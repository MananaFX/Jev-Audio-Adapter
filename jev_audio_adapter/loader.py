"""Strict best2250 loader. Downloads are separate; loading is local-only."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from .assets import verify_adapter, safe_asset, encoder_weight_map
from .model import AudioDecisionModel


def release_parameter(name):
    return name == 'readout' or name.startswith('encoder.adapter.') or (
        name.startswith('backbone.language_model.layers.') and name.endswith(('.a', '.b')))


def apply_release(model, weights):
    """Fail before mutation unless every release tensor has matching shape/dtype."""
    params = dict(model.named_parameters())
    expected = {k for k in params if release_parameter(k)}
    if len(expected) != 75 or set(weights) != expected:
        raise ValueError('Release tensor names mismatch (expected exactly 75)')
    for key, value in weights.items():
        if value.shape != params[key].shape or value.dtype != torch.float32:
            raise ValueError('Release tensor shape/dtype mismatch: ' + key)
        if not torch.isfinite(value).all():
            raise ValueError('Non-finite release tensor: ' + key)
    with torch.no_grad():
        for key, value in weights.items():
            params[key].copy_(value)
    model.requires_grad_(False)
    return model.eval()


def load_model(weights_path, d3_directory, firered_directory):
    verify_adapter(weights_path)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16-capable CUDA GPU required; CPU/T4/quantization unsupported')
    from transformers import AutoTokenizer, WhisperFeatureExtractor, Qwen3_5Model
    from .vendor.firered.configuration_audio_encoder import FireRedAudioEncoderConfig
    from .vendor.firered.modeling_audio_encoder import FireRedAudioEncoder
    d3, fire = Path(d3_directory), Path(firered_directory)
    index = json.loads(safe_asset(d3, 'model.safetensors.index.json').read_text())
    for shard in set(index['weight_map'].values()):
        safe_asset(d3, shard)
    decision = json.loads(safe_asset(d3, 'decision_config.json').read_text())
    if (decision['attention_mode'] != 'noncausal_full_attention'
            or decision['pooling'] != 'last' or len(decision['codes']) != 255
            or len(set(decision['codes'])) != 255 or float(decision.get('temperature', 1)) <= 0):
        raise ValueError('Unsupported decision contract')
    config = json.loads(safe_asset(fire, 'config.json').read_text())
    enc_cfg = FireRedAudioEncoderConfig(**config['audio_encoder_config'])
    enc_cfg._attn_implementation = 'sdpa'
    encoder = FireRedAudioEncoder(enc_cfg)
    mapping = encoder_weight_map(json.loads(safe_asset(fire, 'model.safetensors.index.json').read_text()))
    state = {}
    for shard in sorted(set(mapping.values())):
        with safe_open(str(safe_asset(fire, shard)), framework='pt', device='cpu') as f:
            for name, filename in mapping.items():
                if filename == shard:
                    state[name.removeprefix('audio_encoder.')] = f.get_tensor(name)
    encoder.load_state_dict(state, strict=True)
    del state
    encoder.to(device='cuda', dtype=torch.bfloat16)
    encoder.adapter.float()
    encoder.adapter.register_forward_pre_hook(lambda module, args: (args[0].float(),))
    encoder.adapter.register_forward_hook(lambda module, args, out: out.to(torch.bfloat16))
    backbone, loading = Qwen3_5Model.from_pretrained(str(d3), local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map={'': 'cuda'}, output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('D3 backbone did not load completely')
    if any(not k.startswith(('lm_head.', 'mtp.')) for k in loading.get('unexpected_keys', [])):
        raise ValueError('Unexpected D3 backbone tensors')
    tokenizer = AutoTokenizer.from_pretrained(str(d3), local_files_only=True, trust_remote_code=False)
    processor = json.loads(safe_asset(fire, 'processor_config.json').read_text())
    features = WhisperFeatureExtractor(**processor['feature_extractor'])
    weights = load_file(str(weights_path), device='cpu')
    model = AudioDecisionModel(encoder, backbone, tokenizer, features, decision, weights['readout'].cuda())
    return apply_release(model, weights)
