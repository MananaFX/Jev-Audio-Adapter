"""Pinned public model assets. Network access occurs only on explicit download."""
import hashlib
import json
from pathlib import Path

D3_REPO = 'vllm-sr/d3-flash'
D3_REV = '6838193a0185c0ecfecab937896059d2c3377d61'
FIRE_REPO = 'FireRedTeam/FireRedAudio'
FIRE_REV = '3abf43d7539a2fc05991b9f89295c892f0a034f0'
ADAPTER_REPO = 'MewTiger/Jev-Audio-Adapter'
ADAPTER_SHA256 = 'efd7bb52e3f188fabaeea341ea7b796471e84241568a1c293b7f822e0a9999f7'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_adapter(path):
    if sha256(path) != ADAPTER_SHA256:
        raise ValueError('Weights do not match this best2250 release')


def safe_asset(directory, name):
    root = Path(directory).resolve()
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Unsafe asset filename')
    # Hugging Face snapshots intentionally symlink files to sibling blobs.
    path = root / name
    if not path.is_file():
        raise ValueError('Missing model asset or unsafe filename')
    return path


def encoder_weight_map(index):
    prefix = 'audio_encoder.'
    result = {k: v for k, v in index['weight_map'].items() if k.startswith(prefix)}
    if len(result) != 494:
        raise ValueError('Unexpected FireRed encoder inventory')
    return result


def download_bases(cache_dir=None):
    """Download pinned D3 and only FireRed shards containing encoder tensors.

    A selected FireRed shard may also contain unrelated tensors; they are never
    loaded into this model. No remote Python is executed.
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    d3 = snapshot_download(D3_REPO, revision=D3_REV, cache_dir=cache_dir,
        allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', 'LICENSE*'])
    index_path = hf_hub_download(FIRE_REPO, 'FireRedAudio/model.safetensors.index.json',
                                revision=FIRE_REV, cache_dir=cache_dir)
    weight_map = encoder_weight_map(json.loads(Path(index_path).read_text()))
    for name in sorted(set(weight_map.values()) | {'config.json', 'processor_config.json'}):
        if Path(name).name != name:
            raise ValueError('Unexpected upstream shard filename')
        hf_hub_download(FIRE_REPO, 'FireRedAudio/' + name,
                        revision=FIRE_REV, cache_dir=cache_dir)
    return Path(d3), Path(index_path).parent
