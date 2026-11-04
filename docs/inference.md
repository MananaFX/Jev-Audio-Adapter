# Inference

Code belongs in this GitHub repository. Adapter weights belong in [ModelScope](https://modelscope.cn/models/MewTiger/Jev-Audio-Adapter). The published `model.safetensors` contains only the learned audio adapter, LoRA parameters, and decision readout, not the frozen base models. It is not a standalone Transformers model and cannot be loaded with `AutoModel.from_pretrained` by itself.

## Supported task

Given one audio file, a question, and 1–255 candidate answers, return the selected candidate and candidate logits. This is direct audio-conditioned classification / multiple-choice reasoning, not open-ended speech transcription or text generation. Logits are not calibrated confidence scores.

The implementation preserves the training forward path: FireRed audio encoder, FP32 adapter, Qwen3.5/D3 backbone, rank-16 LoRA (alpha 32), noncausal full-attention layers, causal recurrent layers, last-token pooling, and the trained 255-code FP32 readout. Candidate codes come from the pinned D3 decision configuration, not an assumed alphabet.

## Setup

Use Python 3.11+ and a BF16-capable NVIDIA CUDA GPU. Install the appropriate official PyTorch CUDA wheel for your system, then:

```bash
python -m pip install -r requirements-inference.txt
```

CPU-only inference, T4, quantization, and multi-GPU sharding are not supported by this loader. The pure-PyTorch recurrent fallback is used when optional accelerated kernels are absent. No FlashAttention installation is required. GPU memory depends on audio and prompt length; no fixed minimum is guaranteed.

Download [model.safetensors at the pinned ModelScope revision](https://www.modelscope.cn/models/MewTiger/Jev-Audio-Adapter/resolve/858cff0254c976a615f05f2d4746b8584c811c3d/model.safetensors?view=false) into a directory of your choice. The revision is `858cff0254c976a615f05f2d4746b8584c811c3d`. The loader checks SHA-256 before model loading:

```
efd7bb52e3f188fabaeea341ea7b796471e84241568a1c293b7f822e0a9999f7
```

Run with an audio file that you have the right to use:

```bash
python inference.py \
  --weights ./weights/model.safetensors \
  --audio ./example.wav \
  --question 'What is the dominant sound?' \
  --choices '["Speech", "Music", "Traffic"]' \
  --download-bases
```

`--download-bases` explicitly downloads pinned public frozen model assets into the Hugging Face cache. This is a substantial first download. It never uploads your audio or executes remote model Python. The selected FireRed shard may contain extra tensors, but only `audio_encoder.*` tensors are loaded.

For offline inference, supply previously downloaded snapshots instead:

```bash
HF_HUB_OFFLINE=1 python inference.py \
  --weights ./weights/model.safetensors \
  --d3-dir /path/to/d3-snapshot \
  --firered-dir /path/to/firered-snapshot/FireRedAudio \
  --audio ./example.wav \
  --question 'What is the dominant sound?' \
  --choices '["Speech", "Music", "Traffic"]'
```

Use the exact snapshots recorded below; arbitrary compatible-looking base models are not interchangeable. The local-directory loader checks tensor structure and required files but does not enforce hashes for every frozen base file; obtaining the correct snapshots remains the caller’s responsibility. The adapter file itself is SHA-256 checked. `--d3-dir` contains the index, all its shards, tokenizer assets and `decision_config.json`. `--firered-dir` contains `config.json`, `processor_config.json`, the index and shards holding all 494 encoder tensors.

Output is one JSON object with zero-based `index`, decision `code`, selected `choice`, and candidate `logits`.

## Audio and limits

Audio is decoded with libsndfile, averaged to mono, and resampled to 16 kHz when necessary. Supported duration is 0.1–305 seconds. The official 128-bin log-mel processor uses right zero padding and **no truncation**. Unsupported codecs should be converted to WAV first. Processing long audio can consume much more GPU memory; this CLI processes one recording per call.

## Version contract

- Adapter: best2250, exact SHA-256 above; 75 FP32 tensors, 36,840,448 parameters.
- D3: `vllm-sr/d3-flash` at `6838193a0185c0ecfecab937896059d2c3377d61`.
- FireRed weights: `FireRedTeam/FireRedAudio` at `3abf43d7539a2fc05991b9f89295c892f0a034f0`, `FireRedAudio/` subdirectory.
- Vendored encoder: `FireRedTeam/FireRedAudio` code at `4bf612ee3d97fc7ea82be9fddd4fb87771732ea9`.
- Transformers: 5.8.0. The loader relies on that version's Qwen3.5 recurrent-mask and layer APIs.

## Validation status

This release preparation checked the actual published adapter against meta-instantiated public model configurations: all 75 names, shapes, FP32 dtypes and finite values match. The encoder inventory has 494 tensors. A tiny real Qwen3.5 model produced exactly equal CPU FP32 prompt-forward outputs before and after the inference-only refactor, including a left-padded batch. Four contract tests pass:

```bash
python -m unittest discover -s tests -v
```

The newly assembled full public-base loader has **not yet completed an end-to-end GPU/audio smoke test**. CPU checks do not establish BF16 GPU equivalence, latency, memory use or benchmark accuracy. Treat this as a reference inference implementation pending that test; existing evaluation results describe their evaluated checkpoint and pipeline, not a new validation claim for this CLI.

## Licensing

No separate reuse license is currently granted for the original inference code in this repository. Public visibility alone is not a general reuse or commercial-use license. The adapter remains under its published non-commercial license. The vendored FireRed source retains its Apache-2.0 license and attribution. Frozen model weights, tokenizer assets, training-data restrictions and third-party packages retain their own terms. A code license does not grant commercial rights to the adapter or all upstream assets. See `THIRD_PARTY_NOTICES.md` and the ModelScope model card.
