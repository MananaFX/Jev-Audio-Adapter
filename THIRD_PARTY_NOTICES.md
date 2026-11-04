# Third-party notices

## FireRedAudio encoder source

`jev_audio_adapter/vendor/firered/configuration_audio_encoder.py` and `modeling_audio_encoder.py` are copied without modification from [FireRedTeam/FireRedAudio](https://github.com/FireRedTeam/FireRedAudio/tree/4bf612ee3d97fc7ea82be9fddd4fb87771732ea9/fireredaudio/audio_encoder), revision `4bf612ee3d97fc7ea82be9fddd4fb87771732ea9`.

Their upstream Apache License 2.0 is preserved in `jev_audio_adapter/vendor/firered/LICENSE`. The encoder source itself acknowledges adaptation from the Transformers Qwen2.5-Omni encoder, also Apache-2.0. Do not remove those notices or apply the adapter's non-commercial restriction to this vendored source.

## Dependencies and model assets

Runtime packages are installed separately, not bundled. PyTorch (BSD-style and bundled third-party notices), Transformers/Tokenizers/Accelerate/Hugging Face Hub/Safetensors (Apache-2.0), NumPy/SciPy (BSD-style), and SoundFile (BSD-3-Clause; linked libsndfile has its own LGPL terms) retain their respective upstream licenses. This notice does not replace those distributions' license files.

The download helper retrieves public D3 and FireRed model assets at explicitly pinned revisions. They are not redistributed in this GitHub package. Consult their model cards and licenses before use. The Jev adapter's published license and any upstream data/model restrictions apply independently of this inference code.
