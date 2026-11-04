"""Single-audio multiple-choice inference; prints JSON, never uploads audio."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True, help='Published model.safetensors')
    p.add_argument('--audio', required=True)
    p.add_argument('--question', required=True)
    p.add_argument('--choices', required=True, help='JSON array of candidate strings')
    p.add_argument('--d3-dir')
    p.add_argument('--firered-dir')
    p.add_argument('--download-bases', action='store_true', help='Explicitly download pinned public bases')
    p.add_argument('--cache-dir')
    args = p.parse_args()
    choices = json.loads(args.choices)
    if not isinstance(choices, list) or not 1 <= len(choices) <= 255 or any(not isinstance(x,str) or not x.strip() for x in choices):
        p.error('--choices must contain 1–255 nonempty strings')
    if not Path(args.audio).is_file():
        p.error('--audio file missing')
    if args.download_bases and (args.d3_dir or args.firered_dir):
        p.error('Use --download-bases OR both local base directories')
    from jev_audio_adapter.assets import verify_adapter, download_bases
    verify_adapter(args.weights)
    if args.download_bases:
        d3, fire = download_bases(args.cache_dir)
    elif args.d3_dir and args.firered_dir:
        d3, fire = args.d3_dir, args.firered_dir
    else:
        p.error('Specify --download-bases or both --d3-dir and --firered-dir')
    import torch
    from jev_audio_adapter.loader import load_model
    model = load_model(args.weights, d3, fire)
    with torch.inference_mode():
        scores = model([{'audio_path':args.audio, 'prompt':args.question, 'choices':choices}])[0].float().cpu()
    selected = int(scores.argmax())
    print(json.dumps({'index':selected, 'code':model.decision['codes'][selected],
        'choice':choices[selected], 'logits':scores.tolist()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
