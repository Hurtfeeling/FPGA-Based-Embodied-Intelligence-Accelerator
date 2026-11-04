#!/usr/bin/env python3
"""Prepare exact normalization JSON and language NPZ on the host, without Torch.

Example: python prepare_robot_assets.py --model-dir SOURCE --language-from SOURCE/validation_0.npz
For a new instruction, use --task TEXT (requires host package tokenizers).
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from robot_io.preprocessing import StateActionNormalizer, read_language_assets


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_f32_tensor(path, key):
    # Minimal bounded parser for the two small exported processor files.
    with Path(path).open('rb') as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError('Invalid safetensors header: ' + str(path))
        size = struct.unpack('<Q', size_bytes)[0]
        if size > 1024**2:
            raise ValueError('Unexpectedly large processor header')
        header = json.loads(stream.read(size).decode('utf-8'))
        entry = header[key]
        if entry['dtype'] != 'F32' or entry['shape'] != [6]:
            raise ValueError('Expected F32 [6] for ' + key)
        first, last = entry['data_offsets']
        if first < 0 or last-first != 24:
            raise ValueError('Invalid tensor span for ' + key)
        stream.seek(8+size+first)
        raw = stream.read(last-first)
        if len(raw) != 24:
            raise ValueError('Truncated processor tensor ' + key)
        return np.frombuffer(raw, dtype='<f4').copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'robot_assets')
    language = parser.add_mutually_exclusive_group(required=True)
    language.add_argument('--task', help='Task text, tokenized using the exported tokenizer.json')
    language.add_argument('--language-from', type=Path, help='Reuse tokens/mask from an existing input archive')
    parser.add_argument('--language-text', help='Document the task already represented by --language-from')
    args = parser.parse_args()
    model = args.model_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pre = model/'policy_preprocessor_step_5_normalizer_processor.safetensors'
    post = model/'policy_postprocessor_step_0_unnormalizer_processor.safetensors'
    settings = json.loads((model/'policy_preprocessor.json').read_text(encoding='utf-8'))
    config = next(step['config'] for step in settings['steps'] if step['registry_name']=='normalizer_processor')
    if config['norm_map']['STATE'] != 'MEAN_STD' or config['norm_map']['ACTION'] != 'MEAN_STD':
        raise ValueError('This tool requires the exported MEAN_STD policy')
    stats = dict(eps=config.get('eps', 1e-8),
        state_mean=read_f32_tensor(pre,'observation.state.mean').tolist(),
        state_std=read_f32_tensor(pre,'observation.state.std').tolist(),
        action_mean=read_f32_tensor(post,'action.mean').tolist(),
        action_std=read_f32_tensor(post,'action.std').tolist(),
        joint_order=None, joint_units=None, training_mapping_confirmed=False,
        formula=dict(state='(x-mean)/(std+eps)', action='y*std+mean'),
        sources={pre.name:digest(pre), post.name:digest(post)})
    stats_path = output/'stats.json'
    stats_path.write_text(json.dumps(stats, indent=2)+'\n', encoding='utf-8')
    StateActionNormalizer(stats_path)
    if args.language_from:
        data = read_language_assets(args.language_from)
        language_info = dict(source=str(args.language_from.resolve()),
                             source_sha256=digest(args.language_from), text=args.language_text)
    else:
        try:
            from tokenizers import Tokenizer
        except ImportError:
            parser.error('--task requires tokenizers on the host; --language-from works with NumPy only')
        tokenizer = Tokenizer.from_file(str(model/'tokenizer/tokenizer.json'))
        tokenizer.enable_truncation(max_length=48, direction='left')
        tokenizer.enable_padding(direction='right', pad_id=2, pad_token='<|im_end|>', length=48)
        text = args.task.rstrip('\n')+'\n'
        encoded = tokenizer.encode(text)
        data = dict(language_tokens=np.asarray([encoded.ids], dtype=np.int64),
                    language_mask=np.asarray([encoded.attention_mask], dtype=np.bool_))
        language_info = dict(source='exported tokenizer.json', text=text,
                             tokenizer_sha256=digest(model/'tokenizer/tokenizer.json'))
    np.savez(str(output/'language.npz'), **data)
    read_language_assets(output/'language.npz')
    (output/'language.json').write_text(json.dumps(language_info, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('ROBOT_ASSETS', str(output))


if __name__ == '__main__':
    main()
