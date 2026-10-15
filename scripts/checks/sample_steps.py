"""Samples for listening: the same clips and scores synthesised with different numbers of sampling steps.

Everything else is fixed (EMA weights, guidance 3, seed), so the files of one clip differ only in the
number of steps. `--steps N` is the argument of CFM.sample_cfg: N points from t = 0 to 1, i.e. N - 1 Euler
steps, each one conditional and one unconditional transformer pass.

Writes to <out_dir>:
  valid/<clip>_<mode>_s<N>.wav   validation clips (spread over corpus and language), notes only
                                 (`predict`) and with the clip's F0 (`given`)
  valid/<clip>_recording_voc.wav the clip's own mel and F0 through the vocoder: the best any step count can do
  scores/<score>_s<N>.wav        example scores through the WP7 pipeline with the caption of their `style`
                                 (`f0_vibrato_it` with its F0, the others from the notes)
  figures.json                   per file: mel L1 to the recording (clips), F0 figures, seconds of synthesis
Existing wavs are skipped.

Usage (from tcsinger2-reproduce/, PYTHONPATH=.):
  python scripts/checks/sample_steps.py
"""
import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch

from ldm.infer.pipeline import Singer
from ldm.stage2_samples import SR, Synthesizer, build_dataset, load_stage2, pick_clips, safe_name

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='useful_ckpts/tcsinger2_gtsinger.ckpt')
parser.add_argument('--config', default='configs/tcsinger2.yaml')
parser.add_argument('--steps', default='25,9,5,3,2')
parser.add_argument('--n_clips', type=int, default=10)
parser.add_argument('--scores', default='twinkle_en,twinkle_zh,twinkle_es,f0_vibrato_it')
parser.add_argument('--out_dir', default='samples/steps')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
steps_list = [int(s) for s in args.steps.split(',')]


def write(path, wav):
    sf.write(path, np.clip(wav, -1, 1), SR, subtype='PCM_16')


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, round(time.perf_counter() - t, 3)


model, step = load_stage2(args.ckpt, args.config)
singer = Singer(ckpt=args.ckpt, model=model, step=step)
synth = Synthesizer()
figures_path = os.path.join(args.out_dir, 'figures.json')
figures = json.load(open(figures_path)) if os.path.exists(figures_path) else {}
os.makedirs(os.path.join(args.out_dir, 'valid'), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, 'scores'), exist_ok=True)

dataset = build_dataset('valid', args.config)
picks = pick_clips(dataset, args.n_clips, args.seed, min_frames=375, max_frames=3750)
with model.ema_scope():
    for idx in picks:
        item = dataset[idx]
        name = safe_name(item['name'])
        path = os.path.join(args.out_dir, 'valid', f'{name}_recording_voc.wav')
        if not os.path.exists(path):
            n = item['image'].shape[1]
            write(path, synth.vocoder.vocode(item['image'][:, :n], f0=item['f0'][0, :n]))
        for mode in ('predict', 'given'):
            for s in steps_list:
                path = os.path.join(args.out_dir, 'valid', f'{name}_{mode}_s{s}.wav')
                if os.path.exists(path):
                    continue
                synth.steps = s
                (wav, row, _), secs = timed(lambda: synth.clip(model, dataset, idx, mode, args.seed))
                write(path, wav)
                figures[os.path.relpath(path, args.out_dir)] = {k: (round(v, 4) if isinstance(v, float) else v)
                                                                 for k, v in row.items()} | dict(synth_seconds=secs)
                print(os.path.basename(path), f"mel L1 {row['mel_l1']:.3f}", flush=True)

for score in args.scores.split(','):
    for s in steps_list:
        path = os.path.join(args.out_dir, 'scores', f'{score}_s{s}.wav')
        if os.path.exists(path):
            continue
        (wav, info, _), secs = timed(lambda: singer.sing(f'examples/{score}.json', seed=args.seed, steps=s))
        write(path, wav)
        figures[os.path.relpath(path, args.out_dir)] = dict(mode=info['mode'], caption=info['caption'], measured=info.get('measured'),
                                                         synth_seconds=secs)
        print(os.path.basename(path), info['mode'], flush=True)

with open(figures_path, 'w', encoding='utf-8') as f:
    json.dump(figures, f, indent=1, ensure_ascii=False)
print('wrote', args.out_dir)
