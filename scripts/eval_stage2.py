"""Evaluates a stage-2 checkpoint (WP6): synthesis from notes alone and with a given F0.

A fixed set of clips of a split (spread over corpus and language) is synthesised one by one, with the
EMA weights, in the modes of ldm/stage2_samples.py:
  predict  notes + phonemes + caption + prompt; the predicted F0 also drives the vocoder
           (--f0_freeze_step: the sampling step whose prediction is kept, default the first; -1: every
           step conditions on its own prediction, written to <out_dir>/step<N>_<split>_f0every)
  given    the same plus the clip's own F0
Reported per mode, overall and per corpus (see ldm/stage2_samples.py for the definitions): F0 frame
errors, the share of notes within a semitone, mel L1 against the recording.
Writes <out_dir>/step<N>_<split>/metrics.json, and for the first --write clips the wavs and F0 plots.

Usage (from TCSinger2/, PYTHONPATH=.):
  python scripts/eval_stage2.py --ckpt logs/<run>/checkpoints/trainstep_checkpoints/<milestone>.ckpt
  python scripts/eval_stage2.py --ckpt useful_ckpts/tcsinger2_gtsinger.ckpt --split test_unseen
"""
import argparse
import contextlib
import json
import os

import numpy as np
import pandas as pd
import soundfile as sf

from ldm.stage2_samples import SR, Synthesizer, build_dataset, load_stage2, pick_clips, plot_f0, safe_name

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True)
parser.add_argument('--config', default='configs/tcsinger2.yaml')
parser.add_argument('--split', default='valid')
parser.add_argument('--n_clips', type=int, default=200)
parser.add_argument('--write', type=int, default=16, help='clips whose audio and F0 plot are written')
parser.add_argument('--out_dir', default='samples/wp6')
parser.add_argument('--cfg_scale', type=float, default=3.0)
parser.add_argument('--steps', type=int, default=25)
parser.add_argument('--f0_freeze_step', type=int, default=0)
parser.add_argument('--no_ema', action='store_true', help='use the raw weights instead of the EMA weights')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

model, step = load_stage2(args.ckpt, args.config)
dataset = build_dataset(args.split, args.config)
picks = pick_clips(dataset, args.n_clips, args.seed, min_frames=375, max_frames=3750)
synth = Synthesizer(cfg_scale=args.cfg_scale, steps=args.steps, f0_freeze_step=None if args.f0_freeze_step < 0 else args.f0_freeze_step)
suffix = '' if args.f0_freeze_step == 0 else '_f0every' if args.f0_freeze_step < 0 else f'_f0step{args.f0_freeze_step}'
out_dir = os.path.join(args.out_dir, f'step{step:08d}_{args.split}{suffix}')
os.makedirs(out_dir, exist_ok=True)

rows = []
scope = model.ema_scope() if not args.no_ema else contextlib.nullcontext()
with scope:
    for k, idx in enumerate(picks):
        curves = {}
        for mode in ('predict', 'given'):
            wav, row, curves[mode] = synth.clip(model, dataset, idx, mode, args.seed)
            row.update(corpus=dataset.dataset['dataset'].fillna('GTSinger').iloc[idx], language=dataset.dataset['language'].iloc[idx],
                       singer=dataset.dataset['singer'].iloc[idx])
            rows.append(row)
            if k < args.write:
                sf.write(os.path.join(out_dir, f"{safe_name(row['name'])}_{mode}.wav"), np.clip(wav, -1, 1), SR, subtype='PCM_16')
        if k < args.write:
            plot_f0(os.path.join(out_dir, f"{safe_name(row['name'])}_f0.png"), curves, row['name'])

df = pd.DataFrame(rows)
cols = {'given': ['ffe_out_vs_f0', 'mel_l1'], 'predict': ['ffe_pred_vs_sung', 'voicing_err_pred', 'ffe_out_vs_pred', 'notes_within_1',
                                                           'notes_within_1_sung', 'mel_l1']}
summary = {}
for mode, keys in cols.items():
    part = df[df['mode'] == mode]
    summary[mode] = dict(all={k: round(float(part[k].mean()), 4) for k in keys},
                         by_corpus={c: {k: round(float(g[k].mean()), 4) for k in keys} | dict(clips=len(g)) for c, g in part.groupby('corpus')},
                         by_language={c: {k: round(float(g[k].mean()), 4) for k in keys} | dict(clips=len(g)) for c, g in part.groupby('language')})
with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(ckpt=args.ckpt, step=step, split=args.split, clips=len(picks), ema=not args.no_ema, cfg_scale=args.cfg_scale,
                   steps=args.steps, f0_freeze_step=args.f0_freeze_step, summary=summary, rows=rows), f, ensure_ascii=False, indent=1)
print(f'\n{args.ckpt}\nstep {step} | {len(picks)} {args.split} clips | EMA {not args.no_ema} | guidance {args.cfg_scale}, {args.steps} steps, '
      f'F0 of step {args.f0_freeze_step}')
for mode, keys in cols.items():
    print(f'[{mode}]', ', '.join(f'{k} {v:.3f}' for k, v in summary[mode]['all'].items()))
    for c, v in summary[mode]['by_corpus'].items():
        print(f'    {c:11s}', ', '.join(f'{k} {x:.3f}' for k, x in v.items() if k != 'clips'), f"({v['clips']} clips)")
print('wrote', out_dir)
