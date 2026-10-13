"""How good is the F0 predictor of a stage-2 checkpoint by itself, and how much does it lean on the latent? (WP6)

The predictor reads the first block's output, which sees the noisy latent. In training that latent is
the recording's own, noised to a random time t (0 = pure noise, 1 = clean), so at large t the pitch can
be read off the latent; notes-only synthesis starts at t = 0, where only notes, phonemes, caption and
prompt are left. For each checkpoint and each t this script makes one forward pass per clip on the
recording's latent noised to t and compares the predicted F0 with the sung one (definitions in
ldm/stage2_samples.py; cents = median pitch difference where both are voiced). t = 0 needs no sampling,
so it is a cheap way to follow notes-only pitch over the milestones. EMA weights.

With --notes: the notes of the t = 0 prediction by length (which ones are more than a semitone off).
With --draws K: the t = 0 prediction from K noise draws, averaged, against a single draw (accuracy, and
jitter = median frame-to-frame step in cents inside voiced stretches).

Usage (from TCSinger2/, PYTHONPATH=.):
  python scripts/checks/check_f0_predictor.py --ckpt <milestone> [<milestone> ...]
  python scripts/checks/check_f0_predictor.py --ckpt <milestone> --t 0 --notes --draws 8 --out samples/wp6/f0_predictor_notes.json
"""
import argparse
import gc
import json
import os

import numpy as np
import pandas as pd
import torch

from ldm.data.phone_set import MIDI_PAD_ID
from ldm.stage2_samples import build_dataset, f0_metrics, load_stage2, notes_within, pick_clips

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, nargs='+')
parser.add_argument('--config', default='configs/tcsinger2.yaml')
parser.add_argument('--split', default='valid')
parser.add_argument('--n_clips', type=int, default=200, help='the selection scripts/eval_stage2.py makes')
parser.add_argument('--t', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75])
parser.add_argument('--notes', action='store_true')
parser.add_argument('--draws', type=int, default=1)
parser.add_argument('--out', default='samples/wp6/f0_predictor.json')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()


def jitter(f0):
    both = (f0[1:] > 0) & (f0[:-1] > 0)
    return float(np.median(1200 * np.abs(np.log2(f0[1:][both] / f0[:-1][both])))) if both.sum() > 10 else float('nan')


def semitones_off(seg, note):
    """Median voiced pitch of a note's frames minus the note, in semitones (nan if fewer than 3 voiced frames)."""
    seg = seg[seg > 0]
    return 69 + 12 * np.log2(np.median(seg) / 440.0) - note if len(seg) >= 3 else float('nan')


def note_rows(f0, f0_sung, midi):
    """One row per note of at least 8 frames, as ldm.stage2_samples.notes_within counts them."""
    out, start, prev = [], 0, None
    for t in range(1, len(midi) + 1):
        if t == len(midi) or midi[t] != midi[start]:
            note = float(midi[start])
            sung_note = 0 < note < MIDI_PAD_ID
            if sung_note and t - start >= 8:
                out.append(dict(seconds=(t - start) * 256 / 48000, note=note, pred=semitones_off(f0[start:t], note),
                                sung=semitones_off(f0_sung[start:t], note), step_from_prev=(note - prev) if prev else float('nan')))
            prev = note if sung_note else None
            start = t
    return out


dataset = build_dataset(args.split, args.config)
picks = pick_clips(dataset, args.n_clips, args.seed, min_frames=375, max_frames=3750)
keys = ['ffe', 'voicing_err', 'cents', 'notes_within_1']
rows, notes, draws = [], [], []
for ckpt in args.ckpt:
    model, step = load_stage2(ckpt, args.config)
    with model.ema_scope(), torch.no_grad():
        for idx in picks:
            item = dataset[idx]
            batch = dataset.collater([item])
            n = item['image'].shape[1]
            src = batch[model.cond_stage_key]
            acoustic = {k: v.to(model.device) for k, v in src['acoustic'].items()}
            acoustic['prompt_embed'] = model.get_first_stage_encoding(model.encode_first_stage(acoustic['prompt'])).detach()
            cond = model.get_learned_conditioning(dict(caption=list(src['caption']), acoustic=acoustic, name=src['name'],
                                                       infer=True, f0_mode='predict'))
            x1 = model.get_first_stage_encoding(model.encode_first_stage(batch[model.first_stage_key].to(model.device)))
            torch.manual_seed(args.seed + idx)
            x0 = torch.randn_like(x1)
            f0_sung, midi = item['f0'][0, :n], item['ep_pitches'][0, :n]

            def predict(x, t):
                _, _, f0 = model.apply_model(x, torch.full((1,), int(t * model.num_timesteps), device=x.device, dtype=torch.long), cond)
                return f0[0, 0, :n].float().cpu().numpy()

            for t in args.t:
                f0 = predict(t * x1 + (1. - (1 - model.sigma_min) * t) * x0, t)  # the latent as CFM.p_losses noises it
                ffe, voicing_err, cents = f0_metrics(f0_sung, f0)
                rows.append(dict(step=step, t=t, name=item['name'], corpus=dataset.dataset['dataset'].fillna('GTSinger').iloc[idx],
                                 language=dataset.dataset['language'].iloc[idx], ffe=ffe, voicing_err=voicing_err, cents=cents,
                                 notes_within_1=notes_within(f0, midi), notes_within_1_sung=notes_within(f0_sung, midi)))
                if t == 0 and args.notes:
                    notes += [dict(step=step, name=item['name'], **r) for r in note_rows(f0, f0_sung, midi)]
                if t == 0 and args.draws > 1:
                    many = np.stack([f0] + [predict(torch.randn_like(x1), 0) for _ in range(args.draws - 1)])
                    voiced = (many > 0).mean(0) >= 0.5  # majority vote on voicing, mean of the log F0 of the voiced draws
                    mean = np.where(voiced, np.exp(np.sum(np.where(many > 0, np.log(np.maximum(many, 1)), 0), 0)
                                                   / np.maximum((many > 0).sum(0), 1)), 0)
                    both = (many[0] > 0) & (many[1] > 0)
                    for name, curve in (('one draw', f0), (f'mean of {args.draws} draws', mean), ('sung', f0_sung)):
                        ffe, voicing_err, cents = f0_metrics(f0_sung, curve)
                        draws.append(dict(step=step, curve=name, ffe=ffe, voicing_err=voicing_err, cents=cents,
                                          notes_within_1=notes_within(curve, midi), jitter=jitter(curve),
                                          cents_between_two_draws=float(np.median(1200 * np.abs(np.log2(many[0][both] / many[1][both]))))))
    del model
    gc.collect()
    torch.cuda.empty_cache()

pd.set_option('display.width', 200)
df = pd.DataFrame(rows)
df['group'] = np.where(df['language'] == 'Chinese', 'Chinese', 'other languages')
print(f'\n{len(picks)} {args.split} clips; notes within a semitone in the sung F0: {df["notes_within_1_sung"].mean():.3f} '
      f'({", ".join(f"{g} {v:.3f}" for g, v in df.groupby("group")["notes_within_1_sung"].mean().items())})')
print(df.groupby(['step', 't'])[keys].mean().round(3).to_string())
print(f'\nat t = {df["t"].min():g}, by language group:')
print(df[df['t'] == df['t'].min()].groupby(['step', 'group'])[keys].mean().round(3).to_string())
if notes:
    nd = pd.DataFrame(notes)
    nd['off'], nd['off_sung'] = ~(nd['pred'].abs() <= 1), ~(nd['sung'].abs() <= 1)
    nd['length'] = pd.cut(nd['seconds'], [0, 0.1, 0.2, 0.4, 0.8, 100])
    nd['pitch'] = pd.cut(nd['note'], [0, 48, 55, 62, 69, 76, 128])
    for step, g in nd.groupby('step'):
        off = g[g['off'] & g['pred'].notna()]
        moved = off[off['step_from_prev'].notna() & (off['step_from_prev'] != 0)]
        print(f'\nstep {step}: {len(g)} notes at t = 0; more than a semitone off: predicted {g["off"].mean():.3f}, sung {g["off_sung"].mean():.3f}; '
              f'by time {g.loc[g["off"], "seconds"].sum() / g["seconds"].sum():.3f} and {g.loc[g["off_sung"], "seconds"].sum() / g["seconds"].sum():.3f}')
        print(f'of the predicted ones: within 2 semitones {(off["pred"].abs() <= 2).mean():.2f}, beyond 4 {(off["pred"].abs() > 4).mean():.2f}; '
              f'on the side of the previous note {(np.sign(moved["pred"]) == -np.sign(moved["step_from_prev"])).mean():.2f} of {len(moved)}')
        for col in ('length', 'pitch'):
            print(g.groupby(col, observed=True).agg(notes=('off', 'size'), predicted_off=('off', 'mean'), sung_off=('off_sung', 'mean'),
                                                    predicted_bias=('pred', 'median')).round(3).to_string())
if draws:
    dd = pd.DataFrame(draws)
    print('\nt = 0, noise draws:')
    print(dd.groupby(['step', 'curve'], sort=False)[keys + ['jitter', 'cents_between_two_draws']].mean().round(3).to_string())
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, 'w', encoding='utf-8') as f:
    json.dump(dict(ckpts=args.ckpt, split=args.split, clips=len(picks), rows=rows, notes=notes, draws=draws), f, ensure_ascii=False, indent=1)
print('wrote', args.out)
