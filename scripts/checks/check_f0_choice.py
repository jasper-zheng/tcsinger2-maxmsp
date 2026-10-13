"""Which predicted F0 should condition the sampler and drive the vocoder in notes-only synthesis (WP6)?

In `predict` mode the model predicts the F0 anew at every Euler step, from the first block's output,
which sees the noisy latent. This script compares, on the same clips and the same noise:
  last       every step conditions on its own prediction and the last one drives the vocoder, as the
             released code does (`CFM.sample_cfg` with f0_freeze_step=None)
  freeze@k   the F0 predicted at step k conditions all later steps and drives the vocoder
             (freeze@0: the prediction from pure noise, i.e. from the notes alone; the default of
             `CFM.sample_cfg` since the review of step 50,000)
  twopass    a `last` run, then a second run from the same noise with the first run's final F0 given
Figures per variant (definitions in ldm/stage2_samples.py): the vocoder's F0 against the sung one, the
output's F0 against the vocoder's, notes within a semitone, mel L1, and the jitter of the F0 curve
(median frame-to-frame step in cents inside voiced stretches; the sung curve's jitter is printed too).
For `last` it also prints how the prediction moves from step to step.

Usage (from TCSinger2/, PYTHONPATH=.):
  python scripts/checks/check_f0_choice.py --ckpt logs/<run>/checkpoints/trainstep_checkpoints/<milestone>.ckpt
"""
import argparse
import json
import math
import os

import numpy as np
import soundfile as sf
import torch

from ldm.stage2_samples import HOP, SR, Synthesizer, build_dataset, f0_metrics, load_stage2, notes_within, pick_clips, safe_name

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True)
parser.add_argument('--config', default='configs/tcsinger2.yaml')
parser.add_argument('--split', default='valid')
parser.add_argument('--n_clips', type=int, default=40, help='the first clips of the selection scripts/eval_stage2.py makes')
parser.add_argument('--freeze', type=int, nargs='*', default=[0, 6, 12])
parser.add_argument('--write', type=int, default=4, help='clips whose audio is written per variant')
parser.add_argument('--out_dir', default='samples/wp6')
parser.add_argument('--cfg_scale', type=float, default=3.0)
parser.add_argument('--steps', type=int, default=25)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()


@torch.no_grad()
def euler(model, cond, uncond, noise, freeze_at=None, f0_given=None):
    """The loop of CFM.sample_cfg, spelled out so that a second pass can be given the first one's F0.
    Returns (latent, F0 of the last step [B, 1, T], list of the F0 of every step)."""
    x, f0_fixed, trace = noise.clone(), f0_given, []
    t_span = torch.linspace(0, 1, args.steps, device=noise.device)
    for i, (t0, t1) in enumerate(zip(t_span[:-1], t_span[1:])):
        c, u = cond, uncond
        if f0_fixed is not None:
            acoustic = dict(cond['acoustic'], f0=f0_fixed)
            c, u = dict(cond, acoustic=acoustic, f0_mode='given'), dict(uncond, acoustic=acoustic, f0_mode='given')
        t = torch.full((x.shape[0],), int(t0 * model.num_timesteps), device=x.device, dtype=torch.long)
        v, _, f0 = model.apply_model(x, t, c)
        v_uncond, _, _ = model.apply_model(x, t, u)
        x = x + (t1 - t0) * (v_uncond + args.cfg_scale * (v - v_uncond))
        trace.append(f0[0, 0].float().cpu().numpy())
        if f0_fixed is None and freeze_at == i:
            f0_fixed = f0
    return x, f0, trace


def jitter(f0):
    """Median absolute frame-to-frame step in cents inside voiced stretches."""
    both = (f0[1:] > 0) & (f0[:-1] > 0)
    return float(np.median(1200 * np.abs(np.log2(f0[1:][both] / f0[:-1][both])))) if both.sum() > 10 else float('nan')


model, step = load_stage2(args.ckpt, args.config)
dataset = build_dataset(args.split, args.config)
picks = pick_clips(dataset, 200, args.seed, min_frames=375, max_frames=3750)[:args.n_clips]
synth = Synthesizer(cfg_scale=args.cfg_scale, steps=args.steps)
out_dir = os.path.join(args.out_dir, f'step{step:08d}_{args.split}', 'f0_choice')
os.makedirs(out_dir, exist_ok=True)
variants = ['last'] + [f'freeze@{k}' for k in args.freeze] + ['twopass']

rows, per_step, sung_jitter = [], [], []
with model.ema_scope(), torch.no_grad():
    for k, idx in enumerate(picks):
        item = dataset[idx]
        batch = dataset.collater([item])
        n = item['image'].shape[1]
        src = batch[model.cond_stage_key]
        acoustic = {key: v.to(model.device) for key, v in src['acoustic'].items()}
        acoustic['prompt_embed'] = model.get_first_stage_encoding(model.encode_first_stage(acoustic['prompt'])).detach()
        cond = model.get_learned_conditioning(dict(caption=list(src['caption']), acoustic=acoustic, name=src['name'],
                                                   infer=True, f0_mode='predict'))
        empty = model.get_learned_conditioning(dict(caption=[''], acoustic=acoustic, name=src['name'], infer=True, f0_mode='predict'))
        uncond = dict(cond, caption=empty['caption'])
        torch.manual_seed(args.seed + idx)
        noise = torch.randn((1, model.mel_dim, math.ceil(acoustic['ph'].shape[-1] / 4)), device=model.device)
        f0_sung, midi = item['f0'][0, :n], item['ep_pitches'][0, :n]
        sung_jitter.append(jitter(f0_sung))
        f0_last = None
        for name in variants:
            if name == 'last':
                z, f0, trace = euler(model, cond, uncond, noise)
                f0_last = f0
                # the prediction of every step against the sung curve and against the last step's
                per_step.append([(f0_metrics(f0_sung, tr[:n])[0], f0_metrics(trace[-1][:n], tr[:n])[0]) for tr in trace])
            elif name == 'twopass':
                z, f0, _ = euler(model, cond, uncond, noise, f0_given=f0_last)
            else:
                z, f0, _ = euler(model, cond, uncond, noise, freeze_at=int(name.split('@')[1]))
            mel, f0 = model.decode_first_stage(z)[0, :, :n].float().cpu().numpy(), f0[0, 0, :n].float().cpu().numpy()
            wav = synth.vocoder.vocode(mel, f0=f0)
            f0_out, _ = synth.rmvpe.get_pitch(wav.astype(np.float32), SR, HOP, len(wav) // HOP)
            row = dict(name=item['name'], variant=name, mel_l1=float(np.mean(np.abs(mel - item['image'][:, :n]))),
                       notes_within_1=notes_within(f0, midi), jitter=jitter(f0))
            row['ffe_vs_sung'], row['voicing_err'], row['cents_vs_sung'] = f0_metrics(f0_sung, f0)
            row['ffe_out_vs_f0'], _, row['cents_out'] = f0_metrics(f0, f0_out)
            rows.append(row)
            if k < args.write:
                sf.write(os.path.join(out_dir, f"{safe_name(item['name'])}_{name.replace('@', '')}.wav"), np.clip(wav, -1, 1), SR,
                         subtype='PCM_16')

keys = ['ffe_vs_sung', 'voicing_err', 'cents_vs_sung', 'notes_within_1', 'ffe_out_vs_f0', 'cents_out', 'mel_l1', 'jitter']
summary = {v: {key: round(float(np.nanmean([r[key] for r in rows if r['variant'] == v])), 4) for key in keys} for v in variants}
steps_mean = np.mean(np.array(per_step), axis=0)  # [steps, 2]
with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(ckpt=args.ckpt, step=step, split=args.split, clips=len(picks), summary=summary,
                   sung_jitter=round(float(np.nanmean(sung_jitter)), 4),
                   per_step=dict(ffe_vs_sung=steps_mean[:, 0].round(4).tolist(), ffe_vs_last=steps_mean[:, 1].round(4).tolist()),
                   rows=rows), f, ensure_ascii=False, indent=1)
print(f'\n{args.ckpt}\nstep {step} | {len(picks)} {args.split} clips | guidance {args.cfg_scale}, {args.steps} steps')
print(f"{'variant':10s}", ' '.join(f'{key:>14s}' for key in keys))
for v in variants:
    print(f'{v:10s}', ' '.join(f'{summary[v][key]:14.4f}' for key in keys))
print(f'jitter of the sung F0: {np.nanmean(sung_jitter):.2f} cents')
print('prediction of each step, F0 frame error against the sung curve:  ', ' '.join(f'{x:.3f}' for x in steps_mean[:, 0]))
print('prediction of each step, F0 frame error against the last step\'s:', ' '.join(f'{x:.3f}' for x in steps_mean[:, 1]))
print('wrote', out_dir)
