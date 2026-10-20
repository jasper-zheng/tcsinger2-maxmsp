"""F0 curves for notes-only synthesis, compared without synthesis (WP10). Figures: ldm/f0gen/metrics.py.

Methods (--methods, any number):
  sung             the recording's F0 (the reference figures)
  stage2           the F0 predictor of the stage-2 model at the first sampling step, as notes-only synthesis
                   uses it (CFM.sample_cfg with f0_freeze_step=0): one forward pass at t = 0 from noise
  f0gen:<ckpt>     an F0 generator checkpoint (milestone, resume checkpoint or export); flow models are
                   sampled with --steps and --cfg
Each curve-producing method is run with --seeds seeds; the seed-to-seed spread is the median pitch
difference (cents) between the curves of the first two seeds. The clips are the selection of
scripts/eval_stage2.py (the same rows of the same split).

With --synth K: the first K clips are also synthesised by stage 2 in `given` mode with the curve of every
method (seed 0), vocoded with that curve, and written with an F0 plot (<out_dir>/<split>/synth/); figures:
F0 frame error of the output against the curve, mel L1 against the recording.

Writes <out_dir>/<split>/<label>.json per method (label: sung, stage2, or the checkpoint's name plus
steps/cfg) and prints the table of every method of that split found there.

Usage (from tcsinger2-reproduce/, PYTHONPATH=.):
  python scripts/eval_f0gen.py --methods sung stage2 [--split test_unseen]
  python scripts/eval_f0gen.py --methods f0gen:logs/<run>/checkpoints/<ckpt> --steps 32 --cfg 1.5
  python scripts/eval_f0gen.py --methods sung stage2 f0gen:<ckpt> --synth 16 --table_only
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from ldm.f0gen.dataset import F0Specs, collate
from ldm.f0gen.metrics import curve_rows, summarise
from ldm.stage2_samples import f0_metrics, pick_clips

parser = argparse.ArgumentParser()
parser.add_argument('--methods', nargs='+', default=['sung', 'stage2'])
parser.add_argument('--split', default='valid')
parser.add_argument('--n_clips', type=int, default=200)
parser.add_argument('--seeds', type=int, default=3)
parser.add_argument('--steps', type=int, default=32)
parser.add_argument('--cfg', type=float, default=1.0)
parser.add_argument('--stage2_ckpt', default='useful_ckpts/tcsinger2_gtsinger.ckpt')
parser.add_argument('--stage2_config', default='configs/tcsinger2.yaml')
parser.add_argument('--synth', type=int, default=0, help='clips to synthesise with stage 2 (0 = none)')
parser.add_argument('--out_dir', default='samples/wp10')
parser.add_argument('--table_only', action='store_true', help='only print the table of the results found')
parser.add_argument('--save_curves', action='store_true', help='also write <label>_curves.npz (the curve of every seed and clip)')
args = parser.parse_args()

out_dir = os.path.join(args.out_dir, args.split)
os.makedirs(out_dir, exist_ok=True)
# spec_crop_len as in training (20 s): without it every clip is cut to the dataset default of 1,248 frames, which
# also changes the clip selection and leaves the synthesised clips without F0 after 6.7 s
ds = F0Specs(args.split, main_spec_dir_path='data/sing,data/sing_ext', other_spec_dir_path='', max_prompt_len=1500,
             spec_crop_len=3760)
picks = pick_clips(ds, args.n_clips, 0, min_frames=375, max_frames=3750)
items = [ds[i] for i in picks]
meta = ds.dataset
info = [dict(corpus=meta['dataset'].fillna('GTSinger').iloc[i], language=meta['language'].iloc[i], singer=meta['singer'].iloc[i])
        for i in picks]
_stage2 = {}


def stage2():
    if not _stage2:
        from ldm.stage2_samples import build_dataset, load_stage2
        _stage2['model'], _ = load_stage2(args.stage2_ckpt, args.stage2_config)
        _stage2['ds'] = build_dataset(args.split, args.stage2_config)
    return _stage2['model'], _stage2['ds']


@torch.no_grad()
def stage2_curve(k, seed):
    """The F0 that notes-only synthesis (f0_freeze_step=0) conditions on, for clip k."""
    model, sds = stage2()
    idx = picks[k]
    item = sds[idx]
    assert item['name'] == items[k]['name']
    batch = sds.collater([item])
    n = item['image'].shape[1]
    src = batch[model.cond_stage_key]
    acoustic = {kk: v.to(model.device) for kk, v in src['acoustic'].items()}
    acoustic['prompt_embed'] = model.get_first_stage_encoding(model.encode_first_stage(acoustic['prompt'])).detach()
    cond = model.get_learned_conditioning(dict(caption=list(src['caption']), acoustic=acoustic, name=src['name'],
                                               infer=True, f0_mode='predict'))
    torch.manual_seed(seed + idx)
    x = torch.randn((1, model.mel_dim, -(-acoustic['ph'].shape[-1] // 4)), device=model.device)
    with model.ema_scope():
        _, _, f0 = model.apply_model(x, torch.zeros(1, device=model.device, dtype=torch.long), cond)
    return f0[0, 0, :n].float().cpu().numpy()


def f0gen_curves(path, seed):
    from ldm.f0gen.module import load_f0gen
    model = load_f0gen(path)
    out = []
    for k in range(0, len(items), 16):
        part = items[k:k + 16]
        f0 = model.generate(collate(part), steps=args.steps, cfg=args.cfg, seed=seed + k)
        out += [f0[i, :len(it['r'])].cpu().numpy() for i, it in enumerate(part)]
    del model
    torch.cuda.empty_cache()
    return out


def label_of(method):
    if not method.startswith('f0gen:'):
        return method
    path = method.split(':', 1)[1]
    name = os.path.basename(path.rstrip('/')).replace('.ckpt', '')
    run = path.split('/')[1] if path.startswith('logs/') else ''
    return f"{run.split('_', 1)[-1] + '_' if run else ''}{name}_s{args.steps}_cfg{args.cfg:g}"


def curves_of(method, seed):
    if method == 'sung':
        return [it['f0'] for it in items]
    if method == 'stage2':
        return [stage2_curve(k, seed) for k in range(len(items))]
    return f0gen_curves(method.split(':', 1)[1], seed)


def evaluate(method):
    seeds = 1 if method == 'sung' else args.seeds
    clips, notes, per_seed = [], [], []
    for seed in range(seeds):
        curves = curves_of(method, seed)
        per_seed.append(curves)
        for k, (f0, it) in enumerate(zip(curves, items)):
            clip, nts = curve_rows(f0, it['midi'], it['onset'], it['f0'])
            clips.append(dict(clip, seed=seed, name=it['name'], **info[k]))
            notes += [dict(n, seed=seed, **info[k]) for n in nts]
    import pandas as pd
    summary = dict(all=summarise(clips, notes))
    c, nt = pd.DataFrame(clips), pd.DataFrame(notes)
    for key in ('corpus', 'language'):
        summary[f'by_{key}'] = {g: summarise(c[c[key] == g].to_dict('records'), nt[nt[key] == g].to_dict('records'))
                                for g in sorted(c[key].unique())}
    gt = c['corpus'] == 'GTSinger'
    if args.split == 'test_unseen':
        summary['by_corpus']['GTSinger unseen KO/IT'] = summary['by_corpus'].get('GTSinger')
    if len(per_seed) > 1:
        summary['all']['seed_spread_cents'] = round(float(np.nanmedian([f0_metrics(a, b)[2] for a, b in zip(per_seed[0], per_seed[1])])), 2)
    return summary, clips, per_seed


def synthesise(curves_by_label):
    """Stage 2 in `given` mode with each curve, for the first args.synth clips."""
    import soundfile as sf
    from ldm.stage2_samples import SR, Synthesizer, plot_f0, safe_name
    model, sds = stage2()
    synth = Synthesizer()
    sdir = os.path.join(out_dir, 'synth')
    os.makedirs(sdir, exist_ok=True)
    rows = []
    for k in range(min(args.synth, len(items))):
        idx = picks[k]
        item = sds[idx]
        n = item['image'].shape[1]
        plot = {}
        for label, curves in curves_by_label.items():
            f0 = np.zeros(n, dtype=np.float32)
            f0[:min(n, len(curves[k]))] = curves[k][:n]
            it = dict(item, f0=f0[np.newaxis])
            batch = sds.collater([it])
            torch.manual_seed(idx)
            with model.ema_scope():
                mel, f0_used = model.synthesize(batch, f0_mode='given', cfg_scale=3.0, steps=25)
            mel, f0_used = mel[0, :, :n].float().cpu().numpy(), f0_used[0, :n].float().cpu().numpy()
            wav = synth.vocoder.vocode(mel, f0=f0_used)
            f0_out, _ = synth.rmvpe.get_pitch(wav.astype(np.float32), SR, 256, len(wav) // 256)
            rows.append(dict(name=item['name'], method=label, ffe_out_vs_f0=f0_metrics(f0_used, f0_out)[0],
                             mel_l1=float(np.mean(np.abs(mel - item['image'][:, :n])))))
            sf.write(os.path.join(sdir, f"{safe_name(item['name'])}_{label}.wav"), np.clip(wav, -1, 1), SR, subtype='PCM_16')
            plot[label] = f0_used
        plot_curves(os.path.join(sdir, f"{safe_name(item['name'])}_f0.png"), items[k], plot)
    import pandas as pd
    df = pd.DataFrame(rows)
    table = df.groupby('method')[['ffe_out_vs_f0', 'mel_l1']].mean().round(4)
    with open(os.path.join(sdir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(table=table.to_dict('index'), rows=rows), f, ensure_ascii=False, indent=1)
    print(table.to_string())


def plot_curves(path, item, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ldm.data.phone_set import MIDI_PAD_ID
    semi = lambda f: np.where(f > 0, 69 + 12 * np.log2(np.maximum(f, 1e-3) / 440.0), np.nan)
    t = np.arange(len(item['f0'])) / 187.5
    fig, axes = plt.subplots(len(curves), 1, figsize=(max(8, t[-1] * 1.6), 2.2 * len(curves)), sharex=True, squeeze=False)
    midi = item['midi'].astype(float)
    for ax, (label, f0) in zip(axes[:, 0], curves.items()):
        ax.plot(t, np.where((midi > 0) & (midi < MIDI_PAD_ID), midi, np.nan), color='0.6', lw=3)
        ax.plot(t[:len(f0)], semi(f0[:len(t)]), lw=1, color='tab:red' if label == 'sung' else 'tab:blue')
        ax.set_ylabel(label, fontsize=7)
    axes[0, 0].set_title(item['name'], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


curves_by_label = {}
if not args.table_only or args.synth:
    for method in args.methods:
        label = label_of(method)
        summary, clips, per_seed = evaluate(method)
        curves_by_label[label] = per_seed[0]
        if args.save_curves:
            np.savez_compressed(os.path.join(out_dir, f'{label}_curves.npz'), names=np.array([it['name'] for it in items]),
                                **{f's{seed}_c{k}': np.asarray(c, np.float32) for seed, curves in enumerate(per_seed)
                                   for k, c in enumerate(curves)})
        if not args.table_only:
            with open(os.path.join(out_dir, f'{label}.json'), 'w', encoding='utf-8') as f:
                json.dump(dict(method=method, label=label, split=args.split, clips=len(items), seeds=args.seeds, steps=args.steps,
                               cfg=args.cfg, summary=summary, rows=clips), f, ensure_ascii=False, indent=1)
if args.synth:
    synthesise(curves_by_label)

keys = ['notes_within_1', 'short_notes_within_1', 'notes_flat_cents', 'voicing_err', 'ffe', 'cents', 'vibrato_share',
        'vib_extent_median', 'vib_extent_when_vibrato', 'vib_rate_when_vibrato', 'onset_ms_median', 'r_std', 'jitter', 'seed_spread_cents']
results = {}
for fn in sorted(glob.glob(os.path.join(out_dir, '*.json'))):
    with open(fn, encoding='utf-8') as f:
        d = json.load(f)
    results[d['label']] = d['summary']
print(f'\n{args.split}: {len(items)} clips')
for scope in ('all', 'GTSinger', 'M4Singer', 'OpenSinger', 'Opencpop'):
    print(f'-- {scope}')
    print(f"{'':42s}" + ''.join(f'{k[:13]:>14s}' for k in keys))
    for label, s in results.items():
        row = s['all'] if scope == 'all' else s['by_corpus'].get(scope)
        if row:
            print(f'{label[:42]:42s}' + ''.join(f"{row.get(k, float('nan')):14.3f}" for k in keys))
