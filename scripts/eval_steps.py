"""How few sampling steps do stage 2 and the F0 generator need (WP11)? No retraining; uniform Euler grids only.

Steps are counted as Euler steps throughout. Stage 2's `steps` argument is the number of points of
linspace(0, 1, steps), so Euler steps E = steps - 1 (the default steps=25 is E = 24, each a conditional and an
unconditional pass). The F0 generator's `steps` is already its number of Euler steps (default 32, one pass each).

The clips are those of scripts/eval_f0gen.py (= scripts/eval_stage2.py): the 200 picks of a split; the synthesis
parts use the first --n_clips of them. The initial noise of both samplers depends only on the seed and the clip,
not on the step count, so a configuration's distance to a reference configuration with the same seed is the
discretisation error alone; the same configuration with another seed gives the size of an ordinary change.

Parts (--part; each writes one JSON per configuration and skips configurations whose JSON exists):
  f0ref     F0 generator curves of scripts/eval_f0gen.py --save_curves (<f0_dir>/<split>/*_curves.npz) against
            the curve of the same seed at the most steps found (64): F0 frame error, voicing error, cents
  stage2    stage 2 in `given` mode with the recording's F0, E in --stage2_steps (the first is the reference);
            plus the second entry with seed 1. Per clip: mel L1 to the recording and to the reference output,
            detail ratio (as scripts/eval_vae.py: mean |difference| between neighbouring mel bins / frames,
            output over recording; below 1 = smoothed), F0 of the vocoded output (RMVPE) against the given F0,
            seconds of model.synthesize
  pipeline  notes-only synthesis as `sing.py --f0_model` does it: the F0 generator's curve with K steps (seed 0,
            from the npz files) given to stage 2 with E steps, for K in --grid_f0 x E in --grid_e; reference =
            the first K and E (today's defaults); plus the reference with seed 1 for both samplers
  listen    wavs for listening: the first --n_listen clips (the first clips of the WP10 listening sets) at the
            (K, E) pairs of --listen, the recording vocoded, an F0 plot with the curves of every K; and the scores
            of --scores through ldm.infer.pipeline.Singer at the same pairs
  speed     Singer.sing(measure=False) on --speed_score, warm, median of --repeats: seconds of the F0 generator,
            the stage-2 call, the vocoder and the whole call, per (K, E) of --listen plus every K and E alone

Outputs in <out_dir>: <part>/<split>/*.json, listen/{<split>,scores}/*.wav, summary_<part>.json / tables printed.

Usage (from tcsinger2-reproduce/, PYTHONPATH=.):
  python scripts/eval_steps.py --part stage2 --split valid
  bash scripts/eval_wp11.sh      # everything, resumable
"""
import argparse
import contextlib
import glob
import json
import os
import re
import time

import numpy as np
import soundfile as sf
import torch

from ldm.stage2_samples import SR, HOP, f0_metrics, pick_clips, safe_name

parser = argparse.ArgumentParser()
parser.add_argument('--part', required=True, choices=['f0ref', 'stage2', 'pipeline', 'listen', 'speed'])
parser.add_argument('--split', default='valid')
parser.add_argument('--n_clips', type=int, default=100, help='clips synthesised (the first of the 200 picks)')
parser.add_argument('--out_dir', default='samples/wp11_steps')
parser.add_argument('--f0_dir', default='samples/wp11_steps/f0gen', help='--out_dir of the eval_f0gen.py runs')
parser.add_argument('--f0_model', default='useful_ckpts/f0gen_gtsinger.ckpt')
parser.add_argument('--stage2_ckpt', default='useful_ckpts/tcsinger2_gtsinger.ckpt')
parser.add_argument('--stage2_config', default='configs/tcsinger2.yaml')
parser.add_argument('--cfg_scale', type=float, default=3.0)
parser.add_argument('--stage2_steps', default='48,24,16,12,8,6,4,2', help='Euler steps; the first is the reference')
parser.add_argument('--grid_f0', default='32,16,8,4', help='F0 generator steps; the first is the reference')
parser.add_argument('--grid_e', default='24,12,8,4', help='stage-2 Euler steps; the first is the reference')
parser.add_argument('--listen', default='32:24,16:12,8:8,8:4,4:4', help='K:E pairs')
parser.add_argument('--n_listen', type=int, default=8)
parser.add_argument('--scores', default='twinkle_en,twinkle_zh,twinkle_es')
parser.add_argument('--speed_score', default='examples/twinkle_en.json')
parser.add_argument('--repeats', type=int, default=5)
args = parser.parse_args()
ints = lambda s: [int(x) for x in s.split(',')]
pairs = [tuple(int(x) for x in p.split(':')) for p in args.listen.split(',')]


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def f0_npz(k):
    return os.path.join(args.f0_dir, args.split, f'f0gen_gtsinger_s{k}_cfg1_curves.npz')


def load_curves(k, seed, env=None):
    path = f0_npz(k)
    assert os.path.exists(path), f'{path} missing: run scripts/eval_f0gen.py --steps {k} --save_curves --out_dir {args.f0_dir} first'
    z = np.load(path)
    names = list(z['names'])
    if env is not None:  # the curves must be those of the clips synthesised
        assert names[:len(env.f0items)] == [it['name'] for it in env.f0items], path
    return [z[f's{seed}_c{i}'] for i in range(len(names))], names


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


# ---------------------------------------------------------------------------------------------------------- f0ref
def part_f0ref():
    files = glob.glob(os.path.join(args.f0_dir, args.split, 'f0gen_gtsinger_s*_cfg1_curves.npz'))
    ks = sorted(int(re.search(r'_s(\d+)_cfg', f).group(1)) for f in files)
    assert ks, f'no curves in {args.f0_dir}/{args.split}'
    ref_k = max(ks)
    out = {}
    for k in ks:
        rows = []
        for seed in range(3):
            try:
                ref, names = load_curves(ref_k, seed)
                cur, _ = load_curves(k, seed)
            except KeyError:
                continue
            for name, a, b in zip(names, ref, cur):
                ffe, vde, cents = f0_metrics(a, b)
                rows.append(dict(name=name, seed=seed, ffe=ffe, voicing_err=vde, cents=cents))
        out[k] = dict(ref_steps=ref_k, ffe=float(np.mean([r['ffe'] for r in rows])),
                      voicing_err=float(np.mean([r['voicing_err'] for r in rows])),
                      cents=float(np.median([r['cents'] for r in rows])), n=len(rows))
    # seed-to-seed at the reference: how far another seed is (the yardstick)
    a, _ = load_curves(ref_k, 0)
    b, _ = load_curves(ref_k, 1)
    m = [f0_metrics(x, y) for x, y in zip(a, b)]
    out['seed_spread_at_ref'] = dict(ffe=float(np.mean([x[0] for x in m])), voicing_err=float(np.mean([x[1] for x in m])),
                                     cents=float(np.median([x[2] for x in m])))
    save_json(os.path.join(args.out_dir, 'f0ref', f'{args.split}.json'), out)
    print(f'{args.split}: F0 generator curves against the {ref_k}-step curve of the same seed')
    print(f"{'steps':>8s}{'FFE':>9s}{'voicing':>9s}{'cents':>9s}")
    for k, r in out.items():
        print(f"{str(k):>8s}{r['ffe']:9.4f}{r['voicing_err']:9.4f}{r['cents']:9.2f}")


# ----------------------------------------------------------------------------------------------- synthesis parts
class Env:
    """The clips, the stage-2 model and dataset, vocoder and pitch tracker (loaded once)."""

    def __init__(self):
        from ldm.f0gen.dataset import F0Specs
        from ldm.stage2_samples import Synthesizer, build_dataset, load_stage2
        ds = F0Specs(args.split, main_spec_dir_path='data/sing,data/sing_ext', other_spec_dir_path='', max_prompt_len=1500,
                     spec_crop_len=3760)  # as scripts/eval_f0gen.py, so that the picks are the same
        self.picks = pick_clips(ds, 200, 0, min_frames=375, max_frames=3750)
        self.f0items = [ds[i] for i in self.picks[:args.n_clips]]
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            self.model, self.step = load_stage2(args.stage2_ckpt, args.stage2_config)
            self.synth = Synthesizer()
        self.sds = build_dataset(args.split, args.stage2_config)

    def item(self, k):
        item = self.sds[self.picks[k]]
        assert item['name'] == self.f0items[k]['name'], (item['name'], self.f0items[k]['name'])
        return item

    @torch.no_grad()
    def given(self, k, curve, e, seed=0):
        """Stage 2 (`given`, E Euler steps) with `curve` (None = the recording's F0) for clip k."""
        item = self.item(k)
        n = item['image'].shape[1]
        if curve is not None:
            f0 = np.zeros(n, dtype=np.float32)
            f0[:min(n, len(curve))] = curve[:n]
            item = dict(item, f0=f0[np.newaxis])
        batch = self.sds.collater([item])
        torch.manual_seed(seed * 100003 + self.picks[k])  # seed 0: as eval_f0gen.synthesise
        (mel, f0_used), secs = timed(lambda: self.model.synthesize(batch, f0_mode='given', cfg_scale=args.cfg_scale, steps=e + 1))
        return item, mel[0, :, :n].float().cpu().numpy(), f0_used[0, :n].float().cpu().numpy(), secs

    def measure(self, item, mel, f0_used, ref_mel=None):
        n = mel.shape[1]
        gt = item['image'][:, :n]
        wav = self.synth.vocoder.vocode(mel, f0=f0_used)
        f0_out, _ = self.synth.rmvpe.get_pitch(wav.astype(np.float32), SR, HOP, len(wav) // HOP)
        ffe, vde, cents = f0_metrics(f0_used, f0_out)
        row = dict(name=item['name'], frames=n, mel_l1=float(np.mean(np.abs(mel - gt))),
                   df_out=float(np.mean(np.abs(np.diff(mel, axis=0)))), df_gt=float(np.mean(np.abs(np.diff(gt, axis=0)))),
                   dt_out=float(np.mean(np.abs(np.diff(mel, axis=1)))), dt_gt=float(np.mean(np.abs(np.diff(gt, axis=1)))),
                   ffe_out=ffe, voicing_err_out=vde, cents_out=cents)
        if ref_mel is not None:
            row['mel_l1_ref'] = float(np.mean(np.abs(mel - ref_mel[:, :n])))
        return row, wav


def summary_of(rows):
    w = np.array([r['frames'] for r in rows], dtype=float)
    wmean = lambda key: float(np.sum(w * np.array([r[key] for r in rows])) / np.sum(w))
    out = dict(n=len(rows), mel_l1=float(np.mean([r['mel_l1'] for r in rows])),
               detail_freq=wmean('df_out') / wmean('df_gt'), detail_time=wmean('dt_out') / wmean('dt_gt'),
               ffe_out=float(np.mean([r['ffe_out'] for r in rows])), cents_out=float(np.median([r['cents_out'] for r in rows])),
               s2_seconds=float(np.mean([r['s2_seconds'] for r in rows])))
    if 'mel_l1_ref' in rows[0]:
        out['mel_l1_ref'] = float(np.mean([r['mel_l1_ref'] for r in rows]))
    return out


def run_configs(env, part, configs, ref_name):
    """configs: list of (name, fn(k) -> (item, mel, f0_used, secs)); the one called ref_name is run first and its mels
    are kept (npz) as the reference of the others."""
    d = os.path.join(args.out_dir, part, args.split)
    os.makedirs(d, exist_ok=True)
    ref_path = os.path.join(d, f'{ref_name}_mels.npz')
    configs = sorted(configs, key=lambda c: c[0] != ref_name)
    ref = None
    if os.path.exists(ref_path):
        z = np.load(ref_path)
        ref = [z[f'c{k}'].astype(np.float32) for k in range(len(z.files))]
        if len(ref) < len(env.f0items):
            ref = None
    warm = False
    for name, fn in configs:
        path = os.path.join(d, f'{name}.json')
        if os.path.exists(path) and not (name == ref_name and ref is None):
            continue
        if not warm:  # the first call includes the GPU warm-up; keep it out of the timings
            fn(0)
            warm = True
        rows, mels = [], []
        for k in range(len(env.f0items)):
            item, mel, f0_used, secs = fn(k)
            row, _ = env.measure(item, mel, f0_used, None if name == ref_name else ref[k])
            row['s2_seconds'] = secs
            rows.append(row)
            if name == ref_name:
                mels.append(mel.astype(np.float16))
        if name == ref_name:
            np.savez_compressed(ref_path, **{f'c{k}': m for k, m in enumerate(mels)})
            ref = [m.astype(np.float32) for m in mels]
        s = summary_of(rows)
        save_json(path, dict(config=name, split=args.split, summary=s, rows=rows))
        print(f'{name}: ' + ', '.join(f'{k} {v:.4f}' for k, v in s.items() if k != 'n'), flush=True)
    table = {}
    for name, _ in configs:
        path = os.path.join(d, f'{name}.json')
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                table[name] = json.load(f)['summary']
    save_json(os.path.join(args.out_dir, f'summary_{part}_{args.split}.json'), table)
    keys = ['mel_l1', 'mel_l1_ref', 'detail_freq', 'detail_time', 'ffe_out', 'cents_out', 's2_seconds']
    print(f'\n{part} {args.split}: {len(env.f0items)} clips (reference {ref_name})')
    print(f"{'':16s}" + ''.join(f'{k:>12s}' for k in keys))
    for name, s in table.items():
        print(f'{name:16s}' + ''.join(f"{s.get(k, float('nan')):12.4f}" for k in keys))


def part_stage2():
    env = Env()
    es = ints(args.stage2_steps)
    with env.model.ema_scope():
        configs = [(f'e{e}', lambda k, e=e: env.given(k, None, e)) for e in es]
        configs.append((f'e{es[1]}_seed1', lambda k: env.given(k, None, es[1], seed=1)))
        run_configs(env, 'stage2', configs, f'e{es[0]}')


def part_pipeline():
    env = Env()
    ks, es = ints(args.grid_f0), ints(args.grid_e)
    curves = {(k, 0): load_curves(k, 0, env)[0] for k in ks}
    curves[(ks[0], 1)] = load_curves(ks[0], 1, env)[0]
    with env.model.ema_scope():
        configs = [(f'f{k}_e{e}', lambda i, k=k, e=e: env.given(i, curves[(k, 0)][i], e)) for k in ks for e in es]
        configs.append((f'f{ks[0]}_e{es[0]}_seed1', lambda i: env.given(i, curves[(ks[0], 1)][i], es[0], seed=1)))
        run_configs(env, 'pipeline', configs, f'f{ks[0]}_e{es[0]}')


# ---------------------------------------------------------------------------------------------------------- listen
def plot_curves(path, item, sung, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ldm.data.phone_set import MIDI_PAD_ID
    semi = lambda f: np.where(f > 0, 69 + 12 * np.log2(np.maximum(f, 1e-3) / 440.0), np.nan)
    t = np.arange(len(sung)) / 187.5
    rows = dict(sung=sung, **curves)
    fig, axes = plt.subplots(len(rows), 1, figsize=(max(8, t[-1] * 1.6), 2.0 * len(rows)), sharex=True, squeeze=False)
    midi = item['midi'].astype(float)
    for ax, (label, f0) in zip(axes[:, 0], rows.items()):
        ax.plot(t, np.where((midi > 0) & (midi < MIDI_PAD_ID), midi, np.nan), color='0.6', lw=3)
        ax.plot(t[:len(f0)], semi(f0[:len(t)]), lw=1, color='tab:red' if label == 'sung' else 'tab:blue')
        ax.set_ylabel(label, fontsize=7)
    axes[0, 0].set_title(item['name'], fontsize=8)
    axes[-1, 0].set_xlabel('s')
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def write_wav(path, wav):
    sf.write(path, np.clip(wav, -1, 1), SR, subtype='PCM_16')


def part_listen():
    env = Env()
    ks = sorted({k for k, _ in pairs}, reverse=True)
    curves = {k: load_curves(k, 0, env)[0] for k in ks}
    d = os.path.join(args.out_dir, 'listen', args.split)
    os.makedirs(d, exist_ok=True)
    rows = []
    with env.model.ema_scope():
        for i in range(min(args.n_listen, len(env.f0items))):
            item = env.item(i)
            name = safe_name(item['name'])
            n = item['image'].shape[1]
            path = os.path.join(d, f'{name}_recording_voc.wav')
            if not os.path.exists(path):
                write_wav(path, env.synth.vocoder.vocode(item['image'][:, :n], f0=item['f0'][0, :n]))
            for k, e in pairs:
                path = os.path.join(d, f'{name}_f0s{k}_e{e}.wav')
                if os.path.exists(path):
                    continue
                it, mel, f0_used, secs = env.given(i, curves[k][i], e)
                row, wav = env.measure(it, mel, f0_used)
                write_wav(path, wav)
                rows.append(dict(row, f0_steps=k, e=e, file=os.path.basename(path)))
            plot_curves(os.path.join(d, f'{name}_f0.png'), env.f0items[i], item['f0'][0, :n],
                        {f'F0 gen {k} steps': curves[k][i] for k in ks})
    if rows:
        old = []
        mpath = os.path.join(d, 'metrics.json')
        if os.path.exists(mpath):
            old = json.load(open(mpath, encoding='utf-8'))
        save_json(mpath, old + rows)
    # the example scores through the inference pipeline
    from ldm.infer.pipeline import Singer
    sd = os.path.join(args.out_dir, 'listen', 'scores')
    os.makedirs(sd, exist_ok=True)
    singer = None
    info_rows = []
    for score in args.scores.split(','):
        for k, e in pairs:
            path = os.path.join(sd, f'{score}_f0s{k}_e{e}.wav')
            if os.path.exists(path):
                continue
            if singer is None:
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    singer = Singer(ckpt=args.stage2_ckpt, model=env.model, step=env.step, f0_model=args.f0_model)
            wav, info, _ = singer.sing(f'examples/{score}.json', seed=0, steps=e + 1, f0_steps=k)
            write_wav(path, wav)
            info_rows.append(dict(file=os.path.basename(path), f0_steps=k, e=e, caption=info['caption'], measured=info.get('measured')))
            print(os.path.basename(path), info.get('measured'), flush=True)
    if info_rows:
        mpath = os.path.join(sd, 'metrics.json')
        old = json.load(open(mpath, encoding='utf-8')) if os.path.exists(mpath) else []
        save_json(mpath, old + info_rows)
    print('wrote', d, sd)


# ----------------------------------------------------------------------------------------------------------- speed
def part_speed():
    from ldm.infer.pipeline import Singer
    from ldm.stage2_samples import load_stage2
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        model, step = load_stage2(args.stage2_ckpt, args.stage2_config)
        singer = Singer(ckpt=args.stage2_ckpt, model=model, step=step, f0_model=args.f0_model)
    clock = {}

    def wrap(obj, attr, key):
        fn = getattr(obj, attr)

        def inner(*a, **kw):
            out, secs = timed(lambda: fn(*a, **kw))
            clock[key] = clock.get(key, 0.0) + secs
            return out
        setattr(obj, attr, inner)
    wrap(singer, 'generate_f0', 'f0gen')
    wrap(model, 'synthesize', 'stage2')
    wrap(singer.vocoder, 'vocode', 'vocoder')
    prepared = singer.prepare(args.speed_score)
    audio_s = prepared.n_frames * HOP / SR
    settings = list(pairs) + [(k, pairs[0][1]) for k in ints(args.grid_f0)] + [(pairs[0][0], e) for e in ints(args.stage2_steps)]
    settings = list(dict.fromkeys(settings))
    for _ in range(2):  # warm-up
        singer.sing(args.speed_score, measure=False)
    table = {}
    for k, e in settings:
        runs = []
        for _ in range(args.repeats):
            clock.clear()
            _, secs = timed(lambda: singer.sing(args.speed_score, measure=False, steps=e + 1, f0_steps=k))
            runs.append(dict(clock, total=secs))
        table[f'f{k}_e{e}'] = {key: float(np.median([r[key] for r in runs])) for key in runs[0]}
        table[f'f{k}_e{e}']['realtime_factor'] = audio_s / table[f'f{k}_e{e}']['total']
    out = dict(score=args.speed_score, audio_seconds=audio_s, gpu=torch.cuda.get_device_name(), repeats=args.repeats, table=table)
    save_json(os.path.join(args.out_dir, 'speed.json'), out)
    print(f'{args.speed_score}: {audio_s:.1f} s of audio, {out["gpu"]}, median of {args.repeats}')
    print(f"{'':12s}{'f0gen':>9s}{'stage2':>9s}{'vocoder':>9s}{'total':>9s}{'x real':>9s}")
    for name, r in table.items():
        print(f"{name:12s}{r['f0gen']:9.3f}{r['stage2']:9.3f}{r['vocoder']:9.3f}{r['total']:9.3f}{r['realtime_factor']:9.1f}")


dict(f0ref=part_f0ref, stage2=part_stage2, pipeline=part_pipeline, listen=part_listen, speed=part_speed)[args.part]()
