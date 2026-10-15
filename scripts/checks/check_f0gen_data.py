"""Checks of the F0 generator's data (WP10).

  notes      the note numbers inferred from the manifest pieces (ldm.f0gen.features.note_index) against the
             notes of the GTSinger annotations (<clip>.json next to the wav): share of clips with the same
             number of sung notes, and share of annotated note onsets with an inferred onset within 30 ms
  roundtrip  encode then decode gives the F0 back (voicing exact, pitch within 0.01 cent) on manifest clips
  dataset    a few items and a batch of ldm.f0gen.dataset.F0Specs, with their shapes and value ranges
  overfit    --ckpt of a run trained on data.params.max_items=64: its curves for those 64 training clips
             against the sung ones (F0 frame error, voicing error, cents, vibrato figures)

Usage (from tcsinger2-reproduce/, PYTHONPATH=.):
  python scripts/checks/check_f0gen_data.py [--part notes roundtrip dataset] [--n 2000]
  python scripts/checks/check_f0gen_data.py --part overfit --ckpt logs/<run>_overfit/checkpoints/<ckpt>
"""
import argparse
import ast
import json
import os

import numpy as np
import pandas as pd

from ldm.f0gen import features

parser = argparse.ArgumentParser()
parser.add_argument('--part', nargs='+', default=['notes', 'roundtrip', 'dataset'])
parser.add_argument('--n', type=int, default=2000, help='clips per check')
parser.add_argument('--ckpt', help='for --part overfit')
args = parser.parse_args()
rng = np.random.RandomState(0)
lit = ast.literal_eval


def load(path, cols):
    df = pd.read_csv(path, sep='\t', usecols=lambda c: c in cols)
    return df.iloc[rng.permutation(len(df))[:args.n]]


if 'notes' in args.part:
    df = load('data/sing/new.tsv', ['ph', 'ph2words', 'ep_pitches', 'ep_notedurs', 'ph_durs', 'audio_path', 'language'])
    same = total = onsets = hits = 0
    by_lang = {}
    for row in df.itertuples():
        js = row.audio_path[:-4] + '.json'
        if not os.path.exists(js):
            continue
        ph, w, p, nd, d = (lit(getattr(row, k)) for k in ('ph', 'ph2words', 'ep_pitches', 'ep_notedurs', 'ph_durs'))
        ids = features.note_index(ph, w, p, nd)
        starts = np.concatenate([[0], np.cumsum(d)])
        inferred = sorted({starts[i] for i, k in enumerate(ids) if p[i] > 0 and (i == 0 or ids[i - 1] != k)})
        notes = [(s, n) for word in json.load(open(js, encoding='utf-8')) for s, n in zip(word['note_start'], word['note'])
                 if n > 0 and word['word'] not in ('<AP>', '<SP>')]
        ok = len(inferred) == len(notes)
        same += ok
        total += 1
        lang = by_lang.setdefault(row.language, [0, 0])
        lang[0] += ok
        lang[1] += 1
        # annotation times start at the clip's first word; manifests at 0 as well
        t0 = json.load(open(js, encoding='utf-8'))[0]['start_time']
        for s, _ in notes:
            onsets += 1
            hits += bool(inferred) and np.min(np.abs(np.asarray(inferred) - (s - t0))) <= 0.03
    print(f'notes: {total} GTSinger clips; same number of sung notes in {same / total:.1%}; '
          f'annotated onsets with an inferred onset within 30 ms: {hits / onsets:.1%} of {onsets}')
    print('  same count by language: ' + ', '.join(f'{k} {a / b:.1%} ({b})' for k, (a, b) in sorted(by_lang.items())))

if 'roundtrip' in args.part:
    worst_cents, voicing_bad, n = 0.0, 0, 0
    for path in ('data/sing/new.tsv', 'data/sing_ext/new.tsv'):
        df = load(path, ['ph', 'ph2words', 'ep_pitches', 'ep_notedurs', 'ph_durs', 'ep_types', 'f0_path'])
        for row in df.head(300).itertuples():
            f0 = np.load(row.f0_path)
            ph, w, p, nd, d, ty = (lit(getattr(row, k)) for k in ('ph', 'ph2words', 'ep_pitches', 'ep_notedurs', 'ph_durs', 'ep_types'))
            tr = features.note_tracks([0] * len(ph), d, p, ty, features.note_index(ph, w, p, nd), len(f0))
            ref = features.reference_pitch(tr['midi'])
            r, v, trust = features.encode(f0, ref)
            back = features.decode(r, v, ref)
            voicing_bad += np.sum((back > 0) != (f0 > 0))
            keep = (f0 > 0) & (trust > 0)
            if keep.any():
                worst_cents = max(worst_cents, float(np.max(np.abs(1200 * np.log2(back[keep] / f0[keep])))))
            n += 1
    print(f'roundtrip: {n} clips; voicing differences {voicing_bad}; worst pitch difference {worst_cents:.4f} cents')
    assert voicing_bad == 0 and worst_cents < 0.01

if 'dataset' in args.part:
    from ldm.f0gen.dataset import F0Specs
    ds = F0Specs('valid', main_spec_dir_path='data/sing,data/sing_ext', other_spec_dir_path='', max_tokens=60000)
    items = [ds[i] for i in range(4)]
    for k, v in items[0].items():
        print(f'  {k}: {getattr(v, "shape", v) if not isinstance(v, str) else v}')
    batch = ds.collater(items)
    for k, v in batch.items():
        if hasattr(v, 'shape'):
            print(f'  batch {k}: {tuple(v.shape)} {v.dtype} [{float(v.min()):.2f}, {float(v.max()):.2f}]')

if 'overfit' in args.part:
    from ldm.f0gen.dataset import F0Specs, collate
    from ldm.f0gen.metrics import curve_rows, summarise
    from ldm.f0gen.module import load_f0gen
    model = load_f0gen(args.ckpt)
    ds = F0Specs('train', main_spec_dir_path='data/sing,data/sing_ext', other_spec_dir_path='', max_prompt_len=1500,
                 max_items=64, prompt_drop=0.0)
    ds.split = 'valid'  # the first prompt, from its start, as validation reads it
    items = [ds[i] for i in range(len(ds))]
    for label, curves in (('sung', [it['f0'] for it in items]), ('model', None)):
        if curves is None:
            curves = []
            for k in range(0, len(items), 16):
                part = items[k:k + 16]
                f0 = model.generate(collate(part), steps=32, seed=k).float().cpu().numpy()
                curves += [f0[i, :len(it['r'])] for i, it in enumerate(part)]
        clips, notes = [], []
        for it, f0 in zip(items, curves):
            clip, nts = curve_rows(f0, it['midi'], it['onset'], it['f0'])
            clips.append(clip)
            notes += nts
        s = summarise(clips, notes)
        print(f'overfit, {label} ({model.objective}, step {model.step}, {len(items)} training clips): ' +
              ', '.join(f'{k} {s[k]}' for k in ('ffe', 'voicing_err', 'cents', 'notes_within_1', 'vibrato_share',
                                               'vib_extent_median', 'r_std')))
