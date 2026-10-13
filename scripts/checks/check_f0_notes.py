"""How good are note pitches taken from F0 (WP9, OpenSinger)? Measured on clips that have a score.

The held-out clips of a corpus with annotated notes (default M4Singer) are reduced to what an
aligned OpenSinger clip has: one piece per phoneme (note splits merged), every pitch 0. The pitches
are then filled by postprocess.fill_pitches and compared with the score, weighted by time.

Usage (from TCSinger2/, PYTHONPATH=.):  python scripts/checks/check_f0_notes.py --out data/meta_ext/f0_notes_report.json
"""
import argparse
import ast
import collections
import json

import numpy as np
import pandas as pd

from preprocess import postprocess

REST = ('<AP>', '<SP>')

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', default='data/sing_ext/new.tsv')
parser.add_argument('--dataset', default='M4Singer')
parser.add_argument('--splits', nargs='+', default=['valid', 'test_seen', 'test_unseen'])
parser.add_argument('--out', default='data/meta_ext/f0_notes_report.json')
args = parser.parse_args()
postprocess.ARGS = argparse.Namespace(min_voiced_frames=3, min_voiced_ratio=0.3)
FPS = 48000 / 256

df = pd.read_csv(args.manifest, sep='\t', low_memory=False)
df = df[(df['dataset'] == args.dataset) & df['split'].isin(args.splits) & (df['note_source'] == 'score')]
tot = collections.Counter()
for row in df.to_dict('records'):
    ph, durs, words, score = (ast.literal_eval(row[c]) for c in ('ph', 'ph_durs', 'ph2words', 'ep_pitches'))
    # one piece per phoneme, as in an aligned OpenSinger clip
    m_ph, m_durs, m_words, parts = [], [], [], []
    for k, p in enumerate(ph):
        if m_ph and p == ph[k - 1] and words[k] == words[k - 1] and p not in REST:
            m_durs[-1] += durs[k]
            parts[-1].append(k)
        else:
            m_ph.append(p), m_durs.append(durs[k]), m_words.append(words[k]), parts.append([k])
    blank = dict(ph=m_ph, ph_durs=m_durs, ph2words=m_words, ep_pitches=[0] * len(m_ph), ep_notedurs=list(m_durs),
                 ep_types=[1 if p in REST else 2 for p in m_ph], language='Chinese', note_source='f0')
    stats = collections.defaultdict(collections.Counter)
    filled, _, types = postprocess.fill_pitches(blank, np.load(row['f0_path']), FPS, stats)
    for p, est, ks, t in zip(m_ph, filled, parts, types):
        if p in REST:
            continue
        tot['phonemes'] += 1
        tot['phonemes_multi_note'] += len({score[k] for k in ks}) > 1
        tot['phonemes_no_pitch'] += est == 0
        for k in ks:
            if score[k] <= 0:
                continue
            d = abs(est - score[k]) if est > 0 else None
            tot['s'] += durs[k]
            tot['s_no_pitch'] += durs[k] * (d is None)
            tot['s_exact'] += durs[k] * (d == 0)
            tot['s_within1'] += durs[k] * (d is not None and d <= 1)
            tot['s_within2'] += durs[k] * (d is not None and d <= 2)
            tot['s_octave'] += durs[k] * (d is not None and 11 <= d <= 13)
    # words whose initial and final got different pitches (the score gives both the same note)
    by_word = collections.defaultdict(set)
    for p, est, w in zip(m_ph, filled, m_words):
        if p not in REST and est > 0:
            by_word[w].add(est)
    tot['words'] += len(by_word)
    tot['words_two_pitches'] += sum(len(v) > 1 for v in by_word.values())

s = tot['s']
report = dict(manifest=args.manifest, dataset=args.dataset, splits=args.splits, clips=len(df), sung_hours=round(s / 3600, 2),
              time_exact=round(tot['s_exact'] / s, 4), time_within_1_semitone=round(tot['s_within1'] / s, 4),
              time_within_2_semitones=round(tot['s_within2'] / s, 4), time_octave_off=round(tot['s_octave'] / s, 4),
              time_without_pitch=round(tot['s_no_pitch'] / s, 4),
              phonemes_with_several_score_notes=round(tot['phonemes_multi_note'] / tot['phonemes'], 4),
              words_with_two_pitches=round(tot['words_two_pitches'] / tot['words'], 4))
print(json.dumps(report, indent=1))
with open(args.out, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=1)
