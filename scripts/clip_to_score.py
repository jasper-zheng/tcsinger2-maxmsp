"""Writes the JSON score of a dataset clip (WP7): its words, the notes as the manifest has them and how long
they are sung, with the clip's style. A way to get a score to start from, in any of the nine languages.

  --with_ph     also the annotated phonemes of every word (otherwise they are looked up from the text when sung)
  --with_durs   also the annotated phoneme lengths (the score then reproduces the clip's timing)
  --with_f0     also the clip's F0 curve as "f0": the sung performance, which the output then follows. The notes
                are the clip's, so the curve fits them; with --with_durs the phonemes are placed as sung too
  --f0_fps R    write that curve at R values per second (with "f0_fps") instead of the 187.5 of the model
  --list N      print N clip names of the split per language and exit

Usage (from TCSinger2/, PYTHONPATH=.):
  python scripts/clip_to_score.py --list 3
  python scripts/clip_to_score.py --name "German#DE-Tenor-1#Breathy#Wind#Breathy_Group#0000" --out examples/wind_de.json
  python scripts/clip_to_score.py --name "Chinese#ZH-Alto-1#Glissando#青花瓷#Glissando_Group#0004" --with_f0 --out examples/f0_glissando_zh.json
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())

from ldm.infer.score import F0_FPS, parse_score, score_from_row  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--manifest', default='data/sing/new.tsv')
parser.add_argument('--split', default='valid')
parser.add_argument('--name')
parser.add_argument('--out')
parser.add_argument('--with_ph', action='store_true')
parser.add_argument('--with_durs', action='store_true')
parser.add_argument('--with_f0', action='store_true')
parser.add_argument('--f0_fps', type=float)
parser.add_argument('--list', type=int)
args = parser.parse_args()

df = pd.read_csv(args.manifest, sep='\t', low_memory=False)
if args.list:
    part = df[(df['split'] == args.split) & df['duration'].between(5, 15)]
    for lang, g in part.groupby('language'):
        for name in g['item_name'].head(args.list):
            print(name)
    sys.exit()
rows = df[df['item_name'] == args.name]
if not len(rows) or not args.out:
    sys.exit(f'error: give --out and a --name of {args.manifest}; --list 3 prints some')
score = score_from_row(rows.iloc[0], with_ph=args.with_ph, with_durs=args.with_durs, with_f0=args.with_f0)
if args.with_f0 and args.f0_fps:
    from ldm.infer.f0_input import resample_f0
    score['f0'] = [round(float(v), 2) for v in resample_f0(score['f0'], F0_FPS, dst_fps=args.f0_fps)]
    score['f0_fps'] = args.f0_fps
parse_score(score)  # refuses what the front-end would refuse
with open(args.out, 'w', encoding='utf-8') as f:
    f.write('{' + ',\n '.join(f'{json.dumps(k)}: {json.dumps(score[k], ensure_ascii=False)}' for k in ('language', 'name', 'style')))
    f.write(',\n "words": [\n  ' + ',\n  '.join(json.dumps(w, ensure_ascii=False) for w in score['words']) + '\n ]')
    f.write(''.join(f',\n {json.dumps(k)}: {json.dumps(score[k])}' for k in ('f0_fps', 'f0') if k in score) + '}\n')
print('wrote', args.out)
