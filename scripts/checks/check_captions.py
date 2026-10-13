"""Checks the style captions of a training manifest (format in ldm/data/captions.py).

Usage (from TCSinger2/, PYTHONPATH=.):  python scripts/checks/check_captions.py data/sing/new.tsv [--no_encoder]
"""
import collections, re, sys, numpy as np, pandas as pd
from ldm.data.captions import (EMOTIONS, GENDERS, SINGING_METHODS, TIMBRE_TECHS, VOCAL_RANGES, caption_from_row,
                               parse_caption, parse_technique, row_fields, sample_caption)
T5 = 'useful_ckpts/flan-t5-large'; MAX_TOKENS = 80
df = pd.read_csv(sys.argv[1], sep='\t', low_memory=False, usecols=['item_name', 'singer', 'gender', 'vocal_range', 'emotion', 'singing_method', 'technique', 'caption', 'split'])
bad = collections.Counter()
for r in df.to_dict('records'):
    r['caption'] = r['caption'] if isinstance(r['caption'], str) else ''
    f = parse_caption(r['caption']); g, sep, v = f.get('singer', 'female').partition(' ')
    want = {k: x for k, x in row_fields(r).items() if x is not None}  # a field is expected when its column has a label
    if set(f) != set(want): bad['field missing'] += 1
    elif g not in GENDERS or (sep and v not in VOCAL_RANGES) or f.get('emotion', EMOTIONS[0]) not in EMOTIONS or f.get('singing_method', SINGING_METHODS[0]) not in SINGING_METHODS: bad['value outside vocabulary'] += 1
    elif f != want: bad['caption differs from columns'] += 1
    if r['caption'] != caption_from_row(r): bad['caption not reproducible'] += 1
    if r['technique'] != 'no': parse_technique(r['technique'])
    if r['singer'] in r['caption'] or re.search(r'[A-Z]{2}-\w+-\d', r['caption']): bad['singer ID in caption'] += 1
df['caption'] = df['caption'].fillna('')
print('fields per caption (clips):', dict(sorted(collections.Counter(len(parse_caption(c)) for c in df['caption']).items())))
print('clips', len(df), '| distinct captions', df['caption'].nunique(), '| distinct technique values', df['technique'].nunique(), '| violations:', dict(bad) or 'none')

from transformers import T5Tokenizer
tok = T5Tokenizer.from_pretrained(T5, legacy=True); caps = sorted(df['caption'].unique())
n_tok = [len(x) for x in tok(caps)['input_ids']]
print('T5 tokens per caption: min %d median %d max %d (limit %d) | longest: %s' % (min(n_tok), np.median(n_tok), max(n_tok), MAX_TOKENS, caps[int(np.argmax(n_tok))]))
assert max(n_tok) < MAX_TOKENS

parts = df['item_name'].str.split('#', expand=True)
gtsinger = parts.shape[1] > 4  # GTSinger item names carry the technique folder and the group
if gtsinger: df['tech_dir'], df['group'] = parts[2], parts[4]
if gtsinger: print('technique named in the caption, by group (share of clips):')
for group, tech in (('Mixed_Voice_Group', 'mix'), ('Falsetto_Group', 'falsetto'), ('Breathy_Group', 'breathy'), ('Pharyngeal_Group', 'pharyngeal')) if gtsinger else ():
    g = df[df['group'] == group]; ctrl = df[(df['group'] == 'Control_Group') & (df['tech_dir'] == g['tech_dir'].iloc[0])]
    has = lambda x: x['technique'].str.contains(tech).mean()
    print(f'  {group:18s} {tech:10s} {has(g):.3f}   its control group {has(ctrl):.3f}   (n {len(g)}, {len(ctrl)})')
if gtsinger:
    ctrl = df[(df['group'] == 'Control_Group') & (df['tech_dir'] == 'Mixed_Voice_and_Falsetto')]
    print('  control clips of Mixed_Voice_and_Falsetto that are `none`: %.3f (n %d)' % ((ctrl['technique'] == 'none').mean(), len(ctrl)))

train = df[df['split'] == 'train']
for col in ('technique', 'caption'):
    print(f'{col} values per split / of which unseen in train (clips):', {s: (g[col].nunique(), int((~g[col].isin(set(train[col]))).sum())) for s, g in df.groupby('split')})
print('technique counts:', df['technique'].value_counts().head(12).to_dict(), '| values under 20 clips:', int((df['technique'].value_counts() < 20).sum()))
for col, vals in (('emotion', EMOTIONS), ('singing_method', SINGING_METHODS)):
    print(col, 'by split:', pd.crosstab(df['split'], df[col]).to_dict('index'))
print('techniques in valid/test_seen/test_unseen:', {s: {t: int(g['technique'].str.contains(t).sum()) for t in TIMBRE_TECHS} for s, g in df.groupby('split') if s != 'train'})

rng = np.random.default_rng(0); rows = train.sample(min(2000, len(train)), random_state=0).to_dict('records'); n = 20000
drawn = [sample_caption(rows[i % len(rows)], rng, field_drop=0.1, caption_drop=0.2) for i in range(n)]
present = collections.Counter(k for c in drawn for k in parse_caption(c))
print('sample_caption(field_drop 0.1, caption_drop 0.2), %d draws: empty %.3f (expect 0.200) | field present (expect 0.720 for a field every clip has):' % (n, np.mean([c == '' for c in drawn])), {k: round(v / n, 3) for k, v in present.items()})
print('examples:', drawn[:4])

if '--no_encoder' not in sys.argv:
    import torch
    from ldm.modules.encoders.modules import FrozenTextVocalEmbedder
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = FrozenTextVocalEmbedder(version=T5, device=dev, max_length=MAX_TOKENS)
    z = torch.cat([enc(dict(caption=caps[i:i + 64] + ([''] if i == 0 else []), acoustic=None, name=None, infer=False))['caption'].float() for i in range(0, len(caps), 64)])
    assert z.shape[1:] == (MAX_TOKENS, 1024) and torch.isfinite(z).all(), z.shape
    pooled = torch.nn.functional.normalize(z.mean(1), dim=-1); sim = pooled @ pooled.T; sim.fill_diagonal_(-1)
    i, j = divmod(int(sim.argmax()), len(sim)); names = caps[:64] + [''] + caps[64:]
    print('encoder output', tuple(z.shape), '| cosine of mean-pooled embeddings: closest pair %.4f | median %.3f' % (sim.max(), sim[sim > -1].median()))
    print('  closest pair:', repr(names[i]), '<->', repr(names[j]))
    assert sim.max() < 1 - 1e-4, 'two different captions encode to the same vector'
