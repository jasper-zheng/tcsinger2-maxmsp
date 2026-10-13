import ast, os, sys, collections, numpy as np, pandas as pd
df = pd.read_csv(sys.argv[1], sep='\t', low_memory=False)
print('rows', len(df), '| hours', round(df['duration'].sum() / 3600, 2))
print('by split:', {k: (len(g), round(g['duration'].sum() / 3600, 2)) for k, g in df.groupby('split')})
print('by language:', {k: (len(g), round(g['duration'].sum() / 3600, 2)) for k, g in df.groupby('language')})
split_of = dict(zip(df['mel_path'], df['split'])); bad = collections.Counter(); nprompt = []
for r in df.itertuples():
    pr = ast.literal_eval(r.prompt_mel); nprompt.append(len(pr))
    if r.mel_path in pr: bad['self prompt'] += 1
    if any(p not in split_of for p in pr): bad['prompt not in manifest'] += 1
    elif any(split_of[p] != r.split for p in pr): bad['prompt from other split'] += 1
    if any(os.path.dirname(p) != os.path.dirname(r.mel_path) for p in pr): bad['prompt from other group'] += 1
    n = len(ast.literal_eval(r.ph))
    if any(len(ast.literal_eval(getattr(r, c))) != n for c in ('ph_durs', 'ep_pitches', 'ep_notedurs', 'ep_types')): bad['list length'] += 1
# the raw GTSinger wavs of the train split were deleted to save space (2026-10-10), so audio is only required outside train
train = df['split'] == 'train'
miss = sum(not os.path.exists(p) for c in ('mel_path', 'f0_path') for p in df[c]) + sum(not os.path.exists(p) for p in df.loc[~train, 'audio_path'])
print('violations:', dict(bad) or 'none', '| missing files:', miss, '| prompts per clip: min %d median %d' % (min(nprompt), np.median(nprompt)))
print('train wavs on disk: %d of %d' % (sum(os.path.exists(p) for p in df.loc[train, 'audio_path']), train.sum()))
print('n_frames: min %d max %d | technique top:' % (df['n_frames'].min(), df['n_frames'].max()), df['technique'].value_counts().head(6).to_dict())
rng = np.random.default_rng(0); vals = []
for p in rng.choice(df['mel_path'].values, 300, replace=False):
    m = np.load(p); assert m.shape[0] == 80 and np.isfinite(m).all(); vals.append((m.min(), m.max(), m.mean()))
v = np.array(vals); print('mel sample (300 clips): min %.2f max %.2f mean %.2f' % (v[:, 0].min(), v[:, 1].max(), v[:, 2].mean()))
vr = collections.defaultdict(list)
for r in df.sample(900, random_state=0).itertuples():
    f = np.load(r.f0_path); vr[r.language].append((np.mean(f > 0), np.median(f[f > 0]) if (f > 0).any() else 0, abs(len(f) - int(float(r.duration) * 187.5)) if r.n_frames < 3750 else 0))
print('F0 sample: voiced share / median Hz / max length diff per language:', {k: (round(float(np.mean([a for a, _, _ in x])), 2), int(np.median([b for _, b, _ in x])), int(max(c for _, _, c in x))) for k, x in sorted(vr.items())})
