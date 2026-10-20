"""Does the singing audio prompt set the voice of stage 2? (2026-10-18; WP6/WP7 left this to WP8.)

One score per GTSinger singer of the valid split (annotated phonemes and lengths, F0 predicted, no caption) is sung
with a prompt of every singer of the same gender: the 18 training singers (prompts from their valid/test_seen clips)
and 10 unseen singers (GTSinger's two, the 8 of data/sing_ext test_unseen), and once with the blank prompt.
Each output mel is put to the singer of the nearest centroid among those candidates, by two proxies of the voice:

  vae   time mean of the VAE latent mean (what stage 2 reads from a prompt; not independent of the model)
  mfcc  mean and std of cepstral coefficients 1-20 of the log mel over non-silent frames, z-scored (model-free)

There is no speaker-verification model in the environment. Each proxy is first calibrated on recordings: the
leave-one-out accuracy of the same nearest-centroid rule on the real clips is its ceiling.

In GTSinger a singer sings one language in one range, so a proxy can tell singers apart by the language and register
of the score as well as by the voice. The `centred` figures take that out: per score, the similarity to each
singer's centroid is taken relative to its mean over all prompts of that score, so what is left is how far a prompt
moves the same score towards a singer. `pairwise`: how often the output with prompt p is closer to p's centroid than
the output of the same score with another prompt q is (chance 0.5).

Writes samples/wp7/prompt_check/{metrics.json,outputs.csv} and, for listening, audio/ with a female and a male score
sung with the blank prompt and four prompts, next to the first 8 s of each prompt's recording.
Usage (from TCSinger2/, PYTHONPATH=.; about 12 min on the A6000): python scripts/checks/check_prompt_voice.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.fft import dct

sys.path.insert(0, os.getcwd())

from ldm.infer.pipeline import MAX_PROMPT_FRAMES, SR, Singer  # noqa: E402
from ldm.infer.score import score_from_row  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--ckpt', default='useful_ckpts/tcsinger2_gtsinger.ckpt')
parser.add_argument('--out_dir', default='samples/wp7/prompt_check')
parser.add_argument('--ref_clips', type=int, default=30, help='recordings per singer for the centroids')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
rng = np.random.RandomState(args.seed)
os.makedirs(os.path.join(args.out_dir, 'audio'), exist_ok=True)

gt = pd.read_csv('data/sing/new.tsv', sep='\t', low_memory=False)
ext = pd.read_csv('data/sing_ext/new.tsv', sep='\t', low_memory=False)
seen = gt[gt['split'].isin(['valid', 'test_seen'])].assign(seen=True)
unseen = pd.concat([gt[gt['split'] == 'test_unseen'], ext[ext['split'] == 'test_unseen']]).assign(seen=False)
pool = pd.concat([seen, unseen], ignore_index=True)
pool = pool[pool['duration'].between(3, 15)]

# per singer: one score (GTSinger valid, seen singers only), two prompt clips (>= 6 s), reference clips
scores, prompts, refs = {}, {}, {}
for singer, g in pool.groupby('singer'):
    g = g.sample(frac=1, random_state=rng)
    if g['seen'].iloc[0]:
        cand = g[(g['split'] == 'valid') & g['duration'].between(4, 12)]
        scores[singer] = cand.iloc[0]
        g = g[g['item_name'] != cand.iloc[0]['item_name']]
    p = g[g['duration'] >= 6].iloc[:2]
    prompts[singer] = [r for _, r in p.iterrows()]
    refs[singer] = g[~g['item_name'].isin(p['item_name'])].iloc[:args.ref_clips]
gender = pool.groupby('singer')['gender'].first().to_dict()
is_seen = pool.groupby('singer')['seen'].first().to_dict()
groups = {sex: sorted(s for s in gender if gender[s] == sex) for sex in ('female', 'male')}
print({k: len(v) for k, v in groups.items()}, 'candidate singers;', len(scores), 'scores')

singer_model = Singer(args.ckpt)
vae = singer_model.model.first_stage_model


@torch.no_grad()
def emb_vae(mel):
    x = torch.from_numpy(mel[:, :mel.shape[1] // 4 * 4]).float().to(singer_model.model.device)[None]
    return vae.quant_conv(vae.encoder(x))[0, :vae.embed_dim].mean(1).cpu().numpy()


def emb_mfcc(mel):
    c = dct(mel, type=2, axis=0, norm='ortho')[1:21]
    keep = mel.mean(0) > -4.0  # non-silent frames (floor is -5)
    c = c[:, keep] if keep.sum() > 20 else c
    return np.concatenate([c.mean(1), c.std(1)])


def embed(mel):
    return dict(vae=emb_vae(mel), mfcc=emb_mfcc(mel))


ref_emb = {s: [embed(np.load(p)) for p in refs[s]['mel_path']] for s in gender}
mfcc_all = np.stack([e['mfcc'] for s in ref_emb for e in ref_emb[s]])
mfcc_mu, mfcc_sd = mfcc_all.mean(0), mfcc_all.std(0) + 1e-6
norm = dict(vae=lambda v: v, mfcc=lambda v: (v - mfcc_mu) / mfcc_sd)
unit = lambda v: v / (np.linalg.norm(v) + 1e-9)
proxies = ('vae', 'mfcc')
centroids = {k: {s: unit(np.mean([unit(norm[k](e[k])) for e in ref_emb[s]], 0)) for s in gender} for k in proxies}


def ranked(e, cands, k, exclude=None):
    """Candidates sorted by cosine to their centroid; `exclude` = (singer, index) leaves a clip out of its centroid."""
    v = unit(norm[k](e[k]))
    sims = {}
    for s in cands:
        c = centroids[k][s]
        if exclude is not None and exclude[0] == s:
            others = [unit(norm[k](x[k])) for i, x in enumerate(ref_emb[s]) if i != exclude[1]]
            c = unit(np.mean(others, 0))
        sims[s] = float(v @ c)
    return sorted(sims, key=sims.get, reverse=True), sims


metrics = dict(ckpt=args.ckpt, candidates={k: v for k, v in groups.items()},
               unseen=[s for s in gender if not is_seen[s]], calibration={}, outputs={})
for k in proxies:  # the proxy on recordings: leave-one-out nearest centroid within the gender group
    cal = {}
    for sex, cands in groups.items():
        hits = {'seen': [], 'unseen': []}
        for s in cands:
            for i, e in enumerate(ref_emb[s]):
                hits['seen' if is_seen[s] else 'unseen'].append(ranked(e, cands, k, (s, i))[0][0] == s)
        cal[sex] = {kind: round(float(np.mean(h)), 3) for kind, h in hits.items()}
        cal[sex]['chance'] = round(1 / len(cands), 3)
    metrics['calibration'][k] = cal
print('calibration on recordings:', json.dumps(metrics['calibration']))

rows = []
listen = {sex: next(s for s in sorted(scores) if gender[s] == sex) for sex in groups}
for n, (score_singer, row) in enumerate(sorted(scores.items())):
    sex = gender[score_singer]
    score = score_from_row(row, with_ph=True, with_durs=True)
    score['style'] = {}  # no caption: the prompt alone
    prepared = singer_model.prepare(score)
    for prompt_singer in [None] + groups[sex]:
        prow = prompts[prompt_singer][n % len(prompts[prompt_singer])] if prompt_singer else None
        wav, info, curves = singer_model.sing(prepared, prompt=prow['mel_path'] if prow is not None else None,
                                              seed=args.seed, measure=False)
        assert info['caption'] == ''
        e = embed(curves['mel'])
        out = dict(score=row['item_name'], score_singer=score_singer, gender=sex,
                   prompt_singer=prompt_singer or '', prompt=prow['item_name'] if prow is not None else '',
                   prompt_seen=is_seen[prompt_singer] if prompt_singer else None)
        for k in proxies:
            order, sims = ranked(e, groups[sex], k)
            out[f'{k}_sims'] = sims
            out[f'{k}_top'] = order[0]
            if prompt_singer:
                out[f'{k}_rank'] = order.index(prompt_singer) + 1
                out[f'{k}_margin'] = sims[prompt_singer] - np.mean([v for s, v in sims.items() if s != prompt_singer])
        rows.append(out)
        if listen[sex] == score_singer:
            chosen = [s for s in groups[sex] if is_seen[s] and s != score_singer][:2] + [s for s in groups[sex] if not is_seen[s]][:2]
            if prompt_singer is None or prompt_singer in chosen:
                tag = f'{sex}_{prompt_singer or "blank"}'
                sf.write(os.path.join(args.out_dir, 'audio', f'{tag}_out.wav'), wav, SR)
                if prow is not None:
                    a, sr = sf.read(prow['audio_path'], dtype='float32', always_2d=True)
                    sf.write(os.path.join(args.out_dir, 'audio', f'{tag}_prompt.wav'), a.mean(1)[:int(sr * MAX_PROMPT_FRAMES * 256 / SR)], sr)
    print(f'{n + 1}/{len(scores)} {score_singer}', flush=True)

with open(os.path.join(args.out_dir, 'outputs.json'), 'w') as f:  # with the similarity to every candidate's centroid
    json.dump(rows, f, default=lambda v: v.item() if hasattr(v, 'item') else v)
df = pd.DataFrame(rows)
df.drop(columns=[f'{k}_sims' for k in proxies]).to_csv(os.path.join(args.out_dir, 'outputs.csv'), index=False)
prompted, blank = df[df['prompt_singer'] != ''], df[df['prompt_singer'] == '']


def centred(sub, k, cands):
    """Per output of `sub`: rank of the prompt singer after centring each centroid's similarity over the prompts
    of the same score, and the pairwise wins against the other prompts of that score."""
    ranks, wins = [], []
    for _, g in prompted[prompted.score.isin(sub.score)].groupby('score'):
        m = np.array([[r[f'{k}_sims'][c] for c in cands] for _, r in g.iterrows()])  # prompts x centroids
        c_m = m - m.mean(0, keepdims=True)
        for i, (_, r) in enumerate(g.iterrows()):
            if r.name not in sub.index:
                continue
            j = cands.index(r.prompt_singer)
            ranks.append(int((c_m[i] > c_m[i, j]).sum()) + 1)
            wins += [m[i, j] > m[q, j] for q in range(len(g)) if q != i]
    return np.array(ranks), np.array(wins)


for k in proxies:
    res = {}
    for sex in groups:
        for kind, sub in (('seen', prompted[(prompted.gender == sex) & (prompted.prompt_seen == True)]),  # noqa: E712
                          ('unseen', prompted[(prompted.gender == sex) & (prompted.prompt_seen == False)])):  # noqa: E712
            other = sub[sub.prompt_singer != sub.score_singer]
            res[f'{sex}_{kind}'] = dict(
                n=len(sub), prompt_singer_top1=round(float((sub[f'{k}_rank'] == 1).mean()), 3),
                prompt_singer_top3=round(float((sub[f'{k}_rank'] <= 3).mean()), 3),
                mean_rank=round(float(sub[f'{k}_rank'].mean()), 2),
                score_singer_top1_when_other_prompt=round(float((other[f'{k}_top'] == other.score_singer).mean()), 3),
                chance=round(1 / len(groups[sex]), 3))
            ranks, wins = centred(sub, k, groups[sex])
            res[f'{sex}_{kind}'].update(centred_top1=round(float((ranks == 1).mean()), 3),
                                        centred_top3=round(float((ranks <= 3).mean()), 3),
                                        pairwise=round(float(wins.mean()), 3))
        b = blank[blank.gender == sex]
        res[f'{sex}_blank'] = dict(n=len(b), score_singer_top1=round(float((b[f'{k}_top'] == b.score_singer).mean()), 3))
    metrics['outputs'][k] = res
with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=1)
print(json.dumps(metrics['outputs'], indent=1))
print('wrote', args.out_dir)
