"""Retrieval check of the Custom Audio Encoder (WP5): do matched singing and speech clips score highest?

Every clip of a split is encoded whole (no crop, no padding) and reduced to the embedding the contrastive
loss compares: the time mean of the latent mean, compared by cosine similarity. Singing goes through the
frozen VAE encoder, speech through the trained speech encoder; with --baseline speech goes through the VAE
encoder too, which is what stage 2 would do without this work package.

Reported per split (the triplet manifest data/con/new.tsv):
  singing -> speech   the paired speech clip is ranked first / in the top 10; the top-ranked speech clip is by the
                      right singer; the paired clip is first among the singer's own speech clips
  speech -> singing   the top-ranked singing clip is one of the clip's own (control or technique version); it is
                      by the right singer
  latents             std of the speech latents next to the singing latents, and how far the channel means are
                      apart (in units of the singing std), since stage 2 reads both through one projection

Writes samples/wp5_retrieval/step<N>/metrics.json (or baseline/metrics.json).

Usage: python scripts/eval_con.py --ckpt logs/<run>/checkpoints/trainstep_checkpoints/<milestone>.ckpt
       python scripts/eval_con.py --baseline
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ldm.models.autoencoder1d_con import FROZEN
from ldm.util import instantiate_from_config


@torch.no_grad()
def encode(paths, encoder, model):
    """Pooled embedding [n, C] of each clip, and the per-channel sum, sum of squares and count of all latent frames."""
    pooled, s1, s2, n = [], 0, 0, 0
    for path in paths:
        mel = np.load(path)
        x = torch.from_numpy(mel[:, :mel.shape[1] // 4 * 4]).float().cuda()[None]
        z = model.quant_conv(encoder(x))[0, :model.embed_dim]  # latent mean [C, T]
        pooled.append(z.mean(1))
        s1, s2, n = s1 + z.sum(1), s2 + (z ** 2).sum(1), n + z.shape[1]
    return torch.stack(pooled), (s1 / n, (s2 / n - (s1 / n) ** 2).sqrt())


def retrieval(z_sing, z_speech, sing, speech):
    sim = F.normalize(z_sing, dim=1) @ F.normalize(z_speech, dim=1).t()  # [singing, speech]
    rows = torch.arange(len(sing), device=sim.device)
    target = torch.tensor(sing['pair_id'].map({p: i for i, p in enumerate(speech['pair_id'])}).values, device=sim.device)
    same_singer = torch.tensor(sing['singer'].values[:, None] == speech['singer'].values[None], device=sim.device)
    top = sim.argmax(1)
    rank = (sim > sim[rows, target][:, None]).sum(1)
    within = sim.masked_fill(~same_singer, -2).argmax(1)
    top_sing = sim.argmax(0)  # for each speech clip
    cols = torch.arange(len(speech), device=sim.device)
    singer_right = same_singer[rows, top].float()
    m = lambda t: round(t.float().mean().item(), 4)
    return dict(
        singing_clips=len(sing), speech_clips=len(speech), singers=int(sing['singer'].nunique()),
        chance=dict(pair_top1=round(1 / len(speech), 4), singer_right=m(same_singer), pair_top1_within_singer=m(1 / same_singer.sum(1))),
        sing2speech=dict(pair_top1=m(top == target), pair_top10=m(rank < 10), singer_right=m(singer_right),
                         pair_top1_within_singer=m(within == target)),
        speech2sing=dict(pair_top1=m(target[top_sing] == cols), singer_right=m(same_singer[top_sing, cols])),
        cosine=dict(matched=m(sim[rows, target]), all=m(sim)),
        singer_right_by_singer={s: round(singer_right[torch.tensor(sing['singer'].values == s)].mean().item(), 3)
                                for s in sorted(sing['singer'].unique())},
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', help='a checkpoint of the run, or the export of scripts/export_con.py')
    parser.add_argument('--baseline', action='store_true', help='the VAE encoder for speech (no trained speech encoder)')
    parser.add_argument('--config', default='configs/ae_con.yaml')
    parser.add_argument('--manifest', default='data/con/new.tsv')
    parser.add_argument('--splits', nargs='+', default=['valid', 'test_unseen'])
    parser.add_argument('--out_dir', default='samples/wp5_retrieval')
    args = parser.parse_args()
    assert bool(args.ckpt) != args.baseline, 'give --ckpt or --baseline'

    model = instantiate_from_config(OmegaConf.load(args.config).model)
    if args.baseline:
        name, info = 'baseline', dict(speech_encoder='VAE encoder')
    else:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        assert not unexpected and all(k.startswith(FROZEN) for k in missing), (missing[:5], unexpected[:5])
        name, info = f"step{int(ckpt['global_step']):08d}", dict(ckpt=args.ckpt, global_step=int(ckpt['global_step']))
    model = model.cuda().eval()
    speech_encoder = model.encoder if args.baseline else model.speech_encoder

    df = pd.read_csv(args.manifest, sep='\t', low_memory=False)
    results = dict(info)
    for split in args.splits:
        sing = df[df['split'] == split].reset_index(drop=True)
        speech = sing.drop_duplicates('pair_id').reset_index(drop=True)
        z_sing, (mean_sing, std_sing) = encode(sing['mel_path'], model.encoder, model)
        z_speech, (mean_speech, std_speech) = encode(speech['speech_mel_path'], speech_encoder, model)
        r = retrieval(z_sing, z_speech, sing, speech)
        r['latents'] = dict(singing_std=round(std_sing.mean().item(), 3), speech_std=round(std_speech.mean().item(), 3),
                            channel_mean_distance=round(((mean_speech - mean_sing) / std_sing).pow(2).mean().sqrt().item(), 3))
        results[split] = r
        a, b, c = r['sing2speech'], r['speech2sing'], r['chance']
        print(f"{split}: {r['singing_clips']} singing, {r['speech_clips']} speech clips, {r['singers']} singers\n"
              f"  singing -> speech: pair first {a['pair_top1']:.1%} (chance {c['pair_top1']:.1%}), in top 10 {a['pair_top10']:.1%}, "
              f"right singer {a['singer_right']:.1%} (chance {c['singer_right']:.1%}), pair first within singer "
              f"{a['pair_top1_within_singer']:.1%} (chance {c['pair_top1_within_singer']:.1%})\n"
              f"  speech -> singing: own clip first {b['pair_top1']:.1%}, right singer {b['singer_right']:.1%}\n"
              f"  cosine matched {r['cosine']['matched']:.3f}, all pairs {r['cosine']['all']:.3f} | latents: {r['latents']}")

    out = os.path.join(args.out_dir, name)
    os.makedirs(out, exist_ok=True)
    json.dump(results, open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    print('wrote', os.path.join(out, 'metrics.json'))
