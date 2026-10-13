"""Evaluates a VAE checkpoint (WP4) on the validation split.

  * mel L1 (log10 units) of the reconstruction of every full validation clip, per language, decoding
    the posterior mean and a posterior sample (stage 2 trains on samples); also per third of the mel bins
  * detail ratio: mean absolute difference between neighbouring mel bins (and frames) of the
    reconstruction over that of the input; 1 = as much fine structure as the input, below 1 = smoothed
  * latent statistics: per-channel mean / std, posterior std, KL per latent frame
  * padding: clips padded with the mel floor, as stage 2 pads its batches, must reconstruct the same
    audio and a floor-level tail
  * vocoded reconstruction on the clips of scripts/vocode_gt.py (WP2): mel L1 and F0 frame error of the
    re-analysed audio, to compare with vocoding the ground-truth mel; writes <name>_rec.wav
    (the matching <name>_gt.wav / <name>_voc.wav are in samples/wp2_resynth)
  * duration predictor: absolute error of the predicted phoneme durations

Usage: python scripts/eval_vae.py --ckpt logs/<run>/checkpoints/trainstep_checkpoints/<name>.ckpt
"""
import argparse
import ast
import json
import os
import re

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from omegaconf import OmegaConf

from ldm.data.phone_set import ph2id
from ldm.util import instantiate_from_config
from preprocess.NAT_mel import MelNet
from scripts.vocode_gt import HOP, MEL_ARGS, SR, f0_metrics
from utils.rmvpe import RMVPE
from vocoder.hifigan.hifigan_nsf import HifiGAN_NSF

MEL_FLOOR = -5.0
BANDS = dict(low=slice(0, 27), mid=slice(27, 54), high=slice(54, 80))  # mel bins


def load_mel(path, factor=4):
    mel = np.load(path)
    return torch.from_numpy(mel[:, :mel.shape[1] // factor * factor]).float().cuda()[None]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--config', default='configs/ae_singing.yaml')
    parser.add_argument('--manifest', default='data/sing/new.tsv')
    parser.add_argument('--split', default='valid')
    parser.add_argument('--out_dir', default='samples/wp4_recon')
    parser.add_argument('--max_clips', type=int, default=0, help='0 = the whole split')
    parser.add_argument('--per_language', type=int, default=2, help='clips to vocode, as in vocode_gt.py')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--group_by', default='language', help='column to draw --per_language clips from, e.g. dataset')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    step = int(ckpt.get('global_step', 0))
    model = instantiate_from_config(OmegaConf.load(args.config).model)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model = model.cuda().eval()
    out_dir = os.path.join(args.out_dir, f'step{step:08d}')
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    df = pd.read_csv(args.manifest, sep='\t', low_memory=False)
    df = df[df['split'] == args.split].reset_index(drop=True)
    clips = df.sample(args.max_clips, random_state=args.seed) if args.max_clips else df

    rows, z_sum, z_sq, z_n, post_std, kl = [], 0, 0, 0, [], []
    dur_err, clip_dur_err = [], []
    with torch.no_grad():
        for r in clips.itertuples():
            x = load_mel(r.mel_path)
            post = model.encode(x)
            rec = model.decode(post.mode())
            err = (rec - x).abs()
            l1_mode = err.mean().item()
            l1_sample = (model.decode(post.sample()) - x).abs().mean().item()
            z = post.mode()[0]
            z_sum, z_sq, z_n = z_sum + z.sum(1).double(), z_sq + (z ** 2).sum(1).double(), z_n + z.shape[1]
            post_std.append(post.std.mean().item())
            kl.append(post.kl().item() / z.shape[1])
            rows.append(dict(language=r.language, singer=r.singer, dataset=getattr(r, 'dataset', 'GTSinger'), frames=x.shape[2], l1_mode=l1_mode, l1_sample=l1_sample,
                             **{f'l1_{b}': err[:, sl].mean().item() for b, sl in BANDS.items()},
                             df_rec=rec.diff(dim=1).abs().mean().item(), df_gt=x.diff(dim=1).abs().mean().item(),
                             dt_rec=rec.diff(dim=2).abs().mean().item(), dt_gt=x.diff(dim=2).abs().mean().item()))

            ph = torch.tensor([[ph2id[p] for p in ast.literal_eval(r.ph)]]).cuda()
            feats = [torch.tensor([ast.literal_eval(getattr(r, c))]).cuda() for c in ('ep_notedurs', 'ep_pitches', 'ep_types')]
            pred = torch.expm1(model.pred_duration(ph, feats[0].float(), feats[1].long(), feats[2].long(),
                                                   torch.zeros_like(ph, dtype=torch.bool)))[0].clamp(min=0).cpu().numpy()
            true = np.array(ast.literal_eval(r.ph_durs))
            dur_err += list(np.abs(pred - true))
            clip_dur_err.append(abs(pred.sum() - true.sum()) / true.sum())

        # padded input: 30 clips, each extended by half its length with the mel floor
        pad_rows = []
        for r in clips.head(30).itertuples():
            x = load_mel(r.mel_path)
            n = x.shape[2]
            x_pad = torch.nn.functional.pad(x, (0, n // 8 * 4), value=MEL_FLOOR)
            rec, rec_pad = model.decode(model.encode(x).mode()), model.decode(model.encode(x_pad).mode())
            pad_rows.append(dict(real_part_vs_unpadded=(rec_pad[..., :n - 16] - rec[..., :n - 16]).abs().mean().item(),
                                 tail_vs_floor=(rec_pad[..., n + 16:] - MEL_FLOOR).abs().mean().item()))

        # vocoded reconstructions of the WP2 clips
        vocoder, rmvpe = HifiGAN_NSF('useful_ckpts/hifigan'), RMVPE('useful_ckpts/rmvpe/model.pt')
        mel_net = MelNet(MEL_ARGS)
        mel_net.to('cuda')
        cand = df[df['duration'].between(5, 15)]
        picks = pd.concat([g.sample(min(args.per_language, len(g)), random_state=args.seed) for _, g in cand.groupby(args.group_by)])
        voc_rows = []
        for _, r in picks.iterrows():
            x, f0 = load_mel(r['mel_path']), np.load(r['f0_path'])
            n = min(x.shape[2], len(f0))
            mel, f0 = x[0, :, :n].cpu().numpy(), f0[:n]
            rec = model.decode(model.encode(x).mode())[0, :, :n].cpu().numpy()
            wav = vocoder.vocode(rec, f0=f0)
            mel_voc = mel_net(wav.astype(np.float32)).squeeze(0).cpu().numpy()
            m = min(n, mel_voc.shape[1])
            f0_voc, _ = rmvpe.get_pitch(wav.astype(np.float32), SR, HOP, len(wav) // HOP)
            ffe, vde, cents = f0_metrics(f0, f0_voc)
            name = r['item_name'].replace('#', '_').replace(' ', '_')
            sf.write(os.path.join(out_dir, name + '_rec.wav'), np.clip(wav, -1, 1), SR, subtype='PCM_16')
            voc_rows.append(dict(language=r['language'], item=r['item_name'], mel_l1_rec=float(np.mean(np.abs(mel - rec))),
                                 mel_l1_vocoded=float(np.mean(np.abs(mel[:, :m] - mel_voc[:, :m]))), ffe=ffe, voicing_err=vde,
                                 f0_cents_median=cents))

    res, voc, pad = pd.DataFrame(rows), pd.DataFrame(voc_rows), pd.DataFrame(pad_rows)
    w = res['frames'] / res['frames'].sum()
    z_mean = (z_sum / z_n).cpu().numpy()
    z_std = np.sqrt((z_sq / z_n).cpu().numpy() - z_mean ** 2)
    by_lang = res.groupby('language')[['l1_mode', 'l1_sample']].mean().round(4)
    by_singer = res.groupby('singer')['l1_mode'].mean().round(4)
    by_dataset = res.groupby('dataset')['l1_mode'].agg(['mean', 'count']).round(4)
    wmean = lambda c: float((res[c] * w).sum())
    summary = dict(
        ckpt=args.ckpt, global_step=step, batches=step // 2, split=args.split, clips=len(res),
        mel_l1_mode=round(float((res['l1_mode'] * w).sum()), 4), mel_l1_sample=round(float((res['l1_sample'] * w).sum()), 4),
        mel_l1_mode_by_language=by_lang['l1_mode'].to_dict(), mel_l1_mode_by_singer=by_singer.to_dict(), mel_l1_mode_by_dataset=by_dataset['mean'].to_dict(),
        mel_l1_by_band={b: round(wmean(f'l1_{b}'), 4) for b in BANDS},
        detail_ratio_freq=round(wmean('df_rec') / wmean('df_gt'), 4), detail_ratio_time=round(wmean('dt_rec') / wmean('dt_gt'), 4),
        latent=dict(mean=[round(float(v), 3) for v in z_mean], std=[round(float(v), 3) for v in z_std],
                    overall_std=round(float(np.sqrt((z_std ** 2 + z_mean ** 2).mean() - z_mean.mean() ** 2)), 3),
                    posterior_std=round(float(np.mean(post_std)), 4), kl_per_frame=round(float(np.mean(kl)), 2)),
        padding={k: round(float(v), 4) for k, v in pad.mean().items()},
        vocoded={k: round(float(v), 4) for k, v in voc[['mel_l1_rec', 'mel_l1_vocoded', 'ffe', 'voicing_err', 'f0_cents_median']].mean().items()},
        duration=dict(phoneme_abs_err_ms_median=round(1000 * float(np.median(dur_err)), 1),
                      phoneme_abs_err_ms_mean=round(1000 * float(np.mean(dur_err)), 1),
                      clip_length_rel_err_median=round(float(np.median(clip_dur_err)), 3)))
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(summary=summary, vocoded_clips=voc_rows), f, ensure_ascii=False, indent=1)

    print(f"\n{args.ckpt}\nbatches {step // 2} | {len(res)} {args.split} clips")
    print(f"mel L1: posterior mean {summary['mel_l1_mode']} | posterior sample {summary['mel_l1_sample']}")
    print(by_lang.T.to_string())
    print('mel L1 by band:', summary['mel_l1_by_band'], '| detail ratio: freq', summary['detail_ratio_freq'],
          'time', summary['detail_ratio_time'])
    print('by corpus (mean over clips, clips):', {k: (v['mean'], int(v['count'])) for k, v in by_dataset.iterrows()})
    print('by singer (min, max):', by_singer.idxmin(), by_singer.min(), '|', by_singer.idxmax(), by_singer.max())
    print('latent:', {k: v for k, v in summary['latent'].items() if k not in ('mean', 'std')},
          '| channel std min %.2f max %.2f' % (z_std.min(), z_std.max()))
    print('padding:', summary['padding'])
    print('vocoded reconstruction (WP2 clips; vocoding the ground-truth mel gave mel L1 0.125, FFE 0.013):', summary['vocoded'])
    print('duration predictor:', summary['duration'])
    print('wrote', out_dir)
