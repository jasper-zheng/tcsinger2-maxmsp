"""Vocode ground-truth mel + F0 with the HiFi-GAN vocoder, to check that the extracted features
match what the vocoder expects.

For a few clips per language it writes <name>_gt.wav (original, loudness-normalised like the mel
input) and <name>_voc.wav, and reports
  * mel L1 between the input mel and the mel of the vocoded audio (log10 units)
  * F0 frame error (voicing mismatch or > 20 % F0 deviation) between the input F0 and RMVPE on the
    vocoded audio, and the F0 deviation in cents on frames voiced in both.

Usage: python scripts/vocode_gt.py --manifest data/sing/new.tsv --out_dir samples/wp2_resynth
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
import torch

from preprocess.NAT_mel import MelNet
from utils.rmvpe import RMVPE
from vocoder.hifigan.hifigan_nsf import HifiGAN_NSF

SR, HOP = 48000, 256
MEL_ARGS = dict(audio_sample_rate=SR, audio_num_mel_bins=80, fft_size=1024, win_size=1024, hop_size=HOP, fmin=20, fmax=24000)


def f0_metrics(ref, est):
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    v_ref, v_est = ref > 0, est > 0
    both = v_ref & v_est
    gross = both & (np.abs(est - ref) > 0.2 * np.where(both, ref, 1))
    ffe = (np.sum(v_ref != v_est) + np.sum(gross)) / n
    cents = 1200 * np.abs(np.log2(est[both] / ref[both])) if both.any() else np.zeros(1)
    return float(ffe), float(np.sum(v_ref != v_est) / n), float(np.median(cents))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='data/sing/new.tsv')
    parser.add_argument('--split', default='valid')
    parser.add_argument('--per_language', type=int, default=2)
    parser.add_argument('--out_dir', default='samples/wp2_resynth')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--group_by', default='language', help='column to draw --per_language clips from, e.g. dataset')
    args = parser.parse_args()

    df = pd.read_csv(args.manifest, sep='\t', low_memory=False)
    df = df[(df['split'] == args.split) & df['duration'].between(5, 15)]
    picks = pd.concat([g.sample(min(args.per_language, len(g)), random_state=args.seed) for _, g in df.groupby(args.group_by)])

    vocoder = HifiGAN_NSF('useful_ckpts/hifigan')
    mel_net = MelNet(MEL_ARGS)
    mel_net.to('cuda')  # MelNet.to() works in place and returns nothing
    rmvpe = RMVPE('useful_ckpts/rmvpe/model.pt')
    meter = pyln.Meter(SR)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for _, r in picks.iterrows():
        mel, f0 = np.load(r['mel_path']), np.load(r['f0_path'])
        n = min(mel.shape[1], len(f0))
        mel, f0 = mel[:, :n], f0[:n]
        wav = vocoder.vocode(mel, f0=f0)
        with torch.no_grad():
            mel_voc = mel_net(wav.astype(np.float32)).squeeze(0).cpu().numpy()
        m = min(n, mel_voc.shape[1])
        mel_l1 = float(np.mean(np.abs(mel[:, :m] - mel_voc[:, :m])))
        f0_voc, _ = rmvpe.get_pitch(wav.astype(np.float32), SR, HOP, len(wav) // HOP)
        ffe, vde, cents = f0_metrics(f0, f0_voc)

        gt, _ = sf.read(r['audio_path'], dtype='float32')
        gt = pyln.normalize.loudness(gt, meter.integrated_loudness(gt), -23.0)
        peak = np.abs(gt).max()
        gt = gt / peak if peak > 1 else gt
        name = r['item_name'].replace('#', '_').replace(' ', '_')
        sf.write(os.path.join(args.out_dir, name + '_gt.wav'), gt[:n * HOP], SR, subtype='PCM_16')
        sf.write(os.path.join(args.out_dir, name + '_voc.wav'), np.clip(wav, -1, 1), SR, subtype='PCM_16')
        rows.append(dict(language=r['language'], singer=r['singer'], item=r['item_name'], seconds=round(n * HOP / SR, 1),
                         mel_l1=round(mel_l1, 3), ffe=round(ffe, 3), voicing_err=round(vde, 3), f0_cents_median=round(cents, 1),
                         mel_min=round(float(mel.min()), 2), mel_max=round(float(mel.max()), 2)))
        print(rows[-1])

    res = pd.DataFrame(rows)
    print(res.groupby('language')[['mel_l1', 'ffe', 'voicing_err', 'f0_cents_median']].mean().round(3))
    print('overall:', res[['mel_l1', 'ffe', 'voicing_err', 'f0_cents_median']].mean().round(3).to_dict())
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
