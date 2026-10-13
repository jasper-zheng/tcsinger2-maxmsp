"""Mel spectrograms of VAE reconstructions (WP4) next to the ground truth, for comparing checkpoints by eye.

One row per clip (from the clips of scripts/vocode_gt.py), one column for the input and one per checkpoint;
each panel shows a few seconds from the middle of the clip.

Usage: python scripts/compare_vae_mels.py --ckpts <a.ckpt> <b.ckpt> [--out samples/wp4_recon/mel_compare.png]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from scripts.vocode_gt import HOP, SR

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts', nargs='+', required=True)
    parser.add_argument('--config', default='configs/ae_singing.yaml')
    parser.add_argument('--manifest', default='data/sing/new.tsv')
    parser.add_argument('--languages', nargs='+', default=['Chinese', 'English', 'Italian'])
    parser.add_argument('--seconds', type=float, default=3.0)
    parser.add_argument('--out', default='samples/wp4_recon/mel_compare.png')
    args = parser.parse_args()

    df = pd.read_csv(args.manifest, sep='\t', low_memory=False)
    df = df[(df['split'] == 'valid') & df['duration'].between(5, 15)]
    picks = [df[df['language'] == lang].sample(1, random_state=0).iloc[0] for lang in args.languages]
    mels = [np.load(r['mel_path']) for r in picks]
    mels = [m[:, :m.shape[1] // 4 * 4] for m in mels]

    columns = [('input', mels)]
    for path in args.ckpts:
        ckpt = torch.load(path, map_location='cpu')
        model = instantiate_from_config(OmegaConf.load(args.config).model)
        model.load_state_dict(ckpt['state_dict'], strict=True)
        model = model.cuda().eval()
        with torch.no_grad():
            recs = [model.decode(model.encode(torch.from_numpy(m).float().cuda()[None]).mode())[0].cpu().numpy() for m in mels]
        columns.append((f"batch {int(ckpt['global_step']) // 2:,}", recs))
        del model

    n = int(args.seconds * SR / HOP)
    fig, axes = plt.subplots(len(picks), len(columns), figsize=(6 * len(columns), 3.2 * len(picks)), squeeze=False)
    for i, r in enumerate(picks):
        start = max(0, (mels[i].shape[1] - n) // 2)
        for j, (label, col) in enumerate(columns):
            m = col[i][:, start:start + n]
            l1 = '' if j == 0 else f' | L1 {np.abs(col[i] - mels[i]).mean():.3f}'
            axes[i, j].imshow(m, origin='lower', aspect='auto', vmin=-5, vmax=1.5, cmap='magma', interpolation='nearest',
                              extent=(start * HOP / SR, (start + m.shape[1]) * HOP / SR, 0, 80))
            axes[i, j].set_title(f"{label}{l1}", loc='left', fontsize=10)
            axes[i, j].set_ylabel(f"{r['language']}, {r['singer']}\nmel bin" if j == 0 else '')
    for ax in axes[-1]:
        ax.set_xlabel('seconds')
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print('wrote', args.out, '|', [r['item_name'] for r in picks])
