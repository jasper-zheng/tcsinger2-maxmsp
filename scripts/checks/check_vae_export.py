"""Checks the exported VAE (WP4) the way stage 2 loads it: builds the first stage from configs/tcsinger2.yaml
(so through AutoencoderKL.init_from_ckpt with the config's ckpt_path) and verifies that

  * every weight of the model equals the source milestone's (nothing left at its random initialisation)
  * it reconstructs the validation split with the mel L1 that scripts/eval_vae.py reported for the milestone
  * and prints the latent std, whose inverse is what stage 2 should pick up as scale_factor (scale_by_std)

Nothing is written. Usage: python scripts/checks/check_vae_export.py [--max_clips 200]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/tcsinger2.yaml')
    parser.add_argument('--manifest', default='data/sing/new.tsv')
    parser.add_argument('--max_clips', type=int, default=0, help='0 = the whole validation split')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config).model.params.first_stage_config
    export = torch.load(cfg.params.ckpt_path, map_location='cpu')
    print(f"{cfg.params.ckpt_path}: batch {export['batches']:,} of {export['source']}")
    model = instantiate_from_config(cfg)

    source = torch.load(export['source'], map_location='cpu')['state_dict']
    own = model.state_dict()
    missing = [k for k in own if k not in source]
    differ = [k for k in own if k in source and not torch.equal(own[k], source[k])]
    unused = [k for k in source if k not in own and not k.startswith('loss.')]
    print(f'{len(own)} tensors in the model | not in the milestone: {missing} | different: {differ} | milestone weights not loaded: {unused}')
    assert not missing and not differ and not unused

    model = model.cuda().eval()
    df = pd.read_csv(args.manifest, sep='\t', low_memory=False)
    df = df[df['split'] == 'valid']
    df = df.sample(args.max_clips, random_state=0) if args.max_clips else df
    err, frames, zs = 0.0, 0, []
    with torch.no_grad():
        for path in df['mel_path']:
            mel = np.load(path)
            x = torch.from_numpy(mel[:, :mel.shape[1] // 4 * 4]).float().cuda()[None]
            z = model.encode(x).mode()
            err, frames = err + (model.decode(z) - x).abs().mean().item() * x.shape[2], frames + x.shape[2]
            zs.append(z[0].flatten().cpu())
    std = torch.cat(zs).std().item()
    print(f'mel L1 on {len(df)} validation clips: {err / frames:.4f}')
    metrics = os.path.join('samples/wp4_recon', f"step{export['global_step']:08d}", 'metrics.json')
    if os.path.exists(metrics) and not args.max_clips:
        ref = json.load(open(metrics))['summary']['mel_l1_mode']
        print(f'eval_vae.py reported {ref} for the milestone')
        assert abs(err / frames - ref) < 1e-3
    print(f'latent std over all values {std:.3f} -> scale_factor {1 / std:.4f}')
