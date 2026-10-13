"""Exports a VAE checkpoint (WP4) for stage 2: the weights without the training loss (discriminator, logvar).

Stage 2 builds the VAE with an Identity loss (first_stage_config in configs/tcsinger2.yaml) and loads the
file through AutoencoderKL.init_from_ckpt. The duration predictor stays in; ldm/infer/timing.py uses it.
The export is checked by loading it strictly into a VAE built from --config.

Usage: python scripts/export_vae.py --ckpt logs/<run>/checkpoints/trainstep_checkpoints/<name>.ckpt
"""
import argparse
import os

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', default='useful_ckpts/vae_gtsinger.ckpt')
    parser.add_argument('--config', default='configs/tcsinger2.yaml', help='stage-2 config; its first_stage_config must fit the export')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    sd = {k: v for k, v in ckpt['state_dict'].items() if not k.startswith('loss.')}
    step = int(ckpt['global_step'])

    cfg = OmegaConf.load(args.config).model.params.first_stage_config
    cfg.params.pop('ckpt_path', None)
    instantiate_from_config(cfg).load_state_dict(sd, strict=True)

    tmp = args.out + '.tmp'
    torch.save(dict(state_dict=sd, global_step=step, batches=step // 2, source=args.ckpt), tmp)
    os.replace(tmp, args.out)
    print(f'wrote {args.out}: {len(sd)} tensors, {sum(v.numel() for v in sd.values()) / 1e6:.1f} M values, '
          f'{os.path.getsize(args.out) / 2 ** 30:.2f} GB, from batch {step // 2:,} of {args.ckpt}')
