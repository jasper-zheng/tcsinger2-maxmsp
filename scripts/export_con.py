"""Exports the Custom Audio Encoder (WP5): the trained parts of a checkpoint, without the frozen VAE.

The file holds the speech encoder and the content branch. A model is rebuilt from it by setting both paths in
the config of ldm.models.autoencoder1d_con.AutoencoderKL: ckpt_path (the VAE the encoder was trained on) and
con_ckpt_path (this export). The export is checked by building the model that way and comparing every trained
weight with the checkpoint's.

Usage: python scripts/export_con.py --ckpt logs/<run>/checkpoints/trainstep_checkpoints/<milestone>.ckpt
"""
import argparse
import os

import torch
from omegaconf import OmegaConf

from ldm.models.autoencoder1d_con import FROZEN
from ldm.util import instantiate_from_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', default='useful_ckpts/audio_encoder_gtsinger.ckpt')
    parser.add_argument('--config', default='configs/ae_con.yaml')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    sd = {k: v for k, v in ckpt['state_dict'].items() if not k.startswith(FROZEN)}
    step = int(ckpt['global_step'])
    cfg = OmegaConf.load(args.config).model
    vae = cfg.params.ckpt_path

    tmp = args.out + '.tmp'
    torch.save(dict(state_dict=sd, global_step=step, source=args.ckpt, vae=vae, vae_source=torch.load(vae, map_location='cpu').get('source')), tmp)
    cfg.params.con_ckpt_path = tmp
    own = instantiate_from_config(cfg).state_dict()
    trained = [k for k in own if not k.startswith(FROZEN)]
    assert sorted(trained) == sorted(sd) and all(torch.equal(own[k], sd[k]) for k in sd)
    os.replace(tmp, args.out)
    print(f'wrote {args.out}: {len(sd)} tensors, {sum(v.numel() for v in sd.values()) / 1e6:.1f} M values, '
          f'{os.path.getsize(args.out) / 2 ** 30:.2f} GB, from batch {step:,} of {args.ckpt}; VAE {vae}')
