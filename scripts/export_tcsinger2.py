"""Writes the weights of a stage-2 checkpoint (WP6) to one file for inference: the transformer, its EMA
copy and the latent scale, without optimizer state. The VAE and the text encoder are not inside; they
are loaded from the paths in configs/tcsinger2.yaml (ldm.stage2_samples.load_stage2 reads the file).

Usage: python scripts/export_tcsinger2.py --ckpt logs/<run>/checkpoints/trainstep_checkpoints/<milestone>.ckpt
"""
import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True)
parser.add_argument('--out', default='useful_ckpts/tcsinger2_gtsinger.ckpt')
args = parser.parse_args()

state = torch.load(args.ckpt, map_location='cpu')
sd = {k: v for k, v in state['state_dict'].items() if not k.startswith(('first_stage_model.', 'cond_stage_model.'))}
assert any(k.startswith('model.diffusion_model.') for k in sd) and 'scale_factor' in sd
torch.save(dict(state_dict=sd, global_step=state['global_step'], source=args.ckpt), args.out)
print(f"{args.out}: step {state['global_step']}, {sum(v.numel() for v in sd.values()) / 1e6:.1f} M values, "
      f"scale_factor {float(sd['scale_factor']):.4f}")
