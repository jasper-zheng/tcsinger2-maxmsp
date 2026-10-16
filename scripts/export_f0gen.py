"""Writes an F0 generator checkpoint as a small file with the EMA weights as the network's own (WP10).

The export holds `state_dict` (net.* only), `f0gen_config` (the F0Gen arguments) and `global_step`;
ldm.f0gen.module.load_f0gen reads it, as it reads training checkpoints and milestones.

Usage (from tcsinger2-reproduce/, PYTHONPATH=.):
  python scripts/export_f0gen.py --ckpt logs/<run>/checkpoints/trainstep_checkpoints/<milestone>.ckpt [--out useful_ckpts/f0gen_gtsinger.ckpt]
"""
import argparse

import torch

from ldm.f0gen.module import load_f0gen

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True)
parser.add_argument('--out', default='useful_ckpts/f0gen_gtsinger.ckpt')
args = parser.parse_args()

state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
model = load_f0gen(args.ckpt, device='cpu')
model.ema.copy_to(model.net)
torch.save(dict(state_dict={f'net.{k}': v for k, v in model.net.state_dict().items()}, f0gen_config=dict(state['hyper_parameters']),
                global_step=int(state.get('global_step', 0)), source=args.ckpt), args.out)
again = load_f0gen(args.out, device='cpu')
assert all(torch.equal(a, b) for a, b in zip(again.net.state_dict().values(), model.net.state_dict().values()))
print(f'wrote {args.out}: step {model.step}, objective {model.objective}, from {args.ckpt}')
