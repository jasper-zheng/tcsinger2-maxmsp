"""Export `score2f0` (EWP5 of EXPORTING_WORK_PACKAGES.md) to tcsinger2_f0.pte for neural.gen~, on MLX.

    [neural.gen~ tcsinger2_f0.pte score2f0 f0]

Inputs, in order ("The interface"):
  ph, midi, note_dur, note_type, ph_dur, nucleus, note_id   condition [1, 256] (int64; note_dur, ph_dur float32)
  language                                                  condition [1, 1] int64
  f0_noise                                                  noise [1, 2, 3760]
Output: f0 [1, 1, 1003] at 50 Hz, MIDI pitch, 0 = unvoiced (the F0 buffer~ that `sing` takes).
The module is export.modules.score2f0.F0Pipeline; `--f0_steps` Euler steps of the F0 generator (16, D6).

Writes <out>.pte, <out>.json and <out>.export.json.

Usage (from TCSinger2/, in envs/et151, see export/requirements.txt):
  envs/et151/bin/python export/export_f0.py [--f0_steps 16] [--out export/artifacts/tcsinger2_f0]
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402

from export.export_sing import DESCRIPTIONS, INT_CONDS  # noqa: E402
from export.export_sing import INPUTS as SING_INPUTS  # noqa: E402

OUT = 'export/artifacts/tcsinger2_f0'
INPUTS = SING_INPUTS[:8] + ['f0_noise']


def build(f0_steps):
    from neural_tilde import GenModule
    from export.modules.load import load_f0net, load_vae
    from export.modules.score2f0 import F0Pipeline
    from export.modules.sing import F0_RATE, F0_SAMPLES
    with contextlib.redirect_stdout(io.StringIO()):
        vae, (f0net, _) = load_vae(), load_f0net()
    pipe = F0Pipeline(vae, f0net, f0_steps).eval()
    gm = GenModule(pipe)
    for name in INPUTS[:7]:
        gm.register_condition(name, [1, 256], 'int64' if name in INT_CONDS else 'float32', DESCRIPTIONS[name])
    gm.register_condition('language', [1, 1], 'int64', DESCRIPTIONS['language'])
    gm.register_noise('f0_noise', [1, 2, 3760])
    gm.register_method('score2f0', inputs=INPUTS, out_channels=1, out_length=F0_SAMPLES, out_sample_rate=F0_RATE, test_method=True)
    return gm


def check_metadata(path):
    """The metadata against PROTOCOL.md: kind gen, no signal or buffer input, the inputs in order with their roles."""
    meta = json.load(open(path))
    problems = []
    if meta['kind'] != 'gen':
        problems.append(f"kind {meta['kind']}")
    m = meta['methods']['score2f0']
    names = [i['name'] for i in m['inputs']]
    if names != INPUTS:
        problems.append(f'inputs {names}')
    roles = {i['name']: i['role'] for i in m['inputs']}
    want = dict(f0_noise='noise')
    problems += [f'{k} is {roles[k]}' for k in roles if roles[k] != want.get(k, 'condition')]
    noise = next(i for i in m['inputs'] if i['name'] == 'f0_noise')
    if noise['shape'] != [1, 2, 3760]:
        problems.append(f"f0_noise shape {noise['shape']}")
    if m['output']['shape'] != [1, 1, 1003] or m['output']['sample_rate'] != 50:
        problems.append(f"output {m['output']}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--f0_steps', type=int, default=16, help='Euler steps of the F0 generator (D6)')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)  # export_to_pte does not create it
    t0 = time.time()
    from export import mlx_slots
    mlx_slots.apply()  # temp-slot reuse (EWP4): without it a call holds all its intermediates
    gm = build(args.f0_steps)
    t1 = time.time()
    path = gm.export_to_pte(args.out, delegate='mlx', decompose_conv=True)
    t2 = time.time()
    meta_path = path[:-4] + '.json'
    problems = check_metadata(meta_path)
    info = dict(pte=path, mb=round(os.path.getsize(path) / 2**20, 1), f0_steps=args.f0_steps, build_seconds=round(t1 - t0, 1),
                export_seconds=round(t2 - t1, 1), metadata_problems=problems)
    with open(path[:-4] + '.export.json', 'w') as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))
    if problems:
        sys.exit('metadata: ' + '; '.join(problems))


if __name__ == '__main__':
    main()
