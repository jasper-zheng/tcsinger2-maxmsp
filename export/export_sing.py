"""Export `sing` (EWP4 of EXPORTING_WORK_PACKAGES.md) to tcsinger2_sing.pte for neural.gen~, on MLX.

    [neural.gen~ tcsinger2_sing.pte sing out]

Inputs, in order ("The interface"):
  ph, midi, note_dur, note_type, ph_dur, nucleus, note_id   condition [1, 256] (int64; note_dur, ph_dur float32)
  language                                                  condition [1, 1] int64
  input_ids                                                 condition [1, 80] int64 (neural.tokenizer, max_length 80)
  cfg_scale                                                 attribute float32, default 3.0, 1 to 10
  f0                                                        buffer [1, 1, 1003]: MIDI pitch at 50 Hz, 0 = unvoiced,
                                                            copied as it is (resample false), padded with 0
  latent_noise                                              noise [1, 20, 940]
Output: audio [1, 1, 962560] at 48 kHz, silent after the phrase's last frame.
The module is export.modules.sing.SingPipeline; `--steps` points of the stage-2 Euler grid (8 = 7 steps, D6).

Writes <out>.pte, <out>.json and the tokenizer bundle (<out>.tokenizer.json, <out>.tokenizer.config.json).

Usage (from TCSinger2/, in envs/et151, see export/requirements.txt):
  envs/et151/bin/python export/export_sing.py [--steps 8] [--out export/artifacts/tcsinger2_sing]
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

OUT = 'export/artifacts/tcsinger2_sing'
TOKENIZER = 'useful_ckpts/flan-t5-large/tokenizer.json'
INT_CONDS = ('ph', 'midi', 'note_type', 'nucleus', 'note_id')
INPUTS = ['ph', 'midi', 'note_dur', 'note_type', 'ph_dur', 'nucleus', 'note_id', 'language', 'input_ids', 'cfg_scale', 'f0',
          'latent_noise']
DESCRIPTIONS = dict(
    ph='phoneme id per piece (data/meta/phone_set.json), 430 = padding', midi='note pitch per piece, 0 for rests, 99 = padding',
    note_dur="length of the piece's note in seconds", note_type='1 rest, 2 word on one note, 3 word on several notes, 4 = padding',
    ph_dur='length given by the score in seconds, -1 = predicted', nucleus='1 if the piece is a nucleus',
    note_id="index of the piece's note, -1 = padding", language='index in language_set (Chinese 0 ... Spanish 8)',
    input_ids='caption tokens (neural.tokenizer, flan-t5, 80)')


def build(steps):
    from neural_tilde import GenModule
    from export.modules.load import load_all
    from export.modules.sing import F0_RATE, F0_SAMPLES, N_SAMPLES, SingPipeline
    with contextlib.redirect_stdout(io.StringIO()):
        parts = load_all()
    pipe = SingPipeline(parts, steps).eval()
    gm = GenModule(pipe)
    for name in INPUTS[:7]:
        gm.register_condition(name, [1, 256], 'int64' if name in INT_CONDS else 'float32', DESCRIPTIONS[name])
    gm.register_condition('language', [1, 1], 'int64', DESCRIPTIONS['language'])
    gm.register_condition('input_ids', [1, 80], 'int64', DESCRIPTIONS['input_ids'])
    gm.register_attribute('cfg_scale', 3.0, 1.0, 10.0, 'classifier-free guidance on the caption')
    gm.register_buffer_input('f0', channels=1, length=F0_SAMPLES, sample_rate=F0_RATE, resample=False, pad_value=0.0,
                             description='F0 as MIDI pitch at 50 Hz, 0 = unvoiced (from score2f0, may be edited)')
    gm.register_noise('latent_noise', [1, 20, 940])
    gm.register_method('sing', inputs=INPUTS, out_channels=1, out_length=N_SAMPLES, out_sample_rate=48000, test_method=True)
    gm.register_tokenizer(TOKENIZER, max_length=80)
    return gm


def check_metadata(path):
    """The metadata against PROTOCOL.md: kind gen, no signal input, the inputs in order with their roles."""
    meta = json.load(open(path))
    problems = []
    if meta['kind'] != 'gen':
        problems.append(f"kind {meta['kind']}")
    m = meta['methods']['sing']
    names = [i['name'] for i in m['inputs']]
    if names != INPUTS:
        problems.append(f'inputs {names}')
    roles = {i['name']: i['role'] for i in m['inputs']}
    if 'signal' in roles.values():
        problems.append('a signal input')
    want = dict(cfg_scale='attribute', f0='buffer', latent_noise='noise')
    problems += [f'{k} is {roles[k]}' for k in roles if roles[k] != want.get(k, 'condition')]
    f0 = next(i for i in m['inputs'] if i['name'] == 'f0')
    if f0.get('resample') is not False or f0['shape'] != [1, 1, 1003] or f0['sample_rate'] != 50:
        problems.append(f'f0 spec {f0}')
    ids = next(i for i in m['inputs'] if i['name'] == 'input_ids')
    if ids['dtype'] != 'int64':
        problems.append('input_ids not int64')
    if m['output']['shape'] != [1, 1, 962560] or m['output']['sample_rate'] != 48000:
        problems.append(f"output {m['output']}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--steps', type=int, default=8, help='stage-2 points of linspace(0, 1, steps): steps - 1 Euler steps (D6)')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)  # export_to_pte does not create it
    t0 = time.time()
    from export import mlx_slots
    mlx_slots.apply()  # temp-slot reuse: without it the call holds all its intermediates (about 51 GB)
    gm = build(args.steps)
    t1 = time.time()
    path = gm.export_to_pte(args.out, delegate='mlx', decompose_conv=True)
    t2 = time.time()
    meta_path = path[:-4] + '.json'
    problems = check_metadata(meta_path)
    info = dict(pte=path, mb=round(os.path.getsize(path) / 2**20, 1), steps=args.steps, build_seconds=round(t1 - t0, 1),
                export_seconds=round(t2 - t1, 1), metadata_problems=problems)
    with open(path[:-4] + '.export.json', 'w') as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))
    if problems:
        sys.exit('metadata: ' + '; '.join(problems))


if __name__ == '__main__':
    main()
