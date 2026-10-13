"""The example scores that carry an F0 curve (examples/f0_*.json, examples/clip_de_with_f0.json), WP7.

Every score is sung twice with the same seed, into <out_dir>/<score>_given.wav and <score>_predicted.wav
(+ .json, _f0.png): with its "f0", and with the F0 predicted from the notes (--no_f0).
Reported per score:
  F0 frame error and median pitch deviation of both outputs against the curve of the score: the output
  follows the curve if the first run is close to it and the second is not;
  how much the pitch moves at 4 to 8 Hz (vibrato and fast slides; RMS in cents) in the curve and in each
  output, over the stretches of at least 0.4 s in which both are voiced;
  what check_f0 says about the curve against the notes, and the warnings.

Usage (from TCSinger2/, PYTHONPATH=.): python scripts/checks/check_f0_examples.py
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ldm.infer import pipeline
from ldm.infer.timing import FPS
from ldm.stage2_samples import f0_metrics

MIN_RUN = 75  # frames, 0.4 s
EDGE = 15     # frames left out at both ends of a voiced stretch (onset and release)


def movement(curve, out, sos=butter(2, [4, 8], btype='band', fs=FPS, output='sos')):
    """RMS (cents) of the 4 to 8 Hz part of the pitch of two F0 tracks, over the same frames: the stretches of
    at least 0.4 s in which both are voiced."""
    n = min(len(curve), len(out))
    voiced = np.concatenate([[False], (curve[:n] > 0) & (out[:n] > 0), [False]])
    edges = np.flatnonzero(np.diff(voiced.astype(int)))
    parts = [[], []]
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= MIN_RUN:
            for part, f0 in zip(parts, (curve, out)):
                cents = 1200 * np.log2(f0[a:b])
                part.append(sosfiltfilt(sos, cents - cents.mean())[EDGE:-EDGE])
    return [round(float(np.sqrt(np.mean(np.concatenate(p) ** 2))), 1) if p else 0.0 for p in parts]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', nargs='*', default=sorted(glob.glob('examples/f0_*.json')) + ['examples/clip_de_with_f0.json'])
    parser.add_argument('--ckpt', default='useful_ckpts/tcsinger2_gtsinger.ckpt')
    parser.add_argument('--out_dir', default='samples/wp7/examples')
    args = parser.parse_args()

    singer = pipeline.Singer(args.ckpt)
    rows = []
    for score in args.scores:
        name = os.path.splitext(os.path.basename(score))[0]
        wav, info, curves = singer.sing(score)
        assert info['f0_source'] == 'score' and info['mode'] == 'given', f'{score} has no "f0"'
        pipeline.write_outputs(os.path.join(args.out_dir, f'{name}_given.wav'), wav, info, curves)
        given = curves['f0_used']
        wav, info_p, curves_p = singer.sing(score, f0=False)
        pipeline.write_outputs(os.path.join(args.out_dir, f'{name}_predicted.wav'), wav, info_p, curves_p)
        ffe_p, _, cents_p = f0_metrics(given, curves_p['f0_out'])
        rows.append(dict(score=name, clip=info['name'], seconds=round(info['seconds'], 2),
                         ffe_given=info['measured']['ffe_out_vs_f0_used'], cents_given=info['measured']['cents_out'],
                         ffe_predicted_vs_curve=round(ffe_p, 4), cents_predicted_vs_curve=round(cents_p, 1),
                         movement_curve_and_given=movement(given, curves['f0_out']),
                         movement_curve_and_predicted=movement(given, curves_p['f0_out']),
                         f0_check={k: round(v, 3) for k, v in info['f0_check'].items()}, warnings=info['warnings']))
        r = rows[-1]
        print(f"{name:22s} {r['seconds']:5.1f} s | with the curve: FFE {r['ffe_given']:.2%}, {r['cents_given']:.1f} cents | "
              f"F0 predicted, against the curve: FFE {r['ffe_predicted_vs_curve']:.2%}, {r['cents_predicted_vs_curve']:.1f} cents | "
              f"4-8 Hz movement, curve / output: {'/'.join(f'{v:.0f}' for v in r['movement_curve_and_given'])} cents with it, "
              f"{'/'.join(f'{v:.0f}' for v in r['movement_curve_and_predicted'])} predicted"
              + (f" | warnings: {r['warnings']}" if r['warnings'] else ''), flush=True)
    with open(os.path.join(args.out_dir, 'f0_examples.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
