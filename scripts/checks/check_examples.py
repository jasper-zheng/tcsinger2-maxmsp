"""The example scores of examples/ through the whole pipeline (WP7): inputs from outside the dataset.

Every score is sung four times, into <out_dir>/<score>_<run>.wav (+ .json, _f0.png):
  caption   the style of the score, F0 predicted
  prompt    the same with an audio prompt (a recording of an evaluation clip)
  f0        the F0 of the `caption` run with a vibrato added, given as "f0" in the score
            (<out_dir>/<score>_vibrato_score.json)
  f0wav     the audio of the `caption` run as the F0 source (tracked with RMVPE), with the prompt
Reported: notes within a semitone (F0 predicted), F0 frame error and median pitch deviation of the output
against the F0 that was given, and for `f0` the deviation against the curve without the vibrato: the
output follows the edit if the first is small and the second is not.
Also: the mel of a prompt recording as ldm.infer.pipeline.prompt_mel makes it against the mel of preprocessing.

Usage (from TCSinger2/, PYTHONPATH=.): python scripts/checks/check_examples.py
"""
import argparse
import json
import os

import numpy as np

from ldm.infer import f0_input, pipeline
from ldm.infer.timing import FPS
from ldm.stage2_samples import f0_metrics

PROMPTS = {  # a recording of another singer than the caption describes
    'twinkle_en': '/notebooks/data/GTSinger/English/EN-Tenor-1/Mixed_Voice_and_Falsetto/Cardigan/Control_Group/0000.wav',
    'twinkle_zh': '/notebooks/data/GTSinger/Chinese/ZH-Tenor-1/Mixed_Voice_and_Falsetto/失落沙洲/Control_Group/0000.wav',
    'twinkle_es': '/notebooks/data/GTSinger/Spanish/ES-Soprano-1/Breathy/Si Yo Fuera Un CHICO/Breathy_Group/0002.wav',
}

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='useful_ckpts/tcsinger2_gtsinger.ckpt')
parser.add_argument('--out_dir', default='samples/wp7/examples')
parser.add_argument('--vibrato_hz', type=float, default=5.5)
parser.add_argument('--vibrato_cents', type=float, default=60.0)
args = parser.parse_args()

singer = pipeline.Singer(args.ckpt)

# the prompt path against preprocessing
wav_path = PROMPTS['twinkle_en']
stored = np.load(wav_path.replace('/notebooks/data/GTSinger/', '/notebooks/data/GTSinger_processed/melnone48000/')[:-4] + '_mel.npy')
mel, _ = pipeline.prompt_mel(wav_path, singer.mel_net, max_frames=10 ** 6)
n = min(mel.shape[1], stored.shape[1])
print(f'prompt mel against the preprocessed mel of the same clip: {mel.shape[1]} / {stored.shape[1]} frames, '
      f'mean abs difference {np.abs(mel[:, :n] - stored[:, :n]).mean():.4f}', flush=True)

rows = []
for name, prompt in PROMPTS.items():
    score = f'examples/{name}.json'
    out = lambda run: os.path.join(args.out_dir, f'{name}_{run}.wav')
    wav, info, curves = singer.sing(score)
    pipeline.write_outputs(out('caption'), wav, info, curves)
    rows.append(dict(score=name, run='caption', caption=info['caption'], **info['measured'], warnings=info['warnings']))
    predicted = curves['f0_used']

    wav, info, curves = singer.sing(score, prompt=prompt)
    pipeline.write_outputs(out('prompt'), wav, info, curves)
    rows.append(dict(score=name, run='prompt', **info['measured'], warnings=info['warnings']))

    t = np.arange(len(predicted)) / FPS
    vibrato = np.where(predicted > 0, predicted * 2 ** (args.vibrato_cents / 1200 * np.sin(2 * np.pi * args.vibrato_hz * t)), 0)
    back = f0_input.load_f0(os.path.splitext(out('caption'))[0] + '.json')  # the curve as the output json holds it
    assert len(back) == len(predicted) and np.abs(back - predicted).max() < 0.01
    with open(score, encoding='utf-8') as f:
        with_f0 = dict(json.load(f), f0=[round(float(v), 2) for v in vibrato])
    score_f0 = os.path.join(args.out_dir, f'{name}_vibrato_score.json')
    with open(score_f0, 'w', encoding='utf-8') as f:
        json.dump(with_f0, f, ensure_ascii=False)
    wav, info, curves = singer.sing(score_f0)
    assert info['f0_source'] == 'score' and info['mode'] == 'given'
    pipeline.write_outputs(out('f0'), wav, info, curves)
    _, _, cents_plain = f0_metrics(predicted, curves['f0_out'])
    rows.append(dict(score=name, run='f0', **info['measured'], cents_out_vs_curve_without_vibrato=round(cents_plain, 1),
                     f0_check=info['f0_check'], warnings=info['warnings']))

    wav, info, curves = singer.sing(score, prompt=prompt, f0=out('caption'))
    pipeline.write_outputs(out('f0wav'), wav, info, curves)
    rows.append(dict(score=name, run='f0wav', **info['measured'], f0_check=info['f0_check'], warnings=info['warnings']))

for r in rows:
    figures = ', '.join(f'{k} {v}' for k, v in r.items() if isinstance(v, float))
    print(f"{r['score']:11s} {r['run']:8s} {figures}" + (f" | warnings: {r['warnings']}" if r['warnings'] else ''), flush=True)
with open(os.path.join(args.out_dir, 'examples.json'), 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
