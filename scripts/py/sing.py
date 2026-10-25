"""Sing a score: notes and lyrics, a style, optionally an F0 curve -> wav.

The score is a JSON file (format in ldm/infer/score.py, examples listed in examples/README.md):
  {"language": "English", "tempo": 100,
   "words": [{"text": "twinkle", "notes": [["C4", 1], ["C4", 1]]}, {"rest": 1}, ...],
   "f0": [0, 0, 261.4, 262.0, ...]}                                  (optional)

Style, both optional: a caption (--singer "female alto" --emotion happy --method pop --technique breathy,
or the "style" of the score, or --caption with the whole string) and an audio prompt (--prompt, a recording
of the voice to imitate; its first 8 s are used).
F0, optional: "f0" in the score, a list of Hz per frame (187.5 frames/s, or the score's "f0_fps"), 0 where
unvoiced. The curve must start with the score and follow the same melody; the output follows it. Without
a curve, or with --no_f0, the model predicts the F0 from the notes; with --f0_model the F0 generator
(ldm/f0gen) makes the curve instead (from the notes, the language and the F0 of the prompt).

Writes <out>.wav, <out>_f0.png and <out>.json: phonemes, timings, caption, warnings, figures, and as "f0" the
curve the model sang (predicted or given). Copy it into a score to sing it again after editing it.

The defaults are the released models in useful_ckpts/. scripts-new/6_sing.sh runs this script with the models
the pipeline trained (--ckpt, --config) and its phone set and lexicon.

Usage (from the repository root, PYTHONPATH=.):
  python scripts-new/py/sing.py --score examples/twinkle_en.json --out samples/twinkle.wav --singer "female alto"
  python scripts-new/py/sing.py --score examples/f0_vibrato_it.json --out samples/with_f0.wav --prompt ref.wav
  python scripts-new/py/sing.py --score examples/twinkle_en.json --dry_run     # phonemes and timings only, on the CPU
  python scripts-new/py/sing.py --phonemes English                              # the phonemes "ph" may use
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())  # so that it also runs without PYTHONPATH=.

from ldm.data.captions import EMOTIONS, GENDERS, SINGING_METHODS, TIMBRE_TECHS, VOCAL_RANGES  # noqa: E402
from ldm.data.phone_set import language_set, ph_set  # noqa: E402
from ldm.infer import pipeline  # noqa: E402
from ldm.infer.g2p import LANG_SUFFIX, G2PError  # noqa: E402
from ldm.infer.score import ScoreError  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--score', help='JSON score')
    p.add_argument('--out', help='output wav; <out>.json and <out>_f0.png are written next to it')
    g = p.add_argument_group('style')
    g.add_argument('--singer', help=f'<gender> or <gender> <range>: {GENDERS} x {VOCAL_RANGES}')
    g.add_argument('--emotion', help=str(EMOTIONS))
    g.add_argument('--method', dest='singing_method', help=str(SINGING_METHODS))
    g.add_argument('--technique', help=f"none, or of {TIMBRE_TECHS} joined by ' and ', optionally two parts around ' then '")
    g.add_argument('--caption', help='the whole caption instead, e.g. "Singer: female alto, Emotion: happy"; "" for none')
    g.add_argument('--prompt', help='audio prompt: a recording of the voice (or a mel .npy)')
    g = p.add_argument_group('F0')
    g.add_argument('--no_f0', action='store_true', help='predict the F0 from the notes even if the score has an "f0"')
    g.add_argument('--f0_model', help='F0 generator (e.g. useful_ckpts/f0gen_gtsinger.ckpt) for the F0 when none is given')
    g.add_argument('--f0_steps', type=int, default=32, help='sampling steps of the F0 generator')
    g.add_argument('--f0_cfg', type=float, default=1.0, help='guidance scale of the F0 generator (prompt against none)')
    g = p.add_argument_group('synthesis')
    g.add_argument('--blur', dest='blur', action='store_true', default=pipeline.DEFAULT_BLUR,
                   help='label only the middle 24 frames of every phoneme piece and let the model place the boundaries')
    g.add_argument('--no_blur', dest='blur', action='store_false')
    g.add_argument('--fit', default=pipeline.DEFAULT_FIT, choices=['nucleus', 'scale'], help='how predicted lengths are fitted to a note')
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--cfg_scale', type=float, default=3.0)
    g.add_argument('--steps', type=int, default=25)
    g.add_argument('--ckpt', default='useful_ckpts/tcsinger2_gtsinger.ckpt')
    g.add_argument('--config', default='configs/tcsinger2.yaml')
    g.add_argument('--device', help='cuda, mps or cpu; default: cuda, else mps (Apple GPU), else cpu')
    p.add_argument('--dry_run', action='store_true', help='stop after the front-end: print phonemes and timings (no GPU needed)')
    p.add_argument('--phonemes', metavar='LANGUAGE', help='list the phonemes of a language and exit')
    return p.parse_args()


def print_front_end(info):
    print(f"{info['name']}: {info['language']}, {info['seconds']:.2f} s, {info['n_frames']} frames")
    print('   start    end  word           source   notes        phoneme pieces (ms)')
    for w in info['words']:
        notes = ' '.join(str(m) for m, _ in w['notes']) if w['source'] != 'rest' else '-'
        pieces = ' '.join(f"{p['ph'].rsplit('_', 1)[0] if '_' in p['ph'] else p['ph']}:{p['dur'] * 1000:.0f}"
                          for p in (info['pieces'][i] for i in w['pieces']))
        print(f"  {w['start']:6.2f} {w['end']:6.2f}  {w['text']:<14s} {w['source']:<8s} {notes:<12s} {pieces}")


def main():
    args = parse_args()
    if args.phonemes:
        if args.phonemes not in language_set:
            sys.exit(f'error: unknown language {args.phonemes!r}; expected one of {language_set}')
        suffix = LANG_SUFFIX[args.phonemes]
        print(' '.join(p[:-len(suffix)] for p in ph_set if p.endswith(suffix)))
        return
    if not args.score:
        sys.exit('error: --score is required')
    if not args.dry_run and not args.out:
        sys.exit('error: --out is required (or --dry_run)')
    f0 = False if args.no_f0 else None  # None: the "f0" of the score, if it has one
    style = dict(singer=args.singer, emotion=args.emotion, singing_method=args.singing_method, technique=args.technique)
    try:
        if args.dry_run:
            from ldm.infer.timing import load_duration_model
            prepared = pipeline.prepare(args.score, load_duration_model(args.config, 'cpu'), args.blur, args.fit)
            info = prepared.describe()
            info['caption'] = pipeline.make_caption(prepared.score.style, args.caption, **style)
            _, source, _, w = pipeline.given_f0(prepared, f0)
            info['warnings'] += w
            print_front_end(info)
            print('caption:', info['caption'] or '(none)', '| F0:', source)
        else:
            singer = pipeline.Singer(args.ckpt, args.config, args.device, f0_model=args.f0_model)
            wav, info, curves = singer.sing(args.score, style=style, caption=args.caption, prompt=args.prompt, f0=f0,
                                            blur=args.blur, fit=args.fit, seed=args.seed,
                                            cfg_scale=args.cfg_scale, steps=args.steps, f0_steps=args.f0_steps, f0_cfg=args.f0_cfg)
            pipeline.write_outputs(args.out, wav, info, curves)
            print_front_end(info)
            print('caption:', info['caption'] or '(none)', '| prompt:', info.get('prompt', '(none)'), '| F0:', info['f0_source'])
            print('measured:', ', '.join(f'{k} {v}' for k, v in info['measured'].items()))
            print('wrote', os.path.splitext(args.out)[0] + '.wav')
    except (ScoreError, G2PError, ValueError, FileNotFoundError) as e:
        sys.exit(f'error: {e}')
    for w in info['warnings']:
        print('warning:', w)


if __name__ == '__main__':
    main()
