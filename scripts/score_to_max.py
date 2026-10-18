"""A score to the conditions of the TCSinger2 Max models (EWP3 of EXPORTING_WORK_PACKAGES.md).

The two neural.gen~ objects (tcsinger2_f0.pte `score2f0`, tcsinger2_sing.pte `sing`) take the pieces of a
score: its phonemes placed on its notes, without timing; the models predict the lengths and the frames
themselves. Making the pieces needs the Python front-end (g2p, syllables onto notes), so this tool writes
them, as one JSON file per score that a Max `dict` imports (`import <name>.json`) and sends to both objects:

  ph, midi, note_dur, note_type, ph_dur, nucleus, note_id   [256] each, padded (PAD below)
  language                                                  index in language_set
  caption     the caption text, for neural.tokenizer (-> input_ids of `sing`)
  meta        frontend (python), name, language, n_pieces, n_frames, seconds, warnings
  words, pieces   what the front-end made of the score, with the timings Python predicts (display only)

neural.gen~ reads the keys named after its conditions and ignores the others. No F0 is written: in Max the
curve always comes from `score2f0` (and may be edited there); an "f0" in the score is ignored.
The same file is made inside Max, without Python, by export/max/js (tcs_max.js in a [v8] object).

Usage (from TCSinger2/):
  python scripts/score_to_max.py examples/*.json                      # -> examples/<file>.max.json, next to each score
  python scripts/score_to_max.py examples/*.json --out_dir somewhere  # -> somewhere/<file>.json
  python scripts/score_to_max.py examples/twinkle_en.json --out x.json --singer "female alto" --emotion happy
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())  # so that it also runs without PYTHONPATH=.

import numpy as np  # noqa: E402

from ldm.data.captions import EMOTIONS, GENDERS, SINGING_METHODS, TIMBRE_TECHS, VOCAL_RANGES  # noqa: E402
from ldm.data.phone_set import MIDI_PAD_ID, NOTE_TYPE_PAD, PH_PAD_ID, language_set, ph_set  # noqa: E402
from ldm.infer import pipeline  # noqa: E402
from ldm.infer.g2p import G2PError  # noqa: E402
from ldm.infer.score import ScoreError  # noqa: E402
from ldm.infer.timing import Piece, fill_durations, to_frames  # noqa: E402

FORMAT = 'tcsinger2-max-conditions/1'
N_PIECES = 256  # P of the export
# name -> (dtype, pad). The pads of ph, midi and note_type are those the duration predictor was trained with
# (ldm/data/joinaudiodataset_ae.py); note_id -1 marks a padded piece as belonging to no note.
CONDITIONS = {'ph': (int, PH_PAD_ID), 'midi': (int, MIDI_PAD_ID), 'note_dur': (float, 0.0), 'note_type': (int, NOTE_TYPE_PAD),
              'ph_dur': (float, 0.0), 'nucleus': (int, 0), 'note_id': (int, -1)}
PREDICT = -1.0  # ph_dur of a piece whose length the model predicts
SUFFIX = '.max.json'  # the default output, next to its score (git-ignored: *.max.json)


def note_ids(pieces):
    """Index of every piece's note, from 0: a word's note, and each rest its own. The grouping of
    timing.fill_durations (rests are always given) and of ldm/f0gen/features.tracks_from_pieces."""
    keys = {}
    return [keys.setdefault(('rest', i) if p.midi == 0 else (p.word, p.note), len(keys)) for i, p in enumerate(pieces)]


def conditions(pieces, language, n=N_PIECES):
    """The condition arrays of a score's pieces (timing.build_pieces; the given lengths are all they need),
    each padded to n pieces, plus `language`."""
    if len(pieces) > n:
        raise ValueError(f'the score has {len(pieces)} phoneme pieces, more than the {n} of the Max models; '
                         f'split it at a rest and convert the parts one by one')
    from ldm.data.phone_set import ph2id
    values = dict(ph=[ph2id[p.ph] for p in pieces], midi=[p.midi for p in pieces], note_dur=[p.notedur for p in pieces],
                  note_type=[p.type for p in pieces], ph_dur=[p.dur if p.given else PREDICT for p in pieces],
                  nucleus=[int(p.nucleus) for p in pieces], note_id=note_ids(pieces))
    out = {}
    for name, (kind, pad) in CONDITIONS.items():
        out[name] = [kind(v) for v in values[name]] + [pad] * (n - len(pieces))
    out['language'] = language_set.index(language)
    return out


def pieces_from_conditions(d, dtype=np.float64):
    """The pieces as the models see them: rebuilt from the conditions alone, with the float conditions at
    `dtype` (np.float32 is what neural.gen~ passes). Every note is its own `word` (word = note_id, note = 0)."""
    valid = [i for i, ph in enumerate(d['ph']) if ph != PH_PAD_ID]
    f = lambda v: float(dtype(v))
    return [Piece(ph_set[d['ph'][i]], d['note_id'][i], 0, d['midi'][i], f(d['note_dur'][i]), d['note_type'][i],
                  bool(d['nucleus'][i]), None if d['ph_dur'][i] == PREDICT else f(d['ph_dur'][i]), d['ph_dur'][i] != PREDICT)
            for i in valid]


def self_check(prepared, d, predictor):
    """Problems (empty = none) when the frames are computed again from the conditions alone, as the in-graph
    front-end will: the same predictor, fit and rounding must give the same frames and F0 generator tracks."""
    from ldm.f0gen.features import tracks_from_pieces
    want = prepared.tracks
    want_f0 = tracks_from_pieces(prepared.pieces, want['starts'])
    problems = []
    for dtype in (np.float64, np.float32):
        pieces = pieces_from_conditions(d, dtype)
        fill_durations(pieces, predictor, pipeline.DEFAULT_FIT)
        got = to_frames(pieces)
        got_f0 = tracks_from_pieces(pieces, got['starts'])
        for k in ('starts', 'mel2ph', 'ph', 'midi'):
            if len(got[k]) != len(want[k]) or (got[k] != want[k]).any():
                problems.append(f'{np.dtype(dtype).name}: {k} differs')
        for k in want_f0:
            if np.shape(got_f0[k]) != np.shape(want_f0[k]) or not np.array_equal(got_f0[k], want_f0[k]):
                problems.append(f'{np.dtype(dtype).name}: F0 generator track {k} differs')
    return problems


def graph_check(prepared, d, vae, frontend=None):
    """Problems (empty = none) of the in-graph front-end (export/modules/frontend.py, run here in PyTorch) on the
    conditions, against the Python front-end's frames and F0 generator tracks."""
    import torch
    from export.modules.frontend import FrontEnd
    from ldm.f0gen.features import reference_pitch, tracks_from_pieces
    fe = frontend or FrontEnd(vae).eval()
    conds = [torch.tensor([d[k]], dtype=torch.float32 if kind is float else torch.long) for k, (kind, _) in CONDITIONS.items()]
    with torch.no_grad():
        got = {k: v[0].numpy() for k, v in fe(*conds).items()}
    want = prepared.tracks
    n = int(want['starts'][-1])
    if int(got['n']) != n or not np.array_equal(got['ends'][:len(want['starts']) - 1], want['starts'][1:]):
        return [f"in-graph front-end: piece ends differ ({int(got['n'])} frames, Python {n})"]
    problems = [f'in-graph front-end: {k} differs' for k in ('mel2ph', 'ph', 'midi') if not np.array_equal(got[k][:n], want[k])]
    f0 = tracks_from_pieces(prepared.pieces, want['starts'])
    f0['ref'] = reference_pitch(f0['midi'])
    problems += [f'in-graph front-end: F0 generator track {k} differs' for k in ('ntype', 'onset') if not np.array_equal(got[k][:n], f0[k])]
    problems += [f'in-graph front-end: F0 generator track {k} differs by {np.abs(got[k][..., :n] - f0[k]).max():.1e}'
                 for k in ('pos', 'ref') if np.abs(got[k][..., :n] - f0[k]).max() > 1e-5]
    return problems


def convert(path, predictor, caption=None, style=None):
    """Score file -> (the dict to write, the Prepared score)."""
    prepared = pipeline.prepare(path, predictor, blur=False, fit=pipeline.DEFAULT_FIT)
    info = prepared.describe()
    warnings = list(info['warnings'])
    if prepared.score.f0 is not None:
        warnings.append('the score has an "f0"; it is ignored: in Max the F0 comes from score2f0')
    d = conditions(prepared.pieces, prepared.score.language)
    d['caption'] = pipeline.make_caption(prepared.score.style, caption, **(style or {}))
    d['meta'] = dict(format=FORMAT, frontend='python', name=info['name'], language=info['language'], n_pieces=len(prepared.pieces),
                     n_frames=info['n_frames'], seconds=info['seconds'], pieces_max=N_PIECES, frames_max=pipeline.MAX_FRAMES,
                     warnings=warnings)
    d['words'], d['pieces'] = info['words'], info['pieces']
    return d, prepared


def dumps(d):
    """JSON with one line per condition array, so the file stays readable."""
    lines = [f'  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}' for k, v in d.items() if k not in ('words', 'pieces')]
    for k in ('words', 'pieces'):
        lines.append(f'  {json.dumps(k)}: [\n' + ',\n'.join('    ' + json.dumps(x, ensure_ascii=False) for x in d[k]) + '\n  ]')
    return '{\n' + ',\n'.join(lines) + '\n}\n'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('scores', nargs='+', help='JSON score(s), in the format of ldm/infer/score.py')
    o = p.add_mutually_exclusive_group()
    o.add_argument('--out_dir', help=f'where <score file name>.json goes (default: next to the score, as <name>{SUFFIX})')
    o.add_argument('--out', help='output file, for a single score')
    g = p.add_argument_group('style (overrides the "style" of the score)')
    g.add_argument('--singer', help=f'<gender> or <gender> <range>: {GENDERS} x {VOCAL_RANGES}')
    g.add_argument('--emotion', help=str(EMOTIONS))
    g.add_argument('--method', dest='singing_method', help=str(SINGING_METHODS))
    g.add_argument('--technique', help=f"none, or of {TIMBRE_TECHS} joined by ' and ', optionally two parts around ' then '")
    g.add_argument('--caption', help='the whole caption instead, e.g. "Singer: female alto, Emotion: happy"; "" for none')
    p.add_argument('--config', default='configs/tcsinger2.yaml', help='config with the VAE that holds the duration predictor')
    p.add_argument('--no_check', action='store_true', help='skip the check that the conditions alone give the same frames')
    p.add_argument('--check', action='store_true', help='also run the in-graph front-end of the export (export/modules/frontend.py) '
                   'on the conditions and check that it gives the same frames')
    p.add_argument('--quiet', action='store_true', help='no phoneme table')
    return p.parse_args()


def main():
    args = parse_args()
    if args.out and len(args.scores) > 1:
        sys.exit('error: --out takes a single score; use --out_dir for several')
    from ldm.infer.timing import load_duration_model
    predictor = load_duration_model(args.config, 'cpu')
    style = dict(singer=args.singer, emotion=args.emotion, singing_method=args.singing_method, technique=args.technique)
    failed = 0
    for path in args.scores:
        if path.endswith(SUFFIX):  # an output of this script, e.g. matched by examples/*.json
            continue
        try:
            d, prepared = convert(path, predictor, args.caption, style)
        except (ScoreError, G2PError, ValueError, FileNotFoundError) as e:
            print(f'error: {path}: {e}', file=sys.stderr)
            failed += 1
            continue
        if not args.no_check:
            problems = self_check(prepared, d, predictor)
            if args.check:
                problems += graph_check(prepared, d, predictor.vae)
            if problems:
                d['meta']['warnings'].append('conditions check: ' + '; '.join(problems))
        stem = os.path.splitext(os.path.basename(path))[0]
        out = args.out or (os.path.join(args.out_dir, stem + '.json') if args.out_dir else os.path.join(os.path.dirname(path), stem + SUFFIX))
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            f.write(dumps(d))
        if not args.quiet:
            pipeline.print_front_end(prepared.describe())
        m = d['meta']
        print(f"caption: {d['caption'] or '(none)'} | {m['n_pieces']}/{N_PIECES} pieces, {m['n_frames']}/{m['frames_max']} frames"
              f" | wrote {out}")
        for w in m['warnings']:
            print('warning:', w)
    if failed:
        sys.exit(f'{failed} of {len(args.scores)} score(s) failed')


if __name__ == '__main__':
    main()
