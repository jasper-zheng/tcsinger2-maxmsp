"""Turn the intermediate manifest (preprocess.py + mel_spec_48k.py + extract_f0.py) into the training manifest.

Per clip:
  * drops clips without mel / F0, shorter than --min_duration, with fewer than --min_ph phonemes
    or with negative durations; clips longer than --max_duration are kept (the mel is already
    truncated there) with `duration` capped
  * fills missing style labels with 'no' and writes the style caption (`technique`, `caption`;
    format in ldm/data/captions.py); rows with `technique_labelled` false (corpora without technique
    annotation, WP9) get technique 'no', which leaves the field out of the caption
  * fills note pitches that are missing on sung phonemes from the F0 track; in clips without a
    score (`note_source` f0, WP9) an onset takes the pitch of the phoneme after it
  * attaches `f0_path`, `n_frames` and `prompt_mel`: up to --max_prompts clips of the same
    singer / song / group, nearest by clip number

Usage:
  PYTHONPATH=. python preprocess/postprocess.py --in_tsv data/new/data.tsv --out_tsv data/sing/new.tsv \
      --data_root /notebooks/data/GTSinger --f0_root /notebooks/data/GTSinger_processed/f0
"""
import argparse
import ast
import collections
import csv
import json
import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd

from ldm.data.captions import caption_from_row, technique_text

REST = ('<AP>', '<SP>')
STYLE_COLS = ['emotion', 'range', 'pace', 'singing_method']
TECHS = ['mix', 'falsetto', 'vibrato', 'breathy', 'pharyngeal', 'glissando']
LIST_COLS = ['ph', 'ph_durs', 'ph2words', 'ep_pitches', 'ep_notedurs', 'ep_types', 'txt', 'word_durs', 'tech'] \
            + [t + '_tech' for t in TECHS]
MAX_MIDI = 98


def read_intermediate(path):
    # written by preprocess.py with QUOTE_NONE + backslash escapes
    return pd.read_csv(path, sep='\t', header=0, low_memory=False, quoting=csv.QUOTE_NONE, escapechar='\\')


def piece_midi(f0, start, end, fps):
    """MIDI note of the median voiced F0 inside [start, end) seconds, or None if mostly unvoiced."""
    seg = f0[int(round(start * fps)):int(round(end * fps))]
    voiced = seg[seg > 0]
    if len(voiced) < ARGS.min_voiced_frames or len(voiced) < ARGS.min_voiced_ratio * len(seg):
        return None
    return int(np.clip(round(69 + 12 * np.log2(np.median(voiced) / 440.0)), 1, MAX_MIDI))


def stats_key(row):
    """Key of the pitch statistics: the language, with the corpus in front for manifests that have a `dataset` column."""
    dataset = row.get('dataset')
    return f"{dataset}/{row['language']}" if isinstance(dataset, str) and dataset else row['language']


def fill_pitches(row, f0, fps, stats):
    """Give sung pieces without a note pitch one from the F0 track (or from their word)."""
    ph, durs, words = row['ph'], row['ph_durs'], row['ph2words']
    pitches, notedurs, types = list(row['ep_pitches']), list(row['ep_notedurs']), list(row['ep_types'])
    ends = np.cumsum(durs)
    starts = ends - np.asarray(durs)
    lang = stats_key(row)
    src = {}  # piece -> 'f0' or index of the piece it borrowed from
    # clips without a score (note_source f0): an onset takes the pitch of the phoneme after it, as a
    # score gives it, and its own F0 only if that phoneme has none
    no_score = row.get('note_source') == 'f0'
    onset_est = {}
    for i, p in enumerate(ph):
        if p in REST:
            continue
        stats[lang]['sung_s'] += durs[i]
        est = piece_midi(f0, starts[i], ends[i], fps)
        if pitches[i] > 0:  # annotated: only used to measure how well F0 agrees with the score
            if est is not None:
                d = abs(est - pitches[i])
                stats[lang]['cmp_n'] += 1
                stats[lang]['cmp_exact'] += d == 0
                stats[lang]['cmp_within1'] += d <= 1
                stats[lang]['cmp_octave'] += 11 <= d <= 13
            continue
        stats[lang]['missing_s'] += durs[i]
        if no_score and est is not None and i + 1 < len(ph) and words[i + 1] == words[i] and ph[i + 1] != p:
            onset_est[i] = est
            continue
        if est is not None:
            pitches[i], src[i] = est, 'f0'
            stats[lang]['filled_f0_s'] += durs[i]
    for i, p in enumerate(ph):
        if p in REST or pitches[i] > 0:
            continue
        cands = [j for j in range(len(ph)) if words[j] == words[i] and pitches[j] > 0]
        if cands:
            j = min(cands, key=lambda j: (abs(j - i), j < i))  # nearest, the following one on ties
            pitches[i], src[i] = pitches[j], j
            stats[lang]['filled_borrow_s'] += durs[i]
        elif i in onset_est:
            pitches[i], src[i] = onset_est[i], 'f0'
            stats[lang]['filled_f0_s'] += durs[i]
        else:
            stats[lang]['still_missing_s'] += durs[i]
    for w in {words[i] for i in src}:
        idx = [i for i in range(len(ph)) if words[i] == w]
        note_type = 3 if len({pitches[i] for i in idx if pitches[i] > 0}) > 1 else 2
        for i in idx:
            types[i] = note_type
        for i in idx:
            if i not in src:
                continue
            origin = src[i]
            while origin != 'f0' and origin in src:
                origin = src[origin]
            if origin != 'f0':  # borrowed from an annotated note
                notedurs[i] = notedurs[origin]
                continue
            run = [k for k in idx if k in src and pitches[k] == pitches[i]]  # same filled pitch, same word
            lo = hi = run.index(i)
            while lo > 0 and run[lo - 1] == run[lo] - 1:
                lo -= 1
            while hi < len(run) - 1 and run[hi + 1] == run[hi] + 1:
                hi += 1
            notedurs[i] = round(float(sum(durs[k] for k in run[lo:hi + 1])), 5)
    return pitches, notedurs, types


def process_row(row):
    """Returns (updated fields or None, drop reason, stats)."""
    stats = collections.defaultdict(collections.Counter)
    if not os.path.exists(row['mel_path']):
        return None, 'no_mel', stats
    f0_path = os.path.join(ARGS.f0_root, os.path.relpath(row['audio_path'], ARGS.data_root)[:-4] + '_f0.npy')
    if not os.path.exists(f0_path):
        return None, 'no_f0', stats
    if row['duration'] < ARGS.min_duration:
        return None, 'too_short', stats
    n_ph = len(row['ph'])
    if n_ph < ARGS.min_ph or (ARGS.max_ph and n_ph > ARGS.max_ph):
        return None, 'phoneme_count', stats
    if min(row['ph_durs']) < 0 or min(row['ep_notedurs']) < 0:
        return None, 'negative_duration', stats
    fps = ARGS.sample_rate / ARGS.hop_size
    n_frames = np.load(row['mel_path'], mmap_mode='r').shape[1]
    f0 = np.load(f0_path)
    expect = int(min(row['duration'], ARGS.max_duration) * fps)
    if abs(n_frames - expect) > 2 or len(f0) < n_frames - 2:
        return None, 'length_mismatch', stats
    pitches, notedurs, types = fill_pitches(row, f0, fps, stats)
    technique = 'no'
    if row['technique_labelled']:
        technique = technique_text(row['ph'], row['ph_durs'], {t: row[t + '_tech'] for t in TECHS}, ARGS.max_duration)
    out = dict(f0_path=f0_path, n_frames=int(n_frames), ep_pitches=pitches, ep_notedurs=notedurs, ep_types=types,
               technique=technique, caption=caption_from_row(dict(row, technique=technique)),
               duration=min(row['duration'], ARGS.max_duration))
    return out, None, stats


def add_prompts(df, max_prompts):
    """prompt_mel: mels of the nearest-numbered clips in the same singer/song/group directory."""
    number = df['mel_path'].str.extract(r'(\d+)_mel\.npy$')[0].astype(int)
    prompts = pd.Series([None] * len(df), index=df.index, dtype=object)
    for _, idx in df.groupby(df['mel_path'].map(os.path.dirname)).groups.items():
        nums, paths = number[idx].tolist(), df.loc[idx, 'mel_path'].tolist()
        for k, i in enumerate(idx):
            others = sorted((abs(nums[j] - nums[k]), nums[j], paths[j]) for j in range(len(idx)) if j != k)
            prompts[i] = [p for _, _, p in others[:max_prompts]]
    return prompts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_tsv', default='data/new/data.tsv')
    parser.add_argument('--out_tsv', default='data/sing/new.tsv')
    parser.add_argument('--data_root', required=True, help='root of the wav files (as given to preprocess.py)')
    parser.add_argument('--f0_root', required=True, help='out_root given to extract_f0.py')
    parser.add_argument('--report', default='data/meta/postprocess_report.json')
    parser.add_argument('--speech_in', default=None, help='optional speech manifest from preprocess.py')
    parser.add_argument('--speech_out', default='data/speech/new.tsv')
    parser.add_argument('--min_duration', type=float, default=2.0)
    parser.add_argument('--max_duration', type=float, default=20.0, help='mel length limit used by mel_spec_48k.py')
    parser.add_argument('--min_ph', type=int, default=2)
    parser.add_argument('--max_ph', type=int, default=0, help='0 = no limit')
    parser.add_argument('--max_prompts', type=int, default=10)
    parser.add_argument('--min_voiced_frames', type=int, default=3)
    parser.add_argument('--min_voiced_ratio', type=float, default=0.3)
    parser.add_argument('--sample_rate', type=int, default=48000)
    parser.add_argument('--hop_size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--datasets', nargs='*', default=[], help='keep only these corpora (`dataset` column), e.g. M4Singer Opencpop')
    return parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()
    df = read_intermediate(ARGS.in_tsv)
    if ARGS.datasets:
        df = df[df['dataset'].isin(ARGS.datasets)].reset_index(drop=True)
    n_in = len(df)
    for col in STYLE_COLS:
        df[col] = df[col].fillna('no').replace('', 'no')
    for col in LIST_COLS:
        df[col] = df[col].apply(ast.literal_eval)
    if 'technique_labelled' not in df.columns:  # GTSinger: every clip has technique flags
        df['technique_labelled'] = True
    df['technique_labelled'] = df['technique_labelled'].astype(str).str.lower() == 'true'

    with Pool(ARGS.workers) as pool:
        results = pool.map(process_row, df.to_dict('records'), chunksize=64)
    dropped = collections.Counter(reason for _, reason, _ in results if reason)
    stats = collections.defaultdict(collections.Counter)
    for _, _, st in results:
        for lang, c in st.items():
            stats[lang].update(c)
    keep = [out is not None for out, _, _ in results]
    df = df[keep].reset_index(drop=True)
    for col in pd.DataFrame([out for out, _, _ in results if out is not None]).items():
        df[col[0]] = col[1].values

    df['prompt_mel'] = add_prompts(df, ARGS.max_prompts)
    no_prompt = df['prompt_mel'].apply(len) == 0
    dropped['no_prompt'] = int(no_prompt.sum())
    df = df[~no_prompt].reset_index(drop=True)

    os.makedirs(os.path.dirname(ARGS.out_tsv) or '.', exist_ok=True)
    df.to_csv(ARGS.out_tsv, index=False, sep='\t')

    report = dict(rows_in=n_in, rows_out=len(df), dropped=dict(dropped),
                  hours=round(df['duration'].sum() / 3600, 2),
                  by_split={k: [int(len(g)), round(g['duration'].sum() / 3600, 2)] for k, g in df.groupby('split')},
                  by_language={k: [int(len(g)), round(g['duration'].sum() / 3600, 2)] for k, g in df.groupby('language')},
                  by_dataset={k: [int(len(g)), round(g['duration'].sum() / 3600, 2)] for k, g in df.groupby('dataset')}
                  if 'dataset' in df.columns else {},
                  truncated_clips=int((df['n_frames'] >= int(ARGS.max_duration * ARGS.sample_rate / ARGS.hop_size)).sum()),
                  pitch={lang: {k: round(float(v), 2) for k, v in c.items()} for lang, c in sorted(stats.items())},
                  captions=dict(distinct=int(df['caption'].nunique()),
                                technique={k: int(v) for k, v in df['technique'].value_counts().items()},
                                technique_values_by_split={k: int(g['technique'].nunique()) for k, g in df.groupby('split')}))
    print(f"{n_in} -> {len(df)} clips, {report['hours']} h; dropped {dict(dropped)}")
    print(f"{'language':18s} {'no pitch %':>10s} {'from F0 %':>10s} {'borrowed %':>10s} {'left %':>7s} | F0 vs score: exact  within1  octave")
    for lang, c in sorted(stats.items()):
        sung, n = max(c['sung_s'], 1e-9), max(c['cmp_n'], 1)
        print(f"{lang:18s} {100 * c['missing_s'] / sung:10.2f} {100 * c['filled_f0_s'] / sung:10.2f} "
              f"{100 * c['filled_borrow_s'] / sung:10.2f} {100 * c['still_missing_s'] / sung:7.2f} | "
              f"{c['cmp_exact'] / n:17.3f} {c['cmp_within1'] / n:8.3f} {c['cmp_octave'] / n:7.3f}")

    if ARGS.speech_in:
        sp = read_intermediate(ARGS.speech_in)
        has_mel = sp['mel_path'].map(os.path.exists)
        sp = sp[has_mel].reset_index(drop=True)
        sp['n_frames'] = [int(np.load(p, mmap_mode='r').shape[1]) for p in sp['mel_path']]
        for col in ('ph', 'ph_durs', 'ph2words', 'txt', 'word_durs'):
            sp[col] = sp[col].apply(ast.literal_eval)
        os.makedirs(os.path.dirname(ARGS.speech_out) or '.', exist_ok=True)
        sp.to_csv(ARGS.speech_out, index=False, sep='\t')
        report['speech'] = dict(rows_in=int(len(has_mel)), rows_out=int(len(sp)), no_mel=int((~has_mel).sum()),
                                hours=round(sp['duration'].sum() / 3600, 2))
        print(f"speech: {len(has_mel)} -> {len(sp)} clips")

    with open(ARGS.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
