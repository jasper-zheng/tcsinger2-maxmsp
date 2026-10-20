"""Intermediate manifest of a metadata file: one TSV row per clip, for extract_mels.py and preprocess/postprocess.py.

Every field of the metadata becomes a column (lists are written as Python literals); `wav_fn` becomes
`audio_path`, and `mel_path` says where extract_mels.py writes the clip's mel:
    <mel_root>/melnone48000/<wav path relative to data_root, .wav -> _mel.npy>
Paths are NFC-normalised. A clip whose wav is missing is an error (the metadata and the audio do not belong
together); --allow_missing leaves such clips out instead.

Replaces preprocess/preprocess.py; the rows are the same.

Usage (from the repository root, PYTHONPATH=.):
  python scripts-new/py/make_manifest.py --metadata work/data/meta/metadata.json --data_root work/audio/GTSinger \
      --mel_root work/features/GTSinger --out_tsv work/data/new/data.tsv
"""
import argparse
import csv
import json
import os
import sys
import unicodedata

import pandas as pd

MEL_DIR = 'melnone48000'  # mel mode 'none' at 48 kHz, as extract_mels.py writes it


def nfc(path):
    return unicodedata.normalize('NFC', path)


def mel_path_for(wav_fn, data_root, mel_root):
    return os.path.join(mel_root, MEL_DIR, os.path.relpath(wav_fn, data_root)[:-4] + '_mel.npy')


def write_tsv(df, path):
    """The format preprocess/postprocess.py reads: no quoting, backslash escapes."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    df.to_csv(path + '.tmp', sep='\t', header=True, index=False, encoding='utf-8', escapechar='\\', quoting=csv.QUOTE_NONE)
    os.replace(path + '.tmp', path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--metadata', required=True, help='metadata.json with wav_fn, ph, ph_durs, ...')
    parser.add_argument('--data_root', required=True, help='directory that every wav_fn lies under')
    parser.add_argument('--mel_root', required=True, help='--save_path of extract_mels.py')
    parser.add_argument('--out_tsv', required=True)
    parser.add_argument('--allow_missing', action='store_true', help='leave out clips whose wav is missing')
    args = parser.parse_args()

    with open(args.metadata, encoding='utf-8') as f:
        items = json.load(f)
    missing = [it['wav_fn'] for it in items if not os.path.exists(nfc(it['wav_fn']))]
    if missing and not args.allow_missing:
        sys.exit(f'error: {len(missing)} of {len(items)} wavs of {args.metadata} are missing, e.g. {missing[0]}')

    keys = [k for k in dict.fromkeys(k for it in items for k in it) if k != 'wav_fn']
    rows = []
    for it in items:
        wav = nfc(it['wav_fn'])
        if os.path.exists(wav):
            rows.append({**{k: it.get(k) for k in keys}, 'mel_path': nfc(mel_path_for(it['wav_fn'], args.data_root, args.mel_root)),
                         'audio_path': wav})
    write_tsv(pd.DataFrame(rows, columns=keys + ['mel_path', 'audio_path']), args.out_tsv)
    print(f'{args.out_tsv}: {len(rows)} clips' + (f', {len(missing)} left out without a wav' if missing else ''))


if __name__ == '__main__':
    main()
