"""Builds the triplet manifest for the Custom Audio Encoder (WP5), in the format of ldm/data/joinaudiodataset_con.py.

One row per singing clip of data/sing/new.tsv that has a paired speech clip (same singer, technique folder, song
and phrase number, in the Paired_Speech_Group). Added columns:

  speech_mel_path          mel of the paired speech clip
  pair_id                  row number of that speech clip in the speech manifest; the control and technique versions
                           of a phrase share it, so they must not be used as negatives of each other
  singing_diff{1..4}_path  mels of four other singing clips, the explicit negatives (the dataset draws from them)
  speech_diff{1..4}_path   mels of their paired speech clips
  diff{1..4}_type          how the negative was chosen (below)

Negatives are drawn inside the clip's own split with a fixed seed:

  1, 2  same singer, different song          (same_singer_other_song; same_singer_other_phrase if the singer has one song)
  3, 4  different singer, same lyrics        (other_singer_same_lyrics), where another singer sings the same words,
        otherwise a different singer of the same language (other_singer_same_language), otherwise any other singer

Usage: python preprocess/build_con_manifest.py [--out_tsv data/con/new.tsv]
"""
import argparse
import ast
import os

import numpy as np
import pandas as pd

REST = ('<SP>', '<AP>')


def pair_key(df):
    p = df['item_name'].str.split('#', expand=True)  # language, singer, technique folder, song, group, number
    return p[1] + '#' + p[2] + '#' + p[3] + '#' + p[5], p[3]


def draw(rng, pool, ok, n):
    """Up to n distinct rows of pool for which ok(row) holds."""
    cand = [i for i in pool if ok(i)]
    return [int(i) for i in rng.choice(cand, size=min(n, len(cand)), replace=False)] if cand else []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sing_tsv', default='data/sing/new.tsv')
    parser.add_argument('--speech_tsv', default='data/speech/new.tsv')
    parser.add_argument('--out_tsv', default='data/con/new.tsv')
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    sing = pd.read_csv(args.sing_tsv, sep='\t', low_memory=False)
    speech = pd.read_csv(args.speech_tsv, sep='\t', low_memory=False)
    sing['key'], sing['song'] = pair_key(sing)
    speech['key'] = pair_key(speech)[0]
    assert speech['key'].is_unique
    speech['pair_id'] = np.arange(len(speech))
    df = sing.merge(speech[['key', 'pair_id', 'mel_path', 'split']].rename(columns=dict(mel_path='speech_mel_path', split='speech_split')),
                    on='key', how='inner').reset_index(drop=True)
    assert (df['split'] == df['speech_split']).all(), 'a speech clip is in another split than its singing clip'
    print(f'{len(df)} of {len(sing)} singing clips have a paired speech clip ({df["pair_id"].nunique()} speech clips)')
    df['lyrics'] = df['txt'].map(lambda t: ' '.join(w for w in ast.literal_eval(t) if w not in REST))

    singer, song, pid, lang = (df[c].values for c in ('singer', 'song', 'pair_id', 'language'))
    by = lambda *cols: df.groupby(list(cols) if len(cols) > 1 else cols[0]).indices  # group -> row numbers
    of_singer, of_lyrics, of_language, of_split = by('split', 'singer'), by('split', 'lyrics'), by('split', 'language'), by('split')
    rng = np.random.default_rng(args.seed)
    picks, types = np.zeros((len(df), 4), dtype=int), np.empty((len(df), 4), dtype=object)
    for i, (split, lyrics) in enumerate(zip(df['split'], df['lyrics'])):
        own = of_singer[(split, singer[i])]
        a, ta = draw(rng, own, lambda j: song[j] != song[i], 2), 'same_singer_other_song'
        if not a:
            a, ta = draw(rng, own, lambda j: pid[j] != pid[i], 2), 'same_singer_other_phrase'
        other = lambda j: singer[j] != singer[i]
        b = draw(rng, of_lyrics[(split, lyrics)], other, 2) if lyrics else []
        tb = ['other_singer_same_lyrics'] * len(b)
        for pool, name in ((of_language[(split, lang[i])], 'other_singer_same_language'), (of_split[split], 'other_singer')):
            if len(b) < 2:
                more = draw(rng, pool, lambda j: other(j) and j not in b, 2 - len(b))
                b, tb = b + more, tb + [name] * len(more)
        assert a and len(b) == 2, df['item_name'][i]
        picks[i], types[i] = (a * 2)[:2] + b, [ta] * 2 + tb

    for n in range(4):
        df[f'singing_diff{n + 1}_path'] = df['mel_path'].values[picks[:, n]]
        df[f'speech_diff{n + 1}_path'] = df['speech_mel_path'].values[picks[:, n]]
        df[f'diff{n + 1}_type'] = types[:, n]
    os.makedirs(os.path.dirname(args.out_tsv), exist_ok=True)
    df.drop(columns=['key', 'song', 'speech_split', 'lyrics']).to_csv(args.out_tsv + '.tmp', sep='\t', index=False)
    os.replace(args.out_tsv + '.tmp', args.out_tsv)
    print('wrote', args.out_tsv, '| clips per split:', df['split'].value_counts().to_dict())
    for n in (1, 3):
        print(f'negatives {n} and {n + 1}:', pd.concat([df[f'diff{n}_type'], df[f'diff{n + 1}_type']]).value_counts().to_dict())
