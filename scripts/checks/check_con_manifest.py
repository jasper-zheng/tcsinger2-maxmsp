"""Checks the triplet manifest of the Custom Audio Encoder (WP5), written by preprocess/build_con_manifest.py:

  * every mel path exists (singing, paired speech, the four negatives and their speech clips)
  * the paired speech clip is the one of the same singer, technique folder, song and phrase number
  * every negative is another clip of the manifest, in the same split, with another pair_id, and its type holds
    (same singer for negatives 1 and 2, another singer for 3 and 4)

Prints the counts per split and per negative type. Usage: python scripts/checks/check_con_manifest.py [data/con/new.tsv]
"""
import ast
import os
import sys

import pandas as pd

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/con/new.tsv'
    df = pd.read_csv(path, sep='\t', low_memory=False)
    parts = df['item_name'].str.split('#', expand=True)
    speech = df['speech_mel_path'].str.split('/')
    expected = parts[0] + '/' + parts[1] + '/' + parts[2] + '/' + parts[3] + '/Paired_Speech_Group/' + parts[5] + '_mel.npy'
    assert (speech.str[-6:].str.join('/') == expected).all(), 'a speech clip is not the pair of its singing clip'
    assert df['mel_path'].is_unique and (df.groupby('pair_id')['speech_mel_path'].nunique() == 1).all()
    assert (df.groupby('speech_mel_path')['pair_id'].nunique() == 1).all()

    row = pd.Series(df.index, index=df['mel_path'])
    lyrics = lambda d: d['txt'].map(lambda t: [w for w in ast.literal_eval(t) if w not in ('<SP>', '<AP>')])  # rests aside
    for n in range(1, 5):
        neg = df.loc[row[df[f'singing_diff{n}_path']].values].reset_index(drop=True)  # KeyError if a negative is not in the manifest
        assert (neg['speech_mel_path'] == df[f'speech_diff{n}_path']).all()
        assert (neg['split'] == df['split']).all(), f'negative {n} crosses splits'
        assert (neg['pair_id'] != df['pair_id']).all(), f'negative {n} shares the speech clip'
        same_singer = neg['singer'] == df['singer']
        assert same_singer.all() if n <= 2 else not same_singer.any(), f'negative {n} has the wrong singer'
        t = df[f'diff{n}_type']
        song = lambda d: d['item_name'].str.split('#').str[3]
        assert (song(neg) != song(df))[t == 'same_singer_other_song'].all()
        assert (lyrics(neg[t == 'other_singer_same_lyrics']) == lyrics(df[t == 'other_singer_same_lyrics'])).all()
        assert (neg['language'] == df['language'])[t == 'other_singer_same_language'].all()

    paths = pd.unique(df[['mel_path', 'speech_mel_path'] + [f'{k}_diff{n}_path' for k in ('singing', 'speech') for n in range(1, 5)]].values.ravel())
    missing = [p for p in paths if not os.path.exists(p)]
    assert not missing, f'{len(missing)} missing files, e.g. {missing[:3]}'

    print(f'{path}: {len(df)} singing clips, {df["pair_id"].nunique()} speech clips, {len(paths)} mel files, all present')
    print(pd.DataFrame(dict(singing=df.groupby('split').size(), speech=df.groupby('split')['pair_id'].nunique(),
                            singers=df.groupby('split')['singer'].nunique())).to_string())
    print('singing clips per speech clip:', df.groupby('pair_id').size().value_counts().sort_index().to_dict())
    for n in (1, 3):
        both = pd.concat([df[['split', f'diff{n}_type']].set_axis(['split', 'type'], axis=1),
                          df[['split', f'diff{n + 1}_type']].set_axis(['split', 'type'], axis=1)])
        print(f'negatives {n} and {n + 1}:\n' + both.groupby('split')['type'].value_counts().unstack(fill_value=0).to_string())
    print('all checks passed')
