"""Dataset of the F0 generator (WP10): the stage-2 manifests, F0 files only.

Every F0 of the manifests is held in memory, read from one cache file (CACHE_DIR, or TCSINGER2_F0GEN_CACHE if set; built on first use, about
5 min; rebuilt when a manifest names an F0 file it lacks). Batching is the stage-2 one
(JoinManifestSpecs.epoch_batches: clips of similar length, a frame budget, a share of the extra corpora).

An item: the note tracks and target of ldm.f0gen.features for the clip's first 20 s, and the F0 of a prompt
clip (training: a random one of the clip's prompt list, a random stretch of max_prompt_len frames, dropped
with probability prompt_drop; otherwise the first one, from its start).
"""
import ast
import os
import random

import numpy as np
import pandas as pd
import torch

from ldm.data.phone_set import MIDI_PAD_ID, NOTE_TYPE_PAD, PH_PAD_ID, ph2id
from ldm.data.tcsinger2_dataset import JoinManifestSpecs
from ldm.f0gen import features

CACHE_DIR = os.environ.get('TCSINGER2_F0GEN_CACHE', '/notebooks/data/f0gen_cache')


def load_f0_cache(paths, cache_dir=CACHE_DIR):
    """{f0 path: F0 array} for `paths`, from the cache (built or extended as needed)."""
    data_fn, index_fn = os.path.join(cache_dir, 'f0.npy'), os.path.join(cache_dir, 'index.tsv')
    paths = sorted(set(paths))
    if os.path.exists(index_fn):
        index = pd.read_csv(index_fn, sep='\t')
        if set(paths) <= set(index['path']):
            data = np.load(data_fn, mmap_mode='r')
            out = {}
            for p, a, b in zip(index['path'], index['start'], index['end']):
                out[p] = np.asarray(data[a:b])
            return out
        paths = sorted(set(paths) | set(index['path']))
    print(f'f0gen: building the F0 cache of {len(paths)} files in {cache_dir}', flush=True)
    arrays = [np.load(p).astype(np.float32) for p in paths]
    ends = np.cumsum([len(a) for a in arrays])
    os.makedirs(cache_dir, exist_ok=True)
    np.save(data_fn + '.tmp.npy', np.concatenate(arrays))
    os.replace(data_fn + '.tmp.npy', data_fn)
    pd.DataFrame(dict(path=paths, start=ends - [len(a) for a in arrays], end=ends)).to_csv(index_fn, sep='\t', index=False)
    return dict(zip(paths, arrays))


class F0Specs(JoinManifestSpecs):
    def __init__(self, split, prompt_drop=0.2, **kwargs):
        super().__init__(split, **kwargs)
        self.prompt_drop = prompt_drop
        main = self.df_main
        self.mel2f0 = dict(zip(main['mel_path'], main['f0_path']))
        self.f0 = load_f0_cache(main['f0_path'].tolist())
        lit = ast.literal_eval
        rows = self.dataset
        self.rows = [dict(ph=[ph2id[p] for p in lit(r.ph)], durs=lit(r.ph_durs), pitches=lit(r.ep_pitches), types=lit(r.ep_types),
                          notes=features.note_index(lit(r.ph), lit(r.ph2words), lit(r.ep_pitches), lit(r.ep_notedurs)),
                          prompts=[p for p in lit(r.prompt_mel) if p in self.mel2f0], f0_path=r.f0_path, name=r.item_name,
                          language=self.language_to_id[r.language])
                     for r in rows.itertuples()]

    def prompt_f0(self, row, train):
        if not row['prompts'] or (train and random.random() < self.prompt_drop):
            return np.zeros((2, 0), dtype=np.float32)
        f0 = self.f0[self.mel2f0[random.choice(row['prompts']) if train else row['prompts'][0]]]
        n = self.max_prompt_len or len(f0)
        if len(f0) > n:
            a = random.randint(0, len(f0) - n) if train else 0
            f0 = f0[a:a + n]
        return features.prompt_features(f0)

    def __getitem__(self, idx):
        row = self.rows[idx % len(self.rows)]
        f0 = self.f0[row['f0_path']]
        n = int(min(self.frames[idx % len(self.rows)], len(f0)))
        f0 = f0[:n]
        tr = features.note_tracks(row['ph'], row['durs'], row['pitches'], row['types'], row['notes'], n)
        ref = features.reference_pitch(tr['midi'])
        r, v, trust = features.encode(f0, ref)
        return dict(name=row['name'], language=row['language'], f0=f0, ref=ref, r=r, v=v, trust=trust,
                    ph=tr['ph'], midi=tr['midi'], ntype=tr['ntype'], pos=tr['pos'], onset=tr['onset'],
                    prompt=self.prompt_f0(row, self.split == 'train'))

    def collater(self, items):
        return collate(items)


def collate(items, multiple=4):
    """Pads to the longest clip (a multiple of `multiple` frames) and the longest prompt. mask / prompt_mask:
    1 on real frames."""
    T = max(len(it['r']) for it in items)
    T = -(-T // multiple) * multiple
    P = max(max(it['prompt'].shape[1] for it in items), 1)
    B = len(items)

    def pad(key, value, dtype=torch.float32):
        out = torch.full((B, T), value, dtype=dtype)
        for i, it in enumerate(items):
            out[i, :len(it[key])] = torch.as_tensor(it[key])
        return out

    pos = torch.zeros(B, 3, T)
    prompt = torch.zeros(B, 2, P)
    prompt_mask = torch.zeros(B, P)
    mask = torch.zeros(B, T)
    for i, it in enumerate(items):
        n, p = len(it['r']), it['prompt'].shape[1]
        pos[i, :, :n] = torch.as_tensor(it['pos'])
        prompt[i, :, :p] = torch.as_tensor(it['prompt'])
        prompt_mask[i, :p] = 1
        mask[i, :n] = 1
    return dict(name=[it['name'] for it in items], language=torch.tensor([it['language'] for it in items]),
                f0=pad('f0', 0.0), ref=pad('ref', 60.0), r=pad('r', 0.0), v=pad('v', -1.0), trust=pad('trust', 0.0),
                ph=pad('ph', PH_PAD_ID, torch.long), midi=pad('midi', MIDI_PAD_ID, torch.long),
                ntype=pad('ntype', NOTE_TYPE_PAD, torch.long), onset=pad('onset', 0.0), pos=pos,
                prompt=prompt, prompt_mask=prompt_mask, mask=mask)


class F0SpecsTrain(F0Specs):
    def __init__(self, specs_dataset_cfg, **kwargs):
        super().__init__('train', **specs_dataset_cfg, **kwargs)


class F0SpecsValidation(F0Specs):
    def __init__(self, specs_dataset_cfg, **kwargs):
        super().__init__('valid', **specs_dataset_cfg, **kwargs)
