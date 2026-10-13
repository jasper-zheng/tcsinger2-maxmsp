"""Extract F0 with RMVPE for every clip of a metadata.json, at the mel frame rate.

Each clip gets <out_root>/<wav path relative to data_root, .wav -> _f0.npy>: float32 [T],
T = samples // hop_size, unvoiced frames = 0. Clips already done are skipped, also those another
run of this script finishes meanwhile, so two runs can share one list (e.g. one from each end).

Usage:
  python preprocess/extract_f0.py --metadata data/meta/metadata.json \
      --data_root /notebooks/data/GTSinger --out_root /notebooks/data/GTSinger_processed/f0
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.rmvpe import RMVPE


def f0_path_for(wav_fn, data_root, out_root):
    return os.path.join(out_root, os.path.relpath(wav_fn, data_root)[:-4] + '_f0.npy')


class WavDataset(Dataset):
    def __init__(self, jobs):
        self.jobs = jobs

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        wav_fn, out_fn = self.jobs[i]
        if os.path.exists(out_fn):  # done by a parallel run since the list was made
            return None
        wav, sr = sf.read(wav_fn, dtype='float32', always_2d=True)
        return wav.mean(axis=1), sr, out_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--out_root', required=True)
    parser.add_argument('--ckpt', default='useful_ckpts/rmvpe/model.pt')
    parser.add_argument('--sample_rate', type=int, default=48000)
    parser.add_argument('--hop_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--reverse', action='store_true', help='work through the list from its end')
    args = parser.parse_args()

    with open(args.metadata, encoding='utf-8') as f:
        items = json.load(f)
    jobs = [(it['wav_fn'], f0_path_for(it['wav_fn'], args.data_root, args.out_root)) for it in items]
    todo = [j for j in jobs if not os.path.exists(j[1])]
    if args.reverse:
        todo.reverse()
    print(f'{len(jobs)} clips, {len(jobs) - len(todo)} already done')

    rmvpe = RMVPE(args.ckpt)
    loader = DataLoader(WavDataset(todo), batch_size=None, num_workers=args.num_workers, shuffle=False)
    for job in tqdm(loader, mininterval=30):
        if job is None:
            continue
        wav, sr, out_fn = job
        wav = wav.numpy()
        assert sr == args.sample_rate, f'{out_fn}: sample rate {sr}'
        f0, _ = rmvpe.get_pitch(wav, sample_rate=sr, hop_size=args.hop_size, length=len(wav) // args.hop_size)
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        tmp = out_fn + '.tmp.npy'
        np.save(tmp, f0.astype(np.float32))
        os.replace(tmp, out_fn)
    print('done')
