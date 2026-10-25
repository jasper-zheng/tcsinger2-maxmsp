"""80-bin log mels of the clips of intermediate manifests (make_manifest.py), 48 kHz, hop 256 (187.5 frames/s).

Per clip: mono, resampled to 48 kHz, loudness-normalised to -23 LUFS (scaled down if that would clip), then the
mel of preprocess/NAT_mel.MelNet (FFT and window 1024, fmin 20 Hz, fmax 24 kHz, log10 with floor 1e-5), cut at
--max_duration. Written to the row's `mel_path` as float32 [80, T].

A clip gets no mel, and preprocess/postprocess.py then leaves it out, when it is shorter than 2 s or when
reaching -23 LUFS would take more than 20 dB of gain either way. Clips whose mel exists are skipped, so the
script resumes after an interruption.

Replaces preprocess/mel_spec_48k.py; the mels are the same (compared bit for bit on a GPU). On a CPU they differ
in the last digits.

Usage (from the repository root, PYTHONPATH=.):
  python scripts-new/py/extract_mels.py --tsv work/data/new/data.tsv
"""
import argparse
import collections
import csv
import os

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from preprocess.NAT_mel import MelNet

SR, HOP = 48000, 256
MEL_ARGS = dict(audio_sample_rate=SR, audio_num_mel_bins=80, fft_size=1024, win_size=1024, hop_size=HOP, fmin=20, fmax=24000)
TARGET_LUFS = -23.0
MAX_GAIN_DB = 20.0
MIN_SECONDS = 2.0


class Clips(Dataset):
    """(mel path, normalised wav [1, N] or None, reason for skipping or None) per clip."""

    def __init__(self, jobs):
        self.jobs = jobs
        self.meter = pyln.Meter(SR)

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        audio_path, mel_path = self.jobs[i]
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] != 1:
            wav = wav.mean(0, keepdim=True)
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SR)
        if wav.shape[-1] / SR < MIN_SECONDS:
            return mel_path, None, 'shorter than 2 s'
        x = wav.squeeze(0).numpy()
        if not np.isfinite(x).all():
            return mel_path, None, 'non-finite samples'
        try:
            loudness = self.meter.integrated_loudness(x)
        except ValueError:
            return mel_path, None, 'loudness not measurable'
        if abs(TARGET_LUFS - loudness) > MAX_GAIN_DB:
            return mel_path, None, 'gain over 20 dB'
        x = pyln.normalize.loudness(x, loudness, TARGET_LUFS)
        peak = np.abs(x).max()
        if peak > 1.0:
            x = x / peak
        return mel_path, torch.tensor(x).unsqueeze(0), None


def read_jobs(tsvs):
    """(audio path, mel path) of the rows of the manifests, each clip once."""
    df = pd.concat([pd.read_csv(p, sep='\t', usecols=['audio_path', 'mel_path'], quoting=csv.QUOTE_NONE, escapechar='\\')
                    for p in tsvs], ignore_index=True)
    return list(dict.fromkeys(zip(df['audio_path'], df['mel_path'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tsv', nargs='+', required=True, help='intermediate manifests (make_manifest.py)')
    parser.add_argument('--max_duration', type=float, default=20.0, help='seconds; longer clips are cut')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    jobs = read_jobs(args.tsv)
    todo = [j for j in jobs if not os.path.exists(j[1])]
    print(f'{len(jobs)} clips, {len(jobs) - len(todo)} already done')
    max_frames = int(args.max_duration * SR / HOP)
    mel_net = MelNet(MEL_ARGS)
    mel_net.to(args.device)
    loader = DataLoader(Clips(todo), batch_size=None, num_workers=args.num_workers, shuffle=False)
    skipped = collections.Counter()
    with torch.no_grad():
        for mel_path, wav, reason in tqdm(loader, mininterval=30):
            if reason:
                skipped[reason] += 1
                continue
            mel = mel_net(wav.to(args.device)).cpu().numpy().squeeze(0)[:, :max_frames]
            os.makedirs(os.path.dirname(mel_path), exist_ok=True)
            np.save(mel_path + '.tmp.npy', mel)  # written whole or not at all
            os.replace(mel_path + '.tmp.npy', mel_path)
    print(f'wrote {len(todo) - sum(skipped.values())} mels; no mel for {sum(skipped.values())} clips: {dict(skipped)}')


if __name__ == '__main__':
    main()
