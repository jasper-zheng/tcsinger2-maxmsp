"""Train the frame-level phoneme classifier of the aligner (WP9) on Chinese singing with annotated timings.

Training clips: split `train` and language Chinese in the given manifests (GTSinger, M4Singer,
Opencpop; clips with an `align_score`, i.e. timings made by the aligner, are left out).
Validation: the `valid` split of the same manifests, frame accuracy.

The state is saved to <out>.resume at every evaluation; a rerun of the same command continues from
it, and the file is removed when training has finished.

Usage (from TCSinger2/, PYTHONPATH=.):
  python preprocess/aligner/train.py --manifests data/sing/new.tsv data/sing_ext/new.tsv --out useful_ckpts/aligner_zh.pt
"""
import argparse
import ast
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from preprocess.aligner.common import MEL_FLOOR, FrameClassifier, frame_labels, load_classes


def read_manifests(paths, split, exclude=()):
    cols = ['item_name', 'mel_path', 'ph', 'ph_durs', 'split', 'language', 'singer', 'n_frames']
    df = pd.concat([pd.read_csv(p, sep='\t', usecols=lambda c: c in cols + ['dataset', 'align_score'], low_memory=False) for p in paths], ignore_index=True)
    if 'align_score' in df.columns:  # clips whose timings come from this aligner (OpenSinger) are no training data
        df = df[df['align_score'].isna()]
    corpus = df['dataset'].fillna('GTSinger') if 'dataset' in df.columns else pd.Series('GTSinger', index=df.index)
    return df[(df['language'] == 'Chinese') & (df['split'] == split) & ~corpus.isin(exclude)].reset_index(drop=True)


def augment(mel):
    """Other voices and recordings: a shift along the mel axis, a spectral tilt, sometimes a masked band."""
    n = mel.shape[0]
    bins = np.arange(n)
    src = np.clip(bins + np.random.uniform(-2.5, 2.5), 0, n - 1)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (src - lo)[:, None]
    mel = mel[lo] * (1 - w) + mel[hi] * w
    mel = mel + np.random.uniform(-0.4, 0.4) * (bins[:, None] / (n - 1) - 0.5)
    if np.random.rand() < 0.5:
        a = np.random.randint(n - 8)
        mel[a:a + np.random.randint(1, 9)] = MEL_FLOOR
    return mel


class Clips(Dataset):
    def __init__(self, df, class_id, crop, train, augment=False):
        self.rows = df.to_dict('records')
        self.class_id, self.crop, self.train, self.augment = class_id, crop, train, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        mel = np.load(row['mel_path'])
        labels = frame_labels(ast.literal_eval(row['ph']), ast.literal_eval(row['ph_durs']), mel.shape[1], self.class_id)
        if self.train:
            if mel.shape[1] > self.crop:
                a = np.random.randint(mel.shape[1] - self.crop + 1)
                mel, labels = mel[:, a:a + self.crop], labels[a:a + self.crop]
            if self.augment:
                mel = augment(mel)
            mel = np.maximum(mel + np.random.uniform(-0.3, 0.3), MEL_FLOOR)  # +-6 dB
        return torch.from_numpy(mel.astype(np.float32)), torch.from_numpy(labels)


def collate(batch):
    n = max(m.shape[1] for m, _ in batch)
    mel = torch.full((len(batch), 80, n), MEL_FLOOR)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    for i, (m, l) in enumerate(batch):
        mel[i, :, :m.shape[1]] = m
        labels[i, :len(l)] = l
    return mel, labels, torch.tensor([m.shape[1] for m, _ in batch])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    hit = n = 0
    for mel, labels, lengths in loader:
        pred = model(mel.to(device), lengths).argmax(-1).cpu()
        mask = labels >= 0
        hit += (pred[mask] == labels[mask]).sum().item()
        n += mask.sum().item()
    model.train()
    return hit / max(n, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifests', nargs='+', required=True)
    parser.add_argument('--out', default='useful_ckpts/aligner_zh.pt')
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--crop', type=int, default=1000, help='training crop in mel frames')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=2, help='GRU layers')
    parser.add_argument('--augment', action='store_true', help='spectral augmentation of the training clips')
    parser.add_argument('--exclude_datasets', nargs='*', default=[], help='corpora left out of training and validation, to test on them')
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--max_valid', type=int, default=600)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classes = load_classes()
    class_id = {c: i for i, c in enumerate(classes)}
    train_df, valid_df = (read_manifests(args.manifests, sp, args.exclude_datasets) for sp in ('train', 'valid'))
    valid_df = valid_df.sample(min(args.max_valid, len(valid_df)), random_state=0)
    print(f'{len(classes)} classes | train {len(train_df)} clips, {train_df["n_frames"].sum() / 187.5 / 3600:.1f} h, '
          f'{train_df["singer"].nunique()} singers | valid {len(valid_df)} clips')
    loader = DataLoader(Clips(train_df, class_id, args.crop, True, args.augment), batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, collate_fn=collate, persistent_workers=args.num_workers > 0,
                        worker_init_fn=lambda w: np.random.seed(args.seed * 1000 + w))
    valid = DataLoader(Clips(valid_df, class_id, args.crop, False), batch_size=16, num_workers=min(4, args.num_workers), collate_fn=collate)
    if len(loader) == 0:  # drop_last: the loop below would never take a step
        raise SystemExit(f'error: {len(train_df)} training clips, fewer than one batch of {args.batch_size}')

    config = dict(hidden=args.hidden, layers=args.layers)
    model = FrameClassifier(len(classes), **config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    step, t0, running = 0, time.time(), []
    resume = args.out + '.resume'
    if os.path.exists(resume):
        state = torch.load(resume, map_location='cpu')
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        sched.load_state_dict(state['sched'])
        step = state['step']
        print(f'resumed from {resume} at step {step}')
    while step < args.steps:
        for mel, labels, lengths in loader:
            logits = model(mel.to(device), lengths)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.to(device).reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            running.append(loss.item())
            if step % args.eval_every == 0 or step == args.steps:
                acc = evaluate(model, valid, device)
                print(f'step {step}: loss {np.mean(running):.3f}, valid frame accuracy {acc:.4f}, {time.time() - t0:.0f} s', flush=True)
                running = []
                torch.save(dict(model=model.state_dict(), classes=classes, config=config, step=step, valid_frame_accuracy=acc,
                                manifests=args.manifests, augment=args.augment, exclude_datasets=args.exclude_datasets), args.out)
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step), resume + '.tmp')
                os.replace(resume + '.tmp', resume)
            if step >= args.steps:
                break
    if os.path.exists(resume):
        os.remove(resume)
    print('saved', args.out)
