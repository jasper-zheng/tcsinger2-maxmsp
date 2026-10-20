"""Shared parts of the Chinese singing aligner (WP9): classes, frame labels, model, Viterbi.

The aligner is a frame-level phoneme classifier on the 80-bin mel (187.5 frames/s). A clip is
aligned by a Viterbi pass over its known phoneme sequence, in which a rest may be inserted before
the first syllable, between syllables and after the last one.
"""
import json
import os

import numba
import numpy as np
import torch
from torch import nn

FPS = 48000 / 256
MEL_FLOOR = -5.0
REST = ('<AP>', '<SP>')
INITIALS = {p + '_zh' for p in 'b p m f d t n l g k h j q x zh ch sh r z c s'.split()}


def load_classes(phone_set=None):
    """REST and the Chinese phonemes of the phone set (TCSINGER2_PHONE_SET if set, as in ldm/data/phone_set.py)."""
    phone_set = phone_set or os.environ.get('TCSINGER2_PHONE_SET', 'data/meta/phone_set.json')
    with open(phone_set, encoding='utf-8') as f:
        phones = json.load(f)
    return list(REST) + sorted(p for p in phones if p.endswith('_zh'))


def frame_labels(ph, ph_durs, n_frames, class_id):
    """Class per mel frame from the annotated durations; frames past the annotation are -100."""
    bounds = np.round(np.cumsum([0.0] + list(ph_durs)) * FPS).astype(int)
    labels = np.full(n_frames, -100, dtype=np.int64)
    for p, a, b in zip(ph, bounds[:-1], bounds[1:]):
        labels[a:min(b, n_frames)] = class_id[p]
    return labels


def syllables(phones):
    """Group a rest-free phoneme sequence into syllables: an initial with the final after it, or a phoneme alone."""
    out, i = [], 0
    while i < len(phones):
        if phones[i] in INITIALS and i + 1 < len(phones) and phones[i + 1] not in INITIALS:
            out.append([phones[i], phones[i + 1]])
            i += 2
        else:
            out.append([phones[i]])
            i += 1
    return out


class FrameClassifier(nn.Module):
    def __init__(self, n_classes, n_mels=80, hidden=256, layers=2, dropout=0.1):
        super().__init__()
        self.inp = nn.Conv1d(n_mels, hidden, 5, padding=2)
        self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden, 5, padding=2 * d, dilation=d) for d in (1, 2, 4))
        self.norms = nn.ModuleList(nn.GroupNorm(8, hidden) for _ in range(3))
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, mel, lengths=None):
        """mel [B, 80, T] in log10 units -> logits [B, T, C]. Padding must be MEL_FLOOR."""
        x = self.inp((mel - MEL_FLOOR) / 2)  # the floor becomes 0, like the convolutions' own padding
        for conv, norm in zip(self.convs, self.norms):
            x = x + self.drop(torch.relu(norm(conv(x))))
        x = x.transpose(1, 2)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            x, _ = nn.utils.rnn.pad_packed_sequence(self.rnn(packed)[0], batch_first=True, total_length=x.shape[1])
        else:
            x, _ = self.rnn(x)
        return self.out(x)


def load_model(ckpt, device):
    state = torch.load(ckpt, map_location='cpu')
    model = FrameClassifier(len(state['classes']), **state.get('config', {}))
    model.load_state_dict(state['model'])
    return model.to(device).eval(), state['classes']


@numba.njit(cache=True)
def _viterbi(emit, optional):
    T, S = emit.shape
    NEG = -1e30
    score = np.full(S, NEG)
    score[0] = emit[0, 0]
    if optional[0] and S > 1:
        score[1] = emit[0, 1]
    back = np.zeros((T, S), dtype=np.int8)
    new = np.empty(S)
    for t in range(1, T):
        for s in range(S):
            best, arg = score[s], 0
            if s >= 1 and score[s - 1] > best:
                best, arg = score[s - 1], 1
            if s >= 2 and optional[s - 1] and score[s - 2] > best:
                best, arg = score[s - 2], 2
            new[s] = best + emit[t, s]
            back[t, s] = arg
        score[:] = new
    s = S - 1
    if optional[S - 1] and S > 1 and score[S - 2] > score[S - 1]:
        s = S - 2
    total = score[s]
    path = np.empty(T, dtype=np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= back[t, s]
    return path, total


def align(logp, phones, class_id, min_rest_frames=6):
    """Align a rest-free phoneme sequence to frame log-probabilities [T, C].

    Returns (labels, frames, score): the phonemes with the rests that were found, the number of
    frames of each, and the mean log-probability per frame along the path; None if the clip is
    shorter than its phoneme sequence."""
    rest = np.logaddexp(logp[:, class_id['<AP>']], logp[:, class_id['<SP>']])
    cols, labels, optional = [rest], [None], [True]
    for syl in syllables(phones):
        for p in syl:
            cols.append(logp[:, class_id[p]])
            labels.append(p)
            optional.append(False)
        cols.append(rest)
        labels.append(None)
        optional.append(True)
    if len(phones) > logp.shape[0]:
        return None
    path, total = _viterbi(np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.float64), np.array(optional))
    if total < -1e29:
        return None
    segs = []  # [label, first frame, end frame)
    for s in np.unique(path):
        idx = np.flatnonzero(path == s)
        segs.append([labels[s], int(idx[0]), int(idx[-1]) + 1])
    # a rest too short to be one goes to the phoneme before it (after it, at the start of the clip)
    kept = []
    for k, (label, a, b) in enumerate(segs):
        if label is None and b - a < min_rest_frames and len(segs) > 1:
            if kept:
                kept[-1][2] = b
            else:
                segs[k + 1][1] = a
            continue
        kept.append([label, a, b])
    out_labels, frames = [], []
    for label, a, b in kept:
        if label is None:
            ap, sp = logp[a:b, class_id['<AP>']].sum(), logp[a:b, class_id['<SP>']].sum()
            label = '<AP>' if ap > sp else '<SP>'
        out_labels.append(label)
        frames.append(b - a)
    return out_labels, frames, float(total) / logp.shape[0]
