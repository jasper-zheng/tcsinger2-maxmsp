"""Align phoneme sequences to singing with the frame classifier (WP9), or measure how well it does that.

Alignment (OpenSinger): for every clip of --clips (item_name, wav_fn, rest-free `ph`), writes the
phonemes with the rests that were found, their lengths in mel frames, and the path score.

Evaluation (--eval): the held-out Chinese clips of the manifests (valid, test_seen, test_unseen) are
stripped of their rests and of their note splits, as an OpenSinger label file would be, aligned,
and compared with their annotation. Per corpus and split: boundary errors, the share of frames that get
the annotated phoneme, how much rest time is found and invented, and the path scores of clips aligned to their own and to another clip's
phonemes (for the score threshold that drops clips whose text does not match the audio).

Usage (from TCSinger2/, PYTHONPATH=.):
  python preprocess/aligner/align.py --eval --manifests data/sing/new.tsv data/sing_ext/new.tsv --report data/meta_ext/aligner_report.json
  python preprocess/aligner/align.py --clips data/meta_ext/opensinger_audio.json --out data/meta_ext/opensinger_align.json
"""
import argparse
import ast
import collections
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from preprocess.aligner.common import FPS, MEL_FLOOR, REST, align, load_model


class Mels(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return i, torch.from_numpy(np.load(self.paths[i]).astype(np.float32)) if os.path.exists(self.paths[i]) else None


def collate(batch):
    batch = [(i, m) for i, m in batch if m is not None]
    if not batch:
        return [], None, None
    n = max(m.shape[1] for _, m in batch)
    mel = torch.full((len(batch), 80, n), MEL_FLOOR)
    for k, (_, m) in enumerate(batch):
        mel[k, :, :m.shape[1]] = m
    return [i for i, _ in batch], mel, torch.tensor([m.shape[1] for _, m in batch])


def log_probs(model, paths, lengths_hint, device, batch_size=32, num_workers=8):
    """Yields (index, log-probabilities [T, C]) for every mel that exists; batches hold clips of similar length."""
    order = np.argsort(lengths_hint)
    batches = [order[k:k + batch_size].tolist() for k in range(0, len(order), batch_size)]
    loader = DataLoader(Mels(paths), batch_sampler=batches, num_workers=num_workers, collate_fn=collate)
    with torch.no_grad():
        for idx, mel, lengths in loader:
            if not idx:
                continue
            logp = torch.log_softmax(model(mel.to(device), lengths), dim=-1).cpu().numpy()
            for k, i in enumerate(idx):
                yield i, logp[k, :lengths[k]]


def reference(ph, durs, words):
    """Annotation without note splits: [(phoneme, start s, end s)], rests included."""
    out, t = [], 0.0
    for k, (p, d) in enumerate(zip(ph, durs)):
        if out and k > 0 and p == ph[k - 1] and words[k] == words[k - 1] and p not in REST:
            out[-1][2] = t + d
        else:
            out.append([p, t, t + d])
        t += d
    return out


def run_eval(args, model, classes, device):
    class_id = {c: i for i, c in enumerate(classes)}
    dfs = []
    for p in args.manifests:
        df = pd.read_csv(p, sep='\t', low_memory=False)
        if 'dataset' not in df.columns:
            df['dataset'] = 'GTSinger'
        if 'align_score' in df.columns:  # timings made by the aligner are no reference
            df = df[df['align_score'].isna()]
        dfs.append(df[['item_name', 'mel_path', 'ph', 'ph_durs', 'ph2words', 'split', 'language', 'dataset', 'duration', 'n_frames']])
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df['language'] == 'Chinese') & df['split'].isin(args.eval_splits) & (df['n_frames'] < int(20 * FPS) - 2)]
    if args.eval_datasets:
        df = df[df['dataset'].isin(args.eval_datasets)]
    df = df.reset_index(drop=True)
    rows = df.to_dict('records')
    refs = [reference(ast.literal_eval(r['ph']), ast.literal_eval(r['ph_durs']), ast.literal_eval(r['ph2words'])) for r in rows]
    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, logp in log_probs(model, df['mel_path'].tolist(), df['n_frames'].values, device):
        ref, key = refs[i], f"{rows[i]['dataset']}/{rows[i]['split']}"
        sung = [seg for seg in ref if seg[0] not in REST]
        res = align(logp, [seg[0] for seg in sung], class_id)
        if res is None:
            stats[key]['failed'].append(1)
            continue
        labels, frames, score = res
        ends = np.cumsum(frames) / FPS
        pred = [(l, e - f / FPS, e) for l, f, e in zip(labels, frames, ends)]
        pred_sung = [seg for seg in pred if seg[0] not in REST]
        st = stats[key]
        st['score'].append(score)
        for k, (r, p) in enumerate(zip(sung, pred_sung)):
            j = ref.index(r)
            before_rest = j == 0 or ref[j - 1][0] in REST
            st['err_rest' if before_rest else 'err_inner'].append(abs(p[1] - r[1]))
            if j == len(ref) - 1 or ref[j + 1][0] in REST:
                st['err_rest'].append(abs(p[2] - r[2]))
        n = logp.shape[0]
        ref_rest, pred_rest = np.zeros(n, bool), np.zeros(n, bool)
        for mask, segs in ((ref_rest, ref), (pred_rest, pred)):
            for l, a, b in segs:
                if l in REST:
                    mask[int(round(a * FPS)):int(round(b * FPS))] = True
        ref_lab, pred_lab = np.full(n, '', dtype=object), np.full(n, '', dtype=object)
        for lab, segs in ((ref_lab, ref), (pred_lab, pred)):
            for l, a, b in segs:
                lab[int(round(a * FPS)):int(round(b * FPS))] = 'rest' if l in REST else l
        st['frames'].append(n)
        st['frames_same'].append((ref_lab == pred_lab).sum())
        st['rest_ref'].append(ref_rest.sum())
        st['rest_hit'].append((ref_rest & pred_rest).sum())
        st['sung_ref'].append((~ref_rest).sum())
        st['rest_false'].append((~ref_rest & pred_rest).sum())
        # the same audio against the phonemes of the next clip that has other phonemes
        other = next((refs[j] for j in range(i + 1, min(i + 20, len(refs))) if [s[0] for s in refs[j]] != [s[0] for s in ref]), None)
        if other is not None:
            wrong = align(logp, [seg[0] for seg in other if seg[0] not in REST], class_id)
            if wrong is not None:
                st['score_wrong'].append(wrong[2])

    report = {}
    for key, st in sorted(stats.items()):
        err = np.array(st['err_rest'] + st['err_inner'])
        q = lambda x, p: round(float(np.percentile(x, p)), 3) if len(x) else None
        report[key] = dict(
            clips=len(st['score']), failed=len(st['failed']), boundaries=len(err),
            median_error_ms=round(float(np.median(err)) * 1000, 1),
            within_20ms=round(float((err <= 0.02).mean()), 4), within_50ms=round(float((err <= 0.05).mean()), 4),
            within_50ms_next_to_rest=round(float((np.array(st['err_rest']) <= 0.05).mean()), 4),
            within_50ms_between_phonemes=round(float((np.array(st['err_inner']) <= 0.05).mean()), 4),
            within_100ms=round(float((err <= 0.1).mean()), 4),
            frames_with_the_annotated_phoneme=round(float(sum(st['frames_same']) / sum(st['frames'])), 4),
            rest_time_found=round(float(sum(st['rest_hit']) / max(sum(st['rest_ref']), 1)), 4),
            sung_time_called_rest=round(float(sum(st['rest_false']) / max(sum(st['sung_ref']), 1)), 4),
            score_own_text={p: q(st['score'], p) for p in (1, 5, 50)},
            score_other_text={p: q(st['score_wrong'], p) for p in (50, 95, 99)},
            # a clip is dropped when its score is under the threshold: [right text dropped, wrong text dropped]
            score_threshold={t: [round(float((np.array(st['score']) < t).mean()), 4), round(float((np.array(st['score_wrong']) < t).mean()), 4)]
                             for t in (-1.5, -2.0, -2.5, -3.0)})
        r = report[key]
        print(f"{key:24s} clips {r['clips']:5d} | median {r['median_error_ms']:5.1f} ms | within 20 ms {r['within_20ms']:.3f} | "
              f"within 50 ms {r['within_50ms']:.3f}, 100 ms {r['within_100ms']:.3f}, frames right {r['frames_with_the_annotated_phoneme']:.3f} (next to a rest {r['within_50ms_next_to_rest']:.3f}, between phonemes "
              f"{r['within_50ms_between_phonemes']:.3f}) | rest found {r['rest_time_found']:.3f}, sung called rest "
              f"{r['sung_time_called_rest']:.3f} | score own {r['score_own_text']} other {r['score_other_text']} | "
              f"dropped under a threshold [right text, wrong text] {r['score_threshold']}")
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(dict(ckpt=args.ckpt, groups=report), f, indent=1)


def run_align(args, model, classes, device):
    class_id = {c: i for i, c in enumerate(classes)}
    with open(args.clips, encoding='utf-8') as f:
        clips = json.load(f)
    paths = [os.path.join(args.mel_root, 'melnone48000', os.path.relpath(c['wav_fn'], args.data_root)[:-4] + '_mel.npy') for c in clips]
    out, failed = {}, 0
    for i, logp in log_probs(model, paths, [c['duration'] for c in clips], device):
        res = align(logp, clips[i]['ph'], class_id)
        if res is None:
            failed += 1
            continue
        out[clips[i]['item_name']] = dict(ph=res[0], frames=res[1], score=round(res[2], 4))
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    scores = np.array([v['score'] for v in out.values()])
    print(f'{len(clips)} clips: {len(out)} aligned, {failed} too short for their phonemes, {len(clips) - len(out) - failed} without mel | '
          f'score percentiles 1/5/50: {np.percentile(scores, [1, 5, 50]).round(3).tolist()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='useful_ckpts/aligner_zh.pt')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--manifests', nargs='+', default=['data/sing/new.tsv', 'data/sing_ext/new.tsv'])
    parser.add_argument('--report', default='data/meta_ext/aligner_report.json')
    parser.add_argument('--eval_splits', nargs='+', default=['valid', 'test_seen', 'test_unseen'])
    parser.add_argument('--eval_datasets', nargs='*', default=[], help='corpora to evaluate on (default: all)')
    parser.add_argument('--clips', default='data/meta_ext/opensinger_audio.json')
    parser.add_argument('--data_root', default='/notebooks/data/ext')
    parser.add_argument('--mel_root', default='/notebooks/data/ext_processed')
    parser.add_argument('--out', default='data/meta_ext/opensinger_align.json')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, classes = load_model(args.ckpt, device)
    (run_eval if args.eval else run_align)(args, model, classes, device)
