"""Checks of the inference front-end (WP7) against the validation clips.

A validation clip is turned into a score (ldm.infer.score.score_from_row: its words, their notes and how
long they are sung) and sent through the front-end, as a score from outside the dataset would be.

  --part frontend   no synthesis, all clips of the split up to 20 s:
      phonemes   words whose phonemes equal the annotation, with the training lexicon and without it
      pieces     words that the syllable rule puts on their notes as the annotation has them
      lengths    error of the phoneme lengths (predicted, and fitted to the notes in both ways) against the annotation
      tracks     with the annotated phonemes and lengths the per-frame input must equal the dataset's
  --part synth      --n_clips clips, EMA weights, the same caption, prompt and noise for every input:
      dataset    the batch of the dataset (annotated phonemes and lengths), as scripts/eval_stage2.py
      ph         annotated phonemes, predicted lengths
      text       phonemes from the lyrics, predicted lengths (clips with a word without pronunciation are left out)
      ph_blur, text_blur   the same with boundary blurring
    each in both F0 modes; figures as in ldm/stage2_samples.py
  --part shift      --n_shift clips, dataset batch, the given F0 moved later by 0, 50, 100 and 200 ms:
      F0 frame error of the output against the F0 that was given, and what ldm.infer.f0_input.check_f0 reports

Results in <out_dir>/{frontend,synth_<input>,shift}.json, audio of the first --write clips in <out_dir>/audio.
A part whose file exists is skipped, so the script can be started again after an interruption.

Usage (from TCSinger2/, PYTHONPATH=.):
  python scripts/checks/check_frontend.py --part frontend
  nohup python scripts/checks/check_frontend.py --part synth,shift > logs/wp7_check_frontend.log 2>&1 &
"""
import argparse
import collections
import json
import os

import numpy as np
import soundfile as sf
import torch

from ldm.data.phone_set import ph2id
from ldm.data.tcsinger2_dataset import frame_tracks
from ldm.infer import f0_input, pipeline
from ldm.infer.g2p import G2PError, phonemize
from ldm.infer.score import ScoreError, parse_score, score_from_row
from ldm.infer.timing import FPS, build_pieces, fill_durations
from ldm.stage2_samples import SR, HOP, build_dataset, f0_metrics, notes_within, pick_clips, safe_name

parser = argparse.ArgumentParser()
parser.add_argument('--part', default='frontend')
parser.add_argument('--ckpt', default='useful_ckpts/tcsinger2_gtsinger.ckpt')
parser.add_argument('--split', default='valid')
parser.add_argument('--n_clips', type=int, default=100)
parser.add_argument('--n_shift', type=int, default=32)
parser.add_argument('--write', type=int, default=9)
parser.add_argument('--out_dir', default='samples/wp7/check')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
parts = args.part.split(',')
os.makedirs(args.out_dir, exist_ok=True)

dataset = build_dataset(args.split)
df = dataset.dataset
corpus = df['dataset'].fillna('GTSinger')


def dump(name, obj):
    path = os.path.join(args.out_dir, name + '.json')
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(path + '.tmp', path)
    print('wrote', path, flush=True)


def done(name):
    return os.path.exists(os.path.join(args.out_dir, name + '.json'))


def row_score(row, **kw):
    """Score of a manifest row, or None (longer than 20 s, or a sung word without a pitch)."""
    try:
        return parse_score(score_from_row(row, **kw))
    except ScoreError:
        return None


def frontend_part(predictor):
    stat = collections.defaultdict(collections.Counter)
    errs = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in range(len(df)):
        row = df.iloc[r]
        key = f"{corpus.iloc[r]} {row['language']}"
        ref = row_score(row, with_ph=True)
        if ref is None:
            stat[key]['clips_skipped'] += 1
            continue
        stat[key]['clips'] += 1
        ref_ph, _, _ = phonemize(ref)

        # phonemes from the lyrics
        text = row_score(row)
        for name, use_lex in (('lex', True), ('nolex', False)):
            phs, src, _ = phonemize(text, use_lexicon=use_lex, strict=False)
            for w, a, b, s in zip(text.words, ref_ph, phs, src):
                if w.is_rest:
                    continue
                stat[key][f'{name}_words'] += 1
                stat[key][f'{name}_missing'] += b is None
                stat[key][f'{name}_equal'] += b == a
                if b is not None:
                    stat[key][f'{name}_{s}_words'] += 1
                    stat[key][f'{name}_{s}_equal'] += b == a

        # pieces: the syllable rule against the annotated note-split
        pieces = build_pieces(ref, ref_ph)
        ph, pitches, ph2w = (eval(row[k]) for k in ('ph', 'ep_pitches', 'ph2words'))  # noqa: S307 (our own manifest)
        by_word = collections.defaultdict(list)
        for i, w in enumerate(ph2w):
            by_word[w].append((ph[i], pitches[i]))
        ann = []  # per word of the score; score_from_row makes one rest of every rest phoneme
        for _, v in sorted(by_word.items()):
            ann += [[x] for x in v] if v[0][0] in ('<AP>', '<SP>') else [v]
        rule = collections.defaultdict(list)
        for p in pieces:
            rule[p.word].append((p.ph, p.midi))
        assert len(ann) == len(ref.words), (row['item_name'], len(ann), len(ref.words))
        for wi, w in enumerate(ref.words):
            if w.is_rest:
                continue
            multi = 'multi' if len(w.notes) > 1 else 'single'
            stat[key][f'pieces_{multi}_words'] += 1
            stat[key][f'pieces_{multi}_equal'] += rule[wi] == ann[wi]

        # lengths per phoneme (pieces of one phoneme added up) against the annotation
        true = [d for w in parse_score(score_from_row(row, with_durs=True)).words if not w.is_rest for d in w.ph_durs]
        for fit in ('none', 'scale', 'nucleus'):
            pcs = build_pieces(ref, ref_ph)
            fill_durations(pcs, predictor, fit)
            got = collections.OrderedDict()
            for p in pcs:
                if p.type != 1:
                    got[(p.word, p.index)] = got.get((p.word, p.index), 0.0) + p.dur
            assert len(got) == len(true), (row['item_name'], len(got), len(true))
            e = np.abs(np.array(list(got.values())) - np.array(true))
            errs[key][fit] += list(e)
            nuc = np.array([any(p.nucleus for p in pcs if (p.word, p.index) == k) for k in got])
            errs[key][fit + '_consonant'] += list(e[~nuc])

        # the user's own lengths: the per-frame tracks must be the dataset's
        full = pipeline.prepare(parse_score(score_from_row(row, with_durs=True)), predictor)
        item = dataset[r]
        n = min(full.n_frames, item['ph'].shape[1])
        stat[key]['track_frames'] += n
        stat[key]['track_ph_equal'] += int((full.tracks['ph'][:n] == item['ph'][0, :n]).sum())
        stat[key]['track_midi_equal'] += int((full.tracks['midi'][:n] == item['ep_pitches'][0, :n]).sum())
        stat[key]['track_len_diff_max'] = max(stat[key]['track_len_diff_max'], abs(full.n_frames - item['ph'].shape[1]))
        # the same with every piece rounded to frames by itself, as the dataset does
        _, ph_own, _ = frame_tracks([ph2id[p.ph] for p in full.pieces], [p.dur for p in full.pieces], [p.midi for p in full.pieces],
                                    item['ph'].shape[1])
        stat[key]['track_frames_own'] += len(ph_own)
        stat[key]['track_ph_equal_own'] += int((ph_own == item['ph'][0]).sum())
    out = {}
    for key in sorted(stat):
        s, e = stat[key], errs[key]
        share = lambda a, b: round(s[a] / s[b], 4) if s[b] else None
        out[key] = dict(
            clips=s['clips'], clips_skipped=s['clips_skipped'], words=s['lex_words'],
            phonemes_equal=share('lex_equal', 'lex_words'), phonemes_missing=share('lex_missing', 'lex_words'),
            phonemes_equal_without_lexicon=share('nolex_equal', 'nolex_words'), missing_without_lexicon=share('nolex_missing', 'nolex_words'),
            by_source={src: dict(words=s[f'lex_{src}_words'], equal=share(f'lex_{src}_equal', f'lex_{src}_words'))
                       for src in ('lexicon', 'dict', 'pinyin') if s[f'lex_{src}_words']},
            pieces_equal_single_note=share('pieces_single_equal', 'pieces_single_words'),
            pieces_equal_multi_note=share('pieces_multi_equal', 'pieces_multi_words'), multi_note_words=s['pieces_multi_words'],
            length_error_ms={fit: dict(median=round(float(np.median(v)) * 1000, 1), mean=round(float(np.mean(v)) * 1000, 1))
                             for fit, v in e.items()},
            tracks=dict(ph_equal=share('track_ph_equal', 'track_frames'), midi_equal=share('track_midi_equal', 'track_frames'),
                        ph_equal_dataset_rounding=share('track_ph_equal_own', 'track_frames_own'),
                        len_diff_max_frames=s['track_len_diff_max']))
        o = out[key]
        print(f"{key:20s} {o['clips']:4d} clips | phonemes equal {o['phonemes_equal']:.1%} (missing {o['phonemes_missing']:.1%}), "
              f"without lexicon {o['phonemes_equal_without_lexicon']:.1%} (missing {o['missing_without_lexicon']:.1%}) | pieces equal: "
              f"one note {o['pieces_equal_single_note']:.1%}, several {o['pieces_equal_multi_note'] or 0:.1%} | length error median ms: "
              + ', '.join(f"{fit} {v['median']}" for fit, v in o['length_error_ms'].items())
              + f" | tracks ph {o['tracks']['ph_equal']:.2%} (dataset rounding {o['tracks']['ph_equal_dataset_rounding']:.2%}) "
                f"midi {o['tracks']['midi_equal']:.2%}", flush=True)
    dump('frontend', out)


def measure(singer, mel, f0_used, item, mode, n):
    """The figures of ldm.stage2_samples.Synthesizer.clip for a synthesised mel."""
    wav = singer.vocoder.vocode(mel[:, :n], f0=f0_used[:n])
    f0_out, _ = singer.rmvpe.get_pitch(wav.astype(np.float32), SR, HOP, len(wav) // HOP)
    f0_sung, midi = item['f0'][0, :n], item['ep_pitches'][0, :n]
    out = dict(mel_l1=float(np.mean(np.abs(mel[:, :n] - item['image'][:, :n]))))
    if mode == 'given':
        out['ffe_out_vs_f0'], out['voicing_err_out'], out['cents_out'] = f0_metrics(f0_sung, f0_out)
    else:
        out['ffe_pred_vs_sung'], out['voicing_err_pred'], _ = f0_metrics(f0_sung, f0_used[:n])
        out['ffe_out_vs_pred'], _, out['cents_out'] = f0_metrics(f0_used[:n], f0_out)
        out['notes_within_1'] = notes_within(f0_used[:n], midi)
    return wav, out


def synth_dataset(singer, item, idx, mode, f0=None):
    """The dataset's own batch, as Synthesizer.clip builds it (f0 replaces the clip's F0 as the given curve)."""
    if f0 is not None:
        item = dict(item, f0=f0[np.newaxis].astype(np.float32))
    batch = dataset.collater([item])
    n = item['image'].shape[1]
    torch.manual_seed(args.seed + idx)
    mel, f0_used = singer.model.synthesize(batch, f0_mode=mode, cfg_scale=3.0, steps=25)
    return mel[0, :, :n].float().cpu().numpy(), f0_used[0, :n].float().cpu().numpy(), n


def synth_part(singer, picks):
    inputs = [('dataset', None, False), ('ph', True, False), ('text', False, False), ('ph_blur', True, True), ('text_blur', False, True)]
    audio_dir = os.path.join(args.out_dir, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    for name, with_ph, blur in inputs:
        if done(f'synth_{name}'):
            continue
        rows, skipped = [], 0
        with singer.model.ema_scope():
            for k, idx in enumerate(picks):
                row, item = df.iloc[idx], dataset[idx]
                prepared = None
                if name != 'dataset':
                    score = row_score(row, with_ph=with_ph)
                    try:
                        prepared = singer.prepare(score, blur=blur) if score is not None else None
                    except G2PError:
                        prepared = None
                    if prepared is None:
                        skipped += 1
                        continue
                for mode in ('predict', 'given'):
                    if prepared is None:
                        mel, f0_used, n = synth_dataset(singer, item, idx, mode)
                    else:
                        f0, _ = f0_input.fit_length(item['f0'][0], prepared.n_frames)
                        batch = pipeline.make_batch(prepared, item['caption'], item['prompt'], f0 if mode == 'given' else None)
                        torch.manual_seed(args.seed + idx)
                        mel, f0_used = singer.model.synthesize(batch, f0_mode=mode, cfg_scale=3.0, steps=25)
                        n = min(prepared.n_frames, item['image'].shape[1])
                        mel, f0_used = mel[0, :, :n].float().cpu().numpy(), f0_used[0, :n].float().cpu().numpy()
                    wav, out = measure(singer, mel, f0_used, item, mode, n)
                    rows.append(dict(name=item['name'], idx=int(idx), mode=mode, corpus=corpus.iloc[idx], language=row['language'], **out))
                    if k < args.write:
                        sf.write(os.path.join(audio_dir, f"{safe_name(item['name'])}_{name}_{mode}.wav"), np.clip(wav, -1, 1), SR, subtype='PCM_16')
                if (k + 1) % 20 == 0:
                    print(f'{name}: {k + 1} / {len(picks)} clips', flush=True)
        mean = {}
        for mode in ('predict', 'given'):
            part = [r for r in rows if r['mode'] == mode]
            keys = [key for key in part[0] if isinstance(part[0][key], float)]
            mean[mode] = {key: round(float(np.nanmean([r[key] for r in part])), 4) for key in keys}
        dump(f'synth_{name}', dict(input=name, clips=len(rows) // 2, skipped=skipped, mean=mean, rows=rows))
        print(name, f'({len(rows) // 2} clips, {skipped} skipped)', json.dumps(mean), flush=True)

    # the same clips in every input, so that the means can be compared
    res = {name: json.load(open(os.path.join(args.out_dir, f'synth_{name}.json'), encoding='utf-8')) for name, _, _ in inputs}
    common = set.intersection(*[{r['idx'] for r in v['rows']} for v in res.values()])
    table = {}
    for name, v in res.items():
        table[name] = {}
        for mode, keys in (('given', ['ffe_out_vs_f0', 'mel_l1']), ('predict', ['notes_within_1', 'ffe_pred_vs_sung', 'mel_l1'])):
            part = [r for r in v['rows'] if r['mode'] == mode and r['idx'] in common]
            for key in keys:
                table[name][f'{mode}_{key}'] = round(float(np.nanmean([r[key] for r in part])), 4)
    dump('synth_summary', dict(clips_in_all_inputs=len(common), table=table))
    print(f'\n{len(common)} clips present in every input')
    for name, t in table.items():
        print(f'{name:10s}', ', '.join(f'{k} {v:.4f}' for k, v in t.items()), flush=True)


def shift_part(singer, picks):
    if done('shift'):
        return
    shifts = [0, 50, 100, 200]
    rows = []
    with singer.model.ema_scope():
        for idx in picks:
            row, item = df.iloc[idx], dataset[idx]
            prepared = singer.prepare(row_score(row, with_durs=True))
            midi, sung, nucleus = prepared.frame_masks()
            for ms in shifts:
                k = int(round(ms / 1000 * FPS))
                f0 = np.concatenate([np.zeros(k, dtype=np.float32), item['f0'][0]])[:item['f0'].shape[1]]
                mel, f0_used, n = synth_dataset(singer, item, idx, 'given', f0)
                wav = singer.vocoder.vocode(mel, f0=f0_used)
                f0_out, _ = singer.rmvpe.get_pitch(wav.astype(np.float32), SR, HOP, len(wav) // HOP)
                ffe, voicing, cents = f0_metrics(f0[:n], f0_out)
                m = min(n, prepared.n_frames)
                stats, warnings = f0_input.check_f0(f0[:m], midi[:m], sung[:m], nucleus[:m])
                rows.append(dict(name=item['name'], shift_ms=ms, ffe_out_vs_given=ffe, voicing_err=voicing, cents=cents,
                                 mel_l1=float(np.mean(np.abs(mel - item['image'][:, :n]))), warned=bool(warnings), **stats))
    mean = {}
    for ms in shifts:
        part = [r for r in rows if r['shift_ms'] == ms]
        mean[ms] = {key: round(float(np.mean([r[key] for r in part])), 4) for key in part[0] if key not in ('name', 'shift_ms')}
        print(f'shift {ms:3d} ms:', ', '.join(f'{k} {v:.4f}' for k, v in mean[ms].items()), flush=True)
    dump('shift', dict(clips=len(picks), mean=mean, rows=rows))


if 'frontend' in parts and not done('frontend'):
    from ldm.infer.timing import load_duration_model
    frontend_part(load_duration_model(device='cuda'))
if 'synth' in parts or 'shift' in parts:
    singer = pipeline.Singer(args.ckpt)
    picks = pick_clips(dataset, args.n_clips, args.seed, min_frames=375, max_frames=3750)
    if 'synth' in parts:
        synth_part(singer, picks)
    if 'shift' in parts:
        shift_part(singer, [i for i in picks if row_score(df.iloc[i]) is not None][:args.n_shift])
