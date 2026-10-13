"""Build TCSinger2 metadata from an extracted GTSinger tree (see gtsinger_unzip.py).

Output format follows the authors' processed metadata.json: every phoneme is split at note
boundaries and each piece carries its note pitch / note duration / note type
(1 rest, 2 single-note word, 3 multi-note word).

On top of that this script cleans the annotations (all steps counted in cleaning_log.json):
  * phoneme labels: bare AP/SP, slashed labels, notational variants, language suffix on every phoneme
  * empty labels / timeline gaps: absorbed into a neighbour (< --max_absorb s) or turned into <SP>
  * rest words always get pitch 0 and one piece per phoneme
  * rest notes inside sung words (audio-transcribed notes in Japanese/Korean): the phoneme keeps
    the pitch of its sung part, or borrows the nearest sung pitch of the word
  * note-split pieces shorter than --min_piece_dur are merged into a sibling piece of the same phoneme
  * clips containing a phoneme seen fewer than --min_ph_count times in the corpus are dropped
  * clips whose audio length differs from the annotation by more than --max_len_diff s are dropped
  * emotion typos are resolved from the other clips of the same song

--compat switches all cleaning off and reproduces the authors' metadata (used for validation).

Usage:
  python preprocess/gtsinger_build_metadata.py --data_root /notebooks/data/GTSinger --out_dir data/meta
"""
import argparse
import collections
import json
import os
import unicodedata
from multiprocessing import Pool

import soundfile as sf

LANG_SUFFIX = {'Chinese': '_zh', 'English': '_en', 'French': '_fr', 'German': '_de', 'Italian': '_it',
               'Japanese': '_ja', 'Korean': '_ko', 'Russian': '_ru', 'Spanish': '_es'}
LANGUAGES = list(LANG_SUFFIX)
# position i <-> code i+1 in the word-level `tech` string of the per-clip json
TECHS = ['mix', 'falsetto', 'breathy', 'pharyngeal', 'vibrato', 'glissando']
REST = ('<AP>', '<SP>')
BARE_REST = {'AP': '<AP>', 'SP': '<SP>'}
GLOBALS = ['emotion', 'singing_method', 'pace', 'range']
VOICE = {'Soprano': 'female', 'Alto': 'female', 'Tenor': 'male', 'Bass': 'male'}

# languages annotated in IPA, where ASCII "g" is a typo for IPA "ɡ" (pinyin and ARPAbet use plain letters)
IPA_LANGS = {'French', 'German', 'Italian', 'Japanese', 'Korean', 'Russian', 'Spanish'}
# notational variants of a phoneme that already exists in that language's inventory
PH_FIXES = {
    'Japanese': {'t͡ɕ': 'tɕ', 't͡s': 'ts', 'ʨ': 'tɕ'},
    'Korean': {'t͡ɕ': 'tɕ', 't͡ɕʰ': 'tɕʰ', 't͡ɕ͈': 'tɕ͈', 't͈ɕ': 'tɕ͈', 'd͡ʑ': 'dʑ'},
}


def clean_label(p, lang):
    p = unicodedata.normalize('NFC', p.strip())
    p = BARE_REST.get(p, p)
    if len(p) > 2 and p[0] == p[-1] == '/':
        p = p[1:-1]
    if p == 'g' and lang in IPA_LANGS:
        p = 'ɡ'
    return PH_FIXES.get(lang, {}).get(p, p)


def word_flags(w, i, from_word_string):
    """Technique flags of phoneme i of word w.

    The per-phoneme arrays are the precise annotation (a word-level technique is often marked on
    only some of its phonemes). The authors' metadata uses the coarser word-level `tech` string
    for the seven older languages, which is what from_word_string reproduces."""
    codes = {int(c) for c in str(w.get('tech', '')).split(',') if c.strip().isdigit()}
    if from_word_string:
        return [int(k + 1 in codes) for k in range(len(TECHS))]
    flags = []
    for k, t in enumerate(TECHS):
        v = w.get(t, w.get('vibra') if t == 'vibrato' else None)  # 'vibra': typo in some Chinese clips
        flags.append(int(v[i]) if v is not None and i < len(v) else int(k + 1 in codes))
    return flags


def phoneme_entries(words, lang, clean, log):
    """One entry per annotated phoneme. With clean=True the entries tile the timeline without
    gaps, empty labels or zero-length phonemes."""
    raw = []
    for wi, w in enumerate(words):
        word = unicodedata.normalize('NFC', w['word'])
        if clean:
            word = BARE_REST.get(word.strip(), word.strip())
        for i, (p, s, e) in enumerate(zip(w['ph'], w['ph_start'], w['ph_end'])):
            label = clean_label(p, lang) if clean else p
            if clean and label != unicodedata.normalize('NFC', p):
                log['label_fixed'] += 1
            en = dict(ph=label, start=float(s), end=float(e), word=word, wi=wi,
                      flags=word_flags(w, i, from_word_string=not clean))
            if clean and label in REST and word not in REST:  # breath/silence annotated inside a sung word
                log['rest_inside_word'] += 1
                en.update(word=label, wi=('rest', wi, i), flags=[0] * len(TECHS))
            raw.append(en)
    if not clean:
        return raw

    out, state = [], dict(t=0.0, carry=None, n_sp=0)

    def push(en):
        dur = en['end'] - en['start']
        if en['ph'] == '':
            if dur <= 1e-4:
                log['empty_zero_len_dropped'] += 1
            elif dur < ARGS.max_absorb:
                log['empty_absorbed'] += 1
                if out:
                    out[-1]['end'] = en['end']
                elif state['carry'] is None:
                    state['carry'] = en['start']
            else:
                log['empty_to_sp'] += 1
                state['n_sp'] += 1
                out.append(dict(en, ph='<SP>', word='<SP>', wi=('sp', state['n_sp']), flags=[0] * len(TECHS)))
            return
        if dur <= 1e-4:
            log['zero_len_phoneme_dropped'] += 1
            return
        if state['carry'] is not None:
            en['start'], state['carry'] = state['carry'], None
        out.append(en)

    for en in raw:
        t = state['t']
        if en['start'] - t > 1e-3:
            log['timeline_gap'] += 1
            push(dict(ph='', start=t, end=en['start'], word='', wi=None, flags=[0] * len(TECHS)))
        elif en['start'] < t - 1e-3:
            log['timeline_overlap'] += 1
            en['start'] = min(t, en['end'])
        push(en)
        state['t'] = max(t, en['end'])
    return out


def split_at_notes(entries, notes, log):
    """Split every phoneme entry at note boundaries; each piece points to its note (index or None)."""
    bounds = sorted({b for ns, ne, *_ in notes for b in (ns, ne)})
    live = [k for k, (ns, ne, *_) in enumerate(notes) if ne - ns > 1e-6]
    pieces = []
    for ei, en in enumerate(entries):
        s, e = en['start'], en['end']
        cuts = [s] + [b for b in bounds if s + 1e-6 < b < e - 1e-6] + [e]
        for a, b in zip(cuts[:-1], cuts[1:]):
            mid = (a + b) / 2
            k = next((k for k in live if notes[k][0] - 1e-6 <= mid <= notes[k][1] + 1e-6), None)
            if k is None and live:
                log['piece_outside_notes'] += 1
                k = min(live, key=lambda k: min(abs(mid - notes[k][0]), abs(mid - notes[k][1])))
            pieces.append(dict(ei=ei, dur=b - a, note=k))
    return pieces


def merge_into(pieces, src, dst):
    pieces[dst]['dur'] += pieces[src]['dur']
    del pieces[src]


def clean_pieces(entries, pieces, notes, log):
    """Apply the rest / unpitched / sliver rules. Returns pieces with explicit pitch and note_dur."""
    pitch = lambda pc: notes[pc['note']][2] if pc['note'] is not None else 0
    by_entry = collections.defaultdict(list)
    for pc in pieces:
        by_entry[pc['ei']].append(pc)

    out = []
    for ei, en in enumerate(entries):
        grp = by_entry[ei]
        if en['word'] in REST:
            if len(grp) > 1 or pitch(grp[0]) != 0:
                log['rest_piece_fixed'] += 1
            ndur = sum(notes[k][3] for k in dict.fromkeys(pc['note'] for pc in grp) if k is not None)
            out.append(dict(ei=ei, dur=sum(pc['dur'] for pc in grp), note=None, pitch=0,
                            note_dur=ndur or sum(pc['dur'] for pc in grp)))
            continue
        # rest notes inside a sung phoneme: fold them into the sung part
        sung = [i for i, pc in enumerate(grp) if pitch(pc) > 0]
        if sung and len(sung) < len(grp):
            log['unpitched_piece_merged'] += len(grp) - len(sung)
            i = 0
            while i < len(grp):
                if pitch(grp[i]) > 0:
                    i += 1
                    continue
                dst = i - 1 if i > 0 else i + 1
                merge_into(grp, i, dst)
                i = max(i - 1, 0) if dst < i else i
        elif not sung:
            grp = [dict(ei=ei, dur=sum(pc['dur'] for pc in grp), note=None)]
        # note slivers
        while len(grp) > 1 and min(pc['dur'] for pc in grp) < ARGS.min_piece_dur:
            i = min(range(len(grp)), key=lambda i: grp[i]['dur'])
            nb = [j for j in (i - 1, i + 1) if 0 <= j < len(grp)]
            merge_into(grp, i, max(nb, key=lambda j: grp[j]['dur']))
            log['sliver_merged'] += 1
        out.extend(grp)

    # unpitched phonemes of sung words borrow the nearest sung note of the same word
    for i, pc in enumerate(out):
        if 'pitch' in pc:
            continue
        if pc['note'] is not None and notes[pc['note']][2] > 0:
            continue
        wi = entries[pc['ei']]['wi']
        cands = [j for j, q in enumerate(out) if entries[q['ei']]['wi'] == wi and 'pitch' not in q
                 and q['note'] is not None and notes[q['note']][2] > 0]
        if cands:
            j = min(cands, key=lambda j: (abs(j - i), j < i))  # nearest, the following one on ties
            pc['note'] = out[j]['note']
            log['unpitched_phoneme_borrowed'] += 1
        else:
            pc['note'] = None
            log['sung_phoneme_without_pitch'] += 1
    for pc in out:
        if 'pitch' not in pc:
            pc['pitch'] = notes[pc['note']][2] if pc['note'] is not None else 0
            pc['note_dur'] = notes[pc['note']][3] if pc['note'] is not None else pc['dur']
    return out


def build_singing(words, lang, clean, log):
    entries = phoneme_entries(words, lang, clean, log)
    notes = [(float(ns), float(ne), int(n), float(nd), wi) for wi, w in enumerate(words)
             for n, nd, ns, ne in zip(w['note'], w['note_dur'], w['note_start'], w['note_end'])]
    pieces = split_at_notes(entries, notes, log)
    if clean:
        pieces = clean_pieces(entries, pieces, notes, log)
        word_notes = collections.defaultdict(set)
        for pc in pieces:
            word_notes[entries[pc['ei']]['wi']].add(pc['note'])
        for pc in pieces:
            en = entries[pc['ei']]
            pc['type'] = 1 if en['word'] in REST else (3 if len(word_notes[en['wi']]) > 1 else 2)
    else:
        for pc in pieces:
            en = entries[pc['ei']]
            pc['pitch'] = notes[pc['note']][2]
            pc['note_dur'] = notes[pc['note']][3]
            pc['type'] = 1 if en['word'] in REST or pc['pitch'] == 0 else (3 if len(words[en['wi']]['note']) > 1 else 2)

    item = collections.defaultdict(list)
    last_wi, wid = object(), -1
    for pc in pieces:
        en = entries[pc['ei']]
        if en['wi'] != last_wi:
            last_wi, wid = en['wi'], wid + 1
            item['txt'].append(en['word'])
            item['word_durs'].append(0.0)
        item['word_durs'][-1] = round(item['word_durs'][-1] + pc['dur'], 5)
        item['ph'].append(en['ph'] if en['ph'] in REST or (not clean and en['ph'] == '') else en['ph'] + LANG_SUFFIX[lang])
        item['ph_durs'].append(round(pc['dur'], 5))
        item['ph2words'].append(wid)
        item['ep_pitches'].append(pc['pitch'])
        item['ep_notedurs'].append(round(pc['note_dur'], 5))
        item['ep_types'].append(pc['type'])
        for t, f in zip(TECHS, en['flags']):
            item[t + '_tech'].append(f)
        item['tech'].append(','.join(str(k + 1) for k, f in enumerate(en['flags']) if f) or '0')
    for g in GLOBALS:
        item[g] = next((str(w[g]) for w in words if w.get(g) not in (None, '')), 'no')
    return dict(item)


def build_speech(words, lang, log):
    entries = phoneme_entries(words, lang, True, log)
    item = collections.defaultdict(list)
    last_wi = object()
    for en in entries:
        if en['wi'] != last_wi:
            last_wi = en['wi']
            item['txt'].append(en['word'])
            item['word_durs'].append(0.0)
        dur = en['end'] - en['start']
        item['word_durs'][-1] = round(item['word_durs'][-1] + dur, 5)
        item['ph'].append(en['ph'] if en['ph'] in REST else en['ph'] + LANG_SUFFIX[lang])
        item['ph_durs'].append(round(dur, 5))
        item['ph2words'].append(len(item['txt']) - 1)
    return dict(item)


def process_clip(json_path):
    log = collections.Counter()
    rel = os.path.relpath(json_path, ARGS.data_root)[:-5]
    parts = unicodedata.normalize('NFC', rel).split(os.sep)
    lang, singer, _, _, group, idx = parts
    is_speech = 'Speech' in group
    wav_fn = json_path[:-5] + '.wav'
    if not os.path.exists(wav_fn):
        return None, is_speech, {'missing_wav': 1}, lang
    with open(json_path, encoding='utf-8') as f:
        words = json.load(f)
    try:
        item = build_speech(words, lang, log) if is_speech else build_singing(words, lang, not ARGS.compat, log)
    except Exception as e:
        return None, is_speech, {f'build_error:{type(e).__name__}': 1}, lang
    if not item.get('ph'):
        return None, is_speech, {'empty_clip': 1}, lang
    info = sf.info(wav_fn)
    if not ARGS.compat and abs(info.frames / info.samplerate - sum(item['ph_durs'])) > ARGS.max_len_diff:
        return None, is_speech, {'dropped_length_mismatch': 1}, lang  # audio and annotation do not belong together
    item.update(item_name='#'.join(parts), wav_fn=wav_fn, language=lang, singer=singer,
                gender=VOICE[singer.split('-')[1]], vocal_range=singer.split('-')[1].lower(),
                duration=round(info.frames / info.samplerate, 5))
    if not is_speech:
        speech_fn = os.path.join(os.path.dirname(os.path.dirname(json_path)), 'Paired_Speech_Group', idx + '.wav')
        item['speech_fn'] = speech_fn if os.path.exists(speech_fn) else ''
    return item, is_speech, dict(log), lang


def find_clips(data_root):
    for d, _, files in os.walk(data_root):
        for f in files:
            if f.endswith('.json'):
                yield os.path.join(d, f)


def drop_rare_phonemes(items, min_count, log):
    """Drop clips containing phonemes seen fewer than min_count times (repeat until stable)."""
    while True:
        counts = collections.Counter(p for it in items for p in it['ph'])
        rare = {p for p, c in counts.items() if c < min_count}
        if not rare:
            return items
        kept = []
        for it in items:
            bad = sorted(rare.intersection(it['ph']))
            if bad:
                log['dropped_rare_phoneme'].append([it['item_name'], bad])
            else:
                kept.append(it)
        log['rare_phonemes'].update({p: counts[p] for p in rare})
        items = kept


def fix_emotion(sing, log):
    valid = {'happy', 'sad'}
    song_key = lambda it: tuple(it['item_name'].split('#')[1:4])
    lang_song = lambda it: (it['language'], it['item_name'].split('#')[3])
    votes = collections.defaultdict(collections.Counter)
    for it in sing:
        if it['emotion'] in valid:
            votes[song_key(it)][it['emotion']] += 1
            votes[lang_song(it)][it['emotion']] += 1
    for it in sing:
        if it['emotion'] not in valid:
            v = votes.get(song_key(it)) or votes.get(lang_song(it))
            new = v.most_common(1)[0][0] if v else 'no'
            log['emotion_fixed'][f"{it['emotion']}->{new}"] = log['emotion_fixed'].get(f"{it['emotion']}->{new}", 0) + 1
            it['emotion'] = new


def dump(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--out_dir', default='data/meta')
    parser.add_argument('--compat', action='store_true', help="no cleaning: reproduce the authors' metadata")
    parser.add_argument('--min_piece_dur', type=float, default=0.03)
    parser.add_argument('--max_absorb', type=float, default=0.1)
    parser.add_argument('--min_ph_count', type=int, default=10)
    parser.add_argument('--max_len_diff', type=float, default=0.1, help='max |wav - annotation| length in seconds')
    parser.add_argument('--workers', type=int, default=8)
    return parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()
    ARGS.data_root = os.path.abspath(ARGS.data_root)
    clips = sorted(find_clips(ARGS.data_root))
    print(f'{len(clips)} annotation files under {ARGS.data_root}')

    sing, speech = [], []
    steps = collections.defaultdict(collections.Counter)
    with Pool(ARGS.workers) as pool:
        for item, is_speech, clog, lang in pool.imap(process_clip, clips, chunksize=64):
            steps[lang + ('/speech' if is_speech else '')].update(clog)
            if item is not None:
                (speech if is_speech else sing).append(item)
    log = dict(steps={k: dict(v) for k, v in sorted(steps.items())}, dropped_rare_phoneme=[], rare_phonemes={},
               emotion_fixed={}, built=dict(singing=len(sing), speech=len(speech)))

    os.makedirs(ARGS.out_dir, exist_ok=True)
    if ARGS.compat:
        dump(sing, os.path.join(ARGS.out_dir, 'metadata_compat.json'))
        print(f'compat: wrote {len(sing)} singing clips')
        raise SystemExit

    kept = drop_rare_phonemes(sing + speech, ARGS.min_ph_count, log)
    kept_names = {it['item_name'] for it in kept}
    sing = [it for it in sing if it['item_name'] in kept_names]
    speech = [it for it in speech if it['item_name'] in kept_names]
    fix_emotion(sing, log)
    for it in sing:  # a singing clip whose paired speech was dropped keeps no dangling pointer
        if it['speech_fn'] and '#'.join(os.path.relpath(it['speech_fn'], ARGS.data_root)[:-4].split(os.sep)) not in kept_names:
            it['speech_fn'] = ''
    log['kept'] = dict(singing=len(sing), speech=len(speech))

    phone_set = sorted({p for it in sing + speech for p in it['ph']})
    singers = sorted({it['singer'] for it in sing}, key=lambda s: (LANGUAGES.index(next(it['language'] for it in sing if it['singer'] == s)), s))
    dump(sing, os.path.join(ARGS.out_dir, 'metadata.json'))
    dump(speech, os.path.join(ARGS.out_dir, 'metadata_speech.json'))
    with open(os.path.join(ARGS.out_dir, 'phone_set.json'), 'w', encoding='utf-8') as f:
        json.dump(phone_set, f, ensure_ascii=False, indent=0)
    with open(os.path.join(ARGS.out_dir, 'spker_set.json'), 'w', encoding='utf-8') as f:
        json.dump({s: i + 1 for i, s in enumerate(singers)}, f, indent=1)
    with open(os.path.join(ARGS.out_dir, 'cleaning_log.json'), 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    print(f"singing {len(sing)} clips, speech {len(speech)} clips, {len(phone_set)} phonemes, "
          f"{len(log['dropped_rare_phoneme'])} clips dropped for rare phonemes")
