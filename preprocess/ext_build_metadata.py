"""Build TCSinger2 metadata for the extra corpora (WP9) from the tree written by ext_unpack.py.

The items have the fields of data/meta/metadata.json (see gtsinger_build_metadata.py): note-split
phoneme pieces with note pitch / note duration / note type (1 rest, 2 single-note word,
3 multi-note word), phonemes with the `_zh` suffix. Extra fields:
  dataset              M4Singer | Opencpop | OpenSinger
  technique_labelled   False: the corpus has no technique annotation (the flags are all 0)
  note_source          score | f0 (f0: pitches are 0 here and filled from F0 by postprocess.py)

M4Singer: meta.json is already note-split. A word is an initial plus its final, or a final alone,
plus the slur pieces that follow; the number of sung words equals the number of characters of the
text in every clip.

Opencpop: built from the phoneme, slur and note tiers of the TextGrids (the syllable and character
tiers are shifted against them in a fifth of the songs), cut at the sentences ext_unpack.py wrote.
Its phonemes are mapped onto the GTSinger / M4Singer convention: the glides `y` and `w` merge into
the final with summed duration (`w o` -> `uo`, `y an` -> `ian`, `y v` -> `v`), and `iu`, `ui`, `un`
become `iou`, `uei`, `uen`. A sung phoneme under a rest note keeps pitch 0 for postprocess.py to fill.

OpenSinger has a phoneme sequence per clip and nothing else, so it takes two steps:
  opensinger_audio   the clips that can be aligned (opensinger_audio.json: wav, rest-free phonemes, text),
                     for mel / F0 extraction and for preprocess/aligner/align.py.
                     About 1,100 label files have lost their last phoneme; it is restored from the
                     last character of the text with a character -> syllable table learnt from the
                     intact clips. Clips under --min_duration or over --max_duration are left out.
  opensinger         the metadata, from opensinger_audio.json and the alignment (opensinger_align.json).
                     Every sung phoneme is one piece with pitch 0 (note_source f0).

Usage:
  python preprocess/ext_build_metadata.py --dataset m4singer --data_root /notebooks/data/ext --out_dir data/meta_ext
  python preprocess/ext_build_metadata.py --dataset opencpop --data_root /notebooks/data/ext --out_dir data/meta_ext
  python preprocess/ext_build_metadata.py --dataset opensinger_audio
  python preprocess/ext_build_metadata.py --dataset opensinger [--min_align_score X]
"""
import argparse
import collections
import json
import os
import re
from multiprocessing import Pool

import soundfile as sf

EXTRA_LOG = {}  # counts from a builder's setup step, written into the log file
REST = ('<AP>', '<SP>')
TECHS = ['mix', 'falsetto', 'breathy', 'pharyngeal', 'vibrato', 'glissando']
INITIALS = set('b p m f d t n l g k h j q x zh ch sh r z c s'.split())
VOICE = {'Soprano': 'female', 'Alto': 'female', 'Tenor': 'male', 'Bass': 'male'}
SUFFIX = '_zh'


def merge_slivers(pieces, log):
    """Note pieces of one phoneme shorter than --min_piece_dur go into the longer neighbouring piece (as in WP1)."""
    out = []
    for word in pieces:
        runs = []  # consecutive pieces of the same phoneme
        for pc in word:
            if runs and runs[-1][-1]['ph'] == pc['ph'] and pc['slur']:
                runs[-1].append(pc)
            else:
                runs.append([pc])
        merged = []
        for run in runs:
            while len(run) > 1 and min(pc['dur'] for pc in run) < ARGS.min_piece_dur:
                i = min(range(len(run)), key=lambda i: run[i]['dur'])
                j = max((j for j in (i - 1, i + 1) if 0 <= j < len(run)), key=lambda j: run[j]['dur'])
                run[j]['dur'] += run[i]['dur']
                if i == 0:
                    run[j]['slur'] = run[i]['slur']
                del run[i]
                log['sliver_merged'] += 1
            merged.extend(run)
        out.append(merged)
    return out


def finish_item(words, texts, wav_fn, log, **fields):
    """words: [[piece, ...], ...] with piece = dict(ph, dur, pitch, note_dur, slur); texts: one label per word."""
    words = merge_slivers(words, log)
    item = collections.defaultdict(list)
    for wid, (word, text) in enumerate(zip(words, texts)):
        rest = word[0]['ph'] in REST
        n_notes = 1 + sum(pc['slur'] for pc in word[1:])
        item['txt'].append(text)
        item['word_durs'].append(round(sum(pc['dur'] for pc in word), 5))
        for pc in word:
            item['ph'].append(pc['ph'] if rest else pc['ph'] + SUFFIX)
            item['ph_durs'].append(round(pc['dur'], 5))
            item['ph2words'].append(wid)
            item['ep_pitches'].append(0 if rest else int(pc['pitch']))
            item['ep_notedurs'].append(round(pc['note_dur'], 5))
            item['ep_types'].append(1 if rest else (3 if n_notes > 1 else 2))
            for t in TECHS:
                item[t + '_tech'].append(0)
            item['tech'].append('0')
    bad = sorted(set(item['ph']) - PHONE_SET)
    if bad:
        return None, 'dropped_unknown_phoneme:' + ','.join(bad)
    if min(item['ph_durs']) <= 0:
        return None, 'dropped_nonpositive_duration'
    info = sf.info(wav_fn)
    duration = info.frames / info.samplerate
    if abs(duration - sum(item['ph_durs'])) > ARGS.max_len_diff:
        return None, 'dropped_length_mismatch'
    item.update(wav_fn=wav_fn, language='Chinese', duration=round(duration, 5), emotion='no', pace='no', range='no',
                singing_method='pop', speech_fn='', technique_labelled=False, **fields)
    return dict(item), None


def m4singer_clip(it):
    log = collections.Counter()
    singer, song, idx = it['item_name'].split('#')
    wav_fn = os.path.join(ARGS.data_root, 'M4Singer', 'M4-' + singer, song, idx + '.wav')
    if not os.path.exists(wav_fn):
        return None, {'missing_wav': 1}
    words, texts, chars = [], [], list(it['txt'])
    phs, slurs = it['phs'], it['is_slur']
    for i, ph in enumerate(phs):
        pc = dict(ph=ph, dur=it['ph_dur'][i], pitch=it['notes'][i], note_dur=it['notes_dur'][i], slur=bool(slurs[i]))
        if ph in REST:
            words.append([dict(pc, slur=False)])
            texts.append(ph)
            continue
        after_own_initial = i > 0 and phs[i - 1] in INITIALS and not slurs[i - 1] and words and len(words[-1]) == 1 \
            and words[-1][0]['ph'] == phs[i - 1] and ph not in INITIALS
        if (pc['slur'] or after_own_initial) and words and words[-1][0]['ph'] not in REST:
            words[-1].append(pc)
        else:
            words.append([dict(pc, slur=False)])
            texts.append(None)
    n_sung = sum(t is None for t in texts)
    if n_sung != len(chars):
        return None, {'dropped_word_count': 1}
    chars = iter(chars)
    texts = [next(chars) if t is None else t for t in texts]
    item, reason = finish_item(words, texts, wav_fn, log, item_name=f'M4Singer#M4-{singer}#{song}#{idx}',
                               singer='M4-' + singer, gender=VOICE[singer.split('-')[0]],
                               vocal_range=singer.split('-')[0].lower(), dataset='M4Singer', note_source='score')
    if item is None:
        log[reason] += 1
    return item, dict(log)


def build_m4singer():
    with open(os.path.join(ARGS.data_root, 'M4Singer', 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    return meta, m4singer_clip


# ---------------------------------------------------------------- Opencpop

OCP_REST = {'SP': '<SP>', 'AP': '<AP>'}
OCP_Y = {'i': 'i', 'in': 'in', 'ing': 'ing', 'a': 'ia', 'an': 'ian', 'ang': 'iang', 'ao': 'iao', 'e': 'ie', 'ong': 'iong',
         'ou': 'iou', 'u': 'v', 'v': 'v', 'van': 'van', 've': 've', 'vn': 'vn'}
OCP_W = {'u': 'u', 'o': 'uo', 'a': 'ua', 'ai': 'uai', 'an': 'uan', 'ang': 'uang', 'ei': 'uei', 'en': 'uen', 'eng': 'ueng'}
OCP_FINAL = {'iu': 'iou', 'ui': 'uei', 'un': 'uen'}
NOTE_NAMES = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def read_textgrid(path):
    """{tier name: [(start, end, text), ...]} of a long-format Praat TextGrid."""
    with open(path, 'rb') as f:
        text = f.read().decode('utf-8')
    tiers = {}
    for block in re.split(r'item \[\d+\]:', text)[1:]:
        name = re.search(r'name = "(.*?)"', block).group(1)
        tiers[name] = [(float(a), float(b), c.strip()) for a, b, c in
                       re.findall(r'xmin = (\S+)\s+xmax = (\S+)\s+text = "(.*?)"', block)]
    return tiers


def note_to_midi(name):
    m = re.fullmatch(r'([A-G])(#?)(-?\d)', name)
    return 12 * (int(m.group(3)) + 1) + NOTE_NAMES[m.group(1)] + len(m.group(2)) if m else 0


def at(tier, t):
    """The interval of a tier that contains time t."""
    return next((iv for iv in tier if iv[0] <= t < iv[1]), tier[-1])


def opencpop_clip(seg):
    log = collections.Counter()
    song, idx = seg['song'], f"{seg['index']:04d}"
    wav_fn = os.path.join(ARGS.data_root, 'Opencpop', 'OCP-1', song, idx + '.wav')
    if not os.path.exists(wav_fn):
        return None, {'missing_wav': 1}
    _, chars, _, notes, _, phones, slurs = OCP_TIERS[song]
    span = [k for k, (a, b, _) in enumerate(phones) if a >= seg['start'] - 1e-4 and b <= seg['end'] + 1e-4]
    words, texts, pending = [], [], None  # pending: an initial waiting for its final

    def piece(k, label, start=None):
        a, b, _ = phones[k]
        a = a if start is None else start
        note = at(notes, (a + b) / 2)
        return dict(ph=label, dur=b - a, pitch=note_to_midi(note[2]), note_dur=note[1] - note[0], slur=False)

    for k in span:
        a, b, ph = phones[k]
        slur = k < len(slurs) and slurs[k][2].strip('0') == '1'
        if ph in OCP_REST:
            if pending is not None:
                return None, {'dropped_initial_without_final': 1}
            words.append([piece(k, OCP_REST[ph])])
            texts.append(OCP_REST[ph])
        elif slur and ph not in INITIALS and ph not in ('y', 'w') and pending is None and words \
                and words[-1][0]['ph'] not in REST and words[-1][-1]['ph'] not in INITIALS:
            words[-1].append(dict(piece(k, words[-1][-1]['ph']), slur=True))
        elif ph in ('y', 'w') or ph in INITIALS:
            if pending is not None:
                return None, {'dropped_initial_without_final': 1}
            pending = k
        else:
            head = phones[pending][2] if pending is not None else None
            if head in ('y', 'w'):
                label = (OCP_Y if head == 'y' else OCP_W).get(ph)
                if label is None:
                    return None, {f'dropped_unmapped_syllable:{head} {ph}': 1}
                words.append([piece(k, label, start=phones[pending][0])])
            else:
                label = OCP_FINAL.get(ph, ph) if head else ph
                words.append(([piece(pending, head)] if head else []) + [piece(k, label)])
            texts.append(at(chars, (a + b) / 2)[2])
            pending = None
    if pending is not None:
        return None, {'dropped_initial_without_final': 1}
    for word in words:  # the pieces of a note share its pitch; a sung piece under a rest note is filled from F0 later
        log['sung_piece_under_rest_note'] += sum(pc['ph'] not in REST and pc['pitch'] == 0 for pc in word)
        log['rest_under_pitched_note'] += sum(pc['ph'] in REST and pc['pitch'] != 0 for pc in word)
    item, reason = finish_item(words, texts, wav_fn, log, item_name=f'Opencpop#OCP-1#{song}#{idx}', singer='OCP-1',
                               gender='female', vocal_range='', dataset='Opencpop', note_source='score')
    if item is None:
        log[reason] += 1
    return item, {k: v for k, v in log.items() if v}


def build_opencpop():
    global OCP_TIERS
    with open(os.path.join(ARGS.data_root, 'Opencpop', 'segments.json'), encoding='utf-8') as f:
        segments = json.load(f)
    raw = os.path.join(ARGS.data_root + '_raw', 'Opencpop', 'textgrids')
    OCP_TIERS = {song: list(read_textgrid(os.path.join(raw, song + '.TextGrid')).values())
                 for song in sorted({s['song'] for s in segments})}
    return segments, opencpop_clip


# ---------------------------------------------------------------- OpenSinger

def lab_syllables(phones):
    out, i = [], 0
    while i < len(phones):
        if phones[i] in INITIALS and i + 1 < len(phones) and phones[i + 1] not in INITIALS:
            out.append((phones[i], phones[i + 1]))
            i += 2
        else:
            out.append((phones[i],))
            i += 1
    return out


def opensinger_labels():
    """[(rel, phones, characters, repaired)] of the usable label files, and a log."""
    with open(os.path.join(ARGS.data_root, 'OpenSinger', 'labels.json'), encoding='utf-8') as f:
        labels = json.load(f)
    log, parsed = collections.Counter(), []
    lexicon = collections.defaultdict(collections.Counter)  # character -> syllables seen for it
    for rel, d in sorted(labels.items()):
        phones = [p for p in d.get('lab', '').split() if p != '<EOS>']
        chars = re.findall(r'[\u4e00-\u9fff]', d.get('txt', ''))
        if not phones or not chars:
            log['dropped_no_label'] += 1
            continue
        if any(p + SUFFIX not in PHONE_SET for p in phones):
            log['dropped_unknown_phoneme'] += 1
            continue
        syls = lab_syllables(phones)
        lone = [s for s in syls if len(s) == 1 and s[0] in INITIALS]
        parsed.append((rel, phones, chars, syls, lone))
        if len(syls) == len(chars) and not lone:
            for ch, syl in zip(chars, syls):
                lexicon[ch][syl] += 1
    out = []
    for rel, phones, chars, syls, lone in parsed:
        repaired = False
        if lone or len(syls) != len(chars):  # the label file lost its last phoneme
            last = lexicon.get(chars[-1], {})
            if lone == [syls[-1]] and len(syls) == len(chars):
                cands = {s: n for s, n in last.items() if len(s) == 2 and s[0] == phones[-1]}
            elif not lone and len(syls) == len(chars) - 1:
                cands = {s: n for s, n in last.items() if len(s) == 1}
            else:
                cands = {}
            if not cands:
                log['dropped_label_text_mismatch'] += 1
                continue
            phones = phones + [max(cands, key=cands.get)[-1]]
            repaired = True
            log['last_phoneme_restored'] += 1
        out.append((rel, phones, chars, repaired))
    return out, log


def opensinger_audio_clip(job):
    rel, phones, chars, repaired = job
    wav_fn = os.path.join(ARGS.data_root, 'OpenSinger', rel + '.wav')
    if not os.path.exists(wav_fn):
        return None, {'missing_wav': 1}
    info = sf.info(wav_fn)
    duration = info.frames / info.samplerate
    if duration < ARGS.min_duration:
        return None, {'dropped_too_short': 1}
    if duration > ARGS.max_duration:
        return None, {'dropped_too_long': 1}
    singer, song, idx = rel.split('/')
    return dict(item_name=f'OpenSinger#{singer}#{song}#{idx}', wav_fn=wav_fn, duration=round(duration, 5), singer=singer,
                gender='female' if singer.startswith('OS-Woman') else 'male', ph=[p + SUFFIX for p in phones],
                txt=chars, label_repaired=repaired), {}


def build_opensinger_audio():
    jobs, log = opensinger_labels()
    EXTRA_LOG.update(log)
    return jobs, opensinger_audio_clip


def opensinger_clip(it):
    log = collections.Counter()
    al = OS_ALIGN.get(it['item_name'])
    if al is None:
        return None, {'dropped_not_aligned': 1}
    if ARGS.min_align_score is not None and al['score'] < ARGS.min_align_score:
        return None, {'dropped_low_align_score': 1}
    fps = 48000 / 256
    durs = [n / fps for n in al['frames']]
    durs[-1] += it['duration'] - sum(durs)  # the mel has a few frames less than the audio
    chars = iter(it['txt'])
    words, texts = [], []
    for ph, dur in zip(al['ph'], durs):
        pc = dict(ph=ph if ph in REST else ph[:-len(SUFFIX)], dur=dur, pitch=0, note_dur=dur, slur=False)
        if ph in REST:
            words.append([pc])
            texts.append(ph)
        elif words and len(words[-1]) == 1 and words[-1][0]['ph'] in INITIALS and pc['ph'] not in INITIALS:
            words[-1].append(pc)
        else:
            words.append([pc])
            texts.append(next(chars, '?'))
    item, reason = finish_item(words, texts, it['wav_fn'], log, item_name=it['item_name'], singer=it['singer'],
                               gender=it['gender'], vocal_range='', dataset='OpenSinger', note_source='f0')
    if item is None:
        log[reason] += 1
    else:
        item.update(align_score=round(al['score'], 4), label_repaired=it['label_repaired'])
    return item, dict(log)


def build_opensinger():
    global OS_ALIGN
    with open(os.path.join(ARGS.out_dir, 'opensinger_audio.json'), encoding='utf-8') as f:
        clips = json.load(f)
    with open(os.path.join(ARGS.out_dir, 'opensinger_align.json'), encoding='utf-8') as f:
        OS_ALIGN = json.load(f)
    return clips, opensinger_clip


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['m4singer', 'opencpop', 'opensinger_audio', 'opensinger'])
    parser.add_argument('--min_duration', type=float, default=2.0, help='OpenSinger: shorter clips are left out')
    parser.add_argument('--max_duration', type=float, default=20.0, help='OpenSinger: longer clips are left out')
    parser.add_argument('--min_align_score', type=float, default=None, help='OpenSinger: clips aligned with a lower score are dropped')
    parser.add_argument('--data_root', default='/notebooks/data/ext')
    parser.add_argument('--out_dir', default='data/meta_ext')
    parser.add_argument('--phone_set', default='data/meta/phone_set.json')
    parser.add_argument('--min_piece_dur', type=float, default=0.03)
    parser.add_argument('--max_len_diff', type=float, default=0.1, help='max |wav - annotation| length in seconds')
    parser.add_argument('--workers', type=int, default=8)
    ARGS = parser.parse_args()
    ARGS.data_root = os.path.abspath(ARGS.data_root)
    with open(ARGS.phone_set, encoding='utf-8') as f:
        PHONE_SET = set(json.load(f))

    clips, build = dict(m4singer=build_m4singer, opencpop=build_opencpop, opensinger_audio=build_opensinger_audio,
                        opensinger=build_opensinger)[ARGS.dataset]()
    items, log = [], collections.Counter(EXTRA_LOG)
    with Pool(ARGS.workers) as pool:
        for item, clog in pool.imap(build, clips, chunksize=64):
            log.update(clog)
            if item is not None:
                items.append(item)
    os.makedirs(ARGS.out_dir, exist_ok=True)
    with open(os.path.join(ARGS.out_dir, ARGS.dataset + '.json'), 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False)
    with open(os.path.join(ARGS.out_dir, ARGS.dataset + '_log.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(clips_in=len(clips), clips_out=len(items), steps=dict(sorted(log.items()))), f, ensure_ascii=False, indent=1)
    hours = sum(it['duration'] for it in items) / 3600
    print(f'{ARGS.dataset}: {len(clips)} -> {len(items)} clips, {hours:.2f} h, {len({it["singer"] for it in items})} singers; {dict(log)}')
