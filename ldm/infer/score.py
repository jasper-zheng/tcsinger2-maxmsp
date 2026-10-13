"""The score: what is to be sung, read from a JSON file (WP7).

    {"language": "English", "tempo": 112,
     "style": {"singer": "female alto", "emotion": "happy", "singing_method": "pop", "technique": "breathy"},
     "words": [
       {"rest": 1.0},
       {"text": "where", "notes": [["C4", 1]]},
       {"text": "wind",  "notes": [["A#3", 2], [60, 1]]},
       {"rest": 0.5, "type": "AP"}],
     "f0": [0, 0, 0, 261.4, 262.0, 263.1, ...]}

  language   one of ldm.data.phone_set.language_set
  tempo      beats per minute; lengths are then beats, without it seconds
  style      optional caption fields (ldm/data/captions.py); the command line overrides them
  words      in singing order. A sung word has `text` and `notes`, a list of [pitch, length] with the pitch
             as a MIDI number or a name (C4 = 60, C#4, Db4). The word lasts as long as its notes together.
             A rest is {"rest": length}, with "type" "AP" (breath) or "SP" (silence); by default the first
             word of a score is a silence and any other rest a breath, as mostly in the training data.
  f0         optional: the F0 curve to sing, in Hz per frame from the start of the score, 0 (or null) where
             unvoiced. Frames are 1/187.5 s (48 kHz, hop 256) unless "f0_fps" gives another rate. The output
             follows this curve; without it the model predicts the F0 from the notes. Every output json of
             scripts/sing.py holds the curve that was sung in the same two keys, so it can be copied here.
  optional per sung word:
    ph                  its phonemes, as in data/meta/phone_set.json with or without the language suffix
    ph_durs             seconds per phoneme of `ph` (or of the phonemes found for `text`)
    notes_per_syllable  how many of the notes each syllable takes, e.g. [1, 2]; 0 = the syllable shares
                        the last note of the one before

Chinese: one character or one pinyin syllable per word.
"""
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ldm.data.captions import row_fields
from ldm.data.phone_set import NUM_MIDI, language_set

MAX_SECONDS = 20.0  # clips in training are at most 20 s (spec_crop_len 3760 frames)
F0_FPS = 48000 / 256  # frames per second of the model, and of an "f0" without "f0_fps"
MAX_MIDI = NUM_MIDI - 2  # 98
REST_TYPES = {'AP': '<AP>', 'SP': '<SP>', '<AP>': '<AP>', '<SP>': '<SP>'}
_STEPS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
_WORD_KEYS = {'text', 'notes', 'ph', 'ph_durs', 'notes_per_syllable'}
_REST_KEYS = {'rest', 'type'}
_TOP_KEYS = {'language', 'tempo', 'style', 'words', 'name', 'f0', 'f0_fps'}


class ScoreError(ValueError):
    pass


@dataclass
class Word:
    text: str                                 # '<AP>' / '<SP>' for a rest
    notes: List[Tuple[int, float]]            # (MIDI pitch, seconds); one (0, seconds) for a rest
    ph: Optional[List[str]] = None
    ph_durs: Optional[List[float]] = None
    notes_per_syllable: Optional[List[int]] = None

    @property
    def is_rest(self):
        return self.text in ('<AP>', '<SP>')

    @property
    def seconds(self):
        return sum(n[1] for n in self.notes)


@dataclass
class Score:
    language: str
    words: List[Word]
    style: dict = field(default_factory=dict)
    name: str = 'score'
    f0: Optional[List[float]] = None          # Hz per frame at f0_fps, 0 = unvoiced
    f0_fps: float = F0_FPS

    @property
    def seconds(self):
        return sum(w.seconds for w in self.words)


def note_to_midi(pitch):
    """60, '60', 'C4', 'C#4', 'Db4' -> MIDI number."""
    if isinstance(pitch, (int, float)) and not isinstance(pitch, bool) and float(pitch).is_integer():
        return int(pitch)
    m = re.fullmatch(r'\s*([A-Ga-g])([#b♯♭]?)(-?\d+)\s*', str(pitch))
    if m:
        return _STEPS[m.group(1).upper()] + {'': 0, '#': 1, '♯': 1, 'b': -1, '♭': -1}[m.group(2)] + 12 * (int(m.group(3)) + 1)
    if re.fullmatch(r'\s*\d+\s*', str(pitch)):
        return int(pitch)
    raise ScoreError(f'bad pitch {pitch!r}: use a MIDI number or a name like C4, F#3, Bb4')


def _length(value, beat, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ScoreError(f'{where}: bad length {value!r}, expected a number above 0')
    return float(value) * beat


def parse_score(data, name='score', max_seconds=MAX_SECONDS):
    """Score from the dict of a JSON score file; raises ScoreError with the place of the first problem."""
    if not isinstance(data, dict):
        raise ScoreError('the score must be a JSON object with "language" and "words"')
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise ScoreError(f'unknown key(s) {sorted(unknown)}; allowed: {sorted(_TOP_KEYS)}')
    language = data.get('language')
    if language not in language_set:
        raise ScoreError(f'bad language {language!r}: expected one of {language_set}')
    tempo = data.get('tempo')
    if tempo is not None and (isinstance(tempo, bool) or not isinstance(tempo, (int, float)) or tempo <= 0):
        raise ScoreError(f'bad tempo {tempo!r}: beats per minute, a number above 0')
    beat = 60.0 / tempo if tempo else 1.0
    style = data.get('style') or {}
    if not isinstance(style, dict) or set(style) - {'singer', 'emotion', 'singing_method', 'technique'}:
        raise ScoreError('"style" may hold singer, emotion, singing_method and technique')
    entries = data.get('words')
    if not isinstance(entries, list) or not entries:
        raise ScoreError('"words" must be a list with at least one entry')

    words = []
    for i, en in enumerate(entries):
        where = f'words[{i}]'
        if not isinstance(en, dict):
            raise ScoreError(f'{where}: expected an object like {{"text": ..., "notes": [[pitch, length]]}} or {{"rest": length}}')
        if 'rest' in en:
            if set(en) - _REST_KEYS:
                raise ScoreError(f'{where}: a rest takes "rest" and "type" only, found {sorted(set(en) - _REST_KEYS)}')
            kind = en.get('type', 'SP' if i == 0 else 'AP')
            if kind not in REST_TYPES:
                raise ScoreError(f'{where}: bad rest type {kind!r}, expected "AP" (breath) or "SP" (silence)')
            words.append(Word(REST_TYPES[kind], [(0, _length(en['rest'], beat, where))]))
            continue
        if set(en) - _WORD_KEYS:
            raise ScoreError(f'{where}: unknown key(s) {sorted(set(en) - _WORD_KEYS)}; allowed: {sorted(_WORD_KEYS)}')
        text = en.get('text')
        if not isinstance(text, str) or not text.strip():
            if not en.get('ph'):
                raise ScoreError(f'{where}: a sung word needs "text" (or "ph")')
            text = ' '.join(en['ph'])
        where = f'words[{i}] ({text!r})'
        notes = en.get('notes')
        if not isinstance(notes, list) or not notes:
            raise ScoreError(f'{where}: "notes" must be a list of [pitch, length]')
        parsed = []
        for note in notes:
            if not isinstance(note, (list, tuple)) or len(note) != 2:
                raise ScoreError(f'{where}: bad note {note!r}, expected [pitch, length]')
            try:
                midi = note_to_midi(note[0])
            except ScoreError as e:
                raise ScoreError(f'{where}: {e}') from None
            if not 1 <= midi <= MAX_MIDI:
                raise ScoreError(f'{where}: pitch {note[0]!r} is MIDI {midi}, outside 1..{MAX_MIDI}; a rest is {{"rest": length}}')
            parsed.append((midi, _length(note[1], beat, where)))
        ph, ph_durs, nps = en.get('ph'), en.get('ph_durs'), en.get('notes_per_syllable')
        if ph is not None and (not isinstance(ph, list) or not ph or not all(isinstance(p, str) for p in ph)):
            raise ScoreError(f'{where}: "ph" must be a list of phonemes')
        if ph_durs is not None:
            if not isinstance(ph_durs, list) or not all(isinstance(d, (int, float)) and d > 0 for d in ph_durs):
                raise ScoreError(f'{where}: "ph_durs" must be a list of seconds above 0')
            if ph is not None and len(ph) != len(ph_durs):
                raise ScoreError(f'{where}: {len(ph)} phonemes but {len(ph_durs)} lengths')
        if nps is not None:
            if not isinstance(nps, list) or not all(isinstance(k, int) and k >= 0 for k in nps) or not nps or nps[0] < 1 \
                    or sum(nps) != len(parsed):
                raise ScoreError(f'{where}: "notes_per_syllable" must be whole numbers that start above 0 and add up to the '
                                 f'{len(parsed)} note(s) of the word')
        words.append(Word(text.strip(), parsed, ph, [float(d) for d in ph_durs] if ph_durs else None, nps))

    f0, f0_fps = parse_f0(data.get('f0'), data.get('f0_fps'))
    score = Score(language, words, dict(style), data.get('name') or name, f0, f0_fps)
    if all(w.is_rest for w in words):
        raise ScoreError('the score has no sung word')
    if max_seconds and score.seconds > max_seconds + 1e-6:
        raise ScoreError(f'the score lasts {score.seconds:.2f} s; the model was trained on clips of at most {max_seconds:.0f} s '
                         f'and longer scores are not supported yet. Split it at a rest and sing the parts one by one.')
    return score


def parse_f0(f0, f0_fps=None, where='"f0"'):
    """(values in Hz with 0 for unvoiced, frames per second) of an F0 array as JSON holds it; (None, rate) without one."""
    if f0_fps is None:
        f0_fps = F0_FPS
    elif isinstance(f0_fps, bool) or not isinstance(f0_fps, (int, float)) or f0_fps <= 0:
        raise ScoreError(f'bad "f0_fps" {f0_fps!r}: frames per second, a number above 0')
    if f0 is None:
        return None, float(f0_fps)
    if not isinstance(f0, list) or not f0:
        raise ScoreError(f'{where} must be a list of Hz values, one per frame, 0 where unvoiced')
    for i, v in enumerate(f0):
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or v != v):
            raise ScoreError(f'{where}[{i}]: bad value {v!r}, expected Hz as a number, or 0 / null where unvoiced')
    return [float(v or 0) for v in f0], float(f0_fps)


def load_score(path, max_seconds=MAX_SECONDS):
    import os
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ScoreError(f'{path} is not valid JSON: {e}') from None
    return parse_score(data, os.path.splitext(os.path.basename(path))[0], max_seconds)


def score_from_row(row, with_ph=False, with_durs=False, with_f0=False):
    """Score dict (seconds) of a manifest row, for checks: the words with the notes as the manifest has them.

    A note of a word is a run of pieces with one pitch (a phoneme that is repeated starts a new note), as
    long as its pieces. with_ph adds the annotated phonemes, with_durs also their lengths, with_f0 the
    clip's F0 (the sung performance; it fits the words exactly only together with with_durs)."""
    import ast
    lit = lambda v: ast.literal_eval(v) if isinstance(v, str) else list(v)
    ph, durs, ph2w, pitches, txt = (lit(row[k]) for k in ('ph', 'ph_durs', 'ph2words', 'ep_pitches', 'txt'))
    words = []
    for w in sorted(set(ph2w)):
        idx = [i for i, x in enumerate(ph2w) if x == w]
        if ph[idx[0]] in ('<AP>', '<SP>'):
            for i in idx:  # one rest per rest phoneme
                words.append({'rest': round(durs[i], 5), 'type': ph[i][1:-1]})
            continue
        notes, phs, pdurs = [], [], []
        for k, i in enumerate(idx):
            new_ph = k == 0 or ph[i] != ph[idx[k - 1]]
            if k == 0 or pitches[i] != pitches[idx[k - 1]] or not new_ph:
                notes.append([int(pitches[i]), 0.0])
            notes[-1][1] += durs[i]
            if new_ph:
                phs.append(ph[i])
                pdurs.append(0.0)
            pdurs[-1] += durs[i]
        en = {'text': txt[w], 'notes': [[p, round(d, 5)] for p, d in notes]}  # pitch 0 (rare): parse_score refuses it
        if with_ph or with_durs:
            en['ph'] = phs
        if with_durs:
            en['ph_durs'] = [round(d, 5) for d in pdurs]
        words.append(en)
    style = {k: v for k, v in row_fields(row).items() if v is not None}
    score = {'language': row['language'], 'name': row['item_name'], 'style': style, 'words': words}
    if with_f0:
        import numpy as np
        n = int(row['n_frames']) if 'n_frames' in row else None
        score['f0'] = [round(float(v), 2) for v in np.load(row['f0_path'])[:n]]
    return score
