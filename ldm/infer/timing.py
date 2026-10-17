"""Phonemes and notes to the pieces of the training metadata, their lengths, and the per-frame tracks (WP7).

A piece is a phoneme inside one note, with the note's pitch and length and the word's note type (1 rest,
2 word on one note, 3 word on several notes), as in data/meta/metadata.json.

Which phoneme goes on which note (build_pieces):
  * a word is cut into syllables around its nuclei (vowels, syllabic consonants, Chinese finals). One
    consonant between two nuclei opens the next syllable; of several the first half closes the syllable
    before. In the training annotations the note changes before a single consonant in 55 to 77% of the
    cases and splits two consonants in about 60%.
  * as many notes as syllables: one each. More notes: the extra notes go to the last syllable, whose
    nucleus is repeated on each of them (`notes_per_syllable` in the score chooses another syllable).
    Fewer notes: the syllables are spread evenly over the word's time and each goes on the note it falls in.
  * a word with `ph_durs` is cut at the note boundaries by time, as the metadata builder cuts the annotations.

Lengths (fill_durations): the duration predictor trained with the VAE (WP4) gives every piece a length;
the pieces of one note are then fitted to the note's length. Rests last as long as the score says.
"""
import unicodedata
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ldm.data.phone_set import ph2id
from ldm.data.tcsinger2_dataset import HOP_SIZE, SAMPLE_RATE, blur_boundaries, frame_tracks

FPS = SAMPLE_RATE / HOP_SIZE  # 187.5 mel frames per second
REST = ('<AP>', '<SP>')
IPA_VOWELS = set('aeiouyøœɐɑɔəɛɪɨɯɵʉʊʌʏæ')
ZH_INITIALS = {'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's'}
MIN_PRED = 0.01      # seconds; a predicted length below this is raised to it before the fit
MIN_NUCLEUS = 0.4    # a nucleus keeps at least this share of its predicted length, else the whole note is scaled
TIE = 0.01           # frames: a boundary this close below a half frame rounds up, like one on it (see to_frames)


@dataclass
class Piece:
    ph: str
    word: int                    # index of the word in the score
    note: int                    # index of the note inside the word
    midi: int
    notedur: float
    type: int
    nucleus: bool
    dur: Optional[float] = None  # seconds; None until fill_durations
    given: bool = False          # length fixed by the score (rest or ph_durs)
    index: int = 0               # which phoneme of the word


def is_nucleus(ph):
    """True for the phonemes that carry a syllable: they are the ones stretched over several notes."""
    if ph in REST:
        return False
    base, lang = ph.rsplit('_', 1)
    if lang == 'en':
        return base[-1].isdigit()
    if lang == 'zh':
        return base not in ZH_INITIALS
    d = unicodedata.normalize('NFD', base)
    return d[0] in IPA_VOWELS or '̩' in d


def syllabify(phs):
    """[(onset indices, nucleus index, coda indices)] of a word's phonemes. A word without a nucleus is one
    syllable whose last phoneme stands in for it."""
    nuclei = [i for i, p in enumerate(phs) if is_nucleus(p)]
    if not nuclei:
        return [(list(range(len(phs) - 1)), len(phs) - 1, [])]
    syllables = [[[], n, []] for n in nuclei]
    syllables[0][0] = list(range(nuclei[0]))
    syllables[-1][2] = list(range(nuclei[-1] + 1, len(phs)))
    for j, (a, b) in enumerate(zip(nuclei[:-1], nuclei[1:])):
        between = list(range(a + 1, b))
        k = 0 if len(between) == 1 else len(between) // 2
        syllables[j][2] = between[:k]
        syllables[j + 1][0] = between[k:]
    return [tuple(s) for s in syllables]


def syllable_notes(n_syllables, note_lens, notes_per_syllable=None):
    """For each syllable the notes it is sung on (indices; consecutive syllables may share a note)."""
    K, S = len(note_lens), n_syllables
    if notes_per_syllable is not None:
        if len(notes_per_syllable) != S:
            raise ValueError(f'"notes_per_syllable" has {len(notes_per_syllable)} entries but the word has {S} syllable(s)')
        out, k = [], 0
        for n in notes_per_syllable:
            out.append(list(range(k, k + n)) if n else [k - 1])
            k += n
        return out
    if K >= S:
        return [[j] for j in range(S - 1)] + [list(range(S - 1, K))]
    ends = np.cumsum(note_lens)
    first = [int(min(np.searchsorted(ends, (j + 0.5) / S * ends[-1], side='right'), K - 1)) for j in range(S)]
    first[0] = 0
    out = []
    for j in range(S):
        last = (first[j + 1] - 1 if first[j + 1] > first[j] else first[j]) if j + 1 < S else K - 1
        out.append(list(range(first[j], last + 1)))
    return out


def build_pieces(score, phs):
    """Pieces of a Score whose words have the phonemes `phs` (ldm.infer.g2p.phonemize)."""
    pieces = []
    for wi, (word, wph) in enumerate(zip(score.words, phs)):
        if word.is_rest:
            pieces.append(Piece(word.text, wi, 0, 0, word.seconds, 1, False, word.seconds, True))
            continue
        note_lens = [n[1] for n in word.notes]
        if word.ph_durs is not None:
            if len(word.ph_durs) != len(wph):
                raise ValueError(f'words[{wi}] ({word.text!r}): {len(word.ph_durs)} "ph_durs" for the {len(wph)} phonemes '
                                 f'{" ".join(p.rsplit("_", 1)[0] for p in wph)}')
            scale = word.seconds / sum(word.ph_durs)  # the notes decide how long the word is
            bounds = np.cumsum(note_lens)
            start, new = 0.0, []
            for pi, (p, d) in enumerate(zip(wph, word.ph_durs)):
                end = start + d * scale
                cuts = [start] + [b for b in bounds[:-1] if start + 1e-6 < b < end - 1e-6] + [end]
                for a, b in zip(cuts[:-1], cuts[1:]):
                    k = int(min(np.searchsorted(bounds, (a + b) / 2, side='right'), len(bounds) - 1))
                    new.append(Piece(p, wi, k, word.notes[k][0], word.notes[k][1], 2, is_nucleus(p), b - a, True, pi))
                start = end
        else:
            syllables = syllabify(wph)
            try:
                notes_of = syllable_notes(len(syllables), note_lens, word.notes_per_syllable)
            except ValueError as e:
                raise ValueError(f'words[{wi}] ({word.text!r}): {e}') from None
            new = []
            for (onset, nucleus, coda), ks in zip(syllables, notes_of):
                mk = lambda i, k, nuc=False: Piece(wph[i], wi, k, word.notes[k][0], word.notes[k][1], 2, nuc, index=i)
                new += [mk(i, ks[0]) for i in onset] + [mk(nucleus, k, True) for k in ks] + [mk(i, ks[-1]) for i in coda]
        note_type = 3 if len({p.note for p in new}) > 1 else 2
        for p in new:
            p.type = note_type
        pieces += new
    return pieces


class DurationPredictor:
    """The duration predictor trained with the VAE (`AutoencoderKL.pred_duration`), on a list of pieces."""

    def __init__(self, vae):
        self.vae = vae

    def __call__(self, pieces):
        import torch
        dev = next(self.vae.parameters()).device
        row = lambda values, dtype: torch.tensor([values], dtype=dtype, device=dev)
        ph = row([ph2id[p.ph] for p in pieces], torch.long)
        with torch.no_grad(), torch.autocast(dev.type, enabled=False):
            log_dur = self.vae.pred_duration(ph, row([p.notedur for p in pieces], torch.float), row([p.midi for p in pieces], torch.long),
                                             row([p.type for p in pieces], torch.long), torch.zeros_like(ph, dtype=torch.bool))
        return torch.expm1(log_dur.float())[0].clamp(min=0).cpu().numpy()


def load_duration_model(config='configs/tcsinger2.yaml', device='cpu'):
    """The VAE alone (it holds the duration predictor), for a front-end run without the stage-2 model."""
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):  # the constructor prints its layer list
        vae = instantiate_from_config(OmegaConf.load(config).model.params.first_stage_config)
    return DurationPredictor(vae.to(device).eval())


def fill_durations(pieces, predictor, fit='nucleus'):
    """Sets `dur` of the pieces that have none: predicted, then fitted note by note.

    fit 'nucleus': the consonants of a note keep their predicted length and its nuclei take the rest, unless
    that leaves them less than MIN_NUCLEUS of their own prediction; then, and with fit 'scale', all pieces of
    the note are scaled alike. fit 'none' keeps the predictions (the score's timing is then not kept).
    Returns the raw predictions (seconds per piece)."""
    pred = np.asarray(predictor(pieces), dtype=float)
    groups = {}
    for i, p in enumerate(pieces):
        if not p.given:
            groups.setdefault((p.word, p.note), []).append(i)
    for idx in groups.values():
        want = np.maximum(pred[idx], MIN_PRED)
        total = pieces[idx[0]].notedur
        if fit == 'none':
            lens = want
        else:
            nuc = np.array([pieces[i].nucleus for i in idx])
            rest = total - want[~nuc].sum()
            if fit == 'nucleus' and nuc.any() and rest >= MIN_NUCLEUS * want[nuc].sum():
                lens = np.where(nuc, want * rest / want[nuc].sum(), want)
            else:
                lens = want * total / want.sum()
        for i, d in zip(idx, lens):
            pieces[i].dur = float(d)
    return pred


def to_frames(pieces, blur=False):
    """Per-frame tracks of timed pieces, through the training code.

    Piece boundaries are rounded to frames on the running time, so that they do not drift against a given
    F0. A boundary on a half frame (a note boundary at a whole second is one: 1 s = 187.5 frames) rounds
    up, and so does one up to TIE frames below it, so that the float32 front-end of the Max export
    (EWP3) rounds every boundary the same way. Returns dict(mel2ph, ph, midi: int32 [T]; starts: first
    frame of every piece, with T appended)."""
    ends = np.floor(np.cumsum([p.dur for p in pieces]) * FPS + 0.5 + TIE).astype(int)
    starts = np.concatenate([[0], ends])
    durs = np.diff(starts) / FPS  # whole frames each; frame_tracks rounds them to the same numbers
    mel2ph, ph, midi = frame_tracks([ph2id[p.ph] for p in pieces], durs, [p.midi for p in pieces], int(ends[-1]))
    if blur:
        blur_boundaries(mel2ph, ph, midi, centre=True)
    return dict(mel2ph=mel2ph, ph=ph, midi=midi, starts=starts)
