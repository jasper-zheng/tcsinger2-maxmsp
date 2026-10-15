"""What the F0 generator reads and writes, per mel frame (WP10). Shared by training and inference.

Inputs per frame (note_tracks): phoneme id, MIDI pitch, note type (1 rest, 2 single-note word, 3 multi-note
word), and where the frame lies in its note (frames since the onset, frames to the end, note length). The
tracks are not boundary-blurred: the generator needs the note edges.

Notes. At inference they are exact (ldm.infer.timing.Piece: word, note). The manifests store note-split
pieces without a note number (data/meta/metadata.json), so note_index infers it: a piece opens a new note
when its word changes, its pitch or note length changes, or it repeats the phoneme before it inside the same
word (a vowel held over several notes); rests are notes of their own.

Target per frame (encode / decode): r = semitone(F0) - reference pitch, and voicing (+1 voiced, -1 not).
The reference pitch is the note's MIDI pitch; in rests it is interpolated between the neighbouring notes.
In unvoiced frames r is interpolated from the voiced ones, so that the curve is continuous; r is clipped
to +-R_CLIP, and frames with |r| > R_TRUST (tracking or annotation errors) are left out of the loss.
"""
import numpy as np

from ldm.data.phone_set import MIDI_PAD_ID, NOTE_TYPE_PAD, PH_PAD_ID
from ldm.data.tcsinger2_dataset import frame_tracks

R_CLIP, R_TRUST = 12.0, 7.0
POS_SCALE = 1.0 / np.log(1 + 750.0)  # position features: log(1 + frames) scaled to about 1 at 4 s


def note_index(ph, ph2words, ep_pitches, ep_notedurs):
    """Note number of every piece of a manifest row (see the module text)."""
    out, k = [], -1
    for i in range(len(ph)):
        new = i == 0 or ph2words[i] != ph2words[i - 1] or ep_pitches[i] != ep_pitches[i - 1] \
            or abs(ep_notedurs[i] - ep_notedurs[i - 1]) > 1e-4 or ph[i] == ph[i - 1] \
            or ep_pitches[i] == 0 or ep_pitches[i - 1] == 0
        k += new
        out.append(k)
    return out


def note_tracks(ph_ids, ph_durs, ep_pitches, ep_types, note_ids, n_frames):
    """Per-frame tracks [n_frames] of pieces (ids, seconds, MIDI, note type, note number).

    Returns dict(ph, midi, ntype: int; pos: float [3, T] = log frames since the note's onset, to its end,
    note length (scaled); onset: 1 on a note's first frame; mel2ph as frame_tracks gives it)."""
    mel2ph, ph, midi = frame_tracks(ph_ids, ph_durs, ep_pitches, n_frames)
    piece = mel2ph.astype(np.int64) - 1                     # -1 = no piece
    ok = piece >= 0
    ntype = np.full(n_frames, NOTE_TYPE_PAD, dtype=np.int64)
    ntype[ok] = np.asarray(ep_types, dtype=np.int64)[piece[ok]]
    note = np.full(n_frames, -1, dtype=np.int64)
    note[ok] = np.asarray(note_ids, dtype=np.int64)[piece[ok]]
    pos = np.zeros((3, n_frames), dtype=np.float32)
    onset = np.zeros(n_frames, dtype=np.float32)
    start = 0
    for t in range(1, n_frames + 1):
        if t == n_frames or note[t] != note[start]:
            if note[start] >= 0:
                n = t - start
                idx = np.arange(n)
                pos[0, start:t] = np.log1p(idx)
                pos[1, start:t] = np.log1p(n - 1 - idx)
                pos[2, start:t] = np.log1p(n)
                onset[start] = 1
            start = t
    pos *= POS_SCALE
    return dict(ph=ph.astype(np.int64), midi=midi.astype(np.int64), ntype=ntype, pos=pos, onset=onset, mel2ph=mel2ph)


def reference_pitch(midi):
    """The note pitch per frame (float); rests and padding interpolated between the notes around them."""
    midi = np.asarray(midi)
    sung = (midi > 0) & (midi < MIDI_PAD_ID)
    if not sung.any():
        return np.full(len(midi), 60.0, dtype=np.float32)
    t = np.arange(len(midi))
    return np.interp(t, t[sung], midi[sung].astype(np.float64)).astype(np.float32)


def semitone(f0):
    return 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440.0)


def encode(f0, ref):
    """(r [T], voicing [T] in {-1, 1}, loss mask of r [T]) of an F0 in Hz against the reference pitch."""
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = f0 > 0
    r = np.zeros(len(f0))
    if voiced.any():
        t = np.arange(len(f0))
        r = np.interp(t, t[voiced], (semitone(f0[voiced]) - ref[voiced]))
    trust = voiced & (np.abs(r) <= R_TRUST)
    r = np.clip(r, -R_CLIP, R_CLIP)
    return r.astype(np.float32), np.where(voiced, 1.0, -1.0).astype(np.float32), trust.astype(np.float32)


def decode(r, v, ref):
    """F0 in Hz (0 = unvoiced) from r, voicing (> 0 = voiced) and the reference pitch. Works on numpy or torch."""
    if isinstance(r, np.ndarray):
        return np.where(v > 0, 440.0 * 2 ** ((ref + r - 69) / 12), 0.0).astype(np.float32)
    import torch
    return torch.where(v > 0, 440.0 * 2 ** ((ref + r - 69) / 12), torch.zeros_like(r))


def prompt_features(f0):
    """[2, T] of a prompt clip's F0: semitones relative to 60 / 12 (0 where unvoiced) and voicing (+-1)."""
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = f0 > 0
    pitch = np.where(voiced, (semitone(f0) - 60) / 12, 0.0)
    return np.stack([pitch, np.where(voiced, 1.0, -1.0)]).astype(np.float32)


def tracks_from_pieces(pieces, starts):
    """note_tracks of timed inference pieces (ldm.infer.timing.Piece) with the frame starts of
    timing.to_frames (n pieces + 1 values)."""
    from ldm.data.phone_set import ph2id
    durs = np.diff(starts) / 187.5
    keys, note_ids = {}, []
    for i, p in enumerate(pieces):
        key = ('rest', i) if p.midi == 0 else (p.word, p.note)
        note_ids.append(keys.setdefault(key, len(keys)))
    return note_tracks([ph2id[p.ph] for p in pieces], durs, [p.midi for p in pieces], [p.type for p in pieces],
                       note_ids, int(starts[-1]))
