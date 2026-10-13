"""The optional F0 input: read it, bring it to the model's frame rate and compare it with the score (WP7).

Where the curve comes from. scripts/sing.py takes it from the score only; the files are for callers of
ldm.infer.pipeline.Singer.sing(f0=<path>):
  the score       "f0" in the score's JSON: Hz per frame, 0 (or null) = unvoiced, at 187.5 frames/s (48 kHz,
                  hop 256) or at the score's "f0_fps" (ldm/infer/score.py)
  .json           the same two keys in another JSON file, e.g. the json that scripts/sing.py writes next to
                  an output, or another score; or a file that is just the list
  .txt .csv .tsv  one value per line, or two columns: time in seconds, F0 in Hz
  audio           a sung or hummed recording; the F0 is tracked with RMVPE as for the training data

The curve must lie on the score's time axis: frame 0 is the start of the score. The model sings the given
curve, so what it contradicts in the score is reported (check_f0) instead of being corrected.
"""
import os

import numpy as np

from ldm.data.phone_set import MIDI_PAD_ID
from ldm.infer.timing import FPS

AUDIO_EXT = ('.wav', '.flac', '.ogg', '.mp3', '.m4a')
# Shares above which check_f0 warns. With the clips' own F0, 3 of 32 validation clips are warned about; with the
# F0 moved by 50, 100 and 200 ms it is 19%, 38% and 78% (scripts/checks/check_frontend.py --part shift). So
# the warnings catch a curve that is clearly off the score's time axis, not a small shift.
OFF_NOTES, VOICED_IN_RESTS, UNVOICED_IN_VOWELS = 0.3, 0.3, 0.15
TEXT_EXT = ('.txt', '.csv', '.tsv')


def resample_f0(f0, src_fps, n_out=None, dst_fps=FPS):
    """F0 at another frame rate. Voiced stretches are interpolated, unvoiced gaps stay gaps."""
    f0 = np.asarray(f0, dtype=np.float64)
    n_out = int(round(len(f0) / src_fps * dst_fps)) if n_out is None else n_out
    return curve_at(np.arange(len(f0)) / src_fps, f0, np.arange(n_out) / dst_fps)


def curve_at(times, f0, at):
    """F0 given at `times` (seconds), read at the times `at`: log-linear between voiced points, unvoiced where
    the nearest point is unvoiced, and unvoiced outside the given span."""
    times, f0 = np.asarray(times, dtype=np.float64), np.asarray(f0, dtype=np.float64)
    out = np.zeros(len(at), dtype=np.float32)
    voiced = f0 > 0
    if not voiced.any() or len(at) == 0:
        return out
    logf = np.interp(at, times[voiced], np.log(f0[voiced]))
    right = np.clip(np.searchsorted(times, at), 1, len(times) - 1)
    nearest = np.where(np.abs(at - times[right - 1]) <= np.abs(times[right] - at), right - 1, right) if len(times) > 1 \
        else np.zeros(len(at), dtype=int)
    step = np.median(np.diff(times)) if len(times) > 1 else 1.0 / FPS
    inside = (at >= times[0] - step / 2) & (at <= times[-1] + step / 2)
    keep = voiced[nearest] & inside
    out[keep] = np.exp(logf[keep])
    return out


def track_f0(wav, sr, rmvpe):
    """F0 of a recording at the model's frame rate, as preprocess/extract_f0.py tracks the training clips."""
    import librosa
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 48000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=48000)
    f0, _ = rmvpe.get_pitch(wav, sample_rate=48000, hop_size=256, length=len(wav) // 256)
    return f0.astype(np.float32)


def from_values(values, fps=None, where='the F0'):
    """F0 in Hz at 187.5 frames/s from one value per frame at `fps` (None = 187.5), checked for plausible Hz."""
    f0 = np.nan_to_num(np.asarray(values, dtype=np.float64))
    if fps and abs(fps - FPS) > 1e-6:
        f0 = resample_f0(f0, fps)
    f0 = np.where(f0 > 0, f0, 0).astype(np.float32)
    if (f0 > 0).any() and (np.median(f0[f0 > 0]) < 40 or f0.max() > 2000):
        raise ValueError(f'{where}: values between {f0[f0 > 0].min():.1f} and {f0.max():.1f}; expected Hz (0 = unvoiced)')
    return f0


def load_f0(path, fps=None, rmvpe=None):
    """F0 in Hz at 187.5 frames/s from a file (see the module text). `rmvpe` is needed for audio."""
    from ldm.infer.score import ScoreError, parse_f0
    ext = os.path.splitext(path)[1].lower()
    if ext in AUDIO_EXT:
        import soundfile as sf
        if rmvpe is None:
            from utils.rmvpe import RMVPE
            rmvpe = RMVPE('useful_ckpts/rmvpe/model.pt')
        wav, sr = sf.read(path, dtype='float32', always_2d=True)
        return track_f0(wav.mean(axis=1), sr, rmvpe)
    if ext == '.json':
        import json
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            if data.get('f0') is None:
                raise ValueError(f'F0 file {path}: no "f0" in it')
            data, fps = data['f0'], data.get('f0_fps', fps)
        try:
            values, fps = parse_f0(data, fps, f'F0 file {path}: "f0"')
        except ScoreError as e:
            raise ValueError(str(e)) from None
        return from_values(values, fps, f'F0 file {path}')
    if ext in TEXT_EXT:
        with open(path, encoding='utf-8') as f:
            rows = [line.replace(',', ' ').replace(';', ' ').split() for line in f if line.strip() and not line.lstrip().startswith('#')]
        try:
            data = np.array([[float(x) for x in r] for r in rows])
        except ValueError:  # a header line
            data = np.array([[float(x) for x in r] for r in rows[1:]])
    else:
        raise ValueError(f'F0 file {path}: unknown type {ext!r}; use .json, text {TEXT_EXT} or audio {AUDIO_EXT}')
    data = np.squeeze(np.asarray(data, dtype=np.float64))
    if data.ndim == 2 and data.shape[1] == 2:
        times, f0 = data[:, 0], np.nan_to_num(data[:, 1])
        if np.any(np.diff(times) <= 0):
            raise ValueError(f'F0 file {path}: the times in the first column must increase')
        f0 = curve_at(times, f0, np.arange(int(round(times[-1] * FPS)) + 1) / FPS)
    elif data.ndim == 1:
        f0 = data
    else:
        raise ValueError(f'F0 file {path}: expected one column of Hz or two columns (seconds, Hz), found shape {data.shape}')
    return from_values(f0, fps if data.ndim == 1 else None, f'F0 file {path}')


def fit_length(f0, n_frames):
    """(F0 cut or padded with unvoiced frames to n_frames, warning or None)."""
    diff = (len(f0) - n_frames) / FPS
    out = np.zeros(n_frames, dtype=np.float32)
    out[:min(len(f0), n_frames)] = f0[:n_frames]
    if abs(diff) > 0.1:
        return out, (f'the F0 lasts {len(f0) / FPS:.2f} s and the score {n_frames / FPS:.2f} s; the F0 was '
                     f'{"cut" if diff > 0 else "filled with unvoiced frames"} at its end. Both must start together.')
    return out, None


def check_f0(f0, midi, sung, nucleus):
    """How a given F0 agrees with the score. midi: note pitch per frame; sung / nucleus: frames inside sung
    words / inside vowels. Returns (figures, warnings)."""
    f0, midi = np.asarray(f0), np.asarray(midi)
    voiced = f0 > 0
    note = sung & (midi > 0) & (midi < MIDI_PAD_ID)
    both = voiced & note
    semis = 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440.0) - midi
    stats = dict(voiced_share=float(voiced.mean()),
                 voiced_in_rests=float(voiced[~sung].mean()) if (~sung).any() else 0.0,
                 unvoiced_in_vowels=float((~voiced)[nucleus].mean()) if nucleus.any() else 0.0,
                 off_notes_2_semitones=float((np.abs(semis[both]) > 2).mean()) if both.any() else 0.0,
                 median_offset_semitones=float(np.median(semis[both])) if both.any() else 0.0)
    warnings = []
    if not voiced.any():
        warnings.append('the F0 has no voiced frame')
        return stats, warnings
    off = stats['median_offset_semitones']
    if abs(off) >= 9:
        warnings.append(f'the F0 lies {abs(off):.1f} semitones {"above" if off > 0 else "below"} the notes on average'
                        + (' (an octave)' if abs(abs(off) - 12) <= 2 else '') + '; the output follows the F0, not the notes')
    elif stats['off_notes_2_semitones'] > OFF_NOTES:
        warnings.append(f"{stats['off_notes_2_semitones']:.0%} of the voiced frames are more than 2 semitones from their note "
                        f'(median offset {off:+.1f}); F0 and score may not be the same melody or not on the same time axis')
    if stats['voiced_in_rests'] > VOICED_IN_RESTS:
        warnings.append(f"{stats['voiced_in_rests']:.0%} of the frames in rests are voiced in the F0")
    if stats['unvoiced_in_vowels'] > UNVOICED_IN_VOWELS:
        warnings.append(f"{stats['unvoiced_in_vowels']:.0%} of the frames in vowels are unvoiced in the F0")
    return stats, warnings
