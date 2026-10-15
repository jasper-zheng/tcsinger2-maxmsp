"""How an F0 curve sings its notes (WP10): figures that compare generated, predicted and sung curves.

Per note (sung notes of at least 8 frames, boundaries from the onset track of ldm.f0gen.features):
  off          median voiced pitch minus the note, semitones (nan with fewer than 3 voiced frames)
  short        the note lasts under 0.4 s
  vib_extent   on notes of 0.6 s or more with 0.4 s voiced after the first 0.1 s: peak-to-peak extent (cents)
               of the 3.5-8.5 Hz part of the detrended curve (2 sqrt 2 x its rms)
  vib_rate     the frequency of its spectral peak (Hz)
  vib          vibrato present: extent of at least 30 cents and at least 40% of the 1-15 Hz energy in 3.5-8.5 Hz
  onset_ms     notes after a sung note at least a semitone away: time from the onset until the curve stays
               within half a semitone of the note for 3 frames (the note length if never)
Per clip:
  ffe / voicing_err / cents   against the sung curve (ldm.stage2_samples.f0_metrics)
  r_std        spread of the voiced pitch around the notes inside sung notes (cents, |offset| < 7 semitones)
  jitter       median frame-to-frame step of the voiced pitch (cents)
summarise() turns the rows of many clips into the figures reported in LOGS.md.
"""
import numpy as np

from ldm.data.phone_set import MIDI_PAD_ID
from ldm.f0gen.features import semitone
from ldm.stage2_samples import f0_metrics

FPS = 187.5


def note_spans(midi, onset):
    """(start, end, pitch) of the sung notes: a new note at every onset and every change of MIDI pitch."""
    spans, start = [], 0
    n = len(midi)
    for t in range(1, n + 1):
        if t == n or onset[t] > 0 or midi[t] != midi[start]:
            if 0 < midi[start] < MIDI_PAD_ID:
                spans.append((start, t, int(midi[start])))
            start = t
    return spans


def vibrato(cents):
    """(extent in cents, rate in Hz, share of 1-15 Hz energy in 3.5-8.5 Hz) of a voiced stretch in cents."""
    t = np.arange(len(cents))
    x = cents - np.polyval(np.polyfit(t, cents, 2), t)
    n = 4096
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n)) ** 2
    freq = np.fft.rfftfreq(n, 1 / FPS)
    band, wide = (freq >= 3.5) & (freq <= 8.5), (freq >= 1) & (freq <= 15)
    share = spec[band].sum() / max(spec[wide].sum(), 1e-9)
    xf = np.fft.irfft(np.where(band, np.fft.rfft(x, n), 0), n)[:len(x)]
    return 2 * np.sqrt(2) * float(np.sqrt(np.mean(xf ** 2))), float(freq[band][np.argmax(spec[band])]), float(share)


def curve_rows(f0, midi, onset, f0_sung=None):
    """(clip figures, list of note figures) of an F0 curve [T] with the clip's MIDI and onset tracks [T]."""
    f0, midi, onset = np.asarray(f0, dtype=np.float64), np.asarray(midi), np.asarray(onset)
    n = min(len(f0), len(midi))
    f0, midi, onset = f0[:n], midi[:n], onset[:n]
    semi = np.where(f0 > 0, semitone(f0), np.nan)
    notes, prev = [], None
    for a, b, pitch in note_spans(midi, onset):
        seg = semi[a:b]
        voiced = seg[~np.isnan(seg)]
        row = dict(seconds=(b - a) / FPS, short=(b - a) / FPS < 0.4,
                   off=float(np.median(voiced) - pitch) if len(voiced) >= 3 else np.nan)
        if prev is not None and prev[1] == a and abs(prev[2] - pitch) >= 1:
            close = np.abs(seg - pitch) < 0.5
            run = np.convolve(close.astype(int), np.ones(3, dtype=int), 'valid') == 3
            row['onset_ms'] = 1000 * (int(np.argmax(run)) if run.any() else b - a) / FPS
        skip = int(0.1 * FPS)
        if (b - a) / FPS >= 0.6:
            stretch = seg[skip:]
            # the longest voiced run after the first 0.1 s
            best, cur = (0, 0), None
            for i, ok in enumerate(np.append(~np.isnan(stretch), False)):
                if ok and cur is None:
                    cur = i
                elif not ok and cur is not None:
                    best = max(best, (i - cur, cur))
                    cur = None
            if best[0] >= 0.4 * FPS:
                ext, rate, share = vibrato(100 * (stretch[best[1]:best[1] + best[0]] - pitch))
                row.update(vib_extent=ext, vib_rate=rate, vib=bool(ext >= 30 and share >= 0.4))
        notes.append(row)
        prev = (a, b, pitch)
    sung = (midi > 0) & (midi < MIDI_PAD_ID) & ~np.isnan(semi)
    r = 100 * (semi[sung] - midi[sung]) if sung.any() else np.zeros(1)
    r = r[np.abs(r) < 700]
    both = ~np.isnan(semi[1:]) & ~np.isnan(semi[:-1])
    clip = dict(r_std=float(np.std(r)) if len(r) > 10 else np.nan, r_median=float(np.median(r)) if len(r) else np.nan,
                jitter=float(np.median(100 * np.abs(np.diff(semi)[both]))) if both.sum() > 10 else np.nan)
    if f0_sung is not None:
        clip['ffe'], clip['voicing_err'], clip['cents'] = f0_metrics(np.asarray(f0_sung)[:n], f0)
    return clip, notes


def summarise(clips, notes):
    """Figures of many clips: means of the clip figures and pooled note figures."""
    import pandas as pd
    c, nt = pd.DataFrame(clips), pd.DataFrame(notes)
    out = {k: float(c[k].mean()) for k in ('ffe', 'voicing_err', 'cents', 'r_std', 'r_median', 'jitter') if k in c}
    hit = lambda d: float((d['off'].abs() <= 1).mean()) if len(d) else np.nan  # unvoiced notes (nan) count as misses
    out['notes_within_1'] = hit(nt)
    out['short_notes_within_1'] = hit(nt[nt['short']])
    out['notes_flat_cents'] = float(100 * nt['off'].median())
    if 'vib' in nt:
        long = nt[nt['vib'].notna()]
        out['long_notes'] = len(long)
        out['vibrato_share'] = float(long['vib'].astype(bool).mean())
        out['vib_extent_median'] = float(long['vib_extent'].median())
        vib = long[long['vib'].astype(bool)]
        out['vib_extent_when_vibrato'] = float(vib['vib_extent'].median()) if len(vib) else np.nan
        out['vib_rate_when_vibrato'] = float(vib['vib_rate'].median()) if len(vib) else np.nan
    if 'onset_ms' in nt:
        out['onset_ms_median'] = float(nt['onset_ms'].median())
    out['notes'] = len(nt)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}
