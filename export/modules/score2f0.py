"""The `score2f0` method of tcsinger2_f0.pte (EWP5): the pieces conditions and the F0 generator's noise to the F0
curve at 50 Hz, as "The interface" of EXPORTING_WORK_PACKAGES.md pins it.

    front-end (the F0 generator's tracks, n) -> F0 generator (f0_steps Euler steps, cfg 1, no prompt; its convs
    masked to the phrase's own padded length) -> MIDI pitch per 187.5 fps frame (ref + r, voiced where v > 0) ->
    the 1,003 samples of the F0 buffer~ at 50 Hz (MIDI pitch, 0 = unvoiced)

F0Pipeline(vae, f0net) builds it from export.modules.load (load_vae, load_f0net). stages(...) returns every
intermediate (for the checks), score2f0(...) the curve only.
"""
import numpy as np
import torch
from torch import nn

from export.modules.f0gen import F0Sampler
from export.modules.frontend import N_FRAMES, FrontEnd, _and
from export.modules.sing import F0_RATE, F0_SAMPLES
from ldm.infer.timing import FPS


def curve_tables(n_frames=N_FRAMES, n_samples=F0_SAMPLES, src_fps=FPS, dst_fps=F0_RATE):
    """The input-independent part of f0_input.resample_f0(f0, src_fps, n_samples, dst_fps) for n_frames frames, by
    curve_at's own float64 expressions: each sample's position in frames u, the frames that bound its
    interpolation (the last voiced frame <= floor u, the first >= ceil u) and its nearest frame (curve_at's tie
    rule on the half frames, j = 2 mod 4 at 187.5 -> 50)."""
    times = np.arange(n_frames) / src_fps
    at = np.arange(n_samples) / dst_fps
    right = np.clip(np.searchsorted(times, at), 1, n_frames - 1)
    nearest = np.where(np.abs(at - times[right - 1]) <= np.abs(times[right] - at), right - 1, right)
    step = np.median(np.diff(times))
    inside = (at >= times[0] - step / 2) & (at <= times[-1] + step / 2)
    assert inside.all(), 'a sample lies outside the frames: curve_at would leave it unvoiced'
    u = np.arange(n_samples) * (src_fps / dst_fps)
    assert np.all(u == np.round(u * 4) / 4), 'positions not exact in float32'
    return dict(u=u.astype(np.float32), lo_bound=np.floor(u).astype(np.int64), hi_bound=np.ceil(u).astype(np.int64),
                nearest=nearest.astype(np.int64))


class To50Hz(nn.Module):
    """MIDI pitch per frame [1, T] and its voicing [1, T] (bool) -> the 50 Hz curve [1, 1, S], MIDI pitch, 0 =
    unvoiced: f0_input.resample_f0 of the curve in Hz (its log-linear interpolation between voiced frames is
    linear in MIDI pitch; np.interp holds the end values; a sample is voiced when its nearest frame is)."""

    def __init__(self):
        super().__init__()
        for k, v in curve_tables().items():
            self.register_buffer(k, torch.as_tensor(v), persistent=False)

    def forward(self, midi, voiced):
        m, vo = midi[0], voiced[0]
        T = m.shape[0]
        k = torch.arange(T, device=m.device)
        S = self.u.shape[0]
        none_before = torch.full((S, T), -1, dtype=torch.long, device=m.device)
        none_after = torch.full((S, T), T, dtype=torch.long, device=m.device)
        lo = torch.where(_and(vo[None, :], k[None, :] <= self.lo_bound[:, None]), k[None, :], none_before).amax(-1)
        hi = torch.where(_and(vo[None, :], k[None, :] >= self.hi_bound[:, None]), k[None, :], none_after).amin(-1)
        has_lo, has_hi = lo >= 0, hi < T
        m_lo, m_hi = m[torch.clamp(lo, min=0)], m[torch.clamp(hi, max=T - 1)]
        both = _and(has_lo, has_hi)
        span = torch.where(both, hi - lo, torch.ones_like(hi)).to(m.dtype)  # 0 when the sample sits on a voiced frame
        span = torch.where(span > 0, span, torch.ones_like(span))
        between = m_lo + (self.u - lo.to(m.dtype)) * ((m_hi - m_lo) / span)
        pitch = torch.where(both, between, torch.where(has_lo, m_lo, m_hi))
        keep = vo[self.nearest]
        return torch.where(keep, pitch, torch.zeros_like(pitch))[None, None]


class F0Pipeline(nn.Module):
    def __init__(self, vae, f0net, f0_steps=16):
        super().__init__()
        self.frontend = FrontEnd(vae)
        self.sampler = F0Sampler(f0net, f0_steps, conv_mask=True)
        self.to_50hz = To50Hz()

    def stages(self, ph, midi, note_dur, note_type, ph_dur, nucleus, note_id, language, f0_noise):
        """The inputs of "The interface" in their order: ph, midi, note_dur, note_type, ph_dur, nucleus, note_id
        [1, 256]; language [1, 1]; f0_noise [1, 2, 3760]. Returns a dict of the intermediates and `f0` [1, 1, 1003]."""
        fe = self.frontend(ph, midi, note_dur, note_type, ph_dur, nucleus, note_id)
        x = self.sampler.sample(f0_noise, fe['ph'], fe['midi'], fe['ntype'], fe['pos'], fe['onset'], fe['mask'], language.view(-1))
        r, v = x[:, 0], x[:, 1]
        voiced = _and(v > 0, fe['mask'] > 0)
        pitch = fe['ref'] + r
        # Hz as F0Sampler.forward decodes it (features.decode), for the checks against Singer.generate_f0
        semis = (pitch - 69) / 12
        f0_hz = torch.where(voiced, 440.0 * torch.pow(torch.full_like(semis, 2.0), semis), torch.zeros_like(r))
        f0 = self.to_50hz(pitch, voiced)
        return dict(n=fe['n'], r=r, v=v, f0_hz=f0_hz, f0=f0)

    def score2f0(self, *inputs):
        return self.stages(*inputs)['f0']
