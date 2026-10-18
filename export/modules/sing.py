"""The `sing` method of tcsinger2_sing.pte (EWP4): the pieces conditions, the caption's tokens, the F0 curve and
the latent noise to the audio, as "The interface" of EXPORTING_WORK_PACKAGES.md pins it.

    front-end (frames, n) -> T5 on the caption (with its </s>) -> the 50 Hz MIDI curve to Hz at 187.5 fps -> stage 2 (Euler
    loop with CFG as a batch of 2, the self-attention masked to the phrase) -> VAE decoder (masked) ->
    vocoder (baked NSF noise) -> the audio zeroed after the phrase's last frame

SingPipeline(parts) builds it from export.modules.load.load_all(). stages(...) returns every intermediate (for the
checks), sing(...) the audio only. Constants: the T5 states of the empty caption (the unconditional branch), the
blank prompt's mean latent, the NSF draws of baked_nsf().
"""
import math

import torch
from torch import nn

from export.modules.frontend import N_FRAMES, FrontEnd
from export.modules.load import tokenize
from export.modules.stage2 import Stage2Given, Stage2Sampler, valid_latent_frames
from export.modules.t5 import CaptionEncoder
from export.modules.vae import VAEDecoder, blank_prompt_latent
from export.modules.vocoder import NSFVocoder, baked_nsf

F0_SAMPLES, F0_RATE, HOP = 1003, 50, 256
N_SAMPLES = N_FRAMES * HOP
PAD_ID, EOS_ID = 0, 1  # flan-t5: <pad>, </s>


def with_eos(input_ids):
    """The caption's ids as training tokenized them (T5Tokenizer, which ends the text with </s>): neural.tokenizer
    and gen_cli encode without special tokens, so when the ids hold no </s>, one goes at the first pad position
    (the last position if there is none, as truncation keeps it). Ids that end in </s> pass unchanged."""
    L = input_ids.shape[-1]
    count = (input_ids != PAD_ID).to(torch.long).sum(-1, keepdim=True)
    has_eos = (input_ids == EOS_ID).to(torch.long).sum(-1, keepdim=True) > 0
    at = torch.arange(L, device=input_ids.device)[None] == torch.clamp(count, max=L - 1)
    return torch.where(torch.logical_and(at, torch.logical_not(has_eos)), torch.full_like(input_ids, EOS_ID), input_ids)


def f0_from_50hz(f0_midi, n, n_frames=N_FRAMES):
    """The F0 buffer~ (MIDI pitch at 50 Hz, 0 = unvoiced) [1, 1, S] as Hz per 187.5 fps frame [1, n_frames], 0
    from frame n [1] on: f0_input.resample_f0 of the curve in Hz. Its log-linear interpolation between voiced
    samples is linear in MIDI pitch; a frame is voiced when its nearest sample is. Frame k lies at sample
    4k / 15 and its nearest sample is round(4k / 15), never a tie (a half is at least 1/30 away)."""
    m = f0_midi[0, 0]
    S = m.shape[0]
    dev = m.device
    voiced = m > 0
    j = torch.arange(S, device=dev)
    none_before = torch.full((S, S), -1, dtype=torch.long, device=dev)
    none_after = torch.full((S, S), S, dtype=torch.long, device=dev)
    prev = torch.where(torch.logical_and(voiced[None, :], j[None, :] <= j[:, None]), j[None, :], none_before).amax(-1)  # last voiced <= i
    nxt = torch.where(torch.logical_and(voiced[None, :], j[None, :] >= j[:, None]), j[None, :], none_after).amin(-1)    # first voiced >= i

    # positions in float32 (4k / 15 is correctly rounded, so whole samples stay whole): the MLX delegate drops
    # the floor of an integer div(rounding_mode='floor') whose result is cast to float
    k = torch.arange(n_frames, device=dev)
    u = 4.0 * k.to(m.dtype) / 15.0
    i0f = torch.floor(u)
    i0 = i0f.to(torch.long)
    lo = prev[i0]
    hi = torch.where(i0 + 1 < S, nxt[torch.clamp(i0 + 1, max=S - 1)], torch.full_like(i0, S))
    has_lo, has_hi = lo >= 0, hi < S
    m_lo, m_hi = m[torch.clamp(lo, min=0)], m[torch.clamp(hi, max=S - 1)]
    span = torch.where(torch.logical_and(has_lo, has_hi), hi - lo, torch.ones_like(hi)).to(m.dtype)
    between = m_lo + ((4.0 * k.to(m.dtype) - 15.0 * lo.to(m.dtype)) / 15.0) * ((m_hi - m_lo) / span)  # an exact numerator
    pitch = torch.where(torch.logical_and(has_lo, has_hi), between, torch.where(has_lo, m_lo, m_hi))
    nearest = torch.clamp(torch.floor(u + 0.5).to(torch.long), max=S - 1)  # u is never within 1/30 of a half
    keep = torch.logical_and(voiced[nearest], k < n.view(-1))
    hz = 440.0 * torch.exp((pitch - 69.0) * (math.log(2.0) / 12.0))
    return torch.where(keep, hz, torch.zeros_like(hz))[None]


class SingPipeline(nn.Module):
    def __init__(self, parts, steps=8):
        super().__init__()
        self.frontend = FrontEnd(parts.vae)
        self.caption = CaptionEncoder(parts.t5)
        self.sampler = Stage2Sampler(Stage2Given(parts.dit, blank_prompt_latent(scale_factor=parts.scale_factor)), steps, cfg_batch=True)
        self.vae = VAEDecoder(parts.vae, parts.scale_factor)
        self.vocoder = NSFVocoder(parts.vocoder)
        with torch.no_grad():
            self.register_buffer('empty_caption', self.caption(tokenize([''])).float().clone(), persistent=False)
        rand_ini, sine_noise = baked_nsf(N_SAMPLES)
        self.register_buffer('rand_ini', rand_ini, persistent=False)
        self.register_buffer('sine_noise', sine_noise, persistent=False)

    def stages(self, ph, midi, note_dur, note_type, ph_dur, nucleus, note_id, language, input_ids, cfg_scale, f0, latent_noise):
        """The inputs of "The interface" in their order: ph, midi, note_dur, note_type, ph_dur, nucleus, note_id
        [1, 256]; language [1, 1]; input_ids [1, 80]; cfg_scale [1]; f0 [1, 1, 1003] (MIDI at 50 Hz);
        latent_noise [1, 20, 940]. Returns a dict of the intermediates and `audio` [1, 1, 962560]."""
        fe = self.frontend(ph, midi, note_dur, note_type, ph_dur, nucleus, note_id)
        n = fe['n']
        valid = valid_latent_frames(n)
        caption = self.caption(with_eos(input_ids))
        f0_hz = f0_from_50hz(f0, n)
        latent = self.sampler(latent_noise, caption, self.empty_caption, fe['ph'], fe['midi'], f0_hz, language.view(-1), cfg_scale,
                              valid=valid)
        mel = self.vae(latent, valid)
        wav = self.vocoder(mel, f0_hz, self.rand_ini, self.sine_noise)
        keep = torch.arange(wav.shape[-1], device=wav.device) < n.view(-1, 1, 1) * HOP
        audio = torch.where(keep, wav, torch.zeros_like(wav))
        return dict(n=n, valid=valid, frame_ph=fe['ph'], frame_midi=fe['midi'], f0_hz=f0_hz, caption=caption, latent=latent, mel=mel, audio=audio)

    def sing(self, *inputs):
        return self.stages(*inputs)['audio']
