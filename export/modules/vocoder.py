"""The HiFi-GAN NSF vocoder for export (EWP1).

NSFVocoder(generator) computes HifiGanGenerator.forward(mel, f0) (export.modules.load.load_vocoder) with the
randomness of its NSF source given from outside: SineGen draws the initial phases of the harmonics
(rand [B, 9], the fundamental's set to 0) and the added noise (randn [B, L, 9]); SourceModuleHnNSF then draws a
noise branch that the generator does not use. In the export they are constants (baked_nsf). Without
upsample (f0 repeated per sample), leaky_relu (common.leaky_relu) and in-place indexing; the phase is the
original's: rad % 1, the overflow shift of the first cumsum, sin(cumsum * 2 pi).
"""
import sys

import numpy as np
import torch
from torch import nn

from export.modules.common import leaky_relu

HARMONICS = 9  # SineGen.dim: the fundamental and 8 overtones (HifiGanGenerator.harmonic_num)


def nsf_draws(seed, n_samples, prompt_shape=(1, 20, 94), generator=None):
    """(rand_ini [1, 9], sine_noise [1, n_samples, 9]) as Singer.sing draws them: torch.manual_seed(seed), the
    blank prompt's posterior sample (randn of prompt_shape; None = not drawn), then SineGen's rand and randn."""
    if generator is None:
        torch.manual_seed(seed)
    kw = {} if generator is None else dict(generator=generator)
    if prompt_shape is not None:
        torch.randn(prompt_shape, **kw)
    rand_ini = torch.rand(1, HARMONICS, **kw)
    rand_ini[:, 0] = 0
    return rand_ini, torch.randn(1, n_samples, HARMONICS, **kw)


def baked_nsf(n_samples=962560, seed=0):
    """The NSF constants of the export: drawn from torch.Generator(seed), without a prompt draw."""
    return nsf_draws(seed, n_samples, prompt_shape=None, generator=torch.Generator().manual_seed(seed))


class NSFVocoder(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.g = generator
        assert generator.h['resblock'] == '1' and generator.harmonic_num + 1 == HARMONICS
        self.lrelu_slope = sys.modules[type(generator).__module__].LRELU_SLOPE
        self.hop = int(np.prod(generator.h['upsample_rates']))
        sg = generator.m_source.l_sin_gen
        self.sr, self.sine_amp, self.noise_std, self.voiced_threshold = sg.sampling_rate, sg.sine_amp, sg.noise_std, sg.voiced_threshold
        self.register_buffer('harmonics', torch.arange(1, HARMONICS + 1, dtype=torch.float32), persistent=False)

    def source(self, f0, rand_ini, sine_noise):
        """SourceModuleHnNSF.forward's harmonic source [B, 1, L] for f0 [B, L, 1] Hz per sample."""
        rad = (f0 * self.harmonics / self.sr) % 1                          # [B, L, 9]
        rad = torch.cat([rad[:, :1] + rand_ini[:, None, :], rad[:, 1:]], dim=1)
        over = torch.cumsum(rad, 1) % 1
        shift = ((over[:, 1:] - over[:, :-1]) < 0).to(rad.dtype) * -1.0
        shift = torch.cat([torch.zeros_like(shift[:, :1]), shift], dim=1)
        sines = torch.sin(torch.cumsum(rad + shift, dim=1) * 2 * np.pi) * self.sine_amp
        uv = (f0 > self.voiced_threshold).to(f0.dtype)
        noise = (uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * sine_noise
        sines = sines * uv + noise
        m = self.g.m_source
        return m.l_tanh(m.l_linear(sines)).transpose(1, 2)

    def resblock(self, blk, x):
        """hifigan_nsf.ResBlock1.forward."""
        for c1, c2 in zip(blk.convs1, blk.convs2):
            xt = c2(leaky_relu(c1(leaky_relu(x, self.lrelu_slope)), self.lrelu_slope))
            x = xt + x
        return x

    def forward(self, mel, f0, rand_ini, sine_noise):
        """mel [B, 80, T]; f0 [B, T] Hz per frame (0 = unvoiced); rand_ini [B, 9]; sine_noise [B, T * 256, 9].
        Returns wav [B, 1, T * 256]."""
        g = self.g
        har = self.source(f0.repeat_interleave(self.hop, dim=1)[:, :, None], rand_ini, sine_noise)
        x = g.conv_pre(mel)
        for i in range(g.num_upsamples):
            x = g.ups[i](leaky_relu(x, self.lrelu_slope))
            x = x + g.noise_convs[i](har)
            xs = None
            for j in range(g.num_kernels):
                y = self.resblock(g.resblocks[i * g.num_kernels + j], x)
                xs = y if xs is None else xs + y
            x = xs / g.num_kernels
        return torch.tanh(g.conv_post(leaky_relu(x)))
