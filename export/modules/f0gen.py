"""The F0 generator's sampler for export (EWP1): F0Gen.generate of ldm/f0gen/module.py without an F0 prompt,
at cfg 1, unrolled over `steps` Euler steps of linspace(0, 1, steps + 1), with its noise as an input.

F0Net.forward is written out over the network's own modules, with two changes for the MLX delegate, which
has no pow(Scalar, Tensor): the RoPE frequencies (base ** (arange / D)) are computed once with the same ops as
model.rope (a buffer), and the decode's 2 ** x is pow(Tensor, Tensor); both give the same values.
Without a prompt the style vector is the language embedding plus the prompt encoder's null vector, exactly as
F0Net.style gives it for an empty prompt mask. The frame tracks are those of ldm.f0gen.features
(tracks_from_pieces, reference_pitch), padded as ldm.f0gen.dataset.collate pads them; `mask` is 1 on real frames.

conv_mask (EWP5): PyTorch runs the generator at its own length, n frames padded to nf, a multiple of 4 (collate).
At a longer fixed length the key mask hides the padding from the attention, but the convolutions (pre, each
block's conv, up, post) still read the frames from nf on, where the own run has their zero padding. With
conv_mask those inputs are zeroed from nf on (from nf / 2 at the blocks' half rate), so the frames before nf
are computed as at the own length; down already takes h * mask, and the noise after n never moves (the
velocity is masked). Off by default: the EWP1 and EWP2 checks run without it.
"""
import torch
import torch.nn.functional as F
from torch import nn

from export.modules.common import rotate_pairs
from ldm.f0gen.model import timestep_embedding


class F0Sampler(nn.Module):
    def __init__(self, f0net, steps=16, base=10000.0, conv_mask=False):
        super().__init__()
        self.net, self.steps, self.conv_mask = f0net, steps, conv_mask
        span = torch.linspace(0, 1, steps + 1)
        self.times = [float(t0) for t0 in span[:-1]]
        for i, (t0, t1) in enumerate(zip(span[:-1], span[1:])):
            self.register_buffer(f'dt{i}', (t1 - t0).reshape(()).clone(), persistent=False)
        D = f0net.dim // f0net.blocks[0].heads
        # model.rope's frequencies; the angles and their cos/sin are computed per call as rope does (cos of a
        # longer table, then cut, rounds differently in the vectorised tail)
        self.register_buffer('rope_freqs', 1.0 / base ** (torch.arange(0, D, 2, dtype=torch.float32) / D), persistent=False)

    def block(self, blk, x, c, key_mask, cos, sin, token_keep):
        """ldm.f0gen.model.Block.forward (eval); token_keep [B, T/2, 1] multiplies the conv's input."""
        shift1, scale1, gate1, shift2, scale2, gate2 = blk.ada(c)[:, None].chunk(6, dim=-1)
        h = blk.norm1(x) * (1 + scale1) + shift1
        B, T, D = h.shape
        q, k, v = blk.qkv(h).view(B, T, 3, blk.heads, D // blk.heads).permute(2, 0, 3, 1, 4)
        q, k = blk.q_norm(q), blk.k_norm(k)
        h = F.scaled_dot_product_attention(rotate_pairs(q, cos, sin), rotate_pairs(k, cos, sin), v,
                                           attn_mask=key_mask[:, None, None, :])
        x = x + gate1 * blk.proj(h.transpose(1, 2).reshape(B, T, D))
        h = blk.norm2(x) * (1 + scale2) + shift2
        h = h + blk.conv((h * token_keep).transpose(1, 2)).transpose(1, 2)
        return x + gate2 * blk.mlp(h)

    @staticmethod
    def conv_seq(seq, h, keep):
        """nn.Sequential(conv, act, conv) with each conv's input multiplied by keep."""
        return seq[2](seq[1](seq[0](h * keep)) * keep)

    def velocity(self, x, t, tracks, mask, key_mask, style, cos, sin, keep):
        """ldm.f0gen.model.F0Net.forward(x, t, batch, style); keep = (frame_keep [B, 1, T], token_keep [B, T/2, 1])
        multiply the conv inputs (ones without conv_mask)."""
        net = self.net
        frame_keep, token_keep = keep
        ph, midi, ntype, pos, onset = tracks
        h = net.ph(ph) + net.midi(midi) + net.ntype(ntype)
        h = h.transpose(1, 2) + net.frame(torch.cat([pos, onset[:, None], x], dim=1))
        h = h + self.conv_seq(net.pre, h, frame_keep)
        full = h
        h = net.down(h * mask[:, None]).transpose(1, 2)
        c = net.time(timestep_embedding(t, 256)) + style
        for blk in net.blocks:
            h = self.block(blk, h, c, key_mask, cos, sin, token_keep)
        shift, scale = net.final_ada(c)[:, None].chunk(2, dim=-1)
        h = net.norm(h) * (1 + scale) + shift
        h = net.up((h * token_keep).transpose(1, 2))[..., :full.shape[-1]] + full
        return self.conv_seq(net.post, h, frame_keep) * mask[:, None]

    def keep(self, mask):
        """The conv-input masks: frames < nf and tokens < nf / 2, nf = the mask's length rounded up to a
        multiple of 4 (with remainder: the MLX delegate drops the floor of an integer floor division)."""
        T = mask.shape[-1]
        if not self.conv_mask:
            return torch.ones_like(mask)[:, None], torch.ones_like(mask[:, ::2])[..., None]
        n = mask.sum(-1, keepdim=True)
        nf = n + torch.remainder(4.0 - torch.remainder(n, 4.0), 4.0)
        frames = torch.arange(T, device=mask.device, dtype=mask.dtype)[None]
        frame_keep = (frames < nf).to(mask.dtype)
        return frame_keep[:, None], (frames[:, ::2] < nf).to(mask.dtype)[..., None]

    def sample(self, noise, ph, midi, ntype, pos, onset, mask, language):
        """The Euler loop: the final (r, v) [1, 2, T] from the noise [1, 2, T]."""
        style = self.net.lang(language) + self.net.prompt.null[None]
        key_mask = mask[:, ::2] > 0
        ang = torch.arange(key_mask.shape[1], device=noise.device, dtype=torch.float32)[:, None] * self.rope_freqs[None]
        cos, sin = ang.cos().to(noise.dtype), ang.sin().to(noise.dtype)
        tracks = (ph, midi, ntype, pos, onset)
        keep = self.keep(mask)
        x = noise
        for i, t0 in enumerate(self.times):
            t = torch.full((x.shape[0],), t0, device=x.device)
            x = x + getattr(self, f'dt{i}') * self.velocity(x, t, tracks, mask, key_mask, style, cos, sin, keep)
        return x

    def forward(self, noise, ph, midi, ntype, pos, onset, ref, mask, language):
        """noise [1, 2, T]; ph, midi, ntype [1, T] int64; pos [1, 3, T]; onset, ref, mask [1, T]; language [1].
        Returns F0 in Hz [1, T], 0 where unvoiced or masked."""
        x = self.sample(noise, ph, midi, ntype, pos, onset, mask, language)
        r, v = x[:, 0], x[:, 1]
        semis = (ref + r - 69) / 12
        f0 = torch.where(v > 0, 440.0 * torch.pow(torch.full_like(semis, 2.0), semis), torch.zeros_like(r))  # features.decode
        return f0 * mask
