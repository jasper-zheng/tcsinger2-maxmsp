"""The VAE decoder for export (EWP1): stage 2's latent to the mel, as CFM.decode_first_stage does it.

VAEDecoder(vae, scale_factor) runs AutoencoderKL.decode(1 / scale_factor * z) with the weights of `vae`
(export.modules.load.load_vae), without group_norm (common.group_norm) and interpolate (nearest upsampling as
repeat_interleave). With `valid` (EWP2), GroupNorm takes its statistics and the attention blocks their keys
from the latent's first `valid` frames only (twice as many after each upsampling), so a phrase padded to the
export's fixed length decodes as it does at its own length; without it, the output is the original's.
blank_prompt_latent() is the latent of the blank audio prompt of version 1: the posterior
mean, a constant of `sing` (the VAE encoder is not exported).
"""
import os

import numpy as np
import torch
from torch import nn

from export.modules.common import group_norm, group_norm_masked, upsample_nearest

BLANK_PROMPT_PATH = 'export/artifacts/constants/blank_prompt_mean.pt'


def nonlinearity(x):
    """autoencoder1d.nonlinearity (swish)."""
    return x * torch.sigmoid(x)


def _norm(x, norm, valid):
    return group_norm(x, norm) if valid is None else group_norm_masked(x, norm, valid)


def resnet_block(blk, x, valid=None):
    """autoencoder1d.ResnetBlock1D.forward without a time embedding (dropout is off in eval)."""
    h = blk.conv1(nonlinearity(_norm(x, blk.norm1, valid)))
    h = blk.conv2(nonlinearity(_norm(h, blk.norm2, valid)))
    if blk.in_channels != blk.out_channels:
        x = blk.conv_shortcut(x) if blk.use_conv_shortcut else blk.nin_shortcut(x)
    return x + h


def attn_block(blk, x, valid=None):
    """autoencoder1d.AttnBlock1D.forward (scaled by channels ** -0.5, as the original computes it)."""
    h = _norm(x, blk.norm, valid)
    q, k, v = blk.q(h), blk.k(h), blk.v(h)
    b, c, t = q.shape
    w = torch.bmm(q.permute(0, 2, 1), k) * (int(c) ** (-0.5))
    if valid is not None:
        keys = (torch.arange(t, device=x.device) < valid.view(-1, 1))[:, None, :]
        w = torch.where(keys, w, torch.full_like(w, float('-inf')))
    w = torch.softmax(w, dim=2)
    return x + blk.proj_out(torch.bmm(v, w.permute(0, 2, 1)))


class VAEDecoder(nn.Module):
    def __init__(self, vae, scale_factor):
        super().__init__()
        self.post_quant_conv, self.decoder = vae.post_quant_conv, vae.decoder
        self.register_buffer('inv_scale', 1. / torch.tensor(scale_factor, dtype=torch.float32), persistent=False)

    def forward(self, z, valid=None):
        """z [B, 20, L] stage-2 latent -> mel [B, 80, 4 L]; valid [B] or [1] int64: the latent frames that
        GroupNorm and attention see (None: all)."""
        return self.decode_scaled(self.inv_scale * z, valid)

    def decode_scaled(self, z, valid=None):
        """AutoencoderKL.decode: z already divided by the scale factor."""
        d = self.decoder
        h = d.conv_in(self.post_quant_conv(z))
        h = resnet_block(d.mid.block_1, h, valid)
        h = attn_block(d.mid.attn_1, h, valid)
        h = resnet_block(d.mid.block_2, h, valid)
        for i_level in reversed(range(d.num_layers)):
            up = d.up[i_level]
            for i_block in range(d.num_res_blocks + 1):
                h = resnet_block(up.block[i_block], h, valid)
                if len(up.attn) > 0:
                    h = attn_block(up.attn[i_block], h, valid)
            if i_level in d.up_layers and hasattr(up, 'upsample'):
                h = upsample_nearest(h, 2)
                valid = None if valid is None else valid * 2
                if up.upsample.with_conv:
                    h = up.upsample.conv(h)
        h = d.conv_out(nonlinearity(_norm(h, d.norm_out, valid)))
        return torch.tanh(h) if d.tanh_out else h


def blank_prompt_latent(path=BLANK_PROMPT_PATH, scale_factor=None):
    """The stage-2 latent [1, 20, 94] of the blank prompt (pipeline.blank_prompt()): scale_factor times the VAE
    posterior mean. Computed once with the VAE encoder and kept at `path`."""
    from export.modules.load import ROOT, load_dit, load_vae
    full = os.path.join(ROOT, path)
    if os.path.exists(full):
        return torch.load(full, map_location='cpu')
    from ldm.infer.pipeline import blank_prompt
    if scale_factor is None:
        _, scale_factor, _ = load_dit()
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):  # the VAE constructors print their layer lists
        vae = load_vae(encoder=True)
    with torch.no_grad():
        post = vae.encode(torch.from_numpy(np.ascontiguousarray(blank_prompt()))[None])
        z = torch.tensor(scale_factor, dtype=torch.float32) * post.mean
    os.makedirs(os.path.dirname(full), exist_ok=True)
    torch.save(z, full)
    return z
