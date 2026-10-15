"""The F0 generator's network (WP10).

Per frame: phoneme, MIDI pitch, note type, position in the note (ldm.f0gen.features) and, for the flow
objective, the noisy target (r, voicing) are summed into one sequence, convolved, halved in time by a strided
convolution and run through transformer blocks with adaptive layer norm (DiT style, gates initialised to
zero) and normalised queries and keys. The global condition of the blocks is the flow time, the language and the style vector of the prompt
F0 (convolutions and attention pooling; a learned vector when there is no prompt). A transposed convolution
restores the frame rate, the full-rate features are added back, and a convolution gives two channels per
frame: the velocity of (r, voicing) for the flow objective, or (r, voicing logit) for the regression.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from ldm.data.phone_set import NUM_LANG, NUM_MIDI, NUM_PH, NOTE_TYPE_PAD


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = 1000 * t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def rope(x, base=10000.0):
    """Rotary position embedding of x [B, H, T, D]."""
    T, D = x.shape[-2], x.shape[-1]
    freqs = 1.0 / base ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D)
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * freqs[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        # QK normalisation: without it the attention logits grew to 1e6 by step 10,000 and the first run
        # diverged at step 11,900 (LOGS.md, WP10)
        self.q_norm = nn.LayerNorm(dim // heads, eps=1e-6)
        self.k_norm = nn.LayerNorm(dim // heads, eps=1e-6)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(approximate='tanh'), nn.Linear(int(dim * mlp_ratio), dim))
        self.conv = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)  # local smoothing inside the block
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.dropout = dropout
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, key_mask):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c)[:, None].chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1) + shift1
        B, T, D = h.shape
        q, k, v = self.qkv(h).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        h = F.scaled_dot_product_attention(rope(q), rope(k), v, attn_mask=key_mask[:, None, None, :],
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + gate1 * self.proj(h.transpose(1, 2).reshape(B, T, D))
        h = self.norm2(x) * (1 + scale2) + shift2
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        return x + gate2 * self.mlp(h)


class PromptEncoder(nn.Module):
    """Style vector [B, dim] of a prompt F0 [B, 2, P] with mask [B, P]; the null vector where the mask is empty."""

    def __init__(self, dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(2, dim, 5, padding=2), nn.GELU(),
            nn.Conv1d(dim, dim, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(dim, dim, 5, padding=8, dilation=4), nn.GELU(),
            nn.Conv1d(dim, dim, 5, padding=16, dilation=8), nn.GELU())
        self.score = nn.Conv1d(dim, 1, 1)
        self.out = nn.Linear(dim, dim)
        self.null = nn.Parameter(torch.zeros(dim))

    def forward(self, prompt, mask):
        h = self.convs(prompt)
        w = self.score(h).squeeze(1).masked_fill(mask <= 0, -1e4).softmax(-1)
        vec = self.out(torch.einsum('bdt,bt->bd', h, w))
        return torch.where((mask.sum(-1) > 0)[:, None], vec, self.null.to(vec.dtype)[None])


class F0Net(nn.Module):
    def __init__(self, dim=384, depth=8, heads=6, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.ph = nn.Embedding(NUM_PH, dim)
        self.midi = nn.Embedding(NUM_MIDI, dim)
        self.ntype = nn.Embedding(NOTE_TYPE_PAD + 1, dim)
        self.frame = nn.Conv1d(3 + 1 + 2, dim, 1)  # position (3), onset, noisy target (2)
        self.pre = nn.Sequential(nn.Conv1d(dim, dim, 5, padding=2), nn.GELU(), nn.Conv1d(dim, dim, 5, padding=2))
        self.down = nn.Conv1d(dim, dim, 4, stride=2, padding=1)
        self.time = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.lang = nn.Embedding(NUM_LANG, dim)
        self.prompt = PromptEncoder(dim)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.up = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)
        self.post = nn.Sequential(nn.Conv1d(dim, dim, 5, padding=2), nn.GELU(), nn.Conv1d(dim, 2, 5, padding=2))
        nn.init.zeros_(self.final_ada[1].weight)
        nn.init.zeros_(self.final_ada[1].bias)

    def style(self, batch):
        """The global condition without the time: language + prompt style vector."""
        return self.lang(batch['language']) + self.prompt(batch['prompt'], batch['prompt_mask'])

    def forward(self, x, t, batch, style=None):
        """x: noisy (r, voicing) [B, 2, T] (zeros for the regression); t [B] in [0, 1].
        Returns [B, 2, T]. `style` overrides self.style(batch) (classifier-free guidance)."""
        mask = batch['mask']                                                     # [B, T]
        h = self.ph(batch['ph']) + self.midi(batch['midi']) + self.ntype(batch['ntype'])
        frame = torch.cat([batch['pos'], batch['onset'][:, None], x], dim=1)
        h = h.transpose(1, 2) + self.frame(frame)                               # [B, D, T]
        h = h + self.pre(h)
        full = h
        h = self.down(h * mask[:, None]).transpose(1, 2)                        # [B, T/2, D]
        key_mask = mask[:, ::2] > 0
        c = self.time(timestep_embedding(t, 256)) + (self.style(batch) if style is None else style)
        for block in self.blocks:
            h = block(h, c, key_mask)
        shift, scale = self.final_ada(c)[:, None].chunk(2, dim=-1)
        h = self.norm(h) * (1 + scale) + shift
        h = self.up(h.transpose(1, 2))[..., :full.shape[-1]] + full
        return self.post(h) * mask[:, None]
