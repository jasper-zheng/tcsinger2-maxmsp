"""Stage 2 for export (EWP1): the TCSinger2 DiT in given-F0 mode and its Euler sampler with CFG.

Stage2Given(dit) computes what TCSinger2.forward(x, t, context) returns as its vector field when
context['f0_mode'] == 'given', with the weights of `dit` (export.modules.load.load_dit), but without:
  - complex numbers: RoPE through cos/sin tables (common.rotate_pairs);
  - scatter_: the hard MoE gates (eval mode: argmax, no Gumbel noise) as argmax == arange;
  - nn.MultiheadAttention: the MoE cross-attention written out, through SDPA;
  - leaky_relu: common.leaky_relu;
  - the all-ones attention masks, the load-balancing loss and the F0 head (unused with a given F0).
With `valid` (EWP2), the self-attention takes as keys only the prompt and the first `valid` latent frames, so a
phrase padded to the export's fixed length sings as it does at its own length; valid_latent_frames(n) gives the
count of the phrase's own padded length. Without it, the output is the original's.
Stage2Sampler runs the Euler loop of CFM.sample_cfg (`steps` points of linspace(0, 1, steps), so steps - 1
updates) with classifier-free guidance on the caption.
"""
import torch
import torch.nn.functional as F
from torch import nn

from export.modules.common import leaky_relu, rotate_pairs
from ldm.data.phone_set import MIDI_PAD_ID, PH_PAD_ID
from ldm.modules.diffusionmodules.flag_large_dit_moe import modulate

NUM_TIMESTEPS = 1000  # CFM.num_timesteps: the DiT reads the time as int(t * 1000)
MIN_FRAMES, FRAME_MULTIPLE = 376, 8  # tcsinger2_dataset.collate_items: the shortest padded batch, its multiple


def valid_latent_frames(n_frames):
    """The latent length of a phrase of n_frames mel frames padded as collate_items pads it (at least 376
    frames, a multiple of 8), in tensor ops: n_frames [B] int64 -> [B] int64."""
    size = torch.clamp(n_frames, min=MIN_FRAMES)
    # rounded up with remainder, so the division is exact: the MLX delegate drops the floor of an integer
    # floor division that another op follows (EWP4: (n + 7) // 8 * 2 came out as (n + 7) / 8 * 2)
    size = size + torch.remainder(FRAME_MULTIPLE - torch.remainder(size, FRAME_MULTIPLE), FRAME_MULTIPLE)
    return size // 4


def _conv_leaky_pool(seq, x):
    """nn.Sequential(Conv1d, LeakyReLU, AvgPool1d) of the DiT."""
    return seq[2](leaky_relu(seq[0](x), seq[1].negative_slope))


def _one_hot_argmax(logits):
    """MoE.gumbel_softmax(hard=True): 1 at the argmax of the last dimension."""
    idx = torch.arange(logits.shape[-1], device=logits.device)
    return (logits.argmax(dim=-1, keepdim=True) == idx).to(logits.dtype)


class Stage2Given(nn.Module):
    def __init__(self, dit, prompt_embed=None):
        super().__init__()
        self.dit = dit
        self.register_buffer('rope_cos', dit.freqs_cis.real.float().contiguous(), persistent=False)
        self.register_buffer('rope_sin', dit.freqs_cis.imag.float().contiguous(), persistent=False)
        if prompt_embed is not None:
            self.register_buffer('prompt_embed', prompt_embed.float().clone(), persistent=False)
        else:
            self.prompt_embed = None

    def attention(self, att, x, freqs, y, key_mask=None):
        """flag_large_dit_moe.Attention.forward with all-ones masks (q/k norms are identities here); key_mask
        [B, 1, 1, S] bool limits the self-attention's keys."""
        B, S, _ = x.shape
        H, D = att.n_local_heads, att.head_dim
        xq = att.q_norm(att.wq(x)).view(B, S, H, D)
        xk = att.k_norm(att.wk(x)).view(B, S, att.n_local_kv_heads, D)
        xv = att.wv(x).view(B, S, att.n_local_kv_heads, D)
        cos, sin = freqs
        xq, xk = rotate_pairs(xq, cos, sin), rotate_pairs(xk, cos, sin)
        q = xq.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, xk.permute(0, 2, 1, 3), xv.permute(0, 2, 1, 3), attn_mask=key_mask).permute(0, 2, 1, 3)
        yk = att.ky_norm(att.wk_y(y)).view(B, -1, att.n_local_kv_heads, D)
        yv = att.wv_y(y).view(B, -1, att.n_local_kv_heads, D)
        assert att.n_local_heads == att.n_local_kv_heads  # n_rep 1: the original's repeat is a no-op
        out_y = F.scaled_dot_product_attention(q, yk.permute(0, 2, 1, 3), yv.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        out = out + out_y * att.gate.tanh().view(1, 1, -1, 1)
        return att.wo(out.flatten(-2))

    @staticmethod
    def cross_attention(mha, x, y):
        """nn.MultiheadAttention(batch_first=True)(x, y, y)[0], written out."""
        B, S, C = x.shape
        H = mha.num_heads
        D = C // H
        wq, wk, wv = mha.in_proj_weight.chunk(3)
        bq, bk, bv = mha.in_proj_bias.chunk(3)
        q = F.linear(x, wq, bq).view(B, S, H, D).transpose(1, 2)
        k = F.linear(y, wk, bk).view(B, -1, H, D).transpose(1, 2)
        v = F.linear(y, wv, bv).view(B, -1, H, D).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, S, C)
        return mha.out_proj(out)

    def moe(self, moe, x, caption, language):
        """tcsinger2.MoE.forward in eval mode, without the load-balancing loss."""
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)
        lang_probs = _one_hot_argmax(moe.language_gating(moe.language_embedding(language)))
        lang_probs = lang_probs.repeat_interleave(T, dim=0)
        lang_out = torch.stack([e(x_flat) for e in moe.language_experts.values()], dim=1)
        first = (lang_out * lang_probs.unsqueeze(-1)).sum(dim=1)
        cross = self.cross_attention(moe.cross_attention, first.view(B, T, C), caption).reshape(-1, C)
        cap_probs = _one_hot_argmax(moe.caption_gating(cross))
        cap_out = torch.stack([e(first) for e in moe.caption_experts.values()], dim=1)
        second = (cap_out * cap_probs.unsqueeze(-1)).sum(dim=1)
        return (first + second).view(B, T, C)

    def block(self, blk, x, y, freqs, adaln, caption, language, key_mask=None):
        """tcsinger2.TransformerBlock.forward with adaLN."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = blk.adaLN_modulation(adaln).chunk(6, dim=1)
        h = x + gate_msa.unsqueeze(1) * self.attention(blk.attention, modulate(blk.attention_norm(x), shift_msa, scale_msa),
                                                       freqs, blk.attention_y_norm(y), key_mask)
        out = self.moe(blk.feed_forward, modulate(blk.ffn_norm(h), shift_mlp, scale_mlp), caption, language)
        return h + gate_mlp.unsqueeze(1) * out

    def forward(self, x, t, caption, ph, midi, f0_hz, language, prompt_embed=None, valid=None):
        """x [B, 20, L] latent; t [B] the timestep as the DiT reads it (0..999, float); caption [B, 80, 1024]
        T5 states; ph, midi [B, T] int64 frame tracks; f0_hz [B, T] (0 = unvoiced); language [B] int64;
        prompt_embed [B, 20, P] (default: the buffer); valid [B] or [1] int64: the latent frames the
        self-attention sees (None: all). Returns the vector field [B, 20, L]."""
        m = self.dit
        prompt = self.prompt_embed.expand(x.shape[0], -1, -1) if prompt_embed is None else prompt_embed
        B, Tp = x.shape[0], prompt.shape[2]
        f0_log = torch.log(f0_hz[:, None, :] + 1.0)

        prompt = m.prompt_proj(prompt).transpose(1, 2)
        prompt_global = prompt.mean(dim=1)
        n_pad = Tp * m.downsample_rate
        ph_pad = torch.full((B, n_pad), PH_PAD_ID, dtype=torch.long, device=x.device)
        midi_pad = torch.full((B, n_pad), MIDI_PAD_ID, dtype=torch.long, device=x.device)
        ph_pad_feat = _conv_leaky_pool(m.ph_proj, m.ph_embedding(ph_pad).transpose(1, 2)).transpose(1, 2)
        midi_pad_feat = _conv_leaky_pool(m.midi_proj, m.midi_embedding(midi_pad).transpose(1, 2)).transpose(1, 2)
        f0_pad_feat = _conv_leaky_pool(m.f0_proj, torch.zeros(B, 1, n_pad, device=x.device)).transpose(1, 2)
        midi_feat = _conv_leaky_pool(m.midi_proj, m.midi_embedding(midi).transpose(1, 2)).transpose(1, 2)
        ph_feat = _conv_leaky_pool(m.ph_proj, m.ph_embedding(ph).transpose(1, 2)).transpose(1, 2)
        content = torch.cat([midi_pad_feat, midi_feat], dim=1) + torch.cat([ph_pad_feat, ph_feat], dim=1)
        content = m.final_proj(content.transpose(1, 2)).transpose(1, 2)

        x = m.proj_in(x).transpose(1, 2)
        h = torch.cat([prompt, x], dim=1)
        t_emb = m.t_embedder(t)
        caption = m.c_embedder(caption)
        cap_pool = caption.sum(dim=1) / caption.shape[1]  # the original's mean over an all-ones mask
        adaln = t_emb + m.cap_embedder(cap_pool) + prompt_global
        h = h + content * torch.sigmoid(m.gate_content(content))

        S = h.shape[1]
        freqs = (self.rope_cos[:S].view(1, S, 1, -1), self.rope_sin[:S].view(1, S, 1, -1))
        key_mask = None
        if valid is not None:  # the prompt's tokens come first, then the latent's
            key_mask = (torch.arange(S, device=x.device) < (Tp + valid).view(-1, 1))[:, None, None, :]
        h = self.block(m.pre_transformer, h, caption, freqs, adaln, caption, language, key_mask)
        f0_full = torch.cat([f0_pad_feat, _conv_leaky_pool(m.f0_proj, f0_log).transpose(1, 2)], dim=1)
        h = h + f0_full * torch.sigmoid(m.gate_f0(f0_full))
        for blk in m.blocks:
            h = self.block(blk, h, caption, freqs, adaln, caption, language, key_mask)
        return m.final_layer(h[:, Tp:, :], adaln).transpose(1, 2)


def step_grid(steps):
    """(timesteps as the DiT reads them, Euler step sizes) of CFM.sample_cfg with `steps` points."""
    span = torch.linspace(0, 1, steps)
    times = [float(int(t0 * NUM_TIMESTEPS)) for t0 in span[:-1]]
    return times, [(t1 - t0).reshape(()) for t0, t1 in zip(span[:-1], span[1:])]


class Stage2Sampler(nn.Module):
    """The Euler loop of CFM.sample_cfg in given-F0 mode with CFG on the caption, unrolled.

    cfg_batch: the conditional and the unconditional branch as one batch of 2 (for the export); False runs
    them one after the other, as the original does."""

    def __init__(self, given, steps=8, cfg_batch=True):
        super().__init__()
        self.given, self.steps, self.cfg_batch = given, steps, cfg_batch
        self.times, dts = step_grid(steps)
        for i, dt in enumerate(dts):
            self.register_buffer(f'dt{i}', dt.clone(), persistent=False)

    def forward(self, noise, caption, empty_caption, ph, midi, f0_hz, language, cfg_scale, prompt_embed=None, callback=None,
                valid=None):
        """noise [1, 20, L]; caption, empty_caption [1, 80, 1024]; ph, midi [1, T]; f0_hz [1, T];
        language [1]; cfg_scale [1] or a float; valid [1] int64: see Stage2Given (None: unmasked).
        Returns the latent [1, 20, L]."""
        x = noise
        if self.cfg_batch:
            two = lambda a: torch.cat([a, a], dim=0)
            cap2 = torch.cat([caption, empty_caption], dim=0)
            ph2, midi2, f02, lang2 = two(ph), two(midi), two(f0_hz), two(language)
            prompt2 = None if prompt_embed is None else two(prompt_embed)
        for i, t in enumerate(self.times):
            if self.cfg_batch:
                tt = torch.full((2,), t, device=x.device)
                v2 = self.given(two(x), tt, cap2, ph2, midi2, f02, lang2, prompt2, valid)
                v, v_uncond = v2[:1], v2[1:]
            else:
                tt = torch.full((1,), t, device=x.device)
                v = self.given(x, tt, caption, ph, midi, f0_hz, language, prompt_embed, valid)
                v_uncond = self.given(x, tt, empty_caption, ph, midi, f0_hz, language, prompt_embed, valid)
            v = v_uncond + cfg_scale * (v - v_uncond)
            x = x + getattr(self, f'dt{i}') * v
            if callback is not None:
                callback(i, x)
        return x
