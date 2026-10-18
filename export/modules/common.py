"""Ops the MLX delegate lacks, written with ops it has (B4 of EXPORTING_WORK_PACKAGES.md)."""
import torch


def leaky_relu(x, slope=0.01):
    """F.leaky_relu, to the bit: x where x >= 0, else slope * x."""
    return torch.where(x >= 0, x, x * slope)


def group_norm(x, norm):
    """nn.GroupNorm `norm` on x [B, C, T] through mean and var of each group."""
    B, C, T = x.shape
    g = x.reshape(B, norm.num_groups, -1)
    mean = g.mean(-1, keepdim=True)
    var = g.var(-1, unbiased=False, keepdim=True)
    g = (g - mean) * torch.rsqrt(var + norm.eps)
    return g.reshape(B, C, T) * norm.weight[None, :, None] + norm.bias[None, :, None]


def group_norm_masked(x, norm, valid):
    """group_norm with the statistics of each group taken over the first `valid` frames only; valid [B] or [1]
    int64. All frames are normalised."""
    B, C, T = x.shape
    m = (torch.arange(T, device=x.device) < valid.view(-1, 1)).to(x.dtype)[:, None, None, :]
    g = x.reshape(B, norm.num_groups, C // norm.num_groups, T)
    count = m.sum((-1, -2), keepdim=True) * (C // norm.num_groups)
    mean = (g * m).sum((-1, -2), keepdim=True) / count
    var = (((g - mean) ** 2) * m).sum((-1, -2), keepdim=True) / count
    g = (g - mean) * torch.rsqrt(var + norm.eps)
    return g.reshape(B, C, T) * norm.weight[None, :, None] + norm.bias[None, :, None]


def upsample_nearest(x, k):
    """F.interpolate(x, scale_factor=k, mode='nearest') of x [B, C, T]."""
    return x.repeat_interleave(k, dim=-1)


def rotate_pairs(x, cos, sin):
    """x * (cos + i sin) with x read as complex pairs (x[..., 0::2], x[..., 1::2]): the RoPE of
    torch.view_as_complex without complex numbers. cos, sin broadcast against x[..., 0::2]."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
