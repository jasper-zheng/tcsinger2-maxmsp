import os
from pytorch_memlab import LineProfiler,profile
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps
from torchvision.utils import make_grid
try:
    from pytorch_lightning.utilities.distributed import rank_zero_only
except:
    from pytorch_lightning.utilities import rank_zero_only # torch2
from torchdyn.core import NeuralODE
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.models.diffusion.ddpm_audio import LatentDiffusion_audio, disabled_train
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from omegaconf import ListConfig
import math
import matplotlib.pyplot as plt

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def f0_to_figure(f0_gt, f0_pred=None):
    fig = plt.figure(figsize=(12, 8))
    # f0_gt 是必需的，其它可选
    f0_gt = f0_gt.cpu().numpy()  # 记得把tensor移动到cpu再转成numpy
    plt.plot(f0_gt, color='r', label='gt')
    
    f0_pred = f0_pred.cpu().numpy()
    plt.plot(f0_pred, color='green', label='pred')

    plt.legend()
    return fig

class CFM(LatentDiffusion_audio):

    def __init__(self, **kwargs):

        super(CFM, self).__init__(**kwargs)
        self.sigma_min = 1e-4

    def p_losses(self, x_start, cond, t, noise=None):
        x1 = x_start
        x0 = default(noise, lambda: torch.randn_like(x_start))
        ut = x1 - (1 - self.sigma_min) * x0  # 和ut的梯度没关系
        t_unsqueeze = t.unsqueeze(1).unsqueeze(1).float() / self.num_timesteps
        x_noisy = t_unsqueeze * x1 + (1. - (1 - self.sigma_min) * t_unsqueeze) * x0

        f0_gt=cond['acoustic']['f0'] 
        model_output, lb_loss,f0_pred = self.apply_model(x_noisy, t, cond)
        
        # ---------- 每 5000 步绘制前 5 条曲线 ----------
        if self.training and (self.global_step % 5000 == 0):
            os.makedirs('test', exist_ok=True)
            num_plot = min(5, f0_gt.size(0))
            for idx in range(num_plot):
                gt_curve = f0_gt[idx, 0].detach().cpu()
                pred_curve = f0_pred[idx, 0].detach().cpu()
                fig = f0_to_figure(gt_curve, pred_curve)
                fig_path = os.path.join('test', f'f0_idx_{idx}.png')
                fig.savefig(fig_path)
                plt.close(fig)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        target = ut

        mean_dims = list(range(1,len(target.shape)))
        loss_simple = self.get_loss(model_output, target, mean=False).mean(dim=mean_dims)
        
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        loss_dict.update({f'{prefix}/lb_loss': lb_loss})
        
        loss = loss_simple
        loss = self.l_simple_weight * loss.mean()+lb_loss
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def sample(self, cond, batch_size=16, timesteps=None, shape=None, x_latent=None, t_start=None, **kwargs):
        if shape is None:
            # if self.channels > 0:
            #     shape = (batch_size, self.channels, self.mel_dim, self.mel_length)
            # else:
            #     shape = (batch_size, self.mel_dim, self.mel_length)
            mel_length = math.ceil(cond['acoustic']['mel2ph'].shape[2] * 1 / 4)
            shape = (self.channels, self.mel_dim, mel_length) if self.channels > 0 else (self.mel_dim, mel_length)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        neural_ode = NeuralODE(self.ode_wrapper(cond), solver='euler', sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        t_span = torch.linspace(0, 1, 25 if timesteps is None else timesteps)
        if t_start is not None:
            t_span = t_span[t_start:]

        x0 = torch.randn(shape, device=self.device) if x_latent is None else x_latent
        eval_points, traj = neural_ode(x0, t_span)

        return traj[-1], traj

    def ode_wrapper(self, cond):
        # self.estimator receives x, mask, mu, t, spk as arguments
        return Wrapper(self, cond)

    @torch.no_grad()
    def sample_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning, batch_size=16, timesteps=None, shape=None, x_latent=None, t_start=None, **kwargs):
        if shape is None:
            mel_length = math.ceil(cond['acoustic']['mel2ph'].shape[2] * 1 / 4)
            shape = (self.channels, self.mel_dim, mel_length) if self.channels > 0 else (self.mel_dim, mel_length)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        neural_ode = NeuralODE(self.ode_wrapper_cfg(cond, unconditional_guidance_scale, unconditional_conditioning), solver='euler', sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        t_span = torch.linspace(0, 1, 25 if timesteps is None else timesteps)

        if t_start is not None:
            t_span = t_span[t_start:]

        x0 = torch.randn(shape, device=self.device) if x_latent is None else x_latent
        eval_points, traj = neural_ode(x0, t_span)

        return traj[-1], traj

    def ode_wrapper_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning):
        # self.estimator receives x, mask, mu, t, spk as arguments
        return Wrapper_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning)


    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        # if use_original_steps:
        #     sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
        #     sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        # else:
        sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
        sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas
        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)


class Wrapper(nn.Module):
    def __init__(self, net, cond):
        super(Wrapper, self).__init__()
        self.net = net
        self.cond = cond

    def forward(self, t, x, args):
        t = torch.tensor([t * 1000] * x.shape[0], device=t.device).long()
        results,loss= self.net.apply_model(x, t, self.cond)
        return results


class Wrapper_cfg(nn.Module):

    def __init__(self, net, cond, unconditional_guidance_scale, unconditional_conditioning):
        super(Wrapper_cfg, self).__init__()
        self.net = net
        self.cond = cond
        self.unconditional_conditioning = unconditional_conditioning
        self.unconditional_guidance_scale = unconditional_guidance_scale

        self.f0_preds = []          # 预测的 f0

    def forward(self, t, x, args):
        t = torch.tensor([t * 1000] * x.shape[0], device=t.device).long()
        e_t,loss,f0_pred= self.net.apply_model(x, t, self.cond)
        e_t_uncond,loss,f0_pred_uncond= self.net.apply_model(x, t, self.unconditional_conditioning)
        e_t = e_t_uncond + self.unconditional_guidance_scale * (e_t - e_t_uncond)    
        f0_pred = f0_pred_uncond + self.unconditional_guidance_scale * (f0_pred - f0_pred_uncond)    

        self.f0_preds.append(f0_pred.detach().cpu())

        return e_t
