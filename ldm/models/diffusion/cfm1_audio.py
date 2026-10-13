import math

import torch

from ldm.util import default
from ldm.models.diffusion.ddpm_audio import LatentDiffusion_audio

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}

FROZEN = ('first_stage_model.', 'cond_stage_model.')  # restored from their own files, kept out of checkpoints


class CFM(LatentDiffusion_audio):

    def __init__(self, **kwargs):

        super(CFM, self).__init__(**kwargs)
        self.sigma_min = 1e-4

    def on_train_batch_start(self, batch, batch_idx):
        # the load-balancing weight and the Gumbel temperature follow the training step
        net = self.model.diffusion_model
        net.set_step(self.global_step)
        self.log_dict({'train/gumbel_temperature': net.temperature, 'train/lb_weight': net.loss_w}, on_step=True, on_epoch=False)
        return super().on_train_batch_start(batch, batch_idx)

    # The frozen VAE and text encoder always run in full precision, also when the trainer uses mixed
    # precision for the transformer (trainer.precision: bf16).
    def encode_first_stage(self, x):
        with torch.autocast('cuda', enabled=False):
            return super().encode_first_stage(x.float())

    def decode_first_stage(self, z, *args, **kwargs):
        with torch.autocast('cuda', enabled=False):
            return super().decode_first_stage(z.float(), *args, **kwargs)

    def get_learned_conditioning(self, c):
        with torch.autocast('cuda', enabled=False):
            return super().get_learned_conditioning(c)

    def on_save_checkpoint(self, checkpoint):
        for k in [k for k in checkpoint["state_dict"] if k.startswith(FROZEN)]:
            del checkpoint["state_dict"][k]

    def on_load_checkpoint(self, checkpoint):
        for k, v in self.state_dict().items():
            if k.startswith(FROZEN):
                checkpoint["state_dict"].setdefault(k, v)
        self.model.diffusion_model.set_step(checkpoint.get("global_step", 0))

    def p_losses(self, x_start, cond, t, noise=None):
        x1 = x_start
        x0 = default(noise, lambda: torch.randn_like(x_start))
        ut = x1 - (1 - self.sigma_min) * x0  # 和ut的梯度没关系
        t_unsqueeze = t.unsqueeze(1).unsqueeze(1).float() / self.num_timesteps
        x_noisy = t_unsqueeze * x1 + (1. - (1 - self.sigma_min) * t_unsqueeze) * x0

        model_output, aux, _ = self.apply_model(x_noisy, t, cond)

        prefix = 'train' if self.training else 'val'
        mean_dims = list(range(1, len(ut.shape)))
        loss_simple = self.get_loss(model_output, ut, mean=False).mean(dim=mean_dims).mean()
        loss = self.l_simple_weight * loss_simple + aux['lb'] + aux['f0'] + aux['uv']
        loss_dict = {f'{prefix}/loss_simple': loss_simple, f'{prefix}/lb_loss': aux['lb'],
                     f'{prefix}/f0_loss': aux['f0'], f'{prefix}/uv_loss': aux['uv'], f'{prefix}/loss': loss}
        return loss, loss_dict

    @torch.no_grad()
    def sample_cfg(self, cond, unconditional_guidance_scale=1.0, unconditional_conditioning=None, timesteps=None,
                   shape=None, x_latent=None, f0_freeze_step=0, **kwargs):
        """Euler solution of the flow ODE from noise (t = 0) to data (t = 1).

        cond / unconditional_conditioning: conditioning dicts as returned by get_learned_conditioning,
        with `prompt_embed` in their 'acoustic' part. shape: [B, C, T_latent].
        f0_freeze_step: in `predict` mode the F0 that the conditional branch predicts at this step is given
        to all later steps, as a supplied F0 would be. The default 0 keeps the prediction made from pure
        noise, i.e. from the notes alone. None lets every step condition on its own prediction, as the
        released code does: the predictor then reads the pitch of its own half-made latent and drifts off
        the notes (scripts/checks/check_f0_choice.py).
        Returns (latent, F0 in Hz [B, 1, T_mel] that conditioned the last step: the given F0, or the
        prediction of the conditional branch)."""
        if shape is None:
            n = cond['acoustic']['mel2ph']
            shape = (n.shape[0], self.mel_dim, math.ceil(n.shape[2] / 4))
        x = torch.randn(shape, device=self.device) if x_latent is None else x_latent
        t_span = torch.linspace(0, 1, 25 if timesteps is None else timesteps, device=self.device)
        guided = unconditional_conditioning is not None and unconditional_guidance_scale != 1.0
        f0 = None
        for i, (t0, t1) in enumerate(zip(t_span[:-1], t_span[1:])):
            t = torch.full((x.shape[0],), int(t0 * self.num_timesteps), device=self.device, dtype=torch.long)
            v, _, f0 = self.apply_model(x, t, cond)
            if guided:
                v_uncond, _, _ = self.apply_model(x, t, unconditional_conditioning)
                v = v_uncond + unconditional_guidance_scale * (v - v_uncond)
            x = x + (t1 - t0) * v
            if i == f0_freeze_step and (cond.get('f0_mode') or ('predict' if cond.get('infer') else 'train')) == 'predict':
                acoustic = dict(cond['acoustic'], f0=f0)
                cond = dict(cond, acoustic=acoustic, f0_mode='given')
                if guided:
                    unconditional_conditioning = dict(unconditional_conditioning, acoustic=acoustic, f0_mode='given')
        return x, f0

    @torch.no_grad()
    def sample(self, cond, timesteps=None, shape=None, x_latent=None, **kwargs):
        return self.sample_cfg(cond, timesteps=timesteps, shape=shape, x_latent=x_latent)

    @torch.no_grad()
    def synthesize(self, batch, f0_mode='predict', cfg_scale=3.0, steps=25, f0_freeze_step=0):
        """Mel-spectrograms for a batch as the dataset's collater builds it.

        f0_mode 'predict': from notes, phonemes, caption and prompt, with the model's own F0 (the one
        predicted at step `f0_freeze_step`, see sample_cfg); 'given': additionally conditioned on the F0 in the batch.
        Classifier-free guidance drops the caption only; prompt, notes and phonemes stay in both branches.
        Returns (mel [B, 80, T], F0 used in Hz [B, T])."""
        assert f0_mode in ('predict', 'given'), f0_mode
        src = batch[self.cond_stage_key]
        acoustic = {k: v.to(self.device) for k, v in src['acoustic'].items()}
        cond = self.get_learned_conditioning(dict(caption=list(src['caption']), acoustic=acoustic, name=src['name'],
                                                  infer=True, f0_mode=f0_mode))
        if 'prompt_embed' not in acoustic:
            acoustic['prompt_embed'] = self.get_first_stage_encoding(self.encode_first_stage(acoustic['prompt'])).detach()
        uncond = None
        if cfg_scale != 1.0:
            empty = self.get_learned_conditioning(dict(caption=[''] * len(src['caption']), acoustic=acoustic,
                                                       name=src['name'], infer=True, f0_mode=f0_mode))
            uncond = dict(cond, caption=empty['caption'])
        shape = (acoustic['ph'].shape[0], self.mel_dim, math.ceil(acoustic['ph'].shape[-1] / 4))
        z, f0 = self.sample_cfg(cond, cfg_scale, uncond, timesteps=steps, shape=shape, f0_freeze_step=f0_freeze_step)
        return self.decode_first_stage(z), f0[:, 0]

