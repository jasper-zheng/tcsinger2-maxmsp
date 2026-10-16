"""Training and sampling of the F0 generator (WP10). Trained through main.py (configs/f0gen.yaml, f0gen_reg.yaml).

objective 'flow': conditional flow matching as stage 2 (CFM.p_losses: x_t = t x1 + (1 - (1 - sigma_min) t) x0,
  target velocity x1 - (1 - sigma_min) x0) on x1 = (r, voicing); the r channel counts where the frame is voiced
  and trusted, voicing on every frame. Sampling: Euler from noise, classifier-free guidance against the null
  prompt (the prompt is dropped in training by the dataset, ldm.f0gen.dataset).
objective 'regression': the same network without the noisy input; MSE on r (voiced and trusted frames) and
  cross-entropy of the voicing logit.

generate() returns F0 in Hz per frame. EMA weights are used for validation, samples and export.
"""
import contextlib
import os
import traceback

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

from ldm.f0gen.features import decode
from ldm.f0gen.model import F0Net
from ldm.modules.ema import LitEma


class F0Gen(pl.LightningModule):
    def __init__(self, net_config=None, objective='flow', warmup_steps=2000, weight_decay=0.01, ema_decay=0.999,
                 sigma_min=1e-4, voicing_weight=1.0, monitor='val/loss_ema'):
        super().__init__()
        assert objective in ('flow', 'regression'), objective
        self.save_hyperparameters()
        self.objective, self.sigma_min, self.voicing_weight = objective, sigma_min, voicing_weight
        self.warmup_steps, self.weight_decay = warmup_steps, weight_decay
        self.net = F0Net(**(net_config or {}))
        self.ema = LitEma(self.net, decay=ema_decay)
        self.monitor = monitor
        self.learning_rate = 2e-4  # main.py sets it from base_learning_rate

    @contextlib.contextmanager
    def ema_scope(self):
        self.ema.store(self.net.parameters())
        self.ema.copy_to(self.net)
        try:
            yield
        finally:
            self.ema.restore(self.net.parameters())

    def target(self, batch):
        return torch.stack([batch['r'], batch['v']], dim=1)  # [B, 2, T]

    def losses(self, batch):
        x1 = self.target(batch)
        mask, trust = batch['mask'], batch['trust']
        n_mask, n_trust = mask.sum().clamp(min=1), trust.sum().clamp(min=1)
        if self.objective == 'flow':
            t = torch.rand(x1.shape[0], device=x1.device)
            x0 = torch.randn_like(x1)
            xt = t[:, None, None] * x1 + (1 - (1 - self.sigma_min) * t[:, None, None]) * x0
            out = self.net(xt, t, batch)
            err = (out - (x1 - (1 - self.sigma_min) * x0)) ** 2
            loss_r = (err[:, 0] * trust).sum() / n_trust
            loss_v = (err[:, 1] * mask).sum() / n_mask
        else:
            out = self.net(torch.zeros_like(x1), torch.zeros(x1.shape[0], device=x1.device), batch)
            loss_r = (((out[:, 0] - batch['r']) ** 2) * trust).sum() / n_trust
            bce = F.binary_cross_entropy_with_logits(out[:, 1].float(), (batch['v'] > 0).float(), reduction='none')
            loss_v = (bce * mask).sum() / n_mask
        return loss_r + self.voicing_weight * loss_v, dict(loss_r=loss_r, loss_v=loss_v)

    MAX_BAD_STEPS = 50  # consecutive steps with a non-finite loss before the training is stopped

    def training_step(self, batch, batch_idx):
        loss, parts = self.losses(batch)
        if not torch.isfinite(loss):  # skip the step (Lightning makes no update for None); stop if it persists
            self.bad_steps = getattr(self, 'bad_steps', 0) + 1
            print(f'non-finite loss at step {self.global_step} ({self.bad_steps} in a row); step skipped', flush=True)
            if self.bad_steps >= self.MAX_BAD_STEPS:
                raise RuntimeError(f'{self.bad_steps} steps in a row with a non-finite loss: the training has diverged')
            return None
        self.bad_steps = 0
        self.log_dict({'train/loss': loss, **{f'train/{k}': v for k, v in parts.items()}}, on_step=True, on_epoch=False,
                      prog_bar=True, batch_size=batch['mask'].shape[0])
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        self.ema(self.net)

    def validation_step(self, batch, batch_idx):
        torch.manual_seed(batch_idx)  # the same noise and times at every validation
        with self.ema_scope():
            loss, parts = self.losses(batch)
        self.log_dict({'val/loss_ema': loss, **{f'val/{k}_ema': v for k, v in parts.items()}}, on_step=False, on_epoch=True,
                      batch_size=batch['mask'].shape[0])

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, p in self.net.named_parameters():
            (no_decay if p.ndim < 2 or 'null' in name else decay).append(p)
        opt = torch.optim.AdamW([dict(params=decay, weight_decay=self.weight_decay), dict(params=no_decay, weight_decay=0.0)],
                                lr=self.learning_rate, betas=(0.9, 0.98))
        warm = lambda step: min(1.0, (step + 1) / self.warmup_steps)
        return [opt], [dict(scheduler=torch.optim.lr_scheduler.LambdaLR(opt, warm), interval='step', frequency=1)]

    @torch.no_grad()
    def generate(self, batch, steps=32, cfg=1.0, seed=0, ema=True, return_target=False):
        """F0 in Hz [B, T] (0 = unvoiced) for a collated batch (ldm.f0gen.dataset.collate)."""
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        scope = self.ema_scope() if ema else contextlib.nullcontext()
        with scope:
            B, T = batch['mask'].shape
            if self.objective == 'regression':
                out = self.net(torch.zeros(B, 2, T, device=self.device), torch.zeros(B, device=self.device), batch)
                x = torch.stack([out[:, 0], torch.where(out[:, 1] > 0, 1.0, -1.0)], dim=1)
            else:
                g = torch.Generator(device=self.device).manual_seed(seed)
                x = torch.randn(B, 2, T, device=self.device, generator=g)
                style = self.net.style(batch)
                null = self.net.lang(batch['language']) + self.net.prompt.null[None]
                ts = torch.linspace(0, 1, steps + 1, device=self.device)
                for t0, t1 in zip(ts[:-1], ts[1:]):
                    t = torch.full((B,), float(t0), device=self.device)
                    v = self.net(x, t, batch, style=style)
                    if cfg != 1.0:
                        v_null = self.net(x, t, batch, style=null)
                        v = v_null + cfg * (v - v_null)
                    x = x + (t1 - t0) * v
        f0 = decode(x[:, 0], x[:, 1], batch['ref']) * batch['mask']
        return (f0, x) if return_target else f0


def load_f0gen(path, device='cuda'):
    """An F0 generator from a training checkpoint, a milestone or an export (scripts/export_f0gen.py)."""
    # weights_only=False: the hyperparameters are an OmegaConf DictConfig, which torch >= 2.6 refuses by default
    state = torch.load(path, map_location='cpu', weights_only=False)
    hp = state.get('hyper_parameters') or state.get('f0gen_config')
    model = F0Gen(**hp)
    missing, unexpected = model.load_state_dict(state['state_dict'], strict=False)
    assert not unexpected and all(k.startswith('ema.') for k in missing), (missing[:5], unexpected[:5])
    if missing:  # an export holds the EMA weights as the network's own; copy them into the EMA too
        model.ema = LitEma(model.net)
    model.step = int(state.get('global_step', 0))
    return model.to(device).eval()


class F0SampleLogger(Callback):
    """Every `batch_frequency` steps: F0 curves for `n_clips` fixed validation clips (EMA weights), the figures of
    ldm.f0gen.metrics in TensorBoard (samples/...) and <logdir>/samples/step<N>/metrics.json, F0 plots of `n_plots`."""

    def __init__(self, batch_frequency=5000, n_clips=64, n_plots=8, steps=32, cfg=1.0, **kwargs):
        super().__init__()
        self.batch_frequency, self.n_clips, self.n_plots, self.steps, self.cfg = batch_frequency, n_clips, n_plots, steps, cfg
        self.picks = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = pl_module.global_step
        if step == 0 or step % self.batch_frequency != 0 or trainer.global_rank != 0:
            return
        was_training = pl_module.training
        pl_module.eval()
        try:
            self.run(pl_module, trainer.datamodule.datasets['validation'], os.path.join(trainer.logdir, 'samples', f'step{step:08d}'))
        except Exception:  # noqa: BLE001
            print('F0SampleLogger failed:')
            traceback.print_exc()
        if was_training:
            pl_module.train()

    def run(self, pl_module, dataset, out_dir):
        import json
        from ldm.f0gen.dataset import collate
        from ldm.f0gen.metrics import curve_rows, summarise
        from ldm.stage2_samples import pick_clips
        if self.picks is None:
            self.picks = pick_clips(dataset, self.n_clips, 0, min_frames=375, max_frames=3750)
        os.makedirs(out_dir, exist_ok=True)
        items = [dataset[i] for i in self.picks]
        clips, notes, curves = [], [], []
        for k in range(0, len(items), 16):
            part = items[k:k + 16]
            f0 = pl_module.generate(collate(part), steps=self.steps, cfg=self.cfg, seed=k).float().cpu().numpy()
            curves += [f0[i, :len(it['r'])] for i, it in enumerate(part)]
        for it, f0 in zip(items, curves):
            clip, nts = curve_rows(f0, it['midi'], it['onset'], it['f0'])
            clips.append(dict(clip, name=it['name']))
            notes += nts
        mean = summarise(clips, notes)
        with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(step=int(pl_module.global_step), mean=mean, clips=clips), f, ensure_ascii=False, indent=1)
        for key, value in mean.items():
            if isinstance(value, float) and np.isfinite(value):
                pl_module.logger.experiment.add_scalar(f'samples/{key}', value, pl_module.global_step)
        for it, f0 in list(zip(items, curves))[:self.n_plots]:
            plot(os.path.join(out_dir, f"{it['name'].replace('#', '_').replace(' ', '_').replace('/', '_')}.png"), it, f0)
        print(f'samples at step {pl_module.global_step}: ' + ', '.join(f'{k} {v:.3f}' for k, v in mean.items() if isinstance(v, float)),
              flush=True)


def plot(path, item, f0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ldm.data.phone_set import MIDI_PAD_ID
    semi = lambda f: np.where(f > 0, 69 + 12 * np.log2(np.maximum(f, 1e-3) / 440.0), np.nan)
    t = np.arange(len(item['f0'])) / 187.5
    fig, ax = plt.subplots(figsize=(max(8, t[-1] * 1.6), 3.2))
    midi = item['midi'].astype(float)
    ax.plot(t, np.where((midi > 0) & (midi < MIDI_PAD_ID), midi, np.nan), color='0.6', lw=3, label='notes')
    ax.plot(t, semi(item['f0']), color='tab:red', lw=1, label='sung')
    ax.plot(t[:len(f0)], semi(f0[:len(t)]), color='tab:blue', lw=1, label='generated')
    ax.set_title(item['name'], fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
