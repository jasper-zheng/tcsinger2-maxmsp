"""Synthesis of validation clips with the stage-2 model, and the figures that describe it (WP6).

A clip is synthesised in two modes:
  predict  from notes, phonemes, caption and prompt; the model predicts the F0, which also drives the vocoder
           (the F0 predicted at the first sampling step, see CFM.sample_cfg)
  given    the same inputs plus the clip's own F0

Figures per clip (F0 frame error = share of frames with a voicing error or a pitch more than 20% off):
  given:    ffe_out_vs_f0     F0 of the vocoded output against the given F0
  predict:  ffe_pred_vs_sung  predicted F0 against the sung F0 (how close notes-only pitch is to the performance)
            ffe_out_vs_pred   F0 of the vocoded output against the predicted F0
            notes_within_1    share of notes whose median predicted F0 is within one semitone of the note
  both:     mel_l1            output mel against the recording's mel (log10 units)

`Stage2SampleLogger` is the training callback (configs/tcsinger2.yaml); scripts/eval_stage2.py uses
`Synthesizer` on a checkpoint.
"""
import os
import traceback

import numpy as np
import soundfile as sf
import torch
from pytorch_lightning.callbacks import Callback

from ldm.data.phone_set import MIDI_PAD_ID

SR, HOP = 48000, 256


def f0_metrics(ref, est):
    """(F0 frame error, voicing error, median pitch difference in cents); same definition as scripts/vocode_gt.py."""
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    v_ref, v_est = ref > 0, est > 0
    both = v_ref & v_est
    gross = both & (np.abs(est - ref) > 0.2 * np.where(both, ref, 1))
    ffe = (np.sum(v_ref != v_est) + np.sum(gross)) / n
    cents = 1200 * np.abs(np.log2(est[both] / ref[both])) if both.any() else np.zeros(1)
    return float(ffe), float(np.sum(v_ref != v_est) / n), float(np.median(cents))


def notes_within(f0, midi, semitones=1.0):
    """Share of notes (runs of one MIDI pitch > 0) whose median voiced F0 is within `semitones` of the note."""
    hit = total = 0
    start = 0
    for t in range(1, len(midi) + 1):
        if t == len(midi) or midi[t] != midi[start]:
            note = midi[start]
            if 0 < note < MIDI_PAD_ID and t - start >= 8:
                seg = f0[start:t]
                seg = seg[seg > 0]
                if len(seg) >= 3:
                    total += 1
                    hit += abs(69 + 12 * np.log2(np.median(seg) / 440.0) - note) <= semitones
                else:
                    total += 1
            start = t
    return hit / total if total else float('nan')


def pick_clips(dataset, n, seed=0, min_frames=750, max_frames=2250):
    """n validation clips of 4 to 12 s, spread over corpus and language, the same for a given seed."""
    df = dataset.dataset
    ok = df[(dataset.frames >= min_frames) & (dataset.frames <= max_frames)]
    groups = ok.groupby([ok['dataset'].fillna('GTSinger'), ok['language']], sort=True)
    rng = np.random.RandomState(seed)
    pools = [rng.permutation(g.index.values).tolist() for _, g in groups]
    picks = []
    while len(picks) < n and any(pools):
        for pool in pools:
            if pool and len(picks) < n:
                picks.append(int(pool.pop()))
    return picks


class Synthesizer:
    """Holds the vocoder and the pitch tracker; synthesises and measures single clips."""

    def __init__(self, vocoder_ckpt='useful_ckpts/hifigan', rmvpe_ckpt='useful_ckpts/rmvpe/model.pt', cfg_scale=3.0, steps=25,
                 f0_freeze_step=0):
        from utils.rmvpe import RMVPE
        from vocoder.hifigan.hifigan_nsf import HifiGAN_NSF
        self.vocoder = HifiGAN_NSF(vocoder_ckpt)
        self.rmvpe = RMVPE(rmvpe_ckpt)
        self.cfg_scale, self.steps, self.f0_freeze_step = cfg_scale, steps, f0_freeze_step

    @torch.no_grad()
    def clip(self, model, dataset, idx, mode, seed=0):
        """Synthesises row `idx` of `dataset` alone (no padding). Returns (wav, dict of figures and curves)."""
        item = dataset[idx]
        batch = dataset.collater([item])
        n = item['image'].shape[1]  # frames before the collater's padding
        torch.manual_seed(seed + idx)
        mel, f0 = model.synthesize(batch, f0_mode=mode, cfg_scale=self.cfg_scale, steps=self.steps, f0_freeze_step=self.f0_freeze_step)
        mel, f0 = mel[0, :, :n].float().cpu().numpy(), f0[0, :n].float().cpu().numpy()
        wav = self.vocoder.vocode(mel, f0=f0)
        f0_out, _ = self.rmvpe.get_pitch(wav.astype(np.float32), SR, HOP, len(wav) // HOP)
        f0_sung, midi = item['f0'][0, :n], item['ep_pitches'][0, :n]
        out = dict(name=item['name'], mode=mode, seconds=round(n * HOP / SR, 2), caption=item['caption'],
                   mel_l1=float(np.mean(np.abs(mel - item['image'][:, :n]))))
        if mode == 'given':
            out['ffe_out_vs_f0'], out['voicing_err_out'], out['cents_out'] = f0_metrics(f0_sung, f0_out)
        else:
            out['ffe_pred_vs_sung'], out['voicing_err_pred'], out['cents_pred_vs_sung'] = f0_metrics(f0_sung, f0)
            out['ffe_out_vs_pred'], _, out['cents_out'] = f0_metrics(f0, f0_out)
            out['notes_within_1'] = notes_within(f0, midi)
            out['notes_within_1_sung'] = notes_within(f0_sung, midi)
        curves = dict(f0_used=f0, f0_out=f0_out, f0_sung=f0_sung, midi=midi)
        return wav, out, curves


def plot_f0(path, curves_by_mode, title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c = next(iter(curves_by_mode.values()))
    t = np.arange(len(c['f0_sung'])) * HOP / SR
    semi = lambda f: np.where(f > 0, 69 + 12 * np.log2(np.maximum(f, 1e-3) / 440.0), np.nan)
    fig, ax = plt.subplots(figsize=(max(8, t[-1] * 1.6), 3.6))
    midi = c['midi'].astype(float)
    ax.plot(t, np.where((midi > 0) & (midi < MIDI_PAD_ID), midi, np.nan), color='0.6', lw=3, label='notes')
    ax.plot(t, semi(c['f0_sung']), color='tab:red', lw=1, label='sung F0')
    if 'predict' in curves_by_mode:
        ax.plot(t[:len(curves_by_mode['predict']['f0_used'])], semi(curves_by_mode['predict']['f0_used']), color='tab:green', lw=1,
                label='predicted F0')
    ax.set_xlabel('s')
    ax.set_ylabel('MIDI pitch')
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def safe_name(name):
    return name.replace('#', '_').replace(' ', '_').replace('/', '_')


class Stage2SampleLogger(Callback):
    """Every `batch_frequency` training steps: `max_images` fixed validation clips in both modes, with EMA weights.

    Writes <logdir>/samples/step<N>/<clip>_{predict,given}.wav, <clip>_f0.png and metrics.json, and logs the
    mean figures to TensorBoard under samples/. A failure here is printed and does not stop the training."""

    def __init__(self, batch_frequency, max_images=8, cfg_scale=3.0, steps=25, seed=0, **kwargs):
        super().__init__()
        self.batch_frequency, self.n, self.seed = batch_frequency, max_images, seed
        self.cfg_scale, self.steps = cfg_scale, steps
        self.synth = self.picks = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = pl_module.global_step
        if step == 0 or step % self.batch_frequency != 0 or trainer.global_rank != 0:
            return
        was_training = pl_module.training
        pl_module.eval()
        try:
            self.run(pl_module, trainer.datamodule.datasets['validation'], os.path.join(trainer.logdir, 'samples', f'step{step:08d}'))
        except Exception:  # noqa: BLE001
            print('Stage2SampleLogger failed:')
            traceback.print_exc()
        if was_training:
            pl_module.train()

    def run(self, pl_module, dataset, out_dir):
        import json
        if self.synth is None:
            self.synth = Synthesizer(cfg_scale=self.cfg_scale, steps=self.steps)
            self.picks = pick_clips(dataset, self.n, self.seed)
        os.makedirs(out_dir, exist_ok=True)
        rows = []
        with pl_module.ema_scope():
            for idx in self.picks:
                curves = {}
                for mode in ('predict', 'given'):
                    wav, row, curves[mode] = self.synth.clip(pl_module, dataset, idx, mode, self.seed)
                    sf.write(os.path.join(out_dir, f"{safe_name(row['name'])}_{mode}.wav"), np.clip(wav, -1, 1), SR, subtype='PCM_16')
                    rows.append(row)
                plot_f0(os.path.join(out_dir, f"{safe_name(row['name'])}_f0.png"), curves, row['name'])
        mean = {}
        for key in ('ffe_out_vs_f0', 'ffe_pred_vs_sung', 'ffe_out_vs_pred', 'notes_within_1', 'voicing_err_pred'):
            vals = [r[key] for r in rows if key in r and not np.isnan(r[key])]
            if vals:
                mean[key] = float(np.mean(vals))
        for mode in ('predict', 'given'):
            mean[f'mel_l1_{mode}'] = float(np.mean([r['mel_l1'] for r in rows if r['mode'] == mode]))
        with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(step=int(pl_module.global_step), mean=mean, clips=rows), f, ensure_ascii=False, indent=1)
        for key, value in mean.items():
            pl_module.logger.experiment.add_scalar(f'samples/{key}', value, pl_module.global_step)
        print(f'samples at step {pl_module.global_step}: ' + ', '.join(f'{k} {v:.3f}' for k, v in mean.items()), flush=True)


FROZEN = ('first_stage_model.', 'cond_stage_model.')


def load_stage2(ckpt, config='configs/tcsinger2.yaml', device='cuda'):
    """The stage-2 model with the weights of a resume checkpoint, a milestone or an exported file.

    These files hold the transformer (and its EMA copy) only; the VAE and the text encoder come from
    the paths in the config."""
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    model = instantiate_from_config(OmegaConf.load(config).model)
    state = torch.load(ckpt, map_location='cpu')
    missing, unexpected = model.load_state_dict(state['state_dict'], strict=False)
    lost = [k for k in missing if not k.startswith(FROZEN)]
    assert not lost and not unexpected, (lost[:5], unexpected[:5])
    step = int(state.get('global_step', 0))
    model.model.diffusion_model.set_step(step)
    return model.to(device).eval(), step


def build_dataset(split, config='configs/tcsinger2.yaml', **overrides):
    """The stage-2 dataset of a split with the data options of the config."""
    from omegaconf import OmegaConf
    from ldm.data.tcsinger2_dataset import JoinManifestSpecs
    p = OmegaConf.load(config).data.params
    keys = ('main_spec_dir_path', 'other_spec_dir_path', 'mel_num', 'spec_crop_len', 'drop', 'max_tokens', 'ext_share',
            'max_prompt_len', 'max_batch_clips')
    return JoinManifestSpecs(split, **{**{k: p[k] for k in keys if k in p}, **overrides})
