"""Exports the weights of a training run for the next stage and for inference.

  vae     The VAE without its training loss (discriminator). Stage 2 loads it through
          first_stage_config.params.ckpt_path; the duration predictor inside is used at inference.
          Checked by loading it strictly into the VAE of --config.
  stage2  The transformer, its EMA copy and the latent scale, without optimizer state, VAE or T5. Next to it,
          <out>.yaml: the config the run was trained with, which names the VAE and T5 to load with it
          (ldm.stage2_samples.load_stage2, ldm.infer.pipeline.Singer).
  f0gen   The F0 generator with its EMA weights as the network's own (ldm.f0gen.module.load_f0gen).
          Checked by loading it again.

--run exports the run's newest complete checkpoint in checkpoints/ (at the end of training, the final step);
--ckpt exports a given checkpoint, e.g. a milestone in checkpoints/trainstep_checkpoints/.

Usage (from the repository root, PYTHONPATH=.):
  python scripts-new/py/export.py vae --run work/logs/<time>_ae_singing --out work/ckpts/vae.ckpt
  python scripts-new/py/export.py stage2 --run work/logs/<time>_tcsinger2 --out work/ckpts/tcsinger2.ckpt
  python scripts-new/py/export.py f0gen --ckpt <milestone>.ckpt --out work/ckpts/f0gen.ckpt
"""
import argparse
import glob
import os
import shutil
import zipfile

import torch


def is_complete(path):
    """False for a checkpoint cut off while being written (its zip directory is missing)."""
    try:
        with zipfile.ZipFile(path) as f:
            return len(f.namelist()) > 0
    except (zipfile.BadZipFile, OSError):
        return False


def newest_checkpoint(run):
    for path in sorted(glob.glob(os.path.join(run, 'checkpoints', '*.ckpt')), key=os.path.getmtime, reverse=True):
        if is_complete(path):
            return path
    raise SystemExit(f'error: no complete checkpoint in {run}/checkpoints')


def run_of(ckpt):
    """The run directory of a checkpoint in <run>/checkpoints/ or below (as main.py finds it)."""
    parts = os.path.abspath(ckpt).split(os.sep)
    if 'checkpoints' not in parts:
        raise SystemExit(f'error: {ckpt} is not inside a run directory; give --run')
    return os.sep.join(parts[:len(parts) - 1 - parts[::-1].index('checkpoints')])


def save(obj, out):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    torch.save(obj, out + '.tmp')
    os.replace(out + '.tmp', out)


def export_vae(ckpt, out, config):
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    state = torch.load(ckpt, map_location='cpu')
    sd = {k: v for k, v in state['state_dict'].items() if not k.startswith('loss.')}
    step = int(state['global_step'])
    cfg = OmegaConf.load(config).model.params.first_stage_config
    cfg.params.pop('ckpt_path', None)
    instantiate_from_config(cfg).load_state_dict(sd, strict=True)
    save(dict(state_dict=sd, global_step=step, batches=step // 2, source=ckpt), out)  # two optimizers: 2 steps per batch
    return f'batch {step // 2:,}, {sum(v.numel() for v in sd.values()) / 1e6:.1f} M values'


def export_stage2(ckpt, out, run):
    state = torch.load(ckpt, map_location='cpu')
    sd = {k: v for k, v in state['state_dict'].items() if not k.startswith(('first_stage_model.', 'cond_stage_model.'))}
    assert any(k.startswith('model.diffusion_model.') for k in sd) and 'scale_factor' in sd, f'{ckpt} is no stage-2 checkpoint'
    save(dict(state_dict=sd, global_step=state['global_step'], source=ckpt), out)
    configs = sorted(glob.glob(os.path.join(run, 'configs', '*-project.yaml')))
    if not configs:
        raise SystemExit(f'error: no configs/*-project.yaml in {run}')
    config_out = os.path.splitext(out)[0] + '.yaml'
    shutil.copyfile(configs[-1], config_out)
    return f"step {state['global_step']:,}, {sum(v.numel() for v in sd.values()) / 1e6:.1f} M values, " \
           f"scale_factor {float(sd['scale_factor']):.4f}; config {config_out}"


def export_f0gen(ckpt, out):
    from ldm.f0gen.module import load_f0gen
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = load_f0gen(ckpt, device='cpu')
    model.ema.copy_to(model.net)
    save(dict(state_dict={f'net.{k}': v for k, v in model.net.state_dict().items()}, f0gen_config=dict(state['hyper_parameters']),
              global_step=int(state.get('global_step', 0)), source=ckpt), out)
    again = load_f0gen(out, device='cpu')
    assert all(torch.equal(a, b) for a, b in zip(again.net.state_dict().values(), model.net.state_dict().values()))
    return f'step {model.step:,}, objective {model.objective}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('kind', choices=['vae', 'stage2', 'f0gen'])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run', help='run directory: export its newest complete checkpoint')
    source.add_argument('--ckpt', help='checkpoint to export')
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default='configs/tcsinger2.yaml', help='vae: stage-2 config whose first_stage_config must fit')
    args = parser.parse_args()

    ckpt = args.ckpt or newest_checkpoint(args.run)
    if args.kind == 'vae':
        what = export_vae(ckpt, args.out, args.config)
    elif args.kind == 'stage2':
        what = export_stage2(ckpt, args.out, args.run or run_of(ckpt))
    else:
        what = export_f0gen(ckpt, args.out)
    print(f'wrote {args.out} ({os.path.getsize(args.out) / 2 ** 20:,.0f} MB): {what}, from {ckpt}')


if __name__ == '__main__':
    main()
