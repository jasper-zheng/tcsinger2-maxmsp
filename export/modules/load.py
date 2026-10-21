"""The parts of TCSinger2 as plain nn.Modules (EWP0), built without Lightning or instantiate_from_config, so that
they load in the ExecuTorch env (envs/et151: torch 2.14, transformers 5) as in the repo's .venv.

    from export.modules.load import load_all
    parts = load_all()   # dit, vae, t5, vocoder, f0net, scale_factor; eval mode, CPU, fp32

Weights:
  dit      the stage-2 TCSinger2 with the EMA weights of useful_ckpts/tcsinger2_gtsinger.ckpt (LitEma names)
  vae      AutoencoderKL of useful_ckpts/vae_gtsinger.ckpt without its encoder: decode(z) and pred_duration(...)
  t5       the flan-t5-large encoder (useful_ckpts/flan-t5-large)
  vocoder  the HiFi-GAN NSF generator of useful_ckpts/hifigan, weight norm removed
  f0net    the F0 generator network (WP10) of useful_ckpts/f0gen_gtsinger.ckpt (an export: EMA weights as its own)
  scale_factor  the latent scale of stage 2: the VAE decodes latent / scale_factor

Modules that the model code imports but these parts never use are stubbed while it is imported, when they are
missing (see _stubs): pytorch_lightning (AutoencoderKL subclasses LightningModule) and librosa (the vocoder
package's wrapper). The caption encoder module (ldm.modules.encoders) is not used: T5 comes from transformers.
"""
import contextlib
import os
import sys
import types

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STAGE2_CKPT = 'useful_ckpts/tcsinger2_gtsinger.ckpt'
STAGE2_CONFIG = 'configs/tcsinger2.yaml'
VAE_CKPT = 'useful_ckpts/vae_gtsinger.ckpt'
T5_DIR = 'useful_ckpts/flan-t5-large'
VOCODER_DIR = 'useful_ckpts/hifigan'
F0GEN_CKPT = 'useful_ckpts/f0gen_gtsinger.ckpt'
CAPTION_TOKENS = 80


@contextlib.contextmanager
def _stubs(*names):
    """Stub modules for imports that the loaded parts never use, only where the real package is missing, and only
    while the model code is imported (transformers looks for librosa itself and must not find a stub)."""
    added = []
    for name in names:
        try:
            __import__(name)
        except ImportError:
            mod = types.ModuleType(name)
            if name == 'pytorch_lightning':
                mod.LightningModule = nn.Module
                mod.callbacks = types.ModuleType('pytorch_lightning.callbacks')
                mod.callbacks.Callback = object
                sys.modules['pytorch_lightning.callbacks'] = mod.callbacks
                added.append('pytorch_lightning.callbacks')
            sys.modules[name] = mod
            added.append(name)
    try:
        yield
    finally:
        for name in added:
            sys.modules.pop(name, None)


def _path(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _config(config=STAGE2_CONFIG):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(_path(config)).model.params, resolve=True)


def _load(path):
    # weights_only=False: some files hold OmegaConf objects or numpy values
    return torch.load(_path(path), map_location='cpu', weights_only=False)


def load_dit(ckpt=STAGE2_CKPT, config=STAGE2_CONFIG):
    """The stage-2 transformer with its EMA weights. Returns (dit, scale_factor, global_step)."""
    with _stubs('pytorch_lightning'):
        from ldm.modules.diffusionmodules.tcsinger2 import TCSinger2
    state = _load(ckpt)
    sd = state['state_dict']
    dit = TCSinger2(**_config(config)['unet_config']['params'])
    prefix = 'model.diffusion_model.'
    dit.load_state_dict({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}, strict=True)
    # LitEma(self.model) keeps each parameter of the wrapper under its name without dots
    for name, p in dit.named_parameters():
        p.data.copy_(sd['model_ema.' + ('diffusion_model.' + name).replace('.', '')])
    step = int(state.get('global_step', 0))
    dit.set_step(step)
    return dit.eval(), float(sd['scale_factor']), step


def load_vae(ckpt=VAE_CKPT, config=STAGE2_CONFIG, encoder=False):
    """AutoencoderKL with the decoder, the duration predictor and its encoders; the VAE encoder only with
    encoder=True (for the blank-prompt latent, export.modules.vae.blank_prompt_latent)."""
    with _stubs('pytorch_lightning'):
        from ldm.models.autoencoder1d import AutoencoderKL
    params = dict(_config(config)['first_stage_config']['params'])
    params.pop('ckpt_path', None)
    params.pop('monitor', None)
    vae = AutoencoderKL(**params)
    sd = _load(ckpt)['state_dict']
    del vae.loss
    if not encoder:
        sd = {k: v for k, v in sd.items() if not k.startswith(('encoder.', 'quant_conv.'))}
        del vae.encoder, vae.quant_conv
    vae.load_state_dict(sd, strict=True)
    return vae.eval()


def load_t5(path=T5_DIR):
    """The flan-t5-large encoder (fp32). Called as t5(input_ids=ids).last_hidden_state, without a mask, as in training."""
    from transformers import T5EncoderModel
    return T5EncoderModel.from_pretrained(_path(path), torch_dtype=torch.float32).eval()


def tokenize(captions, path=T5_DIR, max_length=CAPTION_TOKENS):
    """input_ids [B, 80] int64 from tokenizer.json (the HF fast tokenizer, as neural.tokenizer loads it): truncated
    to 80 and right-padded with <pad> = 0."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(_path(path), 'tokenizer.json'))
    tok.enable_truncation(max_length)
    tok.enable_padding(length=max_length, pad_id=0, pad_token='<pad>')
    return torch.tensor([e.ids for e in tok.encode_batch(list(captions))], dtype=torch.long)


def load_vocoder(base_dir=VOCODER_DIR):
    """The HiFi-GAN NSF generator, called as vocoder(mel [B, 80, T], f0 Hz [B, T]) -> wav [B, 1, T * 256]."""
    import glob
    import re
    from utils.commons.hparams import set_hparams
    with _stubs('librosa'):
        from vocoder.hifigan.modules.hifigan_nsf import HifiGanGenerator
    base_dir = _path(base_dir)
    ckpt = max(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=lambda p: int(re.findall(r'steps_(\d+)\.ckpt', p)[0]))
    config = set_hparams(f'{base_dir}/config.yaml', print_hparams=False, global_hparams=False)
    model = HifiGanGenerator(config)
    model.load_state_dict(_load(ckpt)['state_dict']['model_gen'], strict=True)
    model.remove_weight_norm()
    return model.eval()


def load_f0net(ckpt=F0GEN_CKPT):
    """The F0 generator network with the EMA weights. Returns (net, config of F0Gen)."""
    from ldm.f0gen.model import F0Net
    state = _load(ckpt)
    hp = dict(state.get('hyper_parameters') or state['f0gen_config'])
    net = F0Net(**(hp.get('net_config') or {}))
    sd = state['state_dict']
    if any(k.startswith('ema.') for k in sd):  # a training checkpoint: take the EMA copy
        net.load_state_dict({k[len('net.'):]: v for k, v in sd.items() if k.startswith('net.')}, strict=True)
        for name, p in net.named_parameters():
            p.data.copy_(sd['ema.' + name.replace('.', '')])
    else:
        net.load_state_dict({k[len('net.'):]: v for k, v in sd.items()}, strict=True)
    return net.eval(), hp


def load_all():
    dit, scale_factor, step = load_dit()
    f0net, f0_hp = load_f0net()
    parts = types.SimpleNamespace(dit=dit, scale_factor=scale_factor, step=step, vae=load_vae(), t5=load_t5(),
                                  vocoder=load_vocoder(), f0net=f0net, f0_hp=f0_hp)
    for m in (parts.dit, parts.vae, parts.t5, parts.vocoder, parts.f0net):
        m.requires_grad_(False)
    return parts
