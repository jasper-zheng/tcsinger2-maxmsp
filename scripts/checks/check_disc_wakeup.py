"""Does the discriminator of a VAE resume checkpoint (WP4) start learning when the adversarial loss is switched on?

Before disc_start the discriminator is updated with the R1 penalty alone, which flattens it towards a
constant output. This trains the checkpoint's discriminator (with its optimizer state) on real and
reconstructed training mels with the adversarial loss on and the VAE frozen, next to a freshly
initialised discriminator as a control, and prints both every --every batches:

  real / fake   mean logit on real and reconstructed mels (the mse loss wants 1 and 0)
  d_weight      the adaptive weight the generator pass would use (clamped at 1e4 x disc_weight)
  adv/rec       gradient norm of the weighted adversarial term over that of the reconstruction term,
                at the decoder's last layer (disc_weight x disc_factor when d_weight is not clamped)

Nothing is written. Usage: python scripts/checks/check_disc_wakeup.py --ckpt logs/<run>/checkpoints/step=XXXXXXXX.ckpt
"""
import argparse
import copy

import torch
from omegaconf import OmegaConf
from taming.modules.discriminator.model import weights_init

from ldm.util import instantiate_from_config


def generator_side(model, x, global_step):
    """d_weight and the adversarial / reconstruction gradient ratio of a generator pass (no update)."""
    loss, last = model.loss, model.get_last_layer()
    rec, post = model(x)
    _, log = loss(x, rec, post, 0, global_step, last_layer=last, split='train')
    nll = (x - rec).abs().sum() / x.shape[0]
    g_loss = -loss.discriminator(rec.unsqueeze(1)).mean()
    nll_g = torch.autograd.grad(nll, last, retain_graph=True)[0].norm()
    adv_g = torch.autograd.grad(g_loss, last)[0].norm()
    d_weight = log['train/d_weight'].item()
    return d_weight, (d_weight * loss.disc_factor * adv_g / nll_g).item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--config', default='configs/ae_singing.yaml')
    parser.add_argument('--batches', type=int, default=400)
    parser.add_argument('--every', type=int, default=25)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.data.params.num_workers = args.num_workers
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model = instantiate_from_config(config.model)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model = model.cuda().train()
    lr = config.data.params.batch_size * config.model.base_learning_rate
    on = model.loss.discriminator_iter_start + 1  # a global step with the adversarial loss on
    print(f'checkpoint at batch {ckpt["global_step"] // 2}, lr {lr:.2e}')

    discs, opts = {}, {}
    discs['checkpoint'] = model.loss.discriminator
    opts['checkpoint'] = torch.optim.Adam(discs['checkpoint'].parameters(), lr=lr, betas=(0.5, 0.9))
    opts['checkpoint'].load_state_dict(ckpt['optimizer_states'][1])
    torch.manual_seed(0)
    discs['fresh'] = copy.deepcopy(discs['checkpoint'])
    for m in discs['fresh'].modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    discs['fresh'].apply(weights_init)
    opts['fresh'] = torch.optim.Adam(discs['fresh'].parameters(), lr=lr, betas=(0.5, 0.9))
    del ckpt

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print(f'{"batch":>5}' + ''.join(f' | {name:>10}: {"real":>8} {"fake":>8} {"d_weight":>8} {"adv/rec":>8}' for name in discs))
    for i, batch in enumerate(data.train_dataloader()):
        x = model.get_input(batch, model.image_key)[0].cuda()
        with torch.no_grad():
            rec, post = model(x)
        line = f'{i:5d}'
        for name, disc in discs.items():
            model.loss.discriminator = disc
            report = i % args.every == 0 or i == args.batches
            if report:  # before the update, so row 0 is the state the checkpoint is in
                d_weight, ratio = generator_side(model, x, on)
            else:
                disc(rec.unsqueeze(1))  # the generator pass of training also updates the batch-norm statistics
            d_loss, log = model.loss(x, rec, post, 1, on, split='train')
            opts[name].zero_grad()
            d_loss.backward()
            opts[name].step()
            if report:
                line += f' | {"":>10}  {log["train/logits_real"].item():8.4f} {log["train/logits_fake"].item():8.4f} {d_weight:8.1f} {ratio:8.2e}'
        if i % args.every == 0 or i == args.batches:
            print(line, flush=True)
        if i == args.batches:
            break
