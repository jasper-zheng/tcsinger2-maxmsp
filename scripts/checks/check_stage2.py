"""Checks of the stage-2 model and its training set-up (WP6).

  python scripts/checks/check_stage2.py modes               the three F0 modes of the model
  python scripts/checks/check_stage2.py sampler             frame-budget batches and the share of the extra corpora
  python scripts/checks/check_stage2.py overfit --run DIR   a run on 64 clips (data.params.max_items=64): a given F0 is followed
  python scripts/checks/check_stage2.py resume --run DIR    a run that was resumed: counters and schedules continue
"""
import argparse
import glob
import os

import numpy as np
import torch

from ldm.stage2_samples import FROZEN, Synthesizer, build_dataset, load_stage2

parser = argparse.ArgumentParser()
parser.add_argument('what', choices=['modes', 'sampler', 'overfit', 'resume'])
parser.add_argument('--run', help='run directory (logs/<date>_tcsinger2)')
parser.add_argument('--config', default='configs/tcsinger2.yaml')
args = parser.parse_args()


def newest_ckpt(run):
    return max(glob.glob(os.path.join(run, 'checkpoints', '*.ckpt')), key=os.path.getmtime)


if args.what == 'modes':
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    torch.manual_seed(0)
    model = instantiate_from_config(OmegaConf.load(args.config).model).cuda().eval()
    model.scale_factor.fill_(0.12)
    net = model.model.diffusion_model
    torch.nn.init.normal_(net.final_layer.linear.weight, std=0.02)  # zero at the start of training; the output would be 0
    for block in net.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
    ds = build_dataset('valid', args.config)
    batch = ds.collater([ds[i] for i in ds.epoch_batches(0, 8, False)[40][:3]])
    src = batch['caption']
    acoustic = {k: v.cuda() for k, v in src['acoustic'].items()}
    acoustic['prompt_embed'] = model.get_first_stage_encoding(model.encode_first_stage(acoustic['prompt'])).detach()
    x = torch.randn(3, 20, acoustic['ph'].shape[-1] // 4, device='cuda')
    t = torch.tensor([100, 500, 900], device='cuda')

    def run(mode):
        cond = model.get_learned_conditioning(dict(caption=list(src['caption']), acoustic=acoustic, name=src['name'], infer=mode != 'train', f0_mode=mode))
        with torch.no_grad():
            return model.apply_model(x, t, cond)

    out = {m: run(m) for m in ('train', 'given', 'predict')}
    for m, (v, aux, f0) in out.items():
        print(f"{m:8s} output {tuple(v.shape)} std {v.std():.3f} | losses lb {float(aux['lb']):.3f} f0 {float(aux['f0']):.3f} uv {float(aux['uv']):.3f} | "
              f"F0 used: voiced {float((f0 > 0).float().mean()):.2f}, median {float(f0[f0 > 0].median()) if (f0 > 0).any() else 0:.0f} Hz")
    assert float(out['train'][1]['f0']) > 0 and float(out['train'][1]['uv']) > 0
    assert float(out['given'][1]['f0']) == 0 and float(out['predict'][1]['f0']) == 0
    assert torch.allclose(out['train'][0], out['given'][0], atol=1e-5), 'train and given condition on the same F0'
    assert torch.equal(out['given'][2], acoustic['f0']), 'given mode returns the supplied F0'
    assert not torch.allclose(out['given'][0], out['predict'][0], atol=1e-4), 'predict mode conditions on another F0'
    with torch.no_grad():  # another predictor: given mode must not move, predict mode must
        net.f0_regressor.bias += 1.0
        net.uv_classifier.weight.mul_(-1)
    again = {m: run(m) for m in ('given', 'predict')}
    assert torch.equal(again['given'][0], out['given'][0]), 'given mode depends on the F0 predictor'
    assert not torch.allclose(again['predict'][0], out['predict'][0], atol=1e-4)
    for step, (w, tau) in {0: (1.0, 2.0), 50000: (0.01, 1.15), 100000: (0.01, 0.3), 300000: (0.01, 0.3)}.items():
        net.set_step(step)
        assert abs(net.loss_w - w) < 1e-3 and abs(net.temperature - tau) < 1e-6 and abs(net.blocks[0].feed_forward.temperature - tau) < 1e-6, step
    a, b = run('given')[0], run('given')[0]
    assert torch.equal(a, b), 'routing at inference is not deterministic'
    print('modes: ok (given = train conditioning; given ignores the predictor; schedules follow the step; inference is deterministic)')

elif args.what == 'sampler':
    ds = build_dataset('train', args.config)
    main = set(np.flatnonzero(~ds.is_ext).tolist())
    subsets = []
    for epoch in (0, 1):
        batches = ds.epoch_batches(epoch, 8, True)
        chosen = [i for b in batches for i in b]
        assert len(chosen) == len(set(chosen)) and main <= set(chosen), 'a main-corpus clip is missing or a clip is repeated'
        share = ds.frames[[i for i in chosen if i not in main]].sum() / ds.frames[chosen].sum()
        cost = max(len(b) * (ds.frames[b].max() + ds.max_prompt_len) for b in batches)
        assert abs(share - ds.ext_share) < 0.005 and cost <= ds.max_tokens, (share, cost)
        subsets.append(set(chosen) - main)
        print(f'epoch {epoch}: extra share {share:.3f}, largest batch {cost} frames (budget {ds.max_tokens}), clips per batch {min(map(len, batches))}-{max(map(len, batches))}')
    overlap = len(subsets[0] & subsets[1]) / len(subsets[0])
    assert 0.5 < overlap < 0.8, overlap
    val = build_dataset('valid', args.config)
    vb = val.epoch_batches(0, 8, False)
    assert sorted(i for b in vb for i in b) == list(range(len(val)))
    same = all(np.array_equal(val[7][k], val[7][k]) for k in ('prompt', 'ph', 'ep_pitches', 'mel2ph'))
    assert same, 'validation items are not repeatable'
    print(f'sampler: ok (subsets of two epochs share {overlap:.0%} of their extra clips; validation: {len(val)} clips in {len(vb)} batches, repeatable)')

elif args.what == 'overfit':
    cfg = glob.glob(os.path.join(args.run, 'configs', '*-project.yaml'))[0]
    ckpt = newest_ckpt(args.run)
    model, step = load_stage2(ckpt, cfg)
    ds = build_dataset('train', cfg, max_items=64)
    synth = Synthesizer()
    rows = []
    with model.ema_scope():
        pass  # EMA lags far behind in a short run; the raw weights are measured
    for idx in range(6):
        for mode in ('given', 'predict'):
            _, row, _ = synth.clip(model, ds, idx, mode)
            rows.append(row)
    given = [r for r in rows if r['mode'] == 'given']
    pred = [r for r in rows if r['mode'] == 'predict']
    ffe = float(np.mean([r['ffe_out_vs_f0'] for r in given]))
    print(f"step {step}, 6 training clips | given F0: output F0 frame error {ffe:.3f}, mel L1 {np.mean([r['mel_l1'] for r in given]):.3f} | "
          f"notes only: predicted against sung {np.mean([r['ffe_pred_vs_sung'] for r in pred]):.3f}, output against predicted "
          f"{np.mean([r['ffe_out_vs_pred'] for r in pred]):.3f}, notes within a semitone {np.nanmean([r['notes_within_1'] for r in pred]):.2f}, "
          f"mel L1 {np.mean([r['mel_l1'] for r in pred]):.3f}")
    assert ffe < 0.10, 'the output does not follow the given F0'
    print('overfit: ok')

elif args.what == 'resume':
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = sorted(glob.glob(os.path.join(args.run, '**', 'events.out.tfevents.*'), recursive=True), key=os.path.getmtime)
    assert len(events) >= 2, 'the run has one session only'
    series = {}
    for path in events:
        acc = EventAccumulator(path, size_guidance={'scalars': 0})
        acc.Reload()
        for tag in ('lr-AdamW', 'train/gumbel_temperature', 'train/lb_weight', 'train/loss_simple_step'):
            if tag in acc.Tags()['scalars']:
                series.setdefault(tag, []).append([(e.step, e.value) for e in acc.Scalars(tag)])
    for tag, sessions in series.items():
        a, b = sessions[-2][-1], sessions[-1][0]
        print(f'{tag:28s} last of session {len(sessions) - 1}: step {a[0]} value {a[1]:.6g} | first of session {len(sessions)}: step {b[0]} value {b[1]:.6g}')
        assert 0 <= b[0] - a[0] <= 1100, 'the step counter did not continue'
        if tag != 'train/loss_simple_step':
            assert abs(b[1] - a[1]) <= 0.05 * abs(a[1]) + 1e-9, f'{tag} jumped at the resume'
    state = torch.load(newest_ckpt(args.run), map_location='cpu')
    keys = state['state_dict'].keys()
    assert not [k for k in keys if k.startswith(FROZEN)] and any(k.startswith('model_ema.') for k in keys)
    print(f"checkpoint: step {state['global_step']}, EMA updates {int(state['state_dict']['model_ema.num_updates'])}, "
          f"scale_factor {float(state['state_dict']['scale_factor']):.4f}, no frozen weights inside")
    assert 0 <= state['global_step'] - int(state['state_dict']['model_ema.num_updates']) <= 1
    print('resume: ok')
