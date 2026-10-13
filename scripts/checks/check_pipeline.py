import sys, time, torch, numpy as np
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.data.phone_set import NUM_PH, PH_PAD_ID
n_batches = int(sys.argv[1])
spec_dirs = sys.argv[2] if len(sys.argv) > 2 else None  # e.g. data/sing,data/sing_ext instead of the configs' manifests
for cfg_path in ('configs/ae_singing.yaml', 'configs/tcsinger2.yaml'):
    cfg = OmegaConf.load(cfg_path); cfg.data.params.num_workers = 8
    if spec_dirs:
        cfg.data.params['main_spec_dir_path' if 'main_spec_dir_path' in cfg.data.params else 'spec_dir_path'] = spec_dirs
    data = instantiate_from_config(cfg.data); data.prepare_data(); data.setup()
    for name, loader in (('train', data._train_dataloader()), ('val', data._val_dataloader())):
        t0 = time.time(); n = 0; frames = 0; shapes = None
        for batch in loader:
            x = batch['image']; assert torch.isfinite(x).all()
            if 'caption' in batch and isinstance(batch['caption'], dict):
                ac = batch['caption']['acoustic']
                assert ac['ph'].max() <= PH_PAD_ID and ac['ph'].min() >= 0 and ac['midi'].max() <= 99
                assert ac['f0'].shape[-1] == x.shape[-1] == ac['ph'].shape[-1] == ac['mel2ph'].shape[-1], (ac['f0'].shape, x.shape, ac['ph'].shape)
                assert x.shape[-1] % 8 == 0
                shapes = dict(image=tuple(x.shape), prompt=tuple(ac['prompt'].shape), f0=tuple(ac['f0'].shape), caption=batch['caption']['caption'][0][:90])
            else:
                assert batch['ph'].max() <= PH_PAD_ID and batch['f0'].shape[-1] == x.shape[-1]
                shapes = dict(image=tuple(x.shape), ph=tuple(batch['ph'].shape), f0=tuple(batch['f0'].shape))
            n += 1; frames += x.shape[0] * x.shape[-1]
            if n >= n_batches: break
        print(f"[{cfg_path.split('/')[-1]} {name}] {n} batches of {len(loader)} ok in {time.time()-t0:.0f}s, {frames/n:.0f} frames/batch, last: {shapes}")
