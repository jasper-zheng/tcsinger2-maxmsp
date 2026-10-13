"""Loss curves of a VAE run (WP4), from the TensorBoard events of all its sessions.

Writes <run>/losses.csv (tag, batch, value) and <run>/losses.png. The x axis is in batches
(TensorBoard's step for this run; Lightning's global step is twice that).

Usage: python scripts/plot_losses.py logs/<run>
"""
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
TRAIN, VALID, REAL, FAKE = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
SMOOTH = 20  # logged points (x 50 batches) in the rolling mean of training curves
# (title, [(tag, label, color, is_validation)])
PANELS = [
    ('Reconstruction L1 (log10 mel)', [('train/rec_loss', 'train', TRAIN, False), ('val/rec_loss', 'validation', VALID, True)]),
    ('KL loss', [('train/kl_loss', 'train', TRAIN, False), ('val/kl_loss', 'validation', VALID, True)]),
    ('Duration loss', [('durloss_step', 'train', TRAIN, False), ('val/dur_loss', 'validation', VALID, True)]),
    ('Generator loss', [('train/g_loss', 'train', TRAIN, False), ('val/g_loss', 'validation', VALID, True)]),
    ('Discriminator loss', [('train/disc_loss', 'train', TRAIN, False), ('val/disc_loss', 'validation', VALID, True)]),
    ('Adversarial weight (d_weight)', [('train/d_weight', 'train', TRAIN, False)]),
    ('Discriminator logits (train)', [('train/logits_real', 'real', REAL, False), ('train/logits_fake', 'reconstructed', FAKE, False)]),
]


def read_events(run):
    rows = []
    for path in sorted(glob.glob(os.path.join(run, 'tensorboard', 'version_*', 'events.*')), key=os.path.getmtime):
        acc = EventAccumulator(path, size_guidance={'scalars': 0})
        acc.Reload()
        for tag in acc.Tags()['scalars']:
            rows += [(tag, e.step, e.value) for e in acc.Scalars(tag)]
    df = pd.DataFrame(rows, columns=['tag', 'batch', 'value'])
    return df.drop_duplicates(['tag', 'batch'], keep='last').sort_values(['tag', 'batch'])  # a resumed session repeats a few steps


if __name__ == '__main__':
    run = sys.argv[1].rstrip('/')
    df = read_events(run)
    df.to_csv(os.path.join(run, 'losses.csv'), index=False)
    cfg = OmegaConf.merge(*[OmegaConf.load(p) for p in sorted(glob.glob(os.path.join(run, 'configs', '*-project.yaml')))])
    disc_start = cfg.model.params.lossconfig.params.disc_start // 2
    last = int(df['batch'].max())

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': AXIS, 'axes.linewidth': 1, 'text.color': INK, 'axes.labelcolor': INK2,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2})
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), facecolor=SURFACE)
    for ax, (title, series) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        lo, hi, drawn = [], [], 0
        for tag, label, color, is_val in series:
            s = df[df['tag'] == tag]
            if s.empty:
                continue
            x, y = s['batch'].values, s['value']
            if is_val:
                ax.plot(x, y.values, color=color, lw=2, marker='o', ms=6, mec=SURFACE, mew=1.5, label=label, solid_capstyle='round')
            else:
                y = y.rolling(SMOOTH, min_periods=1).mean()
                ax.plot(x, y.values, color=color, lw=2, label=label, solid_capstyle='round', solid_joinstyle='round')
            drawn += 1
            settled = y.values[len(y) // 20:]  # the first 5% is the start-up transient; keep it from setting the scale
            lo.append(settled.min()); hi.append(settled.max())
        if lo and max(hi) > min(lo):
            pad = 0.08 * (max(hi) - min(lo))
            ax.set_ylim(min(lo) - pad, max(hi) + pad)
        if last >= disc_start:
            ax.axvline(disc_start, color=AXIS, lw=1)
            ax.annotate('adversarial loss on', (disc_start, 1), xycoords=('data', 'axes fraction'), xytext=(3, -3),
                        textcoords='offset points', va='top', color=MUTED, fontsize=8)
        ax.set_title(title, loc='left', fontsize=10, color=INK)
        ax.grid(axis='y', color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(length=0)
        ax.xaxis.set_major_formatter(lambda v, _: f'{v:,.0f}')
        ax.margins(x=0.02)
        if drawn > 1:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='upper right')
    for ax in axes[1]:
        ax.set_xlabel('batches')
    note = axes.flat[-1]
    note.axis('off')
    note.text(0, 0.95, f'{os.path.basename(run)}\n\n{last:,} batches so far\nadversarial loss from batch {disc_start:,}\n\n'
              f'Training curves: mean of the last {SMOOTH} logged\nvalues ({SMOOTH * 50:,} batches).\nValidation: once per epoch, fixed crops.\n\n'
              'Each y axis leaves out the first 5% of the run.', va='top', color=INK2, fontsize=9, linespacing=1.4)
    fig.tight_layout()
    fig.savefig(os.path.join(run, 'losses.png'), dpi=110, facecolor=SURFACE)
    print('wrote', os.path.join(run, 'losses.csv'), 'and losses.png;', last, 'batches')
    final = df.groupby('tag').tail(1).set_index('tag')['value']
    print({t: round(float(final[t]), 4) for t in ('train/rec_loss', 'val/rec_loss', 'train/kl_loss', 'val/dur_loss') if t in final})
