"""Draws aligned phoneme boundaries over the mel for a few clips, to look at the aligner's output (WP9).

Clips come from the aligned corpus (default: OpenSinger, from opensinger_align.json). The lowest-
scoring, the median and the highest-scoring clips are drawn, --n in total.

Usage (from TCSinger2/, PYTHONPATH=.):  python scripts/checks/check_aligner.py --out_dir samples/wp9_align
"""
import argparse, json, os, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

parser = argparse.ArgumentParser()
parser.add_argument('--clips', default='data/meta_ext/opensinger_audio.json')
parser.add_argument('--align', default='data/meta_ext/opensinger_align.json')
parser.add_argument('--data_root', default='/notebooks/data/ext')
parser.add_argument('--mel_root', default='/notebooks/data/ext_processed')
parser.add_argument('--out_dir', default='samples/wp9_align')
parser.add_argument('--n', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

FPS = 48000 / 256
clips = {c['item_name']: c for c in json.load(open(args.clips, encoding='utf-8'))}
al = json.load(open(args.align, encoding='utf-8'))
names = sorted(al, key=lambda k: al[k]['score'])
rng = np.random.default_rng(args.seed)
third = args.n // 3
picks = list(rng.choice(names[:len(names) // 50], third, replace=False)) \
    + list(rng.choice(names[len(names) // 3:2 * len(names) // 3], args.n - 2 * third, replace=False)) \
    + list(rng.choice(names[-len(names) // 5:], third, replace=False))
cjk = [f.fname for f in font_manager.fontManager.ttflist if any(k in f.name for k in ('CJK', 'Noto Sans SC', 'WenQuanYi', 'SimHei'))]
os.makedirs(args.out_dir, exist_ok=True)
for k, name in enumerate(picks):
    clip, a = clips[name], al[name]
    rel = os.path.relpath(clip['wav_fn'], args.data_root)[:-4]
    mel = np.load(os.path.join(args.mel_root, 'melnone48000', rel + '_mel.npy'))
    f0_path = os.path.join(args.mel_root, 'f0', rel + '_f0.npy')
    fig, ax = plt.subplots(figsize=(max(8, mel.shape[1] / FPS * 2.2), 4))
    ax.imshow(mel, origin='lower', aspect='auto', extent=(0, mel.shape[1] / FPS, 0, 80), vmin=-5, vmax=1.5, cmap='magma')
    if os.path.exists(f0_path):
        f0 = np.load(f0_path)[:mel.shape[1]]
        ax.plot(np.arange(len(f0)) / FPS, np.where(f0 > 0, 12 * np.log2(np.maximum(f0, 1) / 55), np.nan), color='cyan', lw=1)
    t = 0.0
    for ph, n in zip(a['ph'], a['frames']):
        rest = ph in ('<AP>', '<SP>')
        ax.axvline(t, color='white', lw=0.6)
        ax.text(t + n / FPS / 2, 76 if not rest else 70, ph.replace('_zh', '').strip('<>'), color='yellow' if rest else 'white',
                ha='center', va='top', fontsize=8)
        t += n / FPS
    title = f"{name}   score {a['score']:.2f}   " + ''.join(clip['txt'])
    ax.set_title(title, fontproperties=font_manager.FontProperties(fname=cjk[0]) if cjk else None, fontsize=9)
    ax.set_xlabel('s (cyan: F0, semitones above 55 Hz)')
    fig.tight_layout()
    stem = os.path.join(args.out_dir, f"{k:02d}_score{a['score']:.2f}".replace('-', 'm'))
    fig.savefig(stem + '.png', dpi=110)
    plt.close(fig)
    shutil.copy(clip['wav_fn'], stem + '.wav')
    print(stem + '.png', '|', ' '.join(p.replace('_zh', '') for p in a['ph']))
