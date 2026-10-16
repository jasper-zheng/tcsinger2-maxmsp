"""From a score, a style and optionally an F0 to audio (WP7). scripts/sing.py is the command line for it.

    singer = Singer('useful_ckpts/tcsinger2_gtsinger.ckpt')
    wav, info, curves = singer.sing('examples/twinkle_en.json', style=dict(singer='female alto'), prompt='ref.wav')
    write_outputs('out.wav', wav, info, curves)

Steps: score.py (the JSON), g2p.py (phonemes), timing.py (pieces, lengths, frames), f0_input.py (the F0),
then the batch of the training dataset (collate_items) and CFM.synthesize with the EMA weights, and the
vocoder with the F0 that conditioned the model: the given one, or the one predicted from the notes.
The F0 is given as "f0" in the score, or to sing() as a file (.json, text or a recording) or an array.
Without a given F0, Singer(f0_model=...) generates one with the F0 generator of WP10 (ldm/f0gen) and sings
it like a given one; without f0_model the stage-2 model predicts it.
"""
import contextlib
import io
import json
import os
import warnings as py_warnings
from dataclasses import dataclass

import numpy as np

from ldm.data.captions import build_caption, parse_caption
from ldm.data.phone_set import MIDI_PAD_ID, language_set
from ldm.data.tcsinger2_dataset import collate_items
from ldm.infer import f0_input
from ldm.infer.g2p import phonemize
from ldm.infer.score import Score, load_score, parse_score
from ldm.infer.timing import FPS, DurationPredictor, build_pieces, fill_durations, to_frames

SR, HOP = 48000, 256
MEL_ARGS = dict(audio_sample_rate=SR, audio_num_mel_bins=80, fft_size=1024, win_size=1024, hop_size=HOP, fmin=20, fmax=24000)
MEL_FLOOR = -5.0
MAX_FRAMES = 3760        # spec_crop_len of configs/tcsinger2.yaml
MAX_PROMPT_FRAMES = 1500  # max_prompt_len: prompts were cut to 8 s in training
BLANK_PROMPT_FRAMES = 376  # the blank prompt of training (10% of the clips)
DEFAULT_BLUR = False     # boundary blurring for timings that are not given; see scripts/checks/check_frontend.py
DEFAULT_FIT = 'nucleus'


@dataclass
class Prepared:
    """What the front-end makes of a score."""
    score: Score
    phs: list        # phonemes per word
    sources: list    # where they come from, per word
    pieces: list     # timed pieces
    tracks: dict     # per-frame mel2ph, ph, midi (blurred if asked for) and piece starts
    warnings: list
    blur: bool

    @property
    def n_frames(self):
        return int(self.tracks['starts'][-1])

    def frame_masks(self):
        """Per frame, before any blurring: note pitch (0 in rests), inside a sung word, inside a nucleus."""
        n, starts = self.n_frames, self.tracks['starts']
        midi, sung, nucleus = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        for p, a, b in zip(self.pieces, starts[:-1], starts[1:]):
            midi[a:b], sung[a:b], nucleus[a:b] = p.midi, p.type != 1, p.nucleus
        return midi, sung, nucleus

    def describe(self):
        starts = self.tracks['starts'] / FPS
        words = []
        for wi, (w, phs, src) in enumerate(zip(self.score.words, self.phs, self.sources)):
            idx = [i for i, p in enumerate(self.pieces) if p.word == wi]
            words.append(dict(text=w.text, ph=[p.rsplit('_', 1)[0] if '_' in p else p for p in phs], source=src,
                              start=round(float(starts[idx[0]]), 3), end=round(float(starts[idx[-1] + 1]), 3),
                              notes=[[m, round(d, 4)] for m, d in w.notes], pieces=idx))
        pieces = [dict(ph=p.ph, start=round(float(starts[i]), 3), dur=round(float(starts[i + 1] - starts[i]), 3), midi=p.midi,
                       timing='score' if p.given else 'predicted') for i, p in enumerate(self.pieces)]
        return dict(name=self.score.name, language=self.score.language, seconds=round(self.n_frames / FPS, 3), n_frames=self.n_frames,
                    blur=self.blur, words=words, pieces=pieces, warnings=list(self.warnings))


def prepare(score, predictor, blur=DEFAULT_BLUR, fit=DEFAULT_FIT, use_lexicon=True):
    """Score (object, dict or path) -> Prepared. `predictor`: timing.DurationPredictor."""
    if isinstance(score, str):
        score = load_score(score)
    elif isinstance(score, dict):
        score = parse_score(score)
    phs, sources, warnings = phonemize(score, use_lexicon)
    pieces = build_pieces(score, phs)
    fill_durations(pieces, predictor, fit)
    tracks = to_frames(pieces, blur)
    if tracks['starts'][-1] > MAX_FRAMES:
        raise ValueError(f"the score gives {tracks['starts'][-1]} frames, more than the {MAX_FRAMES} the model was trained on")
    return Prepared(score, phs, sources, pieces, tracks, warnings, blur)


def make_caption(style=None, caption=None, **fields):
    """The caption string: a raw `caption` (checked), or the fields of `style` overridden by `fields`."""
    if caption is not None:
        return build_caption(**parse_caption(caption))
    values = {**(style or {}), **{k: v for k, v in fields.items() if v is not None}}
    return build_caption(**values)


def prompt_mel(path, mel_net=None, max_frames=MAX_PROMPT_FRAMES):
    """Mel [80, T] of an audio prompt: a mel .npy as preprocessing writes it, or audio (mono, 48 kHz, -23 LUFS and
    the mel of preprocess/mel_spec_48k.py). The first `max_frames` frames are used. Returns (mel, warnings)."""
    import torch
    warnings = []
    if path.lower().endswith('.npy'):
        mel = np.load(path).astype(np.float32)
        if mel.ndim != 2 or mel.shape[0] != 80:
            raise ValueError(f'prompt {path}: expected a mel of shape [80, T], found {mel.shape}')
    else:
        import pyloudnorm as pyln
        import soundfile as sf
        import torchaudio
        from preprocess.NAT_mel import MelNet
        wav, sr = sf.read(path, dtype='float32', always_2d=True)  # not torchaudio.load: from torchaudio 2.9 on it needs torchcodec
        wav = torchaudio.functional.resample(torch.from_numpy(wav.mean(1)), orig_freq=sr, new_freq=SR)
        wav = wav[:(max_frames + 1) * HOP].numpy()
        if len(wav) < SR:
            raise ValueError(f'prompt {path}: {len(wav) / SR:.2f} s of audio; at least 1 s is needed, 3 s or more is better')
        loudness = pyln.Meter(SR).integrated_loudness(wav)
        if np.isfinite(loudness):
            wav = pyln.normalize.loudness(wav, loudness, -23.0)
            peak = np.abs(wav).max()
            wav = wav / peak if peak > 1.0 else wav
        else:
            warnings.append(f'prompt {path} is silent')
        if mel_net is None:
            mel_net = MelNet(MEL_ARGS)
        with torch.no_grad():
            mel = mel_net(wav.astype(np.float32)).squeeze(0).cpu().numpy()
    mel = mel[:, :max_frames]
    if mel.shape[1] < 3 * FPS:
        warnings.append(f'the prompt is only {mel.shape[1] / FPS:.1f} s long')
    return mel, warnings


def default_device():
    """cuda, else mps (Apple GPU), else cpu."""
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def blank_prompt():
    return np.full((80, BLANK_PROMPT_FRAMES), MEL_FLOOR, dtype=np.float32)


def given_f0(prepared, f0=None, f0_fps=None, rmvpe=None):
    """The F0 to sing, fitted to the score's frames, or None when the model is to predict it.

    f0: None = the "f0" of the score if it has one; False = none, whatever the score holds; a file
    (f0_input.load_f0); or Hz per frame at f0_fps (None = 187.5). `rmvpe` is needed for a recording.
    Returns (F0 [n_frames] or None, where it came from, figures of check_f0, warnings)."""
    if f0 is False or (f0 is None and prepared.score.f0 is None):
        return None, 'predicted', None, []
    if f0 is None:
        source, f0 = 'score', f0_input.from_values(prepared.score.f0, prepared.score.f0_fps, 'the "f0" of the score')
    elif isinstance(f0, str):
        source, f0 = f0, f0_input.load_f0(f0, f0_fps, rmvpe)
    else:
        source, f0 = 'array', f0_input.from_values(f0, f0_fps)
    f0, w = f0_input.fit_length(f0, prepared.n_frames)
    stats, w2 = f0_input.check_f0(f0, *prepared.frame_masks())
    return f0, source, stats, ([w] if w else []) + w2


def make_batch(prepared, caption, prompt=None, f0=None):
    """The batch CFM.synthesize takes, built with the dataset's collater."""
    n, t = prepared.n_frames, prepared.tracks
    item = dict(prompt=blank_prompt() if prompt is None else prompt, image=np.full((80, n), MEL_FLOOR, dtype=np.float32),
                caption=caption, f0=(np.zeros(n, dtype=np.float32) if f0 is None else f0.astype(np.float32))[np.newaxis],
                name=prepared.score.name, ph=t['ph'][np.newaxis].astype(np.int32), ep_pitches=t['midi'][np.newaxis].astype(np.int32),
                mel2ph=t['mel2ph'][np.newaxis].astype(np.int32), language=language_set.index(prepared.score.language))
    return collate_items([item], pad_value=MEL_FLOOR, max_len=MAX_FRAMES)


class Singer:
    """The stage-2 model with its vocoder and pitch tracker. device: None = default_device(); the vocoder runs
    on the model's device, the pitch tracker on CUDA if there is one, else on the CPU (MPS is no faster for it)."""

    def __init__(self, ckpt='useful_ckpts/tcsinger2_gtsinger.ckpt', config='configs/tcsinger2.yaml', device=None,
                 vocoder_ckpt='useful_ckpts/hifigan', rmvpe_ckpt='useful_ckpts/rmvpe/model.pt', model=None, step=None,
                 f0_model=None):
        from ldm.stage2_samples import load_stage2
        from vocoder.hifigan.hifigan_nsf import HifiGAN_NSF
        self.ckpt = ckpt
        with contextlib.redirect_stdout(io.StringIO()):  # the constructors print their layer lists
            self.model, self.step = (model, step) if model is not None else load_stage2(ckpt, config, device or default_device())
            self.vocoder = HifiGAN_NSF(vocoder_ckpt, device=self.model.device)
        self.predictor = DurationPredictor(self.model.first_stage_model)
        self.f0_model = f0_model
        if f0_model:
            from ldm.f0gen.module import load_f0gen
            self.f0gen = load_f0gen(f0_model, self.model.device)
        else:
            self.f0gen = None
        self.rmvpe_ckpt, self._rmvpe, self._mel_net = rmvpe_ckpt, None, None

    @property
    def rmvpe(self):
        if self._rmvpe is None:
            from utils.rmvpe import RMVPE
            self._rmvpe = RMVPE(self.rmvpe_ckpt)
        return self._rmvpe

    @property
    def mel_net(self):
        if self._mel_net is None:
            from preprocess.NAT_mel import MelNet
            self._mel_net = MelNet(MEL_ARGS)
            self._mel_net.to(self.model.device)  # works in place
        return self._mel_net

    def prepare(self, score, blur=DEFAULT_BLUR, fit=DEFAULT_FIT, use_lexicon=True):
        return prepare(score, self.predictor, blur, fit, use_lexicon)

    def sing(self, score, style=None, caption=None, prompt=None, f0=None, f0_fps=None, blur=DEFAULT_BLUR, fit=DEFAULT_FIT,
             seed=0, cfg_scale=3.0, steps=25, f0_freeze_step=0, measure=True, f0_steps=32, f0_cfg=1.0):
        """score: path, dict, Score or Prepared. style: dict of caption fields that override the score's;
        caption: a raw caption instead. prompt: audio or mel file, or a mel [80, T]. f0: see given_f0; by default
        the "f0" of the score is sung if it has one. f0_steps, f0_cfg: sampling of the F0 generator, if loaded.
        Returns (wav at 48 kHz, info dict, curves dict)."""
        import torch
        prepared = score if isinstance(score, Prepared) else self.prepare(score, blur, fit)
        info = prepared.describe()
        n = prepared.n_frames
        midi, sung, nucleus = prepared.frame_masks()
        info['caption'] = make_caption(prepared.score.style, caption, **(style or {}))
        prompt_path = prompt if isinstance(prompt, str) else None
        if isinstance(prompt, str):
            info['prompt'] = prompt
            prompt, w = prompt_mel(prompt, self.mel_net)
            info['warnings'] += w
        elif prompt is not None:
            prompt = np.asarray(prompt, dtype=np.float32)[:, :MAX_PROMPT_FRAMES]
        if not info['caption'] and prompt is None:
            info['warnings'].append('neither a caption nor an audio prompt: the model chooses the voice')
        f0, info['f0_source'], stats, w = given_f0(prepared, f0, f0_fps, self.rmvpe if isinstance(f0, str) else None)
        info['warnings'] += w
        if stats is not None:
            info['f0_check'] = stats
        if f0 is None and self.f0gen is not None:
            f0 = self.generate_f0(prepared, prompt_path, seed, f0_steps, f0_cfg)
            info.update(f0_source='f0gen', f0_model=self.f0_model, f0_steps=f0_steps, f0_cfg=f0_cfg)
        mode = 'predict' if f0 is None else 'given'
        info.update(mode=mode, seed=seed, cfg_scale=cfg_scale, steps=steps, ckpt=self.ckpt, ckpt_step=self.step)

        batch = make_batch(prepared, info['caption'], prompt, f0)
        torch.manual_seed(seed)
        with self.model.ema_scope():
            mel, f0_used = self.model.synthesize(batch, f0_mode=mode, cfg_scale=cfg_scale, steps=steps, f0_freeze_step=f0_freeze_step)
        mel, f0_used = mel[0, :, :n].float().cpu().numpy(), f0_used[0, :n].float().cpu().numpy()
        wav = self.vocoder.vocode(mel, f0=f0_used)
        curves = dict(mel=mel, f0_used=f0_used, midi=midi, starts=prepared.tracks['starts'], labels=[p.ph for p in prepared.pieces])
        if measure:
            from ldm.stage2_samples import f0_metrics, notes_within
            f0_out, _ = self.rmvpe.get_pitch(wav.astype(np.float32), SR, HOP, len(wav) // HOP)
            curves['f0_out'] = f0_out
            ffe, voicing, cents = f0_metrics(f0_used, f0_out)
            info['measured'] = dict(ffe_out_vs_f0_used=round(ffe, 4), voicing_err_out=round(voicing, 4), cents_out=round(cents, 1))
            if info['f0_source'] in ('predicted', 'f0gen'):
                note_track = np.where(sung & (midi > 0), midi, MIDI_PAD_ID)
                info['measured']['notes_within_1'] = round(float(notes_within(f0_used, note_track)), 4)
        return wav, info, curves

    def prompt_f0(self, path):
        """F0 of a prompt for the F0 generator: tracked in a recording; for a mel .npy of the processed data, the F0
        file next to it (melnone48000/..._mel.npy -> f0/..._f0.npy); otherwise None (no prompt)."""
        if path is None:
            return None
        if path.lower().endswith('.npy'):
            f0_path = path.replace('/melnone48000/', '/f0/').replace('_mel.npy', '_f0.npy')
            return np.load(f0_path) if f0_path != path and os.path.exists(f0_path) else None
        import soundfile as sf
        wav, sr = sf.read(path, dtype='float32', always_2d=True)
        return f0_input.track_f0(wav.mean(axis=1)[:int((MAX_PROMPT_FRAMES + 1) * HOP * sr / SR)], sr, self.rmvpe)

    def generate_f0(self, prepared, prompt_path=None, seed=0, steps=32, cfg=1.0):
        """An F0 for the score from the F0 generator (WP10), from the unblurred note tracks of its pieces."""
        from ldm.f0gen import features
        from ldm.f0gen.dataset import collate
        n = prepared.n_frames
        tr = features.tracks_from_pieces(prepared.pieces, prepared.tracks['starts'])
        pf0 = self.prompt_f0(prompt_path)
        prompt = features.prompt_features(pf0[:MAX_PROMPT_FRAMES]) if pf0 is not None and (pf0 > 0).any() else np.zeros((2, 0), np.float32)
        zeros = np.zeros(n, dtype=np.float32)
        item = dict(name=prepared.score.name, language=language_set.index(prepared.score.language), f0=zeros,
                    ref=features.reference_pitch(tr['midi']), r=zeros, v=zeros, trust=zeros, ph=tr['ph'], midi=tr['midi'],
                    ntype=tr['ntype'], pos=tr['pos'], onset=tr['onset'], prompt=prompt)
        f0 = self.f0gen.generate(collate([item]), steps=steps, cfg=cfg, seed=seed)
        return f0[0, :n].float().cpu().numpy()


def plot_result(path, info, curves):
    # a Figure without pyplot: no backend is chosen, so a notebook's own is left alone (matplotlib.use('Agg')
    # inside an IPython 9 kernel fails with matplotlib 3.8)
    from matplotlib.figure import Figure
    midi, starts = curves['midi'].astype(float), curves['starts']
    t = np.arange(len(midi)) / FPS
    semi = lambda f: np.where(f > 0, 69 + 12 * np.log2(np.maximum(f, 1e-3) / 440.0), np.nan)
    fig = Figure(figsize=(max(8, t[-1] * 1.6), 3.8))
    ax = fig.subplots()
    ax.plot(t, np.where(midi > 0, midi, np.nan), color='0.6', lw=3, label='notes')
    if 'f0_out' in curves:
        ax.plot(t[:len(curves['f0_out'])], semi(curves['f0_out'][:len(t)]), color='tab:orange', lw=1, label='F0 of the output')
    generated = info.get('f0_source') == 'f0gen'
    ax.plot(t, semi(curves['f0_used']), color='tab:green' if info['mode'] == 'predict' or generated else 'tab:blue', lw=1,
            label='predicted F0' if info['mode'] == 'predict' else 'generated F0' if generated else 'given F0')
    lo, hi = ax.get_ylim()
    for a, b, label in zip(starts[:-1], starts[1:], curves['labels']):
        ax.axvline(a / FPS, color='0.85', lw=0.5, zorder=0)
        if b > a:
            ax.text((a + b) / 2 / FPS, hi, label.rsplit('_', 1)[0] if '_' in label else label.strip('<>'), fontsize=5, ha='center', va='bottom')
    ax.set_ylim(lo, hi)
    ax.set_xlabel('s')
    ax.set_ylabel('MIDI pitch')
    ax.set_title(f"{info['name']} | {info['language']} | {info['caption'] or 'no caption'}", fontsize=8, pad=12)
    ax.legend(fontsize=7, loc='lower right')
    with py_warnings.catch_warnings():  # the default font has no CJK glyphs for a title with such a name
        py_warnings.simplefilter('ignore', UserWarning)
        fig.tight_layout()
        fig.savefig(path, dpi=110)


def write_outputs(out_path, wav, info, curves=None):
    """<out>.wav (48 kHz, 16 bit), <out>_f0.png and <out>.json: what was sung, warnings, figures, and as "f0" (with
    "f0_fps") the F0 that conditioned the model, in the form a score takes it, so that a predicted curve can be
    edited and given back."""
    import soundfile as sf
    base = os.path.splitext(out_path)[0]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(base + '.wav', np.clip(wav, -1, 1), SR, subtype='PCM_16')
    text = json.dumps(info, ensure_ascii=False, indent=1)
    if curves is not None:  # the curve on one line, after everything else
        f0 = json.dumps([round(float(v), 2) for v in curves['f0_used']])
        text = text[:text.rindex('}')].rstrip() + f',\n "f0_fps": {FPS},\n "f0": {f0}\n}}'
    with open(base + '.json', 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    if curves is not None:
        plot_result(base + '_f0.png', info, curves)
