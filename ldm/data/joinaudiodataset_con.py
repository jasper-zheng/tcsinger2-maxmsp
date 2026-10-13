import ast
import glob
import logging
import sys

import numpy as np
import pandas as pd
import torch

from ldm.data.phone_set import ph2id, PH_PAD_ID, MIDI_PAD_ID

logger = logging.getLogger(f"main.{__name__}")

sys.path.insert(0, '.')  # nopep8

SR, HOP = 48000, 256
NUM_NEG_COLS = 4


class JoinManifestSpecs(torch.utils.data.Dataset):
    """Singing / speech / content triplets for the Custom Audio Encoder (contrastive training).

    Expected manifest columns (preprocess/build_con_manifest.py writes them)
    -------------------------
        - mel_path                : singing mel-spectrogram .npy, [n_mels, T]
        - speech_mel_path         : mel of the *corresponding* speech clip (same singer and lyrics)
        - pair_id                 : id of that speech clip; singing clips that share it are positives of each other
        - singing_diff{i}_path    : mel of a *different* singing clip, i = 1..4 (1, 2: same singer; 3, 4: another singer)
        - ph, ph_durs, ep_pitches : phonemes with durations and note pitches, for the frame-level content arrays
        - caption, split

    Every clip is cut to a fixed number of frames (spec_crop_len for singing, speech_crop_len for speech): a random
    window in training, the middle one otherwise. A shorter clip is repeated, never padded with the mel floor,
    because the VAE encoder was trained that way and shifts its output on padded input. Speech and singing are not
    time-aligned, so the two windows are independent.

    Each item carries num_neg of the four different singing clips: drawn at random in training, otherwise one of
    the same singer and one of another singer.
    """

    def __init__(self, split, spec_dir_path, mel_num=None, spec_crop_len=1000, speech_crop_len=750, num_neg=2,
                 max_items=0, **kwargs):
        super().__init__()
        self.split = split
        self.sing_len = spec_crop_len
        self.speech_len = speech_crop_len
        self.num_neg = num_neg
        self.mel_num = mel_num

        manifest_files = []
        for dir_path in spec_dir_path.split(','):
            manifest_files += glob.glob(f"{dir_path}/**/*.tsv", recursive=True)
        if not manifest_files:
            raise FileNotFoundError(f"No manifest TSV files found under: {spec_dir_path}")
        df = pd.concat([pd.read_csv(fp, sep='\t', low_memory=False) for fp in manifest_files], ignore_index=True)

        if split == 'train':
            self.dataset = df[df['split'] == 'train']
        elif split in {'valid', 'val'}:
            self.dataset = df[df['split'] == 'valid']
        else:
            raise ValueError(f"Unknown split '{split}'")
        if max_items:  # a small fixed subset, for the overfit check
            self.dataset = self.dataset.sample(max_items, random_state=0)
        self.dataset = self.dataset.reset_index(drop=True)
        print(f'{split} dataset len:', len(self.dataset))

    def _window(self, length, target_len):
        """Frame indices of the window of a clip of `length` frames (wrapping around if the clip is shorter)."""
        if length <= target_len:
            start = 0
        elif self.split == 'train':
            start = np.random.randint(length - target_len + 1)
        else:
            start = (length - target_len) // 2
        return (start + np.arange(target_len)) % length

    def _content(self, row, n_frames):
        """Phoneme and note id of every frame of the clip; frames beyond the annotation are padding."""
        ph = ast.literal_eval(row['ph'])
        ends = np.round(np.cumsum(ast.literal_eval(row['ph_durs'])) * SR / HOP).astype(int)
        piece = np.searchsorted(ends, np.arange(n_frames), side='right')  # index of the piece each frame lies in
        inside = piece < len(ph)
        piece = np.minimum(piece, len(ph) - 1)
        ph_array = np.where(inside, np.array([ph2id[p] for p in ph])[piece], PH_PAD_ID)
        pitch_array = np.where(inside, np.array(ast.literal_eval(row['ep_pitches']))[piece], MIDI_PAD_ID)
        return ph_array.astype(np.int64), pitch_array.astype(np.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]

        spec = np.load(row['mel_path'])  # [n_mels, T]
        ph_array, pitch_array = self._content(row, spec.shape[1])
        win = self._window(spec.shape[1], self.sing_len)

        speech = np.load(row['speech_mel_path'])
        speech = speech[:, self._window(speech.shape[1], self.speech_len)]

        if self.split == 'train':
            cols = np.random.choice(NUM_NEG_COLS, self.num_neg, replace=False) + 1
        else:
            cols = [1, 3, 2, 4][:self.num_neg]
        neg = []
        for i in cols:
            m = np.load(row[f'singing_diff{i}_path'])
            neg.append(m[:, self._window(m.shape[1], self.sing_len)])

        return {
            'image': spec[:, win].astype(np.float32),        # singing mel
            'spec_speech': speech.astype(np.float32),
            'neg_singing': np.stack(neg).astype(np.float32),  # [num_neg, n_mels, T]
            'ph_array': ph_array[win],
            'pitch_array': pitch_array[win],
            'caption': row['caption'] if isinstance(row['caption'], str) else '',
            'pair_id': int(row['pair_id']),
        }


# Convenience wrappers ---------------------------------------------------------
class JoinSpecsTrain(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg, **kwargs):
        super().__init__('train', **{**specs_dataset_cfg, **kwargs})


class JoinSpecsValidation(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg, **kwargs):
        super().__init__('valid', **{**specs_dataset_cfg, **kwargs})
