import sys
import numpy as np
import torch
from typing import TypeVar, Optional, Iterator
import logging
import pandas as pd
from ldm.data.joinaudiodataset_anylen import *
from ldm.data.phone_set import language_set, ph2id, PH_PAD_ID, MIDI_PAD_ID
from ldm.data.captions import caption_from_row, sample_caption
import glob
import math
import ast
import random

logger = logging.getLogger(f'main.{__name__}')

sys.path.insert(0, '.')  # nopep8

"""
adapted from joinaudiodataset_struct_sample_anylen.py
"""


SAMPLE_RATE, HOP_SIZE = 48000, 256
BLUR_KEEP = 24  # frames of a phoneme piece that keep their label when the boundaries are blurred


def frame_tracks(ph_ids, ph_durs, ep_pitches, spec_len=None):
    """Per-frame tracks of a clip from its phoneme pieces (ids, seconds, note pitches).

    Returns (mel2ph, ph_array, ep_pitches_array), each [spec_len]: the 1-based piece number (0 = none),
    the phoneme id (PH_PAD_ID = none) and the MIDI pitch (MIDI_PAD_ID = none) of every mel frame.
    spec_len None: as many frames as the pieces fill. Used by the dataset and by ldm/infer."""
    if spec_len is None:
        spec_len = sum(int(dur * SAMPLE_RATE / HOP_SIZE + 0.5) for dur in ph_durs if dur > 0)
    mel2ph = np.zeros((spec_len,), dtype=np.int32)
    current_frame = 0
    for i_ph, dur in enumerate(ph_durs):
        # 如果 dur = -1，表示无效持续时间，可根据需要决定跳过或赋为0
        if dur <= 0:
            continue
        # 持续时间转为mel帧数
        frames = int(dur * SAMPLE_RATE / HOP_SIZE + 0.5)
        end_frame = current_frame + frames
        if end_frame > spec_len:
            # 如果超过mel长度，则截断
            end_frame = spec_len
        # 通常mel2ph从1开始编号音素
        mel2ph[current_frame:end_frame] = i_ph + 1
        current_frame = end_frame
        if current_frame >= spec_len:
            break

    ph_array = np.full((spec_len,), PH_PAD_ID, dtype=np.int32)
    ep_pitches_array = np.full((spec_len,), MIDI_PAD_ID, dtype=np.int32)
    for i in range(spec_len):
        p = mel2ph[i] - 1
        if p >= 0:
            ph_array[i] = ph_ids[p]
            ep_pitches_array[i] = ep_pitches[p]
    return mel2ph, ph_array, ep_pitches_array


def blur_boundaries(mel2ph, ph_array, ep_pitches_array, centre=False):
    """Boundary blurring, in place: of every run of one piece longer than BLUR_KEEP frames only BLUR_KEEP
    frames keep their piece, phoneme and pitch, the rest become padding. The kept stretch lies at a
    random position (training) or in the middle of the run (centre=True, inference)."""
    spec_len = len(mel2ph)

    def mask(start_idx, length):
        if centre:
            seg_start = start_idx + (length - BLUR_KEEP) // 2
        else:
            seg_start = np.random.randint(start_idx, start_idx + length - BLUR_KEEP + 1)
        seg_end = seg_start + BLUR_KEEP
        # mask前后多余的部分
        for mask_range in [range(start_idx, seg_start), range(seg_end, start_idx + length)]:
            mel2ph[mask_range] = 0
            ph_array[mask_range] = PH_PAD_ID
            ep_pitches_array[mask_range] = MIDI_PAD_ID

    ph_val = mel2ph[0] if spec_len > 0 else -1
    start_idx = 0
    length = 1
    i = 1
    while i < spec_len:
        if mel2ph[i] == ph_val and ph_val != PH_PAD_ID:
            length += 1
        else:
            if length > BLUR_KEEP:
                mask(start_idx, length)
            ph_val = mel2ph[i]
            start_idx = i
            length = 1
        i += 1
    # 最后一个连续段处理
    if length > BLUR_KEEP:
        mask(start_idx, length)


def collate_items(inputs, pad_value=-5, min_len=376, max_len=3760, min_factor=8):
    """Batch of items as JoinManifestSpecs.__getitem__ returns them (also built by ldm/infer)."""
    to_dict = {}
    for l in inputs:
        for k, v in l.items():
            if k in to_dict: # image, acoustic, f0, midi, beats, caption, prompt, name
                to_dict[k].append(v)
            else:
                to_dict[k] = [v]

    pad = dict(min_len=min_len, max_len=max_len, min_factor=min_factor)
    to_dict['image'] = collate_1d_or_2d(to_dict['image'], pad_idx=pad_value, **pad)   # B, C, T
    to_dict['prompt'] = collate_1d_or_2d(to_dict['prompt'], pad_idx=pad_value, **pad)   # B, C, T
    to_dict['f0'] = collate_1d_or_2d(to_dict['f0'], pad_idx=0, **pad)   # B, C, T
    to_dict['mel2ph'] = collate_1d_or_2d(to_dict['mel2ph'], pad_idx=0, **pad).long()
    to_dict['ph'] = collate_1d_or_2d(to_dict['ph'], pad_idx=PH_PAD_ID, **pad).long()
    to_dict['ep_pitches'] = collate_1d_or_2d(to_dict['ep_pitches'], pad_idx=MIDI_PAD_ID, **pad).long()

    to_dict['caption'] = {
        'name': to_dict['name'],
        'caption': to_dict['caption'],
        'acoustic':{
        'f0': to_dict['f0'],
        'ph': to_dict['ph'],
        'midi': to_dict['ep_pitches'],
        'mel2ph': to_dict['mel2ph'],
        'prompt': to_dict['prompt'],
        'language': torch.Tensor(to_dict['language']).long()
        },
        'infer': False
    }

    return to_dict


class JoinManifestSpecs(torch.utils.data.Dataset):
    def __init__(self, split, main_spec_dir_path, other_spec_dir_path, mel_num=80, mode='pad', spec_crop_len=1248,
                 pad_value=-5, drop=0, max_tokens=None, ext_share=None, max_prompt_len=None, max_batch_clips=64,
                 sampler_seed=0, max_items=None, **kwargs):
        """max_tokens: frame budget of a batch, clips x (longest clip + max_prompt_len); None = batches of
        batch_size clips. ext_share: share of an epoch's frames that the extra corpora (rows with a
        `dataset` value, WP9) may take; None = all of them. max_prompt_len: prompts are cut to this
        many frames (a random stretch in training)."""
        super().__init__()
        self.split = split
        self.ext_share = ext_share
        self.max_prompt_len = max_prompt_len
        self.max_batch_clips = max_batch_clips
        self.sampler_seed = sampler_seed
        self.max_items = max_items
        self.max_batch_len = spec_crop_len 
        self.min_batch_len = 376
        self.min_factor = 8
        self.mel_num = mel_num
        self.drop = drop
        self.pad_value = pad_value
        self.max_tokens = max_tokens
        assert mode in ['pad', 'tile']
        self.collate_mode = mode
        manifest_files = []
        for dir_path in main_spec_dir_path.split(','):
            manifest_files += glob.glob(f'{dir_path}/*.tsv')
        df_list = [pd.read_csv(manifest, sep='\t') for manifest in manifest_files]
        self.df_main = pd.concat(df_list, ignore_index=True)
        self.language_to_id = {language: idx for idx, language in enumerate(language_set)}

        has_split = 'split' in self.df_main.columns  # manifests built from the GTSinger metadata carry their split
        if split == 'train':
            self.dataset = self.df_main[self.df_main['split'] == 'train'] if has_split else self.df_main.iloc[300:]
        elif split == 'valid' or split == 'val':
            self.dataset = self.df_main[self.df_main['split'] == 'valid'] if has_split else self.df_main.iloc[:300]
        elif split == 'test':
            self.df_main = self.add_name_num(self.df_main)
            self.dataset = self.df_main
        elif has_split and split in set(self.df_main['split']):  # e.g. test_seen, test_unseen: read like validation
            self.dataset = self.df_main[self.df_main['split'] == split]
        else:
            raise ValueError(f'Unknown split {split}')
        if max_items and split == 'train':  # a small fixed subset, for overfitting checks
            self.dataset = self.dataset.sample(max_items, random_state=0)
        self.dataset = self.dataset.reset_index()
        print('dataset len:', len(self.dataset), "drop_rate", self.drop)
        if 'dataset' not in self.dataset.columns:
            self.dataset['dataset'] = np.nan
        self.frames = np.minimum(self.dataset['n_frames'].values, self.max_batch_len) if 'n_frames' in self.dataset.columns \
            else np.minimum((self.dataset['duration'].values * 187.5).astype(int), self.max_batch_len)
        self.is_ext = self.dataset['dataset'].notna().values

    def epoch_batches(self, epoch, batch_size, shuffle):
        """Batches (lists of row numbers) of clips of similar length for one epoch.

        Training with ext_share: every clip of the main corpus and a random subset of the extra
        corpora, drawn anew for each epoch, so that the extra data makes up ext_share of the frames."""
        rng = np.random.RandomState(self.sampler_seed + epoch)
        main, ext = np.flatnonzero(~self.is_ext), np.flatnonzero(self.is_ext)
        if self.split == 'train' and self.ext_share is not None and len(ext) and len(main):
            budget = self.frames[main].sum() * self.ext_share / (1 - self.ext_share)
            ext = rng.permutation(ext)
            ext = ext[:int(np.searchsorted(np.cumsum(self.frames[ext]), budget)) + 1]
        chosen = np.concatenate([main, ext])
        chosen = chosen[np.argsort(self.frames[chosen], kind='stable')]
        batches, batch = [], []
        for i in chosen:  # ascending length, so frames[i] is the longest clip of the batch so far
            if self.max_tokens:
                full = len(batch) >= self.max_batch_clips or \
                       (len(batch) + 1) * (self.frames[i] + (self.max_prompt_len or 0)) > self.max_tokens
            else:
                full = len(batch) >= batch_size
            if batch and full:
                batches.append(batch)
                batch = []
            batch.append(int(i))
        if batch and (self.max_tokens or self.split != 'train'):
            batches.append(batch)
        if self.split == 'train' and self.max_items:  # a small subset is repeated, so that an epoch is not a few steps
            batches = batches * math.ceil(200 / len(batches))
        if shuffle:
            batches = [batches[k] for k in rng.permutation(len(batches))]
        if self.split == 'train':
            total = self.frames[chosen].sum()
            corpus = self.dataset['dataset'].fillna('GTSinger').values[chosen]
            lang = self.dataset['language'].values[chosen]
            share = lambda keys: ', '.join(f'{k} {self.frames[chosen][keys == k].sum() / total:.1%}' for k in sorted(set(keys)))
            print(f'epoch {epoch}: {len(chosen)} clips, {total / 187.5 / 3600:.1f} h, {len(batches)} batches | {share(corpus)} | {share(lang)}')
        return batches

    def add_name_num(self, df):
        """each file may have different caption, we add num to filename to identify each audio-caption pair"""
        name_count_dict = {}
        change = []
        for t in df.itertuples():
            name = getattr(t, 'name')
            if name in name_count_dict:
                name_count_dict[name] += 1
            else:
                name_count_dict[name] = 0
            change.append((t[0], name_count_dict[name]))
        for t in change:
            df.loc[t[0], 'name'] = str(df.loc[t[0], 'name']) + f'_{t[1]}'
        return df

    def ordered_indices(self):
        index2dur = self.dataset[['duration']].sort_values(by='duration')
        return list(index2dur.index)

    def collater(self, inputs):
        assert self.collate_mode == 'pad'
        return collate_items(inputs, pad_value=self.pad_value, min_len=self.min_batch_len, max_len=self.max_batch_len,
                             min_factor=self.min_factor)

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        data = self.dataset.iloc[idx]

        item = {}
        start = -1
        valid_spec = True
        # try:
        spec = np.load(data['mel_path'])  # mel spec [80, T]
        spec_len = spec.shape[1]
        
        # 随机选择
        if spec.shape[1] > self.max_batch_len:
            start=0
            spec = spec[:, start: start + self.max_batch_len]
            spec_len = self.max_batch_len

        # training: a random other clip of the same singer and song, blanked 10% of the time (text-only
        # control) and cut to a random stretch; validation: always the first one, from its start
        train = self.split == 'train'
        prompt_list = ast.literal_eval(data['prompt_mel'])
        prompt = np.load(random.choice(prompt_list) if train else prompt_list[0])  # [80, T]
        max_prompt = min(self.max_prompt_len or self.max_batch_len, self.max_batch_len)
        if train and np.random.rand() < 0.1:
            prompt = np.ones((self.mel_num, self.min_batch_len)).astype(np.float32) * self.pad_value
        elif prompt.shape[1] > max_prompt:
            a = np.random.randint(prompt.shape[1] - max_prompt + 1) if train else 0
            prompt = prompt[:, a:a + max_prompt]

        # 读取f0
        f0_path = data['f0_path'] if 'f0_path' in data else data['audio_path'].replace('.wav', '_f0.npy')
        f0 = np.load(f0_path)  # [T,]
        # 截断或对齐f0长度与mel一致
        if f0.shape[0] > spec_len:
            # if start < 0:
            start = 0
            f0 = f0[start:start+spec_len]
        else:
            min_len = min(f0.shape[0], spec_len)
            f0 = f0[:min_len]
            spec = spec[:, :min_len]
            spec_len = min_len
        f0 = f0[np.newaxis, :]  # [1, T]

        ph = ast.literal_eval(data['ph'])
        ph_durs = ast.literal_eval(data['ph_durs'])
        ep_pitches = ast.literal_eval(data['ep_pitches'])
        
        # per-frame phoneme and note pitch
        ph = [ph2id[p] for p in ph]
        mel2ph, ph_array, ep_pitches_array = frame_tracks(ph, ph_durs, ep_pitches, spec_len)

        # 对mel2ph中连续相同ph的片段长度超过24的进行mask
        if train and random.random() > 0.5:  # boundary blurring, in training only
            blur_boundaries(mel2ph, ph_array, ep_pitches_array)

        # Adding language_id
        language_id = self.language_to_id[data['language']]

        # format in ldm/data/captions.py; `drop` is the rate of the empty (unconditional) caption
        if self.split == 'train':
            caption = sample_caption(data, field_drop=0.1, caption_drop=self.drop)
        else:
            caption = caption_from_row(data)

        item['prompt'] = prompt
        item['image'] = spec
        item['caption'] = caption
        item['f0'] = f0
        item['name'] = data['item_name']
        item['ph'] = ph_array[np.newaxis, :].astype(np.int32)
        item['ep_pitches'] = ep_pitches_array[np.newaxis, :].astype(np.int32)
        item['mel2ph'] = mel2ph[np.newaxis, :].astype(np.int32)
        item['language'] = language_id

        if self.split == 'test':
            item['f_name'] = data['item_name']
        return item

    def __len__(self):
        return len(self.dataset)


class JoinSpecsTrain(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('train', **specs_dataset_cfg)


class JoinSpecsValidation(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('valid', **specs_dataset_cfg)


class JoinSpecsTest(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('test', **specs_dataset_cfg)


class DDPIndexBatchSampler(Sampler):    # 让长度相似的音频的indices合到一个batch中以避免过长的pad
    def __init__(self, indices, batch_size, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, max_tokens=80000) -> None:
        
        if num_replicas is None:
            if not dist.is_initialized():
                # raise RuntimeError("Requires distributed package to be available")
                print("Not in distributed mode")
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_initialized():
                # raise RuntimeError("Requires distributed package to be available")
                rank = 0
            else:
                rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.indices = indices
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.batches = self.build_batches()
        print(f"rank: {self.rank}, batches_num {len(self.batches)}")
        # If the dataset length is evenly divisible by replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        # print(num_replicas, len(self.batches))
        
        if self.drop_last and len(self.batches) % self.num_replicas != 0:
            self.batches = self.batches[:len(self.batches)//self.num_replicas*self.num_replicas]
        if len(self.batches) > self.num_replicas:
            self.batches = self.batches[self.rank::self.num_replicas]
        else: # may happen in sanity checking
            self.batches = [self.batches[0]]
        print(f"after split batches_num {len(self.batches)}")
        self.shuffle = shuffle
        if self.shuffle:
            self.batches = np.random.permutation(self.batches)
        self.seed = seed

    def set_epoch(self,epoch):
        self.epoch = epoch
        if self.shuffle:
            np.random.seed(self.seed+self.epoch)
            self.batches = np.random.permutation(self.batches)

    def build_batches(self):
        batches, batch = [], []
        for index in self.indices:
            batch.append(index)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
        if not self.drop_last and len(batch) > 0:
            batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)



class ListBatchSampler(Sampler):
    """Yields batches that were built beforehand (JoinManifestSpecs.epoch_batches)."""

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
