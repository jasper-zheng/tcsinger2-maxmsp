import sys
import numpy as np
import torch
from typing import TypeVar, Optional, Iterator
import logging
import pandas as pd
from ldm.data.joinaudiodataset_anylen import *
import glob
import math
import ast
import random

logger = logging.getLogger(f'main.{__name__}')

sys.path.insert(0, '.')  # nopep8

"""
adapted from joinaudiodataset_struct_sample_anylen.py
"""

language_set = ['Chinese','English','French','German','Italian','Japanese','Korean','Russian','Spanish']

ph_set = [
    "tʰ_ko",
    "ɔ_it",
    "d_it",
    "dʑ_ja",
    "ɣ_es",
    "in",
    "ʒ_fr",
    "tʲ_ko",
    "p_it",
    "ɕ͈_ko",
    "t͡ʃː_it",
    "UW1",
    "dʲ_ko",
    "IH2",
    "NG",
    "ZH",
    "ɡ_es",
    "ɾ_ja",
    "ɛ_ko",
    "IH1",
    "j",
    "ʎ_fr",
    "ɨː_ja",
    "kː_it",
    "a",
    "OW1",
    "f",
    "AO0",
    "ou",
    "k_ja",
    "l_it",
    "o_ja",
    "F",
    "ʂː_ru",
    "AW1",
    "i_ko",
    "ɰ_ko",
    "tɕ_ru",
    "uai",
    "z_it",
    "ie",
    "u_ko",
    "mː_it",
    "ʃ_it",
    "j_fr",
    "ch",
    "n_es",
    "ɛ_de",
    "ɴ_ja",
    "tɕʷ_ko",
    "b_it",
    "iou",
    "IY1",
    "ɛ_fr",
    "k͈ʷ_ko",
    "s̪_ru",
    "b_ja",
    "aj_de",
    "c_ru",
    "e_it",
    "pf_de",
    "ç_es",
    "<SP>",
    "v_it",
    "ŋ_fr",
    "ʔ_ja",
    "S",
    "tː_ja",
    "ɑ_fr",
    "x_es",
    "r_es",
    "sh",
    "ɡ_fr",
    "ʌː_ko",
    "ɟ_es",
    "zʲ_ru",
    "AW0",
    "AA0",
    "Y",
    "ɟ_de",
    "e_ru",
    "ə_ru",
    "P",
    "t_de",
    "f_ru",
    "l_fr",
    "k_ko",
    "ɐ_ru",
    "t͡s_it",
    "UH0",
    "s_ja",
    "ʃ_fr",
    "β_es",
    "n_ja",
    "AY2",
    "ʌ_ko",
    "p_ko",
    "dʲ_ru",
    "ŋ_de",
    "iang",
    "b_ru",
    "o",
    "AA2",
    "a_ru",
    "t",
    "o_ru",
    "x_ru",
    "ɛ̃_fr",
    "oː_ko",
    "AE1",
    "v",
    "D",
    "z̪_ru",
    "b_es",
    "UW2",
    "uo",
    "EY1",
    "œ_de",
    "øː_de",
    "yː_de",
    "CH",
    "o_fr",
    "bʲ_ja",
    "u",
    "sː_it",
    "<AP>",
    "w_es",
    "UH2",
    "ø_fr",
    "ʎ_es",
    "ɪ_de",
    "er",
    "k_de",
    "ç_ru",
    "L",
    "ʐ_ru",
    "G",
    "ʑ_ja",
    "sʲ_ru",
    "t_fr",
    "dː_it",
    "ɥ_fr",
    "s_fr",
    "x_de",
    "ɨ_ru",
    "EY2",
    "kʰ_de",
    "ə_fr",
    "l",
    "ð_es",
    "n̪ː_ru",
    "a_it",
    "ɕː_ru",
    "z_de",
    "w_ja",
    "ɲ_ja",
    "r_ru",
    "TH",
    "rː_ru",
    "t͈_ko",
    "j_ja",
    "t_ko",
    "w_fr",
    "AW2",
    "uː_ko",
    "t̪ː_ru",
    "c_ko",
    "OY1",
    "e_es",
    "ʁ_fr",
    "ŋ_ko",
    "eː_ko",
    "ɔ̃_fr",
    "fʲ_ru",
    "j_es",
    "ʃ_de",
    "mː_ko",
    "d",
    "ɲ_ko",
    "OW0",
    "j_it",
    "AY1",
    "h_ja",
    "dz_ja",
    "f_es",
    "AA1",
    "ʎ_ru",
    "k_fr",
    "cʰ_de",
    "ɲ_fr",
    "c͈_ko",
    "θ_es",
    "pː_it",
    "d_ko",
    "pʰ_de",
    "e_ko",
    "d̪_es",
    "EH0",
    "m_es",
    "ɲ_de",
    "UW0",
    "ɫ_ru",
    "ŋ_es",
    "ɔ_de",
    "M",
    "B",
    "tʃ_fr",
    "V",
    "pʲ_ko",
    "IY0",
    "eː_ja",
    "aː_ja",
    "sʰ_ko",
    "ɨ̥_ja",
    "z_fr",
    "d_de",
    "m_fr",
    "s͈_ko",
    "uang",
    "k",
    "eː_de",
    "hː_it",
    "ɵ_ru",
    "m_it",
    "ʏ_de",
    "f_it",
    "ɕ_ja",
    "ɰ̃_ja",
    "uei",
    "i",
    "ei",
    "ɴː_ja",
    "tɕ_ko",
    "ʊ_ru",
    "OY2",
    "i_es",
    "IY2",
    "ian",
    "u_fr",
    "pʷ_ko",
    "ɦ_ko",
    "j_ko",
    "rʲ_ru",
    "c_es",
    "iao",
    "en",
    "t_ja",
    "y_fr",
    "ə_de",
    "tɕ͈_ko",
    "AE0",
    "tɕ_ja",
    "d̪_ru",
    "a_de",
    "ER1",
    "tʃ_es",
    "ɪ_ru",
    "ɸʷ_ko",
    "ER2",
    "ç_de",
    "tɕʰ_ko",
    "ɥ_ko",
    "T",
    "EH1",
    "d_ja",
    "ɲ_it",
    "DH",
    "eng",
    "ç_ko",
    "tː_it",
    "i_it",
    "ɲː_ru",
    "h",
    "o_it",
    "vʲ_ru",
    "sʷ_ko",
    "AH0",
    "oː_ja",
    "ç_ja",
    "ɕʰ_ko",
    "uː_de",
    "ɸ_ja",
    "q",
    "fː_it",
    "m_ko",
    "ɐ_de",
    "k̚_ko",
    "g",
    "x_ko",
    "s_ko",
    "ɕ_ru",
    "tʃ_de",
    "i_ja",
    "N",
    "JH",
    "IH0",
    "t̪s̪ː_ru",
    "dʑ_ko",
    "n̩_de",
    "n_de",
    "t̚_ko",
    "AH1",
    "t͡ʃ_it",
    "zh",
    "ʐː_ru",
    "o_es",
    "c_fr",
    "ʁ_de",
    "vn",
    "n_fr",
    "ɯ̥_ja",
    "ɟ_ja",
    "r_it",
    "an",
    "lː_it",
    "ang",
    "a_fr",
    "ɡ_de",
    "r",
    "nː_ja",
    "ɐ_ko",
    "k_ru",
    "R",
    "u_it",
    "AE2",
    "f_fr",
    "rː_it",
    "ɾʲ_ja",
    "p_ja",
    "bʲ_ru",
    "tʷ_ko",
    "mʲ_ja",
    "æ_ru",
    "e_fr",
    "m_de",
    "m_ru",
    "n",
    "ɟʝ_es",
    "t̪_es",
    "ia",
    "n_it",
    "iong",
    "ɡ_it",
    "u_ru",
    "v_ru",
    "ɲ_ru",
    "ɔʏ_de",
    "tʰ_de",
    "mʲ_ru",
    "pʰ_ko",
    "ʎ_it",
    "t̪s̪_ru",
    "HH",
    "m̩_de",
    "a_ja",
    "ɛː_ko",
    "OW2",
    "uen",
    "t̪_ru",
    "e_ja",
    "p_de",
    "i_ru",
    "bʲ_ko",
    "j_de",
    "AH2",
    "k_it",
    "ɯ_ja",
    "oː_de",
    "EY0",
    "d͡ʒ_it",
    "ɨ_ja",
    "aː_de",
    "p̚_ko",
    "s_it",
    "ɛ_it",
    "ai",
    "l_es",
    "k_es",
    "K",
    "ʉ_ru",
    "o_ko",
    "EH2",
    "v_de",
    "ɾ_ko",
    "s",
    "p_ru",
    "UH1",
    "ts_ja",
    "i̥_ja",
    "ts_de",
    "ʂ_ru",
    "z",
    "b_de",
    "pʲ_ru",
    "Z",
    "v_fr",
    "aw_de",
    "k͈_ko",
    "f_de",
    "n̪_ru",
    "œ_fr",
    "ɟ_fr",
    "SH",
    "d͡ʒː_it",
    "ɾʲ_ko",
    "ua",
    "mʲ_ko",
    "ɨː_ko",
    "h_de",
    "pː_ja",
    "c",
    "s_de",
    "l_de",
    "m",
    "j_ru",
    "cʰ_ko",
    "ɾ_es",
    "p",
    "w_ko",
    "p_es",
    "ER0",
    "ɨ_ko",
    "z_ja",
    "b_fr",
    "s̪ː_ru",
    "ɡ_ja",
    "s_es",
    "ʝ_es",
    "c_ja",
    "h_ko",
    "d_fr",
    "kʷ_ko",
    "ɲ_es",
    "iː_ja",
    "uan",
    "van",
    "ao",
    "AO1",
    "ve",
    "e",
    "mʲ_fr",
    "t͈ʲ_ko",
    "ɡ_ru",
    "ʊ_de",
    "AY0",
    "n_ko",
    "ing",
    "kʰ_ko",
    "W",
    "iː_de",
    "nː_it",
    "p͈_ko",
    "ɟ_ru",
    "a_es",
    "x",
    "ɔ_fr",
    "m_ja",
    "i_fr",
    "ɛ_ru",
    "ɑ̃_fr",
    "ong",
    "bː_it",
    "t_it",
    "AO2",
    "tsʲ_ru",
    "u_es",
    "p_fr",
    "iː_ko",
    "b",
    "ɭ_ko"
]

class JoinManifestSpecs(torch.utils.data.Dataset):
    def __init__(self, split, main_spec_dir_path, other_spec_dir_path, mel_num=80, mode='pad', spec_crop_len=1248,
                 pad_value=-5, drop=0, max_tokens=80000, **kwargs):
        super().__init__()
        self.split = split
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

        if split == 'train':
            self.dataset = self.df_main.iloc[300:]
        elif split == 'valid' or split == 'val':
            self.dataset = self.df_main.iloc[:300]
        elif split == 'test':
            self.df_main = self.add_name_num(self.df_main)
            self.dataset = self.df_main
        else:
            raise ValueError(f'Unknown split {split}')
        self.dataset.reset_index(inplace=True)
        print('dataset len:', len(self.dataset), "drop_rate", self.drop)

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
        to_dict = {}
        for l in inputs:
            for k, v in l.items():
                if k in to_dict: # image, acoustic, f0, midi, beats, caption, prompt, name
                    to_dict[k].append(v)
                else:
                    to_dict[k] = [v]

        if self.collate_mode == 'pad':
            to_dict['image'] = collate_1d_or_2d(to_dict['image'], pad_idx=self.pad_value, min_len=self.min_batch_len,   # B, C, T
                                                max_len=self.max_batch_len, min_factor=self.min_factor)
            to_dict['prompt'] = collate_1d_or_2d(to_dict['prompt'], pad_idx=self.pad_value, min_len=self.min_batch_len,   # B, C, T
                                                max_len=self.max_batch_len, min_factor=self.min_factor)
            to_dict['f0'] = collate_1d_or_2d(to_dict['f0'], pad_idx=0, min_len=self.min_batch_len,   # B, C, T
                                                max_len=self.max_batch_len, min_factor=self.min_factor)
            to_dict['mel2ph'] = collate_1d_or_2d(to_dict['mel2ph'], pad_idx=0, min_len=self.min_batch_len,   # B, C, T
                                                max_len=self.max_batch_len, min_factor=self.min_factor).long()
            to_dict['ph'] = collate_1d_or_2d(to_dict['ph'], pad_idx=475, 
                                                min_len=self.min_batch_len,   # B, C, T
                                                max_len=self.max_batch_len, min_factor=self.min_factor).long()

            to_dict['ep_pitches'] = collate_1d_or_2d(to_dict['ep_pitches'], pad_idx=99, 
                                                min_len=self.min_batch_len,   # B, C, T
                                                max_len=self.max_batch_len, min_factor=self.min_factor).long()

        else:
            raise NotImplementedError

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
            print('too long', spec.shape[1])
            start=0
            spec = spec[:, start: start + self.max_batch_len]
            spec_len = self.max_batch_len

        prompt_list = ast.literal_eval(data['prompt_mel'])
        prompt_mel = random.choice(prompt_list)
        prompt=np.load(prompt_mel)  # [80, T]
        if np.random.rand() < 0.1:
            prompt = np.ones((self.mel_num, self.min_batch_len)).astype(np.float32) * self.pad_value
        elif np.random.rand() > 0.9:
            prompt_mel=spec
        if prompt.shape[1] > self.max_batch_len:
            prompt = prompt[:, :self.max_batch_len]

        # 读取f0
        f0_path = data['audio_path'].replace('.wav', '_f0.npy')
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
        
        # mel2ph
        audio_sample_rate = 48000
        hop_size = 256

        mel2ph = np.zeros((spec_len,), dtype=np.int32)
        current_frame = 0
        for i_ph, dur in enumerate(ph_durs):
            # 如果 dur = -1，表示无效持续时间，可根据需要决定跳过或赋为0
            if dur <= 0:
                # 如跳过：继续下一个音素
                # 或者将dur视作0，不填充
                continue
            # 持续时间转为mel帧数
            frames = int(dur * audio_sample_rate / hop_size + 0.5)
            end_frame = current_frame + frames
            if end_frame > spec_len:
                # 如果超过mel长度，则截断
                end_frame = spec_len
            # 将对应区间的mel帧标记为此音素
            # 通常mel2ph从1开始编号音素
            mel2ph[current_frame:end_frame] = i_ph + 1
            # 更新下一个音素的起始帧位置
            current_frame = end_frame
            # 如果已经填满整个mel长度，则跳出循环
            if current_frame >= spec_len:
                break

        # ph id
        ph2id = {ph: idx for idx, ph in enumerate(ph_set)}
        ph = [ph2id[p] for p in ph]

        ph_array = np.full((spec_len,), 475, dtype=np.int32)
        ep_pitches_array = np.full((spec_len,), 99, dtype=np.int32)

        for i in range(spec_len):
            p = mel2ph[i] - 1
            if p >= 0:
                ph_array[i] = ph[p]               # 若ph是字符串需映射为int
                ep_pitches_array[i] = ep_pitches[p]

        # 对mel2ph中连续相同ph的片段长度超过24的进行mask
        ph_val = mel2ph[0] if spec_len > 0 else -1
        start_idx = 0
        length = 1
        i = 1
        mask_not= random.random() > 0.5
        while i < spec_len:
            if mel2ph[i] == ph_val and ph_val != 475:
                length += 1
            else:
                if length > 24 and mask_not:  
                    seg_start = np.random.randint(start_idx, start_idx + length - 24 + 1)
                    seg_end = seg_start + 24
                    # mask前后多余的部分
                    for mask_range in [range(start_idx, seg_start), range(seg_end, start_idx + length)]:
                        mel2ph[mask_range] = 0
                        ph_array[mask_range] = 475
                        ep_pitches_array[mask_range] = 99
                ph_val = mel2ph[i]
                start_idx = i
                length = 1
            i += 1
        # 最后一个连续段处理
        if length > 24 and mask_not:
            seg_start = np.random.randint(start_idx, start_idx + length - 24 + 1)
            seg_end = seg_start + 24
            for mask_range in [range(start_idx, seg_start), range(seg_end, start_idx + length)]:
                mel2ph[mask_range] = 0
                ph_array[mask_range] = 475
                ep_pitches_array[mask_range] = 99


        # Adding language_id
        language_id = self.language_to_id[data['language']]

        caption=''
        if np.random.rand() > 0.1:
            caption += f', A Singer: {data["singer"]}'
        if np.random.rand() > 0.1:
            caption += f', in Language: {data["language"]},'
        if data['emotion'] !='no' and np.random.rand() > 0.1:
            caption += f', with Emotion: {data["emotion"]} and Singing Method: {data["singing_method"]},'
        if data['technique'] != 'no' and np.random.rand() > 0.1:
            caption += f', using Technique: {data["technique"]}'

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


