import sys
import numpy as np
import torch
import logging
import pandas as pd
import glob
import ast
from ldm.data.phone_set import ph2id, PH_PAD_ID, MIDI_PAD_ID, NOTE_TYPE_PAD
logger = logging.getLogger(f'main.{__name__}')

sys.path.insert(0, '.')  # nopep8


class JoinManifestSpecs(torch.utils.data.Dataset):
    def __init__(self, split, spec_dir_path, mel_num=None, spec_crop_len=None,drop=0,**kwargs):
        super().__init__()
        self.split = split
        self.batch_max_length = spec_crop_len
        self.batch_min_length = 50
        self.mel_num = mel_num
        self.drop = drop
        manifest_files = []
        for dir_path in spec_dir_path.split(','):
            manifest_files += glob.glob(f'{dir_path}/**/*.tsv',recursive=True)
        df_list = [pd.read_csv(manifest,sep='\t') for manifest in manifest_files]
        df = pd.concat(df_list,ignore_index=True)

        has_split = 'split' in df.columns  # manifests built from the GTSinger metadata carry their split
        if split == 'train':
            self.dataset = df[df['split'] == 'train'] if has_split else df.iloc[100:]
        elif split == 'valid' or split == 'val':
            self.dataset = df[df['split'] == 'valid'] if has_split else df.iloc[:100]
        elif split == 'test':
            df = self.add_name_num(df)
            self.dataset = df
        else:
            raise ValueError(f'Unknown split {split}')
        self.dataset.reset_index(inplace=True)
        print('dataset len:', len(self.dataset))

    def add_name_num(self,df):
        """each file may have different caption, we add num to filename to identify each audio-caption pair"""
        name_count_dict = {}
        change = []
        for t in df.itertuples():
            name = getattr(t,'name')
            if name in name_count_dict:
                name_count_dict[name] += 1
            else:
                name_count_dict[name] = 0
            change.append((t[0],name_count_dict[name]))
        for t in change:
            df.loc[t[0],'name'] = df.loc[t[0],'name'] + f'_{t[1]}'
        return df

    def __getitem__(self, idx):
        data = self.dataset.iloc[idx]
        item = {}
        spec = np.load(data['mel_path']) # mel spec [80, 3750]
        f0=np.load(data['f0_path'] if 'f0_path' in data else data['audio_path'].replace('.wav','_f0.npy')) # f0 [1,3750]
        f0 = np.expand_dims(f0, axis=0)  
        f0=f0[:,:spec.shape[1]]
        f0 = np.pad(f0, ((0, 0), (0, spec.shape[1] - f0.shape[1])))  # same length as the mel, so both take the same crop
        fixed = self.split != 'train'  # fixed crops outside training keep the validation loss comparable
        
        ph = ast.literal_eval(data['ph'])
        ph_durs = ast.literal_eval(data['ph_durs'])
        ep_pitches = ast.literal_eval(data['ep_pitches'])
        ep_notedurs = ast.literal_eval(data['ep_notedurs'])
        ep_types = ast.literal_eval(data['ep_types'])
        
        ph = [ph2id[p] for p in ph]

        if spec.shape[1] < self.batch_max_length:
            spec = np.tile(spec, reps=(self.batch_max_length//spec.shape[1])+1)
        if f0.shape[1] < self.batch_max_length:
            f0 = np.tile(f0, reps=(self.batch_max_length//f0.shape[1])+1)

        # 随机选择
        if spec.shape[1] > self.batch_max_length:
            start = 0 if fixed else np.random.randint(spec.shape[1] - self.batch_max_length)
            spec = spec[:, start: start + self.batch_max_length]
            f0=f0[:, start: start + self.batch_max_length]

        PH_MAX_LEN = 30
        if len(ph) > PH_MAX_LEN:
            start_tok = 0 if fixed else np.random.randint(len(ph) - PH_MAX_LEN + 1)
            slice_fn = lambda x: x[start_tok : start_tok + PH_MAX_LEN]
            ph, ph_durs, ep_pitches, ep_notedurs, ep_types = map(slice_fn,
                (ph, ph_durs, ep_pitches, ep_notedurs, ep_types))
        elif len(ph) < PH_MAX_LEN:
            pad_n   = PH_MAX_LEN - len(ph)
            ph          += [PH_PAD_ID]  * pad_n   # ← ph 序列 PAD
            ph_durs     += [0.0]  * pad_n   # ← 时长仍用 0.0
            ep_pitches  += [MIDI_PAD_ID]   * pad_n   # ← pitch PAD
            ep_notedurs += [0.0]  * pad_n   # ← 音符时值 PAD = 0.0
            ep_types    += [NOTE_TYPE_PAD]    * pad_n   # ← type PAD

        item['image'] = spec[:,:self.batch_max_length]
        item['f0'] = f0[:,:self.batch_max_length]
        item['ph'] = np.array(ph)
        item['pitches'] = np.array(ep_pitches)
        item['notedurs']=np.array(ep_notedurs)
        item['notetypes'] = np.array(ep_types)
        item['ph_durs'] = np.array(ph_durs)
        if self.split == 'test':
            item['f_name'] = data['name']
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



