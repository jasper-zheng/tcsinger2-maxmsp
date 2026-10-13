import sys
import numpy as np
import torch
import logging
import pandas as pd
import glob
import ast
logger = logging.getLogger(f'main.{__name__}')

sys.path.insert(0, '.')  # nopep8

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

        if split == 'train':
            self.dataset = df.iloc[100:]
        elif split == 'valid' or split == 'val':
            self.dataset = df.iloc[:100]
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
        f0=np.load(data['audio_path'].replace('.wav','_f0.npy')) # f0 [1,3750]
        f0 = np.expand_dims(f0, axis=0)  
        f0=f0[:,:spec.shape[1]]
        
        ph = ast.literal_eval(data['ph'])
        ph_durs = ast.literal_eval(data['ph_durs'])
        ep_pitches = ast.literal_eval(data['ep_pitches'])
        ep_notedurs = ast.literal_eval(data['ep_notedurs'])
        ep_types = ast.literal_eval(data['ep_types'])
        
        ph2id = {ph: idx for idx, ph in enumerate(ph_set)}
        ph = [ph2id[p] for p in ph]

        if spec.shape[1] < self.batch_max_length:
            spec = np.tile(spec, reps=(self.batch_max_length//spec.shape[1])+1)
        if f0.shape[1] < self.batch_max_length:
            f0 = np.tile(f0, reps=(self.batch_max_length//f0.shape[1])+1)

        # 随机选择
        if spec.shape[1] > self.batch_max_length:
            start = np.random.randint(spec.shape[1] - self.batch_max_length)
            spec = spec[:, start: start + self.batch_max_length]
        if f0.shape[1] > self.batch_max_length:
            f0=f0[:, start: start + self.batch_max_length]

        PH_MAX_LEN = 30
        if len(ph) > PH_MAX_LEN:
            start_tok = np.random.randint(len(ph) - PH_MAX_LEN + 1)
            slice_fn = lambda x: x[start_tok : start_tok + PH_MAX_LEN]
            ph, ph_durs, ep_pitches, ep_notedurs, ep_types = map(slice_fn,
                (ph, ph_durs, ep_pitches, ep_notedurs, ep_types))
        elif len(ph) < PH_MAX_LEN:
            pad_n   = PH_MAX_LEN - len(ph)
            ph          += [475]  * pad_n   # ← ph 序列 PAD = 475
            ph_durs     += [0.0]  * pad_n   # ← 时长仍用 0.0
            ep_pitches  += [99]   * pad_n   # ← pitch PAD = 99
            ep_notedurs += [0.0]  * pad_n   # ← 音符时值 PAD = 0.0
            ep_types    += [4]    * pad_n   # ← type PAD = 4

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



