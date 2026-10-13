import sys
import numpy as np
import torch
import logging
import pandas as pd
import glob
from typing import Dict, Any, Optional, List
import ast

logger = logging.getLogger(f"main.{__name__}")

sys.path.insert(0, '.')  # nopep8

# 定义填充常量
PAD_MEL_VAL = -5.0
PAD_PH_ID = 475
PAD_PITCH_ID = 99


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
    """A PyTorch Dataset that joins multiple TSV manifest files and returns
    time‑aligned singing/speech mel‑spectrogram chunks together with f0 curves.

    Expected manifest columns
    -------------------------
    Required
        - mel_path            : path to the singing mel‑spectrogram .npy file
        - audio_path          : path to the original singing .wav file (used to locate *_f0.npy)
        - speech_mel_path     : path to the *corresponding* speech mel‑spectrogram .npy file
    Optional (for contrastive / diff training, up to 4 pairs)
        - singing_diff{i}_path : path to mel‑spectrogram of a *different* singing clip (i = 1..4)
        - speech_diff{i}_path  : path to mel‑spectrogram of the *matching* speech clip

    All mel files are assumed to be 2‑D numpy arrays of shape [n_mels, T].
    """

    def __init__(
        self,
        split: str,
        spec_dir_path: str,
        mel_num: Optional[int] = None,
        spec_crop_len: int = 3750,
        drop: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.split = split
        self.batch_max_length = spec_crop_len
        self.batch_min_length = 50
        self.mel_num = mel_num
        self.drop = drop

        # 创建音素到ID的映射
        self.ph2id = {p: i for i, p in enumerate(ph_set)}
        self.default_phoneme_id = PAD_PH_ID

        # ------------------------------------------------------------------
        # Load & concatenate all manifest TSVs found in *spec_dir_path*
        # ------------------------------------------------------------------
        manifest_files = []
        for dir_path in spec_dir_path.split(','):
            manifest_files += glob.glob(f"{dir_path}/**/*.tsv", recursive=True)

        if not manifest_files:
            raise FileNotFoundError(
                f"No manifest TSV files found under: {spec_dir_path}")

        df_list = [pd.read_csv(fp, sep='\t') for fp in manifest_files]
        df = pd.concat(df_list, ignore_index=True)

        # ------------------------------------------------------------------
        # Split handling
        # ------------------------------------------------------------------
        if split == 'train':
            self.dataset = df.iloc[100:]
        elif split in {'valid', 'val'}:
            self.dataset = df.iloc[:100]
        elif split == 'test':
            df = self._add_name_num(df)
            self.dataset = df
        else:
            raise ValueError(f"Unknown split '{split}'")

        self.dataset.reset_index(drop=True, inplace=True)
        print(f'{split} dataset len:', len(self.dataset))

    @staticmethod
    def _pad_or_crop(
        arr: np.ndarray,
        target_len: int,
        *,
        pad_value: float = PAD_MEL_VAL,
    ) -> np.ndarray:
        """Return the **first** ``target_len`` frames of *arr*; if *arr* is
        shorter, right‑pad with *pad_value*.

        This deterministic behaviour matches the requirement of always starting
        from time‑index 0 (no random cropping).
        """
        T = arr.shape[-1]
        if T >= target_len:
            return arr[..., :target_len]
        pad_width = [(0, 0)] * arr.ndim
        pad_width[-1] = (0, target_len - T)
        return np.pad(arr, pad_width, mode='constant', constant_values=pad_value)

    def _add_name_num(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append an index suffix so that (audio, caption) pairs remain unique."""
        name_count_dict: Dict[str, int] = {}
        new_names: List[str] = []
        for name in df['name']:
            count = name_count_dict.get(name, -1) + 1
            name_count_dict[name] = count
            new_names.append(f"{name}_{count}")
        df = df.copy()
        df['name'] = new_names
        return df

    # ------------------------------------------------------------------
    # PyTorch dataset interface ----------------------------------------
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataset.iloc[idx]
        item: Dict[str, Any] = {"neg_singing": [], "neg_speech": []}

        # --------------------------------------------------------------
        # Frame‑level phoneme & pitch arrays ---------------------------
        # --------------------------------------------------------------
        ph = ast.literal_eval(row['ph'])
        ph_durs = ast.literal_eval(row['ph_durs'])
        ep_pitches = ast.literal_eval(row['ep_pitches'])

        L = self.batch_max_length
        mel2ph = np.zeros((L,), dtype=np.int32)
        cur = 0
        sr = 48000
        hop = 256
        for i_ph, dur in enumerate(ph_durs):
            if dur <= 0:
                continue
            frames = int(dur * sr / hop + 0.5)
            end = min(cur + frames, L)
            mel2ph[cur:end] = i_ph + 1
            cur = end
            if cur >= L:
                break

        # Construct frame‑level arrays with padding --------------------
        ph_array = np.full((L,), PAD_PH_ID, dtype=np.int32)
        pitch_array = np.full((L,), PAD_PITCH_ID, dtype=np.int32)
        for t in range(L):
            ph_idx = mel2ph[t] - 1
            if ph_idx >= 0 and ph_idx < len(ph) and ph_idx < len(ep_pitches):
                p = ph[ph_idx]
                ph_array[t] = self.ph2id.get(p, self.default_phoneme_id)
                pitch_array[t] = ep_pitches[ph_idx]

        caption = row['caption']

        # --------------------------------------------------------------
        #  Singing spec & f0 -----------------------------------------
        # --------------------------------------------------------------
        spec_singing = self._pad_or_crop(np.load(row['mel_path']), L, pad_value=PAD_MEL_VAL)

        f0_path = row['audio_path'].replace('.wav', '_f0.npy')
        f0 = np.load(f0_path)  # [1, T] or [T]
        if f0.ndim == 1:
            f0 = f0[None, :]  # [1, T]
        f0 = self._pad_or_crop(f0, L, pad_value=0)

        # --------------------------------------------------------------
        #  Speech spec -----------------------------------------------
        # --------------------------------------------------------------
        if 'speech_mel_path' not in row or pd.isna(row['speech_mel_path']):
            raise KeyError("Manifest missing required column 'speech_mel_path'.")
        spec_speech = self._pad_or_crop(np.load(row['speech_mel_path']), L, pad_value=PAD_MEL_VAL)

        # --------------------------------------------------------------
        #  Diff pairs (singing / speech) -------------------
        # --------------------------------------------------------------
        for i in range(1, 5):
            sing_key   = f'singing_diff{i}_path'
            speech_key = f'speech_diff{i}_path'

            if sing_key in row and pd.notna(row[sing_key]):
                item["neg_singing"].append(
                    self._pad_or_crop(np.load(row[sing_key]), L))
            if speech_key in row and pd.notna(row[speech_key]):
                item["neg_speech"].append(
                    self._pad_or_crop(np.load(row[speech_key]), L))

        # --------------------------------------------------------------
        #  Assemble output ------------------------------------------
        # --------------------------------------------------------------
        item.update({
            'image'       : spec_singing,   # singing mel
            'f0'          : f0,
            'spec_speech' : spec_speech,
            'ph_array'    : ph_array,
            'pitch_array' : pitch_array,
            'neg_singing' : item['neg_singing'], 
            'neg_speech'  : item['neg_speech'],
            'caption'     : caption
        })
        if self.split == 'test':
            item['f_name'] = row['name']
        return item


# Convenience wrappers ---------------------------------------------------------
class JoinSpecsTrain(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('train', **specs_dataset_cfg)


class JoinSpecsValidation(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('valid', **specs_dataset_cfg)


class JoinSpecsTest(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('test', **specs_dataset_cfg)