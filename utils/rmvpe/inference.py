# RMVPE pitch extractor. Vendored from GTSinger/Tech-Recognition/modules/pe/rmvpe
# (https://github.com/AaronZ345/GTSinger), itself based on https://github.com/Dream-High/RMVPE.
# Changes: no dependency on that repo's utils package or on pyworld; batch helpers removed;
# get_pitch resamples a gap-filled contour so frames at voicing edges are not averaged with zeros.
import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from .constants import *
from .model import E2E0
from .spec import MelSpectrogram
from .utils import to_local_average_f0, to_viterbi_f0


def resample_align_curve(points: np.ndarray, original_timestep: float, target_timestep: float, align_length: int):
    t_max = (len(points) - 1) * original_timestep
    curve_interp = np.interp(
        np.arange(0, t_max, target_timestep),
        original_timestep * np.arange(len(points)),
        points
    ).astype(points.dtype)
    delta_l = align_length - len(curve_interp)
    if delta_l < 0:
        curve_interp = curve_interp[:align_length]
    elif delta_l > 0:
        curve_interp = np.concatenate((curve_interp, np.full(delta_l, fill_value=curve_interp[-1])), axis=0)
    return curve_interp


class RMVPE:
    def __init__(self, model_path, hop_length=160, device=None):
        self.resample_kernel = {}
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model = E2E0(4, 1, (2, 2)).eval().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX
        ).to(self.device)
        self.hop_length = hop_length

    @torch.no_grad()
    def mel2hidden(self, mel):
        n_frames = mel.shape[-1]
        mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode='constant')
        hidden = self.model(mel)
        return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        if use_viterbi:
            f0 = to_viterbi_f0(hidden, thred=thred)
        else:
            f0 = to_local_average_f0(hidden, thred=thred)
        return f0

    def postprocess(self, f0, fmin=50, fmax=1000, min_gap=2):
        f0[f0 < fmin] = 0
        f0[f0 > fmax] = 0
        # eliminate glitch
        # min_gap: if successive positive f0 positions < min_gap, zero these positions
        # eg: if min_gap=2, [0, 500, 500, 0] => [0, 0, 0, 0]
        for idx in range(f0.shape[0] - min_gap - 1):
            if f0[idx] == 0 and f0[idx + min_gap + 1] == 0 and np.sum(f0[idx: idx + min_gap + 2]) > 0:
                f0[idx: idx + min_gap + 2] = 0
        return f0

    def infer_from_audio(self, audio, sample_rate=16000, thred=0.03, use_viterbi=False):
        audio = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        if sample_rate == 16000:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, 16000, lowpass_filter_width=128)
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
        mel = self.mel_extractor(audio_res, center=True)
        hidden = self.mel2hidden(mel)
        f0 = self.decode(hidden, thred=thred, use_viterbi=use_viterbi).squeeze(0)
        return f0

    def get_pitch(self, waveform, sample_rate, hop_size, length, interp_uv=False, fmin=50, fmax=1000):
        """F0 of `waveform` at the frame rate sample_rate / hop_size, `length` frames, unvoiced = 0."""
        f0 = self.infer_from_audio(waveform, sample_rate=sample_rate)
        f0 = self.postprocess(f0, fmin, fmax)
        uv = f0 == 0
        if uv.all():
            return np.zeros(length, dtype=f0.dtype), np.ones(length, dtype=bool)
        # fill unvoiced gaps before resampling, so that voiced frames next to a gap keep their pitch
        filled = f0.copy()
        filled[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        time_step = hop_size / sample_rate
        f0_res = resample_align_curve(filled, 0.01, time_step, length)
        uv_res = resample_align_curve(uv.astype(np.float32), 0.01, time_step, length) > 0.5
        if not interp_uv:
            f0_res[uv_res] = 0
        return f0_res, uv_res
