# TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis

#### Yu Zhang, Wenxiang Guo, Changhao Pan, Dongyu Yao, Zhiyuan Zhu, Ziyue Jiang, Yuhan Wang, Tao Jin, Zhou Zhao | Zhejiang University

PyTorch implementation of **[TCSinger 2 (ACL 2025)](https://arxiv.org/abs/2505.14910): Customizable Multilingual Zero-shot Singing Voice Synthesis**.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2505.14910)
[![Demo](https://img.shields.io/badge/🚀%20Demo%20Page-blue)](https://aaronz345.github.io/TCSinger2Demo/) 
[![weixin](https://img.shields.io/badge/-WeChat@语音之家-000000?logo=wechat&logoColor=07C160)](https://mp.weixin.qq.com/s/Uvt27dt2S3TpBjJccl9mTA)
[![weixin](https://img.shields.io/badge/-WeChat@PaperWeekly-000000?logo=wechat&logoColor=07C160)](https://mp.weixin.qq.com/s/c6CPx14y_wASU_dSXKcAJQ)
[![zhihu](https://img.shields.io/badge/-知乎-000000?logo=zhihu&logoColor=0084FF)](https://zhuanlan.zhihu.com/p/1928562845236303816)
[![GitHub Stars](https://img.shields.io/github/stars/AaronZ345/TCSinger2?style=social&label=GitHub+Stars)](https://github.com/AaronZ345/TCSinger2)

Visit our [demo page](https://aaronz345.github.io/TCSinger2Demo/) for audio samples.

## News
- 2025.07: We released the code of TCSinger 2!
- 2025.07: We realeased the code of [STARS](https://github.com/gwx314/STARS)!
- 2025.05: TCSinger 2 is accepted by ACL 2025!

## Key Features
- We present **TCSinger 2**, a multi-task multilingual zero-shot SVS model with style transfer and style control based on various prompts.
- We introduce the **Blurred Boundary Content Encoder** for robust modeling and smooth transitions of phoneme and note boundaries.
- We design the **Custom Audio Encoder** using contrastive learning to extract styles from various prompts, while the **Flow-based Custom Transformer** with Cus-MOE and F0, enhances synthesis quality and style modeling.
- Experimental results show that TCSinger 2 outperforms baseline models in subjective and objective metrics across multiple tasks: **zero-shot style transfer, multi-level style control, cross-lingual style transfer, and speech-to-singing style transfer**.

## Quick Start

This repository trains TCSinger 2 on GTSinger, optionally with M4Singer, Opencpop and OpenSinger. Its pipeline fixes
the released data preparation, training and inference code, and adds an F0 generator for synthesis from notes alone.
The inputs are notes and lyrics (a JSON score), a style (a text caption and/or an audio prompt) and, optionally, an F0
curve that the output follows.

The pipeline is in [`scripts-new/`](scripts-new/README.md); its README has the details (data, disk space, times,
resuming, the work directory). Run every step from the repository root.

### Setup

Needs Linux, Python 3.11, an NVIDIA GPU with CUDA 12.1 (training was done on one RTX A6000 48 GB) and a C compiler.

1. Edit `scripts-new/config.sh`: the venv (`TCS_VENV`), the work directory for data, runs and exports (`TCS_WORK`,
   default `work/`), the GTSinger source and, optionally, the extra corpora.
2. Install the packages and download FLAN-T5-large, RMVPE and the pronunciation dictionaries:

```bash
bash scripts-new/0_setup.sh
```

3. Download the [HiFi-GAN vocoder](https://drive.google.com/drive/folders/1ve9cm_Yn3CQWSqkzMuRL33Uj1BNh51lR?usp=drive_link)
   into `useful_ckpts/hifigan`.

## Train your own model

### Data Preparation

1. Get [GTSinger](https://github.com/AaronZ345/GTSinger): either its nine language zips (set `GTSINGER_ZIPS`) or the
   extracted tree from Hugging Face (set `GTSINGER_ROOT`):
   `huggingface-cli download GTSinger/GTSinger --repo-type dataset --local-dir <dir> --exclude 'processed/*'`.
2. Build the metadata (cleaned annotations, note-split phonemes, phone set, singer-held-out splits), the mels, the F0
   (RMVPE), the style captions and the training manifest:

```bash
bash scripts-new/1_data_gtsinger.sh
```

3. Optionally add the Chinese corpora M4Singer, Opencpop and OpenSinger (set `EXTRA_CORPORA` and their archives).
   OpenSinger has no timings, so a small aligner is trained to place its phonemes:

```bash
bash scripts-new/2_data_extra.sh
```

Every step can be run again after an interruption: it skips what has finished.

### Training TCSinger 2

Each command starts its run, resumes it from its newest checkpoint, or skips it once finished, and then exports the
model to `work/ckpts/`. Extra arguments go to `main.py` (e.g. `--max_steps 300000` to train further).

1. The VAE and the duration predictor (about 17 h on an RTX A6000):

```bash
nohup bash scripts-new/3_train_vae.sh > /dev/null 2>&1 &
```

2. The main TCSinger 2 model (about 29 h on an RTX A6000):

```bash
nohup bash scripts-new/4_train_stage2.sh > /dev/null 2>&1 &
```

3. The F0 generator for notes-only synthesis (about 6.5 h; it needs only the data, not the models above):

```bash
nohup bash scripts-new/5_train_f0gen.sh > /dev/null 2>&1 &
```

`bash scripts-new/run_all.sh` runs the data steps and all three trainings in order. Logs are in `work/logs/`.

*Notes*
- One GPU was used and tested (`GPUS` in `scripts-new/config.sh`).
- The Custom Audio Encoder for speech prompts (`configs/ae_con.yaml`) is not part of the pipeline; the main model does
  not use it.
- Evaluation scripts are in `scripts/`.

### Inference with TCSinger 2

With the models the pipeline trained:

```bash
bash scripts-new/6_sing.sh --score examples/twinkle_en.json --out samples/twinkle.wav --singer "female alto"
bash scripts-new/6_sing.sh --score examples/twinkle_en.json --out samples/twinkle_f0gen.wav --f0_model work/ckpts/f0gen.ckpt
bash scripts-new/6_sing.sh --score examples/f0_vibrato_it.json --out samples/with_f0.wav --prompt voice.wav
```

*The score is a JSON file of words with their notes (format in `ldm/infer/score.py`); the style is a caption and/or an
audio prompt; the F0 curve is optional and goes into the score as `"f0"`. Without it the main model predicts the F0
from the notes, or the F0 generator makes it with `--f0_model`. `examples/README.md` lists the example scores, among
them eight with the F0 of a recorded performance; `bash scripts-new/6_sing.sh -h` lists the options. This replaces
the released `scripts/test_sing.py`, which resynthesised training rows.*

`python scripts-new/py/sing.py` takes the same options and defaults to models in `useful_ckpts/`
(`tcsinger2_gtsinger.ckpt`, `vae_gtsinger.ckpt`) with `configs/tcsinger2.yaml`.

Inference also runs on macOS with Apple silicon (the models run on MPS, about 2 s per 10 s score on an M4 Max once
loaded). Copy `useful_ckpts/` from the training machine, then:

```bash
bash scripts/setup_env_mac.sh        # .venv with torch 2.11 (requirements-mac.txt) and the pronunciation dictionaries in data/mfa_dict
source .venv/bin/activate
python scripts-new/py/sing.py --score examples/twinkle_en.json --out samples/twinkle.wav --singer "female alto"
```

*`--device` defaults to `cuda`, else `mps`, else `cpu`. To sing many scores, load the model once from Python
(`ldm.infer.pipeline.Singer`) instead of calling the script per score.*


## Acknowledgements

This implementation uses parts of the code from the following Github repos:
[Make-An-Audio-3](https://github.com/Text-to-Audio/Make-An-Audio-3),
[TCSinger](https://github.com/AaronZ345/TCSinger)
[Lumina-T2X](https://github.com/Alpha-VLLM/Lumina-T2X)
as described in our code.

## Citations ##

If you find this code useful in your research, please cite our work:
```bib
@article{zhang2025tcsinger,
  title={TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis},
  author={Zhang, Yu and Guo, Wenxiang and Pan, Changhao and Yao, Dongyu and Zhu, Zhiyuan and Jiang, Ziyue and Wang, Yuhan and Jin, Tao and Zhao, Zhou},
  journal={arXiv preprint arXiv:2505.14910},
  year={2025}
}
```

## Disclaimer ##

Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's singing without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

 ![visitors](https://visitor-badge.laobi.icu/badge?page_id=AaronZ345/TCSinger2)
