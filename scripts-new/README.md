# Training pipeline

Trains the three models of this TCSinger2 reproduction from the raw corpora, and sings with them:

| model | what it does | trained by |
|---|---|---|
| VAE (stage 1) | compresses 80-bin mels into 20-channel latents at a quarter of the frame rate; predicts phoneme lengths at inference | `3_train_vae.sh` |
| stage 2 | flow-matching transformer: notes, lyrics, language, a style caption and an audio prompt in, a mel out; follows an F0 curve when one is given, predicts one otherwise | `4_train_stage2.sh` |
| F0 generator | makes a sung F0 curve (vibrato, slides) from the notes, for notes-only synthesis | `5_train_f0gen.sh` |

The HiFi-GAN vocoder (48 kHz), FLAN-T5-large (caption encoder) and RMVPE (pitch tracker) are pretrained and downloaded.

## Steps

Run everything from the repository root. Each step is resumable: run the same command again after an interruption and
it skips what has finished. `run_all.sh` runs steps 1 to 5 in order.

| step | command | output in `work/` | time (measured here) |
|---|---|---|---|
| 0 | `bash scripts-new/0_setup.sh` | Python packages, `useful_ckpts/{flan-t5-large,rmvpe}`, `data/mfa_dict/` | |
| 1 | `bash scripts-new/1_data_gtsinger.sh` | `data/meta/` (metadata, phone set, splits, reports, lexicon), `data/sing/new.tsv`, mels and F0 | |
| 2 | `bash scripts-new/2_data_extra.sh` (optional) | `data/meta_ext/`, `data/sing_ext/new.tsv`, `ckpts/aligner_zh.pt` | about 2.5 h for all three corpora |
| 3 | `nohup bash scripts-new/3_train_vae.sh > /dev/null 2>&1 &` | `logs/<time>_ae_singing/`, `ckpts/vae.ckpt` | 17 h, RTX A6000 |
| 4 | `nohup bash scripts-new/4_train_stage2.sh > /dev/null 2>&1 &` | `logs/<time>_tcsinger2/`, `ckpts/tcsinger2.ckpt` + `ckpts/tcsinger2.yaml` | 29 h, RTX A6000 (17 h, A100) |
| 5 | `nohup bash scripts-new/5_train_f0gen.sh > /dev/null 2>&1 &` | `logs/<time>_f0gen/`, `ckpts/f0gen.ckpt` | 6.5 h, RTX A6000 |
| 6 | `bash scripts-new/6_sing.sh --score examples/twinkle_en.json --out samples/x.wav` | a wav, its F0 plot and a JSON of what was sung | seconds |

Step 5 needs only the data, so it can run before steps 3 and 4. Each step writes its log to `work/logs/<step>.log`
(`data_gtsinger`, `data_extra`, `train_vae`, `train_stage2`, `train_f0gen`).

## Setting up

Needs Linux, Python 3.11, an NVIDIA GPU with CUDA 12.1 (one RTX A6000 48 GB was used; stage 2 takes about 19 GB at its
frame budget, the F0 generator 16 GB), a C compiler (for `pyworld`) and the disk space below.

1. Edit `scripts-new/config.sh` (or set the same variables in the environment). At least: `TCS_VENV` for a venv, the
   GTSinger source, and `EXTRA_CORPORA` with its archives if the extra corpora are wanted.
2. `bash scripts-new/0_setup.sh`. It creates the venv (or installs into the active environment if `TCS_VENV` is
   empty), downloads FLAN-T5-large and RMVPE, and GTSinger's MFA pronunciation dictionaries.
3. Download the HiFi-GAN vocoder by hand from the
   [authors' Google Drive](https://drive.google.com/drive/folders/1ve9cm_Yn3CQWSqkzMuRL33Uj1BNh51lR) into
   `useful_ckpts/hifigan/` (`config.yaml`, `model_ckpt_steps_1000000.ckpt`). Step 0 says if it is missing.

## Data

| corpus | where to get it | settings in `config.sh` |
|---|---|---|
| GTSinger (required; 80 h, 20 singers, 9 languages) | the nine language zips, or the extracted tree on Hugging Face: `huggingface-cli download GTSinger/GTSinger --repo-type dataset --local-dir <dir> --exclude 'processed/*'` | `GTSINGER_ZIPS`, or `GTSINGER_ROOT` for an extracted tree |
| M4Singer (Chinese, 30 h, 20 singers) | `m4singer.zip` from the M4Singer authors | `M4SINGER_ZIP`, `EXTRA_CORPORA` contains `m4singer` |
| Opencpop (Chinese, 5 h, 1 singer) | `wavs.zip`, `textgrids.zip`, `midis.zip`, by application; the zips are password-protected | `OPENCPOP_DIR`, `opencpop`; export `OPENCPOP_ZIP_PASSWORD` for step 2 (never write it into a file) |
| OpenSinger (Chinese, 52 h, 76 singers) | `OpenSinger.tar.gz` | `OPENSINGER_TAR`, `opensinger` |

The released models were built from the GTSinger zips; the Hugging Face tree has the same layout but was not compared
with them. GTSinger, M4Singer and OpenSinger are CC BY-NC-SA 4.0. Opencpop's licence has not been checked (it is
believed to be CC BY-NC-ND 4.0); confirm it before sharing anything trained on it.

What step 1 does to GTSinger: cleans the annotations (labels, gaps, rest notes inside words, typos), splits every
phoneme at note boundaries, drops clips with phonemes seen fewer than 10 times, holds out the singers of
`GTSINGER_UNSEEN` as `test_unseen` and whole songs as `valid` and `test_seen`, extracts mels (loudness-normalised to
-23 LUFS, cut at 20 s) and RMVPE F0 at the mel frame rate (187.5 frames/s), fills note pitches that the annotation
lacks (Japanese, Korean) from the F0, writes the style caption of each clip and its list of prompt clips.
`work/data/meta/report.md` and `postprocess_report.json` describe the result.

What step 2 adds: the extra corpora resampled to 48 kHz and mapped onto GTSinger's phonemes, with their own held-out
singers and songs consistent with GTSinger's splits. OpenSinger has lyrics but no timings or notes: a small aligner is
trained on the annotated Chinese clips, OpenSinger's phonemes are placed with it, and its notes are taken from the F0.
The aligner reached 84% of boundaries within 50 ms on unseen singers.

## Work directory

```
work/                        TCS_WORK
  audio/GTSinger/            GTSINGER_ROOT when extracted from zips (51 GB)
  audio/ext/, audio/ext_raw/ EXT_ROOT: the extra corpora at 48 kHz (29 GB), Opencpop's raw files
  features/                  TCS_FEATURES: GTSinger/ and ext/ (melnone48000/, f0/; 21 + 18 GB), f0gen_cache/
  data/meta/                 metadata.json, phone_set.json, splits/, report.md, lexicon.json, ...
  data/meta_ext/             the same for the extra corpora, the aligner's output
  data/new*/                 intermediate manifests (one row per clip, before filtering)
  data/sing/new.tsv          training manifest of GTSinger
  data/sing_ext/new.tsv      training manifest of the extra corpora (data/sing_ext_m4_ocp/: without OpenSinger)
  data/mfa_dict/             pronunciation dictionaries for inference
  logs/                      training runs (<time>_ae_singing, <time>_tcsinger2, <time>_f0gen) and step logs
  ckpts/                     vae.ckpt, tcsinger2.ckpt + tcsinger2.yaml, f0gen.ckpt, aligner_zh.pt
  done/                      one marker per finished data step
```

Training runs keep their two newest resume checkpoints in `checkpoints/` and a weights-only milestone every 10,000
steps (VAE: batches) in `checkpoints/trainstep_checkpoints/` (VAE 1.3 GB, stage 2 0.85 GB each).

## Resuming, redoing, training further

- After an interruption, run the same command again. Data steps resume clip by clip where that is slow (mels, F0);
  training resumes from its newest checkpoint (every 1,000 steps). Training that has reached `max_steps` is skipped,
  and its export is rewritten only when the run has a newer checkpoint than the export.
- To redo a data step, delete its marker in `work/done/`. The steps after it are not redone by themselves; delete
  their markers as well. Changing `EXTRA_CORPORA` redoes the steps of step 2 that depend on it.
- Arguments after a training command go to `main.py` and are kept by the run: `--max_steps 300000` trains a finished
  run further (a multiple of the checkpoint interval: 1,000 steps; for the VAE 2,000, as it counts 2 steps per batch),
  and config keys such as `data.params.max_tokens=32000` change a setting, also on resume.
  A new run starts when no run of that config has a checkpoint; to start over, move the old run out of `work/logs/`.
- Export a milestone instead of the final step:
  `python scripts-new/py/export.py stage2 --ckpt work/logs/<run>/checkpoints/trainstep_checkpoints/<milestone>.ckpt --out work/ckpts/tcsinger2.ckpt`
  (also `vae`, `f0gen`; with `PYTHONPATH=.`).

## Singing with the models

`6_sing.sh` runs `py/sing.py` with the exports in `work/ckpts/` and the phone set and lexicon in `work/data/meta/`;
`bash scripts-new/6_sing.sh --help` lists the options. The score format is described at the top of
`ldm/infer/score.py`; `examples/` has scores in all nine languages, some with an F0 curve. Add
`--f0_model work/ckpts/f0gen.ckpt` to let the F0 generator make the curve when the score has none.

From Python, set `TCSINGER2_PHONE_SET=work/data/meta/phone_set.json` before importing `ldm` (a checkpoint only works
with the phone set it was trained with; the lexicon is read from the same directory), then
`ldm.infer.pipeline.Singer('work/ckpts/tcsinger2.ckpt', 'work/ckpts/tcsinger2.yaml', f0_model=...)`.
`tcsinger2.yaml` names the VAE by its absolute path; edit `model.params.first_stage_config.params.ckpt_path` there if
the files move.

## Not in this pipeline

- Evaluation (`scripts/eval_*.py`) and the checks in `scripts/checks/`.
- The audio encoder for speech prompts (`configs/ae_con.yaml`, WP5): stage 2 does not use it.
- The regression control of the F0 generator (`configs/f0gen_reg.yaml`).
- The speech manifest of GTSinger's paired speech clips (only the audio encoder reads it). Step 1 still writes
  `metadata_speech.json`.

## For maintainers: what replaced what

| `scripts-new/` | replaces |
|---|---|
| `0_setup.sh`, `requirements.txt` | `scripts/setup_env.sh` with `requirements-env.txt` + `constraints-env.txt` (assumed the Paperspace image's torch) |
| `1_data_gtsinger.sh` | the WP1 and WP2 command lists in `WORK_PACKAGES.md`, `scripts/build_lexicon.py` |
| `2_data_extra.sh` | the WP9 command list, `scripts/wp9_features.sh` |
| `3_train_vae.sh`, `4_train_stage2.sh`, `5_train_f0gen.sh` | `scripts/train_vae.sh`, `train_tcsinger2.sh`, `train_f0gen.sh`, `train_wp10.sh` |
| `py/make_manifest.py` | `preprocess/preprocess.py` (same rows; a missing wav is now an error instead of a dropped row) |
| `py/extract_mels.py` | `preprocess/mel_spec_48k.py` (same mels, bit for bit; one process, writes where the manifest says) |
| `py/export.py` | `scripts/export_vae.py`, `export_tcsinger2.py` (now also writes the run's config), `export_f0gen.py` |
| `py/build_lexicon.py`, `py/sing.py` | `scripts/build_lexicon.py`, `scripts/sing.py` |

The processing itself is not copied: the steps call `preprocess/` (metadata builders, splits, F0, post-processing,
aligner) and `main.py` with the configs in `configs/`. Library changes made for the pipeline, none of which changes
results: `TCSINGER2_F0GEN_CACHE` sets the F0 generator's cache directory (`ldm/f0gen/dataset.py`), the aligner reads
the phone set of `TCSINGER2_PHONE_SET` (`preprocess/aligner/common.py`), and `preprocess/ext_split_report.py` takes
`--corpora`.
