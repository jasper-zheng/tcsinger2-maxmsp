# Settings of the training pipeline. Edit this file, or override any value from the environment:
#   TCS_WORK=/data/tcsinger2 bash scripts-new/1_data_gtsinger.sh
# Paths may be absolute or relative to the repository root (every step runs from there).
# Paths must not contain spaces or commas (they go into config overrides).

# Python environment to activate before every step; empty = use the python that is active.
TCS_VENV=${TCS_VENV:-}

# Everything the pipeline writes: metadata, manifests, training runs, exported models (see README.md).
TCS_WORK=${TCS_WORK:-work}
# Audio and features are large; put them on another disk if needed.
TCS_AUDIO=${TCS_AUDIO:-$TCS_WORK/audio}          # 48 kHz wavs (GTSinger about 51 GB, extra corpora about 29 GB)
TCS_FEATURES=${TCS_FEATURES:-$TCS_WORK/features} # mels and F0 (GTSinger about 21 GB, extra corpora about 18 GB)

# GTSinger. Either give the directory with the nine language zips (they are extracted into GTSINGER_ROOT),
# or leave GTSINGER_ZIPS empty and point GTSINGER_ROOT at an extracted tree
# (<root>/<Language>/<Singer>/<Technique>/<Song>/<Group>/NNNN.{wav,json}).
GTSINGER_ZIPS=${GTSINGER_ZIPS:-}
GTSINGER_ROOT=${GTSINGER_ROOT:-$TCS_AUDIO/GTSinger}
# Singers held out as test_unseen (every clip of them).
GTSINGER_UNSEEN=${GTSINGER_UNSEEN:-"IT-Bass-2 KO-Soprano-1"}

# Extra Chinese corpora (optional, 2_data_extra.sh): any of m4singer opencpop opensinger; empty = GTSinger only.
# A corpus is unpacked from its archive into EXT_ROOT; with the archive unset, EXT_ROOT must already hold it
# as unpacked by an earlier run.
EXTRA_CORPORA=${EXTRA_CORPORA:-}
EXT_ROOT=${EXT_ROOT:-$TCS_AUDIO/ext}
M4SINGER_ZIP=${M4SINGER_ZIP:-}       # m4singer.zip
OPENCPOP_DIR=${OPENCPOP_DIR:-}       # directory with wavs.zip, textgrids.zip, midis.zip; also export OPENCPOP_ZIP_PASSWORD
OPENSINGER_TAR=${OPENSINGER_TAR:-}   # OpenSinger.tar.gz
# OpenSinger clips that the aligner places with a lower path score are dropped (their lyrics do not match the audio).
OPENSINGER_MIN_ALIGN_SCORE=${OPENSINGER_MIN_ALIGN_SCORE:--2.0}
ALIGNER_STEPS=${ALIGNER_STEPS:-15000}

# Training: the GPUs as Lightning takes them (one GPU tested: "0,").
GPUS=${GPUS:-0,}
# CPU workers of the data steps.
WORKERS=${WORKERS:-8}
