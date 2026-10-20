# Sourced by every step of the pipeline: settings, environment and helpers. Not meant to be run.
set -euo pipefail

NEW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$NEW_DIR")"
cd "$REPO"
source "$NEW_DIR/config.sh"

abs() { [ -n "$1" ] && realpath -m "$1" || true; }
TCS_WORK=$(abs "$TCS_WORK"); TCS_AUDIO=$(abs "$TCS_AUDIO"); TCS_FEATURES=$(abs "$TCS_FEATURES")
GTSINGER_ZIPS=$(abs "$GTSINGER_ZIPS"); GTSINGER_ROOT=$(abs "$GTSINGER_ROOT"); EXT_ROOT=$(abs "$EXT_ROOT")
M4SINGER_ZIP=$(abs "$M4SINGER_ZIP"); OPENCPOP_DIR=$(abs "$OPENCPOP_DIR"); OPENSINGER_TAR=$(abs "$OPENSINGER_TAR")

# Layout of the work directory (README.md).
DATA=$TCS_WORK/data            # metadata (meta, meta_ext), intermediate (new*) and training manifests (sing, sing_ext)
META=$DATA/meta
META_EXT=$DATA/meta_ext
LOGS=$TCS_WORK/logs            # training runs and the log of every step
CKPTS=$TCS_WORK/ckpts          # exported models: vae.ckpt, tcsinger2.ckpt + tcsinger2.yaml, f0gen.ckpt, aligner_zh.pt
DONE=$TCS_WORK/done            # one marker per finished step
GTS_FEATURES=$TCS_FEATURES/GTSinger
EXT_FEATURES=$TCS_FEATURES/ext
mkdir -p "$DATA" "$LOGS" "$CKPTS" "$DONE"

# The library reads its vocabulary, pronunciation dictionaries and F0 cache from these.
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
export TCSINGER2_PHONE_SET=$META/phone_set.json
export TCSINGER2_MFA_DICT=$DATA/mfa_dict
export TCSINGER2_F0GEN_CACHE=$TCS_FEATURES/f0gen_cache

if [ -n "$TCS_VENV" ]; then
    if [ -f "$TCS_VENV/bin/activate" ]; then
        set +u; source "$TCS_VENV/bin/activate"; set -u
    elif [ "$(basename "$0")" != 0_setup.sh ]; then
        echo "error: no venv at TCS_VENV=$TCS_VENV (run scripts-new/0_setup.sh)" >&2; exit 1
    fi
fi

log() { echo "==== $(date -u '+%F %T') UTC $*"; }
die() { echo "error: $*" >&2; exit 1; }
need() { [ -e "$1" ] || die "$1 is missing: $2"; }

# Appends everything the calling script prints to $LOGS/<name>.log (and still shows it).
log_to() { exec > >(tee -a "$LOGS/$1.log") 2>&1; log "$0 $*"; }

# step NAME COMMAND...: runs COMMAND unless the step has finished before (marker $DONE/NAME).
# To redo a step, delete its marker; the steps after it are not redone by themselves.
step() {
    local name=$1; shift
    if [ -e "$DONE/$name" ]; then log "$name: done before, skipped"; return; fi
    log "$name: start"
    "$@"
    touch "$DONE/$name"
    log "$name: done"
}

# newest_run NAME: the newest run directory $LOGS/<time>_NAME that has a checkpoint to resume from.
newest_run() {
    local d
    for d in $(ls -dt "$LOGS"/*_"$1" 2>/dev/null); do
        if [ -n "$(find "$d/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit 2>/dev/null)" ]; then echo "$d"; return; fi
    done
}

# max_steps CONFIG RUN ARGS...: --max_steps of ARGS, else that of the run's newest saved config, else the config's.
max_steps() {
    local config=$1 run=$2; shift 2
    while [ $# -gt 0 ]; do
        case $1 in --max_steps) echo "$2"; return ;; --max_steps=*) echo "${1#*=}"; return ;; esac
        shift
    done
    # main.py saves the trainer settings of every session and merges them in time order on resume
    local saved; saved=$( [ -n "$run" ] && ls "$run"/configs/*-lightning.yaml 2>/dev/null | sort | tail -1 || true)
    grep -m1 -E '^ *max_steps:' "${saved:-$config}" | awk '{print $2}'
}

# train CONFIG SCALE_LR [OVERRIDE...] -- [ARG...]: trains with main.py until max_steps and sets RUN.
# Starts a run of CONFIG in $LOGS with the OVERRIDEs (config keys, key=value), resumes the newest one, or
# does nothing when the run has its rolling checkpoint of max_steps (training has finished). ARGs go to
# main.py every time, e.g. --max_steps 300000 to train further (a multiple of the checkpoint interval).
train() {
    local config=$1 scale_lr=$2; shift 2
    local overrides=()
    while [ $# -gt 0 ] && [ "$1" != -- ]; do overrides+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    local name; name=$(basename "$config" .yaml)
    RUN=$(newest_run "$name")
    local steps; steps=$(max_steps "$config" "$RUN" "$@")
    if [ -n "$RUN" ] && [ -e "$RUN/checkpoints/step=$(printf '%08d' "$steps").ckpt" ]; then
        log "$RUN has finished ($steps steps)"
        return
    fi
    if [ -n "$RUN" ]; then
        log "resuming $RUN (to step $steps)"
        python main.py -r "$RUN" -t --gpus "$GPUS" --no-test true --scale_lr "$scale_lr" "$@"
    else
        log "starting a new run of $config (to step $steps)"
        python main.py --base "$config" -l "$LOGS" -t --gpus "$GPUS" --no-test true --scale_lr "$scale_lr" "${overrides[@]}" "$@"
        RUN=$(newest_run "$name")
    fi
    [ -n "$RUN" ] && [ -e "$RUN/checkpoints/step=$(printf '%08d' "$steps").ckpt" ] \
        || die "training stopped before step $steps; run the same command again to resume"
    log "$RUN has finished ($steps steps)"
}

# export_run KIND OUT: exports the newest checkpoint of $RUN (py/export.py KIND) unless OUT is newer than it.
export_run() {
    local kind=$1 out=$2 newest
    newest=$(ls -t "$RUN"/checkpoints/*.ckpt)
    newest=${newest%%$'\n'*}
    if [ -e "$out" ] && [ "$out" -nt "$newest" ]; then log "$out is up to date"; return; fi
    python scripts-new/py/export.py "$kind" --run "$RUN" --out "$out"
}

# The training manifests stage 2 and the F0 generator read: GTSinger, plus the extra corpora if built.
manifest_dirs() {
    need "$DATA/sing/new.tsv" "run scripts-new/1_data_gtsinger.sh first"
    if [ -e "$DATA/sing_ext/new.tsv" ]; then echo "$DATA/sing,$DATA/sing_ext"; else echo "$DATA/sing"; fi
}
