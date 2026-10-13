#!/usr/bin/env bash
# Mel and F0 extraction for the extra corpora (WP9). Every step skips what is already there, so the
# script can be run again after an interruption. Output is appended to logs/wp9_features.log.
#
# Usage: nohup bash scripts/wp9_features.sh > /dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source /notebooks/envs/tcsinger2/bin/activate
export PYTHONPATH=.
R=/notebooks/data/ext
P=/notebooks/data/ext_processed
log=logs/wp9_features.log
{
    echo "==== $(date '+%F %T') start"
    # data/new_ext: clips with metadata (M4Singer, Opencpop, later OpenSinger); data/new_ext_audio: OpenSinger before alignment
    for tsv in data/new_ext data/new_ext_audio; do
        if ls "$tsv"/*.tsv > /dev/null 2>&1; then
            python preprocess/mel_spec_48k.py --tsv_path "$tsv" --num_gpus 1 --max_duration 20 --data_root $R --save_path $P
        fi
    done
    for meta in data/meta_ext/metadata.json data/meta_ext/opensinger_audio.json; do
        if [ -f "$meta" ]; then
            python preprocess/extract_f0.py --metadata "$meta" --data_root $R --out_root $P/f0
        fi
    done
    echo "==== $(date '+%F %T') done"
} >> "$log" 2>&1
