#!/usr/bin/env bash
# Step 1: GTSinger -> training manifest $TCS_WORK/data/sing/new.tsv, with mels and F0.
#
#   unzip       the nine language zips into GTSINGER_ROOT (only if GTSINGER_ZIPS is set)
#   metadata    note-split phonemes with pitches and lengths, cleaned annotations, phone set
#   splits      train / valid / test_seen / test_unseen (held-out singers: GTSINGER_UNSEEN), data report
#   manifest    one row per clip with its mel path
#   mels        80-bin mels, loudness-normalised, cut at 20 s (GPU)
#   f0          RMVPE F0 at the mel frame rate (GPU)
#   training manifest   filters, captions, missing note pitches from F0, prompt lists (about 3 min)
#   lexicon     pronunciations of the training lyrics, for inference
#
# Every step that has finished is skipped when the script runs again (markers in $TCS_WORK/done); the mel and
# F0 steps also resume inside, clip by clip. Output is appended to $TCS_WORK/logs/data_gtsinger.log.
#
# Usage: bash scripts-new/1_data_gtsinger.sh
source "$(dirname "$0")/lib.sh"
log_to data_gtsinger

if [ -n "$GTSINGER_ZIPS" ]; then
    step gtsinger_unzip python -I preprocess/gtsinger_unzip.py --zip_dir "$GTSINGER_ZIPS" --out_dir "$GTSINGER_ROOT" --workers "$WORKERS"
fi
[ -d "$GTSINGER_ROOT" ] || die "no GTSinger tree at GTSINGER_ROOT=$GTSINGER_ROOT (set GTSINGER_ZIPS or GTSINGER_ROOT in scripts-new/config.sh)"

step gtsinger_metadata python preprocess/gtsinger_build_metadata.py --data_root "$GTSINGER_ROOT" --out_dir "$META" --workers "$WORKERS"
step gtsinger_splits python preprocess/gtsinger_split_report.py --meta_dir "$META" --unseen_singers $GTSINGER_UNSEEN
step gtsinger_manifest python scripts-new/py/make_manifest.py --metadata "$META/metadata.json" --data_root "$GTSINGER_ROOT" \
    --mel_root "$GTS_FEATURES" --out_tsv "$DATA/new/data.tsv"
step gtsinger_mels python scripts-new/py/extract_mels.py --tsv "$DATA/new/data.tsv" --num_workers "$WORKERS"
step gtsinger_f0 python preprocess/extract_f0.py --metadata "$META/metadata.json" --data_root "$GTSINGER_ROOT" \
    --out_root "$GTS_FEATURES/f0" --num_workers "$WORKERS"
step gtsinger_training_manifest python preprocess/postprocess.py --in_tsv "$DATA/new/data.tsv" --out_tsv "$DATA/sing/new.tsv" \
    --data_root "$GTSINGER_ROOT" --f0_root "$GTS_FEATURES/f0" --report "$META/postprocess_report.json" --workers "$WORKERS"

# Rebuilt every time (cheap): 2_data_extra.sh adds the extra corpora to it.
python scripts-new/py/build_lexicon.py --metadata "$META/metadata.json" $(ls "$META_EXT/metadata.json" 2>/dev/null) --out "$META/lexicon.json"
log "GTSinger done: $DATA/sing/new.tsv (report: $META/report.md, $META/postprocess_report.json)"
