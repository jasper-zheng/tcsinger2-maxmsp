#!/usr/bin/env bash
# Step 2 (optional): the extra Chinese corpora of EXTRA_CORPORA -> $TCS_WORK/data/sing_ext/new.tsv.
# Needs step 1 (the phone set, GTSinger's splits, and its Chinese clips for the aligner).
#
#   unpack      archive -> 48 kHz wavs in EXT_ROOT (Opencpop: the password in OPENCPOP_ZIP_PASSWORD; its
#               whole songs are cut into sentences, raw files in <EXT_ROOT>_raw)
#   metadata    M4Singer, Opencpop: note-split phonemes from the annotation, mapped to GTSinger's phonemes
#               OpenSinger: the clips that can be aligned (it has phoneme sequences, no timings, no notes)
#   splits      held-out singers (2 M4Singer, 6 OpenSinger), held-out songs agreeing with GTSinger's
#   manifests, mels, F0             as in step 1
#   OpenSinger only:
#     aligner   a frame-level phoneme classifier trained on the annotated Chinese clips (GPU, ALIGNER_STEPS)
#     align     phoneme timings of every OpenSinger clip; clips under OPENSINGER_MIN_ALIGN_SCORE are dropped
#     metadata, splits, manifest again with OpenSinger; its notes are taken from its F0, one per syllable
#   training manifest   as in step 1
#
# Steps that have finished are skipped when the script runs again; the steps that depend on which corpora are
# in EXTRA_CORPORA are redone when that changes. Output is appended to $TCS_WORK/logs/data_extra.log.
# The original build of all three took about 45 min (unpacking and metadata, CPU), 1 h (features, GPU) and
# 35 min (aligner).
#
# Usage: bash scripts-new/2_data_extra.sh
source "$(dirname "$0")/lib.sh"
log_to data_extra

[ -n "$EXTRA_CORPORA" ] || { log "EXTRA_CORPORA is empty: nothing to do"; exit 0; }
need "$DATA/sing/new.tsv" "run scripts-new/1_data_gtsinger.sh first"
has() { [[ " $EXTRA_CORPORA " == *" $1 "* ]]; }
NAMES=(); for c in $EXTRA_CORPORA; do
    case $c in m4singer) NAMES+=(M4Singer) ;; opencpop) NAMES+=(Opencpop) ;; opensinger) NAMES+=(OpenSinger) ;; *) die "unknown corpus $c in EXTRA_CORPORA" ;; esac
done
SET=$(tr ' ' '\n' <<< "$EXTRA_CORPORA" | grep . | sort | paste -sd+)   # e.g. m4singer+opensinger, part of the marker names
ANNOTATED=$(has m4singer || has opencpop && echo 1 || true)                 # corpora with phoneme timings and notes
build() { python preprocess/ext_build_metadata.py --data_root "$EXT_ROOT" --out_dir "$META_EXT" --phone_set "$META/phone_set.json" --workers "$WORKERS" "$@"; }
splits() { python preprocess/ext_split_report.py --meta_dir "$META_EXT" --gtsinger_meta "$META/metadata.json" --corpora "$@"; }
manifest() { python scripts-new/py/make_manifest.py --data_root "$EXT_ROOT" --mel_root "$EXT_FEATURES" "$@"; }
f0() { python preprocess/extract_f0.py --data_root "$EXT_ROOT" --out_root "$EXT_FEATURES/f0" --num_workers "$WORKERS" "$@"; }
postprocess() { python preprocess/postprocess.py --data_root "$EXT_ROOT" --f0_root "$EXT_FEATURES/f0" --workers "$WORKERS" "$@"; }

# ---- audio and metadata of each corpus
if has m4singer; then
    if [ -n "$M4SINGER_ZIP" ]; then
        step m4singer_unpack python -I preprocess/ext_unpack.py --dataset m4singer --archive "$M4SINGER_ZIP" --out_root "$EXT_ROOT" --workers "$WORKERS"
    fi
    need "$EXT_ROOT/M4Singer/meta.json" "set M4SINGER_ZIP in scripts-new/config.sh"
    step m4singer_metadata build --dataset m4singer
fi
if has opencpop; then
    if [ -n "$OPENCPOP_DIR" ]; then
        step opencpop_unpack python -I preprocess/ext_unpack.py --dataset opencpop --archive "$OPENCPOP_DIR" --out_root "$EXT_ROOT" --workers "$WORKERS"
    fi
    need "$EXT_ROOT/Opencpop/segments.json" "set OPENCPOP_DIR in scripts-new/config.sh"
    need "${EXT_ROOT}_raw/Opencpop/textgrids" "set OPENCPOP_DIR in scripts-new/config.sh"
    step opencpop_metadata build --dataset opencpop
fi
if has opensinger; then
    if [ -n "$OPENSINGER_TAR" ]; then
        step opensinger_unpack python -I preprocess/ext_unpack.py --dataset opensinger --archive "$OPENSINGER_TAR" --out_root "$EXT_ROOT" --workers "$WORKERS"
    fi
    need "$EXT_ROOT/OpenSinger/labels.json" "set OPENSINGER_TAR in scripts-new/config.sh"
    step opensinger_clips build --dataset opensinger_audio
fi

# ---- splits, manifests and features of the annotated corpora, and of the OpenSinger clips to align
ANNOTATED_NAMES=($(printf '%s\n' "${NAMES[@]}" | grep -v OpenSinger || true))
if [ -n "$ANNOTATED" ]; then
    step "ext_splits.$SET" splits "${ANNOTATED_NAMES[@]}"
    step "ext_manifest.$SET" manifest --metadata "$META_EXT/metadata.json" --out_tsv "$DATA/new_ext/data.tsv"
    step "ext_mels.$SET" python scripts-new/py/extract_mels.py --tsv "$DATA/new_ext/data.tsv" --num_workers "$WORKERS"
    step "ext_f0.$SET" f0 --metadata "$META_EXT/metadata.json"
fi

if has opensinger; then
    step opensinger_clips_manifest manifest --metadata "$META_EXT/opensinger_audio.json" --out_tsv "$DATA/new_ext_audio/opensinger.tsv"
    step opensinger_clips_mels python scripts-new/py/extract_mels.py --tsv "$DATA/new_ext_audio/opensinger.tsv" --num_workers "$WORKERS"
    step opensinger_clips_f0 f0 --metadata "$META_EXT/opensinger_audio.json"

    # ---- the aligner, trained on GTSinger's Chinese clips and the annotated corpora
    ALIGN_DATA=("$DATA/sing/new.tsv")
    if [ -n "$ANNOTATED" ]; then
        step "ext_training_manifest_annotated.$SET" postprocess --in_tsv "$DATA/new_ext/data.tsv" --out_tsv "$DATA/sing_ext_m4_ocp/new.tsv" \
            --datasets "${ANNOTATED_NAMES[@]}" --report "$META_EXT/postprocess_report_m4_ocp.json"
        ALIGN_DATA+=("$DATA/sing_ext_m4_ocp/new.tsv")
    fi
    step "aligner.$SET" python preprocess/aligner/train.py --manifests "${ALIGN_DATA[@]}" --augment --steps "$ALIGNER_STEPS" \
        --out "$CKPTS/aligner_zh.pt" --num_workers "$WORKERS"
    step "opensinger_align.$SET" python preprocess/aligner/align.py --ckpt "$CKPTS/aligner_zh.pt" --clips "$META_EXT/opensinger_audio.json" \
        --data_root "$EXT_ROOT" --mel_root "$EXT_FEATURES" --out "$META_EXT/opensinger_align.json"
    step "opensinger_metadata.$SET" build --dataset opensinger --min_align_score "$OPENSINGER_MIN_ALIGN_SCORE"

    # ---- splits and manifest of all corpora (the mels and F0 of OpenSinger exist already)
    step "ext_splits_all.$SET" splits "${NAMES[@]}"
    step "ext_manifest_all.$SET" manifest --metadata "$META_EXT/metadata.json" --out_tsv "$DATA/new_ext/data.tsv"
fi

step "ext_training_manifest.$SET" postprocess --in_tsv "$DATA/new_ext/data.tsv" --out_tsv "$DATA/sing_ext/new.tsv" \
    --report "$META_EXT/postprocess_report.json"
python scripts-new/py/build_lexicon.py --metadata "$META/metadata.json" "$META_EXT/metadata.json" --out "$META/lexicon.json"
log "extra corpora done: $DATA/sing_ext/new.tsv (report: $META_EXT/report.md, $META_EXT/postprocess_report.json)"
