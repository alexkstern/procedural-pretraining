#!/usr/bin/env bash
# Single-pass (NO multi-epoch) C4 pretraining on one A100 80GB.
# Same 124M GPT-2, same 655M-token budget as the cycled runs, but over a large
# enough C4 cache (~1.5M docs ~= 715M tokens) that 655M tokens is <1 epoch, so
# no document is seen twice. Isolates the procedural benefit from the
# repeated-data confound.
#
# Usage:
#   ./run_a100_singlepass.sh union      # union 2500-ckpt -> C4, single pass
#   ./run_a100_singlepass.sh baseline   # C4 from scratch, single pass
#
# Requires the large cache to exist first (build it once, see BIGCACHE note):
#   python3 -m downstream.semantic.data.helpers.c4_utils cache_data \
#       --dataset_name allenai/c4 \
#       --out_dir ./downstream/semantic/data/datasets/c4_gpt2_clean_large \
#       --tokenizer_name gpt2 --c4_samples 1500000
set -euo pipefail
cd "$(dirname "$0")"

WHICH="${1:?usage: run_a100_singlepass.sh <task|baseline>}"
C4_CONFIG=downstream/semantic/configs/c4.yaml
BIGCACHE=downstream/semantic/data/datasets/c4_gpt2_clean_large

[ -d "$BIGCACHE" ] || { echo "Large C4 cache missing: $BIGCACHE — build it first (see header)." >&2; exit 1; }

COMMON=(--config "$C4_CONFIG" --use_c4_1m true --wandb_project procedural_pretraining
        --bsz 16 --gradient_accumulation_steps 2)

if [ "$WHICH" = "baseline" ]; then
    echo "== Single-pass C4 baseline (from scratch) =="
    python downstream/semantic/c4.py "${COMMON[@]}" \
        --wandb_name c4_baseline_singlepass \
        --output_dir output/c4_baseline_singlepass
else
    TASK="$WHICH"
    PROC_SAVE_DIR="pretrained_models/procedural/${TASK}/len64"
    CKPT=$(find "$PROC_SAVE_DIR" -name 'pytorch_model_*_step*.pth' 2>/dev/null | sort -V | tail -1)
    [ -n "$CKPT" ] || { echo "No $TASK checkpoint under $PROC_SAVE_DIR" >&2; exit 1; }
    echo "== Single-pass $TASK -> C4 | ckpt: $CKPT =="
    python downstream/semantic/c4.py "${COMMON[@]}" \
        --pretrained_path "$CKPT" \
        --wandb_name "${TASK}_to_c4_singlepass" \
        --output_dir "output/c4_${TASK}_singlepass"
fi
