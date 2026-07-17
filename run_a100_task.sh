#!/usr/bin/env bash
# Chain procedural pretraining on ONE task -> C4 pretraining, single A100 80GB.
# Usage: ./run_a100_task.sh <task>      e.g. ./run_a100_task.sh union
#
# Stage 1 uses the stock procedural_pretraining/configs/<task>.yaml.
# Stage 2 is the stock C4 config with the resulting checkpoint transferred in.
# Batch: bsz 16 x grad accum 2 = effective batch 32 sequences (paper-faithful;
# bsz 32 OOMs at seq 2048 on 80GB). All runs log to the procedural_pretraining
# W&B project.
set -euo pipefail
cd "$(dirname "$0")"

TASK="${1:?usage: run_a100_task.sh <task>  (e.g. union, sort, set)}"
PROC_CONFIG="procedural_pretraining/configs/${TASK}.yaml"
PROC_SAVE_DIR="pretrained_models/procedural/${TASK}/len64"
C4_CONFIG=downstream/semantic/configs/c4.yaml
C4_DATA_DIR=downstream/semantic/data/datasets/c4_gpt2_clean

[ -f "$PROC_CONFIG" ] || { echo "No config: $PROC_CONFIG" >&2; exit 1; }

# --- C4 data prep (one-time; shared across tasks) ---
if [ ! -d "$C4_DATA_DIR" ]; then
    echo "== Preparing C4 dataset (100k samples) =="
    python3 -m downstream.semantic.data.helpers.c4_utils cache_data \
        --dataset_name allenai/c4 \
        --out_dir "./$C4_DATA_DIR" \
        --tokenizer_name gpt2 \
        --c4_samples 100000
fi

# --- Stage 1: procedural pretraining (skipped if a checkpoint already exists) ---
# Checkpoints land in a run-named subdir, e.g.
# union/len64/union-64-12_12_768-2501steps/pytorch_model_1_step2500.pth
find_ckpt() { find "$PROC_SAVE_DIR" -name 'pytorch_model_*_step*.pth' 2>/dev/null | sort -V | tail -1; }

CKPT=$(find_ckpt)
if [ -z "$CKPT" ]; then
    echo "== Stage 1: procedural pretraining ($TASK) =="
    python -m procedural_pretraining.cli --config "$PROC_CONFIG"
    CKPT=$(find_ckpt)
fi
[ -n "$CKPT" ] || { echo "No stage-1 checkpoint under $PROC_SAVE_DIR" >&2; exit 1; }
echo "== Stage 1 checkpoint ($TASK): $CKPT =="

# --- Stage 2: C4 pretraining from the procedural checkpoint ---
echo "== Stage 2: C4 pretraining ($TASK -> C4) =="
python downstream/semantic/c4.py \
    --config "$C4_CONFIG" \
    --pretrained_path "$CKPT" \
    --wandb_project procedural_pretraining \
    --wandb_name "${TASK}_to_c4" \
    --output_dir "output/c4_${TASK}" \
    --bsz 16 \
    --gradient_accumulation_steps 2
