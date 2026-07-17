#!/usr/bin/env bash
# Chains the two default stages of the repo on a single A100 80GB:
#   1. Procedural pretraining on the `set` task (stock config).
#   2. Standard C4 pretraining from that checkpoint.
#
# Everything uses the stock configs. The only deviation from the default
# C4 run: the per-device batch is raised 4 -> 32 and gradient accumulation
# lowered 8 -> 1. The effective batch is unchanged (32 sequences), so the
# optimization trajectory matches the reference setup — this just trades
# accumulation overhead for one large batch, which an 80GB card fits easily.
#
# Requirements before running:
#   pip install -r requirements.txt
#   export WANDB_API_KEY=...   (or run `wandb login`; configs have wandb on)
set -euo pipefail
cd "$(dirname "$0")"

PROC_CONFIG=procedural_pretraining/configs/set.yaml
PROC_SAVE_DIR=pretrained_models/procedural/set/len64
C4_CONFIG=downstream/semantic/configs/c4.yaml
C4_DATA_DIR=downstream/semantic/data/datasets/c4_gpt2_clean

# --- C4 data prep (one-time; ~100k tokenized samples for gpt2-base) ---
if [ ! -d "$C4_DATA_DIR" ]; then
    echo "== Preparing C4 dataset (100k samples) =="
    python3 -m downstream.semantic.data.helpers.c4_utils cache_data \
        --dataset_name allenai/c4 \
        --out_dir "./$C4_DATA_DIR" \
        --tokenizer_name gpt2 \
        --c4_samples 100000
fi

# --- Stage 1: procedural pretraining (skipped if a checkpoint already exists) ---
# Checkpoints land in a run-named subdirectory of save_dir,
# e.g. len64/set-64-12_12_768-2501steps/pytorch_model_1_step2500.pth
find_ckpt() {
    find "$PROC_SAVE_DIR" -name 'pytorch_model_*_step*.pth' 2>/dev/null | sort -V | tail -1
}

CKPT=$(find_ckpt)
if [ -z "$CKPT" ]; then
    echo "== Stage 1: procedural pretraining (set task) =="
    python -m procedural_pretraining.cli --config "$PROC_CONFIG"
    CKPT=$(find_ckpt)
fi
[ -n "$CKPT" ] || { echo "No stage-1 checkpoint found under $PROC_SAVE_DIR" >&2; exit 1; }
echo "== Stage 1 checkpoint: $CKPT =="

# --- Stage 2: C4 pretraining ---
echo "== Stage 2: C4 pretraining =="
python downstream/semantic/c4.py \
    --config "$C4_CONFIG" \
    --pretrained_path "$CKPT" \
    --bsz 32 \
    --gradient_accumulation_steps 1
