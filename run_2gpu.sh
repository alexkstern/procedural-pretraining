#!/usr/bin/env bash
# Chains the two default stages of the repo on a 2-GPU machine:
#   1. Procedural pretraining on the `set` task (single GPU — the training
#      code has no multi-GPU support, and the stage only takes minutes).
#   2. Standard C4 pretraining from that checkpoint, on 2 GPUs via torchrun.
#
# Everything uses the stock configs. The only deviation from the default
# single-GPU C4 run: gradient_accumulation_steps is halved (8 -> 4) so that
# the effective batch stays at 2 GPUs x bsz 4 x accum 4 = 32 sequences,
# identical to the reference setup.
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

# --- Stage 1: procedural pretraining (GPU 0) ---
echo "== Stage 1: procedural pretraining (set task, single GPU) =="
CUDA_VISIBLE_DEVICES=0 python -m procedural_pretraining.cli --config "$PROC_CONFIG"

# Latest checkpoint by step number (pytorch_model_<epoch>_step<step>.pth)
CKPT=$(ls -1 "$PROC_SAVE_DIR"/pytorch_model_*_step*.pth | sort -V | tail -1)
echo "== Stage 1 checkpoint: $CKPT =="

# --- Stage 2: C4 pretraining (2 GPUs) ---
echo "== Stage 2: C4 pretraining on 2 GPUs =="
torchrun --nproc_per_node 2 downstream/semantic/c4.py \
    --config "$C4_CONFIG" \
    --pretrained_path "$CKPT" \
    --gradient_accumulation_steps 4
