#!/usr/bin/env bash
# Stage 2 only: C4 pretraining on a single A100 80GB from an existing
# stage-1 procedural checkpoint. Use after run_a100.sh has completed
# stage 1 (checkpoints land in a run-named subdirectory of save_dir,
# e.g. len64/set-64-12_12_768-2501steps/pytorch_model_1_step2500.pth,
# which the flat glob in run_a100.sh misses).
#
# Same batch settings as run_a100.sh: bsz 32 x grad accum 1 = effective
# batch 32 sequences, identical to the reference bsz 4 x accum 8.
set -euo pipefail
cd "$(dirname "$0")"

PROC_SAVE_DIR=pretrained_models/procedural/set/len64
C4_CONFIG=downstream/semantic/configs/c4.yaml

# Latest checkpoint by step number, searched recursively
CKPT=$(find "$PROC_SAVE_DIR" -name 'pytorch_model_*_step*.pth' | sort -V | tail -1)
[ -n "$CKPT" ] || { echo "No stage-1 checkpoint found under $PROC_SAVE_DIR" >&2; exit 1; }
echo "== Stage 1 checkpoint: $CKPT =="

echo "== Stage 2: C4 pretraining =="
python downstream/semantic/c4.py \
    --config "$C4_CONFIG" \
    --pretrained_path "$CKPT" \
    --bsz 32 \
    --gradient_accumulation_steps 1
