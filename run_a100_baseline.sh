#!/usr/bin/env bash
# Baseline C4 pretraining on a single A100 80GB: random init, no procedural
# checkpoint (pretrained_path stays null as in the stock c4.yaml). This is
# the comparison run for run_a100_stage2.sh.
#
# Same batch settings: bsz 16 x grad accum 2 = effective batch 32 sequences,
# identical to the reference bsz 4 x accum 8.
set -euo pipefail
cd "$(dirname "$0")"

C4_CONFIG=downstream/semantic/configs/c4.yaml

echo "== Baseline: C4 pretraining from scratch =="
python downstream/semantic/c4.py \
    --config "$C4_CONFIG" \
    --wandb_project procedural_pretraining \
    --wandb_name c4_baseline \
    --output_dir output/c4_baseline \
    --bsz 16 \
    --gradient_accumulation_steps 2
