#!/usr/bin/env bash
# Overnight: union -> C4, then sort -> C4, sequentially on one A100.
# Each is procedural warm-up (2501 steps) followed by C4 pretraining (10k steps).
set -euo pipefail
cd "$(dirname "$0")"

./run_a100_task.sh union
./run_a100_task.sh sort

echo "== All done: union_to_c4 and sort_to_c4 complete =="
