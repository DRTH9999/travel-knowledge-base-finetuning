#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /root/autodl-tmp/travel-finetune/activate.sh
cd "$TASK_ROOT"
TASK_OUTPUT="${1:-outputs/revision_v2_epoch1}"
mkdir -p logs
TASK_LOG="logs/train-revision-epoch1-$(date +%Y%m%d-%H%M%S).log"
python -u training/train_revision_epoch.py --epochs 1 --output-dir "$TASK_OUTPUT" 2>&1 | tee "$TASK_LOG"
python epoch_pipeline/verify_coverage.py --output-dir "$TASK_OUTPUT"
