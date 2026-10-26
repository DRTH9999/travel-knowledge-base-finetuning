#!/usr/bin/env bash
set -euo pipefail

source /root/autodl-tmp/travel-finetune/activate.sh
cd /root/autodl-tmp/travel-finetune/sft_pilot_v1
mkdir -p logs
TASK_LOG="logs/eval-validation-round2-512-$(date +%Y%m%d-%H%M%S).log"
python -u evaluation/evaluate.py \
  --split validation \
  --adapter-dir outputs/smoke_20261023_210443/adapter \
  --max-new-tokens 512 \
  --output-dir eval_outputs/validation_round2_512 \
  2>&1 | tee "$TASK_LOG"
