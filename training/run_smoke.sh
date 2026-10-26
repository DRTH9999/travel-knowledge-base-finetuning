#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ACTIVATE_PATH=/root/autodl-tmp/travel-finetune/activate.sh
if [[ ! -f "$ACTIVATE_PATH" ]]; then
  printf 'Missing training environment: %s\n' "$ACTIVATE_PATH" >&2
  exit 1
fi
source "$ACTIVATE_PATH"
cd "$TASK_ROOT"
mkdir -p logs
TASK_LOG="logs/train-smoke-$(date +%Y%m%d-%H%M%S).log"
printf 'Training smoke log: %s/%s\n' "$TASK_ROOT" "$TASK_LOG"
python -u training/train_sft.py "$@" 2>&1 | tee "$TASK_LOG"
