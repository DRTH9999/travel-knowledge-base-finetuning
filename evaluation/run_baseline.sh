#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /root/autodl-tmp/travel-finetune/activate.sh
cd "$TASK_ROOT"
TASK_ADAPTER="${1:-outputs/smoke_20261023_210443/adapter}"
TASK_OUTPUT="${2:-eval_outputs/validation_$(date +%Y%m%d-%H%M%S)}"
if [[ ! -f "$TASK_ADAPTER/adapter_model.safetensors" ]]; then
  printf 'Saved LoRA weights not found: %s\n' "$TASK_ADAPTER" >&2
  exit 1
fi
python evaluation/validate_eval.py
mkdir -p logs
TASK_LOG="logs/eval-validation-$(date +%Y%m%d-%H%M%S).log"
python -u evaluation/evaluate.py --split validation --adapter-dir "$TASK_ADAPTER" --output-dir "$TASK_OUTPUT" 2>&1 | tee "$TASK_LOG"
