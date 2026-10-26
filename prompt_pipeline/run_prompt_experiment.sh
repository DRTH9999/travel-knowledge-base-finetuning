#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /root/autodl-tmp/travel-finetune/activate.sh
cd "$TASK_ROOT"
TASK_TAG="${1:-prompt_v2_validation}"
if [[ ! "$TASK_TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
  printf 'Use letters, numbers, underscores or hyphens for the experiment name.\n' >&2
  exit 1
fi
TASK_OUTPUT="eval_outputs/$TASK_TAG"
TASK_ARCHIVE="/root/autodl-tmp/qwen3_${TASK_TAG}_results.tar.gz"
for TASK_PATH in "$TASK_OUTPUT" "$TASK_ARCHIVE" "${TASK_ARCHIVE}.sha256"; do
  if [[ -e "$TASK_PATH" ]]; then
    printf 'Existing result: %s. Choose a new experiment name.\n' "$TASK_PATH" >&2
    exit 1
  fi
done
sha256sum -c prompt_pipeline/checksums.sha256
mkdir -p logs
TASK_LOG="logs/prompt-${TASK_TAG}-$(date +%Y%m%d-%H%M%S).log"
python -u prompt_pipeline/evaluate_prompts.py --output-dir "$TASK_OUTPUT" 2>&1 | tee "$TASK_LOG"
python prompt_pipeline/validate_results.py --output-dir "$TASK_OUTPUT"
tar -czf "$TASK_ARCHIVE" \
  "$TASK_OUTPUT" "$TASK_LOG" prompt_pipeline \
  data/eval_v1/manifest.json data/eval_v1/validation.jsonl data/eval_v1/prompt_template.txt \
  evaluation/common.py evaluation/evaluate.py \
  eval_outputs/revision_v2_epoch1_512/report.json \
  eval_outputs/revision_v2_epoch1_512/adapter.predictions.jsonl \
  outputs/revision_v2_epoch1/epoch_report.json \
  outputs/revision_v2_epoch1/adapter/adapter_config.json
sha256sum "$TASK_ARCHIVE" > "${TASK_ARCHIVE}.sha256"
printf 'PROMPT_EXPERIMENT_COMPLETE\nResults: %s\n' "$TASK_ARCHIVE"
