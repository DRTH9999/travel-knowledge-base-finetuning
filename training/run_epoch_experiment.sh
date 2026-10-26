#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /root/autodl-tmp/travel-finetune/activate.sh
cd "$TASK_ROOT"
TASK_TAG="${1:-revision_v2_epoch1}"
if [[ ! "$TASK_TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
  printf 'Use letters, numbers, underscores or hyphens for the experiment name.\n' >&2
  exit 1
fi
TASK_OUTPUT="outputs/$TASK_TAG"
TASK_EVAL="eval_outputs/${TASK_TAG}_512"
TASK_ARCHIVE="/root/autodl-tmp/qwen3_${TASK_TAG}_results.tar.gz"
for TASK_PATH in "$TASK_OUTPUT" "$TASK_EVAL" "$TASK_ARCHIVE" "${TASK_ARCHIVE}.sha256"; do
  if [[ -e "$TASK_PATH" ]]; then
    printf 'Existing result: %s. Choose a new experiment name.\n' "$TASK_PATH" >&2
    exit 1
  fi
done
sha256sum -c epoch_pipeline/checksums.sha256
python training/train_revision_epoch.py --validate-only
bash training/run_revision_epoch.sh "$TASK_OUTPUT"
TASK_EVAL_LOG="logs/eval-epoch-${TASK_TAG}-$(date +%Y%m%d-%H%M%S).log"
python -u evaluation/evaluate.py --split validation \
  --adapter-dir "$TASK_OUTPUT/adapter" --max-new-tokens 512 \
  --output-dir "$TASK_EVAL" 2>&1 | tee "$TASK_EVAL_LOG"
tar -czf "$TASK_ARCHIVE" \
  "$TASK_OUTPUT" "$TASK_EVAL" logs \
  data/revision_v2 data/eval_v1/manifest.json data/eval_v1/validation.jsonl \
  epoch_pipeline training/train_revision_epoch.py training/run_revision_epoch.sh training/run_epoch_experiment.sh \
  revision_pipeline/common.py revision_pipeline/tokenize_revision.py revision_pipeline/legacy_training_core.py \
  evaluation/common.py evaluation/evaluate.py
sha256sum "$TASK_ARCHIVE" > "${TASK_ARCHIVE}.sha256"
printf 'EPOCH_EXPERIMENT_COMPLETE\nResults: %s\n' "$TASK_ARCHIVE"
