#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
TASK_PY="$TASK_ROOT/.serving-venv/bin/python"
"$TASK_PY" -m serving.smoke "$@"
