#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
TASK_PY="$TASK_ROOT/.serving-venv/bin/python"
[[ -x "$TASK_PY" ]] || { echo "Run bash serving/setup_cloud.sh first."; exit 1; }
"$TASK_PY" -m serving.verify_package
"$TASK_PY" - <<'PY'
import os,socket
host=os.getenv('TRAVEL_HOST','127.0.0.1')
port=int(os.getenv('TRAVEL_PORT','8008'))
with socket.socket() as sock:
    sock.bind((host,port))
print('SERVING_PORT_AVAILABLE')
PY
export PYTHONUNBUFFERED=1
exec "$TASK_PY" -m serving.app
