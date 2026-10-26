#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
TASK_BASE_PY="${TRAVEL_BASE_PYTHON:-/root/autodl-tmp/travel-finetune/.venv/bin/python}"
TASK_SERVE_ENV="$TASK_ROOT/.serving-venv"
[[ -x "$TASK_BASE_PY" ]] || { echo "Verified training Python not found: $TASK_BASE_PY"; exit 1; }
"$TASK_BASE_PY" -m serving.verify_package
"$TASK_BASE_PY" - <<'PY'
from pathlib import Path
import importlib.metadata,json
spec=json.loads(Path('serving/candidate.json').read_text(encoding='utf-8'))
for name,expected in spec['runtime_versions'].items():
    actual=importlib.metadata.version(name)
    if actual != expected:
        raise RuntimeError(f'Existing ML runtime mismatch: {name} {actual} != {expected}')
print('EXISTING_ML_RUNTIME_OK')
PY
"$TASK_BASE_PY" -m venv --system-site-packages "$TASK_SERVE_ENV"
"$TASK_BASE_PY" - "$TASK_SERVE_ENV" <<'PY'
from pathlib import Path
import site,sys
env=Path(sys.argv[1])
target=env/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
if not target.is_dir():
    raise RuntimeError(f'Expected Linux venv site-packages: {target}')
# Reuse the verified parent's ML wheels, while new HTTP packages stay in this venv.
paths=[str(Path(p).resolve()) for p in site.getsitepackages() if Path(p).is_dir()]
(target/'verified_training_runtime.pth').write_text('\n'.join(paths)+'\n',encoding='utf-8')
PY
"$TASK_SERVE_ENV/bin/python" -m pip install --no-cache-dir -r serving/requirements.txt -c serving/runtime_constraints.txt
"$TASK_SERVE_ENV/bin/python" -m pip check
"$TASK_SERVE_ENV/bin/python" - <<'PY'
from pathlib import Path
import os,secrets
folder=Path(os.getenv('TRAVEL_RUNTIME_DIR','serving_runtime'))
folder.mkdir(parents=True,exist_ok=True)
key=folder/'api.key'
try:
    with key.open('x',encoding='utf-8') as f:
        f.write(secrets.token_urlsafe(32)+'\n')
except FileExistsError:
    pass
key.chmod(0o600)
print('PRIVATE_KEY_FILE_READY (value not printed)')
PY
"$TASK_SERVE_ENV/bin/python" -m serving.preflight
TASK_RUNTIME_DIR="${TRAVEL_RUNTIME_DIR:-$TASK_ROOT/serving_runtime}"
mkdir -p "$TASK_RUNTIME_DIR"
"$TASK_SERVE_ENV/bin/python" -m pip freeze --all > "$TASK_RUNTIME_DIR/installed-requirements.txt"
echo "SERVING_SETUP_OK"
