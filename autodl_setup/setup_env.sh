#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FT_ROOT=/root/autodl-tmp/travel-finetune
BASE_PYTHON=/root/miniconda3/bin/python
FT_PYPI_INDEX_URL="${FT_PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
FT_TORCH_WHEEL_URL="${FT_TORCH_WHEEL_URL:-https://mirrors.aliyun.com/pytorch-wheels/cu130/torch-2.12.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=4bafc356fbb622e2756179406825c3a56c17b401196435a1487c5b40c657706c}"

# The user selected reuse after both wheel download routes were slow.
case "${FT_REUSE_BASE_PACKAGES:-1}" in
    1) exec bash "$SCRIPT_DIR/reuse_base_torch.sh" ;;
    0) ;; # Explicit fully isolated mode; installs a separate Torch wheel.
    *) printf 'FT_REUSE_BASE_PACKAGES must be 0 or 1\n' >&2; exit 1 ;;
esac

"$BASE_PYTHON" - <<'PY'
import platform
import sys
assert sys.version_info[:2] == (3, 12), f"Expected Python 3.12, got {sys.version}"
assert platform.system() == 'Linux', 'Run this script on the AutoDL Linux instance.'
assert platform.machine() == 'x86_64', 'Expected Linux x86_64.'
libc, version = platform.libc_ver()
assert libc == 'glibc' and tuple(map(int, version.split('.')[:2])) >= (2, 28), (
    f'Official PyTorch wheel requires glibc >= 2.28; detected {libc} {version}'
)
print('Base Python and platform check passed:', sys.version.split()[0], libc, version)
PY

nvidia-smi
df -h /root/autodl-tmp
mkdir -p "$FT_ROOT/logs" "$FT_ROOT/cache" "$FT_ROOT/tmp"

if [[ -e "$FT_ROOT/.venv" && ! -f "$FT_ROOT/.venv/pyvenv.cfg" ]]; then
    printf 'Refusing to reuse a directory without pyvenv.cfg: %s\n' "$FT_ROOT/.venv" >&2
    exit 1
fi
if [[ ! -f "$FT_ROOT/.venv/pyvenv.cfg" ]]; then
    "$BASE_PYTHON" -m venv "$FT_ROOT/.venv"
fi

cat > "$FT_ROOT/activate.sh" <<'BASH'
source /root/autodl-tmp/travel-finetune/.venv/bin/activate
export HF_HOME=/root/autodl-tmp/travel-finetune/cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/travel-finetune/cache/datasets
export MODELSCOPE_CACHE=/root/autodl-tmp/travel-finetune/cache/modelscope
export XDG_CACHE_HOME=/root/autodl-tmp/travel-finetune/cache
export TMPDIR=/root/autodl-tmp/travel-finetune/tmp
export TOKENIZERS_PARALLELISM=false
BASH

source "$FT_ROOT/activate.sh"
python - <<'PY'
import pathlib
import sys
assert sys.prefix != sys.base_prefix, 'A separate venv must be active.'
assert sys.version_info[:2] == (3, 12), 'The existing venv is not Python 3.12.'
config = (pathlib.Path(sys.prefix) / 'pyvenv.cfg').read_text().lower()
assert 'include-system-site-packages = false' in config, 'Global packages must be isolated.'
print('Isolated venv:', sys.executable)
PY

python -m pip install --no-cache-dir --upgrade pip --index-url "$FT_PYPI_INDEX_URL"
if python - <<'PY'
import importlib.metadata
import sys
try:
    installed = importlib.metadata.version('torch')
except importlib.metadata.PackageNotFoundError:
    sys.exit(1)
sys.exit(0 if installed == '2.12.1+cu130' else 1)
PY
then
    printf 'Reusing installed torch==2.12.1+cu130\n'
else
    python -m pip install --no-cache-dir --timeout 120 --retries 5 \
        "$FT_TORCH_WHEEL_URL" --index-url "$FT_PYPI_INDEX_URL"
fi
python -m pip install --no-cache-dir --timeout 120 --retries 5 \
    -r "$SCRIPT_DIR/requirements.txt" --index-url "$FT_PYPI_INDEX_URL"
python -m pip check
python "$SCRIPT_DIR/check_env.py" --report "$FT_ROOT/logs/environment.json"
python -m pip freeze > "$FT_ROOT/logs/installed-requirements.txt"
printf '\nSETUP_OK\nNew terminals: source %s/activate.sh\n' "$FT_ROOT"
