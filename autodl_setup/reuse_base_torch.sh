#!/usr/bin/env bash
# User-confirmed mode: venv may read the original image's site-packages.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FT_ROOT=/root/autodl-tmp/travel-finetune
BASE_PYTHON=/root/miniconda3/bin/python
FT_PYPI_INDEX_URL="${FT_PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"

"$BASE_PYTHON" - <<'PY'
import importlib.metadata
import sys
import torch
assert sys.version_info[:2] == (3, 12), 'Expected the original Python 3.12 interpreter.'
assert importlib.metadata.version('torch') == '2.12.1+cu130', 'Unexpected base Torch version.'
assert torch.cuda.is_available(), 'The base Torch cannot use CUDA.'
assert torch.cuda.is_bf16_supported(), 'The base Torch cannot use BF16.'
assert (torch.ones(1, device='cuda') + 1).item() == 2
print('Base Torch verified:', torch.__version__, torch.__file__)
PY

mkdir -p "$FT_ROOT/logs" "$FT_ROOT/cache" "$FT_ROOT/tmp"
if [[ -e "$FT_ROOT/.venv" && ! -f "$FT_ROOT/.venv/pyvenv.cfg" ]]; then
    printf 'Refusing to reuse a directory without pyvenv.cfg: %s\n' "$FT_ROOT/.venv" >&2
    exit 1
fi
"$BASE_PYTHON" -m venv --system-site-packages "$FT_ROOT/.venv"

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
import importlib.metadata
import pathlib
import sys
import torch
assert sys.prefix != sys.base_prefix, 'A separate venv must be active.'
config = (pathlib.Path(sys.prefix) / 'pyvenv.cfg').read_text().lower()
assert 'include-system-site-packages = true' in config, 'Base package reuse is not enabled.'
assert importlib.metadata.version('torch') == '2.12.1+cu130', 'The venv sees an unexpected Torch.'
assert torch.cuda.is_available(), 'CUDA is unavailable from the venv.'
print('Venv:', sys.executable)
print('Reused Torch:', torch.__version__, torch.__file__)
PY

# No --upgrade or --ignore-installed: keep satisfying inherited packages.
python -m pip install --no-cache-dir --timeout 120 --retries 5 \
    -r "$SCRIPT_DIR/requirements.txt" --index-url "$FT_PYPI_INDEX_URL"
python -m pip check
python "$SCRIPT_DIR/check_env.py" --report "$FT_ROOT/logs/environment.json"
# --all records inherited packages too; --local records only venv-local packages.
python -m pip freeze --all > "$FT_ROOT/logs/installed-requirements.txt"
python -m pip freeze --local > "$FT_ROOT/logs/venv-local-requirements.txt"
printf '\nSETUP_OK\nBase-package reuse is enabled.\nNew terminals: source %s/activate.sh\n' "$FT_ROOT"
