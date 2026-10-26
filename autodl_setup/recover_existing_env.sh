bash <<'BASH'
set -euo pipefail
/root/miniconda3/bin/python -m venv --system-site-packages /root/autodl-tmp/travel-finetune/.venv
source /root/autodl-tmp/travel-finetune/activate.sh
python - <<'PY'
import importlib.metadata
import sys
import torch
assert sys.prefix != sys.base_prefix, 'The training venv must be active.'
assert importlib.metadata.version('torch') == '2.12.1+cu130', 'Unexpected Torch version.'
assert torch.cuda.is_available(), 'CUDA is unavailable.'
print('Python:', sys.executable)
print('Torch:', torch.__version__)
print('Torch path:', torch.__file__)
print('CUDA:', torch.version.cuda)
PY
python -m pip install --no-cache-dir --timeout 120 --retries 5 \
  -r /root/autodl-tmp/autodl_setup/requirements.txt \
  --index-url https://mirrors.aliyun.com/pypi/simple/
python -m pip check
python /root/autodl-tmp/autodl_setup/check_env.py --report /root/autodl-tmp/travel-finetune/logs/environment.json
python -m pip freeze --all > /root/autodl-tmp/travel-finetune/logs/installed-requirements.txt
python -m pip freeze --local > /root/autodl-tmp/travel-finetune/logs/venv-local-requirements.txt
printf '\nSETUP_OK\n'
BASH
