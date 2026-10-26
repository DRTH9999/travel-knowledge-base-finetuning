"""Check real imports and NF4 GPU operations in the cloud training venv."""
import argparse
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Activate the separate training venv first.')

    import torch
    import bitsandbytes as bnb
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer
    from modelscope import snapshot_download

    # Importing these public APIs also checks lazy imports used by training.
    apis = [Dataset, LoraConfig, get_peft_model, AutoModelForCausalLM,
            AutoTokenizer, BitsAndBytesConfig, SFTConfig, SFTTrainer,
            snapshot_download]
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this venv.')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is unavailable; review the selected environment.')

    layer = bnb.nn.Linear4bit(
        64, 32, bias=False, compute_dtype=torch.bfloat16,
        quant_type='nf4', compress_statistics=True,
    ).to('cuda:0')
    inputs = torch.randn(2, 64, device='cuda:0', dtype=torch.bfloat16,
                         requires_grad=True)
    output = layer(inputs)
    output.float().square().mean().backward()
    torch.cuda.synchronize()
    if inputs.grad is None or not torch.isfinite(inputs.grad).all().item():
        raise RuntimeError('NF4 input-gradient check failed.')
    if not torch.isfinite(output).all().item():
        raise RuntimeError('NF4 forward check returned non-finite values.')

    gpu = torch.cuda.get_device_properties(0)
    packages = ['torch', 'transformers', 'trl', 'peft', 'accelerate',
                'datasets', 'bitsandbytes', 'modelscope']
    report = {
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'executable': sys.executable,
        'venv_configuration': (Path(sys.prefix) / 'pyvenv.cfg').read_text(),
        'torch_module_path': torch.__file__,
        'package_install_roots': {
            p: str(importlib.metadata.distribution(p).locate_file('')) for p in packages
        },
        'packages': {p: importlib.metadata.version(p) for p in packages},
        'torch_cuda_runtime': torch.version.cuda,
        'gpu': gpu.name,
        'vram_gib': round(gpu.total_memory / 1024**3, 2),
        'compute_capability': list(torch.cuda.get_device_capability(0)),
        'bf16_supported': True,
        'public_api_imports': [api.__name__ for api in apis],
        'nf4_small_layer_forward_and_input_backward': 'passed',
        'scope': 'Environment check only; no Qwen3 model or dataset was loaded.',
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('ENV_CHECK_OK')


if __name__ == '__main__':
    main()
