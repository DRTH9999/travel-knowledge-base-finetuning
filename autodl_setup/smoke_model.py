"""Check local Qwen3-8B NF4 loading and a short non-thinking generation."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path,
                        default=Path('/root/autodl-tmp/models/Qwen3-8B'))
    parser.add_argument('--report', type=Path,
                        default=Path('/root/autodl-tmp/travel-finetune/logs/model-smoke.json'))
    args = parser.parse_args()

    import torch
    import bitsandbytes as bnb
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('The checked CUDA/BF16 environment must be available.')
    model_path = str(args.model_dir.resolve())
    config = AutoConfig.from_pretrained(model_path, local_files_only=True,
                                         trust_remote_code=False)
    if config.model_type != 'qwen3':
        raise RuntimeError('Expected the selected Qwen3 model.')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                               trust_remote_code=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa',
    ).eval()
    load_seconds = time.perf_counter() - started
    quantized_layers = sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())
    if not getattr(model, 'is_loaded_in_4bit', False) or quantized_layers == 0:
        raise RuntimeError('The model was not loaded with bitsandbytes 4-bit layers.')

    messages = [{'role': 'user', 'content': '请只回复：环境测试通过。'}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda:0')
    torch.manual_seed(42)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=64, do_sample=True,
            temperature=0.7, top_p=0.8, top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started
    generated = output[0, inputs['input_ids'].shape[-1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not answer or '<think>' in answer or '</think>' in answer:
        raise RuntimeError(f'Check the non-thinking output: {answer!r}')

    report = {
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'model_path': model_path,
        'torch_version': torch.__version__,
        'quantization': 'bitsandbytes NF4, double quantization, BF16 compute',
        'attention': 'PyTorch SDPA', 'enable_thinking': False,
        'quantized_linear_layers': quantized_layers,
        'load_seconds': round(load_seconds, 2),
        'generation_seconds': round(generation_seconds, 2),
        'generated_tokens': int(generated.numel()),
        'peak_allocated_vram_gib': round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        'output': answer,
        'scope': 'Short inference smoke check only; not fine-tuning or a quality evaluation.',
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('MODEL_SMOKE_OK')


if __name__ == '__main__':
    main()
