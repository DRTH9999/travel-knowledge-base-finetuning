"""Train exactly one epoch on frozen revision_v2 with observed sample coverage."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epoch_pipeline.preflight import preflight
from revision_pipeline.common import json_bytes, read_jsonl, require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'data/revision_v2')
    parser.add_argument('--model-dir', type=Path, default=Path('/root/autodl-tmp/models/Qwen3-8B'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--epochs', type=int, choices=[1], default=1)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    manifest, raw, tokens, checks = preflight(ROOT, args.dataset_dir)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    print('EPOCH_TRAIN_PREFLIGHT_OK', flush=True)
    if args.validate_only:
        return
    require(os.environ.get('WORLD_SIZE', '1') == '1', 'Use one GPU process for this experiment.')
    require(sys.prefix != sys.base_prefix, 'Activate the existing training venv first.')
    output = (args.output_dir or ROOT / 'outputs' / ('revision_v2_epoch1_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    require(not output.exists() or not any(output.iterdir()), 'Output directory must be new or empty; preserve earlier results.')
    from revision_pipeline.tokenize_revision import export
    tokenizer_check = export(args.dataset_dir, args.model_dir, check_only=True)
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, set_seed
    from revision_pipeline.legacy_training_core import load_base, verify_saved_adapter
    from epoch_pipeline.training_core import run_epoch
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA/BF16 required.')
    for package, version in (('transformers', '5.20.2'), ('trl', '1.16.0'), ('peft', '0.21.2'), ('accelerate', '1.15.0')):
        require(importlib.metadata.version(package) == version, f'Expected {package} {version}.')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = dict(status='RUNNING', experiment='revision_v2_epoch1_v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
                  dataset_version=manifest['dataset_version'], dataset=checks['dataset'], token_checks=checks['tokens'], plan=checks['plan'],
                  tokenizer_recheck=tokenizer_check['status'], initialization='fresh_base_and_fresh_lora_no_adapter_resume',
                  model_dir=str(args.model_dir.resolve()), output_dir=str(output),
                  versions={p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'trl', 'peft', 'accelerate', 'datasets', 'bitsandbytes')},
                  model_config_sha256=sha((args.model_dir / 'config.json').read_bytes()),
                  dataset_manifest_sha256=sha((args.dataset_dir / 'source_manifest.json').read_bytes()),
                  eval_manifest_sha256=sha((ROOT / 'data/eval_v1/manifest.json').read_bytes()),
                  training_code_sha256={p: sha((ROOT / p).read_bytes()) for p in (
                    'training/train_revision_epoch.py', 'epoch_pipeline/training_core.py', 'epoch_pipeline/coverage.py',
                    'epoch_pipeline/preflight.py', 'revision_pipeline/legacy_training_core.py')},
                  quality_evaluation_performed=False)
    (output / 'epoch_report.json').write_bytes(json_bytes(report))
    try:
        set_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
        tokenizer.padding_side = 'right'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        torch.cuda.reset_peak_memory_stats()
        model = prepare_model_for_kbit_training(load_base(args.model_dir), use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False})
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules='all-linear', bias='none', task_type='CAUSAL_LM'))
        model.print_trainable_parameters()
        report['training'] = run_epoch(model, tokenizer, tokens, raw, checks['plan'], output)
        torch.cuda.synchronize()
        report['training_peak_allocated_vram_gib'] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        report['training_peak_reserved_vram_gib'] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        report['training_and_load_seconds'] = round(time.perf_counter() - started, 2)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = PeftModel.from_pretrained(load_base(args.model_dir), str(output / 'adapter'), is_trainable=False,
            autocast_adapter_dtype=False, local_files_only=True).eval()
        model.config.use_cache = True
        report['reload'] = verify_saved_adapter(model, output / 'adapter')
        probe_index = manifest['parent_sample_count']
        prompt = read_jsonl(args.dataset_dir / 'train.sft.jsonl')[probe_index]['prompt']
        inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda:0')
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(outputs[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
        require(bool(answer), 'Reloaded model produced an empty answer.')
        report['reload_generation'] = dict(sample_id=raw[probe_index]['id'], answer=answer,
            note='Functionality check on a training sample, not independent quality evaluation.')
        report.update(status='REVISION_EPOCH_TRAIN_OK', total_seconds=round(time.perf_counter() - started, 2))
        (output / 'epoch_report.json').write_bytes(json_bytes(report))
        print(json.dumps({k: v for k, v in report.items() if k != 'training'}, ensure_ascii=False, indent=2), flush=True)
        print(json.dumps(dict(coverage='EPOCH_COVERAGE_OK', unique_trained_samples=report['training']['coverage']['unique_trained_samples'],
            optimizer_steps=report['training']['optimizer_steps'], parent_samples=50, supplement_samples=32,
            actual_epochs=report['training']['actual_epochs'], final_training_loss=report['training']['final_training_loss']), ensure_ascii=False), flush=True)
        print('REVISION_EPOCH_TRAIN_OK', flush=True)
    except Exception as exc:
        report.update(status='REVISION_EPOCH_TRAIN_FAILED', error=str(exc), traceback=traceback.format_exc())
        (output / 'epoch_report.json').write_bytes(json_bytes(report))
        raise


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
