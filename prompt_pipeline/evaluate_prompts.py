"""Compare original and revised prompts using the same frozen epoch1 adapter."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evaluation'))
from evaluate import DIMENSIONS, format_flags, load_model
from evaluation.common import json_bytes, jsonl_bytes, require, sha
from prompt_pipeline.protocol import VARIANTS, preflight
from prompt_pipeline.review import write_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path('/root/autodl-tmp/models/Qwen3-8B'))
    parser.add_argument('--adapter-dir', type=Path, default=ROOT / 'outputs/revision_v2_epoch1/adapter')
    parser.add_argument('--baseline-dir', type=Path, default=ROOT / 'eval_outputs/revision_v2_epoch1_512')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'eval_outputs/prompt_v2_validation')
    parser.add_argument('--limit', type=int, help='Local/debug subset, explicitly marked incomplete; not eligible for final selection.')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    require(args.limit is None or 1 <= args.limit <= 24, 'Debug limit must be 1-24.')
    spec, templates, examples, baseline_report, previous, tokenizer, prepared, checks = preflight(
        ROOT, args.model_dir, args.adapter_dir, args.baseline_dir)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    print('PROMPT_PREFLIGHT_OK', flush=True)
    if args.validate_only:
        return
    runtime = {p: importlib.metadata.version(p) for p in baseline_report['versions']}
    require(runtime == baseline_report['versions'], 'Generation packages differ from the frozen epoch1 baseline.')
    require(sys.prefix != sys.base_prefix, 'Activate the existing training/inference venv first.')
    import torch
    from transformers import GenerationConfig, set_seed
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA/BF16 required.')
    selected_count = args.limit or len(examples)
    selected = examples[:selected_count]
    output = args.output_dir.resolve()
    require(not output.exists() or not any(output.iterdir()), 'Output directory must be new or empty.')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='RUNNING', experiment=spec['experiment'], created_at_utc=datetime.now(timezone.utc).isoformat(),
        split='validation', samples=selected_count, complete_split=selected_count == 24,
        model_dir=str(args.model_dir.resolve()), adapter_dir=str(args.adapter_dir.resolve()),
        baseline_dir=str(args.baseline_dir.resolve()), adapter_sha256=spec['adapter_sha256'],
        adapter_config_sha256=spec['adapter_config_sha256'], model_config_sha256=spec['base_model_config_sha256'],
        eval_manifest_sha256=spec['eval_manifest_sha256'], eval_file_sha256=spec['validation_file_sha256'],
        chat_template_sha256=spec['chat_template_sha256'], generation=spec['generation'], versions=runtime,
        template_sha256={v: sha(t.encode('utf-8')) for v, t in templates.items()},
        experiment_manifest_sha256=sha((ROOT / 'prompt_pipeline/experiment_manifest.json').read_bytes()),
        code_sha256={name: sha((ROOT / name).read_bytes()) for name in (
            'prompt_pipeline/evaluate_prompts.py', 'prompt_pipeline/protocol.py', 'prompt_pipeline/review.py',
            'evaluation/evaluate.py', 'evaluation/common.py')},
        controls=dict(same_adapter_for_both_variants=True, same_question_evidence_history=True,
            unchanged_payloads=selected_count, generation_settings_unchanged=True, original_control_reproduced=None),
        tokenizer_eos_token_id=tokenizer.eos_token_id, training_performed=False, test_predictions_generated=False,
        manual_review_status='pending', semantic_correctness_auto_scored=False, variants={})
    (output / 'report.json').write_bytes(json_bytes(report))
    inputs = [dict(variant=v, **p) for v in VARIANTS for p in prepared[v][:selected_count]]
    (output / 'inputs.jsonl').write_bytes(jsonl_bytes(inputs))
    predictions = []
    try:
        set_seed(42)
        model = load_model(args.model_dir, args.adapter_dir)
        require(not model.training, 'Model must be in evaluation mode.')
        require(not any(p.requires_grad for p in model.parameters()), 'Frozen adapter inference required.')
        config = GenerationConfig(max_new_tokens=spec['generation']['max_new_tokens'], do_sample=False, num_beams=1,
            use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        warmup_config = GenerationConfig.from_dict(dict(config.to_dict(), max_new_tokens=1))
        for variant in VARIANTS:
            set_seed(42)
            torch.cuda.reset_peak_memory_stats()
            warmup = torch.tensor([prepared[variant][0]['ids']], dtype=torch.long, device='cuda:0')
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                warmup_result = model.generate(input_ids=warmup, attention_mask=torch.ones_like(warmup), generation_config=warmup_config)
            torch.cuda.synchronize()
            warmup_seconds = time.perf_counter() - started
            del warmup, warmup_result
            current = []
            with (output / f'{variant}.predictions.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
                for index, (row, item) in enumerate(zip(selected, prepared[variant]), 1):
                    ids = torch.tensor([item['ids']], dtype=torch.long, device='cuda:0')
                    mask = torch.ones_like(ids)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(input_ids=ids, attention_mask=mask, generation_config=config)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                    new_ids = generated[0, ids.shape[-1]:].tolist()
                    p = dict(id=row['id'], variant=variant, task_type=row['task_type'], adapter_sha256=spec['adapter_sha256'],
                        input_sha256=item['input_sha256'], payload_sha256=item['payload_sha256'], prompt_tokens=item['prompt_tokens'],
                        generated_tokens=len(new_ids), output_token_ids=new_ids, generation_seconds=round(seconds, 4),
                        answer=tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
                        hit_output_limit=len(new_ids) >= spec['generation']['max_new_tokens'] and
                            (not new_ids or new_ids[-1] != tokenizer.eos_token_id))
                    p['format_flags'] = format_flags(row, p)
                    predictions.append(p)
                    current.append(p)
                    stream.write(json.dumps(p, ensure_ascii=False) + '\n')
                    stream.flush()
                    print(f"{variant}: {index}/{selected_count} {row['id']} ({seconds:.2f}s)", flush=True)
            report['variants'][variant] = dict(samples=len(current),
                automatic_format_flag_sample_counts={k: sum(bool(p['format_flags'][k]) for p in current) for k in current[0]['format_flags']},
                mean_output_tokens=sum(p['generated_tokens'] for p in current) / len(current),
                mean_generation_seconds=sum(p['generation_seconds'] for p in current) / len(current),
                peak_allocated_vram_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 3), warmup_seconds=round(warmup_seconds, 4))
            if variant == VARIANTS[0]:
                mismatch = [p['id'] for p, prior in zip(current, previous) if p['answer'] != prior['answer']]
                report['controls'].update(original_control_reproduced=not mismatch, mismatch_ids=mismatch,
                    original_control_identical_answers=len(current) - len(mismatch))
                (output / 'report.json').write_bytes(json_bytes(report))
                require(not mismatch, 'PROMPT_CONTROL_MISMATCH: original prompt did not reproduce the frozen baseline; inspect before comparing prompts.')
                print('PROMPT_CONTROL_REPRODUCED', flush=True)
        require(sha((args.adapter_dir / 'adapter_model.safetensors').read_bytes()) == spec['adapter_sha256'], 'Adapter bytes changed during comparison.')
        write_review(output, selected, predictions, DIMENSIONS)
        original = [p for p in predictions if p['variant'] == VARIANTS[0]]
        revised = [p for p in predictions if p['variant'] == VARIANTS[1]]
        report.update(status='PROMPT_COMPARISON_OK', output_dir=str(output),
            changed_answers=sum(a['answer'] != b['answer'] for a, b in zip(original, revised)),
            interpretation='Paired generation and control checks completed; no quality improvement or final prompt selection is automatically inferred.')
        (output / 'report.json').write_bytes(json_bytes(report))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print('PROMPT_COMPARISON_OK', flush=True)
    except Exception as exc:
        report.update(status='PROMPT_CONTROL_MISMATCH' if 'PROMPT_CONTROL_MISMATCH' in str(exc) else 'PROMPT_COMPARISON_FAILED',
            error=str(exc), traceback=traceback.format_exc())
        (output / 'report.json').write_bytes(json_bytes(report))
        raise


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
