"""Small, versioned revision-data experiment from a fresh base model.

Defaults match the original 10-step QLoRA run. Uses the tested training core,
with dataset size and IDs read from revision metadata instead of hardcoded 50.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision_pipeline.common import json_bytes, read_jsonl, require, sha, validate_dataset, validate_tokens
from revision_pipeline.legacy_training_core import load_base, run_steps, verify_saved_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/revision_v2")
    parser.add_argument("--model-dir", type=Path, default=Path("/root/autodl-tmp/models/Qwen3-8B"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    require(1 <= args.max_steps <= 20, "Revision short experiment allows 1-20 optimizer steps.")
    require(args.batch_size > 0 and args.gradient_accumulation > 0, "Invalid batch settings.")
    require(0 < args.learning_rate <= 1e-3 and math.isfinite(args.learning_rate), "Invalid learning rate.")
    manifest, raw, checks = validate_dataset(args.dataset_dir, require_tokenized=True)
    tokens, token_checks = validate_tokens(args.dataset_dir, raw, max_length=args.max_length)
    print("REVISION_TRAIN_PREFLIGHT_OK", json.dumps(token_checks), flush=True)
    if args.validate_only:
        return
    require(os.environ.get("WORLD_SIZE", "1") == "1", "Use a single GPU process for this experiment.")
    require(sys.prefix != sys.base_prefix, "Activate the training venv first.")
    eval_manifest = ROOT / "data/eval_v1/manifest.json"
    require(eval_manifest.is_file() and sha(eval_manifest.read_bytes()) == manifest["reserved_eval_manifest_sha256"],
            "Frozen evaluation manifest changed or is missing; preserve the existing evaluation package.")
    from revision_pipeline.tokenize_revision import export
    tokenizer_check = export(args.dataset_dir, args.model_dir, check_only=True)
    print("REVISION_TOKENIZER_CHECK_OK", flush=True)
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, set_seed
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA/BF16 required.")
    for package, version in (("trl", "1.16.0"), ("transformers", "5.20.2"), ("peft", "0.21.2")):
        require(importlib.metadata.version(package) == version, f"Expected {package} {version}.")
    output = (args.output_dir or ROOT / "outputs" / ("revision_v2_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    require(not output.exists() or not any(output.iterdir()), "Output directory must be new or empty.")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = dict(status="RUNNING", created_at_utc=datetime.now(timezone.utc).isoformat(),
                  dataset_version=manifest["dataset_version"], dataset=checks, token_checks=token_checks,
                  tokenizer_recheck=tokenizer_check["status"],
                  initialization="fresh_base_and_fresh_lora_no_adapter_resume",
                  model_dir=str(args.model_dir.resolve()), output_dir=str(output),
                  versions={p: importlib.metadata.version(p) for p in ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "bitsandbytes")},
                  model_config_sha256=sha((args.model_dir / "config.json").read_bytes()),
                  dataset_manifest_sha256=sha((args.dataset_dir / "source_manifest.json").read_bytes()),
                  eval_manifest_sha256=sha(eval_manifest.read_bytes()), quality_evaluation_performed=False,
                  coverage_note="All rows are collator-checked; short max_steps training does not imply every training row was used.")
    (output / "revision_report.json").write_bytes(json_bytes(report))
    try:
        set_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        torch.cuda.reset_peak_memory_stats()
        model = prepare_model_for_kbit_training(load_base(args.model_dir), use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
        report["training"] = run_steps(model, tokenizer, tokens, output, max_steps=args.max_steps,
                                       batch_size=args.batch_size, gradient_accumulation=args.gradient_accumulation,
                                       learning_rate=args.learning_rate, max_length=args.max_length)
        torch.cuda.synchronize()
        report["training_peak_allocated_vram_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        report["training_peak_reserved_vram_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        report["training_and_load_seconds"] = round(time.perf_counter() - started, 2)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = PeftModel.from_pretrained(load_base(args.model_dir), str(output / "adapter"), is_trainable=False,
                                          autocast_adapter_dtype=False, local_files_only=True).eval()
        model.config.use_cache = True
        report["reload"] = verify_saved_adapter(model, output / "adapter")
        probe_index = manifest["parent_sample_count"]
        prompt = read_jsonl(args.dataset_dir / "train.sft.jsonl")[probe_index]["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(outputs[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        require(bool(answer), "Reloaded model produced an empty answer.")
        report["reload_generation"] = dict(sample_id=raw[probe_index]["id"], answer=answer,
                                            note="Training sample functionality check, not independent quality evaluation.")
        report.update(status="REVISION_TRAIN_OK", total_seconds=round(time.perf_counter() - started, 2))
        (output / "revision_report.json").write_bytes(json_bytes(report))
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print("REVISION_TRAIN_OK", flush=True)
    except Exception as exc:
        report.update(status="REVISION_TRAIN_FAILED", error=str(exc), traceback=traceback.format_exc())
        (output / "revision_report.json").write_bytes(json_bytes(report))
        raise


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
