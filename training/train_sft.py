"""Run 1-20 Qwen3 QLoRA optimizer steps and verify adapter save/reload.

Default: 10 optimizer steps on the existing 50-row pilot. No evaluation score
or quality improvement is claimed. --validate-only needs only the standard library.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_rows(directory, max_length):
    rows = read_jsonl(directory / "train.tokenized.jsonl")
    stats = json.loads((directory / "tokenizer_report.json").read_text(encoding="utf-8"))
    require(len(rows) == len(stats["sample_lengths"]) == 50, "Expected the checked 50-row pilot.")
    for i, (row, size) in enumerate(zip(rows, stats["sample_lengths"]), 1):
        ids, labels, mask = (row[k] for k in ("input_ids", "labels", "attention_mask"))
        boundary = size["prompt_tokens"]
        require(all(type(x) is int and x >= 0 for x in ids), f"Invalid token IDs: row {i}")
        require(all(type(x) is int for x in labels), f"Invalid label types: row {i}")
        require(len(ids) == len(labels) == len(mask) == size["total_tokens"], f"Lengths disagree: row {i}")
        require(0 < boundary < len(ids) <= max_length, f"Invalid boundary/length: row {i}; no truncation is allowed.")
        require(mask == [1] * len(ids), f"Unexpected padding in exported row {i}")
        require(labels[:boundary] == [-100] * boundary, f"Prompt participates in loss: row {i}")
        require(labels[boundary:] == ids[boundary:], f"Answer labels changed: row {i}")
        require(ids[-1] == stats["eos_token_id"], f"Missing EOS: row {i}")
    return rows, dict(samples=len(rows), max_tokens=max(len(r["input_ids"]) for r in rows),
                      supervised_tokens=sum(sum(x != -100 for x in r["labels"]) for r in rows),
                      tokenized_sha256=sha_file(directory / "train.tokenized.jsonl"))


def verify_collator(trainer, rows):
    """Check the actual TRL collator, including variable-length batch padding."""
    require(len(trainer.train_dataset) == len(rows), "Trainer changed the sample count.")
    for start in range(0, len(rows), 2):
        features = [trainer.train_dataset[i] for i in range(start, min(start + 2, len(rows)))]
        batch = trainer.data_collator(features)
        for offset, feature in enumerate(features):
            original = rows[start + offset]
            length = len(original["input_ids"])
            require(feature["labels"] == original["labels"], "Trainer preprocessing changed labels.")
            require(batch["input_ids"][offset, :length].tolist() == original["input_ids"], "Collator changed input IDs.")
            require(batch["labels"][offset, :length].tolist() == original["labels"], "Collator changed loss labels.")
            require(batch["attention_mask"][offset, :length].tolist() == [1] * length, "Invalid attention mask.")
            require(all(x == -100 for x in batch["labels"][offset, length:].tolist()), "Padding contributes to loss.")
            require(all(x == 0 for x in batch["attention_mask"][offset, length:].tolist()), "Padding has attention.")
    return dict(checked_samples=len(rows), original_labels_preserved=True, padding_labels_masked=True)


def make_gradient_guard(torch, TrainerCallback):
    class GradientGuard(TrainerCallback):
        def __init__(self):
            self.checked_steps = 0
            self.losses = []

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            has_nonzero = False
            found_grad = False
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    require(parameter.grad is None, f"Frozen base parameter has gradients: {name}")
                    continue
                require("lora_" in name, f"Unexpected trainable base parameter: {name}")
                if parameter.grad is not None:
                    found_grad = True
                    require(torch.isfinite(parameter.grad).all().item(), f"Non-finite gradients: {name}")
                    has_nonzero |= bool(torch.count_nonzero(parameter.grad).item())
            require(found_grad and has_nonzero, "No nonzero LoRA gradients reached the optimizer.")
            self.checked_steps += 1

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss = float(logs["loss"])
                require(math.isfinite(loss), "Training loss is not finite.")
                self.losses.append(dict(step=state.global_step, loss=loss))

    return GradientGuard()


def run_steps(model, tokenizer, rows, output, *, max_steps=10, batch_size=1,
              gradient_accumulation=4, learning_rate=1e-4, max_length=2048,
              bf16=True, use_cpu=False):
    """Shared training core; also exercised locally with a tiny random Qwen3."""
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    settings = SFTConfig(
        output_dir=str(output), max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate, lr_scheduler_type="constant", warmup_steps=0,
        optim="adamw_torch", weight_decay=0.0, max_grad_norm=1.0,
        bf16=bf16, fp16=False, use_cpu=use_cpu,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=max_length, packing=False, padding_free=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        # Existing labels already encode answer-only loss; do not regenerate masks.
        completion_only_loss=False, assistant_only_loss=False,
        remove_unused_columns=False, dataloader_num_workers=0,
        logging_strategy="steps", logging_steps=1, logging_nan_inf_filter=False,
        save_strategy="no", eval_strategy="no", report_to="none", push_to_hub=False,
        seed=42, data_seed=42,
    )
    guard = make_gradient_guard(torch, TrainerCallback)
    trainer = SFTTrainer(model=model, args=settings, train_dataset=Dataset.from_list(rows),
                         processing_class=tokenizer, callbacks=[guard])
    collator = verify_collator(trainer, rows)
    trainable = [(n, p) for n, p in trainer.model.named_parameters() if p.requires_grad]
    require(trainable and all("lora_" in n for n, _ in trainable), "Only LoRA parameters should be trainable.")
    probe_name, probe = next((n, p) for n, p in trainable if "lora_B" in n)
    before = probe.detach().float().cpu().clone()
    write_json(output / "run_config.json", dict(
        max_steps=max_steps, batch_size=batch_size, gradient_accumulation=gradient_accumulation,
        effective_batch_size=batch_size * gradient_accumulation,
        learning_rate=learning_rate, max_length=max_length, packing=False,
        gradient_checkpointing=True, trainable_parameters=sum(p.numel() for _, p in trainable),
        label_policy="use_verified_precomputed_labels_only", collator_checks=collator,
        evaluation="none; training pipeline smoke check only",
    ))
    result = trainer.train()
    require(trainer.state.global_step == max_steps, "Training did not reach the requested optimizer steps.")
    require(guard.checked_steps == max_steps and guard.losses, "Gradient/loss checks did not run for training.")
    require(math.isfinite(result.training_loss), "Non-finite final training loss.")
    delta = (probe.detach().float().cpu() - before).abs().max().item()
    require(math.isfinite(delta) and delta > 0, "LoRA probe weights did not update.")
    adapter = output / "adapter"
    trainer.model.save_pretrained(str(adapter), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter))
    trainer.save_state()
    write_json(output / "train_metrics.json", dict(result.metrics))
    require((adapter / "adapter_config.json").is_file(), "Adapter config was not saved.")
    require((adapter / "adapter_model.safetensors").is_file(), "Adapter weights were not saved.")
    report = dict(optimizer_steps=trainer.state.global_step, gradient_checked_steps=guard.checked_steps,
                  loss_history=guard.losses, final_training_loss=result.training_loss,
                  probe_parameter=probe_name, probe_max_absolute_update=delta,
                  trainable_parameters=sum(p.numel() for _, p in trainable), collator_checks=collator)
    return report


def load_base(model_dir):
    import torch
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    )
    require(model.config.model_type == "qwen3", "Expected Qwen3.")
    require(getattr(model, "is_loaded_in_4bit", False), "Expected 4-bit loading.")
    require(any(isinstance(m, bnb.nn.Linear4bit) for m in model.modules()), "No NF4 layers loaded.")
    model.config.use_cache = False
    return model


def verify_saved_adapter(model, adapter):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    saved = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
    loaded = get_peft_model_state_dict(model)
    require(saved and set(saved) == set(loaded), "Reloaded adapter has different parameter names.")
    for name, tensor in saved.items():
        require(torch.isfinite(tensor).all().item(), f"Saved weights are not finite: {name}")
        require(torch.equal(tensor, loaded[name].detach().cpu()), f"Reloaded weights differ: {name}")
    return dict(parameter_tensors=len(saved), saved_and_reloaded_weights_equal=True,
                adapter_sha256=sha_file(adapter / "adapter_model.safetensors"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/pilot_v1")
    parser.add_argument("--model-dir", type=Path, default=Path("/root/autodl-tmp/models/Qwen3-8B"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validate-only", action="store_true", help="Check the existing pilot; no ML imports or GPU use.")
    args = parser.parse_args()
    require(1 <= args.max_steps <= 20, "This smoke script allows 1-20 optimizer steps only.")
    require(0 < args.learning_rate <= 1e-3 and math.isfinite(args.learning_rate), "Invalid learning rate.")
    require(args.batch_size > 0 and args.gradient_accumulation > 0 and args.max_length > 0, "Invalid batch/length settings.")
    return args


def main():
    args = parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    # Also verifies frozen source excerpts and citation metadata using the shipped checker.
    subprocess.run([sys.executable, str(ROOT / "data_pipeline/validate_pilot.py"),
                    "--dataset-dir", str(args.dataset_dir), "--require-tokenized"], check=True)
    rows, data_report = validate_rows(args.dataset_dir, args.max_length)
    print("TRAINING_DATA_PREFLIGHT_OK", json.dumps(data_report), flush=True)
    if args.validate_only:
        return

    require(os.environ.get("WORLD_SIZE", "1") == "1", "This pilot script is for a single GPU process.")
    require(sys.prefix != sys.base_prefix, "Activate the training venv first.")
    args.model_dir = args.model_dir.resolve()
    require(args.model_dir.is_dir(), "Downloaded local model directory is missing.")
    subprocess.run([sys.executable, str(ROOT / "data_pipeline/tokenizer_export.py"),
                    "--model-dir", str(args.model_dir), "--dataset-dir", str(args.dataset_dir),
                    "--check-only"], check=True)

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, set_seed

    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA/BF16 must be available.")
    require(importlib.metadata.version("trl") == "1.16.0", "Expected the selected TRL 1.16.0.")
    require(importlib.metadata.version("transformers") == "5.20.2", "Expected Transformers 5.20.2.")
    require(importlib.metadata.version("peft") == "0.21.2", "Expected PEFT 0.21.2.")
    output = args.output_dir or (ROOT / "outputs" / ("smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    output = output.resolve()
    require(not output.exists() or not any(output.iterdir()), "Output directory is not empty; select a new path.")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = dict(status="RUNNING", time_utc=datetime.now(timezone.utc).isoformat(),
                  model_dir=str(args.model_dir), dataset=data_report, output_dir=str(output),
                  versions={p: importlib.metadata.version(p) for p in
                            ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "bitsandbytes")},
                  gpu=torch.cuda.get_device_name(0), quality_evaluation_performed=False)
    write_json(output / "smoke_report.json", report)
    try:
        set_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        torch.cuda.reset_peak_memory_stats()
        model = load_base(args.model_dir)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
            bias="none", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
        report["training"] = run_steps(
            model, tokenizer, rows, output, max_steps=args.max_steps, batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation, learning_rate=args.learning_rate,
            max_length=args.max_length)
        torch.cuda.synchronize()
        report["training_peak_allocated_vram_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        report["training_peak_reserved_vram_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        report["training_and_first_load_seconds"] = round(time.perf_counter() - started, 2)
        # Release the first model before loading a fresh base and saved adapter.
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = PeftModel.from_pretrained(load_base(args.model_dir), str(output / "adapter"),
                                          is_trainable=False, local_files_only=True,
                                          autocast_adapter_dtype=False).eval()
        model.config.use_cache = True
        report["reload"] = verify_saved_adapter(model, output / "adapter")
        prompt = read_jsonl(args.dataset_dir / "train.sft.jsonl")[4]["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(generated[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        require(bool(answer), "Reloaded adapter produced an empty response.")
        report["reload_generation"] = dict(sample_id="travel-sft-pilot-005", answer=answer,
                                            note="Training example; functionality check, not an independent evaluation.")
        report.update(status="TRAIN_SMOKE_OK", total_seconds=round(time.perf_counter() - started, 2))
        write_json(output / "smoke_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print("TRAIN_SMOKE_OK", flush=True)
    except Exception as exc:
        report.update(status="TRAIN_SMOKE_FAILED", error=str(exc), traceback=traceback.format_exc())
        write_json(output / "smoke_report.json", report)
        raise


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
