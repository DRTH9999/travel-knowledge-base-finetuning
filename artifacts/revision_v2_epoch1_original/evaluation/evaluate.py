"""Compare the local base Qwen3 and one saved LoRA on identical held-out inputs.

Default split: validation. Reference answers are for review and never enter the
model input. Automatic results cover format only, not semantic correctness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import re
import time
import traceback

from common import ROOT, json_bytes, read_jsonl, require, sha, validate_frozen


DIMENSIONS = {
    "faithfulness": "事实忠实：0=有关键无依据或错误事实；1=轻微问题；2=有依据且无明显事实错误",
    "completeness": "完整性：0=遗漏核心需求；1=部分覆盖；2=覆盖主要问题与条件",
    "citation_support": "引用支持：0=关键引用错误或缺失；1=部分支持；2=关键结论均有对应依据",
    "answerability": "回答边界：0=编造缺失信息或明显误拒答；1=边界表达不完整；2=可答与缺失部分处理恰当",
    "constraints_format": "约束与表达：0=违背关键约束或难以使用；1=轻微问题；2=符合用户约束且清晰",
}


def format_flags(row, prediction):
    text = prediction["answer"]
    cited = {int(x) for x in re.findall(r"\[(\d+)\]", text)}
    valid = set(range(1, len(row["contexts"]) + 1))
    expected = {r for c in row["expected_claims"] for r in c["citations"]}
    distractors = {c["reference_index"] for c in row["contexts"] if c["review_role"] == "distractor"}
    return dict(empty_answer=not bool(text.strip()),
                contains_think_tags="<think>" in text or "</think>" in text,
                invalid_reference_ids=sorted(cited - valid),
                irrelevant_reference_ids=sorted(cited & distractors),
                missing_gold_reference_ids=sorted(expected - cited),
                hit_output_limit=prediction["hit_output_limit"])


def write_review(output, rows, predictions):
    indexed = {(p["model"], p["id"]): p for p in predictions}
    manual = []
    lines = ["# 固定证据问答对照审核", "", "自动检查只涉及格式，不能判断语义正确性。请对照证据逐题审核。", "",
             "每个维度 0–2 分；未审核时保持 null，不能把缺失分数当成零分或已通过。", ""]
    for key, definition in DIMENSIONS.items():
        lines.append(f"- {key}：{definition}")
    for row in rows:
        lines += ["", f"## {row['id']} · {row['task_type']}", "", f"问题：{row['question']}", ""]
        if row["history"]:
            lines += ["历史：" + json.dumps(row["history"], ensure_ascii=False), ""]
        lines += ["参考答案：", "", row["reference_answer"], ""]
        for label in ("base", "adapter"):
            p = indexed[(label, row["id"])]
            lines += [f"### {label}", "", p["answer"] or "（空回答）", "",
                      "自动格式提示：" + json.dumps(p["format_flags"], ensure_ascii=False), ""]
            manual.append(dict(id=row["id"], model=label, scores={key: None for key in DIMENSIONS},
                               reviewer="", notes="", status="pending"))
        lines += ["### 原文证据", ""]
        for c in row["contexts"]:
            lines += [f"[{c['reference_index']}] {c['relative_path']} L{c['line_start']}-{c['line_end']}", "",
                      "```text", c["content"], "```", ""]
        if not row["contexts"]:
            lines += ["没有检索片段。", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "manual_reviews.json").write_bytes(json_bytes(dict(rubric=DIMENSIONS, reviews=manual)))


def load_model(model_dir, adapter_dir=None):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    require(model.config.model_type == "qwen3" and getattr(model, "is_loaded_in_4bit", False), "Expected NF4 Qwen3.")
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False,
                                         autocast_adapter_dtype=False, local_files_only=True)
    model.config.use_cache = True
    return model.eval()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/eval_v1")
    parser.add_argument("--model-dir", type=Path, default=Path("/root/autodl-tmp/models/Qwen3-8B"))
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--final-test", action="store_true", help="Explicitly select final-only frozen test evaluation.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, help="Debug subset; results will be marked incomplete.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true", help="Validate sources and real tokenizer rendering; no model loading.")
    args = parser.parse_args()
    require(args.split != "test" or args.final_test, "Frozen test requires explicit --final-test; use validation for tuning.")
    require(1 <= args.max_new_tokens <= 1024, "max-new-tokens must be 1-1024.")
    require(args.limit is None or args.limit > 0, "limit must be positive.")
    return args


def main():
    args = parse_args()
    manifest, data_checks = validate_frozen(args.dataset_dir)
    rows = read_jsonl(args.dataset_dir / f"{args.split}.jsonl")
    total_available = len(rows)
    if args.limit:
        rows = rows[:args.limit]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepared = []
    for row in rows:
        # Only the business prompt is passed to the model; no reference answers or review metadata.
        text = tokenizer.apply_chat_template([dict(role="user", content=row["prompt"])], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        require(len(ids) + args.max_new_tokens <= args.max_total_tokens,
                f"Input/output budget exceeded for {row['id']}; no silent truncation is allowed.")
        prepared.append(dict(ids=ids, input_sha256=sha(json_bytes(ids))))
    preflight = dict(status="EVAL_PREFLIGHT_OK", split=args.split, samples=len(rows),
                     complete_split=len(rows) == total_available, max_prompt_tokens=max(len(p["ids"]) for p in prepared),
                     max_new_tokens=args.max_new_tokens, model_weights_loaded=False)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.validate_only:
        return
    require(args.adapter_dir is not None and (args.adapter_dir / "adapter_model.safetensors").is_file(), "Specify a saved --adapter-dir.")
    import torch
    from transformers import GenerationConfig, set_seed
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA/BF16 must be available.")
    output = args.output_dir or ROOT / "eval_outputs" / (args.split + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output = output.resolve()
    require(not output.exists() or not any(output.iterdir()), "Output directory must be new or empty.")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status="RUNNING", created_at_utc=datetime.now(timezone.utc).isoformat(),
                  split=args.split, samples=len(rows), complete_split=len(rows) == total_available,
                  model_dir=str(args.model_dir.resolve()), adapter_dir=str(args.adapter_dir.resolve()),
                  model_config_sha256=sha((args.model_dir / "config.json").read_bytes()),
                  adapter_sha256=sha((args.adapter_dir / "adapter_model.safetensors").read_bytes()),
                  eval_manifest_sha256=sha((args.dataset_dir / "manifest.json").read_bytes()),
                  eval_file_sha256=manifest["files_sha256"][f"{args.split}.jsonl"],
                  chat_template_sha256=sha(tokenizer.chat_template.encode("utf-8")),
                  versions={p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "bitsandbytes")},
                  generation=dict(do_sample=False, num_beams=1, enable_thinking=False,
                                  max_new_tokens=args.max_new_tokens, warmup_new_tokens=1,
                                  quantization="NF4_double_quant_BF16_SDPA"),
                  manual_review_status="pending", semantic_correctness_auto_scored=False,
                  data_checks=data_checks, models={})
    (output / "report.json").write_bytes(json_bytes(report))
    predictions = []
    try:
        config = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                                  use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        warmup_config = GenerationConfig.from_dict(dict(config.to_dict(), max_new_tokens=1))
        for label, adapter in (("base", None), ("adapter", args.adapter_dir)):
            set_seed(42)
            torch.cuda.reset_peak_memory_stats()
            model = load_model(args.model_dir, adapter)
            # Exclude one-time CUDA/kernel initialization from per-question latency.
            warmup_inputs = torch.tensor([prepared[0]["ids"]], dtype=torch.long, device="cuda:0")
            torch.cuda.synchronize()
            warmup_started = time.perf_counter()
            with torch.inference_mode():
                warmup_tokens = model.generate(input_ids=warmup_inputs, attention_mask=torch.ones_like(warmup_inputs),
                                                generation_config=warmup_config)
            torch.cuda.synchronize()
            warmup_seconds = time.perf_counter() - warmup_started
            del warmup_inputs, warmup_tokens
            with (output / f"{label}.predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                for index, (row, item) in enumerate(zip(rows, prepared), 1):
                    inputs = torch.tensor([item["ids"]], dtype=torch.long, device="cuda:0")
                    mask = torch.ones_like(inputs)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        tokens = model.generate(input_ids=inputs, attention_mask=mask, generation_config=config)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                    new_ids = tokens[0, inputs.shape[-1]:].tolist()
                    prediction = dict(id=row["id"], model=label, task_type=row["task_type"],
                                      input_sha256=item["input_sha256"], prompt_tokens=len(item["ids"]),
                                      generated_tokens=len(new_ids), generation_seconds=round(seconds, 4),
                                      answer=tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
                                      hit_output_limit=len(new_ids) >= args.max_new_tokens and
                                                       (not new_ids or new_ids[-1] != tokenizer.eos_token_id))
                    prediction["format_flags"] = format_flags(row, prediction)
                    predictions.append(prediction)
                    stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(f"{label}: {index}/{len(rows)} {row['id']} ({seconds:.2f}s)", flush=True)
            subset = [p for p in predictions if p["model"] == label]
            flag_counts = {key: sum(bool(p["format_flags"][key]) for p in subset) for key in subset[0]["format_flags"]}
            report["models"][label] = dict(samples=len(subset), automatic_format_flag_sample_counts=flag_counts,
                                           warmup_seconds=round(warmup_seconds, 4),
                                           mean_generation_seconds=sum(p["generation_seconds"] for p in subset) / len(subset),
                                           peak_allocated_vram_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 3))
            del model, inputs, mask, tokens
            gc.collect()
            torch.cuda.empty_cache()
        write_review(output, rows, predictions)
        report.update(status="EVAL_GENERATION_OK", output_dir=str(output),
                      interpretation="Generation completed; format flags are review hints, not accuracy or quality scores.")
        (output / "report.json").write_bytes(json_bytes(report))
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print("EVAL_GENERATION_OK", flush=True)
    except Exception as exc:
        report.update(status="EVAL_FAILED", error=str(exc), traceback=traceback.format_exc())
        (output / "report.json").write_bytes(json_bytes(report))
        raise


if __name__ == "__main__":
    main()
