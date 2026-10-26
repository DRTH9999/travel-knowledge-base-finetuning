"""Render a versioned dataset with Qwen3's real non-thinking tokenizer."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision_pipeline.common import ROOT, json_bytes, jsonl_bytes, read_jsonl, require, sha, validate_dataset


def export(directory, model_dir, *, check_only=False):
    from transformers import AutoTokenizer
    directory, model_dir = Path(directory), Path(model_dir)
    _, raw, _ = validate_dataset(directory)
    messages = read_jsonl(directory / "train.messages.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    require(tokenizer.eos_token_id is not None, "Tokenizer EOS is missing.")
    strings, tokenized, lengths = [], [], []
    for source, row in zip(raw, messages):
        target = source["target_answer"]
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=False,
                                             add_generation_prompt=False, enable_thinking=False).rstrip("\r\n")
        require(full.startswith(prompt), f"Nonthinking prefix mismatch: {source['id']}")
        completion = full[len(prompt):]
        require(completion == target + tokenizer.eos_token, f"Unexpected completion framing: {source['id']}")
        prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        ids = tokenizer.encode(full, add_special_tokens=False)
        require(ids == prefix_ids + completion_ids, "Tokenizer boundary merge changed label boundary.")
        labels = [-100] * len(prefix_ids) + completion_ids
        require(tokenizer.decode(completion_ids, skip_special_tokens=False) == completion, "Answer decoding differs.")
        require(ids[-1] == tokenizer.eos_token_id and len(ids) <= 2048, "EOS/length check failed; no truncation allowed.")
        strings.append(dict(prompt=prompt, completion=completion))
        tokenized.append(dict(input_ids=ids, attention_mask=[1] * len(ids), labels=labels))
        lengths.append(dict(id=source["id"], prompt_tokens=len(prefix_ids), completion_tokens=len(completion_ids), total_tokens=len(ids)))
    if check_only:
        require(tokenized == read_jsonl(directory / "train.tokenized.jsonl"), "Current tokenizer IDs or masks differ from export.")
        require(strings == read_jsonl(directory / "train.sft.jsonl"), "Current tokenizer rendering differs from export.")
    report = dict(status="REVISION_TOKENIZER_CHECK_OK" if check_only else "REVISION_TOKENIZER_EXPORT_OK",
                  dataset_version="revision_v2", samples=len(raw), model_weights_loaded=False,
                  transformers_version=importlib.metadata.version("transformers"),
                  tokenizers_version=importlib.metadata.version("tokenizers"),
                  eos_token=tokenizer.eos_token, eos_token_id=tokenizer.eos_token_id,
                  chat_template_sha256=sha(tokenizer.chat_template.encode("utf-8")),
                  tokenizer_assets_sha256={n: sha((model_dir / n).read_bytes()) for n in ("tokenizer.json", "tokenizer_config.json", "config.json") if (model_dir / n).is_file()},
                  mode="enable_thinking=False", loss_region="verified_answer_and_terminal_eos_only",
                  all_prompt_labels_masked=True, truncated_samples=0,
                  max_total_tokens=max(x["total_tokens"] for x in lengths),
                  total_supervised_tokens=sum(x["completion_tokens"] for x in lengths),
                  suggested_max_length=2048, sample_lengths=lengths)
    if not check_only:
        (directory / "train.sft.jsonl").write_bytes(jsonl_bytes(strings))
        (directory / "train.tokenized.jsonl").write_bytes(jsonl_bytes(tokenized))
        (directory / "tokenizer_report.json").write_bytes(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/revision_v2")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    report = export(args.dataset_dir, args.model_dir, check_only=args.check_only)
    print(json.dumps({k: v for k, v in report.items() if k != "sample_lengths"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
