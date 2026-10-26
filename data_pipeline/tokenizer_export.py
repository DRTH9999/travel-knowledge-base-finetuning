"""Render Qwen3 non-thinking prompt/completion and verify loss-mask boundaries.

Does not load model weights, initialize CUDA, install packages, or start training.
Use --check-only on AutoDL with its downloaded model directory before training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=root / "data/pilot_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(args.dataset_dir / "train.messages.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True,
                                             trust_remote_code=False)
    assert tokenizer.eos_token and tokenizer.eos_token_id is not None
    records, tokens, sizes = [], [], []
    for n, row in enumerate(rows, 1):
        assert [x["role"] for x in row["prompt"]] == ["user"]
        assert [x["role"] for x in row["completion"]] == ["assistant"]
        target = row["completion"][0]["content"]
        assert "<think>" not in target and "</think>" not in target
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        full = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=False,
                                             add_generation_prompt=False, enable_thinking=False)
        assert full.startswith(prompt), f"Qwen3 nonthinking prefix mismatch at sample {n}"
        # Drop whitespace AFTER the terminal EOS only. Do not strip answer content.
        full = full.rstrip("\r\n")
        assert full.endswith(tokenizer.eos_token), f"Missing terminal EOS at {n}"
        completion = full[len(prompt):]
        assert completion == target + tokenizer.eos_token, f"Unexpected answer framing at {n}"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = tokenizer.encode(full, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        assert input_ids == prompt_ids + completion_ids, f"Tokenizer boundary mismatch at {n}"
        labels = [-100] * len(prompt_ids) + completion_ids
        assert all(x == -100 for x in labels[:len(prompt_ids)])
        assert labels[len(prompt_ids):] == input_ids[len(prompt_ids):]
        assert tokenizer.decode(labels[len(prompt_ids):], skip_special_tokens=False) == completion
        assert input_ids[-1] == tokenizer.eos_token_id
        assert len(input_ids) == len(labels) and completion_ids
        records.append(dict(prompt=prompt, completion=completion))
        tokens.append(dict(input_ids=input_ids, attention_mask=[1] * len(input_ids), labels=labels))
        sizes.append(dict(id=f"travel-sft-pilot-{n:03d}", prompt_tokens=len(prompt_ids),
                          completion_tokens=len(completion_ids), total_tokens=len(input_ids),
                          target_character_count=len(target)))
    if args.check_only:
        assert records == read_jsonl(args.dataset_dir / "train.sft.jsonl"), "Rendered strings differ from export"
        assert tokens == read_jsonl(args.dataset_dir / "train.tokenized.jsonl"), "Token IDs/masks differ from export"
    else:
        write_jsonl(args.dataset_dir / "train.sft.jsonl", records)
        write_jsonl(args.dataset_dir / "train.tokenized.jsonl", tokens)
    lengths = sorted(x["total_tokens"] for x in sizes)
    report = dict(status="TOKENIZER_CHECK_OK" if args.check_only else "TOKENIZER_EXPORT_OK",
                  samples=len(rows), local_files_only=True, model_weights_loaded=False,
                  transformers_version=importlib.metadata.version("transformers"),
                  tokenizers_version=importlib.metadata.version("tokenizers"),
                  eos_token=tokenizer.eos_token, eos_token_id=tokenizer.eos_token_id,
                  chat_template_sha256=sha(tokenizer.chat_template.encode("utf-8")),
                  tokenizer_assets_sha256={name: sha((args.model_dir / name).read_bytes())
                                           for name in ("tokenizer.json", "tokenizer_config.json", "config.json")
                                           if (args.model_dir / name).exists()},
                  mode="enable_thinking=False",
                  prefix_contains_empty_think_prefill="<think>\n\n</think>" in records[0]["prompt"],
                  supervised_region="direct_answer_and_one_terminal_eos_only",
                  truncated_samples=0, all_prefix_labels_masked=True,
                  total_supervised_tokens=sum(x["completion_tokens"] for x in sizes),
                  min_total_tokens=min(lengths), max_total_tokens=max(lengths),
                  p95_total_tokens=lengths[min(len(lengths)-1, int(len(lengths)*.95))],
                  max_prompt_tokens=max(x["prompt_tokens"] for x in sizes),
                  max_completion_tokens=max(x["completion_tokens"] for x in sizes),
                  suggested_pilot_max_length=2048 if max(lengths) <= 2048 else 4096 if max(lengths) <= 4096 else 8192,
                  sample_lengths=sizes)
    output_name = "tokenizer_check_cloud.json" if args.check_only else "tokenizer_report.json"
    (args.dataset_dir / output_name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "sample_lengths"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
