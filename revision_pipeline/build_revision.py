"""Preserve the original 50 samples and freeze 32 source-grounded supplements."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.common import excerpt
from revision_pipeline.common import ROOT, json_bytes, jsonl_bytes, read_jsonl, render, require, sha, validate_dataset
from revision_pipeline.curated_revision import SAMPLES


def build(pilot, output):
    parent = json.loads((pilot / "source_manifest.json").read_text(encoding="utf-8"))
    parent_rows, parent_messages = read_jsonl(pilot / "samples.raw.jsonl"), read_jsonl(pilot / "train.messages.jsonl")
    require(len(parent_rows) == len(parent_messages) == 50 and len(SAMPLES) == 32, "Unexpected authored batch size.")
    docs = {d["doc_id"]: d for d in parent["documents"] if d["split"] == "train_pilot"}
    files = {"prompt_template.txt": (pilot / "prompt_template.txt").read_bytes()}
    require(sha(files["prompt_template.txt"]) == parent["prompt_template_sha256"], "Parent business template changed.")
    for doc in docs.values():
        content = (pilot / doc["snapshot_path"]).read_bytes()
        require(sha(content) == doc["sha256"], "Parent source snapshot changed.")
        files[doc["snapshot_path"]] = content
    template = files["prompt_template.txt"].decode("utf-8")
    rows, messages = list(parent_rows), list(parent_messages)
    for number, sample in enumerate(SAMPLES, 1):
        contexts = []
        for ref, spec in enumerate(sample["references"], 1):
            alias, *headings = spec.lstrip("!").split("/")
            doc = docs[alias]
            start, end, text = excerpt(files[doc["snapshot_path"]].decode("utf-8-sig"), headings)
            checksum = sha(text.encode("utf-8"))
            contexts.append(dict(reference_index=ref, doc_id=alias, chunk_id=f"{alias}:L{start}-L{end}:{checksum[:12]}",
                                 relative_path=doc["relative_path"], source="local", title=doc["destination"], url="",
                                 heading_path=headings, line_start=start, line_end=end, content=text,
                                 content_sha256=checksum, role_in_review="distractor" if spec.startswith("!") else "support"))
        claims = [dict(c, evidence_chunk_ids=[contexts[i - 1]["chunk_id"] for i in c["citations"]],
                       claim_kind="grounded_fact_or_recommendation" if c["citations"] else "missing_information_statement")
                  for c in sample["claims"]]
        answer = "\n\n".join(c["text"] + "".join(f"[{i}]" for i in c["citations"]) for c in claims)
        row = dict(id=f"travel-sft-revision-{number:03d}", split="train_revision", task_type=sample["task_type"],
                   destination=sample["destination"], item_names=[sample["destination"]], original_question=sample["question"],
                   rewritten_query=sample["question"], history=[], contexts=contexts, target_answer=answer, claims=claims,
                   answerability=sample.get("answerability", "answerable"), missing_information=sample.get("missing_information", []),
                   task_tags=["targeted_behavior_revision"],
                   response_requirements=[dict(name=name, claim_indices=indices) for name, indices in sample["response_requirements"]],
                   generation_provenance="assistant_authored_from_training_sources_not_validation_answer_copy",
                   review_status="assistant_source_checked_user_pending")
        prompt = render(template, row)
        row.update(prompt_character_count=len(prompt), target_character_count=len(answer))
        rows.append(row)
        messages.append(dict(prompt=[dict(role="user", content=prompt)], completion=[dict(role="assistant", content=answer)]))
    files["samples.raw.jsonl"] = jsonl_bytes(rows)
    files["train.messages.jsonl"] = jsonl_bytes(messages)
    review = ["# revision_v2：原50条保留，新增32条审核稿", "", "子需求映射只是审核标注，不会输入模型；语义与业务口径仍需审核。", ""]
    for row in rows:
        review += [f"## {row['id']} · {row['destination']} · {row['task_type']}", "", row["original_question"], "", "目标回答：", "", row["target_answer"], ""]
        for item in row.get("response_requirements", []):
            review.append(f"- 子需求：{item['name']} → 回答段落 {item['claim_indices']}")
        review.append("")
        for c in row["contexts"]:
            review += [f"证据[{c['reference_index']}] {c['relative_path']} L{c['line_start']}-{c['line_end']}", "",
                       "```text", c["content"], "```", ""]
    files["review_samples.md"] = "\n".join(review).encode("utf-8")
    supplement_tasks = dict(Counter(s["task_type"] for s in SAMPLES))
    manifest = dict(dataset_version="revision_v2", created_at_utc=datetime.now(timezone.utc).isoformat(),
                    sample_count=len(rows), parent_sample_count=len(parent_rows), supplement_sample_count=len(SAMPLES),
                    parent_raw_sha256=sha((pilot / "samples.raw.jsonl").read_bytes()),
                    parent_messages_sha256=sha((pilot / "train.messages.jsonl").read_bytes()),
                    parent_manifest_sha256=sha((pilot / "source_manifest.json").read_bytes()),
                    reserved_eval_manifest_sha256=sha((ROOT / "data/eval_v1/manifest.json").read_bytes()),
                    documents=list(docs.values()), reserved_doc_ids=sorted(d["doc_id"] for d in parent["documents"] if d["split"] != "train_pilot"),
                    supplement_task_counts=supplement_tasks, prompt_template_sha256=parent["prompt_template_sha256"],
                    review_status="assistant_source_checked_user_pending",
                    context_selection="assistant_selected_exact_excerpts_not_live_retrieval",
                    frozen_files_sha256={name: sha(content) for name, content in sorted(files.items())})
    if (output / "source_manifest.json").exists():
        old, _, report = validate_dataset(output)
        require(old["frozen_files_sha256"] == manifest["frozen_files_sha256"], "Revision changed; create a new dataset version.")
        return report
    output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (output / "source_manifest.json").write_bytes(json_bytes(manifest))
    _, _, report = validate_dataset(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=ROOT / "data/pilot_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/revision_v2")
    args = parser.parse_args()
    print(json.dumps(build(args.pilot_dir, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
