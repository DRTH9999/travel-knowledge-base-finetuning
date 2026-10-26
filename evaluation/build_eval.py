"""Freeze 24 validation + 24 test questions from reserved source snapshots."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from common import ROOT, excerpt, json_bytes, jsonl_bytes, normalized, read_jsonl, render_prompt, require, sha, validate_frozen
from curated_eval import TEST, VALIDATION


def review_markdown(split, rows):
    text = [f"# {split} 评测题与原文依据", "", "问答与历史均为助手合成；来源未经实时外部核验。", "",
            "参考答案和证据相关性标签仅用于审核，不会放入模型输入。", ""]
    for row in rows:
        text += [f"## {row['id']} · {row['task_type']}", "", f"问题：{row['question']}", ""]
        if row["history"]:
            text += ["历史：" + json.dumps(row["history"], ensure_ascii=False), ""]
        text += ["参考答案：", "", row["reference_answer"], "", "证据：", ""]
        for context in row["contexts"]:
            text += [f"[{context['reference_index']}] {context['relative_path']} L{context['line_start']}-{context['line_end']} ({context['review_role']})", "",
                     "```text", context["content"], "```", ""]
        if not row["contexts"]:
            text += ["无检索片段。", ""]
    return "\n".join(text).encode("utf-8")


def build(pilot, output):
    source = json.loads((pilot / "source_manifest.json").read_text(encoding="utf-8"))
    docs = {d["doc_id"]: d for d in source["documents"]}
    files = {"prompt_template.txt": (pilot / "prompt_template.txt").read_bytes()}
    require(sha(files["prompt_template.txt"]) == source["prompt_template_sha256"], "Pilot template changed.")
    template = files["prompt_template.txt"].decode("utf-8")
    held_docs = []
    for doc in docs.values():
        if doc["split"] == "train_pilot":
            continue
        snapshot = (pilot / doc["snapshot_path"]).read_bytes()
        require(sha(snapshot) == doc["sha256"], "Pilot source snapshot changed.")
        files[doc["snapshot_path"]] = snapshot
        held_docs.append(dict(doc, eval_split="validation" if doc["destination"] == "厦门" else "test"))
    for split, destination, authored in (("validation", "厦门", VALIDATION), ("test", "张家界", TEST)):
        rows = []
        for index, item in enumerate(authored, 1):
            contexts = []
            for ref, spec in enumerate(item["references"], 1):
                alias, *headings = spec.lstrip("!").split("/")
                doc = docs[alias]
                require(doc["destination"] == destination and doc["split"] != "train_pilot", "Wrong holdout source.")
                start, end, text = excerpt(files[doc["snapshot_path"]].decode("utf-8-sig"), headings)
                contexts.append(dict(reference_index=ref, doc_id=alias, relative_path=doc["relative_path"],
                                     line_start=start, line_end=end, heading_path=headings, content=text,
                                     sha256=sha(text.encode("utf-8")),
                                     review_role="distractor" if spec.startswith("!") else "support"))
            history = item.get("history", [])
            rewritten = item.get("rewritten_query", item["question"])
            gold = "\n\n".join(c["text"] + "".join(f"[{r}]" for r in c["citations"]) for c in item["claims"])
            rows.append(dict(id=f"travel-eval-{split}-{index:03d}", split=split, destination=destination,
                             task_type=item["task_type"], question=item["question"], rewritten_query=rewritten,
                             history=history, contexts=contexts,
                             prompt=render_prompt(template, contexts, history, destination, rewritten),
                             reference_answer=gold, expected_claims=item["claims"],
                             answerability=item.get("answerability", "answerable"),
                             missing_information=item.get("missing_information", []), task_tags=item.get("task_tags", []),
                             review_status="assistant_source_checked_user_pending"))
        require(len(rows) == 24, "Expected 24 rows per split.")
        files[f"{split}.jsonl"] = jsonl_bytes(rows)
        files[f"review.{split}.md"] = review_markdown(split, rows)
    manifest = dict(dataset_version="eval_v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
                    documents=held_docs, training_doc_ids=sorted(d["doc_id"] for d in docs.values() if d["split"] == "train_pilot"),
                    training_question_hashes=sorted(sha(normalized(r["original_question"]).encode("utf-8"))
                                                    for r in read_jsonl(pilot / "samples.raw.jsonl")),
                    files_sha256={name: sha(data) for name, data in sorted(files.items())},
                    source_authority_verified=False, source_date_verified=False,
                    context_selection="assistant_selected_exact_source_excerpts_not_live_retrieval",
                    question_provenance="assistant_authored_without_looking_at_holdout_model_outputs",
                    test_policy="frozen; do not use test results to choose checkpoints or tune prompts",
                    limitations=["One held-out destination per split; results do not establish general tourism performance.",
                                 "Shared facts and templates exist within each destination; rows are not independent fact groups.",
                                 "Gold answers still require business review; automatic checks do not establish semantic entailment."])
    if (output / "manifest.json").exists():
        old, report = validate_frozen(output)
        require(old["files_sha256"] == manifest["files_sha256"], "Frozen eval changed; create a new data version.")
        require(old["training_question_hashes"] == manifest["training_question_hashes"], "Training pilot changed; review evaluation isolation.")
        return report
    output.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    _, report = validate_frozen(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=ROOT / "data/pilot_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/eval_v1")
    args = parser.parse_args()
    report = build(args.pilot_dir, args.output)
    print(json.dumps(dict(status="EVAL_DATA_BUILD_OK", output=str(args.output), splits=report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
