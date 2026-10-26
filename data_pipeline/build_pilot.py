"""Build a traceable pilot without calling a generation API or training a model."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

from curated_pilot import SAMPLES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAG = Path(r"E:\260515\travel_knowledge_base")
DESTINATIONS = {"cd": "成都", "hz": "杭州", "sy": "三亚", "yn": "云南",
                "xm": "厦门", "zjj": "张家界"}
CATEGORIES = {"t": ("交通指南", "交通指南"), "s": ("景点攻略", "景点推荐"),
              "r": ("线路推荐", "线路推荐"), "f": ("美食推荐", "美食推荐"),
              "h": ("酒店信息", "住宿推荐")}
SPLITS = {d: ("validation_reserved" if d == "厦门" else
              "test_reserved" if d == "张家界" else "train_pilot")
          for d in DESTINATIONS.values()}
TASK_NAMES = {"single_evidence": "单片段问答", "multi_evidence": "多片段整合",
              "constrained_planning": "多条件规划", "history_followup": "历史追问",
              "insufficient_evidence": "信息不足或部分可答", "distractor_filtering": "干扰筛选"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def extract_template(path: Path):
    for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "ANSWER_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("ANSWER_PROMPT not found")


def freeze_sources(rag_root: Path, output: Path):
    original = rag_root / "旅游" / "数据"
    docs = []
    for prefix, destination in DESTINATIONS.items():
        for suffix, (category, stem) in CATEGORIES.items():
            relative = f"{category}/{destination}{stem}.md"
            content = (original / relative).read_bytes()
            docs.append(dict(doc_id=f"{prefix}_{suffix}", relative_path=relative,
                             destination=destination, category=category, split=SPLITS[destination],
                             sha256=sha(content), byte_count=len(content),
                             character_count=len(content.decode("utf-8-sig")),
                             snapshot_path=f"sources/{relative}",
                             source_date=None, authority_verified=False))
    template = extract_template(rag_root / "atguigu/config/prompt.py")
    template_hash = sha(template.encode("utf-8"))
    manifest_path = output / "source_manifest.json"
    if manifest_path.exists():
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
        old = {d["doc_id"]: d["sha256"] for d in frozen["documents"]}
        assert old == {d["doc_id"]: d["sha256"] for d in docs}, "Source changed: create a new dataset version"
        assert frozen["prompt_template_sha256"] == template_hash, "Prompt changed: create a new version"
        for doc in docs:
            assert sha((output / doc["snapshot_path"]).read_bytes()) == doc["sha256"], "Snapshot modified"
        assert sha((output / "prompt_template.txt").read_bytes()) == template_hash
        return frozen, template
    for doc in docs:
        target = output / doc["snapshot_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / doc["relative_path"]).read_bytes())
    (output / "prompt_template.txt").write_bytes(template.encode("utf-8"))
    manifest = dict(dataset_version="pilot_v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
                    source_root_at_build=str(original), documents=docs,
                    prompt_template_sha256=template_hash,
                    source_partition_status="pilot_reservation_not_final_evaluation_design",
                    context_selection="assistant_selected_exact_heading_excerpts_not_live_retrieval",
                    generation_provenance="assistant_authored_from_local_documents_no_paid_generation_api",
                    review_status="assistant_checked_user_pending")
    write_json(manifest_path, manifest)
    return manifest, template


def excerpt(document: str, headings: list[str]):
    lines = document.splitlines()
    stack = []
    hits = []
    for i, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            continue
        level, title = len(match[1]), match[2]
        stack = [(n, t) for n, t in stack if n < level] + [(level, title)]
        if [t for n, t in stack if n >= 2] == headings:
            hits.append((i, level))
    assert len(hits) == 1, f"Nonunique or missing section: {headings}: {hits}"
    start, level = hits[0]
    end = len(lines)
    for j in range(start + 1, len(lines)):
        match = re.match(r"^(#{1,6})\s+", lines[j])
        if match and len(match[1]) <= level:
            end = j
            break
    while end > start and not lines[end - 1].strip():
        end -= 1
    return start + 1, end, "\n".join(lines[start:end])


def make_examples(output: Path, manifest, template: str):
    docs = {d["doc_id"]: d for d in manifest["documents"]}
    examples, messages = [], []
    for index, authored in enumerate(SAMPLES, 1):
        sample_id = f"travel-sft-pilot-{index:03d}"
        contexts = []
        for reference_index, spec in enumerate(authored["references"], 1):
            irrelevant = spec.startswith("!")
            alias, *headings = spec.lstrip("!").split("/")
            doc = docs[alias]
            assert doc["split"] == "train_pilot", f"Reserved source in training: {alias}"
            text = (output / doc["snapshot_path"]).read_text(encoding="utf-8-sig")
            start, end, content = excerpt(text, headings)
            chunk_id = f"{alias}:L{start}-L{end}:{sha(content.encode('utf-8'))[:12]}"
            contexts.append(dict(reference_index=reference_index, doc_id=alias, chunk_id=chunk_id,
                                 relative_path=doc["relative_path"], source="local", title=doc["destination"],
                                 url="", heading_path=headings, line_start=start, line_end=end,
                                 content=content, content_sha256=sha(content.encode("utf-8")),
                                 role_in_review="distractor" if irrelevant else "support"))
        claims = []
        for claim in authored["claims"]:
            citations = claim["citations"]
            for n in citations:
                assert 1 <= n <= len(contexts)
                assert contexts[n - 1]["role_in_review"] == "support"
            claims.append(dict(text=claim["text"], citations=citations,
                               evidence_chunk_ids=[contexts[n - 1]["chunk_id"] for n in citations],
                               claim_kind="grounded_fact_or_recommendation" if citations else "missing_information_statement",
                               review_note="assistant_source_review_user_review_pending"))
        answer = "\n\n".join(c["text"] + "".join(f"[{n}]" for n in c["citations"]) for c in claims)
        history = authored.get("history", [])
        context_str = "".join(f"[{c['reference_index']}][{c['source']}][{c['title']}][{c['url']}]\n{c['content']}\n\n"
                              for c in contexts)
        history_str = "".join(f"[{m['role']}]: {m['text']}\n\n" for m in history)
        item_names = authored.get("item_names", [authored["destination"]])
        rewritten = authored.get("rewritten_query", authored["question"])
        prompt = template.format(context=context_str, history=history_str,
                                 item_names=",".join(item_names), question=rewritten)
        assert len(prompt) < 10000, f"Prompt exceeds existing RAG character limit: {sample_id}"
        row = dict(id=sample_id, split="train_pilot", task_type=authored["task_type"],
                   destination=authored["destination"], item_names=item_names,
                   original_question=authored["question"], rewritten_query=rewritten,
                   history=history, contexts=contexts, target_answer=answer, claims=claims,
                   answerability=authored.get("answerability", "answerable"),
                   missing_information=authored.get("missing_information", []),
                   task_tags=authored.get("task_tags", []),
                   generation_provenance="assistant_authored_synthetic_not_production_log",
                   review_status="assistant_checked_user_pending",
                   prompt_character_count=len(prompt), target_character_count=len(answer))
        examples.append(row)
        messages.append(dict(prompt=[dict(role="user", content=prompt)],
                             completion=[dict(role="assistant", content=answer)]))
    assert len(examples) == 50
    write_jsonl(output / "samples.raw.jsonl", examples)
    write_jsonl(output / "train.messages.jsonl", messages)
    return examples


def source_duplicate_report(output: Path, manifest):
    occurrences = defaultdict(list)
    for doc in manifest["documents"]:
        lines = (output / doc["snapshot_path"]).read_text(encoding="utf-8-sig").splitlines()
        for number, line in enumerate(lines, 1):
            text = line.strip()
            if len(text) >= 30 and not text.startswith(("#", "-", "*", ">")):
                occurrences[text].append(dict(doc_id=doc["doc_id"], split=doc["split"], line=number))
    duplicates = [dict(text=t, locations=locs) for t, locs in occurrences.items() if len(locs) > 1]
    crossing = [d for d in duplicates if len({x["split"] for x in d["locations"]}) > 1]
    return dict(method="exact_trimmed_body_lines_at_least_30_characters_excluding_list_and_heading_lines",
                duplicate_line_groups=len(duplicates), cross_split_duplicate_line_groups=len(crossing),
                duplicates=duplicates,
                limitation="Not a semantic leakage detector; templates and general travel advice can still overlap.")


def write_review(output: Path, examples):
    rows = ["# 旅游 RAG SFT 首批 50 条样例审核稿", "",
            "状态：助手已按原文检查，等待用户审核。所有问答和历史为合成内容，不是真实用户日志。", "",
            "证据是人工选择的原文标题片段，不代表现有检索系统的实际召回结果；来源日期未提供。",
            "首批只训练最终回答，不训练查询改写、HyDE 或分类。样例中引用编号每题重新编号。", "",
            "## 覆盖统计", "", "| 类型 | 数量 |", "| --- | ---: |"]
    for task, count in Counter(x["task_type"] for x in examples).items():
        rows.append(f"| {TASK_NAMES[task]} | {count} |")
    rows += ["", "训练来源：成都、杭州、三亚、云南。厦门和张家界整组预留，不参与本批训练。", "",
             "建议重点看 017（行程）、038（纠正旧回答）、042（部分可答）、045（不编造实时信息）、050（干扰引用）。", "",
             "审核时关注：答案是否受证据支持、建议是否合适、引用是否准确，以及拒答与篇幅是否符合预期。", ""]
    for x in examples:
        rows += [f"## {x['id']}｜{TASK_NAMES[x['task_type']]}｜{x['destination']}", "",
                 f"**用户问题：** {x['original_question']}", ""]
        if x["history"]:
            rows += ["**历史语境（合成）：**", ""]
            for m in x["history"]:
                rows.append(f"- {m['role']}：{m['text']}")
            rows += ["", f"**送入回答节点的改写问题：** {x['rewritten_query']}", ""]
        if "history_correction" in x["task_tags"]:
            rows += ["本题故意在历史答案中设置错误；期望模型根据当前证据纠正旧回答。", ""]
        rows += ["**目标回答：**", "", x["target_answer"], "", "**原文证据：**", ""]
        for c in x["contexts"]:
            review_role = "干扰片段，不应引用" if c["role_in_review"] == "distractor" else "支持片段"
            rows += [f"[{c['reference_index']}] `{c['relative_path']}`，L{c['line_start']}–L{c['line_end']}（{review_role}）", "",
                     "```text", c["content"], "```", ""]
        rows += ["**逐段引用检查：**", ""]
        for n, c in enumerate(x["claims"], 1):
            refs = "、".join(f"[{i}]" for i in c["citations"]) or "缺失信息声明，不主张已查到事实"
            rows.append(f"- 回答第 {n} 段 → {refs}")
        if x["missing_information"]:
            rows += ["", "**未提供的信息：** " + "、".join(x["missing_information"])]
        rows.append("")
    (output / "review_samples.md").write_text("\n".join(rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-root", type=Path, default=DEFAULT_RAG)
    parser.add_argument("--output", type=Path, default=ROOT / "data/pilot_v1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, template = freeze_sources(args.rag_root, args.output)
    examples = make_examples(args.output, manifest, template)
    write_review(args.output, examples)
    write_json(args.output / "source_duplicate_report.json", source_duplicate_report(args.output, manifest))
    print(json.dumps(dict(samples=len(examples), tasks=dict(Counter(x["task_type"] for x in examples)),
                          destinations=dict(Counter(x["destination"] for x in examples)),
                          output=str(args.output)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
