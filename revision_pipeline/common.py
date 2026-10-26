"""Standard-library validation for versioned evidence-grounded training data."""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DESTINATIONS = {"成都", "杭州", "三亚", "云南"}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(data):
    return (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(rows):
    return ("\n".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) for x in rows) + "\n").encode("utf-8")


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def normalized(text):
    return re.sub(r"[\W_]+", "", text).lower()


def render(template, row):
    contexts = "".join(f"[{c['reference_index']}][local][{c['title']}][]\n{c['content']}\n\n" for c in row["contexts"])
    history = "".join(f"[{h['role']}]: {h['text']}\n\n" for h in row["history"])
    return template.format(context=contexts, history=history, item_names=",".join(row["item_names"]), question=row["rewritten_query"])


def validate_tokens(directory, rows, *, max_length=2048):
    directory = Path(directory)
    tokens = read_jsonl(directory / "train.tokenized.jsonl")
    rendered = read_jsonl(directory / "train.sft.jsonl")
    report = json.loads((directory / "tokenizer_report.json").read_text(encoding="utf-8"))
    require(len(tokens) == len(rendered) == len(report["sample_lengths"]) == len(rows), "Token export sample counts differ.")
    for row, token, string, size in zip(rows, tokens, rendered, report["sample_lengths"]):
        ids, labels, mask = token["input_ids"], token["labels"], token["attention_mask"]
        boundary = size["prompt_tokens"]
        require(size["id"] == row["id"], "Token export IDs/order differ from raw data.")
        require(all(type(x) is int and x >= 0 for x in ids), "Invalid input token IDs.")
        require(all(type(x) is int for x in labels), "Invalid label types.")
        require(len(ids) == len(labels) == len(mask) == size["total_tokens"], "Token/label lengths differ.")
        require(0 < boundary < len(ids) <= max_length, "Invalid boundary or overlong sample; truncation is forbidden.")
        require(mask == [1] * len(ids), "Unexpected padding inside exported sample.")
        require(labels[:boundary] == [-100] * boundary, "Prompt tokens contribute to loss.")
        require(labels[boundary:] == ids[boundary:], "Answer labels differ from answer token IDs.")
        require(ids[-1] == report["eos_token_id"], "Terminal EOS is missing.")
        require(string["completion"] == row["target_answer"] + report["eos_token"], "Rendered completion changed.")
    return tokens, dict(sample_count=len(tokens), max_tokens=max(len(x["input_ids"]) for x in tokens),
                        supervised_tokens=sum(sum(y != -100 for y in x["labels"]) for x in tokens),
                        tokenized_sha256=sha((directory / "train.tokenized.jsonl").read_bytes()))


def validate_dataset(directory, *, require_tokenized=False):
    directory = Path(directory)
    manifest = json.loads((directory / "source_manifest.json").read_text(encoding="utf-8"))
    require(manifest["dataset_version"] == "revision_v2", "Unexpected dataset version.")
    for name, expected in manifest["frozen_files_sha256"].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), "Invalid source path.")
        require(path.is_file() and sha(path.read_bytes()) == expected, f"Frozen file changed: {name}")
    rows = read_jsonl(directory / "samples.raw.jsonl")
    messages = read_jsonl(directory / "train.messages.jsonl")
    parent_count = manifest["parent_sample_count"]
    require(len(rows) == len(messages) == manifest["sample_count"], "Dataset count differs from manifest.")
    require(len(rows) == parent_count + manifest["supplement_sample_count"] and parent_count > 0, "Invalid parent/supplement counts.")
    require(sha(jsonl_bytes(rows[:parent_count])) == manifest["parent_raw_sha256"], "Original pilot rows were altered.")
    require(sha(jsonl_bytes(messages[:parent_count])) == manifest["parent_messages_sha256"], "Original pilot messages were altered.")
    require(len({r["id"] for r in rows}) == len(rows), "Duplicate sample IDs.")
    require(len({normalized(r["original_question"]) for r in rows}) == len(rows), "Duplicate exact questions.")
    docs = {d["doc_id"]: d for d in manifest["documents"]}
    require(len(docs) == 20 and not set(docs) & set(manifest["reserved_doc_ids"]), "Held-out source leaked into training.")
    require(all(d["destination"] in ALLOWED_DESTINATIONS and d["split"] == "train_pilot" for d in docs.values()), "Nontraining source.")
    for doc in docs.values():
        require(sha((directory / doc["snapshot_path"]).read_bytes()) == doc["sha256"], "Source inventory hash differs from snapshot.")
    template = (directory / "prompt_template.txt").read_text(encoding="utf-8")
    used, novel_numbers = set(), []
    for row, message in zip(rows, messages):
        require(row["destination"] in ALLOWED_DESTINATIONS, "Held-out destination leaked into training.")
        require(message["prompt"] == [dict(role="user", content=render(template, row))], "Business input mismatch.")
        require(message["completion"] == [dict(role="assistant", content=row["target_answer"])], "Target answer mismatch.")
        require("<think>" not in row["target_answer"] and "</think>" not in row["target_answer"], "Unexpected reasoning content.")
        for number, context in enumerate(row["contexts"], 1):
            doc = docs[context["doc_id"]]
            require(context["reference_index"] == number and context["relative_path"] == doc["relative_path"], "Invalid context identity.")
            require(context["source"] == "local" and context["url"] == "" and context["title"] == doc["destination"], "Invalid business source metadata.")
            lines = (directory / doc["snapshot_path"]).read_text(encoding="utf-8-sig").splitlines()
            require(1 <= context["line_start"] <= context["line_end"] <= len(lines), "Invalid source line interval.")
            exact = "\n".join(lines[context["line_start"] - 1:context["line_end"]])
            require(exact == context["content"] and sha(exact.encode("utf-8")) == context["content_sha256"], "Evidence was rewritten.")
            used.add(doc["doc_id"])
        target = []
        for claim in row["claims"]:
            for ref in claim["citations"]:
                require(type(ref) is int and 1 <= ref <= len(row["contexts"]), "Citation ID outside context.")
                require(row["contexts"][ref - 1]["role_in_review"] == "support", "Claim cites distractor.")
            require(claim["evidence_chunk_ids"] == [row["contexts"][i - 1]["chunk_id"] for i in claim["citations"]], "Claim provenance mapping differs.")
            require(claim["citations"] or row["answerability"] != "answerable", "Unsupported claim in answerable target.")
            target.append(claim["text"] + "".join(f"[{i}]" for i in claim["citations"]))
        require("\n\n".join(target) == row["target_answer"], "Claims/answer disagree.")
        if row["answerability"] != "answerable":
            require(bool(row["missing_information"]), "Missing uncertainty rationale.")
        if row.get("response_requirements"):
            covered = set()
            for requirement in row["response_requirements"]:
                require(requirement["claim_indices"], "Empty requirement mapping.")
                require(all(1 <= i <= len(row["claims"]) for i in requirement["claim_indices"]), "Invalid requirement claim mapping.")
                covered.update(requirement["claim_indices"])
            require(covered == set(range(1, len(row["claims"]) + 1)), "Target segment has no declared requirement mapping.")
        stripped = re.sub(r"\[\d+\]", "", row["target_answer"])
        visible = "\n".join([row["rewritten_query"]] + [c["content"] for c in row["contexts"]] + [h["text"] for h in row["history"]])
        missing_numbers = set(re.findall(r"\d+(?:\.\d+)?", stripped)) - set(re.findall(r"\d+(?:\.\d+)?", visible))
        if missing_numbers:
            novel_numbers.append(dict(id=row["id"], numbers=sorted(missing_numbers)))
        require(all(url in visible for url in re.findall(r"https?://[^\s)>]+", row["target_answer"])), "New target URL not in evidence.")
    require(not novel_numbers, f"Unmatched numeric facts require source review: {novel_numbers}")
    supplements = rows[parent_count:]
    require(all(r.get("response_requirements") for r in supplements), "Supplement lacks coverage annotations.")
    require(dict(Counter(r["task_type"] for r in supplements)) == manifest["supplement_task_counts"], "Supplement task counts differ.")
    near = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            ratio = SequenceMatcher(None, normalized(a["original_question"]), normalized(b["original_question"])).ratio()
            if ratio >= .85:
                near.append(dict(ids=[a["id"], b["id"]], question_similarity=round(ratio, 4)))
    report = dict(status="REVISION_DATA_CHECK_OK", dataset_version=manifest["dataset_version"],
                  samples=len(rows), parent_samples=parent_count, supplement_samples=len(supplements),
                  original_pilot_preserved=True, used_training_documents=len(used),
                  destination_counts=dict(Counter(r["destination"] for r in rows)),
                  answerability_counts=dict(Counter(r["answerability"] for r in rows)),
                  supplement_task_counts=manifest["supplement_task_counts"], reserved_source_in_training=0,
                  exact_duplicate_questions=0, question_near_duplicate_flags=near,
                  declared_requirement_mappings_checked=True, semantic_coverage_auto_verified=False,
                  user_business_review="pending", tokenized=False,
                  limitations=["Requirements are author annotations, not proof of semantic coverage.",
                               "Shared evidence/facts are intentional behavior exercises; 82 rows are not 82 independent fact groups.",
                               "Evidence is selected exact source text, not live RAG retrieval; no real-time tourism fact checking."])
    if (directory / "train.tokenized.jsonl").exists():
        _, token_report = validate_tokens(directory, rows)
        report.update(tokenized=True, **token_report)
    else:
        require(not require_tokenized, "Tokenizer export is required.")
    return manifest, rows, report
