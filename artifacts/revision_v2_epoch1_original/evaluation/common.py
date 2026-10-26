"""Shared data/provenance helpers; standard library only."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized(text):
    return re.sub(r"[\W_]+", "", text).lower()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(rows):
    return ("\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows) + "\n").encode("utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def excerpt(document, headings):
    lines, stack, hits = document.splitlines(), [], []
    for index, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            level, title = len(match[1]), match[2]
            stack = [(n, t) for n, t in stack if n < level] + [(level, title)]
            if [t for n, t in stack if n >= 2] == headings:
                hits.append((index, level))
    require(len(hits) == 1, f"Missing or ambiguous section: {headings}")
    start, level = hits[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        match = re.match(r"^(#{1,6})\s+", lines[index])
        if match and len(match[1]) <= level:
            end = index
            break
    while end > start and not lines[end - 1].strip():
        end -= 1
    return start + 1, end, "\n".join(lines[start:end])


def render_prompt(template, contexts, history, destination, question):
    content = "".join(f"[{c['reference_index']}][local][{destination}][]\n{c['content']}\n\n" for c in contexts)
    previous = "".join(f"[{h['role']}]: {h['text']}\n\n" for h in history)
    return template.format(context=content, history=previous, item_names=destination, question=question)


def validate_frozen(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files_sha256"].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), f"Invalid manifest path: {name}")
        require(path.is_file() and sha(path.read_bytes()) == digest, f"Frozen file changed: {name}")
    template = (directory / "prompt_template.txt").read_text(encoding="utf-8")
    docs = {d["doc_id"]: d for d in manifest["documents"]}
    seen_questions = set()
    report = {}
    for split, destination in (("validation", "厦门"), ("test", "张家界")):
        rows = read_jsonl(directory / f"{split}.jsonl")
        require(len(rows) == 24, f"Expected 24 {split} rows.")
        used = set()
        for index, row in enumerate(rows, 1):
            require(row["id"] == f"travel-eval-{split}-{index:03d}", "Unstable evaluation ID.")
            require(row["split"] == split and row["destination"] == destination, "Split/destination mismatch.")
            fingerprint = sha(normalized(row["question"]).encode("utf-8"))
            require(fingerprint not in seen_questions, "Duplicate held-out question.")
            require(fingerprint not in manifest["training_question_hashes"], "Training question leaked into holdouts.")
            seen_questions.add(fingerprint)
            for ref, context in enumerate(row["contexts"], 1):
                doc = docs[context["doc_id"]]
                require(doc["eval_split"] == split, "Cross-split context.")
                require(context["doc_id"] not in manifest["training_doc_ids"], "Training source used in holdouts.")
                require(context["reference_index"] == ref, "Invalid reference numbering.")
                text = (directory / doc["snapshot_path"]).read_text(encoding="utf-8-sig").splitlines()
                original = "\n".join(text[context["line_start"] - 1:context["line_end"]])
                require(original == context["content"], "Source evidence was rewritten.")
                require(sha(original.encode("utf-8")) == context["sha256"], "Excerpt hash mismatch.")
                used.add(context["doc_id"])
            rebuilt = []
            for claim in row["expected_claims"]:
                for ref in claim["citations"]:
                    require(1 <= ref <= len(row["contexts"]), "Invalid gold citation.")
                    require(row["contexts"][ref - 1]["review_role"] == "support", "Gold cites irrelevant evidence.")
                require(claim["citations"] or row["answerability"] != "answerable", "Unsupported answerable gold claim.")
                rebuilt.append(claim["text"] + "".join(f"[{r}]" for r in claim["citations"]))
            require("\n\n".join(rebuilt) == row["reference_answer"], "Gold answer/claims disagree.")
            if row["answerability"] != "answerable":
                require(bool(row["missing_information"]), "Missing answerability rationale.")
            expected = render_prompt(template, row["contexts"], row["history"], destination, row["rewritten_query"])
            require(row["prompt"] == expected, "Prompt does not match the business template.")
        report[split] = dict(samples=len(rows), source_documents=len(used),
                             tasks=dict(Counter(r["task_type"] for r in rows)),
                             answerability=dict(Counter(r["answerability"] for r in rows)))
    return manifest, report
