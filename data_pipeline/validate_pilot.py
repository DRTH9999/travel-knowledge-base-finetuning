"""Independent checks of generated artifacts and their frozen source evidence.

These are structural/provenance checks, not a claim-entailment or business-quality
score. They never mark user review or training acceptance as completed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def normalized(text):
    return re.sub(r"[\W_]+", "", text)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=root / "data/pilot_v1")
    parser.add_argument("--require-tokenized", action="store_true")
    args = parser.parse_args()
    directory = args.dataset_dir
    manifest = json.loads((directory / "source_manifest.json").read_text(encoding="utf-8"))
    docs = {d["doc_id"]: d for d in manifest["documents"]}
    assert len(docs) == 30
    assert Counter(d["split"] for d in docs.values()) == dict(train_pilot=20, validation_reserved=5, test_reserved=5)
    for doc in docs.values():
        content = (directory / doc["snapshot_path"]).read_bytes()
        assert sha(content) == doc["sha256"], f"Changed source snapshot: {doc['doc_id']}"
        assert len(content) == doc["byte_count"]
    template = (directory / "prompt_template.txt").read_text(encoding="utf-8")
    assert sha(template.encode("utf-8")) == manifest["prompt_template_sha256"]
    raw = read_jsonl(directory / "samples.raw.jsonl")
    messages = read_jsonl(directory / "train.messages.jsonl")
    assert len(raw) == len(messages) == 50
    assert len({r["id"] for r in raw}) == 50
    assert len({normalized(r["original_question"]) for r in raw}) == 50
    assert len({normalized(r["target_answer"]) for r in raw}) == 50
    used_docs, unique_chunks = set(), set()
    context_occurrences = 0
    number_flags = []
    for n, (sample, message) in enumerate(zip(raw, messages), 1):
        assert sample["id"] == f"travel-sft-pilot-{n:03d}"
        assert sample["split"] == "train_pilot"
        assert sample["review_status"] == "assistant_checked_user_pending"
        assert len(message["prompt"]) == len(message["completion"]) == 1
        assert message["prompt"][0]["role"] == "user"
        assert message["completion"][0]["role"] == "assistant"
        target = sample["target_answer"]
        assert message["completion"][0]["content"] == target
        assert target.strip() and "<think>" not in target and "</think>" not in target
        contexts = sample["contexts"]
        for index, c in enumerate(contexts, 1):
            doc = docs[c["doc_id"]]
            assert doc["split"] == "train_pilot", f"Reserved source leaked: {sample['id']}"
            assert doc["relative_path"] == c["relative_path"]
            assert c["title"] == doc["destination"]
            assert c["source"] == "local" and c["url"] == ""
            assert c["reference_index"] == index
            lines = (directory / doc["snapshot_path"]).read_text(encoding="utf-8-sig").splitlines()
            assert 1 <= c["line_start"] <= c["line_end"] <= len(lines)
            expected = "\n".join(lines[c["line_start"] - 1:c["line_end"]])
            assert c["content"] == expected, f"Altered evidence: {sample['id']} ref {index}"
            assert c["content_sha256"] == sha(expected.encode("utf-8"))
            used_docs.add(c["doc_id"])
            unique_chunks.add(c["chunk_id"])
            context_occurrences += 1
        rebuilt = []
        for claim in sample["claims"]:
            citations = claim["citations"]
            assert len(set(citations)) == len(citations)
            for ref in citations:
                assert isinstance(ref, int) and 1 <= ref <= len(contexts)
                assert contexts[ref - 1]["role_in_review"] == "support"
            assert claim["evidence_chunk_ids"] == [contexts[i - 1]["chunk_id"] for i in citations]
            if not citations:
                assert claim["claim_kind"] == "missing_information_statement"
                assert sample["answerability"] in ("unanswerable", "partially_answerable")
            rebuilt.append(claim["text"] + "".join(f"[{i}]" for i in citations))
        assert "\n\n".join(rebuilt) == target
        cited = {int(x) for x in re.findall(r"\[(\d+)\]", target)}
        supports = {c["reference_index"] for c in contexts if c["role_in_review"] == "support"}
        if sample["answerability"] != "unanswerable":
            assert cited == supports, f"Citation mismatch: {sample['id']}"
        else:
            assert not cited and sample["missing_information"]
        if sample["answerability"] == "partially_answerable":
            assert cited and sample["missing_information"]
        context_str = "".join(f"[{c['reference_index']}][local][{c['title']}][]\n{c['content']}\n\n" for c in contexts)
        history_str = "".join(f"[{h['role']}]: {h['text']}\n\n" for h in sample["history"])
        expected_prompt = template.format(context=context_str, history=history_str,
                                          item_names=",".join(sample["item_names"]), question=sample["rewritten_query"])
        assert message["prompt"][0]["content"] == expected_prompt
        assert len(expected_prompt) == sample["prompt_character_count"] < 10000
        assert len(target) == sample["target_character_count"]
        # Detect novel Arabic numerals. This is a narrow anomaly scan, not a fact checker.
        answer_numbers = set(re.findall(r"\d+(?:\.\d+)?", re.sub(r"\[\d+\]", "", target)))
        visible = "\n".join([c["content"] for c in contexts] + [sample["rewritten_query"]] +
                            [h["text"] for h in sample["history"]])
        extras = sorted(answer_numbers - set(re.findall(r"\d+(?:\.\d+)?", visible)))
        if extras:
            number_flags.append(dict(id=sample["id"], unmatched_numbers=extras))
        urls = re.findall(r"https?://[^\s)>]+", target)
        assert all(url in visible for url in urls), f"Novel URL in {sample['id']}"
    near_duplicates, shared_evidence = [], []
    for a, left in enumerate(raw):
        for right in raw[a + 1:]:
            ratio = SequenceMatcher(None, normalized(left["original_question"]), normalized(right["original_question"])).ratio()
            if ratio >= .8:
                near_duplicates.append(dict(ids=[left["id"], right["id"]], question_similarity=round(ratio, 4)))
            ls = {c["chunk_id"] for c in left["contexts"] if c["role_in_review"] == "support"}
            rs = {c["chunk_id"] for c in right["contexts"] if c["role_in_review"] == "support"}
            if ls & rs and len(ls & rs) / len(ls | rs) >= .65:
                shared_evidence.append(dict(ids=[left["id"], right["id"]], shared_chunk_ids=sorted(ls & rs)))
    assert not number_flags, f"Unmatched numerals require review: {number_flags}"
    token_status = "not_exported_yet"
    token_report = None
    if (directory / "train.tokenized.jsonl").exists():
        tokenized = read_jsonl(directory / "train.tokenized.jsonl")
        rendered = read_jsonl(directory / "train.sft.jsonl")
        token_report = json.loads((directory / "tokenizer_report.json").read_text(encoding="utf-8"))
        assert len(tokenized) == len(rendered) == len(raw)
        for row, string, stats, original in zip(tokenized, rendered, token_report["sample_lengths"], raw):
            ids, labels, attention = row["input_ids"], row["labels"], row["attention_mask"]
            boundary = stats["prompt_tokens"]
            assert len(ids) == len(labels) == len(attention) == stats["total_tokens"]
            assert attention == [1] * len(ids)
            assert labels[:boundary] == [-100] * boundary
            assert labels[boundary:] == ids[boundary:]
            assert all(x >= 0 for x in labels[boundary:])
            assert ids[-1] == labels[-1] == token_report["eos_token_id"]
            assert string["completion"] == original["target_answer"] + token_report["eos_token"]
            assert len(ids) <= token_report["suggested_pilot_max_length"]
        token_status = "local_render_and_label_checks_passed_cloud_recheck_pending"
    elif args.require_tokenized:
        raise AssertionError("Tokenized exports are required")
    report = dict(status="DATA_CHECK_OK", sample_count=len(raw),
                  task_counts=dict(Counter(r["task_type"] for r in raw)),
                  destination_counts=dict(Counter(r["destination"] for r in raw)),
                  answerability_counts=dict(Counter(r["answerability"] for r in raw)),
                  source_documents_total=len(docs), used_training_documents=len(used_docs),
                  context_occurrences=context_occurrences, unique_source_excerpts=len(unique_chunks),
                  reserved_source_in_training_count=0, exact_duplicate_questions=0,
                  exact_duplicate_answers=0, question_near_duplicates=near_duplicates,
                  shared_evidence_pairs=shared_evidence, unmatched_arabic_number_flags=number_flags,
                  max_business_prompt_characters=max(r["prompt_character_count"] for r in raw),
                  model_weights_loaded=False, training_started=False, tokenizer_status=token_status,
                  semantic_entailment_auto_verified=False, user_review_status="pending",
                  limitations=["Source evidence was selected by the assistant, not by live RAG retrieval.",
                               "All questions, answers and histories are synthetic; 50 rows are not 50 independent fact groups.",
                               "No source dates or authority verification; no live tourism information was checked.",
                               "Exact-line duplicate checks cannot rule out semantic overlap across reserved destinations.",
                               "No held-out questions have been generated; no baseline or fine-tuned model evaluation has run."])
    if token_report:
        report.update(max_total_tokens=token_report["max_total_tokens"],
                      suggested_pilot_max_length=token_report["suggested_pilot_max_length"])
    (directory / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_paths = [p for p in directory.rglob("*") if p.is_file() and p.name != "checksums.sha256"]
    checksum_lines = [f"{sha(p.read_bytes())}  {p.relative_to(directory).as_posix()}" for p in sorted(checksum_paths)]
    (directory / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("shared_evidence_pairs", "limitations")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
