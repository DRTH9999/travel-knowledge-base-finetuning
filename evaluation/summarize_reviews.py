"""Summarize explicitly completed human scores; never fabricate missing scores."""
import argparse
import json
from pathlib import Path

from common import json_bytes, read_jsonl, require
from evaluate import DIMENSIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.result_dir
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    require(report["status"] == "EVAL_GENERATION_OK", "Generation must finish first.")
    reviews = json.loads((directory / "manual_reviews.json").read_text(encoding="utf-8"))["reviews"]
    require(len(reviews) == report["samples"] * 2, "Incomplete review rows.")
    expected_keys = {(r["id"], r["model"]) for label in ("base", "adapter")
                     for r in read_jsonl(directory / f"{label}.predictions.jsonl")}
    seen, grouped = set(), {"base": [], "adapter": []}
    for review in reviews:
        key = (review["id"], review["model"])
        require(key not in seen, "Duplicate review row.")
        seen.add(key)
        require(review["model"] in grouped, "Unknown model label.")
        require(review["status"] == "reviewed" and review["reviewer"].strip(), "Finish and identify every review first.")
        require(set(review["scores"]) == set(DIMENSIONS), "Review dimensions differ.")
        require(all(type(v) is int and 0 <= v <= 2 for v in review["scores"].values()), "Scores must be integers 0-2; null is not zero.")
        grouped[review["model"]].append(review)
    expected_ids = {r["id"] for r in grouped["base"]}
    require(seen == expected_keys, "Review rows do not match actual model predictions.")
    require(expected_ids == {r["id"] for r in grouped["adapter"]} and len(expected_ids) == report["samples"], "Unpaired model reviews.")
    output = dict(status="MANUAL_REVIEW_SUMMARY_OK", split=report["split"], complete_split=report["complete_split"],
                  samples_per_model=report["samples"], score_scale="Each dimension 0-2; total 0-10; human rubric score, not accuracy.", models={})
    for label, items in grouped.items():
        output["models"][label] = dict(mean_scores={key: sum(r["scores"][key] for r in items) / len(items) for key in DIMENSIONS},
                                      mean_total=sum(sum(r["scores"].values()) for r in items) / len(items))
    (directory / "manual_summary.json").write_bytes(json_bytes(output))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
