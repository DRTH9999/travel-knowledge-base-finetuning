"""Replay assessed inputs for serving compatibility, not a new quality test."""
from datetime import datetime, timezone
import json

import httpx

from .client import check_response, client_args, fixture_rows, options
from .common import ROOT, json_bytes


def main():
    p = options(__doc__)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--limit", type=int)
    p.add_argument("--ack-test-is-regression", action="store_true")
    a = p.parse_args()
    if a.split == "test" and not a.ack_test_is_regression:
        p.error("Assessed final questions are only compatibility regression; add --ack-test-is-regression.")
    if a.limit is not None and not 1 <= a.limit <= 24:
        p.error("--limit must be 1-24.")
    base, headers = client_args(a)
    expected = json.loads((ROOT / "serving/candidate.json").read_text(encoding="utf-8"))
    rows = [r for r in fixture_rows() if r["split"] == a.split]
    rows = rows[:a.limit] if a.limit else rows
    folder = a.output_dir / ("replay-" + a.split + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    folder.mkdir(parents=True, exist_ok=False)
    results = []
    with httpx.Client(headers=headers, timeout=150, trust_env=False) as client:
        identity = check_response(client.get(base + "/identity"))
        assert identity["backend"] == "transformers_peft_nf4"
        assert identity["actual"]["adapter_sha256"] == expected["adapter_sha256"]
        with (folder / "predictions.jsonl").open("x", encoding="utf-8") as f:
            for row in rows:
                response = client.post(base + "/chat/completions", json=dict(model=expected["name"],
                    messages=[dict(role="user", content=row["prompt"])], temperature=0, max_tokens=512))
                body = check_response(response)
                assert response.headers["X-Input-SHA256"] == row["expected_input_sha256"], row["id"]
                assert body["usage"]["prompt_tokens"] == row["expected_prompt_tokens"], row["id"]
                answer = body["choices"][0]["message"]["content"]
                result = dict(id=row["id"], input_sha256=row["expected_input_sha256"],
                              answer=answer, previous_answer=row["previous_answer"],
                              matches_previous_answer=answer.strip() == row["previous_answer"].strip(),
                              finish_reason=body["choices"][0]["finish_reason"], usage=body["usage"],
                              request_id=body["id"], semantic_review_status="pending")
                results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                print(row["id"], "INPUT_MATCH", "SAME_TEXT" if result["matches_previous_answer"] else "REVIEW_DIFFERENCE", flush=True)
    report = dict(status="SERVING_HTTP_REPLAY_OK", split=a.split, samples=len(results), complete_split=len(rows) == 24,
                  identity=identity, all_inputs_match=True, same_text_count=sum(r["matches_previous_answer"] for r in results),
                  purpose="compatibility regression of already assessed inputs, not new independent quality evaluation",
                  semantic_review_status="pending", rag_integration_verified=False)
    (folder / "report.json").write_bytes(json_bytes(report))
    print(json.dumps(dict(status=report["status"], samples=len(results), output_dir=str(folder)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
