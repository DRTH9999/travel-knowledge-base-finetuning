"""Two real GPU requests, HTTP negatives and normal/stream consistency; no paid API."""
from datetime import datetime, timezone
import json
import time

import httpx

from .client import check_response, client_args, fixture_rows, options
from .common import ROOT, json_bytes


def main():
    a = options(__doc__).parse_args()
    base, headers = client_args(a)
    expected = json.loads((ROOT / "serving/candidate.json").read_text(encoding="utf-8"))
    row = next(r for r in fixture_rows() if r["split"] == "validation")
    with httpx.Client(headers=headers, timeout=150, trust_env=False) as client:
        assert client.get(base.removesuffix("/v1") + "/readyz").status_code == 200
        identity = check_response(client.get(base + "/identity"))
        assert identity["backend"] == "transformers_peft_nf4", "This smoke command requires the real model backend."
        assert identity["actual"]["adapter_sha256"] == expected["adapter_sha256"]
        assert check_response(client.get(base + "/models"))["data"][0]["id"] == expected["name"]
        payload = dict(model=expected["name"], messages=[dict(role="user", content=row["prompt"])],
                       temperature=0, max_tokens=512)
        started = time.perf_counter()
        response = client.post(base + "/chat/completions", json=payload)
        normal = check_response(response)
        normal_seconds = time.perf_counter() - started
        assert response.headers["X-Input-SHA256"] == row["expected_input_sha256"]
        assert normal["usage"]["prompt_tokens"] == row["expected_prompt_tokens"]
        text, usage, finish, done, first_seconds = "", None, None, 0, None
        started = time.perf_counter()
        with client.stream("POST", base + "/chat/completions", json=dict(payload, stream=True,
                           stream_options=dict(include_usage=True))) as streamed:
            streamed.raise_for_status()
            assert streamed.headers["X-Input-SHA256"] == row["expected_input_sha256"]
            for line in streamed.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done += 1
                    continue
                event = json.loads(data)
                assert "error" not in event, event
                if event.get("usage"):
                    usage = event["usage"]
                if event["choices"]:
                    choice = event["choices"][0]
                    content = choice["delta"].get("content", "")
                    if content and first_seconds is None:
                        first_seconds = time.perf_counter() - started
                    text += content
                    finish = choice.get("finish_reason") or finish
        assert done == 1 and finish == normal["choices"][0]["finish_reason"]
        assert usage == normal["usage"] and text == normal["choices"][0]["message"]["content"]
        assert text.strip() and "<think>" not in text and "</think>" not in text
        unauth = client.get(base + "/models", headers={"Authorization": "Bearer invalid"})
        invalid = client.post(base + "/chat/completions", json=dict(payload, temperature=0.3))
        missing = client.post(base + "/chat/completions", json=dict(payload, model="missing"))
        assert (unauth.status_code, invalid.status_code, missing.status_code) == (401, 400, 404)
    result = dict(status="SERVING_HTTP_SMOKE_OK", checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  real_gpu_backend=True, identity=identity, fixture_id=row["id"], input_sha256=row["expected_input_sha256"],
                  normal_stream_equal=True, answer=text, usage=usage, finish_reason=finish,
                  normal_seconds=normal_seconds, stream_first_content_seconds=first_seconds,
                  negative_statuses=[401, 400, 404],
                  semantic_quality_evaluated=False, rag_integration_verified=False)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    output = a.output_dir / ("http-smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    output.write_bytes(json_bytes(result))
    print(json.dumps(dict(status=result["status"], real_gpu_requests=2, output=str(output)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
