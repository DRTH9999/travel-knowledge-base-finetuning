"""CPU protocol/lifecycle tests. The fake engine cannot be selected by production CLI."""
from dataclasses import replace
import json
import socket
import threading
import time

from fastapi.testclient import TestClient
import pytest

from serving.app import create_app
from serving.common import ROOT, ServiceError, Settings, json_bytes, sha
from serving.engine import Prepared, TextEncoder
from serving.runtime import Runtime


class FakeEngine:
    def __init__(self, settings):
        self.failed = False
        self.identity = dict(backend="cpu_test_fake", actual=dict(adapter_sha256="test-only"))
        self.release = threading.Event()
        self.started = threading.Event()
        self.last_prompt = ""

    def prepare(self, prompt, max_new_tokens):
        self.last_prompt = prompt
        if prompt == "OVERFLOW":
            raise ServiceError("No truncation.", 400, "context_length_exceeded")
        ids = list(prompt.encode())
        return Prepared(ids, sha(json_bytes(ids)))

    def generate(self, prepared, max_new_tokens, cancel, deadline, emit):
        self.started.set()
        if self.last_prompt in ("BLOCK", "WAIT"):
            while not self.release.is_set():
                if self.last_prompt == "WAIT" and (cancel.is_set() or time.monotonic() >= deadline):
                    raise ServiceError("Timed out.", 504, "generation_timeout")
                time.sleep(0.005)
        emit("依据资料")
        time.sleep(0.02)
        if self.last_prompt == "ERROR":
            raise RuntimeError("Simulated generation error.")
        emit("回答。[1]")
        return dict(answer="依据资料回答。[1]", completion_tokens=7, output_token_ids=[1, 2, 3, 4, 5, 6, 7],
                    finish_reason="stop", generation_seconds=0.02, peak_allocated_vram_gib=0)


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key="test-private-key", runtime_dir=tmp_path, request_timeout=2, cancel_grace=0.02)


@pytest.fixture
def service(settings):
    app = create_app(settings, FakeEngine)
    with TestClient(app, headers={"Authorization": "Bearer " + settings.api_key}) as client:
        yield app, client


def payload(**kwargs):
    return dict(model="qwen3-travel-epoch1", messages=[dict(role="user", content="问题")], **kwargs)


def wait_done(job):
    assert job.done.wait(2)


def test_auth_and_identity(service):
    _, client = service
    assert client.get("/healthz").json() == dict(status="alive")
    assert client.get("/readyz").status_code == 200
    assert client.get("/v1/models", headers={"Authorization": "bad"}).status_code == 401
    assert client.get("/v1/models").json()["data"][0]["id"] == "qwen3-travel-epoch1"
    assert client.get("/v1/identity").json()["backend"] == "cpu_test_fake"
    assert client.post("/v1/chat/completions", json=dict(payload(), model="wrong")).status_code == 404


def test_normal_and_stream_protocol(service):
    _, client = service
    response = client.post("/v1/chat/completions", json=payload(max_tokens=32))
    assert response.status_code == 200
    normal = response.json()
    assert normal["choices"][0]["message"]["content"] == "依据资料回答。[1]"
    assert normal["choices"][0]["finish_reason"] == "stop"
    assert int(response.headers["X-Prompt-Tokens"]) == len("问题".encode())
    streamed = client.post("/v1/chat/completions", json=payload(stream=True, max_completion_tokens=32,
                            stream_options=dict(include_usage=True)))
    assert streamed.status_code == 200
    data = [line[6:] for line in streamed.text.splitlines() if line.startswith("data: ")]
    assert data[-1] == "[DONE]" and data.count("[DONE]") == 1
    chunks = [json.loads(x) for x in data[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == normal["choices"][0]["message"]["content"]
    assert chunks[-1]["usage"] == normal["usage"]
    assert streamed.headers["X-Input-SHA256"] == response.headers["X-Input-SHA256"]


@pytest.mark.parametrize("change", [dict(temperature=0.3), dict(top_p=0.9), dict(max_tokens=513),
    dict(max_tokens=True), dict(max_tokens=0), dict(max_tokens=4, max_completion_tokens=4),
    dict(n=2), dict(stop=["END"]), dict(tools=[]), dict(stream="true"),
    dict(messages=[dict(role="system", content="x")]),
    dict(messages=[dict(role="user", content="x"), dict(role="user", content="y")]),
    dict(stream_options=dict(include_usage=True))])
def test_rejects_unsupported_generation_without_starting(service, change):
    app, client = service
    result = client.post("/v1/chat/completions", json=dict(payload(), **change))
    assert result.status_code == 400
    assert app.state.runtime.active is None and not app.state.runtime.engine.started.is_set()


def test_context_overflow_does_not_poison_next_request(service):
    _, client = service
    response = client.post("/v1/chat/completions", json=dict(payload(), messages=[dict(role="user", content="OVERFLOW")]))
    assert response.status_code == 400 and response.json()["error"]["code"] == "context_length_exceeded"
    assert client.post("/v1/chat/completions", json=payload()).status_code == 200


def test_stream_error_has_no_success_finish_and_next_request_runs(service):
    _, client = service
    response = client.post("/v1/chat/completions", json=dict(payload(stream=True), messages=[dict(role="user", content="ERROR")]))
    events = [json.loads(x[6:]) for x in response.text.splitlines() if x.startswith("data: ") and x != "data: [DONE]"]
    assert any("error" in e for e in events)
    assert not any(c.get("finish_reason") in ("stop", "length") for e in events for c in e.get("choices", []))
    assert client.post("/v1/chat/completions", json=payload()).status_code == 200


def test_busy_and_cancel_do_not_release_live_worker(service):
    app, client = service
    runtime = app.state.runtime
    job = runtime.start("BLOCK", 32)
    assert runtime.engine.started.wait(1)
    result = client.post("/v1/chat/completions", json=payload())
    assert result.status_code == 429 and result.headers["Retry-After"] == "2"
    job.cancel.set()
    assert runtime.active is job and not job.done.is_set()
    assert client.post("/v1/chat/completions", json=payload()).status_code == 429
    runtime.engine.release.set()
    wait_done(job)
    assert job.error.code == "generation_timeout"
    assert client.post("/v1/chat/completions", json=payload()).status_code == 200


def test_timed_out_stuck_worker_not_ready_until_it_exits(settings):
    runtime = Runtime(replace(settings, request_timeout=0.03, cancel_grace=0.01), FakeEngine(settings))
    job = runtime.start("BLOCK", 32)
    assert runtime.engine.started.wait(1)
    time.sleep(0.06)
    assert not runtime.healthy()
    with pytest.raises(ServiceError) as exc:
        runtime.start("next", 32)
    assert exc.value.status == 503
    runtime.engine.release.set()
    wait_done(job)
    assert runtime.healthy()


def test_http_timeout_recovers(settings):
    app = create_app(replace(settings, request_timeout=0.06), FakeEngine)
    with TestClient(app, headers={"Authorization": "Bearer " + settings.api_key}) as client:
        response = client.post("/v1/chat/completions", json=dict(payload(), messages=[dict(role="user", content="WAIT")]))
        assert response.status_code == 504
        job = app.state.runtime.active
        if job:
            wait_done(job)
        assert client.post("/v1/chat/completions", json=payload()).status_code == 200


def test_real_tokenizer_48_frozen_inputs_and_long_input_rejection():
    from transformers import AutoTokenizer
    path = ROOT / ".cache/qwen3-8b-tokenizer"
    assert path.is_dir(), "This local verification requires the real cached Qwen3 tokenizer."
    spec = json.loads((ROOT / "serving/candidate.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    encoder = TextEncoder(tokenizer, spec, 2048)
    rows = [json.loads(x) for x in (ROOT / "serving/compatibility.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 48
    for row in rows:
        encoded = encoder.prepare(row["prompt"], 512)
        assert encoded.input_sha256 == row["expected_input_sha256"]
        assert len(encoded.ids) == row["expected_prompt_tokens"]
    with pytest.raises(ServiceError) as exc:
        encoder.prepare("长证据" * 4000 + "最后问题必须保留", 512)
    assert exc.value.code == "context_length_exceeded"


@pytest.fixture
def tcp_service(settings):
    import uvicorn
    app = create_app(settings, FakeEngine)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False, lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs=dict(sockets=[sock]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started
    try:
        yield app, f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        assert not thread.is_alive()


def test_actual_langchain_client_over_tcp(tcp_service, settings):
    from langchain.chat_models import init_chat_model
    _, url = tcp_service
    llm = init_chat_model(model="qwen3-travel-epoch1", model_provider="openai", base_url=url,
                          api_key=settings.api_key, temperature=0, max_tokens=32, max_retries=0)
    message = [dict(role="user", content="问题")]
    assert llm.invoke(message).content == "依据资料回答。[1]"
    parts = list(llm.stream(message))
    assert "".join(part.content for part in parts) == "依据资料回答。[1]"


def test_real_disconnect_cancels_worker_and_slot_recovers(tcp_service, settings):
    import httpx
    app, url = tcp_service
    with httpx.Client(headers={"Authorization": "Bearer " + settings.api_key}, timeout=2, trust_env=False) as client:
        with client.stream("POST", url + "/chat/completions", json=dict(payload(stream=True), messages=[dict(role="user", content="WAIT")])) as response:
            assert response.status_code == 200
            job = app.state.runtime.active
            assert next(response.iter_lines()).startswith("data: ")
        wait_done(job)
        assert job.cancel.is_set() and job.error.code == "generation_timeout"
        assert client.post(url + "/chat/completions", json=payload()).status_code == 200
