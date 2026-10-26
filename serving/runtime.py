from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from queue import Queue
import threading
import time
import uuid

from .common import ServiceError

logger = logging.getLogger("travel_serving")


@dataclass
class Job:
    request_id: str = field(default_factory=lambda: "chatcmpl-" + uuid.uuid4().hex)
    created: int = field(default_factory=lambda: int(time.time()))
    started: float = field(default_factory=time.monotonic)
    cancel: threading.Event = field(default_factory=threading.Event)
    ready: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    chunks: Queue = field(default_factory=Queue)
    prepared: object = None
    result: dict | None = None
    error: ServiceError | None = None
    first_chunk_seconds: float | None = None
    thread: threading.Thread | None = None
    deadline: float = 0


class Runtime:
    def __init__(self, settings, engine):
        self.settings, self.engine = settings, engine
        self.lock = threading.Lock()
        self.active = None
        self.stopping = False

    def healthy(self):
        job = self.active
        return (not self.stopping and not self.engine.failed and
                not (job and time.monotonic() > job.deadline + self.settings.cancel_grace))

    def start(self, prompt, max_new_tokens):
        with self.lock:
            if not self.healthy():
                raise ServiceError("Model unavailable; check readiness and service logs.", 503, "model_not_ready")
            if self.active is not None:
                raise ServiceError("One generation is already active; retry after it completes.", 429, "model_busy")
            job = Job()
            job.deadline = job.started + self.settings.request_timeout
            self.active = job
            job.thread = threading.Thread(target=self._run, args=(job, prompt, max_new_tokens), daemon=True)
            try:
                job.thread.start()
            except BaseException:
                self.active = None
                raise
            return job

    def _run(self, job, prompt, max_new_tokens):
        try:
            job.prepared = self.engine.prepare(prompt, max_new_tokens)
            if job.cancel.is_set() or time.monotonic() >= job.deadline:
                raise ServiceError("Request expired before generation.", 504, "generation_timeout")
            job.ready.set()

            def emit(text):
                if not job.cancel.is_set():
                    if job.first_chunk_seconds is None:
                        job.first_chunk_seconds = time.monotonic() - job.started
                    job.chunks.put(text)

            job.result = self.engine.generate(job.prepared, max_new_tokens, job.cancel, job.deadline, emit)
            if job.cancel.is_set() or time.monotonic() >= job.deadline:
                job.result = None
                raise ServiceError("Generation cancelled or timed out.", 504, "generation_timeout")
        except ServiceError as exc:
            job.error = exc
        except Exception:
            logger.exception("Generation failed: %s", job.request_id)
            job.error = ServiceError("Generation failed; see server logs using the request ID.")
        finally:
            event = dict(request_id=job.request_id, input_sha256=getattr(job.prepared, "input_sha256", None),
                         prompt_tokens=len(job.prepared.ids) if job.prepared else None,
                         elapsed_seconds=round(time.monotonic() - job.started, 4),
                         first_chunk_seconds=job.first_chunk_seconds,
                         cancelled=job.cancel.is_set(), error_code=job.error.code if job.error else None)
            if job.result:
                event.update({k: job.result[k] for k in ("completion_tokens", "finish_reason", "generation_seconds",
                                                          "peak_allocated_vram_gib") if k in job.result})
            try:
                self.settings.runtime_dir.mkdir(parents=True, exist_ok=True)
                with (self.settings.runtime_dir / "requests.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            except OSError:
                logger.exception("Could not save request metrics: %s", job.request_id)
            # Never release the GPU slot from the HTTP task: only the worker can release it.
            with self.lock:
                self.active = None
                job.done.set()

    def shutdown(self):
        self.stopping = True
        job = self.active
        if job:
            job.cancel.set()
            job.thread.join(timeout=self.settings.cancel_grace)
