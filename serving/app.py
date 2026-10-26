from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import logging
import os
from queue import Empty
import time
from typing import Literal

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import ServiceError, Settings, json_bytes
from .engine import TransformersEngine
from .runtime import Runtime


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user"]
    content: str = Field(min_length=1, max_length=100_000, strict=True)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_usage: bool = Field(default=False, strict=True)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[Message] = Field(min_length=1, max_length=1)
    stream: bool = Field(default=False, strict=True)
    max_tokens: int | None = Field(default=None, ge=1, le=512, strict=True)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=512, strict=True)
    temperature: float | None = Field(default=0.0, ge=0, le=0, strict=True)
    top_p: float | None = Field(default=None, ge=1, le=1, strict=True)
    n: int = Field(default=1, ge=1, le=1, strict=True)

    stream_options: StreamOptions | None = None

    @model_validator(mode="after")
    def compatible(self):
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("Specify only one of max_tokens/max_completion_tokens.")
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true.")
        return self

    @property
    def output_budget(self):
        return self.max_tokens or self.max_completion_tokens or 512


def error_body(exc, request_id=None):
    value = dict(message=str(exc), type="server_error" if exc.status >= 500 else "invalid_request_error",
                 param=None, code=exc.code)
    if request_id:
        value["request_id"] = request_id
    return dict(error=value)


def create_app(settings=None, engine_factory=None):
    settings = settings or Settings.from_env()
    if not settings.api_key:
        raise RuntimeError("An API key is required.")
    engine_factory = engine_factory or TransformersEngine

    @asynccontextmanager
    async def lifespan(app):
        engine = await asyncio.to_thread(engine_factory, settings)
        runtime = Runtime(settings, engine)
        app.state.runtime = runtime
        settings.runtime_dir.mkdir(parents=True, exist_ok=True)
        (settings.runtime_dir / "server_identity.json").write_bytes(json_bytes(
            dict(engine.identity, model_name=settings.model_name, pid=os.getpid())))
        try:
            yield
        finally:
            await asyncio.to_thread(runtime.shutdown)

    app = FastAPI(title="Travel SFT single-GPU service", version="1.0.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None)

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        headers = {"Retry-After": "2"} if exc.status == 429 else None
        return JSONResponse(error_body(exc), status_code=exc.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Never echo the prompt or a rejected credential in errors.
        fields = [".".join(map(str, e["loc"])) for e in exc.errors()]
        return JSONResponse(error_body(ServiceError("Unsupported or invalid request fields: " + ", ".join(fields),
                                                     400, "invalid_request")), status_code=400)

    async def authorize(request: Request):
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied.encode(), ("Bearer " + settings.api_key).encode()):
            raise ServiceError("A valid Bearer API key is required.", 401, "invalid_api_key")

    @app.get("/healthz")
    async def health():
        return dict(status="alive")

    @app.get("/readyz")
    async def ready():
        runtime = app.state.runtime
        ok = runtime.healthy()
        return JSONResponse(dict(status="ready" if ok else "unavailable", busy=runtime.active is not None),
                            status_code=200 if ok else 503)

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models():
        return dict(object="list", data=[dict(id=settings.model_name, object="model", created=0, owned_by="local")])

    @app.get("/v1/identity", dependencies=[Depends(authorize)])
    async def identity():
        return dict(app.state.runtime.engine.identity, model_name=settings.model_name)

    async def monitor(request, job):
        if await request.is_disconnected():
            job.cancel.set()
            raise ServiceError("Client disconnected.", 499, "client_disconnected")
        if time.monotonic() >= job.deadline:
            job.cancel.set()
            raise ServiceError("Generation timed out; slot stays held until the worker exits.", 504, "generation_timeout")

    def usage(job):
        return dict(prompt_tokens=len(job.prepared.ids), completion_tokens=job.result["completion_tokens"],
                    total_tokens=len(job.prepared.ids) + job.result["completion_tokens"])

    def chunk(job, delta, reason=None):
        return dict(id=job.request_id, object="chat.completion.chunk", created=job.created, model=settings.model_name,
                    choices=[dict(index=0, delta=delta, finish_reason=reason)])

    def sse(value):
        return "data: " + json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n\n"

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    async def chat(body: ChatRequest, request: Request):
        if body.model != settings.model_name:
            raise ServiceError("Unknown model name; use /v1/models.", 404, "model_not_found")
        runtime = app.state.runtime
        job = runtime.start(body.messages[0].content, body.output_budget)
        handed_to_stream = False
        try:
            # Token-budget errors happen before response headers or GPU generation.
            while not job.ready.is_set() and not job.done.is_set():
                await monitor(request, job)
                await asyncio.sleep(0.01)
            if job.error:
                raise job.error
            headers = {"X-Request-ID": job.request_id, "X-Input-SHA256": job.prepared.input_sha256,
                       "X-Prompt-Tokens": str(len(job.prepared.ids)),
                       "X-Adapter-SHA256": runtime.engine.identity["actual"]["adapter_sha256"]}
            if not body.stream:
                while not job.done.is_set():
                    await monitor(request, job)
                    await asyncio.sleep(0.01)
                if job.error:
                    raise job.error
                return JSONResponse(dict(id=job.request_id, object="chat.completion", created=job.created,
                    model=settings.model_name, choices=[dict(index=0, message=dict(role="assistant", content=job.result["answer"]),
                    finish_reason=job.result["finish_reason"])], usage=usage(job)), headers=headers)

            async def events():
                try:
                    yield sse(chunk(job, dict(role="assistant", content="")))
                    while True:
                        await monitor(request, job)
                        try:
                            text = job.chunks.get_nowait()
                        except Empty:
                            if job.done.is_set():
                                break
                            await asyncio.sleep(0.01)
                        else:
                            yield sse(chunk(job, dict(content=text)))
                            await asyncio.sleep(0)
                    if job.error:
                        raise job.error
                    yield sse(chunk(job, {}, job.result["finish_reason"]))
                    if body.stream_options and body.stream_options.include_usage:
                        yield sse(dict(id=job.request_id, object="chat.completion.chunk", created=job.created,
                                       model=settings.model_name, choices=[], usage=usage(job)))
                    yield "data: [DONE]\n\n"
                except ServiceError as exc:
                    # Headers are already sent. Emit a protocol error, never a false normal finish.
                    yield sse(error_body(exc, job.request_id))
                    yield "data: [DONE]\n\n"
                finally:
                    if not job.done.is_set():
                        job.cancel.set()

            headers.update({"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

            class JobStreamingResponse(StreamingResponse):
                async def __call__(self, scope, receive, send):
                    try:
                        await super().__call__(scope, receive, send)
                    finally:
                        # Also covers disconnect before the first iterator step.
                        if not job.done.is_set():
                            job.cancel.set()

            handed_to_stream = True
            return JobStreamingResponse(events(), media_type="text/event-stream", headers=headers)
        finally:
            if not handed_to_stream and not job.done.is_set():
                job.cancel.set()

    return app


def main():
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    host = os.getenv("TRAVEL_HOST", "127.0.0.1")
    port = int(os.getenv("TRAVEL_PORT", "8008"))
    uvicorn.run(create_app(settings), host=host, port=port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
