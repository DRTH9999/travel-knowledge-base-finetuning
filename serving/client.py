"""Shared HTTP helpers. Keys stay in environment/private files, never reports."""
import argparse
import json
import os
from pathlib import Path

from .common import ROOT


def options(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--base-url", default="http://127.0.0.1:8008/v1")
    runtime = Path(os.getenv("TRAVEL_RUNTIME_DIR", str(ROOT / "serving_runtime")))
    p.add_argument("--api-key-file", type=Path, default=Path(os.getenv("TRAVEL_API_KEY_FILE", str(runtime / "api.key"))))
    p.add_argument("--output-dir", type=Path, default=runtime)
    return p


def client_args(a):
    key = os.getenv("TRAVEL_API_KEY", "").strip()
    if not key:
        key = a.api_key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("Empty API key.")
    return a.base_url.rstrip("/"), {"Authorization": "Bearer " + key}


def fixture_rows():
    return [json.loads(line) for line in (ROOT / "serving/compatibility.jsonl").read_text(encoding="utf-8").splitlines()]


def check_response(response):
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise RuntimeError("Server returned an error: " + body["error"]["code"])
    return body
