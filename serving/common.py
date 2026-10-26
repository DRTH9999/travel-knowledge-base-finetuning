from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def json_bytes(value):
    # Identical to the frozen evaluator's input-token hash convention.
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ServiceError(Exception):
    def __init__(self, message, status=500, code="generation_error"):
        super().__init__(message)
        self.status, self.code = status, code


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path("/root/autodl-tmp/models/Qwen3-8B")
    adapter_dir: Path = Path("/root/autodl-tmp/travel-finetune/sft_pilot_v1/outputs/revision_v2_epoch1/adapter")
    candidate_path: Path = ROOT / "serving/candidate.json"
    runtime_dir: Path = ROOT / "serving_runtime"
    model_name: str = "qwen3-travel-epoch1"
    api_key: str = ""
    max_total_tokens: int = 2048
    request_timeout: float = 120.0
    cancel_grace: float = 5.0

    @classmethod
    def from_env(cls):
        runtime = Path(os.getenv("TRAVEL_RUNTIME_DIR", str(ROOT / "serving_runtime"))).resolve()
        key = os.getenv("TRAVEL_API_KEY", "").strip()
        if not key:
            key_path = Path(os.getenv("TRAVEL_API_KEY_FILE", str(runtime / "api.key")))
            if key_path.is_file():
                key = key_path.read_text(encoding="utf-8").strip()
        if not key or key.startswith("CHANGE_ME"):
            raise RuntimeError("Set TRAVEL_API_KEY or create the private API key with setup_cloud.sh.")
        total = int(os.getenv("TRAVEL_MAX_TOTAL_TOKENS", "2048"))
        timeout = float(os.getenv("TRAVEL_REQUEST_TIMEOUT", "120"))
        if not 513 <= total <= 2048:
            raise RuntimeError("This initial compatibility service only accepts a total budget of 513-2048 tokens.")
        if not 1 <= timeout <= 600:
            raise RuntimeError("TRAVEL_REQUEST_TIMEOUT must be between 1 and 600 seconds.")
        return cls(model_dir=Path(os.getenv("TRAVEL_MODEL_DIR", str(cls.model_dir))).resolve(),
                   adapter_dir=Path(os.getenv("TRAVEL_ADAPTER_DIR", str(cls.adapter_dir))).resolve(),
                   runtime_dir=runtime, api_key=key, max_total_tokens=total, request_timeout=timeout)


def check_candidate(settings, runtime=True):
    candidate = json.loads(settings.candidate_path.read_text(encoding="utf-8"))
    paths = {"model_config_sha256": settings.model_dir / "config.json",
             "adapter_sha256": settings.adapter_dir / "adapter_model.safetensors",
             "adapter_config_sha256": settings.adapter_dir / "adapter_config.json"}
    actual = {}
    for key, path in paths.items():
        if not path.is_file() or file_sha(path) != candidate[key]:
            raise RuntimeError(f"Candidate identity mismatch: {key} ({path}).")
        actual[key] = candidate[key]
    for name, digest in candidate["tokenizer_assets_sha256"].items():
        path = settings.model_dir / name
        if not path.is_file() or file_sha(path) != digest:
            raise RuntimeError(f"Tokenizer asset mismatch: {name}.")
    versions = {}
    if runtime:
        for package, expected in candidate["runtime_versions"].items():
            versions[package] = importlib.metadata.version(package)
            if versions[package] != expected:
                raise RuntimeError(f"Runtime mismatch: {package}: {versions[package]} != {expected}. Keep the verified runtime.")
    return dict(candidate=candidate, actual=actual, versions=versions,
                base_weights_fully_hashed=False, scope="config/tokenizer and adapter identity; not a full base-weight audit")
