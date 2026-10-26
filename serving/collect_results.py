"""Collect service evidence only. Never collect the API key or Python environment."""
import argparse
import json
import os
import tarfile
from pathlib import Path

from .common import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/qwen3_service_a_v1_results.tar.gz"))
    a = p.parse_args()
    runtime = Path(os.getenv("TRAVEL_RUNTIME_DIR", str(ROOT / "serving_runtime"))).resolve()
    required = [runtime / "preflight.json", runtime / "server_identity.json"]
    if not all(f.is_file() for f in required) or not list(runtime.glob("http-smoke-*.json")):
        raise RuntimeError("Complete cloud preflight, server startup and HTTP smoke first.")
    preflight = json.loads(required[0].read_text(encoding="utf-8"))
    identity = json.loads(required[1].read_text(encoding="utf-8"))
    if preflight.get("status") != "SERVING_PREFLIGHT_OK" or not preflight.get("gpu_checked"):
        raise RuntimeError("A real cloud CUDA preflight is required, not a tokenizer-only check.")
    if identity.get("backend") != "transformers_peft_nf4":
        raise RuntimeError("A real model backend is required, not a protocol test engine.")
    files = [f for f in runtime.rglob("*") if f.is_file() and not f.is_symlink() and f.suffix in (".json", ".jsonl")]
    installed = runtime / "installed-requirements.txt"
    if installed.is_file():
        files.append(installed)
    files += [ROOT / "serving/candidate.json", ROOT / "serving/local_validation.json", ROOT / "serving/bundle_manifest.json"]
    with tarfile.open(a.output, "x:gz") as tar:
        for f in sorted(files):
            arc = "serving_runtime/" + f.relative_to(runtime).as_posix() if f.is_relative_to(runtime) else f.relative_to(ROOT).as_posix()
            tar.add(f, arcname=arc, recursive=False)
    print("SERVING_RESULTS_PACKED", a.output, a.output.stat().st_size)


if __name__ == "__main__":
    main()
