"""Read-only candidate/tokenizer/runtime checks; never loads the 8B weights."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .common import ROOT, Settings, check_candidate, json_bytes
from .engine import TextEncoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=Path(os.getenv("TRAVEL_MODEL_DIR", str(Settings.model_dir))))
    p.add_argument("--adapter-dir", type=Path, default=Path(os.getenv("TRAVEL_ADAPTER_DIR", str(Settings.adapter_dir))))
    p.add_argument("--tokenizer-only", action="store_true", help="Local rendering check only; not a GPU/runtime pass.")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    settings = Settings(model_dir=a.model_dir.resolve(), adapter_dir=a.adapter_dir.resolve())
    identity = check_candidate(settings, runtime=not a.tokenizer_only)
    from transformers import AutoTokenizer
    encoder = TextEncoder(AutoTokenizer.from_pretrained(str(a.model_dir), local_files_only=True, trust_remote_code=False),
                          identity["candidate"], settings.max_total_tokens)
    rows = [json.loads(line) for line in (ROOT / "serving/compatibility.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        prepared = encoder.prepare(row["prompt"], 512)
        assert prepared.input_sha256 == row["expected_input_sha256"], row["id"]
        assert len(prepared.ids) == row["expected_prompt_tokens"], row["id"]
    result = dict(status="TOKENIZER_ONLY_OK" if a.tokenizer_only else "SERVING_PREFLIGHT_OK",
                  checked_at_utc=datetime.now(timezone.utc).isoformat(), identity=identity,
                  real_tokenizer_inputs_verified=len(rows), gpu_checked=False, model_weights_loaded=False,
                  max_prompt_tokens=max(r["expected_prompt_tokens"] for r in rows), max_total_tokens=2048)
    if not a.tokenizer_only:
        index = json.loads((a.model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        files = sorted(set(index["weight_map"].values()))
        for name in files:
            target = (a.model_dir / name).resolve()
            if not target.is_relative_to(a.model_dir.resolve()) or not target.is_file() or target.stat().st_size == 0:
                raise RuntimeError(f"Missing or invalid model shard: {name}")
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA/BF16 unavailable.")
        probe = torch.ones((8, 8), device="cuda:0", dtype=torch.bfloat16)
        assert torch.isfinite(probe @ probe).all().item()
        del probe
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        result.update(gpu_checked=True, model_shards_present=len(files), gpu=torch.cuda.get_device_name(0),
                      free_gpu_gib=round(free / 1024**3, 3), total_gpu_gib=round(total / 1024**3, 3))
    output = a.output or Path(os.getenv("TRAVEL_RUNTIME_DIR", str(ROOT / "serving_runtime"))) / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json_bytes(result))
    print(json.dumps(dict(status=result["status"], verified_inputs=len(rows), gpu_checked=result["gpu_checked"],
                          model_weights_loaded=False, output=str(output)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
