"""Create an allowlisted upload zip; no keys, weights, environments or raw archives."""
import argparse
from datetime import datetime, timezone
import json
import zipfile

from .common import ROOT, file_sha, json_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="qwen3_model_service_a_v1.zip")
    a = p.parse_args()
    source = ROOT / "serving"
    allowed_suffixes = {".py", ".json", ".jsonl", ".txt", ".md", ".sh"}
    files = [f for f in source.iterdir() if f.is_file() and f.suffix in allowed_suffixes and f.name != "bundle_manifest.json"]
    required = {"README.md", "candidate.json", "compatibility.jsonl", "local_validation.json", "setup_cloud.sh", "run_cloud.sh"}
    assert required <= {f.name for f in files}
    assert not any("key" in f.name or "secret" in f.name for f in files)
    manifest = dict(package="qwen3_model_service_a_v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
                    files_sha256={f.relative_to(ROOT).as_posix(): file_sha(f) for f in sorted(files)})
    (source / "bundle_manifest.json").write_bytes(json_bytes(manifest))
    files.append(source / "bundle_manifest.json")
    target = ROOT / a.output
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, "serving_v1/" + f.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None and len(z.namelist()) == len(files)
        for f in files:
            assert z.read("serving_v1/" + f.relative_to(ROOT).as_posix()) == f.read_bytes()
    print(json.dumps(dict(path=str(target), byte_count=target.stat().st_size, files=len(files),
                          sha256=file_sha(target)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
