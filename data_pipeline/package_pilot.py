"""Package only checked data and its reproducible scripts, never model weights."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data/pilot_v1"
    quality = json.loads((data / "quality_report.json").read_text(encoding="utf-8"))
    assert quality["status"] == "DATA_CHECK_OK"
    assert quality["tokenizer_status"] == "local_render_and_label_checks_passed_cloud_recheck_pending"
    assert quality["sample_count"] == 50
    required = ("train.sft.jsonl", "train.tokenized.jsonl", "review_samples.md", "checksums.sha256")
    assert all((data / name).exists() for name in required)
    output = root / "sft_pilot_v1.zip"
    files = [p for p in data.rglob("*") if p.is_file()]
    files += [p for p in (root / "data_pipeline").iterdir() if p.is_file() and p.suffix in (".py", ".md", ".ps1")]
    assert not any(p.suffix in (".safetensors", ".bin", ".pth") for p in files)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, "sft_pilot_v1/" + p.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(files)
    print(json.dumps(dict(path=str(output), byte_count=output.stat().st_size, files=len(files),
                          sha256=hashlib.sha256(output.read_bytes()).hexdigest()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
