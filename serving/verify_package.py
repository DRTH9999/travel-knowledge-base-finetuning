"""Verify only files listed in our bundle; runtime files and secrets are excluded."""
import json
from pathlib import Path

from .common import ROOT, file_sha


def main():
    manifest = json.loads((ROOT / "serving/bundle_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files_sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or file_sha(path) != digest:
            raise RuntimeError(f"Package file missing or changed: {name}")
    print("SERVING_PACKAGE_OK", len(manifest["files_sha256"]))


if __name__ == "__main__":
    main()
