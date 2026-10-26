"""Validate versioned revision data without assuming 50 training rows."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision_pipeline.common import ROOT, json_bytes, validate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/revision_v2")
    parser.add_argument("--require-tokenized", action="store_true")
    args = parser.parse_args()
    _, _, report = validate_dataset(args.dataset_dir, require_tokenized=args.require_tokenized)
    (args.dataset_dir / "quality_report.json").write_bytes(json_bytes(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
