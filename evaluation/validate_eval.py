"""Check frozen evaluation provenance, isolation, citations and business inputs."""
import argparse
import json
from pathlib import Path

from common import ROOT, validate_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/eval_v1")
    args = parser.parse_args()
    _, report = validate_frozen(args.dataset_dir)
    print(json.dumps(dict(status="EVAL_DATA_CHECK_OK", splits=report,
                          semantic_entailment_auto_verified=False), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
