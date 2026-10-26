"""Read-only preflight for the controlled epoch experiment."""
import json
from pathlib import Path

from epoch_pipeline.coverage import make_plan
from revision_pipeline.common import require, sha, validate_dataset, validate_tokens


def check_dependencies(root):
    config = json.loads((root / 'epoch_pipeline/dependencies.json').read_text(encoding='utf-8'))
    for name, expected in config['project_files_sha256'].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), 'Dependency path outside project.')
        require(path.is_file() and sha(path.read_bytes()) == expected, f'Existing project dependency changed or missing: {name}')
    return config


def preflight(root, directory):
    deps = check_dependencies(root)
    manifest, raw, checks = validate_dataset(directory, require_tokenized=True)
    token_rows, token_checks = validate_tokens(directory, raw)
    require(sha((directory / 'source_manifest.json').read_bytes()) == deps['dataset_manifest_sha256'], 'Expected the existing frozen 82-row revision_v2 dataset.')
    require(sha((directory / 'train.tokenized.jsonl').read_bytes()) == deps['tokenized_sha256'], 'Frozen training token export changed.')
    require(sha((root / 'data/eval_v1/manifest.json').read_bytes()) == manifest['reserved_eval_manifest_sha256'], 'Frozen evaluation manifest changed.')
    plan = make_plan(raw, manifest['parent_sample_count'])
    require(plan['sample_count'] == 82 and plan['optimizer_steps'] == 21 and plan['last_update_micro_batches'] == 2, 'Unexpected controlled one-epoch plan.')
    return manifest, raw, token_rows, dict(status='EPOCH_TRAIN_PREFLIGHT_OK', plan=plan, dataset=checks, tokens=token_checks)
