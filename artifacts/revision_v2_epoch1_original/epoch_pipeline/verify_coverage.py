"""Verify the completed one-epoch training traces without loading a model."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epoch_pipeline.coverage import audit_coverage
from epoch_pipeline.preflight import preflight
from revision_pipeline.common import require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/revision_v2_epoch1')
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'data/revision_v2')
    args = parser.parse_args()
    _, raw, tokens, checks = preflight(ROOT, args.dataset_dir)
    actual = audit_coverage(args.output_dir, raw, tokens, checks['plan'])
    saved = json.loads((args.output_dir / 'coverage_report.json').read_text(encoding='utf-8'))
    require(all(saved[k] == v for k, v in actual.items()), 'Saved coverage report does not match the actual trace.')
    state = json.loads((args.output_dir / 'trainer_state.json').read_text(encoding='utf-8'))
    report = json.loads((args.output_dir / 'epoch_report.json').read_text(encoding='utf-8'))
    require(report['status'] == 'REVISION_EPOCH_TRAIN_OK', 'Epoch training including save/reload must complete first.')
    require(state['global_step'] == report['training']['optimizer_steps'] == 21 and abs(state['epoch'] - 1.0) < 1e-8, 'Unexpected final epoch/step.')
    require(report['training']['coverage'] == saved, 'Training and coverage reports differ.')
    require(report['reload']['saved_and_reloaded_weights_equal'] is True, 'Saved/reloaded equality was not confirmed.')
    require(sha((args.output_dir / 'adapter/adapter_model.safetensors').read_bytes()) == report['reload']['adapter_sha256'], 'Saved adapter bytes differ from the trained adapter.')
    print(json.dumps(dict(status='EPOCH_COVERAGE_OK', unique_trained_samples=actual['unique_trained_samples'],
                          parent_samples_trained=actual['parent_samples_trained'], supplement_samples_trained=actual['supplement_samples_trained'],
                          optimizer_steps=state['global_step'], epoch=state['epoch'],
                          last_update_micro_batches=actual['last_update']['micro_batches']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
