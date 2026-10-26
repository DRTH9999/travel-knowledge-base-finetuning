"""Read-only validation of paired outputs; no automatic semantic scoring."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evaluation'))
from evaluate import DIMENSIONS, format_flags
from evaluation.common import json_bytes, read_jsonl, require, sha
from prompt_pipeline.protocol import ANCHOR, VARIANTS, check_project, load, prompt_pair


def validate(root, output, baseline_dir, allow_incomplete=False):
    spec, templates = check_project(root)
    report = load(output / 'report.json')
    require(report['status'] == 'PROMPT_COMPARISON_OK', 'Paired generation must complete first.')
    require(report['split'] == 'validation' and report['test_predictions_generated'] is False and report['training_performed'] is False, 'Unexpected experiment scope.')
    require(report['adapter_sha256'] == spec['adapter_sha256'] and report['generation'] == spec['generation'], 'Model or generation configuration changed.')
    require(report['experiment_manifest_sha256'] == sha((root / 'prompt_pipeline/experiment_manifest.json').read_bytes()), 'Experiment protocol changed.')
    require(report['template_sha256'] == {v: sha(t.encode('utf-8')) for v, t in templates.items()}, 'Reported templates changed.')
    for name, expected in report['code_sha256'].items():
        require(sha((root / name).read_bytes()) == expected, f'Runtime source changed: {name}')
    require(report['complete_split'] or allow_incomplete, 'Incomplete debug results cannot stand in for the full validation comparison.')
    n = report['samples']
    require(1 <= n <= 24 and report['complete_split'] == (n == 24), 'Invalid sample count.')
    examples = read_jsonl(root / 'data/eval_v1/validation.jsonl')[:n]
    inputs = read_jsonl(output / 'inputs.jsonl')
    require(len(inputs) == n * 2 and len({(i['id'], i['variant']) for i in inputs}) == n * 2, 'Unpaired recorded inputs.')
    input_map = {(i['id'], i['variant']): i for i in inputs}
    previous = read_jsonl(baseline_dir / 'adapter.predictions.jsonl')[:n]
    for name, expected in spec['baseline_files_sha256'].items():
        require(sha((baseline_dir / name).read_bytes()) == expected, 'Frozen baseline files changed.')
    preds = {v: read_jsonl(output / f'{v}.predictions.jsonl') for v in VARIANTS}
    require(all(len(v) == n for v in preds.values()), 'Incomplete prediction files.')
    for index, row in enumerate(examples):
        _, payload_hash = prompt_pair(row, templates)
        for variant in VARIANTS:
            p, item = preds[variant][index], input_map[(row['id'], variant)]
            require(p['id'] == row['id'] and p['variant'] == variant, 'Prediction identity differs from its paired question.')
            require(p['input_sha256'] == item['input_sha256'] == sha(json_bytes(item['ids'])), 'Recorded input token hash differs.')
            require(p['prompt_tokens'] == item['prompt_tokens'] == len(item['ids']), 'Prompt length differs.')
            require(p['payload_sha256'] == item['payload_sha256'] == payload_hash, 'Question/evidence/history payload changed.')
            require(p['adapter_sha256'] == spec['adapter_sha256'], 'Per-question adapter identity changed.')
            require(len(p['output_token_ids']) == p['generated_tokens'] and 1 <= p['generated_tokens'] <= spec['generation']['max_new_tokens'], 'Generated token count changed.')
            expected_limit = p['generated_tokens'] >= spec['generation']['max_new_tokens'] and (not p['output_token_ids'] or p['output_token_ids'][-1] != report['tokenizer_eos_token_id'])
            require(p['hit_output_limit'] == expected_limit and p['format_flags'] == format_flags(row, p), 'Automatic format flags differ from actual outputs.')
        a, b = (preds[v][index] for v in VARIANTS)
        require(a['input_sha256'] == previous[index]['input_sha256'] and a['answer'] == previous[index]['answer'], 'Original control was not reproduced.')
        require(a['input_sha256'] != b['input_sha256'], 'Prompt variants did not change the model input.')
    require(report['controls']['original_control_reproduced'] is True and report['controls']['original_control_identical_answers'] == n, 'Original control check failed.')
    manual = load(output / 'manual_reviews.json')['reviews']
    require(len(manual) == n * 2 and len({(r['id'], r['variant']) for r in manual}) == n * 2, 'Unpaired review records.')
    require(all(set(r['scores']) == set(DIMENSIONS) and r['status'] == 'pending' and all(v is None for v in r['scores'].values()) for r in manual), 'Do not claim human or semantic review automatically.')
    for variant in VARIANTS:
        expected = {key: sum(bool(p['format_flags'][key]) for p in preds[variant]) for key in preds[variant][0]['format_flags']}
        require(expected == report['variants'][variant]['automatic_format_flag_sample_counts'], 'Reported format counts differ.')
    return dict(status='PROMPT_RESULTS_CHECK_OK', samples_per_variant=n, paired_payloads_verified=n,
        original_control_reproduced=n, complete_split=n == 24, manual_review_status='pending', semantic_quality_automatically_scored=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'eval_outputs/prompt_v2_validation')
    parser.add_argument('--baseline-dir', type=Path, default=ROOT / 'eval_outputs/revision_v2_epoch1_512')
    parser.add_argument('--allow-incomplete', action='store_true', help='Local/debug integrity checks only; not a complete validation result.')
    args = parser.parse_args()
    print(json_bytes(validate(ROOT, args.output_dir, args.baseline_dir, args.allow_incomplete)).decode('utf-8'))


if __name__ == '__main__':
    main()
