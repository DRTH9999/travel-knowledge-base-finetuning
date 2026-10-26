"""Audit prompt-comparison artifacts using trusted local code, not downloaded code."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = ROOT / 'artifacts/prompt_v2_validation_original'
RESULT = RAW / 'eval_outputs/prompt_v2_validation'
BASELINE = RAW / 'eval_outputs/revision_v2_epoch1_512'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evaluation'))
from evaluation.common import json_bytes, read_jsonl, sha
from prompt_pipeline.protocol import VARIANTS, load, preflight
from prompt_pipeline.validate_results import validate


def main():
    checks = validate(ROOT, RESULT, BASELINE)
    assert checks['status'] == 'PROMPT_RESULTS_CHECK_OK' and checks['complete_split'] and checks['samples_per_variant'] == 24
    with zipfile.ZipFile(ROOT / 'sft_prompt_v2_validation.zip') as bundle:
        checksum = bundle.read('prompt_pipeline/checksums.sha256')
        assert checksum == (RAW / 'prompt_pipeline/checksums.sha256').read_bytes()
        shipped = []
        for line in checksum.decode('utf-8').splitlines():
            expected, name = line.split('  ', 1)
            assert sha((RAW / name).read_bytes()) == sha(bundle.read(name)) == expected, name
            shipped.append(name)
    spec = load(ROOT / 'prompt_pipeline/experiment_manifest.json')
    assert (ROOT / 'prompt_pipeline/experiment_manifest.json').read_bytes() == (RAW / 'prompt_pipeline/experiment_manifest.json').read_bytes()
    for name, expected in spec['project_files_sha256'].items():
        assert sha((RAW / name).read_bytes()) == expected, name
    previous = ROOT / 'artifacts/revision_v2_epoch1_original'
    for name in ('report.json', 'adapter.predictions.jsonl'):
        assert (BASELINE / name).read_bytes() == (previous / 'eval_outputs/revision_v2_epoch1_512' / name).read_bytes()
    for name in ('epoch_report.json', 'adapter/adapter_config.json'):
        assert (RAW / 'outputs/revision_v2_epoch1' / name).read_bytes() == (previous / 'outputs/revision_v2_epoch1' / name).read_bytes()
    _, templates, examples, _, _, tokenizer, prepared, token_checks = preflight(
        ROOT, ROOT / '.cache/qwen3-8b-tokenizer', previous / 'outputs/revision_v2_epoch1/adapter', BASELINE)
    report = load(RESULT / 'report.json')
    assert report['generation'] == spec['generation'] and report['adapter_sha256'] == spec['adapter_sha256']
    input_map = {(p['id'], p['variant']): p for p in read_jsonl(RESULT / 'inputs.jsonl')}
    predictions = {v: read_jsonl(RESULT / f'{v}.predictions.jsonl') for v in VARIANTS}
    for variant in VARIANTS:
        for row, expected, actual in zip(examples, prepared[variant], predictions[variant]):
            stored = input_map[(row['id'], variant)]
            assert stored['ids'] == expected['ids'] and stored['input_sha256'] == expected['input_sha256']
            assert actual['answer'] == tokenizer.decode(actual['output_token_ids'], skip_special_tokens=True).strip()
            assert actual['input_sha256'] == expected['input_sha256'] and actual['payload_sha256'] == expected['payload_sha256']
            assert actual['answer'] in (RESULT / 'comparison.md').read_text(encoding='utf-8')
    same, changed = [], []
    for a, b in zip(*(predictions[v] for v in VARIANTS)):
        (same if a['answer'] == b['answer'] else changed).append(a['id'])
    assert len(same) == 4 and len(changed) == report['changed_answers'] == 20
    for name, expected in report['code_sha256'].items():
        assert sha((RAW / name).read_bytes()) == expected == sha((ROOT / name).read_bytes()), name
    manual = load(RESULT / 'manual_reviews.json')['reviews']
    assert len(manual) == 48 and all(r['status'] == 'pending' and all(v is None for v in r['scores'].values()) for r in manual)
    verification = dict(status='PROMPT_V2_ARTIFACT_VERIFICATION_OK', checked_at_utc=datetime.now(timezone.utc).isoformat(),
        archive_sha256=sha((ROOT / 'qwen3_prompt_v2_validation_results.tar.gz').read_bytes()),
        source_archive_contains_adapter_weights=False,
        adapter_sha256=spec['adapter_sha256'], existing_local_adapter_bytes_rehashed=True,
        delivered_files_byte_verified=len(shipped), runtime_source_matches_delivery_and_reports=True,
        source_questions_evidence_history_unchanged=24, original_control_reproduced=24,
        input_tokens_recomputed_with_real_tokenizer=48, output_token_ids_decoded_and_answers_verified=48,
        token_preflight=token_checks, result_integrity=checks, identical_answer_ids=same, changed_answer_ids=changed,
        complete_split=True, original_manual_reviews_pending=48, training_performed=False, test_predictions_generated=False,
        base_model_identity_scope='Reported path/config/template/package identity, not full base-weight rehashing or independent 8B rerun.',
        semantic_review_separate=True)
    (OUT / 'artifact_verification.json').write_bytes(json_bytes(verification))
    print(json.dumps(dict(status=verification['status'], original_control_reproduced=24,
        real_token_inputs_verified=48, decoded_outputs_verified=48, same_answers=4, changed_answers=20), ensure_ascii=False))


if __name__ == '__main__':
    main()
