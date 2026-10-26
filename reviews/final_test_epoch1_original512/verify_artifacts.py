"""Verify the frozen final-test run; never execute Python from the result archive."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = ROOT / 'artifacts/final_test_epoch1_original512_original'
RESULT = RAW / 'eval_outputs/final_test_epoch1_original512'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evaluation'))
from evaluate import format_flags
from evaluation.common import json_bytes, read_jsonl, sha, validate_frozen, normalized
from revision_pipeline.common import validate_dataset


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return sha(path.read_bytes())


def main():
    report = load(RESULT / 'report.json')
    proposed = load(ROOT / 'reviews/prompt_v2_validation/candidate_freeze_proposal.json')
    validation = load(ROOT / 'artifacts/revision_v2_epoch1_original/eval_outputs/revision_v2_epoch1_512/report.json')
    assert report['status'] == 'EVAL_GENERATION_OK' and report['split'] == 'test' and report['complete_split'] and report['samples'] == 24
    for key in ('adapter_sha256', 'adapter_config_sha256', 'base_model_config_sha256', 'chat_template_sha256', 'eval_manifest_sha256'):
        if key == 'adapter_config_sha256':
            actual = digest(RAW / 'outputs/revision_v2_epoch1/adapter/adapter_config.json')
        elif key == 'base_model_config_sha256':
            actual = report['model_config_sha256']
        else:
            actual = report[key]
        assert actual == proposed[key], key
    assert report['generation'] == proposed['generation'] == validation['generation']
    for key in ('model_config_sha256', 'model_dir', 'adapter_sha256', 'chat_template_sha256', 'versions', 'eval_manifest_sha256'):
        assert report[key] == validation[key], key
    assert digest(RAW / 'data/eval_v1/prompt_template.txt') == proposed['original_business_template_sha256']
    for name in ('data/eval_v1/manifest.json', 'data/eval_v1/test.jsonl', 'data/eval_v1/prompt_template.txt',
                 'evaluation/common.py', 'evaluation/evaluate.py', 'prompt_pipeline/experiment_manifest.json'):
        assert (RAW / name).read_bytes() == (ROOT / name).read_bytes(), name
    epoch_root = ROOT / 'artifacts/revision_v2_epoch1_original/outputs/revision_v2_epoch1'
    for name in ('epoch_report.json', 'adapter/adapter_config.json'):
        assert (RAW / 'outputs/revision_v2_epoch1' / name).read_bytes() == (epoch_root / name).read_bytes(), name
    assert digest(epoch_root / 'adapter/adapter_model.safetensors') == report['adapter_sha256']
    manifest, checks = validate_frozen(ROOT / 'data/eval_v1')
    assert digest(ROOT / 'data/eval_v1/test.jsonl') == report['eval_file_sha256'] == manifest['files_sha256']['test.jsonl']
    training_manifest, training_rows, _ = validate_dataset(ROOT / 'data/revision_v2', require_tokenized=True)
    train_docs = {d['doc_id'] for d in training_manifest['documents']}
    examples = read_jsonl(ROOT / 'data/eval_v1/test.jsonl')
    used_docs = {c['doc_id'] for r in examples for c in r['contexts']}
    assert len(train_docs) == 20 and len(used_docs) == 5 and not train_docs & used_docs
    overlap = {normalized(r['original_question']) for r in training_rows} & {normalized(r['question']) for r in examples}
    assert not overlap
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / '.cache/qwen3-8b-tokenizer'), local_files_only=True, trust_remote_code=False)
    assert sha(tokenizer.chat_template.encode('utf-8')) == report['chat_template_sha256']
    predictions = {k: read_jsonl(RESULT / f'{k}.predictions.jsonl') for k in ('base', 'adapter')}
    comparison = (RESULT / 'comparison.md').read_text(encoding='utf-8')
    for label, items in predictions.items():
        assert len(items) == 24 and len({p['id'] for p in items}) == 24
        for row, p in zip(examples, items):
            assert p['id'] == row['id'] and p['model'] == label and p['task_type'] == row['task_type']
            rendered = tokenizer.apply_chat_template([dict(role='user', content=row['prompt'])], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            assert p['input_sha256'] == sha(json_bytes(ids)) and p['prompt_tokens'] == len(ids)
            assert len(ids) + 512 <= 2048 and 1 <= p['generated_tokens'] <= 512
            assert p['format_flags'] == format_flags(row, p) and p['answer'] in comparison
        counts = {key: sum(bool(p['format_flags'][key]) for p in items) for key in items[0]['format_flags']}
        assert counts == report['models'][label]['automatic_format_flag_sample_counts']
    assert all(a['input_sha256'] == b['input_sha256'] for a, b in zip(predictions['base'], predictions['adapter']))
    manual = load(RESULT / 'manual_reviews.json')['reviews']
    assert len(manual) == 48 and all(r['status'] == 'pending' and all(v is None for v in r['scores'].values()) for r in manual)
    verification = dict(status='FINAL_TEST_ARTIFACT_VERIFICATION_OK', checked_at_utc=datetime.now(timezone.utc).isoformat(),
        archive_sha256=digest(ROOT / 'qwen3_final_test_epoch1_original512_results.tar.gz'), split='test', samples_per_model=24,
        candidate_adapter_sha256=report['adapter_sha256'], original_business_template_sha256=proposed['original_business_template_sha256'],
        candidate_identity_matches_pretest_proposal=True, prior_downloaded_adapter_bytes_rehashed=True,
        adapter_weights_in_this_archive=False, source_code_and_frozen_data_match=True,
        inference_input_hashes_recomputed=48, paired_inputs_verified=24,
        training_source_documents=20, test_source_documents=5, overlapping_source_documents=0, exact_training_test_questions_overlap=0,
        frozen_data_checks=checks, generated_token_counts_source='Reported counts and unchanged evaluator; output token IDs not included, no local 8B regeneration.',
        base_identity_scope='Model path/config/template/package hashes and pinned local candidate; full base weights not rehashed.',
        original_human_reviews_pending=48, test_results_used_for_retraining_or_prompt_revision_in_this_review=False,
        semantic_review_is_separate=True)
    (OUT / 'artifact_verification.json').write_bytes(json_bytes(verification))
    print(json.dumps(dict(status=verification['status'], paired_inputs=24, real_tokenizer_inputs_verified=48,
        original_template=True, candidate_adapter_matched=True, source_overlap=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
