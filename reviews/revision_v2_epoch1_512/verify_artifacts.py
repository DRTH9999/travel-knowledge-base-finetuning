"""Verify downloaded data, runtime source, coverage, weights and paired inputs.

Downloaded Python files are compared as bytes; they are never imported/executed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = ROOT / 'artifacts/revision_v2_epoch1_original'
TRAIN = RAW / 'outputs/revision_v2_epoch1'
NEW = RAW / 'eval_outputs/revision_v2_epoch1_512'
OLD_RAW = ROOT / 'artifacts/revision_v2_original'
OLD = OLD_RAW / 'eval_outputs/revision_v2_512'
sys.path.insert(0, str(ROOT))
from epoch_pipeline.coverage import audit_coverage
from epoch_pipeline.preflight import preflight
from evaluation.common import json_bytes, read_jsonl, validate_frozen
from revision_pipeline.common import sha, validate_dataset
from reviews.revision_v2_512.verify_artifacts import check_adapter


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return sha(path.read_bytes())


def main():
    train, old_train = load(TRAIN / 'epoch_report.json'), load(OLD_RAW / 'outputs/revision_v2_10steps/revision_report.json')
    new, old = load(NEW / 'report.json'), load(OLD / 'report.json')
    state, config = load(TRAIN / 'trainer_state.json'), load(TRAIN / 'run_config.json')
    assert train['status'] == 'REVISION_EPOCH_TRAIN_OK' and new['status'] == 'EVAL_GENERATION_OK'
    assert state['global_step'] == train['training']['optimizer_steps'] == train['training']['gradient_checked_steps'] == 21
    assert abs(state['epoch'] - 1) < 1e-8 and train['training']['actual_epochs'] == 1.0
    assert config['num_train_epochs'] == 1 and config['max_steps'] == -1
    assert all(math.isfinite(p['loss']) for p in train['training']['loss_history'])
    assert train['training']['probe_max_absolute_update'] > 0
    for key in ('model_dir', 'model_config_sha256', 'versions', 'dataset_manifest_sha256', 'eval_manifest_sha256', 'initialization'):
        assert train[key] == old_train[key], key
    old_cfg = load(OLD_RAW / 'outputs/revision_v2_10steps/run_config.json')
    for key in ('batch_size', 'gradient_accumulation', 'learning_rate', 'max_length', 'packing', 'gradient_checkpointing', 'trainable_parameters', 'label_policy'):
        assert config[key] == old_cfg[key], key
    manifest, raw, tokens, local_checks = preflight(ROOT, ROOT / 'data/revision_v2')
    returned_manifest, returned_rows, returned_checks = validate_dataset(RAW / 'data/revision_v2', require_tokenized=True)
    assert returned_rows == raw and returned_manifest == manifest
    for path in (ROOT / 'data/revision_v2').rglob('*'):
        if path.is_file():
            relative = path.relative_to(ROOT / 'data/revision_v2')
            assert path.read_bytes() == (RAW / 'data/revision_v2' / relative).read_bytes(), relative
    assert train['dataset_manifest_sha256'] == digest(RAW / 'data/revision_v2/source_manifest.json')
    assert train['token_checks']['tokenized_sha256'] == digest(RAW / 'data/revision_v2/train.tokenized.jsonl')
    coverage = audit_coverage(TRAIN, returned_rows, tokens, local_checks['plan'])
    saved_coverage = load(TRAIN / 'coverage_report.json')
    assert all(saved_coverage[k] == v for k, v in coverage.items())
    assert saved_coverage == train['training']['coverage']
    assert saved_coverage['model_forward_calls'] == 82 and saved_coverage['metadata_excluded_from_model_input'] is True
    assert coverage['unique_trained_samples'] == 82 and coverage['parent_samples_trained'] == 50 and coverage['supplement_samples_trained'] == 32
    assert coverage['last_update']['micro_batches'] == 2 and coverage['last_update']['num_items_in_batch'] == 168

    with zipfile.ZipFile(ROOT / 'sft_revision_v2_epoch1.zip') as bundle:
        checksum = bundle.read('epoch_pipeline/checksums.sha256')
        assert checksum == (RAW / 'epoch_pipeline/checksums.sha256').read_bytes()
        shipped_files = []
        for line in checksum.decode('utf-8').splitlines():
            expected, name = line.split('  ', 1)
            assert digest(RAW / name) == hashlib.sha256(bundle.read(name)).hexdigest() == expected, name
            shipped_files.append(name)
    for name, expected in train['training_code_sha256'].items():
        assert digest(RAW / name) == expected == digest(ROOT / name), name
    dependency = load(RAW / 'epoch_pipeline/dependencies.json')
    bundled_deps, absent_deps = [], []
    for name, expected in dependency['project_files_sha256'].items():
        if (RAW / name).is_file():
            assert digest(RAW / name) == expected, name
            bundled_deps.append(name)
        else:
            absent_deps.append(name)
    assert absent_deps == ['revision_pipeline/__init__.py']
    _, frozen = validate_frozen(ROOT / 'data/eval_v1')
    for name in ('manifest.json', 'validation.jsonl'):
        assert (RAW / 'data/eval_v1' / name).read_bytes() == (ROOT / 'data/eval_v1' / name).read_bytes()
    for key in ('model_config_sha256', 'model_dir', 'eval_manifest_sha256', 'eval_file_sha256', 'chat_template_sha256', 'versions', 'generation'):
        assert new[key] == old[key], key
    assert new['eval_manifest_sha256'] == digest(ROOT / 'data/eval_v1/manifest.json')
    assert new['eval_file_sha256'] == digest(ROOT / 'data/eval_v1/validation.jsonl')
    adapter = TRAIN / 'adapter'
    weight_sha = digest(adapter / 'adapter_model.safetensors')
    assert weight_sha == train['reload']['adapter_sha256'] == new['adapter_sha256']
    assert weight_sha != old['adapter_sha256']
    tensor_checks = check_adapter(adapter / 'adapter_model.safetensors')
    new_lora = load(adapter / 'adapter_config.json')
    old_lora = load(OLD_RAW / 'outputs/revision_v2_10steps/adapter/adapter_config.json')
    for key in ('r', 'lora_alpha', 'lora_dropout', 'bias', 'task_type', 'base_model_name_or_path'):
        assert new_lora[key] == old_lora[key], key
    assert set(new_lora['target_modules']) == set(old_lora['target_modules'])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / '.cache/qwen3-8b-tokenizer'), local_files_only=True, trust_remote_code=False)
    assert sha(tokenizer.chat_template.encode('utf-8')) == new['chat_template_sha256']
    examples = read_jsonl(ROOT / 'data/eval_v1/validation.jsonl')
    same, changed = {}, {}
    for label in ('base', 'adapter'):
        before = read_jsonl(OLD / f'{label}.predictions.jsonl')
        after = read_jsonl(NEW / f'{label}.predictions.jsonl')
        assert len(before) == len(after) == 24
        same[label], changed[label] = [], []
        for row, p, o in zip(examples, after, before):
            assert row['id'] == p['id'] == o['id']
            prompt = tokenizer.apply_chat_template([dict(role='user', content=row['prompt'])], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            assert sha(json_bytes(ids)) == p['input_sha256'] == o['input_sha256']
            assert len(ids) == p['prompt_tokens'] == o['prompt_tokens']
            assert p['answer'] in (NEW / 'comparison.md').read_text(encoding='utf-8')
            (same[label] if p['answer'] == o['answer'] else changed[label]).append(p['id'])
    assert len(same['base']) == 24 and len(same['adapter']) == 1
    manual = load(NEW / 'manual_reviews.json')['reviews']
    assert len(manual) == 48 and all(r['status'] == 'pending' and all(v is None for v in r['scores'].values()) for r in manual)
    loss_differences = [abs(a['loss'] - b['loss']) for a, b in zip(train['training']['loss_history'][:10], old_train['training']['loss_history'])]
    source_expected = {p.relative_to(ROOT / 'data/revision_v2').as_posix() for p in (ROOT / 'data/revision_v2').rglob('*') if p.is_file()}
    source_actual = {p.relative_to(RAW / 'data/revision_v2').as_posix() for p in (RAW / 'data/revision_v2').rglob('*') if p.is_file()}
    report = dict(status='EPOCH1_ARTIFACT_VERIFICATION_OK', checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  archive_sha256=digest(ROOT / 'qwen3_revision_v2_epoch1_results.tar.gz'),
                  adapter_sha256=weight_sha, previous_adapter_sha256=old['adapter_sha256'], tensor_checks=tensor_checks,
                  independently_recomputed_coverage=coverage, cloud_forward_and_metadata_guard_report=saved_coverage,
                  expected_current_shipped_files_verified=len(shipped_files), bundled_runtime_dependencies_verified=bundled_deps,
                  dependency_not_in_result_archive=absent_deps, returned_training_code_matches_shipped_and_report=True,
                  returned_dataset_matches_local_frozen_82=True, returned_data_extra_unreferenced_files=sorted(source_actual - source_expected),
                  eval_inputs_real_tokenizer_verified=24, same_eval_settings_versions_template=True,
                  base_answers_identical=24, adapter_answers_identical=1, unchanged_answer_ids=same, changed_answer_ids=changed,
                  first_10_training_losses_max_abs_difference=max(loss_differences), bitwise_training_reproducibility_claimed=False,
                  cloud_save_reload_report=train['reload'], full_8b_reload_repeated_locally=False,
                  base_weight_identity_scope='Same path/config/template/versions and 24 identical base outputs; full base weight files not rehashed.',
                  original_human_reviews_pending=48, test_predictions_generated_or_used=False,
                  frozen_eval_data_checks=frozen, semantic_review_is_separate_from_integrity_checks=True)
    (OUT / 'artifact_verification.json').write_bytes(json_bytes(report))
    print(json.dumps(dict(status=report['status'], unique_trained_samples=coverage['unique_trained_samples'],
        optimizer_updates=coverage['completed_optimizer_updates'], last_update_batches=coverage['last_update']['micro_batches'],
        original_rows=coverage['parent_samples_trained'], supplement_rows=coverage['supplement_samples_trained'],
        paired_inputs=24, identical_base_answers=24, identical_adapter_answers=1,
        extra_unreferenced_source_files=len(report['returned_data_extra_unreferenced_files'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
