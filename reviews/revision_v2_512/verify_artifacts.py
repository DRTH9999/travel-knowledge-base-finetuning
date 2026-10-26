"""Read downloaded artifacts and verify identities without loading model weights."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = ROOT / 'artifacts/revision_v2_original'
NEW = RAW / 'eval_outputs/revision_v2_512'
OLD = ROOT / 'artifacts/validation_round2_512_original'
sys.path.insert(0, str(ROOT))
from evaluation.common import json_bytes, read_jsonl, validate_frozen
from revision_pipeline.common import validate_dataset


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_adapter(path):
    import numpy as np
    binary = path.read_bytes()
    header_size = struct.unpack('<Q', binary[:8])[0]
    header = json.loads(binary[8:8 + header_size])
    payload = memoryview(binary)[8 + header_size:]
    spans, tensors, elements, nonzero_b = [], [], 0, 0
    for name, item in header.items():
        if name == '__metadata__':
            continue
        start, end = item['data_offsets']
        size = math.prod(item['shape'])
        assert item['dtype'] == 'BF16' and end - start == size * 2
        assert 0 <= start < end <= len(payload)
        values = np.frombuffer(payload[start:end], dtype='<u2')
        assert not np.any((values & 0x7f80) == 0x7f80), name
        assert 'lora_' in name and 16 in item['shape']
        if '.lora_B.' in name:
            assert np.any((values & 0x7fff) != 0), name
            nonzero_b += 1
        elements += size
        tensors.append(name)
        spans.append((start, end))
    spans.sort()
    assert spans[0][0] == 0 and spans[-1][1] == len(payload)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert len(tensors) == 504 and elements == 43646976 and nonzero_b == 252
    return dict(tensors=len(tensors), parameters=elements, dtype='BF16', all_values_finite=True,
                nonzero_lora_B_tensors=nonzero_b, payload_size_and_offsets_valid=True)


def main():
    train = load(RAW / 'outputs/revision_v2_10steps/revision_report.json')
    cfg = load(RAW / 'outputs/revision_v2_10steps/run_config.json')
    state = load(RAW / 'outputs/revision_v2_10steps/trainer_state.json')
    new, old = load(NEW / 'report.json'), load(OLD / 'report.json')
    assert train['status'] == 'REVISION_TRAIN_OK' and new['status'] == 'EVAL_GENERATION_OK'
    assert train['training']['optimizer_steps'] == train['training']['gradient_checked_steps'] == state['global_step'] == 10
    assert cfg['max_steps'] == 10 and cfg['batch_size'] == 1 and cfg['gradient_accumulation'] == 4
    assert all(math.isfinite(r['loss']) for r in train['training']['loss_history'])
    assert train['training']['probe_max_absolute_update'] > 0
    for key in ('model_config_sha256', 'model_dir', 'eval_manifest_sha256', 'eval_file_sha256', 'chat_template_sha256', 'versions', 'generation'):
        assert new[key] == old[key], key
    manifest, frozen_checks = validate_frozen(ROOT / 'data/eval_v1')
    _, _, training_checks = validate_dataset(ROOT / 'data/revision_v2', require_tokenized=True)
    assert new['eval_manifest_sha256'] == sha(ROOT / 'data/eval_v1/manifest.json')
    assert new['eval_file_sha256'] == sha(ROOT / 'data/eval_v1/validation.jsonl')
    assert train['dataset_manifest_sha256'] == sha(ROOT / 'data/revision_v2/source_manifest.json')
    assert train['token_checks']['tokenized_sha256'] == sha(ROOT / 'data/revision_v2/train.tokenized.jsonl')
    for name in ('source_manifest.json', 'quality_report.json', 'tokenizer_report.json'):
        assert (RAW / 'data/revision_v2' / name).read_bytes() == (ROOT / 'data/revision_v2' / name).read_bytes(), name
    with zipfile.ZipFile(ROOT / 'sft_revision_v2.zip') as bundle:
        checksum = (RAW / 'revision_pipeline/checksums.sha256').read_bytes()
        assert checksum == bundle.read('revision_pipeline/checksums.sha256')
        for line in checksum.decode('utf-8').splitlines():
            expected, name = line.split('  ', 1)
            assert hashlib.sha256(bundle.read(name)).hexdigest() == expected
    adapter = RAW / 'outputs/revision_v2_10steps/adapter'
    actual_sha = sha(adapter / 'adapter_model.safetensors')
    assert actual_sha == train['reload']['adapter_sha256'] == new['adapter_sha256']
    assert actual_sha != old['adapter_sha256']
    tensor_checks = check_adapter(adapter / 'adapter_model.safetensors')
    lora = load(adapter / 'adapter_config.json')
    assert lora['r'] == 16 and lora['lora_alpha'] == 32 and lora['lora_dropout'] == .05
    rows = read_jsonl(ROOT / 'data/eval_v1/validation.jsonl')
    predictions = {k: read_jsonl(NEW / f'{k}.predictions.jsonl') for k in ('base', 'adapter')}
    previous = {k: read_jsonl(OLD / f'{k}.predictions.jsonl') for k in ('base', 'adapter')}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / '.cache/qwen3-8b-tokenizer'), local_files_only=True, trust_remote_code=False)
    assert hashlib.sha256(tokenizer.chat_template.encode('utf-8')).hexdigest() == new['chat_template_sha256']
    identical, changed = {}, {}
    comparison = (NEW / 'comparison.md').read_text(encoding='utf-8')
    for label in ('base', 'adapter'):
        assert len(predictions[label]) == len(previous[label]) == 24
        identical[label], changed[label] = [], []
        for row, p, o in zip(rows, predictions[label], previous[label]):
            assert row['id'] == p['id'] == o['id']
            prompt = tokenizer.apply_chat_template([dict(role='user', content=row['prompt'])], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            assert hashlib.sha256(json_bytes(ids)).hexdigest() == p['input_sha256'] == o['input_sha256']
            assert len(ids) == p['prompt_tokens'] == o['prompt_tokens']
            assert p['answer'] in comparison
            assert len(p['answer']) > 0
            (identical[label] if p['answer'] == o['answer'] else changed[label]).append(p['id'])
    assert len(identical['base']) == 24 and len(identical['adapter']) == 6
    manual = load(NEW / 'manual_reviews.json')['reviews']
    assert len(manual) == 48 and all(r['status'] == 'pending' and all(v is None for v in r['scores'].values()) for r in manual)
    report = dict(status='REVISION_V2_ARTIFACT_VERIFICATION_OK', checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  archive_sha256=sha(ROOT / 'qwen3_revision_v2_results.tar.gz'),
                  actual_adapter_sha256=actual_sha, previous_adapter_sha256=old['adapter_sha256'],
                  adapter_tensor_checks=tensor_checks, training_state=dict(steps=state['global_step'], epoch=state['epoch'],
                    expected_examples_presented=state['global_step'] * cfg['batch_size'] * cfg['gradient_accumulation'],
                    coverage_scope='40 row presentations inferred from 10 steps, batch 1, accumulation 4 and epoch 40/82; sample IDs not recorded.',
                    cloud_reload_equality_reported=train['reload']['saved_and_reloaded_weights_equal'],
                    full_8b_reload_repeated_locally=False),
                  paired_inputs_real_tokenizer_verified=24, same_generation_settings=True, same_model_path_config_versions=True,
                  identical_answer_ids=identical, changed_answer_ids=changed,
                  local_training_checks=training_checks, frozen_eval_checks=frozen_checks,
                  original_delivery_checksum_manifest_matches=True, original_manual_reviews_48_pending_preserved=True,
                  base_identity_scope='Path, configuration, template and version identity; full 8B base weight files not rehashed.',
                  test_predictions_generated_or_used=False,
                  semantic_review_not_provided_by_automatic_checks=True)
    (OUT / 'artifact_verification.json').write_bytes(json_bytes(report))
    print(json.dumps(dict(status=report['status'], adapter_tensors=tensor_checks['tensors'],
                          paired_inputs=24, identical_base_answers=len(identical['base']),
                          identical_adapter_answers=len(identical['adapter']), epoch=state['epoch']), ensure_ascii=False))


if __name__ == '__main__':
    main()
