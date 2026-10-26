"""Standard-library audit of completed training batches and optimizer updates."""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

from revision_pipeline.common import json_bytes, read_jsonl, require, sha

INDEX_KEY = '_coverage_indices'


def make_plan(raw, parent_count, *, batch_size=1, gradient_accumulation=4):
    require(len(raw) > 0 and 0 < parent_count < len(raw), 'Invalid original/supplement counts.')
    require(batch_size == 1 and gradient_accumulation == 4, 'This controlled experiment uses batch=1 and accumulation=4.')
    require(len({r['id'] for r in raw}) == len(raw), 'Sample IDs must be unique.')
    batches = math.ceil(len(raw) / batch_size)
    updates = math.ceil(batches / gradient_accumulation)
    return dict(num_train_epochs=1, max_steps=-1, sample_count=len(raw), parent_count=parent_count,
                supplement_count=len(raw) - parent_count, batch_size=batch_size, gradient_accumulation=gradient_accumulation,
                micro_batches=batches, optimizer_steps=updates,
                last_update_micro_batches=batches % gradient_accumulation or gradient_accumulation,
                seed=42, data_seed=42, learning_rate=1e-4, max_length=2048, packing=False, dataloader_drop_last=False)


def token_fingerprint(row):
    return sha(json_bytes(dict(input_ids=row['input_ids'], attention_mask=row['attention_mask'], labels=row['labels'])))


def check_update(batch_events, step, plan):
    expected = plan['last_update_micro_batches'] if step == plan['optimizer_steps'] else plan['gradient_accumulation']
    require(len(batch_events) == expected, f'Unexpected number of training batches in optimizer step {step}.')
    require(all(e['current_gradient_accumulation_steps'] == expected for e in batch_events), 'Runtime accumulation length differs from actual batch group.')
    supervised = sum(sum(e['supervised_tokens']) for e in batch_events)
    normalizers = [e['num_items_in_batch'] for e in batch_events]
    if all(n is None for n in normalizers):
        normalization = 'runtime_current_gradient_accumulation_steps'
    else:
        require(all(n == supervised for n in normalizers), 'Loss normalizer does not count the actual accumulation group answer tokens.')
        normalization = 'actual_group_supervised_token_count'
    return dict(optimizer_step=step, micro_batches=len(batch_events),
                sample_ids=[i for e in batch_events for i in e['sample_ids']],
                supervised_tokens=supervised, loss_normalization=normalization,
                num_items_in_batch=normalizers[0], current_gradient_accumulation_steps=expected)


def audit_coverage(output, raw, token_rows, plan):
    output = Path(output)
    events = read_jsonl(output / 'training_trace.jsonl')
    updates = read_jsonl(output / 'optimizer_trace.jsonl')
    require(len(events) == plan['micro_batches'], 'Incomplete training batch trace.')
    require(len(updates) == plan['optimizer_steps'], 'Incomplete optimizer update trace.')
    expected = {r['id']: token_fingerprint(t) for r, t in zip(raw, token_rows)}
    expected_supervised = {r['id']: sum(v != -100 for v in t['labels']) for r, t in zip(raw, token_rows)}
    counts = Counter()
    for number, e in enumerate(events, 1):
        require(e['micro_batch'] == number and e['forward_and_backward_completed'] is True, 'Invalid completed-batch event.')
        require(len(e['sample_ids']) == len(e['token_fingerprints']) == len(e['supervised_tokens']) == 1, 'Expected one real sample per batch.')
        for sample_id, fingerprint, supervised in zip(e['sample_ids'], e['token_fingerprints'], e['supervised_tokens']):
            require(sample_id in expected and fingerprint == expected[sample_id], 'Observed training data do not match frozen tokens/labels.')
            require(supervised == expected_supervised[sample_id], 'Observed supervised token count changed.')
            counts[sample_id] += 1
    require(set(counts) == set(expected) and all(n == 1 for n in counts.values()), 'One-epoch coverage requires every row exactly once.')
    for step, actual in enumerate(updates, 1):
        require(actual['optimizer_step'] == step and actual['gradient_checked_steps'] == step, 'Optimizer/gradient step sequence differs.')
        require(actual['optimizer_step_skipped'] is False, 'An optimizer update was skipped.')
        group = [e for e in events if e['optimizer_step'] == step]
        checked = check_update(group, step, plan)
        require(all(actual[k] == v for k, v in checked.items()), 'Optimizer update does not match its completed training batches.')
    parent_ids = {r['id'] for r in raw[:plan['parent_count']]}
    report = dict(status='EPOCH_COVERAGE_OK', dataset_samples=len(raw), unique_trained_samples=len(counts),
                  total_sample_presentations=sum(counts.values()), every_sample_exactly_once=True,
                  parent_samples_trained=len(parent_ids & set(counts)),
                  supplement_samples_trained=len(set(counts) - parent_ids),
                  completed_micro_batches=len(events), completed_optimizer_updates=len(updates),
                  last_update=updates[-1], per_sample_counts=dict(sorted(counts.items())),
                  training_trace_sha256=sha((output / 'training_trace.jsonl').read_bytes()),
                  optimizer_trace_sha256=sha((output / 'optimizer_trace.jsonl').read_bytes()),
                  coverage_scope='Recorded after each successful forward and backward; each group is matched to a non-skipped optimizer update. Not a quality score.')
    return report
