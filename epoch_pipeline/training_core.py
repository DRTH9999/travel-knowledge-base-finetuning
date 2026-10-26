"""One-epoch TRL training with metadata excluded from model input."""
from __future__ import annotations

import json
import math

from epoch_pipeline.coverage import INDEX_KEY, audit_coverage, check_update, token_fingerprint
from revision_pipeline.common import json_bytes, require
from revision_pipeline.legacy_training_core import make_gradient_guard, verify_collator


def append_event(path, event):
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(event, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def run_epoch(model, tokenizer, token_rows, raw, plan, output):
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    events = []
    forward_calls = 0
    trace_path = output / 'training_trace.jsonl'
    update_path = output / 'optimizer_trace.jsonl'
    trace_path.touch(exist_ok=False)
    update_path.touch(exist_ok=False)
    dataset_rows = [dict(row, **{INDEX_KEY: index}) for index, row in enumerate(token_rows)]

    class TrackingCollator:
        def __init__(self, delegate):
            self.delegate = delegate

        def __call__(self, features):
            indices = [f[INDEX_KEY] for f in features]
            require(all(type(i) is int and 0 <= i < len(raw) for i in indices), 'Invalid sample tracking index.')
            clean = [{k: v for k, v in f.items() if k != INDEX_KEY} for f in features]
            batch = self.delegate(clean)
            batch[INDEX_KEY] = indices
            return batch

    class TrackingTrainer(SFTTrainer):
        _pending_coverage = None

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            indices = inputs.pop(INDEX_KEY)
            require(self.is_in_train and self.model.training, 'Coverage tracking is restricted to training.')
            observed, supervised = [], []
            for offset, index in enumerate(indices):
                mask = inputs['attention_mask'][offset].detach().cpu().tolist()
                length = sum(mask)
                require(mask == [1] * length + [0] * (len(mask) - length), 'Expected right padding.')
                row = {k: inputs[k][offset, :length].detach().cpu().tolist() for k in ('input_ids', 'attention_mask', 'labels')}
                require(row == token_rows[index], 'Actual model inputs/labels differ from the tracked sample.')
                require(all(v == -100 for v in inputs['labels'][offset, length:].detach().cpu().tolist()), 'Padding contributes to loss.')
                observed.append(token_fingerprint(row))
                supervised.append(sum(v != -100 for v in row['labels']))
            require(INDEX_KEY not in inputs, 'Tracking metadata reached model inputs.')
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
            self._pending_coverage = dict(optimizer_step=self.state.global_step + 1,
                sample_ids=[raw[i]['id'] for i in indices], token_fingerprints=observed, supervised_tokens=supervised,
                current_gradient_accumulation_steps=self.current_gradient_accumulation_steps,
                num_items_in_batch=None if num_items_in_batch is None else int(num_items_in_batch))
            return result

        def training_step(self, model, inputs, num_items_in_batch=None):
            self._pending_coverage = None
            previous_forward_calls = forward_calls
            result = super().training_step(model, inputs, num_items_in_batch)
            require(self._pending_coverage is not None and forward_calls == previous_forward_calls + 1, 'Expected one tracked forward per training batch.')
            require(torch.isfinite(result).all().item(), 'Non-finite backward-completed batch loss.')
            event = dict(micro_batch=len(events) + 1, **self._pending_coverage, forward_and_backward_completed=True)
            events.append(event)
            append_event(trace_path, event)
            self._pending_coverage = None
            return result

    def forward_guard(module, args, kwargs):
        nonlocal forward_calls
        require(INDEX_KEY not in kwargs, 'Coverage metadata must never reach model forward.')
        forward_calls += 1

    settings = SFTConfig(output_dir=str(output), num_train_epochs=1, max_steps=-1,
        per_device_train_batch_size=plan['batch_size'], gradient_accumulation_steps=plan['gradient_accumulation'],
        learning_rate=plan['learning_rate'], lr_scheduler_type='constant', warmup_steps=0,
        optim='adamw_torch', weight_decay=0.0, max_grad_norm=1.0, bf16=True, fp16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        max_length=plan['max_length'], packing=False, padding_free=False,
        dataset_kwargs={'skip_prepare_dataset': True}, completion_only_loss=False, assistant_only_loss=False,
        remove_unused_columns=False, dataloader_num_workers=0, dataloader_drop_last=False,
        logging_strategy='steps', logging_steps=1, logging_nan_inf_filter=False,
        save_strategy='no', eval_strategy='no', report_to='none', push_to_hub=False,
        seed=plan['seed'], data_seed=plan['data_seed'])
    gradient_guard = make_gradient_guard(torch, TrainerCallback)
    trainer = TrackingTrainer(model=model, args=settings, train_dataset=Dataset.from_list(dataset_rows),
                              processing_class=tokenizer, callbacks=[gradient_guard])
    trainer.data_collator = TrackingCollator(trainer.data_collator)
    collator_checks = verify_collator(trainer, token_rows)
    require(len(events) == forward_calls == 0, 'Collator checks must not count as training.')

    class UpdateGuard(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            require(not trainer.accelerator.optimizer_step_was_skipped, 'An optimizer update was skipped.')
            require(gradient_guard.checked_steps == state.global_step, 'Gradient check did not match optimizer update.')
            group = [e for e in events if e['optimizer_step'] == state.global_step]
            event = check_update(group, state.global_step, plan)
            event.update(gradient_checked_steps=gradient_guard.checked_steps, optimizer_step_skipped=False)
            append_event(update_path, event)

    trainer.add_callback(UpdateGuard())
    trainable = [(n, p) for n, p in trainer.model.named_parameters() if p.requires_grad]
    require(trainable and all('lora_' in n for n, _ in trainable), 'Only LoRA parameters should be trainable.')
    probe_name, probe = next((n, p) for n, p in trainable if 'lora_B' in n)
    before = probe.detach().float().cpu().clone()
    config = dict(plan, label_policy='use_verified_precomputed_labels_only', collator_checks=collator_checks,
                  trainable_parameters=sum(p.numel() for _, p in trainable), initialization='fresh_base_and_fresh_lora_no_resume',
                  gradient_checkpointing=True, metadata_excluded_from_model_input=True,
                  coverage_recording='actual forward+backward and confirmed optimizer update; excludes collator checks/prefetch',
                  expected_update_micro_batches=[4] * (plan['optimizer_steps'] - 1) + [plan['last_update_micro_batches']],
                  evaluation='No validation generation inside training; separate fixed-evidence evaluation afterward.')
    (output / 'run_config.json').write_bytes(json_bytes(config))
    hook = trainer.model.register_forward_pre_hook(forward_guard, with_kwargs=True)
    try:
        result = trainer.train()
    finally:
        hook.remove()
    require(trainer.state.global_step == gradient_guard.checked_steps == plan['optimizer_steps'], 'Unexpected optimizer/gradient step count.')
    require(abs(trainer.state.epoch - 1.0) < 1e-8, 'Training did not complete exactly one epoch.')
    require(math.isfinite(result.training_loss) and gradient_guard.losses, 'Invalid training loss.')
    coverage = audit_coverage(output, raw, token_rows, plan)
    require(forward_calls == coverage['completed_micro_batches'], 'Actual model forward count differs from completed batch trace.')
    coverage.update(model_forward_calls=forward_calls, metadata_excluded_from_model_input=True, actual_trainer_epoch=trainer.state.epoch)
    delta = (probe.detach().float().cpu() - before).abs().max().item()
    require(math.isfinite(delta) and delta > 0, 'LoRA probe weights did not update.')
    adapter = output / 'adapter'
    trainer.model.save_pretrained(str(adapter), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter))
    trainer.save_state()
    (output / 'train_metrics.json').write_bytes(json_bytes(dict(result.metrics)))
    (output / 'coverage_report.json').write_bytes(json_bytes(coverage))
    print('EPOCH_COVERAGE_OK', flush=True)
    return dict(optimizer_steps=trainer.state.global_step, gradient_checked_steps=gradient_guard.checked_steps,
                actual_epochs=trainer.state.epoch, final_training_loss=result.training_loss,
                loss_history=gradient_guard.losses, probe_parameter=probe_name, probe_max_absolute_update=delta,
                trainable_parameters=sum(p.numel() for _, p in trainable), collator_checks=collator_checks, coverage=coverage)
