"""Frozen protocol and prompt preparation; metadata never enters model inputs."""
from __future__ import annotations

import json
from pathlib import Path
from string import Formatter

from evaluation.common import json_bytes, read_jsonl, render_prompt, require, sha, validate_frozen

VARIANTS = ('original_prompt', 'revised_prompt')
ANCHOR = '\n\n【参考内容】\n'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_project(root):
    spec = load(root / 'prompt_pipeline/experiment_manifest.json')
    require(spec['experiment'] == 'fixed_epoch1_prompt_v2_validation', 'Unexpected prompt experiment version.')
    for name, expected in spec['project_files_sha256'].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file() and sha(path.read_bytes()) == expected,
                f'Frozen project dependency changed or missing: {name}')
    base = (root / 'data/eval_v1/prompt_template.txt').read_text(encoding='utf-8')
    rules = (root / 'prompt_pipeline/rules_v2.txt').read_text(encoding='utf-8').strip()
    candidate = (root / 'prompt_pipeline/prompt_template_v2.txt').read_text(encoding='utf-8')
    require(base.count(ANCHOR) == 1 and candidate == base.replace(ANCHOR, '\n\n' + rules + ANCHOR, 1),
            'Candidate must only insert the declared generic rules before the unchanged payload.')
    fields = lambda template: [name for _, name, _, _ in Formatter().parse(template) if name is not None]
    require(fields(candidate) == fields(base) == ['context', 'history', 'item_names', 'question'], 'Unexpected template fields.')
    return spec, {VARIANTS[0]: base, VARIANTS[1]: candidate}


def prompt_pair(row, templates):
    prompts = {variant: render_prompt(template, row['contexts'], row['history'], row['destination'], row['rewritten_query'])
               for variant, template in templates.items()}
    require(prompts[VARIANTS[0]] == row['prompt'], 'Original prompt differs from frozen evaluation input.')
    payload = prompts[VARIANTS[0]].split(ANCHOR, 1)[1]
    require(prompts[VARIANTS[1]].split(ANCHOR, 1)[1] == payload, 'Question/evidence/history payload changed between variants.')
    return prompts, sha(payload.encode('utf-8'))


def preflight(root, model_dir, adapter_dir, baseline_dir):
    spec, templates = check_project(root)
    _, data_checks = validate_frozen(root / 'data/eval_v1')
    examples = read_jsonl(root / 'data/eval_v1/validation.jsonl')
    require(len(examples) == 24, 'Expected full 24-question validation set.')
    for name, expected in spec['baseline_files_sha256'].items():
        require((baseline_dir / name).is_file() and sha((baseline_dir / name).read_bytes()) == expected, f'Previous epoch1 baseline changed: {name}')
    baseline_report = load(baseline_dir / 'report.json')
    baseline_predictions = read_jsonl(baseline_dir / 'adapter.predictions.jsonl')
    require(baseline_report['status'] == 'EVAL_GENERATION_OK' and baseline_report['split'] == 'validation' and
            baseline_report['samples'] == 24 and baseline_report['complete_split'], 'Previous baseline must be completed validation.')
    require(spec['generation'] == baseline_report['generation'], 'Generation settings must match the previous epoch1 run.')
    require(spec['adapter_sha256'] == baseline_report['adapter_sha256'] == sha((adapter_dir / 'adapter_model.safetensors').read_bytes()), 'Wrong candidate adapter weights.')
    require(spec['adapter_config_sha256'] == sha((adapter_dir / 'adapter_config.json').read_bytes()), 'Candidate adapter configuration changed.')
    require(spec['base_model_config_sha256'] == baseline_report['model_config_sha256'] == sha((model_dir / 'config.json').read_bytes()), 'Base model configuration changed.')
    require(spec['eval_manifest_sha256'] == baseline_report['eval_manifest_sha256'] == sha((root / 'data/eval_v1/manifest.json').read_bytes()), 'Evaluation manifest changed.')
    require(spec['validation_file_sha256'] == baseline_report['eval_file_sha256'] == sha((root / 'data/eval_v1/validation.jsonl').read_bytes()), 'Validation questions changed.')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    require(sha(tokenizer.chat_template.encode('utf-8')) == baseline_report['chat_template_sha256'] == spec['chat_template_sha256'], 'Chat template changed.')
    require(len(baseline_predictions) == 24, 'Expected 24 previous adapter predictions.')
    prepared = {variant: [] for variant in VARIANTS}
    for row, previous in zip(examples, baseline_predictions):
        require(row['id'] == previous['id'] and previous['model'] == 'adapter', 'Previous prediction identity mismatch.')
        pair, payload_hash = prompt_pair(row, templates)
        for variant in VARIANTS:
            rendered = tokenizer.apply_chat_template([dict(role='user', content=pair[variant])], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            require(len(ids) + spec['generation']['max_new_tokens'] <= spec['max_total_tokens'], f'Input budget exceeded: {row["id"]}; truncation forbidden.')
            item = dict(id=row['id'], ids=ids, prompt_tokens=len(ids), input_sha256=sha(json_bytes(ids)), payload_sha256=payload_hash)
            if variant == VARIANTS[0]:
                require(item['input_sha256'] == previous['input_sha256'] and len(ids) == previous['prompt_tokens'], 'Original prompt does not reproduce previous input tokens.')
            prepared[variant].append(item)
        require(prepared[VARIANTS[0]][-1]['input_sha256'] != prepared[VARIANTS[1]][-1]['input_sha256'], 'Revised prompt had no input effect.')
    report = dict(status='PROMPT_PREFLIGHT_OK', samples=24, split='validation', test_generation_enabled=False,
        adapter_sha256=spec['adapter_sha256'], unchanged_payloads_verified=24, original_inputs_match_previous=24,
        max_prompt_tokens={v: max(p['prompt_tokens'] for p in prepared[v]) for v in VARIANTS},
        max_new_tokens=spec['generation']['max_new_tokens'], max_total_tokens=spec['max_total_tokens'],
        truncated_samples=0, model_weights_loaded=False, runtime_package_identity_checked_before_generation=True)
    return spec, templates, examples, baseline_report, baseline_predictions, tokenizer, prepared, report
