"""Create paired source-review materials; scores stay explicitly pending."""
import json

from evaluation.common import json_bytes
from prompt_pipeline.protocol import VARIANTS


def write_review(output, examples, predictions, dimensions):
    indexed = {(p['id'], p['variant']): p for p in predictions}
    pending = []
    lines = ['# 同一适配器：原提示词与修订提示词对照', '',
             '自动标记只提供格式线索，不能判断事实主体、条件或回答边界；所有题都需要对照证据检查。', '',
             '保持同一完整1轮适配器、问题、证据、历史与512新Token上限，仅改变通用回答规则。', '']
    for key, description in dimensions.items():
        lines.append(f'- {key}：{description}')
    for row in examples:
        lines += ['', f"## {row['id']} · {row['task_type']}", '', '问题：' + row['question'], '']
        if row['history']:
            lines += ['历史：' + json.dumps(row['history'], ensure_ascii=False), '']
        lines += ['参考答案（仅审核使用，不进入模型输入）：', '', row['reference_answer'], '']
        for variant in VARIANTS:
            p = indexed[(row['id'], variant)]
            lines += [f'### {variant}', '', p['answer'] or '（空回答）', '',
                      '自动提示：' + json.dumps(p['format_flags'], ensure_ascii=False), '']
            pending.append(dict(id=row['id'], variant=variant, scores={k: None for k in dimensions},
                reviewer='', notes='', status='pending'))
        lines += ['### 可见证据', '']
        for c in row['contexts']:
            lines += [f"[{c['reference_index']}] {c['relative_path']} L{c['line_start']}-{c['line_end']}", '', '```text', c['content'], '```', '']
        if not row['contexts']:
            lines += ['没有检索片段。', '']
    (output / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    (output / 'manual_reviews.json').write_bytes(json_bytes(dict(rubric=dimensions, reviews=pending)))
