"""Record assistant source review; preserve raw artifacts and pending human forms."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = ROOT / 'artifacts/revision_v2_original'
NEW = RAW / 'eval_outputs/revision_v2_512'
OLD = ROOT / 'artifacts/validation_round2_512_original'
DIMENSIONS = ['faithfulness', 'completeness', 'citation_support', 'answerability', 'constraints_format']
NAMES = dict(faithfulness='事实忠实', completeness='完整性', citation_support='引用支持', answerability='回答边界', constraints_format='约束与表达')

# Every changed adapter answer below was read alongside its complete visible evidence.
# Same rubric as round 2; no penalty merely for omitting a redundant citation.
CHANGED_REVIEWS = {
    2: ([2, 1, 2, 2, 2], '退步：正确说明不能步行、需要轮渡，但遗漏用户同时询问的码头、船票、候船、乘船与入岛流程。未触及输出上限，不能归因于512 Token截断。'),
    3: ([2, 2, 2, 2, 2], '打车建议直接对应多人、带行李和不想换乘的诉求；[1]支持。比旧回答更直接点明适用条件，量表分数保持。'),
    5: ([2, 2, 2, 2, 2], '简餐与正餐两项均正确且受[1]支持。引用前多余逗号可整理，但不影响核心可用性，不单独扣分。'),
    6: ([2, 2, 2, 2, 2], '已修复旧漏答：同时给出沉浸感、清晨夜晚氛围，以及行李搬运、上岛组织、返程节奏复杂的代价；[1]支持两面。'),
    7: ([2, 2, 2, 2, 2], '明确表示适合，并以沙滩、骑行、散步和轻松餐饮回应用户偏好，均来自[1]。'),
    9: ([2, 2, 2, 2, 2], '住宿片区和出发前核对清单均覆盖，分别由[1][2]支持。'),
    10: ([2, 1, 2, 2, 2], '退步：说明第一天老城及鼓浪屿应留完整时段，却没安排第三天海边；鼓浪屿也没有明确放在第二天。核心问题要求三天分别留老城、岛和海边，新增引用[2]不能弥补漏答。'),
    11: ([2, 2, 2, 2, 2], '白天给交通组合，晚上给中山路/轮渡片区；在交通与住宿搭配的问境下，第二段已提供住宿区域方向，未强行因省略“住”字判为漏答。[1][2]支持主要内容。'),
    12: ([2, 1, 2, 2, 1], '部分改善：不再补造岛上寄存服务，减轻行李和评估岛外住宿属于依据行李/上岛成本提出的建议，非具体酒店服务承诺。但仍未清楚比较住一晚的氛围收益与搬运行李成本，“住岛外，再安排一晚”去向和安排含糊，需要澄清。'),
    13: ([2, 2, 2, 2, 2], '按吃当地小吃和边走边逛优先老城，理由由[1]支持。[2]是另一段支持住宿取舍的材料，没有引用它不等于错引或漏答，不要求为凑齐参考编号复述海边。'),
    14: ([1, 2, 1, 2, 2], '轻微退步：保留老城+鼓浪屿、允许缩减环岛路，仍回应两天不赶的需求；但新增“和曾厝垵、厦大周边合并安排”没有当前[2]的明确支持，[2]只列第三天选项和两天可缩减。记录为证据外的规划扩写，不据此断言一定赶行程。'),
    15: ([2, 2, 2, 2, 2], '厦门站及市中心作为短住换乘优先区域完全由[1]支持；[2]为同义补充，自动缺引提示不构成实际错误。'),
    16: ([1, 2, 2, 2, 2], '轻微退步：公交适用预算、时间和片区的关键建议仍受[1]支持，但把“覆盖面较广”改成“公交是厦门的特色”，增加无当前依据的概括性判断。按轻微忠实性问题记录，不把主要建议的引用同时判错。'),
    17: ([0, 2, 0, 2, 2], '原关键问题未解决：不是必须住岛上、可选中山路/轮渡的方向有依据；仍断言岛上住宿资源有限、没有大型超市/餐厅/洗衣店等设施，并把它挂到只讲本岛片区便利的[2]，属于无依据断言及语义错引。'),
    18: ([2, 2, 2, 2, 2], '仍依据当前轮渡证据纠正错误历史，包含上岛流程及三类码头核对，引用对应。可更明确承认历史说法有误，但当前事实方向正确。'),
    19: ([2, 1, 2, 1, 2], '部分改善：已回答核对码头/航线/班次等有依据的内容，修复旧回答把已知清单也拒答的问题；仍没有明确说明今天末班轮渡和票价未提供、无法确认，所以不算完整修复部分可答边界。'),
    22: ([2, 2, 2, 2, 2], '没有编造宠物政策或押金，拒答边界正确。旧答指向酒店，新答退回“景区官方公告”的通用句，渠道针对性变弱；但业务提示词本就要求这句通用措辞，沿用之前不单独扣分的口径，将模板问题与模型事实错误区分记录。'),
    23: ([2, 2, 2, 2, 2], '仍只引用真正支持老城/鼓浪屿住宿的[2]，没有被小吃及海边干扰片段带偏，关键事实与引用均匹配。'),
}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    verification = load(OUT / 'artifact_verification.json')
    assert verification['status'] == 'REVISION_V2_ARTIFACT_VERIFICATION_OK'
    previous = load(ROOT / 'reviews/validation_round2_512/assistant_reviews.json')
    prior_reviews = {(r['id'], r['model']): r for r in previous['reviews']}
    examples = rows(ROOT / 'data/eval_v1/validation.jsonl')
    predictions = {label: rows(NEW / f'{label}.predictions.jsonl') for label in ('base', 'adapter')}
    old_predictions = {label: rows(OLD / f'{label}.predictions.jsonl') for label in ('base', 'adapter')}
    reviews, changes, inherited = [], [], []
    for index, example in enumerate(examples, 1):
        for label in ('base', 'adapter'):
            old, new = old_predictions[label][index - 1], predictions[label][index - 1]
            assert old['id'] == new['id'] == example['id'] and old['input_sha256'] == new['input_sha256']
            prior = prior_reviews[(example['id'], label)]
            assert prior['prediction_sha256'] == digest(old['answer'].encode('utf-8'))
            if old['answer'] == new['answer']:
                score, rationale = prior['scores'], prior['rationale']
                method = 'prior_source_review_reused_after_exact_input_answer_evidence_identity'
                inherited.append([example['id'], label])
            else:
                assert label == 'adapter' and index in CHANGED_REVIEWS
                score_list, rationale = CHANGED_REVIEWS[index]
                score = dict(zip(DIMENSIONS, score_list))
                method = 'changed_adapter_answer_read_against_complete_visible_evidence'
                changes.append(example['id'])
            assert set(score) == set(DIMENSIONS) and all(type(v) is int and 0 <= v <= 2 for v in score.values())
            reviews.append(dict(id=example['id'], model=label, scores=score, total=sum(score.values()),
                                rationale=rationale, review_method=method, human_signoff=False,
                                status='assistant_reviewed_user_pending', prediction_sha256=digest(new['answer'].encode('utf-8'))))
    assert len(changes) == len(CHANGED_REVIEWS) == 18 and len(inherited) == 30
    paired = {e['id']: {r['model']: r for r in reviews if r['id'] == e['id']} for e in examples}
    score_changes = {e['id']: paired[e['id']]['adapter']['total'] - prior_reviews[(e['id'], 'adapter')]['total'] for e in examples}
    stats = {}
    for run, prediction_set in [('previous_50rows_10steps', old_predictions), ('revision_82rows_10steps', predictions)]:
        stats[run] = {}
        for label, items in prediction_set.items():
            stats[run][label] = dict(samples=len(items), mean_output_tokens=sum(p['generated_tokens'] for p in items) / len(items),
                                    mean_generation_seconds=sum(p['generation_seconds'] for p in items) / len(items),
                                    output_limit_count=sum(p['hit_output_limit'] for p in items),
                                    empty_answer_count=sum(not bool(p['answer'].strip()) for p in items),
                                    invalid_reference_count=sum(bool(p['format_flags']['invalid_reference_ids']) for p in items),
                                    missing_gold_reference_count=sum(bool(p['format_flags']['missing_gold_reference_ids']) for p in items),
                                    missing_gold_reference_ids=[p['id'] for p in items if p['format_flags']['missing_gold_reference_ids']],
                                    image_url_or_placeholder_count=sum(bool(re.search(r'https?://|<图片URL', p['answer'])) for p in items))
    model_scores = {}
    for label in ('base', 'adapter'):
        items = [r for r in reviews if r['model'] == label]
        model_scores[label] = dict(mean_scores={d: sum(r['scores'][d] for r in items) / len(items) for d in DIMENSIONS},
                                   mean_total=sum(r['total'] for r in items) / len(items))
    summary = dict(status='REVISION_V2_ASSISTANT_REVIEW_COMPLETE', reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
                   assessor='Codex assistant; same assistant authored training/eval examples; nonblind and not independent human review',
                   human_signoff=False, score_scale='5 dimensions each 0-2; 0-10 rubric totals, not accuracy',
                   archive_sha256=verification['archive_sha256'], adapter_sha256=verification['actual_adapter_sha256'],
                   reviewed_changed_adapter_answers=18, inherited_after_identity_check=30, review_count=48,
                   identical_base_answers=24, identical_adapter_answers=6,
                   previous_adapter_mean_total=previous['summary']['models']['adapter']['mean_total'],
                   models=model_scores, observations=stats,
                   adapter_vs_previous_rubric=dict(higher_ids=[k for k, v in score_changes.items() if v > 0],
                                                  tied_ids=[k for k, v in score_changes.items() if v == 0],
                                                  lower_ids=[k for k, v in score_changes.items() if v < 0]),
                   target_failure_followup={'006': 'resolved_in_this_answer', '012': 'unsupported_storage_removed_but_tradeoff_still_unclear',
                                            '017': 'unsupported_facility_assertion_and_wrong_citation_remain',
                                            '019': 'known_checklist_restored_but_missing_realtime_boundary_omitted',
                                            '020': 'unchanged_missing_room_stock_and_price_boundary'},
                   clear_new_completeness_regressions=['travel-eval-validation-002', 'travel-eval-validation-010'],
                   mild_new_evidence_overreach=['travel-eval-validation-014', 'travel-eval-validation-016'],
                   test_predictions_generated_or_used=False,
                   limitations=['Only one held-out destination and 24 synthetic questions sharing five source documents.',
                                'Validation has already guided revision design; this is an iterative regression set, not independent final evidence.',
                                'Same assistant authored data and assessed outputs; no blind assessment or independent human signoff.',
                                '10 steps cover epoch 0.4878; no per-sample training trace, and coverage is not the only possible cause of residual errors.',
                                'Base identity uses model path/config/template/version and unchanged outputs, not full base-weight rehashing.',
                                'Single-run timing is uncontrolled and depends on answer length; no throughput or latency advantage claim.',
                                'Visible-source faithfulness is assessed; tourism facts were not externally verified in real time.'])
    save('assistant_reviews.json', dict(summary=summary, reviews=reviews))
    save('assistant_summary.json', summary)

    comparison = ['# 原始模型、旧适配器与revision_v2逐题对照', '',
                  '审核主体为助手；非独立人工、非盲审，业务验收待完成。原始下载文件和待人工评分表保持不变。', '',
                  '每题列出用户问题、历史、原始模型、旧/新适配器、审核理由及全部可见证据。不能把自动缺引提示直接当成语义错误。', '']
    for index, example in enumerate(examples):
        old_review = prior_reviews[(example['id'], 'adapter')]
        current = paired[example['id']]['adapter']
        comparison += [f"## {example['id']} · {example['task_type']}", '', '问题：' + example['question'], '']
        if example['history']:
            comparison += ['历史：' + json.dumps(example['history'], ensure_ascii=False), '']
        comparison += ['参考答案（审核辅助，不输入模型）：', '', example['reference_answer'], '',
                       f"### 原始模型 · {paired[example['id']]['base']['total']}/10", '', predictions['base'][index]['answer'], '',
                       f"### 旧适配器 · {old_review['total']}/10", '', old_predictions['adapter'][index]['answer'], '',
                       '旧审核：' + old_review['rationale'], '', f"### 新适配器 · {current['total']}/10", '',
                       predictions['adapter'][index]['answer'], '', '新审核：' + current['rationale'], '',
                       '分项分数：' + json.dumps(current['scores'], ensure_ascii=False), '',
                       '自动提示：' + json.dumps(predictions['adapter'][index]['format_flags'], ensure_ascii=False), '']
        for c in example['contexts']:
            comparison += [f"证据[{c['reference_index']}] {c['relative_path']} L{c['line_start']}-{c['line_end']}", '', '```text', c['content'], '```', '']
        if not example['contexts']:
            comparison += ['没有检索片段。', '']
    (OUT / '逐题对照.md').write_text('\n'.join(comparison), encoding='utf-8')

    old_stats, new_stats = stats['previous_50rows_10steps']['adapter'], stats['revision_82rows_10steps']['adapter']
    train = load(RAW / 'outputs/revision_v2_10steps/revision_report.json')
    report = ['# revision_v2：82条数据、10步短训结果审核', '',
              '审核日期：2026-10-24（Asia/Shanghai）。审核主体为助手，非独立人工、非盲审；业务验收仍待完成。', '',
              '## 结论', '',
              '训练链路及产物核验通过；回答出现局部改善，同时出现新的漏答，目前不建议进入部署。原五类问题中，006在本次回答中修复，012与019部分改善，017与020仍未解决。新增明确漏答为002上岛流程、010第三天海边安排。', '',
              '基础模型24份回答与上一轮逐字一致；新适配器有18份回答改变、6份不变。已重新阅读18份改变的回答及其全部证据，其余30份在输入、回答和证据一致后沿用旧审核。', '',
              '## 原五类问题追踪', '',
              '| 题号 | 本轮变化 | 判定 |', '|---|---|---|',
              '| 006 住岛优缺点 | 已补出搬运行李、上岛组织、返程节奏复杂 | 本次回答修复 |',
              '| 012 大箱子与住一晚 | 删除无依据的寄存建议；仍没清楚比较氛围收益与行李成本，“住岛外，再安排一晚”含糊 | 部分改善 |',
              '| 017 历史追问 | 仍编造岛上住宿资源有限、没有大型超市/餐厅/洗衣店，并错引[2] | 未解决 |',
              '| 019 今日轮渡与核对清单 | 已回答已知核对清单；没说明末班时间与票价缺资料、无法确认 | 部分改善 |',
              '| 020 今晚房态和预算 | 与旧答逐字一致，只说适合人群，没交代库存/价格未知 | 未解决 |', '',
              '## 新出现的退步', '',
              '- 002：从完整上岛流程缩成“不能直接步行，必须轮渡”，漏掉用户同时要求的具体交通环节。',
              '- 010：旧答虽简短，但列全三天；新答讲了第一天和鼓浪屿，不再安排第三天海边。引用从[1]变为[1][2]，自动缺引提示消失，语义却变得不完整。',
              '- 014：仍允许缩减环岛路，但新增与曾厝垵、厦大周边“合并安排”；可见证据仅列第三天选项及两天可缩减，缺少对该合并方案的明确支持，记为轻微证据外规划扩写。',
              '- 016：主要公交建议正确，但将“覆盖面较广”改成“公交是厦门的特色”，新增不受当前证据支持的概括，记轻微忠实性问题。',
              '- 022：酒店查询渠道由“酒店官方”退为“景区官方”，针对性变弱；因当前提示词要求该通用措辞，沿用旧口径不单独扣分，后续应作为独立模板问题处理。', '',
              '## 自动提示与助手量表', '',
              '| 指标 | 旧50条、10步 | 新82条、10步 |', '|---|---:|---:|']
    for title, key, fmt in [('空回答', 'empty_answer_count', 'd'), ('触及输出上限', 'output_limit_count', 'd'),
                            ('越界引用题数', 'invalid_reference_count', 'd'), ('自动缺引提示题数', 'missing_gold_reference_count', 'd'),
                            ('无来源图片URL/占位符题数', 'image_url_or_placeholder_count', 'd'),
                            ('平均输出Token', 'mean_output_tokens', '.2f'), ('平均单题生成秒数', 'mean_generation_seconds', '.3f')]:
        report.append(f"| {title} | {format(old_stats[key], fmt)} | {format(new_stats[key], fmt)} |")
    report += ['', '新适配器剩余的两道自动缺引提示是013和015，现有关键结论由已引片段支持，不强制为了凑齐参考编号引用冗余材料。自动提示不能当作错误题数或准确率。017的语义错引、002/010的漏答仍需要阅读证据发现。', '',
               '| 量表维度 | 旧适配器 | 新适配器 |', '|---|---:|---:|']
    for key in DIMENSIONS:
        before = previous['summary']['models']['adapter']['mean_scores'][key]
        report.append(f"| {NAMES[key]}（0–2） | {before:.3f} | {model_scores['adapter']['mean_scores'][key]:.3f} |")
    report += [f"| 总分（0–10） | {summary['previous_adapter_mean_total']:.3f} | {model_scores['adapter']['mean_total']:.3f} |", '',
               f"按这份助手量表，新适配器比旧适配器分数高的题有{len(summary['adapter_vs_previous_rubric']['higher_ids'])}道，低的有{len(summary['adapter_vs_previous_rubric']['lower_ids'])}道，持平{len(summary['adapter_vs_previous_rubric']['tied_ids'])}道。小幅总分变化不能掩盖017的关键编造和新增漏答，也不是准确率或已通过业务验收的证据。", '',
               '单次耗时不作为速度提升结论；两个适配器输出内容和长度不同，GPU负载也未受控。', '',
               '## 训练和身份核验', '',
               f"- 训练状态 `{train['status']}`；10步、10次梯度检查，82条真实批处理标签/填充屏蔽检查通过，抽查参数更新非零。",
               f"- 平均训练损失{train['training']['final_training_loss']:.6f}，训练显存峰值{train['training_peak_allocated_vram_gib']:.3f} GiB。新旧训练集合不同，不把loss变化当质量改善。",
               '- 下载的适配器SHA256与训练及评测报告相符；实际文件检查504个BF16 LoRA张量、43,646,976个参数均有限，252个LoRA B张量均非零。云端报告保存重载相等；本机未再次加载完整8B模型重验。',
               '- 24题的输入哈希已用真实Qwen3 Tokenizer从本地冻结提示词重新计算；新旧两轮、两模型逐题一致。模型目录/配置、聊天模板、依赖和512生成设置一致，基础模型回答也24份逐字相同。',
               '- 下载数据清单、质量报告、Tokenizer报告与本地82条版本一致；云端校验清单与原交付ZIP一致。包内没有全部训练脚本或逐样本训练轨迹，不能独立重建实际执行代码与抽样顺序；基础模型全部权重也未重新哈希。', '',
               '## 训练覆盖与下一步', '',
               'trainer_state.json记录epoch=0.487804878。结合单GPU、batch=1、梯度累积4和10步，当前约进行了40次样例呈现，只到82条的一半左右；82条全部通过标签检查，不表示82条全部完成了参数更新。覆盖不足是下一轮可检验的因素，不能认定是所有剩余错误的唯一原因。', '',
               '建议保留这次10步适配器，沿用现有82条、基础模型与提示词，先做覆盖完整一轮数据的受控实验，再用相同512验证设置复查遗漏、部分可答和历史追问。现有入口限制最多20步，完整一轮按当前批量约需21个优化器步；需要先扩展入口并检查最后不足4条的累积处理，不能直接向现有入口传21步。具体方案见《下一步实验方案.md》，当前未修改训练入口、未启动新训练。', '',
               '## 评估范围', '',
               '这里只评估固定证据问答，未做完整RAG检索、服务流式、并发或部署验证。厦门验证题已经指导数据修订，本轮属于迭代回归；同一助手参与编写与审核，24题共享5份短文档，不构成独立最终泛化结论。原文未做外部实时核验，审核的是对可见证据的忠实程度。张家界测试集未运行预测。', '',
               '## 文件', '',
               '- `逐题对照.md`：24题的基础模型、旧/新适配器、参考答案、全部证据和审核说明。',
               '- `assistant_reviews.json`：本轮48份助手审核记录；原始待人工评分表未改写。',
               '- `assistant_summary.json`：分项汇总与变化清单。',
               '- `artifact_verification.json`：训练、数据、输入哈希与适配器文件核验。',
               '- `下一步实验方案.md`：仅下一步受控实验的方案。', '']
    (OUT / '结果审核报告.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(dict(status=summary['status'], inherited=30, new_reviews=18,
                          previous_score=summary['previous_adapter_mean_total'], current_score=model_scores['adapter']['mean_total'],
                          higher=len(summary['adapter_vs_previous_rubric']['higher_ids']), tied=len(summary['adapter_vs_previous_rubric']['tied_ids']),
                          lower=len(summary['adapter_vs_previous_rubric']['lower_ids'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
