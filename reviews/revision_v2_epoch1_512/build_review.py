"""Assistant evidence review of the full-epoch run; raw human forms stay pending."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = ROOT / 'artifacts/revision_v2_epoch1_original'
NEW = RAW / 'eval_outputs/revision_v2_epoch1_512'
PREVIOUS = ROOT / 'artifacts/revision_v2_original/eval_outputs/revision_v2_512'
PILOT = ROOT / 'artifacts/validation_round2_512_original'
DIMENSIONS = ['faithfulness', 'completeness', 'citation_support', 'answerability', 'constraints_format']
NAMES = dict(faithfulness='事实忠实', completeness='完整性', citation_support='引用支持', answerability='回答边界', constraints_format='约束与表达')

# Scores follow the existing five-dimension rubric, with reasons recorded below.
# No deduction merely for a redundant reference ID omitted or harmless punctuation.
ASSESSMENTS = {
    2: ([2, 2, 2, 2, 2], '已恢复10步时漏掉的上岛环节：不能直接步行、需要轮渡，以及码头、船票、候船、乘船、步行入岛均覆盖，[1]支持。'),
    3: ([2, 2, 2, 2, 2], '点对点打车直接回应多人、行李与不想研究换乘，距离相对可控等信息也来自[1]，没有扩写票价或具体平台。'),
    4: ([2, 2, 2, 2, 2], '保留整岛建筑、坡道、小巷、氛围和步行体验的重点，并建议留主要时段慢逛；虽省略“不是单个地标”的原句，核心对比已回答。'),
    5: ([1, 2, 1, 2, 2], '新出现的轻微主体混用：简餐选沙茶面、正餐选姜母鸭正确；但将姜母鸭进一步说成“适合搭配海鲜或滨海休闲场景”。原文把滨海/海边休闲场景赋给海鲜，未给出姜母鸭的该限定或搭配建议，句末[1]不能完整支撑这部分扩写。这里不判断现实中这样搭配是否可行。'),
    6: ([2, 2, 2, 2, 2], '保持已修复的两面回答：氛围收益与行李搬运、上岛组织、返程复杂度均来自同一[1]。第二段可补就近[1]，但唯一资料已标明且不存在错误来源，不把标注位置优化当作语义错引。'),
    7: ([2, 2, 2, 2, 2], '用海边休闲与城市慢节奏的住宿定位回应看海、骑行和散步偏好，沙滩/餐饮等描述均受[1]支持。'),
    8: ([2, 2, 2, 2, 2], '明确鼓浪屿补充、本岛系统觅食，并补出中山路、八市、老城主线；海边休闲定位也在同一[1]中，增加篇幅未添加新事实。'),
    9: ([2, 2, 2, 2, 2], '住宿方向及三类码头核对均清楚，分别由[1][2]支持。没有再次逐字列出航线/班次，但已回应主要上岛前核对事项，不要求复述片段所有事实。'),
    10: ([2, 2, 2, 2, 2], '已恢复10步时漏掉的第三天海边安排；第一天老城、鼓浪屿完整时段、第三天海边形成三天顺序，鼓浪屿虽未写“第二天”字样但位置已明确。分开本岛和岛上节奏的理由受[2]支持。'),
    11: ([2, 2, 2, 2, 2], '地铁加网约车与夜间中山路/轮渡片区均来自对应片段；住宿方向可从问题语境理解，符合不自驾和夜间便利要求。'),
    12: ([2, 2, 2, 2, 2], '已说清慢游住一晚与行李/上岛成本较高的权衡，并说明步行为主、不适合带大件高频往返，分别引用[1][2]；没有再补造寄存服务，也不再有“住岛外再安排一晚”的歧义。'),
    13: ([2, 2, 2, 2, 2], '按觅食和边走边逛选老城，并说明海边松弛的替代方向，两个来源分别对应。旧答只引用[1]也足够支持其更窄的结论，因此不能把本轮补齐[2]当作自动证明旧答错误。'),
    14: ([1, 2, 1, 2, 1], '条件混用仍在：第一段在“2天主线”后接环岛路/曾厝垵/厦大周边收尾并挂[1]，但[1]只有本岛老城+鼓浪屿，[2]把这些收尾选项置于第三天。第二段说两天应缩减，方向正确，却未消除前段给两天追加收尾的歧义；未清楚落实不赶行程的取舍。'),
    15: ([2, 2, 2, 2, 2], '厦门站、市中心、换乘便利和停留长短的取舍均在给定资料中。表达“短时短住”略重复，第二段交通/城市功能也参考了[1]，可让引用更就近，但主结论与两份来源没有冲突。'),
    16: ([2, 2, 2, 2, 2], '删去了10步回答中无当前依据的“公交是厦门的特色”，保留预算、时间弹性与热门片区的正确适用条件，[1]支持。'),
    17: ([2, 2, 2, 2, 2], '关键改善：不再编造岛上住宿稀缺或缺超市/餐厅/洗衣店，而是用[2]支持中山路/轮渡的便利，再用[1]说明不是必须住岛上。住一晚的体验措辞是对[1]的合理概括，不涉及具体设施或服务承诺。'),
    18: ([2, 2, 2, 2, 2], '更明确地纠正历史中的步行上岛错误，说明轮渡必要性并保留三类码头核对，[1][2]支持。本题未要求列全购票候船流程，不照搬002的完整性要求扣分。'),
    19: ([2, 2, 2, 2, 2], '已同时说明末班时间/票价未提供、需查最新信息，并给出已有三类码头核对事项；不再误拒答或省略未知边界。[1]标在资料缺失句后，建议移到核对清单后使映射更清楚；当前只有这一个片段且清单确实来自它，未判为实质错引。'),
    20: ([2, 2, 2, 1, 2], '边界明显改善：给出片区适用人群和便利，并明确不能保证具体房态，没有承诺有房或编造价格。但“如果今晚查不到”仍是条件式说法，未直接说明当前资料缺少实时库存与价格、无法确认300元内双床房；价格边界尚不够明确。'),
    21: ([2, 2, 2, 2, 2], '正确说明缺少指定日期天气/温度，指向气象预报，较旧通用景区公告更贴合业务；没有编造天气数值。按原口径，渠道针对性改进不额外增加满分之上的分数。'),
    22: ([2, 2, 2, 2, 2], '没有资料时明确无法确认宠物政策与押金，并建议联系酒店；不补造收费或引用，渠道也比旧答更贴切。'),
    23: ([2, 2, 2, 2, 2], '仍只选直接支持住宿方向的[2]，没有被小吃与海边干扰片段带偏，事实与引用一致。'),
    24: ([2, 2, 2, 2, 2], '对准确酒店房价正确拒答，并指向实时预订平台/酒店官网；没有把小吃资料当成房价依据，也未添加数字。'),
}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def answer_hash(answer):
    return hashlib.sha256(answer.encode('utf-8')).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    verification = load(OUT / 'artifact_verification.json')
    assert verification['status'] == 'EPOCH1_ARTIFACT_VERIFICATION_OK'
    examples = rows(ROOT / 'data/eval_v1/validation.jsonl')
    prior = load(ROOT / 'reviews/revision_v2_512/assistant_reviews.json')
    pilot_reviews = load(ROOT / 'reviews/validation_round2_512/assistant_reviews.json')
    indexed_prior = {(r['id'], r['model']): r for r in prior['reviews']}
    predictions = {k: rows(NEW / f'{k}.predictions.jsonl') for k in ('base', 'adapter')}
    previous = {k: rows(PREVIOUS / f'{k}.predictions.jsonl') for k in ('base', 'adapter')}
    pilot = rows(PILOT / 'adapter.predictions.jsonl')
    reviews, inherited, changed = [], [], []
    for index, example in enumerate(examples, 1):
        for label in ('base', 'adapter'):
            p, old = predictions[label][index - 1], previous[label][index - 1]
            assert example['id'] == p['id'] == old['id'] and p['input_sha256'] == old['input_sha256']
            previous_review = indexed_prior[(example['id'], label)]
            assert previous_review['prediction_sha256'] == answer_hash(old['answer'])
            if p['answer'] == old['answer']:
                score, reason = previous_review['scores'], previous_review['rationale']
                method = 'previous_review_reused_after_exact_input_answer_and_evidence_identity'
                inherited.append([example['id'], label])
            else:
                assert label == 'adapter' and index in ASSESSMENTS
                values, reason = ASSESSMENTS[index]
                score = dict(zip(DIMENSIONS, values))
                method = 'changed_answer_source_reviewed_in_full_epoch_round'
                changed.append(example['id'])
            assert set(score) == set(DIMENSIONS) and all(type(v) is int and 0 <= v <= 2 for v in score.values())
            reviews.append(dict(id=example['id'], model=label, scores=score, total=sum(score.values()), rationale=reason,
                review_method=method, prediction_sha256=answer_hash(p['answer']), human_signoff=False,
                status='assistant_reviewed_user_pending'))
    assert len(reviews) == 48 and len(changed) == len(ASSESSMENTS) == 23 and len(inherited) == 25
    paired = {e['id']: {r['model']: r for r in reviews if r['id'] == e['id']} for e in examples}
    deltas = {e['id']: paired[e['id']]['adapter']['total'] - indexed_prior[(e['id'], 'adapter')]['total'] for e in examples}
    scores = {}
    for label in ('base', 'adapter'):
        items = [r for r in reviews if r['model'] == label]
        scores[label] = dict(mean_scores={d: sum(r['scores'][d] for r in items) / 24 for d in DIMENSIONS}, mean_total=sum(r['total'] for r in items) / 24)
    stats = {}
    for name, ps in [('pilot_50rows_10steps', pilot), ('revision_82rows_10steps', previous['adapter']), ('revision_82rows_epoch1', predictions['adapter'])]:
        stats[name] = dict(mean_output_tokens=sum(p['generated_tokens'] for p in ps) / 24,
            mean_generation_seconds=sum(p['generation_seconds'] for p in ps) / 24,
            empty_answers=sum(not bool(p['answer'].strip()) for p in ps),
            output_limit_count=sum(p['hit_output_limit'] for p in ps),
            invalid_reference_count=sum(bool(p['format_flags']['invalid_reference_ids']) for p in ps),
            missing_gold_reference_count=sum(bool(p['format_flags']['missing_gold_reference_ids']) for p in ps),
            image_url_or_placeholder_count=sum(bool(re.search(r'https?://|<图片URL', p['answer'])) for p in ps))
    summary = dict(status='EPOCH1_ASSISTANT_REVIEW_COMPLETE', reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
        assessor='Codex assistant; same assistant involved in data authoring; not independent human review or blind assessment',
        human_signoff=False, score_scale='Five dimensions each 0-2, total 0-10; auxiliary rubric scores, not accuracy',
        archive_sha256=verification['archive_sha256'], adapter_sha256=verification['adapter_sha256'],
        review_count=48, changed_adapter_answers_reviewed=23, inherited_after_identity_checks=25,
        base_answers_identical=24, adapter_answers_identical=1, model_scores=scores, adapter_observations=stats,
        score_history=dict(pilot_50rows_10steps=pilot_reviews['summary']['models']['adapter'],
                           revision_82rows_10steps=prior['summary']['models']['adapter'], revision_82rows_epoch1=scores['adapter']),
        paired_rubric_vs_82row_10steps=dict(higher_ids=[k for k, v in deltas.items() if v > 0],
            tied_ids=[k for k, v in deltas.items() if v == 0], lower_ids=[k for k, v in deltas.items() if v < 0]),
        original_failure_followup={'006': 'benefits_and_costs_preserved', '012': 'tradeoff_clarified_no_unverified_storage',
            '017': 'unsupported_facility_claims_removed', '019': 'known_checklist_and_unknown_times_prices_both_answered',
            '020': 'room_availability_boundary_improved_but_realtime_price_and_no_data_statement_remain_weak'},
        prior_clear_omission_regressions_recovered=['travel-eval-validation-002', 'travel-eval-validation-010'],
        residual_source_alignment_issues=['travel-eval-validation-005', 'travel-eval-validation-014'],
        realtime_boundary_needs_clarification=['travel-eval-validation-020'],
        citation_locality_editorial_notes=['travel-eval-validation-006', 'travel-eval-validation-015', 'travel-eval-validation-019'],
        test_predictions_generated_or_used=False,
        limitations=['One validation destination and 24 synthetic questions sharing five documents; no broad tourism accuracy claim.',
                     'Validation has repeatedly guided iteration; this is regression evidence, not an independent final test.',
                     'Same assistant authored samples and assessed answers; independent human signoff remains pending.',
                     'Training mode and coverage instrumentation changed to epoch mode; same reported settings do not establish bitwise identical training or causal gains.',
                     'Full base weights were not rehashed and full 8B reload was not repeated locally.',
                     'Timing is from single uncontrolled runs with different answer lengths; no throughput or latency conclusion.',
                     'Review checks support in visible source snapshots, not external real-time factual truth.'])
    assert len(summary['paired_rubric_vs_82row_10steps']['higher_ids']) == 7
    assert len(summary['paired_rubric_vs_82row_10steps']['lower_ids']) == 2
    assert abs(scores['adapter']['mean_total'] - 9.75) < 1e-8
    save('assistant_reviews.json', dict(summary=summary, reviews=reviews))
    save('assistant_summary.json', summary)

    detail = ['# 24题对照：50条10步、82条10步、82条完整1轮', '',
              '审核主体为助手，非独立人工、非盲审；原始待人工评分表保持pending。', '',
              '分数用于辅助整理证据审核，不是准确率。每题给出问题、历史、参考答案、三版适配器回答、本轮理由与全部证据。原始模型回答24份与前轮逐字一致，其审核沿用。', '']
    for index, e in enumerate(examples):
        current = paired[e['id']]['adapter']
        detail += [f"## {e['id']} · {e['task_type']}", '', '问题：' + e['question'], '']
        if e['history']:
            detail += ['历史：' + json.dumps(e['history'], ensure_ascii=False), '']
        detail += ['参考答案（仅供审核，未输入模型）：', '', e['reference_answer'], '',
                   '### 50条10步', '', pilot[index]['answer'], '',
                   f"### 82条10步 · {indexed_prior[(e['id'], 'adapter')]['total']}/10", '', previous['adapter'][index]['answer'], '',
                   f"### 82条完整1轮 · {current['total']}/10", '', predictions['adapter'][index]['answer'], '',
                   '审核：' + current['rationale'], '', '分项：' + json.dumps(current['scores'], ensure_ascii=False), '',
                   '自动提示：' + json.dumps(predictions['adapter'][index]['format_flags'], ensure_ascii=False), '']
        for c in e['contexts']:
            detail += [f"证据[{c['reference_index']}] {c['relative_path']} L{c['line_start']}-{c['line_end']}", '', '```text', c['content'], '```', '']
        if not e['contexts']:
            detail += ['没有检索片段。', '']
    (OUT / '逐题对照.md').write_text('\n'.join(detail), encoding='utf-8')

    train = load(RAW / 'outputs/revision_v2_epoch1/epoch_report.json')
    report = ['# 完整1轮结果审核：覆盖已完成，主要旧问题改善', '',
        '审核日期：2026-10-24（Asia/Shanghai）。审核主体为助手，非独立人工、非盲审；业务验收仍待完成。', '',
        '## 结论', '',
        '本轮训练与覆盖核验通过，82条样例各一次、21次优化器更新已从下载轨迹重新核对。相对82条10步，回答有明显的具体改善：上岛流程和第三天行程恢复，住一晚与行李成本的取舍更清楚，历史追问不再编造岛上缺少设施，轮渡问题同时处理已知清单和未知实时信息。', '',
        '这版适合作为下一阶段的候选保留。仍需复核005的事实主体混用、014的两天/三天条件混用，以及020实时价格和资料缺失边界不够明确；目前不直接视为生产验收通过。', '',
        '## 原五类问题及两处漏答', '',
        '| 题号 | 完整1轮表现 | 判定 |', '|---|---|---|',
        '| 006 住岛优缺点 | 氛围收益与行李/上岛/返程成本都保留 | 保持已修复内容 |',
        '| 012 大箱子与住一晚 | 说明慢游和较高行李/上岛成本，提醒不适合带大件高频往返，无寄存臆测 | 本次回答修复 |',
        '| 017 历史追问 | 删除住宿稀缺、没有超市/餐厅/洗衣店等断言，用对应证据推荐中山路/轮渡 | 本次回答修复 |',
        '| 019 今日轮渡和核对清单 | 明确资料没有末班时间/票价，再给已有的三类码头核对事项 | 本次回答修复；引用位置可更清楚 |',
        '| 020 今晚房态和预算 | 补出不能保证具体房态，但仍用条件式“如果查不到”，没有直接说明缺少实时库存与价格 | 改善，边界仍需明确 |',
        '| 002 上岛流程 | 码头、船票、候船、乘船、步行入岛重新齐全 | 10步新增漏答已恢复 |',
        '| 010 三天行程 | 重新安排第三天海边，前后顺序和岛内/本岛节奏均覆盖 | 10步新增漏答已恢复 |', '',
        '## 剩余问题', '',
        '1. **005：主体混用。** 核心的沙茶面简餐、姜母鸭正餐正确，但新增“姜母鸭适合搭配海鲜或滨海休闲场景”。原文把滨海休闲赋给海鲜，不能把相邻事实自动转移给姜母鸭。这是可见证据支持问题，不是在核验现实餐饮搭配是否可行。',
        '2. **014：条件混用。** 第一段在“两天主线”后接环岛路/曾厝垵/厦大周边收尾，且引用只讲两天主线的[1]；原文的收尾选项属于第三天。第二段虽说两天应缩减，整段仍有追加行程与缩减行程的歧义。',
        '3. **020：边界表述。** “不能保证具体房态”已经改善旧答，但应直接说明当前资料没有今晚库存与实时价格，无法确认300元内双床房。现有句子没有编造有房，只是边界不够明确。',
        '4. **引用就近性。** 006第二段可补[1]；015部分交通理由还来自[1]；019建议把[1]放在核对清单后。当前没有把错误资料当依据，这些按相同口径记为表达与映射优化，不与017旧版的实质语义错引混为一类。', '',
        '## 自动提示与辅助量表', '',
        '| 指标 | 50条10步 | 82条10步 | 82条完整1轮 |', '|---|---:|---:|---:|']
    for title, key, fmt in [('空回答', 'empty_answers', 'd'), ('触及输出上限', 'output_limit_count', 'd'),
                            ('越界引用题数', 'invalid_reference_count', 'd'), ('自动缺引提示题数', 'missing_gold_reference_count', 'd'),
                            ('无来源图片URL/占位符题数', 'image_url_or_placeholder_count', 'd'),
                            ('平均输出Token', 'mean_output_tokens', '.2f'), ('平均单题生成秒数', 'mean_generation_seconds', '.3f')]:
        report.append('| ' + title + ' | ' + ' | '.join(format(v[key], fmt) for v in stats.values()) + ' |')
    report += ['', '本轮自动格式提示全部为0，仍不能看出005/014的语义问题，也不能等同于100%准确率。旧轮013/015的缺引提示本来就可能是冗余来源未引用，不把5→2→0简单当成错误题数减少。输出更完整也更长，单次生成耗时随之变化，不作速度或吞吐结论。', '',
        '| 助手量表维度 | 82条10步 | 完整1轮 |', '|---|---:|---:|']
    for d in DIMENSIONS:
        report.append(f"| {NAMES[d]}（0–2） | {prior['summary']['models']['adapter']['mean_scores'][d]:.3f} | {scores['adapter']['mean_scores'][d]:.3f} |")
    report += [f"| 总分（0–10） | {prior['summary']['models']['adapter']['mean_total']:.3f} | {scores['adapter']['mean_total']:.3f} |", '',
        '按该助手量表，7题比82条10步高、2题低、15题持平。23份改变的适配器回答已重新对照完整证据阅读；25份回答在输入/证据/文本一致后沿用旧审核，包括基础模型24份和适配器001。分数并非独立人工准确率或业务验收结论。', '',
        '## 训练、覆盖及身份核验', '',
        '- 82条各一次：原50条与新增32条全部覆盖，轨迹输入/标签指纹与冻结Token导出逐条一致。21次优化器更新未跳过，均有有限/非零梯度检查。',
        '- 最后一次更新处理2个单条批次，监督Token数168；记录中的损失归一化也使用这168个实际监督Token，实际累积长度2。',
        f"- epoch=1.0，平均训练损失{train['training']['final_training_loss']:.6f}，训练显存峰值{train['training_peak_allocated_vram_gib']:.3f} GiB。完整轮次与较低loss不能代替回答审核。",
        '- 实际适配器SHA256与训练/评测报告相符；504个BF16 LoRA张量、43,646,976个参数均有限，252个LoRA B张量非零。保存重载相等来自云端报告，本机未再加载完整8B权重。',
        '- 本轮附带的训练/评测代码与交付包哈希匹配，完整82条数据及冻结验证文件与本地版本相同；运行身份比上一轮只有部分清单更易独立核对。一个空包初始化文件未包含在结果包，不影响已核对的实质代码。',
        '- 24题输入哈希已用真实Qwen3 Tokenizer重算；新旧两轮两模型一致。基础模型24份回答逐字一致，512生成参数、模型路径/配置、模板、依赖及LoRA配置相同。LoRA目标模块列表顺序不同但模块集合相同。',
        f"- 固定配置不代表训练逐位复现：本轮前10步与上轮loss最大绝对差{verification['first_10_training_losses_max_abs_difference']:.6f}。训练代码增加了轮数模式与覆盖记录，未做多种子或严格确定性重复实验，因此不把全部变化解释为已证明的单因素因果收益。", '',
        '数据目录还保留20个未被清单引用的额外文件，符合前次解压后遗留的文件名问题；它们不在本次冻结来源/样例中，实际Token及训练覆盖均已逐条对上。本轮保留原始产物，未做清理。', '',
        '## 下一步建议', '',
        '先保留当前完整1轮适配器作为候选，不因格式提示归零就直接部署。对005、014、020和引用位置做业务复核，明确是否接受这些残留；若继续改模型或提示词，用厦门验证集迭代，张家界保持冻结。模型、提示词与审核口径确定后，再进行一次张家界最终测试，并把它作为最终评估而非继续按题补数据的依据。具体见《下一步建议.md》。', '',
        '## 范围和限制', '',
        '厦门24题已多次指导训练设计，是迭代回归集，不能作为独立最终泛化证据；这些题共享5份短文档。同一助手参与样例编写与审核，独立人工业务签字仍待完成。原始manual_reviews.json的48项继续为pending，未用助手审核冒充人工完成。', '',
        '本轮只检查固定证据问答，不包含完整RAG检索、实时数据查询、流式服务、并发和部署验证。基础模型权重未全部重新哈希，旅游资料未做外部实时核验。张家界测试集尚未运行预测。', '',
        '## 材料', '', '- `逐题对照.md`：三版适配器、问题、历史、参考答案与完整证据。',
        '- `assistant_reviews.json` / `assistant_summary.json`：48份助手审核及分项汇总。',
        '- `artifact_verification.json`：代码、数据、覆盖、输入哈希和权重核验。', '- `下一步建议.md`：候选固定、残留复核及最终测试的进入条件。', '']
    (OUT / '结果审核报告.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(dict(status=summary['status'], changed_reviewed=23, inherited=25,
        previous_score=prior['summary']['models']['adapter']['mean_total'], current_score=scores['adapter']['mean_total'],
        higher=7, lower=2, tied=15), ensure_ascii=False))


if __name__ == '__main__':
    main()
