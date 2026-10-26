"""Record prompt comparison review separately from untouched pending human forms."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = ROOT / 'artifacts/prompt_v2_validation_original'
RESULT = RAW / 'eval_outputs/prompt_v2_validation'
VARIANTS = ('original_prompt', 'revised_prompt')
DIMENSIONS = ['faithfulness', 'completeness', 'citation_support', 'answerability', 'constraints_format']
NAMES = dict(faithfulness='事实忠实', completeness='完整性', citation_support='引用支持', answerability='回答边界', constraints_format='约束与表达')

# Same rubric as the previous epoch1 review. Changed answers were read against
# the complete visible evidence, without treating a legal ID as semantic support.
ASSESSMENTS = {
    2: ([2, 2, 2, 2, 2], '仍说明不能直接步行、必须轮渡，并保留码头、船票、候船、乘船、步行入岛各环节，[1]支持，未退回之前的流程漏答。'),
    3: ([2, 2, 2, 2, 2], '点对点打车及行李/多人/时间条件仍受[1]支持，调整措辞没有改变事实或新增平台价格。'),
    5: ([2, 2, 2, 2, 2], '本次改善：只把沙茶面安排简餐、姜母鸭安排地方正餐，删除了把海鲜的滨海休闲场景迁给姜母鸭的扩写；两个结论均受[1]支持。'),
    6: ([2, 2, 2, 2, 2], '氛围收益及搬运行李/上岛/返程复杂度都保留，均来自唯一[1]；第二段仍可补就近引用，未因新增规则就自动认为引用位置已经改善。'),
    7: ([2, 2, 2, 2, 2], '海边生活感、沙滩、骑行、海边散步和餐饮定位与[1]一致，没有把区域建议变成具体酒店服务承诺。'),
    8: ([2, 2, 2, 2, 2], '缩短为鼓浪屿补充、本岛系统觅食，仍直接回应是否把全部用餐安排岛上的问题。没有要求列全食物和街区，删去背景不属于核心漏答。'),
    9: ([2, 2, 2, 2, 2], '住宿片区及三类码头核对仍分别由[1][2]支持。没有复述全部调整提醒，不影响问题主要需求。'),
    10: ([2, 2, 2, 2, 2], '老城、鼓浪屿完整时段与第三天海边仍构成清楚顺序，全部由[1]支持；删去独立背景解释后未引用[2]不构成错引或漏答。自动缺引提示属于冗余参考未用。'),
    11: ([2, 2, 2, 2, 2], '交通组合及夜间重便利的住宿区域都有对应依据；“比海边区域更稳妥”是对[2]便利/慢游取向的合理比较，没有新增设施事实。'),
    12: ([2, 2, 2, 2, 2], '住一晚慢游与行李/上岛成本的权衡、步行为主和不宜大件高频往返都保留，引用对应，未重现寄存臆测。'),
    13: ([2, 2, 2, 2, 2], '仍优先老城并说明海边可选；第二段[2]把吃喝上岛便利和海边散步松弛区分开。第一段“海边住宿可选”也应更就近标[2]，但整体两段来源足以支持取舍，未构成新的事实错误。'),
    14: ([1, 2, 1, 2, 1], '目标问题未解决：第一句两天主线与[1]对齐，但第二句“如果只有2天，第3天可以缩减”仍混用天数条件，接着又把海边点作为收尾。[2]原意是三天可安排、两天可缩减这部分，当前没有清楚区分，仍可能把可删内容加回两天行程。沿用上轮轻微忠实/引用/约束问题的分数。'),
    15: ([2, 2, 2, 2, 2], '短住优先厦门站/市中心、重换乘城市功能的建议与整体证据一致。第二段交通和城市功能具体来自[1]，引用可以更精确；与原版相似，不因重复描述或省略长住比较强行扣分。'),
    17: ([0, 2, 0, 2, 2], '关键退步：把“住鼓浪屿一晚”说成能增加老城散步、轮渡上岛和夜间活动的便利，并挂[1]。但[1]只讲可选住岛和氛围，相关便利在[2]明确属于中山路/轮渡片区，主体与来源均被混用。后两段仍推荐本岛片区，所以完整性保留；这条错误却正影响用户重视便利的住宿判断，按关键忠实性及语义错引计，而非普通措辞问题。'),
    18: ([2, 2, 2, 2, 2], '仍纠正直接步行的历史错误，补齐上岛流程和码头核对，事实都在两段证据内。后段“轮渡必要”更宜标[1]、核对事项标[2]，属于引用就近性优化，没有引入017那种错误主体事实。'),
    20: ([2, 2, 2, 2, 2], '本次改善：直接说资料没有今晚300元内双床房实时库存，不能确认是否可订，再给片区适用人群；预算内是否可订的未知边界已经明确，没有声称查过房态。[1]可移到已知片区定位后，但唯一资料与结论映射可辨。'),
    21: ([2, 2, 2, 2, 2], '仍明确无法确认指定日期下雨与最高温，指向实时天气预报，没有编造数据或混用景区渠道。'),
    22: ([2, 2, 2, 2, 2], '无资料时明确无法确认宠物政策和押金，没有补造事实；删去了联系酒店的操作建议，实用性可优化，但用户没有额外要求查询渠道，沿用语义审核口径不单独扣分。'),
    23: ([2, 2, 2, 2, 2], '仍只用[2]支持住宿方向，生活/吃饭/上岛组织的描述对应，没有被干扰片段带偏。'),
    24: ([2, 2, 2, 2, 2], '仍正确拒答无资料的酒店准确房价；删去官网/平台查询建议使后续行动较少，但没有编造价格或用小吃作依据，保留事实与边界评分。'),
}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def ahash(answer):
    return hashlib.sha256(answer.encode('utf-8')).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    verification = load(OUT / 'artifact_verification.json')
    assert verification['status'] == 'PROMPT_V2_ARTIFACT_VERIFICATION_OK'
    source = load(ROOT / 'reviews/revision_v2_epoch1_512/assistant_reviews.json')
    prior = {r['id']: r for r in source['reviews'] if r['model'] == 'adapter'}
    examples = rows(ROOT / 'data/eval_v1/validation.jsonl')
    ps = {v: rows(RESULT / f'{v}.predictions.jsonl') for v in VARIANTS}
    reviews, inherited, changed = [], [], []
    for index, e in enumerate(examples, 1):
        before = ps['original_prompt'][index - 1]
        assert prior[e['id']]['prediction_sha256'] == ahash(before['answer'])
        for variant in VARIANTS:
            p = ps[variant][index - 1]
            assert p['id'] == e['id'] and p['payload_sha256'] == before['payload_sha256']
            if variant == 'original_prompt' or p['answer'] == before['answer']:
                scores, reason = prior[e['id']]['scores'], prior[e['id']]['rationale']
                method = 'prior_semantic_review_reused_after_exact_question_evidence_history_answer_identity'
                inherited.append([e['id'], variant])
            else:
                assert index in ASSESSMENTS
                values, reason = ASSESSMENTS[index]
                scores = dict(zip(DIMENSIONS, values))
                method = 'changed_prompt_answer_read_against_complete_evidence'
                changed.append(e['id'])
            assert set(scores) == set(DIMENSIONS) and all(type(v) is int and 0 <= v <= 2 for v in scores.values())
            reviews.append(dict(id=e['id'], variant=variant, scores=scores, total=sum(scores.values()), rationale=reason,
                review_method=method, prediction_sha256=ahash(p['answer']), human_signoff=False, status='assistant_reviewed_user_pending'))
    assert len(changed) == len(ASSESSMENTS) == 20 and len(inherited) == 28
    paired = {e['id']: {r['variant']: r for r in reviews if r['id'] == e['id']} for e in examples}
    deltas = {e['id']: paired[e['id']]['revised_prompt']['total'] - paired[e['id']]['original_prompt']['total'] for e in examples}
    metrics = {}
    for variant in VARIANTS:
        items = [r for r in reviews if r['variant'] == variant]
        preds = ps[variant]
        metrics[variant] = dict(mean_total=sum(r['total'] for r in items) / 24,
            mean_scores={d: sum(r['scores'][d] for r in items) / 24 for d in DIMENSIONS},
            mean_output_tokens=sum(p['generated_tokens'] for p in preds) / 24,
            mean_generation_seconds=sum(p['generation_seconds'] for p in preds) / 24,
            format_flag_counts={k: sum(bool(p['format_flags'][k]) for p in preds) for k in preds[0]['format_flags']})
    summary = dict(status='PROMPT_V2_ASSISTANT_REVIEW_COMPLETE', reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
        assessor='Codex assistant; same assistant involved in data/rule authoring; not independent human review or a blind assessment',
        human_signoff=False, score_scale='Five dimensions 0-2, total 0-10; auxiliary rubric score, not accuracy',
        archive_sha256=verification['archive_sha256'], adapter_sha256=verification['adapter_sha256'],
        original_control_reproduced=24, paired_payloads_verified=24, same_answers=4, changed_answers=20,
        newly_source_reviewed=20, prior_semantic_reviews_reused=28, review_count=48,
        variants=metrics, paired_rubric=dict(higher_ids=[k for k, d in deltas.items() if d > 0],
            tied_ids=[k for k, d in deltas.items() if d == 0], lower_ids=[k for k, d in deltas.items() if d < 0]),
        target_issues={'005': 'subject_mixing_removed', '014': 'day_condition_confusion_remains', '020': 'unknown_booking_availability_boundary_improved'},
        critical_regressions={'017': 'island_overnight_stay_inherits_mainland_neighborhood_convenience_and_wrong_citation'},
        automatic_missing_reference_010_is_semantically_benign=True,
        recommendation='Keep original prompt for the existing epoch1 candidate; do not promote prompt_v2 wholesale based on this comparison.',
        candidate_model_or_prompt_changed_by_this_review=False, test_predictions_generated=False,
        limitations=['Validation has repeatedly guided rules and training; it is not independent final evaluation.',
            'Only 24 synthetic questions from one destination share five short documents.',
            'Same assistant authored material and reviewed outputs; human signoff remains pending.',
            'Identical revised answers reuse semantic assessment based on exact question/evidence/history/answer, not on identical prompt tokens; revised prefix tokens differ by design.',
            'Small rubric-score differences are not statistical evidence or a general benchmark.',
            'Latency is uncontrolled and answer lengths differ; no speed or cost advantage claim.',
            'No adapter weights are in this result archive; identity is bound to the previously downloaded weight file and recorded hashes.'])
    assert len(summary['paired_rubric']['higher_ids']) == 2 and len(summary['paired_rubric']['lower_ids']) == 1
    assert abs(metrics['original_prompt']['mean_total'] - 9.75) < 1e-8
    assert abs(metrics['revised_prompt']['mean_total'] - 233 / 24) < 1e-8
    save('assistant_reviews.json', dict(summary=summary, reviews=reviews))
    save('assistant_summary.json', summary)

    detail = ['# 固定完整1轮适配器：原提示词与修订提示词审核', '',
        '审核主体为助手，非独立人工、非盲审。原始48项人工评分表保持pending。每题同时查看问题、历史、参考答案与完整证据。', '',
        '语义量表保持不变；不因修订规则写得更严格就改变同一回答的评分口径，也不把自动缺引提示当作语义错误。', '']
    for index, e in enumerate(examples):
        detail += [f"## {e['id']} · {e['task_type']}", '', '问题：' + e['question'], '']
        if e['history']:
            detail += ['历史：' + json.dumps(e['history'], ensure_ascii=False), '']
        detail += ['参考答案（仅供审核，不输入模型）：', '', e['reference_answer'], '']
        for variant in VARIANTS:
            r = paired[e['id']][variant]
            detail += [f"### {variant} · {r['total']}/10", '', ps[variant][index]['answer'], '',
                '审核：' + r['rationale'], '', '分项：' + json.dumps(r['scores'], ensure_ascii=False), '',
                '自动提示：' + json.dumps(ps[variant][index]['format_flags'], ensure_ascii=False), '']
        for c in e['contexts']:
            detail += [f"证据[{c['reference_index']}] {c['relative_path']} L{c['line_start']}-{c['line_end']}", '', '```text', c['content'], '```', '']
        if not e['contexts']:
            detail += ['没有检索片段。', '']
    (OUT / '逐题对照.md').write_text('\n'.join(detail), encoding='utf-8')

    report = ['# 提示词v2对照审核：局部改善，新增关键错配', '',
        '审核日期：2026-10-24（Asia/Shanghai）。审核主体为助手，非独立人工、非盲审；业务验收仍待完成。', '',
        '## 结论与建议', '',
        '不建议整体采用本次修订提示词。它修复005的食物主体混用、改善020实时房态边界，但014条件混用仍在，并使原本正确的017出现新的住宿便利性主体混用和错误引用。建议保留完整1轮适配器与原提示词作为下一步最终评估候选，停止依据这24题反复堆叠规则。', '',
        '原提示词24份回答全部复现前轮结果，模型身份、生成设置和问题/证据/历史载荷已核对，控制组成立。两组有4份回答相同、20份改变；20份改变回答已重新对照证据审核，另外28份语义审核在问题/证据/历史/回答一致后沿用，包括24份原组与4份修订组。修订组输入前缀不同，未声称它们的完整输入Token与原组相同。', '',
        '## 三项目标与新增退步', '',
        '| 题号 | 修订后表现 | 判断 |', '|---|---|---|',
        '| 005 食物主体 | 删除姜母鸭对应滨海场景和搭配海鲜的扩写，简餐/正餐各自对应 | 改善 |',
        '| 014 天数条件 | 仍说“如果只有2天，第3天可以缩减”，随后加入海边收尾 | 条件混用未解决 |',
        '| 020 实时房态 | 明确资料没有今晚预算内双床房库存，不能确认是否可订，再答片区定位 | 改善 |',
        '| 017 历史住宿追问 | 将住鼓浪屿一晚说成增加老城散步、轮渡上岛、夜间活动的便利，错误标[1] | 新的关键退步 |', '',
        '### 017为什么不是单纯的措辞问题', '',
        '修订回答第一段把“住一晚”与“增加老城散步、轮渡上岛和夜间活动的便利性”相连并标[1]。[1]只说明住鼓浪屿可选、住一晚有氛围；老城散步/轮渡/夜间活动的便利在[2]属于中山路与轮渡片区。错误迁移了住宿选择所依赖的关键属性，直接影响重视出入便利的用户。后两段仍推荐本岛片区，不能消除前段的错误理由。', '',
        '本次新增通用规则本来要求主体对应，生成却仍出现该类错配，说明规则文本正确不等于模型会稳定遵循。也不能把005的修复推广为同类问题已经解决。', '',
        '## 其他题与引用', '',
        '- 002上岛流程与010三天行程仍完整，未出现早前10步版本的核心漏答。',
        '- 012行李取舍、018历史纠错、019已知清单与未知轮渡信息保持正确。',
        '- 010出现自动缺少[2]提示，但三天安排完整由[1]支撑，当前回答没有复述需要[2]的独立背景理由，不按实质错引扣分。',
        '- 006/015/019等引用就近性仍可优化；022/024删去了查询渠道建议，实用性略减，但没有编造事实，按原语义量表不另扣分。', '',
        '## 辅助量表与自动观察', '',
        '| 指标 | 原提示词 | 修订提示词 |', '|---|---:|---:|']
    for title, fn in [('空回答', lambda m: m['format_flag_counts']['empty_answer']),
                      ('触及输出上限', lambda m: m['format_flag_counts']['hit_output_limit']),
                      ('越界引用题数', lambda m: m['format_flag_counts']['invalid_reference_ids']),
                      ('自动缺引提示题数', lambda m: m['format_flag_counts']['missing_gold_reference_ids'])]:
        report.append(f"| {title} | {fn(metrics['original_prompt'])} | {fn(metrics['revised_prompt'])} |")
    report += [f"| 平均输出Token | {metrics['original_prompt']['mean_output_tokens']:.3f} | {metrics['revised_prompt']['mean_output_tokens']:.3f} |",
        f"| 平均单题生成秒数 | {metrics['original_prompt']['mean_generation_seconds']:.3f} | {metrics['revised_prompt']['mean_generation_seconds']:.3f} |", '',
        '单次时延不作为性能结论，两组输出长度不同且运行负载未受控；格式提示也无法发现017的语义错配。', '',
        '| 助手量表维度 | 原提示词 | 修订提示词 |', '|---|---:|---:|']
    for d in DIMENSIONS:
        report.append(f"| {NAMES[d]}（0–2） | {metrics['original_prompt']['mean_scores'][d]:.3f} | {metrics['revised_prompt']['mean_scores'][d]:.3f} |")
    report += [f"| 总分（0–10） | {metrics['original_prompt']['mean_total']:.3f} | {metrics['revised_prompt']['mean_total']:.3f} |", '',
        '按该助手量表，修订组2题提高、1题下降、21题持平。9.750与9.708的微小平均差异不作统计显著性或总体能力高低的结论；不采用修订版的主要理由是017新增了关键错误，而不是机械比较小数。分数不是准确率，也不是独立人工验收。', '',
        '## 身份与输入核验', '',
        '- 适配器保持完整1轮版本，哈希916558566d8ed3bae3d479e7c181da433077fa35931dc916e66e5a4499e262f9；本次包不含权重，已与上轮下载的实际权重文件重新哈希核对。',
        '- 模板、实验清单、运行源码与交付包逐文件一致；原始验证文件与前轮基线未改。',
        '- 48份实际输入Token已用真实Qwen3 Tokenizer重算一致，48份输出Token解码也与保存回答一致；两组24题的问题/证据/历史载荷相同。最长输入562/788 Token，均留足512输出空间，无截断。',
        '- 原提示词24份回答与前轮逐字相同；两组非思考、NF4/BF16/SDPA、贪心、512设置与同一适配器实例一致。',
        '- 未执行训练，未生成张家界预测。基础模型全部权重未重新哈希，也未在本机重复加载完整8B模型。', '',
        '## 下一步', '',
        '保留原提示词+完整1轮适配器的候选组合，已知005主体、014条件、020边界残留继续明示，不把候选记录当作这些问题已解决。候选建议见 `candidate_freeze_proposal.json`，最终评估顺序见《下一步建议.md》。本次审核未替换模板、未部署模型，也未把助手评分写入原始人工表。', '',
        '厦门24题已反复用于训练/规则迭代，不能充当独立最终测试；同一助手参与素材编写与审核。下一步应在固定候选和审核口径后，使用仍冻结的张家界作一次最终评估，再判断是否进入服务和RAG验证。', '']
    (OUT / '结果审核报告.md').write_text('\n'.join(report), encoding='utf-8')
    spec = load(ROOT / 'prompt_pipeline/experiment_manifest.json')
    save('candidate_freeze_proposal.json', dict(status='assistant_recommended_candidate_for_final_evaluation', human_signoff=False,
        selected_prompt_variant_recommendation='original_prompt', model='Qwen3-8B with revision_v2 epoch1 LoRA',
        cloud_adapter_dir='outputs/revision_v2_epoch1/adapter', adapter_sha256=spec['adapter_sha256'],
        adapter_config_sha256=spec['adapter_config_sha256'], base_model_config_sha256=spec['base_model_config_sha256'],
        original_business_template_sha256=spec['project_files_sha256']['data/eval_v1/prompt_template.txt'],
        chat_template_sha256=spec['chat_template_sha256'], eval_manifest_sha256=spec['eval_manifest_sha256'],
        generation=spec['generation'], known_validation_residuals=['005 subject/attribute mixing', '014 day-condition mixing', '020 realtime availability/price boundary wording'],
        rejected_candidate_template_sha256=spec['project_files_sha256']['prompt_pipeline/prompt_template_v2.txt'],
        selection_reason='Revised rules improve 005/020 but introduce a critical 017 attribution/citation regression; 014 remains.',
        production_acceptance_claimed=False, final_test_predictions_generated=False))
    print(json.dumps(dict(status=summary['status'], control_reproduced=24, changed_reviewed=20, inherited=28,
        original_score=metrics['original_prompt']['mean_total'], revised_score=metrics['revised_prompt']['mean_total'],
        higher=2, lower=1, tied=21, recommendation='retain_original_prompt'), ensure_ascii=False))


if __name__ == '__main__':
    main()
