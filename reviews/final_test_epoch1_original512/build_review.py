"""Evidence review of all 48 final-test answers; do not tune on this output."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = ROOT / 'artifacts/final_test_epoch1_original512_original'
RESULT = RAW / 'eval_outputs/final_test_epoch1_original512'
DIMENSIONS = ['faithfulness', 'completeness', 'citation_support', 'answerability', 'constraints_format']
NAMES = dict(faithfulness='事实忠实', completeness='完整性', citation_support='引用支持', answerability='回答边界', constraints_format='约束与表达')

# All entries are assistant judgements after reading each answer and its complete
# visible source, using the rubric already used for validation. No human signoff.
BASE = {
    1: ([0,2,1,2,0], '公路转场的核心结论正确；补入旅游专线/包车车程约1小时等当前片段未提供的信息，并输出不存在于证据中的图片占位符。主结论引用正确，额外时间无依据。'),
    2: ([0,2,1,2,0], '交通类型回答齐全，但新增环保车30分钟至1小时、索道15–20分钟、步行换乘5–10分钟及具体索道等未提供细节，末尾还有图片占位符。'),
    3: ([1,2,2,2,0], '点对点网约车/出租车的主建议受[1]支持；具体平台、预约和等待等扩写未由片段提供，按额外建议的轻微证据边界问题记录，不等同于已承诺某服务。图片占位符违反无图片不输出的规则。'),
    4: ([2,2,2,2,2], '明确一天不适合两个景区都完整体验，推荐按兴趣选一个；后续兼顾需取舍的条件建议未承诺两者都完整，符合[1]。'),
    5: ([0,2,0,2,1], '已覆盖地方菜方向与口味，但三下锅具体食材、山货清淡清新等结论超出片段并挂[1]；生成在512上限处中断。核心问题已回答，截断主要影响收尾，完整性不因缺参考答案逐字复述扣分。'),
    6: ([0,2,1,2,0], '晚到/天门山住市区的主建议正确；补出07:00–17:00开放时间、具体交通类型等未给信息，并生成example.com图片URL。以公告为准的限定不能使无依据时刻变成有来源。'),
    7: ([2,2,2,2,0], '减少清晨通勤与远离市区节点的取舍均有依据；旺季预订和位置优先等是一般建议，未补造分钟数或保证服务。主要缺陷为无来源图片占位符。'),
    8: ([1,2,2,2,0], '不为网红餐厅折返的核心与[1]一致；额外推测餐厅位于外围、交通不便及营业时间差异，未由当前片段支持，属轻微外扩；输出图片占位符。'),
    9: ([2,2,2,2,0], '武陵源住宿及就近用餐均正确并分别引用[1][2]；其他餐厅筛选与特殊需求建议没有作具体服务或时间保证。仍输出无来源图片占位符。'),
    10: ([0,2,0,2,0], '三天框架覆盖，但第二天后仍推荐武陵源以方便次日行程，而第三天是天门山，[2]反而提示天门山住市区；又将索道预约票建议挂[2]，出现条件与引用混用。还有图片占位符。'),
    11: ([2,2,2,2,0], '天门山市区近郊、返程方便与森林公园武陵源方向的区别正确，查询开放/门票是待核对建议，未给出未提供的值。问题为图片占位符。'),
    12: ([2,2,1,2,0], '市区正餐与夜间活动均确有可见依据，但把夜生活丰富只挂到[2]，该具体理由实际来自[1]，引用映射部分不完整；另有图片占位符。'),
    13: ([0,2,1,2,0], '四天不必不断加景点的结论与[1]一致；自行扩写大观台、玻璃栈道、天门洞、具体餐饮与票务等未给行程，引用无法支撑这些事实。还有图片占位符。'),
    14: ([1,2,1,2,0], '近景区住宿和提前接驳主要正确，但把避免清晨/傍晚出行、白天游览减少换乘的建议挂[1]，以及末班核对挂[3]，这些片段未提供相应理由；记轻微外扩和引用问题。另有图片占位符。'),
    15: ([1,2,1,2,2], '不必频繁换酒店的主要结论正确；补入张家界大剧院等未在片段中的景点和对应住区例子，并挂[2]，属于轻微事实与引用外扩。无输出截断或图片问题。'),
    16: ([2,2,2,2,2], '明确不必须赶武陵源，可以按行李与休息需求住市区；[2]支持抵达日/行李的便利方向。未用[1]的自动缺引提示本身不是错误，不要求为了凑齐编号再列全部早晚分支。'),
    17: ([0,2,0,2,0], '市区方向正确，但声称市区酒店通常有更完善的行李寄存和搬运服务并挂[2]，片段只讲城市配套与交通衔接，未提供酒店服务保证或地区服务比较。另有图片占位符。'),
    18: ([0,2,0,2,0], '纠正不同片区、不宜一天高强度串联的核心正确；将天门山轻松游、索道玻璃栈道项目、森林公园体力消耗等未给细节挂[2]，新增支持理由超出片段。另有图片占位符。'),
    19: ([2,2,2,2,2], '拒绝提供资料没有的明日开放时间和运营信息，没有生成未来时刻。[1]只含一般天气提醒，不足以预测明日状态，但当前回答并未拿它断言明天开放或关闭；不把自动distractor提示直接判成已确认错引。'),
    20: ([0,2,1,2,0], '能说明缺具体酒店与价格并给出优缺点，但以资料未列酒店推导“住宿选择有限”，还泛称部分度假酒店/民宿提供家庭房，当前证据未给这些事实。拒答主要未知信息的方向正确，仍有无依据扩写及图片占位符。'),
    21: ([2,2,2,2,2], '分别列出两份不同建议，明确不能据此保证两景区完整体验；取舍是条件建议，未伪造确定承诺，引用对应。'),
    22: ([2,2,2,2,2], '没有资料时不编造包间和最低消费，建议向商家核对，边界正确。未追问餐厅名可优化，但不是核心漏答。'),
    23: ([2,2,2,2,0], '主建议武陵源由[2]支持；附加“若以天门山为主可看市区节点”确由[3]支持，虽属本题旁支，也不能只因其被标distractor就判为语义错引。问题为图片占位符。'),
    24: ([2,2,2,2,2], '没有提供列车时刻或余票，拒答边界正确；景区公告措辞不贴合铁路，但来自现有通用模板，按之前口径不单独扣分。'),
}

ADAPTER = {
    1: ([2,2,2,2,2], '机场定位与继续公路转场均由[1]支持，没有补出车程或班次。'),
    2: ([2,2,2,2,2], '不能全按步行、环保车/索道/换乘应计入行程均来自[1]，未编造交通时间。'),
    3: ([2,2,2,2,2], '点对点打车回应行李及不想多段公交，适用节点与[1]一致，没有服务保证或价格外推。'),
    4: ([2,2,2,2,2], '明确一天不能要求两个核心景区都完整，回答直接、条件与[1]一致。'),
    5: ([2,2,2,2,2], '四类地方菜方向、重口/下饭/烟熏与整体酸辣咸香均来自[1]，没有复现基线的配料臆测或山货清淡外推。第二段可补就近引用。'),
    6: ([2,2,2,2,2], '完整说明市区适合晚到、天门山及交通衔接，内容均在[1]；没有给具体营业时间或酒店服务。'),
    7: ([2,2,2,2,2], '减少清晨通勤的优势与离市区节点更远的代价都覆盖，同一[1]支持，第二段可重复标注以更清楚。'),
    8: ([2,2,2,2,2], '明确不为网红店折返，按行程就近用餐，受[1]支持。'),
    9: ([2,2,2,2,2], '住宿和午饭两项均覆盖，分别由[1][2]支持，没有新增交通或餐厅事实。'),
    10: ([2,2,2,2,2], '抵达住宿、森林公园完整时段、天门山返程以及两类住区联动均正确复述，三天与相应条件没有混用。'),
    11: ([2,2,2,2,2], '直接选择天门山并解释市区交通衔接，森林公园在武陵源方向及不同住区出发节奏也有[2]支持；未重复完整一天的背景不算漏答。'),
    12: ([2,2,2,2,2], '市区夜间餐饮与完整地方菜的依据来自[1][2]，合并表述但两个引用共同支持主要理由。'),
    13: ([2,2,2,2,2], '按四天放缓、交通休息留余量和不天天重行程回答，来源分别对应；没有强行增加景点。'),
    14: ([2,2,2,2,2], '住宿贴近主景区、武陵源减少清晨通勤、老人孩子放慢交通并提前接驳均覆盖，[1][3][2]对应。'),
    15: ([2,2,2,2,2], '不必频繁换酒店，主住区按核心景区选择，另区景点放抵达/返程等内容均在[1][2]，未添新景点。'),
    16: ([1,2,1,2,2], '主建议可住市区符合题意；但转述[1]时把“抵达较早且次日直奔森林公园”简化成只要次日直奔就可直接去武陵源，省略了较早这一前提，扩大适用范围。第二段又将“下午才到”直接归为较晚，宜更明确按实际抵达时间判断。未引[2]本身不构成错误。'),
    17: ([2,2,2,2,2], '结合历史的天门山、返程及行李需求推荐市区，两段分别有[2][1]，没有补造寄存搬运服务。'),
    18: ([2,2,2,2,2], '用当前证据纠正历史同一区域的错误，明确两片区与不宜同一天高强度串联，[1][2]支持；可更直接承认此前说错，但核心已经纠正。'),
    19: ([2,2,2,2,2], '不编造明日开放与首班时刻，未知边界正确。天气句的范围可更清楚：原文只给一般天气对索道/观景体验的提醒，不等于明日运营公告；当前回答没有把一般提醒当成明天开放/关闭的事实。“体验”和“运营”应区分，先记措辞歧义，不将自动无关引用标记当作确定事实错误。'),
    20: ([2,1,2,1,2], '优缺点有依据，但没有回答哪家酒店、家庭房与价格，也没有明确说资料缺少今晚库存/实时房价、无法确认这些内容。属于部分可答时遗漏未知边界，不能因未编造数值就认定需求已全部覆盖。'),
    21: ([2,2,2,2,2], '回答不能保证两个景区完整、需按具体行程取舍，结论与两段可见建议一致。解释非常简略，没有分别呈现两份资料如何不同；题目问能否确定，未强制要求逐段比较，因此不按未逐字复现参考答案扣分，也不据此宣称冲突解释能力充分。'),
    22: ([2,2,2,2,2], '明确无法确认今日包间与最低消费并建议联系餐厅，没有编造事实；可补问餐厅名。'),
    23: ([2,2,2,2,2], '优先武陵源、减少通勤与适用条件均来自[2]，没有把小吃或其他片区材料错误作为当前住宿理由。'),
    24: ([2,2,2,2,2], '明确住宿资料没有高铁时刻/余票，未给出无依据数字；铁路核对渠道可更具体，但用户问能否据此给出，拒答已明确。'),
}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(v) for v in path.read_text(encoding='utf-8').splitlines() if v.strip()]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    integrity = load(OUT / 'artifact_verification.json')
    assert integrity['status'] == 'FINAL_TEST_ARTIFACT_VERIFICATION_OK'
    examples = rows(ROOT / 'data/eval_v1/test.jsonl')
    predictions = {k: rows(RESULT / f'{k}.predictions.jsonl') for k in ('base','adapter')}
    reviews = []
    for label, assessments in (('base',BASE),('adapter',ADAPTER)):
        assert set(assessments) == set(range(1,25))
        for index, (example, p) in enumerate(zip(examples, predictions[label]),1):
            assert p['id'] == example['id']
            values, reason = assessments[index]
            scores = dict(zip(DIMENSIONS,values))
            assert all(type(v) is int and 0 <= v <= 2 for v in values)
            reviews.append(dict(id=example['id'],model=label,scores=scores,total=sum(values),rationale=reason,
                prediction_sha256=hashlib.sha256(p['answer'].encode('utf-8')).hexdigest(),
                status='assistant_reviewed_user_pending',human_signoff=False,review_method='new_source_review_of_final_test_answer'))
    paired={e['id']:{r['model']:r for r in reviews if r['id']==e['id']} for e in examples}
    models={}
    for label in ('base','adapter'):
        items=[r for r in reviews if r['model']==label]; ps=predictions[label]
        tasks={}
        for task in sorted({e['task_type'] for e in examples}):
            ids={e['id'] for e in examples if e['task_type']==task}; selected=[r for r in items if r['id'] in ids]
            tasks[task]=dict(samples=len(selected),mean_total=sum(r['total'] for r in selected)/len(selected))
        models[label]=dict(mean_scores={d:sum(r['scores'][d] for r in items)/24 for d in DIMENSIONS},
            mean_total=sum(r['total'] for r in items)/24,tasks=tasks,
            mean_output_tokens=sum(p['generated_tokens'] for p in ps)/24,
            mean_generation_seconds=sum(p['generation_seconds'] for p in ps)/24,
            automatic_format_flag_counts={key:sum(bool(p['format_flags'][key]) for p in ps) for key in ps[0]['format_flags']},
            image_url_or_placeholder_sample_count=sum(bool(re.search(r'https?://|<图片URL',p['answer'])) for p in ps),
            image_url_or_placeholder_ids=[p['id'] for p in ps if re.search(r'https?://|<图片URL',p['answer'])])
    pair_stats=dict(adapter_higher=sum(v['adapter']['total']>v['base']['total'] for v in paired.values()),
                    tied=sum(v['adapter']['total']==v['base']['total'] for v in paired.values()),
                    adapter_lower=sum(v['adapter']['total']<v['base']['total'] for v in paired.values()))
    summary=dict(status='FINAL_TEST_ASSISTANT_REVIEW_COMPLETE',reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
        assessor='Codex assistant; involved in data authoring; not independent human review or blind assessment',human_signoff=False,
        score_scale='Five dimensions each 0-2, total 0-10; auxiliary rubric score, not accuracy or production acceptance',
        split='test',destination='张家界',samples_per_model=24,new_answers_reviewed=48,prior_reviews_reused=0,
        archive_sha256=integrity['archive_sha256'],adapter_sha256=integrity['candidate_adapter_sha256'],
        original_business_template_sha256=integrity['original_business_template_sha256'],models=models,
        paired_rubric=pair_stats,confirmed_adapter_review_issues={'016':'arrival-condition qualifier omitted','020':'missing realtime hotel inventory/price boundary'},
        adapter_editorial_or_capability_caveats={'019':'general weather vs tomorrow operation wording needs care; automatic label not decisive',
                                               '021':'safe conclusion but short conflict explanation; no claim of demonstrated detailed conflict resolution'},
        automatic_flag_caveats={'019':'weather context is insufficient for tomorrow operations, but no future status was asserted',
                                '023_base':'conditional Tianmenshan alternative is actually supported by context 3 despite distractor role',
                                '016_adapter':'missing context 2 is not itself the semantic issue; dropped arrival condition is'},
        test_results_used_for_data_prompt_or_checkpoint_selection=False,
        candidate_unchanged=True,production_acceptance_claimed=False,
        recommended_next_stage='Controlled model-serving and actual RAG integration evaluation; confirm implementation stack before making that new choice.',
        limitations=['24 synthetic test questions share five documents and one held-out destination; no broad tourism accuracy claim.',
            'Same assistant helped author reference material and reviewed outputs; independent business signoff remains pending.',
            'The pretest candidate was fixed; this review does not tune on test outcomes. Reusing outcomes for future tuning would require a new independent final set.',
            'Only fixed-evidence answering tested; actual retrieval, streaming, concurrency, deployment and qwen-flash API comparison remain unverified.',
            'No pre-agreed production score threshold; rubric means cannot be converted into a deployment pass rate.',
            'Timing is from single uncontrolled runs with different output lengths, not a throughput benchmark.',
            'Visible source support was assessed, not current external tourism facts.'])
    assert len(reviews)==48 and abs(models['adapter']['mean_total']-236/24)<1e-8
    save('assistant_reviews.json',dict(summary=summary,reviews=reviews));save('assistant_summary.json',summary)
    detail=['# 张家界最终测试：逐题证据审核','','同一已固定的完整1轮适配器与原提示词；审核主体为助手，非独立人工、非盲审。原始48项人工评分保持pending。','']
    for index,e in enumerate(examples):
        detail += [f"## {e['id']} · {e['task_type']}",'','问题：'+e['question'],'']
        if e['history']:detail += ['历史：'+json.dumps(e['history'],ensure_ascii=False),'']
        detail += ['参考答案（仅审核，不进入模型输入）：','',e['reference_answer'],'']
        for label in ('base','adapter'):
            r=paired[e['id']][label];p=predictions[label][index]
            detail += [f"### {label} · {r['total']}/10",'',p['answer'],'','审核：'+r['rationale'],'',
                '分项：'+json.dumps(r['scores'],ensure_ascii=False),'','自动提示：'+json.dumps(p['format_flags'],ensure_ascii=False),'']
        for c in e['contexts']:
            detail += [f"证据[{c['reference_index']}] {c['relative_path']} L{c['line_start']}-{c['line_end']}",'','```text',c['content'],'```','']
        if not e['contexts']:detail += ['没有检索片段。','']
    (OUT/'逐题对照.md').write_text('\n'.join(detail),encoding='utf-8')
    report=['# 张家界最终固定证据测试审核','','审核日期：2026-10-24（Asia/Shanghai）。助手审核，非独立人工、非盲审；业务签字仍待完成。','',
        '## 结论','',
        '最终测试技术链路已完成：24题两模型完整生成，候选适配器、原业务模板、冻结测试文件和运行代码均核验一致，48份输入Token已真实重算。微调模型在本批未用作开发选择的目的地资料上更贴近证据，明显减少基线的无依据时间、酒店服务和图片扩写。','',
        '支持把现有候选推进到受控模型服务与真实RAG集成验证。仍有016条件省略和020部分可答边界遗漏，019天气措辞、021冲突解释也需注意；本结果不能直接认定生产验收通过，更不能换算成线上准确率。测试后未修改模型、提示词或训练数据。','',
        '## 对照结果','', '| 指标 | 原始Qwen3-8B | 完整1轮适配器 |','|---|---:|---:|']
    for title,key in [('空回答','empty_answer'),('触及512输出上限','hit_output_limit'),('越界引用题数','invalid_reference_ids'),
                      ('自动无关引用提示题数','irrelevant_reference_ids'),('自动缺引提示题数','missing_gold_reference_ids')]:
        report.append(f"| {title} | {models['base']['automatic_format_flag_counts'][key]} | {models['adapter']['automatic_format_flag_counts'][key]} |")
    report += [f"| 无来源图片URL/占位符题数 | {models['base']['image_url_or_placeholder_sample_count']} | {models['adapter']['image_url_or_placeholder_sample_count']} |",
        f"| 平均输出Token | {models['base']['mean_output_tokens']:.2f} | {models['adapter']['mean_output_tokens']:.2f} |",
        f"| 平均单题生成秒数 | {models['base']['mean_generation_seconds']:.3f} | {models['adapter']['mean_generation_seconds']:.3f} |",'',
        '自动引用标记是线索，不等于错误题数。019引用的是天气提醒，没有据此断言明天开放或关闭；023基线引用的旁支[3]确实支持它附加的天门山条件建议。二者即使被标为distractor，也不能不看结论就判错。016适配器未引[2]本身不是问题，真正要看的是[1]中较早到达的条件被省略。','',
        '平均耗时不同伴随输出长度明显不同；GPU负载未受控，不作严格速度或吞吐优势结论。','',
        '## 主要改善','',
        '- 001/002：微调模型只给资料支持的转场与内部交通安排；基线额外生成约1小时、30分钟至1小时、15–20分钟等当前证据没有的时间。',
        '- 005：微调模型保留地方菜方向和已给口味；基线补造具体配料与清淡山野风味，并在输出上限处结束。',
        '- 007/009/010/014/015：住宿、交通、行程和用餐需求能依对应片段整合；微调模型没有明显新增片段外景点或票务政策。',
        '- 017：微调模型结合历史中的天门山、返程和行李需求推荐市区，未像基线那样外推酒店通常提供更完善的寄存/搬运服务。',
        '- 018：微调模型能依据当前资料纠正历史中“两景区同片区、一天顺路”的错误。',
        '- 022/024：没有资料时不编造餐厅余位、最低消费、列车时刻或余票。','',
        '## 残留与解释范围','',
        '1. **016到达条件。** 原文允许直接去武陵源的前提是抵达较早且次日进森林公园。适配器只保留次日进森林公园，扩大了适用范围；后段按下午才到建议市区，主方向可用，条件转述仍不严谨。',
        '2. **020实时房态/价格。** 只回答武陵源优缺点，没有明确交代具体酒店、今晚家庭房库存和实时价格无法确认。没有编造房价是优点，但不代表完整回答了部分可答问题。',
        '3. **019天气措辞。** 没有编造明日索道状态和时间，拒答合理；“天气影响体验”与“明日是否运营”应区分。当前记措辞需要人工复核，不把自动无关引用提示当作确定事实错误。',
        '4. **021资料冲突。** 不能保证两个景区都完整玩到的结论正确，但没有分别说明两份建议如何不同。问题未要求逐段对比，所以未按不复现参考答案扣分；这也不证明详细冲突解释能力充分。','',
        '## 助手量表（不是准确率）','', '| 维度 | 原始模型 | 微调模型 |','|---|---:|---:|']
    for d in DIMENSIONS:report.append(f"| {NAMES[d]}（0–2） | {models['base']['mean_scores'][d]:.3f} | {models['adapter']['mean_scores'][d]:.3f} |")
    report += [f"| 总分（0–10） | {models['base']['mean_total']:.3f} | {models['adapter']['mean_total']:.3f} |",'',
        f"按本量表，适配器高于基线{pair_stats['adapter_higher']}题、持平{pair_stats['tied']}题、低于基线{pair_stats['adapter_lower']}题；016是基线答法较好的实例。以上为同一助手的证据审核，不是盲审或独立人工结论，也不对外部实时事实作认证。",'',
        '| 任务 | 题数 | 原始模型均分 | 微调模型均分 |','|---|---:|---:|---:|']
    for task,m in models['base']['tasks'].items():
        report.append(f"| {task} | {m['samples']} | {m['mean_total']:.3f} | {models['adapter']['tasks'][task]['mean_total']:.3f} |")
    report += ['', '## 身份与来源核验','',
        '- 当前适配器SHA256为916558566d8ed3bae3d479e7c181da433077fa35931dc916e66e5a4499e262f9，与测试前候选、训练记录及现有下载权重一致。最终包不重复携带权重。',
        '- 原业务模板SHA256为f4f298d92c2d7d7dbb90ba3512cf95ccbd17cd537a5b0a3cd4a14748eb7a1ac0；未套用提示词v2。测试题文件与此前冻结版本相同，源码与已核验评测实现相同。',
        '- 训练20份来源与测试5份来源不重叠，82条训练问题与测试题没有规范化文本完全重复；每题两模型输入哈希相同，48份均经真实Tokenizer重算。',
        '- 原始manual_reviews.json的48项保持pending；本报告的助手评分单独保存。输出Token数量来自记录，最终包未保存输出Token IDs，因此未声称重新解码或在本机重生成48份回答。','',
        '## 下一阶段与限制','',
        '微调训练、保存重载、开发集回归、提示词对照、固定候选最终测试的技术链路已完成。下一步可做受控模型服务与真实RAG验证：先确认服务实现方案，再验证加载、流式、网络调用、真实检索证据、历史与引用、长输入、并发和失败处理。当前未选定新的推理技术栈，未改RAG接口，也未替换qwen-flash。',
        '当前只比较原始Qwen3-8B与其适配器，没有调用qwen-flash作付费基线，不据此宣称已超过现有云模型。24题共享一个目的地的5份短文档，无法代表全部旅游业务；没有预先约定生产通过分数，均分不能转成部署通过率。',
        '张家界现已用于这次最终评估。保留候选和结果；若将这些错题用于后续调模型/数据/提示词，应另准备独立最终测试，不再称本集合未见。当前审核不对本批测试题做定向修订或训练。','',
        '文件：`逐题对照.md`、`assistant_reviews.json`、`assistant_summary.json`、`artifact_verification.json`与`下一步建议.md`。','']
    (OUT/'结果审核报告.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(dict(status=summary['status'],new_reviews=48,scores={k:v['mean_total'] for k,v in models.items()},
        paired=pair_stats,adapter_issues=list(summary['confirmed_adapter_review_issues'])),ensure_ascii=False))


if __name__=='__main__':main()
