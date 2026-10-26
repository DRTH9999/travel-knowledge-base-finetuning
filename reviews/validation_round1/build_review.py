"""Reproduce the assistant's individually authored evidence review of round 1.

Scores and rationales below were assigned after reading all 48 outputs against
their visible evidence. This is an assistant review, not human sign-off or an
automatic semantic grader. Original downloaded files and human-review forms
remain unchanged.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
RAW = ROOT / "artifacts/validation_round1_original"
DIMENSIONS = ["faithfulness", "completeness", "citation_support", "answerability", "constraints_format"]

# Order: faithfulness, completeness, citation support, answerability, constraints/format.
# Faithfulness is relative to the supplied evidence, not a claim about current
# real-world tourism facts. Suggestions are distinguished from asserted facts.
REVIEWS = {
    1: {
        "base": ([0, 2, 0, 2, 1], "补入当前片段没有的公交3路、45路和步行/直达接驳方式，并把这些细节挂在[1]上；两站区别已回答，末尾停在空的【图片】标题。"),
        "adapter": ([2, 2, 2, 2, 2], "完整保留厦门站适合岛内片区、厦门北站偏高铁转场且需接驳的区别，[1]支持全部陈述。"),
    },
    2: {
        "base": ([0, 2, 0, 2, 1], "正确回答不能步行及轮渡流程，但自行给出第一码头/黄厝码头、半天游览等证据外细节；[1]不能支持具体码头示例，结尾被截断。"),
        "adapter": ([2, 2, 2, 2, 2], "给出轮渡必要性以及码头、船票、候船、乘船、步行入岛各环节；已实质回答不能从城区直接步行上岛。"),
    },
    3: {
        "base": ([1, 2, 1, 2, 0], "打车建议有依据，但把距离可控扩展为费用相对可控，补入当前片段未列举的平台、岛外地区和服务覆盖提醒；没有图片证据却输出图片URL占位符。"),
        "adapter": ([2, 2, 2, 2, 2], "以打车回应不研究换乘的出行诉求，所述距离与使用场景受[1]支持；可进一步点明网约车/出租车及行李场景，但不构成核心漏答。"),
    },
    4: {
        "base": ([0, 2, 0, 2, 2], "主结论正确，却将至少2-3小时、最好半天等具体时长标为[1]；实际片段只说留主要时段慢逛，没有这些数字。"),
        "adapter": ([2, 2, 2, 2, 2], "准确说明片区氛围、建筑和步行体验比单个地标重要，引用匹配。"),
    },
    5: {
        "base": ([0, 2, 0, 2, 0], "简餐/正餐分配正确，但增加温补功效、体质偏热及肠胃敏感等片段没有的健康相关判断，并错挂[1]；还生成example.com图片链接且结尾截断。这里只判断无证据，不核验健康论断真伪。"),
        "adapter": ([2, 2, 2, 2, 2], "沙茶面放简餐、姜母鸭放正餐补充，两项均由同一片段[1]支持。"),
    },
    6: {
        "base": ([0, 2, 1, 2, 1], "覆盖优缺点，但补入快艇、岛上酒店建筑风格和博物馆等未给出的事实；还说不适合长时间游玩，偏离当前片段慢游/住一晚体验的口径，末尾被截断。"),
        "adapter": ([2, 1, 2, 2, 2], "只复述沉浸感、夜晚和清晨氛围等优点，漏掉用户明确询问的行李搬运、上岛组织和返程复杂度。所引用的已说内容本身正确。"),
    },
    7: {
        "base": ([0, 2, 0, 2, 1], "是否适合的结论正确，但全程约30公里、白城沙滩/胡里山炮台等细节不在片段里，错误挂[1]，末尾未完成。"),
        "adapter": ([2, 2, 2, 2, 2], "正确指出该区适合海边休闲与慢节奏，足以回应看海、骑行和散步的住宿选择。"),
    },
    8: {
        "base": ([1, 2, 2, 2, 0], "核心觅食主线有依据；鼓浪屿餐饮选择较少、多为特色小店的补充描述未获当前片段支持，并输出不存在的图片占位符。"),
        "adapter": ([2, 2, 2, 2, 2], "明确区分鼓浪屿补充体验与本岛系统觅食，已回答不宜把全部用餐放岛上，[1]支持。"),
    },
    9: {
        "base": ([0, 1, 1, 2, 1], "住宿建议正确；上岛部分自行指定第一码头等，未完成返程码头说明，也未引用支持码头/航线核对的[2]，生成在句中触顶。"),
        "adapter": ([2, 2, 2, 2, 2], "住宿和出发前核对事项均覆盖，并分别引用[1]和[2]。"),
    },
    10: {
        "base": ([0, 0, 0, 2, 1], "第一天又建议去鼓浪屿，并加入南普陀寺及预约要求，既无当前证据支持又偏离资料的分天安排；输出未到第三天海边部分。"),
        "adapter": ([2, 2, 2, 2, 2], "老城/鼓浪屿/海边按三天排序，虽简短但回答核心问题；三天安排已由[1]全部支持。自动提示缺少[2]只是未复述背景理由，不属于本回答的错误引用。"),
    },
    11: {
        "base": ([0, 2, 1, 2, 1], "地铁+网约车和住宿方向正确，但把组合交通扩展为地铁覆盖厦大、沙坡尾、曾厝垵等具体片区；证据并未保证这些地点地铁直接覆盖，结尾截断。"),
        "adapter": ([2, 2, 2, 2, 2], "交通组合及中山路/轮渡住宿均有依据并覆盖约束；重复用户问题和把用餐列在交通下属于可优化的表达，不影响核心可用性。"),
    },
    12: {
        "base": ([1, 2, 0, 2, 1], "保留慢游与大行李/频繁往返取舍，且有建议先确认寄存服务的限定；但把寄存建议挂[2]、景点名挂[1]，对应片段均未提供这些内容，尾句被截断。"),
        "adapter": ([1, 0, 0, 2, 2], "仅建议在岛上酒店或其他地方寄存，未回答住一晚氛围与行李/上岛成本的权衡；[2]只说步行和不宜携大件高频往返，不能支持寄存服务。这是未核实的新增建议，不将其等同于已证伪的服务承诺。"),
    },
    13: {
        "base": ([0, 2, 0, 2, 0], "老城方向有依据，但夜晚部分区域灯光较暗等描述不在[2]中，却带引用输出；同时生成图片占位符。"),
        "adapter": ([2, 2, 2, 2, 2], "按吃喝便利选老城、海边松弛选海边的比较由[2]直接支持；缺少[1]的自动提示不是语义引用错误，亦无需逐字复现参考答案中的美食清单。"),
    },
    14: {
        "base": ([0, 0, 0, 2, 0], "未明确建议缩减环岛路，而是把老城/鼓浪屿压同一天并增加大量未提供景点，第二天继续长海边线路；违背不赶景点与资料主线，无来源引用，且被截断。"),
        "adapter": ([2, 2, 2, 2, 2], "保留老城+鼓浪屿，并明确两天时环岛路可以缩减；分别引用[1][2]，覆盖用户限制。"),
    },
    15: {
        "base": ([1, 2, 1, 2, 0], "核心区域正确；市中心具体等同于中山路/轮渡、厦门站住宿多为经济型等描述不在当前片段中，还出现图片占位符。"),
        "adapter": ([2, 2, 2, 2, 2], "厦门站及市中心适合短时停留、省事这一结论完全由[1]支持；[2]是同义补充，未引用它不构成错误。"),
    },
    16: {
        "base": ([0, 2, 0, 2, 2], "公交适用性正确，但补出单次2元、一日票和6:30至22:00等当前片段完全未给出的价格/时刻；尾加以公告为准不能为具体数字补足证据。"),
        "adapter": ([2, 2, 2, 2, 2], "公交适合预算敏感、时间弹性大以及相关片区的结论有[1]支持，没有补造票价班次。"),
    },
    17: {
        "base": ([1, 2, 1, 2, 1], "已回答不是必须住岛上、可考虑中山路/轮渡；交通班次多、部分岛上酒店不提供寄存等未见于证据，开头不是必须的结论未标[1]，附加段落被截断。"),
        "adapter": ([0, 2, 0, 2, 2], "住宿方向与不是必须住岛上正确，但断言岛上住宿资源有限、没有大型超市/餐厅/洗衣店等生活配套，当前两段证据均未提供；还把该断言挂在仅介绍本岛住宿优势的[2]上。自动编号合法检查未发现此语义错误。"),
    },
    18: {
        "base": ([1, 2, 2, 2, 0], "核心纠错与码头/航线要求引用正确；早班避开人流高峰、旺季预订等扩写未获片段明确支持，并输出图片占位符。"),
        "adapter": ([2, 2, 2, 2, 2], "以当前轮渡证据纠正旧历史，包含上岛流程和三类码头核对，[1][2]匹配；若明确说此前说法需纠正会更自然。"),
    },
    19: {
        "base": ([1, 2, 1, 2, 2], "正确区分缺失的当日时刻/票价与已知核对事项，部分可答处理较好；但自行补入第一码头、轮渡站等具体名称，当前片段并未给出。"),
        "adapter": ([1, 0, 0, 0, 2], "拒答时刻/票价正确，但把出发前核对事项也说成未提供，遗漏[1]明确列出的出发、上岛、返程码头等已知信息，是部分可答场景的过度拒答。"),
    },
    20: {
        "base": ([2, 2, 2, 2, 2], "先回答适合人群并引用[1]，再明确无法确认今晚房态与价格，没有编造库存或报价。"),
        "adapter": ([2, 1, 2, 1, 2], "片区适合人群回答正确；用户虽允许先答静态部分，但仍应补明无法确认今晚300元内双床房，当前回答没有交代这部分的信息边界。"),
    },
    21: {
        "base": ([2, 2, 2, 2, 2], "没有把无关景点概览当成天气依据，明确缺少指定日期的天气和气温；景区公告式措辞来自原业务模板，天气来源指向应单独改进模板。"),
        "adapter": ([2, 2, 2, 2, 2], "资料不足时没有编造天气/气温；拒答较通用，可在后续提示词实验中明确指向天气预报，不把遵循当前模板的通用措辞单独扣分。"),
    },
    22: {
        "base": ([2, 2, 0, 2, 2], "宠物政策和押金的拒答边界正确，但本题根本没有任何检索片段，仍生成[1]，属于无来源编号。"),
        "adapter": ([2, 2, 2, 2, 2], "无证据时不编造宠物政策、押金或引用编号；补问酒店名称会更有操作性，但当前拒答边界正确。"),
    },
    23: {
        "base": ([1, 2, 2, 2, 0], "住宿主结论及[2]正确，夜间活动较多等扩写未获当前片段明确支持；生成example.com图片链接，所有给定片段都没有图片URL。"),
        "adapter": ([2, 2, 2, 2, 2], "选中真正支持老城/鼓浪屿住宿的[2]，没有把小吃或海边片段混作当前建议依据。"),
    },
    24: {
        "base": ([2, 2, 2, 2, 2], "识别小吃片段不能支持酒店实时房价，拒答并建议核对预订平台或酒店，无编造引用。"),
        "adapter": ([2, 2, 2, 2, 2], "拒答无证据酒店房价且未编造数字；景区公告的通用来源措辞不贴合酒店，应归入后续业务模板优化项。"),
    },
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = read_jsonl(ROOT / "data/eval_v1/validation.jsonl")
    predictions = {label: read_jsonl(RAW / f"{label}.predictions.jsonl") for label in ("base", "adapter")}
    raw_report = json.loads((RAW / "report.json").read_text(encoding="utf-8"))
    assert raw_report["eval_manifest_sha256"] == sha(ROOT / "data/eval_v1/manifest.json")
    assert raw_report["eval_file_sha256"] == sha(ROOT / "data/eval_v1/validation.jsonl")
    assert len(rows) == len(REVIEWS) == 24
    reviewed = []
    for index, row in enumerate(rows, 1):
        for label in ("base", "adapter"):
            p = predictions[label][index - 1]
            scores, rationale = REVIEWS[index][label]
            assert p["id"] == row["id"] and len(scores) == len(DIMENSIONS)
            assert all(type(v) is int and 0 <= v <= 2 for v in scores)
            reviewed.append(dict(id=row["id"], model=label, task_type=row["task_type"],
                                 scores=dict(zip(DIMENSIONS, scores)), total=sum(scores), rationale=rationale,
                                 status="assistant_reviewed_user_pending",
                                 assessor="Codex assistant, source-by-source review", human_signoff=False,
                                 prediction_sha256=hashlib.sha256(p["answer"].encode("utf-8")).hexdigest()))
    assert all(a["input_sha256"] == b["input_sha256"] for a, b in zip(predictions["base"], predictions["adapter"]))
    summary = dict(status="ASSISTANT_EVIDENCE_REVIEW_COMPLETE", reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
                   scope="24 validation questions, 48 answers, original 256-new-token run only",
                   assessor="Codex assistant; not an independent human or blind review", human_signoff=False,
                   test_set_evaluated=False, automatically_graded_semantics=False,
                   original_archive_sha256=sha(ROOT / "qwen3_eval_validation_round1.tar.gz"),
                   input_identity_verified=True, score_scale="5 evidence-review dimensions, each 0-2; total 0-10; not accuracy",
                   limitations=["The same assistant authored the gold answers and reviewed outputs; independent human review remains needed.",
                                "Base model reached the generation cap in 11/24 cases; score differences are budget-conditional.",
                                "One held-out destination and shared source facts; no general tourism accuracy claim.",
                                "No external current-tourism fact checking was performed; faithfulness is relative to visible evidence."],
                   models={})
    for label in ("base", "adapter"):
        items = [r for r in reviewed if r["model"] == label]
        ps = predictions[label]
        tokens, seconds = sum(p["generated_tokens"] for p in ps), sum(p["generation_seconds"] for p in ps)
        summary["models"][label] = dict(samples=24,
            mean_scores={key: sum(r["scores"][key] for r in items) / 24 for key in DIMENSIONS},
            mean_total=sum(r["total"] for r in items) / 24,
            faithfulness_issue_ids=[r["id"] for r in items if r["scores"]["faithfulness"] < 2],
            completeness_issue_ids=[r["id"] for r in items if r["scores"]["completeness"] < 2],
            citation_issue_ids=[r["id"] for r in items if r["scores"]["citation_support"] < 2],
            mean_generated_tokens=tokens / 24, median_generated_tokens=statistics.median(p["generated_tokens"] for p in ps),
            total_generated_tokens=tokens, total_generation_seconds=seconds,
            aggregate_tokens_per_generation_second=tokens / seconds,
            throughput_interpretation="Includes prefill and variable output lengths; not a fixed-length decode benchmark.",
            output_limit_count=sum(p["hit_output_limit"] for p in ps),
            image_block_count=sum("【图片】" in p["answer"] for p in ps),
            image_url_or_placeholder_count=sum("https://" in p["answer"] or "<图片URL" in p["answer"] for p in ps))
    by_pair = {r["id"]: {v["model"]: v for v in reviewed if v["id"] == r["id"]} for r in rows}
    summary["paired_rubric_total_comparison"] = dict(
        adapter_higher=sum(p["adapter"]["total"] > p["base"]["total"] for p in by_pair.values()),
        tied=sum(p["adapter"]["total"] == p["base"]["total"] for p in by_pair.values()),
        adapter_lower=sum(p["adapter"]["total"] < p["base"]["total"] for p in by_pair.values()))
    (OUTPUT / "assistant_reviews.json").write_text(json.dumps(dict(summary=summary, reviews=reviewed), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "assistant_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 厦门验证集第一轮证据审核", "", "审核日期：2026-10-24（Asia/Shanghai）。", "",
             "审核方式：助手逐条对照已冻结的原文片段与48份原始回答；不是独立人工、盲审或自动语义评分。用户业务审核仍待完成。", "",
             "原始文件保持不变：`artifacts/validation_round1_original/`。本审核只使用验证集，没有生成或读取张家界测试集预测。", "",
             "## 结论", "", "本轮样例中，10步适配器减少了证据外扩写和无来源图片输出，但仍有关键事实边界、错引、漏答及部分可答时过度拒答。当前产物用于行为适配实验，不建议直接替换线上答案模型。", "",
             "原始模型11题触及256个新Token上限，需在相同512新Token设置下重跑两路，才能更完整地区分模型能力与输出预算影响。已有无依据事实往往出现在截断前，增加预算不能自动消除它们。", "",
             "## 可复核的主要发现", "",
             "- 006：适配器只回答住岛吸引力，遗漏搬运行李、上岛组织和返程复杂度。",
             "- 012：适配器以[2]支撑酒店/岛上寄存建议，但[2]只讲步行和大件行李，不提供寄存服务依据，也没有回答住一晚的取舍。",
             "- 017：适配器断言岛上没有大型超市、餐厅、洗衣店等配套并引[2]；两段给定证据没有这类断言。",
             "- 019：适配器把末班时刻、票价、出发前核对事项全部说成未提供；实际上[1]已提供需核对的码头/航线类别。",
             "- 020：适配器答出片区适合人群，但没有说明今晚预算内房态/价格无法确认。",
             "- 010/013/015：适配器的自动缺引提示不构成语义引用错误；它实际引用的[1]或[2]已经足以支持回答。反之，017编号合法却存在语义错引。", "",
             "## 输出长度与耗时", "",
             "| 指标 | 原始模型 | 试跑适配器 |", "|---|---:|---:|"]
    for title, key, fmt in [("平均输出Token", "mean_generated_tokens", ".2f"),
                            ("总输出Token", "total_generated_tokens", "d"),
                            ("总生成耗时（秒）", "total_generation_seconds", ".2f"),
                            ("输出上限触发题数", "output_limit_count", "d"),
                            ("含无来源图片URL/占位符的题数", "image_url_or_placeholder_count", "d")]:
        left, right = summary["models"]["base"][key], summary["models"]["adapter"][key]
        lines.append(f"| {title} | {format(left, fmt)} | {format(right, fmt)} |")
    lines += ["", "适配器总耗时更短与平均输出大幅缩短同时发生，不能据此认定解码吞吐提高。按输出Token/整段生成时间汇总分别约41.10和22.86 Token/秒，但包含prefill及不同输出长度，也不能反向据此认定适配器固有解码速度更慢。", "",
              "## 助手量表评分", "", "下表是本次助手按相同规则逐题审阅的辅助量表，非准确率、非人工验收结论。一个缺陷可能同时影响多个维度，总分不应替代具体例子或独立复核。", "",
              "| 维度（每项0–2） | 原始模型均分 | 适配器均分 |", "|---|---:|---:|"]
    names = dict(faithfulness="事实忠实", completeness="完整性", citation_support="引用支持", answerability="回答边界", constraints_format="约束与表达")
    for key in DIMENSIONS:
        lines.append(f"| {names[key]} | {summary['models']['base']['mean_scores'][key]:.3f} | {summary['models']['adapter']['mean_scores'][key]:.3f} |")
    lines += [f"| 总分（0–10） | {summary['models']['base']['mean_total']:.3f} | {summary['models']['adapter']['mean_total']:.3f} |", "",
              "评分补充规则：不因没有逐字复现参考答案或没有引用全部gold编号而扣分；足够支持实际回答的引用即可。完全可答题不因普通证据外扩写再重复扣拒答边界分；部分可答题需要分别处理已知与缺失内容。普通条件性建议与具体事实断言区分处理。空证据题不要求引用，但不能生成虚构编号。", "",
              "## 逐题审核", ""]
    for index, row in enumerate(rows, 1):
        lines += [f"### {row['id']}", "", f"问题：{row['question']}", ""]
        for label in ("base", "adapter"):
            review = by_pair[row["id"]][label]
            lines += [f"**{label}：{review['total']}/10**（忠实/完整/引用/边界/表达：" + "/".join(str(review["scores"][k]) for k in DIMENSIONS) + "）", "",
                      review["rationale"], "", "```text", predictions[label][index - 1]["answer"], "```", ""]
        lines += ["**本题原文证据**", ""]
        for context in row["contexts"]:
            lines += [f"[{context['reference_index']}] {context['relative_path']} L{context['line_start']}-{context['line_end']}", "",
                      "```text", context["content"], "```", ""]
        if not row["contexts"]:
            lines += ["没有参考片段。", ""]
    lines += ["## 下一步", "",
              "1. 保留本轮，给两个模型相同的512新Token预算，在厦门验证集重跑，检查是否仍触顶，再复核原始模型的完整回答。",
              "2. 用成都、杭州、三亚、云南资料定向补充多问覆盖、优缺点、部分可答、证据未提供服务条件等样例；不要把厦门验证题直接拷进训练集。",
              "3. 针对天气/酒店/轮渡采用更贴合的信息核对来源，作为独立的提示词实验；不把提示词改进收益混同为模型训练收益。",
              "4. 业务口径审核后再确定训练扩充规模和正式训练参数；张家界测试集仍留作最终评估。", ""]
    (OUTPUT / "审核报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(status=summary["status"], reviewed_answers=len(reviewed),
                         scores={label: summary["models"][label]["mean_total"] for label in ("base", "adapter")},
                         paired=summary["paired_rubric_total_comparison"], output=str(OUTPUT)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
