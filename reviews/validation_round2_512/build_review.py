"""Evidence-review the 512-token control run and compare it with round 1.

Previously reviewed answers are reused only after exact text/hash identity checks.
All eleven extended base-model answers were separately read against the same
frozen evidence. These are assistant assessments, not independent human scores.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DIMENSIONS = ["faithfulness", "completeness", "citation_support", "answerability", "constraints_format"]

# Updates apply only to the eleven answers that were cut at 256 new tokens.
# Score order remains identical to round 1; new suffixes were source-reviewed.
CHANGED_BASE_REVIEWS = {
    1: ([0, 2, 0, 2, 0], "两站接驳区别已回答，3路/45路、步行直达等无来源细节仍在；这次结束完整，但新增两条example.com图片链接，片段未提供任何图片。"),
    2: ([0, 2, 0, 2, 0], "轮渡流程完整且不再截断；第一码头/黄厝码头及半天游览等无依据扩写仍在，末尾新增example.com图片链接。"),
    5: ([0, 2, 0, 2, 0], "完整输出后，温补功效/体质相关判断仍无给定证据支持；原先截断的图片段补成两条example.com链接。这里只核对证据边界，不核验健康论断真伪。"),
    6: ([0, 2, 1, 2, 0], "优缺点均已覆盖，但快艇、博物馆、岛上酒店风格等仍不在片段中，且不适合长时间游玩偏离慢游口径；结尾新增无来源图片链接。"),
    7: ([0, 2, 0, 2, 0], "是否适合的主结论有依据；30公里和具体景点等仍无依据，新增部分路段施工/封闭说法及无来源图片，引用[1]不能支持这些细节。"),
    9: ([0, 2, 1, 2, 0], "住宿及三类码头/班次核对这次已完整覆盖，并补出[2]；但自行指定码头、建议提前1-2小时、具体平台及图片链接等超出片段。"),
    10: ([0, 2, 0, 2, 0], "第三天海边部分这次已生成，三天结构完整；第一天又可去鼓浪屿、南普陀寺预约及岛上无机动车等仍无当前依据，多项被挂在[1]，末尾生成三条example.com图片链接。"),
    11: ([0, 2, 1, 2, 0], "交通和住宿要求完整回答；地铁直接覆盖具体片区的外推仍在，延长部分又新增6:30至23:00运营时间等未给出的信息和无来源图片。"),
    12: ([1, 2, 0, 2, 0], "慢游与大行李取舍已有回答；寄存和具体景点错引问题仍在，新增厦门市区酒店/寄存服务建议仍被挂在只讲步行的[2]，并新增图片链接。条件性建议与确定服务承诺区分处理。"),
    14: ([0, 2, 1, 2, 0], "这次明确回答不用走完整个环岛路，完整性比256上限时改善，[2]能支持缩减方向；但第一天密集安排及新增30公里、10-15公里、2-3小时等无依据细节仍违背证据边界，并生成图片链接。"),
    17: ([0, 2, 0, 2, 1], "仍在512上限处结束，但核心是否必须住岛上及住宿区域已回答；延长部分新增岛上房价较高、7:00-17:30开放时间，并引用根本不存在的[3]。原先未引[1]已在末段补出，不能因此忽略新错引。"),
}


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    raw1 = ROOT / "artifacts/validation_round1_original"
    raw2 = ROOT / "artifacts/validation_round2_512_original"
    reports = [json.loads((p / "report.json").read_text(encoding="utf-8")) for p in (raw1, raw2)]
    for key in ("model_config_sha256", "adapter_sha256", "eval_manifest_sha256", "eval_file_sha256", "chat_template_sha256", "versions"):
        assert reports[0][key] == reports[1][key], key
    assert reports[0]["model_dir"] == reports[1]["model_dir"]
    assert reports[1]["eval_manifest_sha256"] == digest(ROOT / "data/eval_v1/manifest.json")
    assert reports[1]["eval_file_sha256"] == digest(ROOT / "data/eval_v1/validation.jsonl")
    assert reports[0]["generation"]["max_new_tokens"] == 256 and reports[1]["generation"]["max_new_tokens"] == 512
    assert {k: v for k, v in reports[0]["generation"].items() if k != "max_new_tokens"} == {
        k: v for k, v in reports[1]["generation"].items() if k != "max_new_tokens"}
    previous = json.loads((ROOT / "reviews/validation_round1/assistant_reviews.json").read_text(encoding="utf-8"))
    old_reviews = {(r["id"], r["model"]): r for r in previous["reviews"]}
    rows = read_jsonl(ROOT / "data/eval_v1/validation.jsonl")
    predictions = [{label: read_jsonl(p / f"{label}.predictions.jsonl") for label in ("base", "adapter")} for p in (raw1, raw2)]
    reviews, changed, inherited = [], [], []
    for i, row in enumerate(rows):
        for label in ("base", "adapter"):
            before, after = predictions[0][label][i], predictions[1][label][i]
            assert before["id"] == after["id"] == row["id"]
            assert before["input_sha256"] == after["input_sha256"]
            assert after["input_sha256"] == predictions[1]["base"][i]["input_sha256"]
            old = old_reviews[(row["id"], label)]
            assert old["prediction_sha256"] == hashlib.sha256(before["answer"].encode("utf-8")).hexdigest()
            if before["answer"] == after["answer"]:
                scores, reason = old["scores"], old["rationale"]
                method = "reused_previous_evidence_review_after_exact_answer_and_evidence_identity_check"
                inherited.append([row["id"], label])
            else:
                assert label == "base" and i + 1 in CHANGED_BASE_REVIEWS
                numbers, reason = CHANGED_BASE_REVIEWS[i + 1]
                scores = dict(zip(DIMENSIONS, numbers))
                method = "extended_base_answer_source_reviewed_in_this_round"
                changed.append(row["id"])
            assert all(type(v) is int and 0 <= v <= 2 for v in scores.values())
            reviews.append(dict(id=row["id"], model=label, scores=scores, total=sum(scores.values()),
                                rationale=reason, review_method=method, human_signoff=False,
                                status="assistant_reviewed_user_pending",
                                prediction_sha256=hashlib.sha256(after["answer"].encode("utf-8")).hexdigest()))
    assert len(reviews) == 48 and len(changed) == 11 and len(inherited) == 37
    assert set(int(n.rsplit("-", 1)[1]) for n in changed) == set(CHANGED_BASE_REVIEWS)
    summary = dict(status="ROUND2_CONTROL_REVIEW_COMPLETE", reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
                   assessor="Codex assistant; not independent human review or a blind assessment", human_signoff=False,
                   score_scale="5 dimensions each 0-2; total 0-10; evidence-review rubric, not accuracy",
                   archive_sha256=digest(ROOT / "qwen3_eval_validation_round2_512.tar.gz"),
                   controls=dict(same_model_config_and_path=True, same_adapter=True, same_data=True, same_template=True,
                                 same_versions=True, paired_inputs_verified=24,
                                 base_weight_identity_scope="Same model directory and config hash; full base weight files were not independently rehashed.",
                                 generation_setting_change="max_new_tokens: 256 -> 512"),
                   identical_adapter_answers=24, identical_base_answers=13,
                   newly_reviewed_base_answers=11, reused_after_exact_identity=37,
                   test_set_evaluated=False, rounds={}, models={})
    for number, pred in enumerate(predictions, 1):
        stats = {}
        for label, ps in pred.items():
            tokens, seconds = sum(p["generated_tokens"] for p in ps), sum(p["generation_seconds"] for p in ps)
            stats[label] = dict(total_tokens=tokens, mean_tokens=tokens / 24,
                                total_generation_seconds=seconds, mean_generation_seconds=seconds / 24,
                                output_limit_count=sum(p["hit_output_limit"] for p in ps),
                                output_limit_ids=[p["id"] for p in ps if p["hit_output_limit"]],
                                invalid_reference_sample_count=sum(bool(p["format_flags"]["invalid_reference_ids"]) for p in ps),
                                missing_gold_reference_sample_count=sum(bool(p["format_flags"]["missing_gold_reference_ids"]) for p in ps),
                                image_url_or_placeholder_count=sum(bool(re.search(r"https?://|<图片URL", p["answer"])) for p in ps))
        summary["rounds"][str(number)] = stats
    for label in ("base", "adapter"):
        items = [r for r in reviews if r["model"] == label]
        summary["models"][label] = dict(mean_scores={k: sum(r["scores"][k] for r in items) / 24 for k in DIMENSIONS},
                                       mean_total=sum(r["total"] for r in items) / 24)
    paired = {r["id"]: {v["model"]: v for v in reviews if v["id"] == r["id"]} for r in rows}
    summary["paired_rubric_total_comparison"] = dict(
        adapter_higher=sum(p["adapter"]["total"] > p["base"]["total"] for p in paired.values()),
        tied=sum(p["adapter"]["total"] == p["base"]["total"] for p in paired.values()),
        adapter_lower=sum(p["adapter"]["total"] < p["base"]["total"] for p in paired.values()))
    summary["limitations"] = [
        "One held-out destination and shared source facts; no general tourism accuracy claim.",
        "The same assistant authored the gold answers and reviewed outputs; independent human review remains pending.",
        "Base case 017 still reached 512 tokens; it remains budget-conditional.",
        "Elapsed times are single-run observations; GPU load/clocks were not controlled or measured.",
        "Base model identity was checked by local path and config hash, not by rehashing all base weights.",
        "Visible-evidence faithfulness was reviewed, not real-time external tourism factual truth.",
    ]
    save_json(OUT / "assistant_reviews.json", dict(summary=summary, reviews=reviews))
    save_json(OUT / "assistant_summary.json", summary)
    lines = ["# 厦门验证集512 Token复评审核", "", "审核日期：2026-10-24（Asia/Shanghai）。", "",
             "审核主体为助手，非独立人工、非盲审。原始下载产物和待人工评分表保持不变。", "",
             "## 结论", "",
             "提高输出上限主要改善了原始模型的生成完整性，没有修复适配器的漏答、错引、无依据断言和部分可答时的过度拒答。适配器24份回答与第一轮逐字一致，不是因Token上限而提前截断。", "",
             "本次只重新审核原始模型延长的11份回答；另外37份回答在文本、输入和证据逐项一致后沿用第一轮审核。覆盖两路共48份回答。", "",
             "不建议继续机械增加到1024 Token来处理这些语义问题。第017题仍触顶，应记录该限制，但适配器五类问题已足够明确，下一阶段应定向修订训练数据。", "",
             "## 对照设置核验", "", "模型路径与配置、适配器、数据清单、验证题、聊天模板哈希以及依赖版本一致；24题两路与跨轮输入哈希一致。基础模型全部权重未重新哈希。报告记录的生成配置中只有max_new_tokens从256变为512。没有读取或生成张家界测试集预测。", "",
             "## 两轮观察", "", "| 指标 | 原始模型256 | 原始模型512 | 适配器256 | 适配器512 |", "|---|---:|---:|---:|---:|"]
    for title, key, fmt in [("触及输出上限题数", "output_limit_count", "d"),
                            ("越界引用题数", "invalid_reference_sample_count", "d"),
                            ("自动缺引提示题数", "missing_gold_reference_sample_count", "d"),
                            ("无来源图片URL/占位符题数", "image_url_or_placeholder_count", "d"),
                            ("平均输出Token", "mean_tokens", ".2f"),
                            ("平均单题生成秒数", "mean_generation_seconds", ".3f")]:
        values = [summary["rounds"][str(n)][label][key] for label in ("base", "adapter") for n in (1, 2)]
        lines.append("| " + title + " | " + " | ".join(format(v, fmt) for v in values) + " |")
    lines += ["", "单次运行时延不能当作严格吞吐基准；适配器文本和950个总输出Token未变，耗时仍有变化，不能把所有跨轮耗时差异归因于上限。原始模型更多地生成了原来被截断的内容，其中既有补全的信息，也有新的无依据数字和图片链接。", "",
              "## 关键实例", "",
              "- 原始模型009：补齐码头/航线核对并引用[2]，但又加入资料没有的提前1-2小时到码头及图片链接。",
              "- 原始模型010：第三天海边安排已生成，但南普陀寺预约、岛上无机动车等证据外扩写和错误引用仍在。",
              "- 原始模型014：现在能明确说不用走完整个环岛路，但新添30公里、10-15公里、2-3小时等片段没有的数字。",
              "- 原始模型017：两段证据中新增不存在的引用[3]，支撑未经提供的7:00-17:30开放时间；该题仍在512上限结束。",
              "- 适配器006/012/017/019/020：与上一轮完全相同，分别仍有遗漏不便、寄存建议错引、编造配套、过度拒答和未说明实时房态边界。", "",
              "## 助手量表（辅助，不是准确率）", "", "原始模型的完整性均分提升，但也暴露了更多图片及引用问题；不能以综合分变化推断增大预算使模型固有能力变好或变差。", "",
              "| 维度 | 原始模型512 | 适配器512 |", "|---|---:|---:|"]
    names = dict(faithfulness="事实忠实", completeness="完整性", citation_support="引用支持", answerability="回答边界", constraints_format="约束与表达")
    for key in DIMENSIONS:
        lines.append(f"| {names[key]}（0–2） | {summary['models']['base']['mean_scores'][key]:.3f} | {summary['models']['adapter']['mean_scores'][key]:.3f} |")
    lines += [f"| 总分（0–10） | {summary['models']['base']['mean_total']:.3f} | {summary['models']['adapter']['mean_total']:.3f} |", "",
              "## 本轮逐题记录", ""]
    for i, row in enumerate(rows):
        lines += [f"### {row['id']}", "", "问题：" + row["question"], ""]
        for label in ("base", "adapter"):
            item = paired[row["id"]][label]
            lines += [f"**{label}：{item['total']}/10**", "", item["rationale"], "",
                      "审核方式：" + item["review_method"], "", "```text", predictions[1][label][i]["answer"], "```", ""]
        for context in row["contexts"]:
            lines += [f"证据[{context['reference_index']}]：{context['relative_path']} L{context['line_start']}-{context['line_end']}", "",
                      "```text", context["content"], "```", ""]
        if not row["contexts"]:
            lines += ["没有检索片段。", ""]
    (OUT / "复评审核报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(status=summary["status"], identical_adapter_answers=24,
                         newly_reviewed_base_answers=11, inherited_exact_answers=37,
                         scores={k: v["mean_total"] for k, v in summary["models"].items()},
                         paired=summary["paired_rubric_total_comparison"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
