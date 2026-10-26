# 固定完整1轮适配器，比较两套提示词

状态更新（2026-10-24）：云端对照及本地核验已完成，原组24份回答复现前轮。修订版改善005/020，但014仍混用条件，017新增关键主体/引用错配；建议保留原提示词，不整体采用v2。具体见 `reviews/prompt_v2_validation/结果审核报告.md`。下面保留实验操作说明供复现。

本轮复用现有Qwen3-8B完整1轮适配器，比较原提示词与修订提示词。两组均使用原厦门24题、相同证据和历史、非思考模式、NF4/BF16、贪心生成及512新Token上限。

模型权重保持不变，本轮不训练，也不生成张家界测试预测。目的是检查通用回答规则能否改善残留的主体混用、条件混用、实时边界和引用位置，同时检查其他题是否退步。

## 新增的四项通用要求

具体原文在 `rules_v2.txt`，完整候选模板在 `prompt_template_v2.txt`：

1. 主体与属性对应，不把相邻对象的特点、场景或服务迁移过来。
2. 保留天数、时段、人群等条件，区分必要、可选和可缩减的安排。
3. 覆盖各个子问题；部分可答时分别说明已有依据与当前无法确认的内容，实时缺口表述直接。
4. 引用就近且支持对应结论，不凑编号，也不为缺失事实补造依据。

候选只在原业务模板的参考内容之前插入这四项要求，其余规则、问题、证据、历史和实体内容全部保留。规则没有放入厦门题目的正确答案、专有事实或题号。原训练数据和冻结评测集也保持不变。

## 上传并运行

将 `sft_prompt_v2_validation.zip` 上传到 JupyterLab 左侧 `/autodl-tmp/`。终端执行：

```bash
source /root/autodl-tmp/travel-finetune/activate.sh
python -m zipfile -e /root/autodl-tmp/sft_prompt_v2_validation.zip /root/autodl-tmp/travel-finetune/sft_pilot_v1
cd /root/autodl-tmp/travel-finetune/sft_pilot_v1
bash prompt_pipeline/run_prompt_experiment.sh
```

脚本会校验文件与候选适配器，加载一次现有模型，先用原提示词重新回答24题。只有原回答能逐题复现上一轮结果时，才继续用修订提示词回答同样24题。两组合计48份回答，不重复训练或重新下载模型。

预期依次看到：

```text
PROMPT_PREFLIGHT_OK
PROMPT_CONTROL_REPRODUCED
PROMPT_COMPARISON_OK
PROMPT_RESULTS_CHECK_OK
PROMPT_EXPERIMENT_COMPLETE
```

`PROMPT_CONTROL_REPRODUCED` 应对应24份原提示词回答与上一轮逐字相同；这用于检查比较条件。其他完成标记表示生成、文件检查和打包成功，不能直接判断修订提示词更好。

## 下载结果

完成后刷新左侧 `/autodl-tmp/`，下载：

```text
qwen3_prompt_v2_validation_results.tar.gz
```

放回本地 `E:\260515\知识库项目模型微调\`，再审核两组回答的变化。包里包括原/修订模板、运行代码、实际输入Token、两组预测、前轮基线和待人工评分表；不重复打包适配器权重。

## 如何查看与判断

输出目录为 `eval_outputs/prompt_v2_validation/`：

- `comparison.md`：24题逐题展示原提示词/修订提示词回答、证据与参考答案。
- `original_prompt.predictions.jsonl`、`revised_prompt.predictions.jsonl`：两组原始生成记录。
- `inputs.jsonl`：实际输入Token、模板变体与问题/证据/历史载荷哈希。
- `report.json`：适配器身份、模板、生成设置、原提示词复现结果和格式提示。
- `manual_reviews.json`：两组48项待人工审核记录，分数保持null，不自动变成通过。

重点复核主体/属性是否串用、条件是否保留、部分可答是否说明每项缺口，以及关键结论的引用是否贴切。全24题都需检查，不只盯着上一轮的三处残留；已改善的历史追问、流程和行程不能退步。

这批厦门题已用于迭代，结果只适合做提示词选择和回归，不能当作独立最终测试或生产准确率。两组规则长短不同，单次生成耗时也不作为严格性能结论。模型和提示词选定后，再安排冻结张家界最终测试。

## 单独检查与错误处理

只检查身份和输入，不加载模型权重：

```bash
python prompt_pipeline/evaluate_prompts.py --validate-only
```

只核对已经生成的结果：

```bash
python prompt_pipeline/validate_results.py
```

如出现 `PROMPT_CONTROL_MISMATCH`，脚本会保留原提示词回答和报告并停止，不继续生成修订组；先查看对应 `report.json` 和终端日志。不要跳过复现检查，也不要接着运行张家界测试。

如需要从头另做一次运行，使用新名称保留旧产物：

```bash
bash prompt_pipeline/run_prompt_experiment.sh prompt_v2_validation_b
```

两组设置从冻结清单读取，不能临时改变512上限或适配器后仍称为同一对照。调试的 `--limit` 结果会明确标为不完整，默认结果校验器拒绝把它当作24题完整结果。

## 本地检查范围

真实Qwen3 Tokenizer已核对24题两套输入：原提示词与前轮输入逐题一致；两组问题/证据/历史载荷一致；最长输入分别为562与788 Token，均完整保留并预留512输出空间，无截断。

本地随机小Qwen3仅用于检查两组生成、基线复现、结果校验和异常停止，不用来评价提示词质量。测试协议在独立缓存副本中绑定小模型和调试基线，生产清单与现有8B权重保持原样。具体证据见 `local_validation.json`。

本轮8B云端原始产物现已保留在 `artifacts/prompt_v2_validation_original/`，本次建议不整体采用修订提示词。原组/修订组代码和模板分别保留，不改原冻结模板。生成与适配器加载沿用已核验实现，相关接口见 [Transformers GenerationConfig](https://huggingface.co/docs/transformers/en/main_classes/text_generation) 和 [PEFT模型加载](https://huggingface.co/docs/peft/en/package_reference/peft_model)。
