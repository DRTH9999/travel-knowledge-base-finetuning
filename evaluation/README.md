# 下一步：固定证据验证集对照

已读取用户下载到本地的 `qwen3_sft_smoke_20261023.tar.gz`，原始 `smoke_report.json` 的状态为 `TRAIN_SMOKE_OK`，完成 10 个优化器步和 10 次梯度检查。压缩包中适配器权重的 SHA256 与报告一致：`c75a2da1a2a35b51b0653a33a83a43584d23f8532a4d3bde069c853c1e3e2336`。报告及核验记录保存在本机 `artifacts/cloud_smoke_20261023_210443/`。

本阶段进行固定证据问答对照，比较原始 Qwen3-8B 与这次 10 步试跑适配器。它复用云端已有模型和环境，生成答案和审核材料，不执行训练，不调用 qwen-flash 等付费生成 API。

## 1. 数据安排

| 集合 | 来源 | 数量 | 本阶段用途 |
| --- | --- | --- | --- |
| 原训练小样本 | 成都、杭州、三亚、云南 | 50 | 保持现有训练数据 |
| validation | 厦门 5 份文档 | 24 | 现在比较两个模型的回答 |
| test | 张家界 5 份文档 | 24 | 已冻结，留给最终评估，不默认运行 |

每个评测集合包括：单片段问答 8、多片段整合 5、多条件咨询 3、历史追问/纠错 2、信息不足 4、干扰筛选 2。每集分别有 19 条可答、2 条部分可答、3 条不可答问题。测试集中保留了一条原文确实存在不同两天游建议的样例，没有伪造来源冲突。

来源正文、问题、证据、参考答案、业务提示词和审核稿的哈希均在 `data/eval_v1/manifest.json` 中冻结。重新构建时若内容改变，脚本要求新版本；已冻结的测试结果不得用于调参或选择检查点。

每个集合只有一个目的地、5 份短文档，问题和历史是合成内容，证据是按标题选取的原文摘录。24 题不等于 24 个独立事实组，也不是完整 RAG 检索评测。原文的价格、天气和班次等没有进行实时外部核验；参考答案仍需业务口径审核。

## 2. 上传与检查

把本机的 `sft_evaluation_v1.zip` 上传到 AutoDL 的 `/root/autodl-tmp/`。其中包含 `evaluation/` 和 `data/eval_v1/`，不包含模型权重。

在 AutoDL Bash 终端依次执行；某一步报错时先处理错误，再继续。

```bash
source /root/autodl-tmp/travel-finetune/activate.sh
python -m zipfile -e \
  /root/autodl-tmp/sft_evaluation_v1.zip \
  /root/autodl-tmp/travel-finetune/sft_pilot_v1
cd /root/autodl-tmp/travel-finetune/sft_pilot_v1
sha256sum -c evaluation/checksums.sha256
python evaluation/validate_eval.py
python evaluation/evaluate.py --validate-only
```

文件检查应全部为 `OK`；两次数据检查分别输出 `EVAL_DATA_CHECK_OK` 和 `EVAL_PREFLIGHT_OK`。最后一条使用云端真实 Tokenizer 检查全部 24 道验证题和输入/输出长度预算，不加载模型权重。

## 3. 运行验证集对照

云端适配器目录来自已经核验的原始报告。运行：

```bash
bash evaluation/run_baseline.sh \
  outputs/smoke_20261023_210443/adapter \
  eval_outputs/validation_round1
```

脚本先让基础模型回答 24 题，释放模型，再加载同一基础模型与指定适配器回答相同 24 题。终端显示 `base: 1/24 ...` 和 `adapter: 1/24 ...` 的进度；每完成一题立即落盘，完整日志保存在 `logs/eval-validation-日期时间.log`。

两个模型使用相同的问题、历史和证据顺序、Qwen3 非思考模板、NF4 双重量化、BF16 计算、SDPA、贪心生成、最多 256 个新 Token。每个模型先做一次 1 Token 预热，单题耗时不包含这次预热；预热耗时单独记录。每题输入 Token 的哈希记录在两份结果中，可核对输入一致。参考答案和证据相关性标签不会进入模型输入。

输出目录必须为空或不存在。需要重跑时使用新的名称，例如 `eval_outputs/validation_round2`，保留前一次结果。

结束标志：

```text
EVAL_GENERATION_OK
```

这是程序输出，不是要执行的命令；它表示答案生成完成，业务质量仍待审核。

## 4. 查看结果与打包

产物在 `eval_outputs/validation_round1/`：

- `comparison.md`：逐题展示问题、历史、参考答案、两个模型的回答、格式提示和原文证据。
- `base.predictions.jsonl`、`adapter.predictions.jsonl`：原始生成结果、输入哈希、Token 数、生成耗时和输出上限提示。
- `report.json`：数据/模型/适配器哈希、生成设置、实际版本、格式异常计数、平均单题生成耗时和显存峰值。
- `manual_reviews.json`：待填的人工审核记录，未审核项为 `null`，不自动计分。

下载结果前执行：

```bash
tar -czf /root/autodl-tmp/qwen3_eval_validation_round1.tar.gz \
  -C /root/autodl-tmp/travel-finetune/sft_pilot_v1 \
  eval_outputs/validation_round1
```

然后在 JupyterLab 左侧 `/autodl-tmp/` 下载 `qwen3_eval_validation_round1.tar.gz`。它含原始回答与证据对照，便于下一步逐题审核。

## 5. 如何判断效果

自动格式提示只检查空回答、思考标签、越界引用编号、引用无关片段、缺少参考答案引用和触及输出上限。编号存在或与参考答案相同，不证明引用确实支持结论；不同引用方式也可能合理，这些标记只是审核线索。

人工审核按 5 个维度各 0–2 分：事实忠实、完整性、引用支持、回答边界、约束与表达。评分规则写在 `comparison.md` 和 `manual_reviews.json` 中。逐条填写 `scores`、`reviewer`、`notes`，并将 `status` 改为 `reviewed` 后，可运行：

```bash
python evaluation/summarize_reviews.py --result-dir eval_outputs/validation_round1
```

若有未审核条目，脚本拒绝生成评分汇总；不会把 `null` 当作 0 或通过。汇总是人工量表分数，不是自动准确率，也不把一次 10 步试跑的差异当成已确证的正式微调收益。

## 6. 后续工作

先阅读验证集对照、确定首批答案的业务口径，再依据真实资料覆盖扩充训练数据并制定正式训练配置。正式训练时用验证集选择检查点；模型与提示词确定后，才对冻结测试集做最终评估。

默认命令只运行验证集。测试集生成要求明确传入 `--split test --final-test`，当前阶段不执行它。若已用测试结果改进模型或提示词，需重新准备独立最终测试资料。

## 本机验证与官方依据

本机检查记录见 `evaluation/local_validation.json`。使用随机初始化的小型 Qwen3 的结果只验证脚本功能，不代表真实 8B 问答质量。云端完整验证集结果以本次运行产物为准。

- [Transformers 生成接口](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [PEFT 模型与适配器加载](https://huggingface.co/docs/peft/en/package_reference/peft_model)
