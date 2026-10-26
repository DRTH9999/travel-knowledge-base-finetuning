# 第二轮：补充数据后短训，再复评

状态更新（2026-10-24）：本轮云端10步训练及24题512上限复评已经完成，下载产物已核验。结论是局部改善并伴随新漏答；具体见 `reviews/revision_v2_512/结果审核报告.md`。下面保留10步运行说明供复现。用户已授权下一步完整1轮实验；新的入口和覆盖记录已通过本地小模型验证，最新操作见 `epoch_pipeline/README.md`，8B云端完整1轮待执行。

这轮检查的目标是：加入针对性样例后，旧适配器的漏答、过度拒答、无依据断言和引用问题是否减少，同时检查其他题目是否退步。

原 50 条训练样例保留，新增 32 条，合计 82 条，版本为 `revision_v2`。新增数据仅使用成都、杭州、三亚、云南的 20 份训练资料；厦门验证题和张家界测试题未加入训练。

| 新增类型 | 数量 | 希望改善的行为 |
| --- | ---: | --- |
| 多问与取舍 | 8 | 同时回答优点、不便和用户约束，避免只答一半 |
| 部分可答 | 12 | 先回答有依据的部分，再说明实时信息等缺口 |
| 证据边界 | 8 | 避免由一个地区便利推断另一个地区没有设施，避免承诺未提供的酒店服务 |
| 语义引用 | 4 | 引用支持当前结论的片段，避免为了凑引用复述无关内容 |

## 1. 先查看新样例

打开同目录的 `新增32条样例审核.md`，逐条对照问题、目标回答和原文证据。全部 82 条还在 `data/revision_v2/review_samples.md`。

目前是助手依据资料检查后的审核稿，用户业务审核仍待完成。子需求与回答段落的对应关系属于作者标注，自动检查不能证明语义完整或业务口径正确。资料没有进行外部实时核验，不能把资料记载当作今天的票价、房态或运营状态。

如需修改样例，应新建数据版本并重新构建、导出；不要只改目标回答后直接沿用旧的 Token 文件。

## 2. 上传与检查

将本地 `sft_revision_v2.zip` 上传到 JupyterLab 左侧 `/autodl-tmp/`。该压缩包仅包含本轮新增目录和脚本，不包含模型权重。

在 JupyterLab 终端依次执行：

```bash
cd /root/autodl-tmp
source /root/autodl-tmp/travel-finetune/activate.sh
python -m zipfile -e sft_revision_v2.zip /root/autodl-tmp/travel-finetune/sft_pilot_v1
cd /root/autodl-tmp/travel-finetune/sft_pilot_v1
sha256sum -c revision_pipeline/checksums.sha256
python revision_pipeline/validate_revision.py --require-tokenized
python revision_pipeline/tokenize_revision.py --model-dir /root/autodl-tmp/models/Qwen3-8B --check-only
python training/train_revision.py --validate-only
```

检查结果应包含：

- 校验清单所有文件为 `OK`。
- `REVISION_DATA_CHECK_OK`，总样本 82，原样本 50，新增 32。
- `REVISION_TOKENIZER_CHECK_OK`，最长 833 Token，无截断，输入标签全部屏蔽。
- `REVISION_TRAIN_PREFLIGHT_OK`。

若此前用 `unzip` 后出现英文路径校验正常、中文路径报 `No such file or directory`，先用上面的 Python 解压命令重新解压，再运行校验。原包中的 21 个中文路径文件已核对存在并使用 UTF-8 标记；部分解压环境对中文文件名的处理可能与预期不同。`Frozen file changed` 在这里也可能表示文件不存在。所有校验通过后再训练，不应跳过来源检查。

训练前检查还会比对现有冻结评测清单。原项目应保留已经使用过的 `data/eval_v1/` 和 `evaluation/`。本包不重复打包它们，也不需要重新下载模型或安装依赖。

原 `training/train_sft.py` 继续保留给 50 条试跑；本轮入口为 `training/train_revision.py`，按新数据清单读取样本数量。为了不覆盖旧入口，已验证的通用训练实现按原文件原样复制为 `revision_pipeline/legacy_training_core.py`，新入口只调用其中训练、加载和重载检查函数，不调用其旧的 50 条 CLI 校验。

## 3. 运行一次 10 步短训

看过新增样例且上述检查通过后，运行：

```bash
bash training/run_revision.sh
```

这会从现有 Qwen3-8B 基础模型创建一个新 LoRA，不接着旧的 10 步适配器训练。NF4 双重量化、BF16、SDPA、LoRA r=16 / alpha=32 / dropout=0.05 / all-linear、batch=1、梯度累积=4、学习率 0.0001、随机种子 42、最大长度 2048 均沿用原短训设置。步数仍为 10，先控制实验规模，再用验证集决定后续训练量。

所有 82 条都会进行真实训练批处理器的标签和填充屏蔽检查；10 个优化器步骤通常只涉及约 40 次单条样本训练，**不表示全部 82 条都参与了参数更新**。这轮是短实验，不保证补充的问题已经学会，也不是正式训练收敛结论。

成功终端标记为 `REVISION_TRAIN_OK`，输出：

- `outputs/revision_v2_10steps/adapter/`：新适配器和 Tokenizer。
- `outputs/revision_v2_10steps/revision_report.json`：数据身份、训练配置、梯度检查、参数更新、保存重载结果。
- `outputs/revision_v2_10steps/train_metrics.json`、`trainer_state.json`、`run_config.json`。
- `logs/train-revision-v2-*.log`：终端日志。

训练后的单题生成来自训练样本，只检查能否正常加载和回答，不作为独立质量评估。输出目录非空时脚本会停止，避免覆盖已有结果；需要另做实验可传新目录，例如 `bash training/run_revision.sh outputs/revision_v2_10steps_b`，并在后续评测中同步改用新目录。

## 4. 用原厦门验证集复评

确认训练成功后，在同一终端执行：

```bash
python -u evaluation/evaluate.py \
  --split validation \
  --adapter-dir outputs/revision_v2_10steps/adapter \
  --max-new-tokens 512 \
  --output-dir eval_outputs/revision_v2_512
```

这会按既有脚本对基础模型和新适配器分别生成 24 题，使用冻结证据与原业务提示词，保持非思考模式和 512 新 Token 上限。张家界测试集继续留到最终评估。

终端 `EVAL_GENERATION_OK` 只表示生成完成。需要查看 `eval_outputs/revision_v2_512/comparison.md`，并对比旧的 512 轮适配器回答：重点看 006 的不便是否补齐、012 的寄存建议是否仍错引、017 是否仍编造设施缺失、019 是否能先给已知清单、020 是否明确房态和价格未知；同时检查其余 19 题有无退步。`missing_gold_reference_ids` 等自动标记只是审核提示，不能直接当作准确率或错误题数。

由于此前验证题已经用于发现问题和设计训练方向，这轮验证适合做迭代回归，不能视为独立的最终泛化结论。不要只根据训练 loss 降低或平均审核分提高就进入部署。

## 5. 打包下载结果

训练和验证均完成后执行：

```bash
tar -czf /root/autodl-tmp/qwen3_revision_v2_results.tar.gz \
  outputs/revision_v2_10steps \
  eval_outputs/revision_v2_512 \
  logs \
  data/revision_v2/source_manifest.json \
  data/revision_v2/quality_report.json \
  data/revision_v2/tokenizer_report.json \
  revision_pipeline/checksums.sha256
```

回到 JupyterLab 左侧 `/autodl-tmp/`，刷新后下载 `qwen3_revision_v2_results.tar.gz`，放回本地项目目录，再核对实际产物和逐题变化。

## 已完成的本地验证与边界

- 数据结构、20 份原文片段及行号/哈希、引用编号、原 50 条保留、来源隔离、真实 Qwen3 Tokenizer 标签和 EOS 检查已通过；最长 833 Token，无截断。
- 新入口已在本机随机初始化的两层小 Qwen3 上验证 NF4/BF16 两步训练、82 条批处理标签检查、梯度更新、适配器保存、逐张量重载一致和单题生成功能。
- 本地小模型测试仅验证程序链路，不代表 Qwen3-8B 训练显存、耗时或回答效果。具体本地证据见 `local_validation.json`。本轮8B云端结果现已下载到 `artifacts/revision_v2_original/`，对应训练报告为 `outputs/revision_v2_10steps/revision_report.json`，语义审核位于 `reviews/revision_v2_512/`。

训练标签与量化适配器的官方说明可参照 [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer) 和 [PEFT 量化训练](https://huggingface.co/docs/peft/en/developer_guides/quantization)。本项目实际行为以当前锁定版本的检查和产物为准。
