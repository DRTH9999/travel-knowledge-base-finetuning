# 同一82条数据：完整训练1轮并复评

本轮沿用现有 `revision_v2` 数据、Qwen3-8B、QLoRA/NF4/BF16、LoRA r=16/alpha=32/dropout=0.05/all-linear、batch=1、梯度累积4、学习率0.0001、种子42、最大长度2048及原业务提示词。从基础模型新建适配器，完整训练1轮，再用原厦门24题与512新Token上限复评。

这是在相同数据下检查增加训练覆盖的效果。完整1轮不保证收敛、旧问题修复或达到部署要求；后续仍需逐题核对。上一轮的17题设施编造、20题房态边界，以及新出现的2题/10题漏答都要继续检查。

## 上传并运行

将本地 `sft_revision_v2_epoch1.zip` 上传到 JupyterLab 左侧 `/autodl-tmp/`。这是一份小更新包，只新增 `epoch_pipeline/` 和三个新的训练脚本，复用云端现有数据、环境、模型及评测程序。

终端执行：

```bash
source /root/autodl-tmp/travel-finetune/activate.sh
python -m zipfile -e /root/autodl-tmp/sft_revision_v2_epoch1.zip /root/autodl-tmp/travel-finetune/sft_pilot_v1
cd /root/autodl-tmp/travel-finetune/sft_pilot_v1
bash training/run_epoch_experiment.sh
```

脚本会顺序执行：

1. 校验新增文件与现有训练依赖，确认数据仍为原82条冻结版本。
2. 完整训练1轮并记录样例实际进入前向/反向计算的情况。
3. 复核82条覆盖、最后2条累积、21步更新及适配器保存重载。
4. 运行原评测脚本，分别生成基础模型与新适配器的24题验证回答，使用512新Token上限。
5. 打包训练、覆盖轨迹、预测、日志、数据及运行代码，输出结果压缩包。

任何一步失败都会停止，保留当时的输出和日志，不会继续跑后面的步骤。预先检查输出目录与结果包不存在，防止覆盖旧结果。

## 如何识别完成

按顺序看到：

```text
EPOCH_TRAIN_PREFLIGHT_OK
EPOCH_COVERAGE_OK
REVISION_EPOCH_TRAIN_OK
EVAL_GENERATION_OK
EPOCH_EXPERIMENT_COMPLETE
```

覆盖结果应为：82个不同样例，每个一次；原50条、新增32条全部覆盖；21个优化器步骤；最后一步处理2个单条批次；epoch=1.0。

这些标记表示程序、覆盖和生成检查通过。回答效果还需对照上一轮逐题审核，不能直接把 `EVAL_GENERATION_OK` 当成质量达标。

## 下载结果

最后一项成功后，刷新 JupyterLab 左侧 `/autodl-tmp/`，下载：

```text
qwen3_revision_v2_epoch1_results.tar.gz
qwen3_revision_v2_epoch1_results.tar.gz.sha256
```

将结果包放到本地 `E:\260515\知识库项目模型微调\`，用于对照旧50条10步、82条10步和本轮82条完整1轮的变化。结果包内包含实际运行代码、训练数据和覆盖记录，便于检查运行身份与来源。

## 分步操作与异常处理

只做检查，不加载模型：

```bash
python training/train_revision_epoch.py --validate-only
```

只运行训练及覆盖复核：

```bash
bash training/run_revision_epoch.sh
```

仅在上述训练已经成功、验证输出目录尚未生成时，单独运行复评：

```bash
python -u evaluation/evaluate.py \
  --split validation \
  --adapter-dir outputs/revision_v2_epoch1/adapter \
  --max-new-tokens 512 \
  --output-dir eval_outputs/revision_v2_epoch1_512
```

单独核对已有训练覆盖及适配器文件：

```bash
python epoch_pipeline/verify_coverage.py --output-dir outputs/revision_v2_epoch1
```

如整套实验已生成部分结果，需要从头另跑时，选择一个新名称：

```bash
bash training/run_epoch_experiment.sh revision_v2_epoch1_b
```

训练目录、验证目录和最终包名会一起换名。旧目录继续保留，默认名称不再覆盖。

若提示 `Existing project dependency changed or missing`，先按报错核对现有项目文件；不要绕过哈希检查。原来的50条与82条10步入口继续保留；完整轮数入口是新的 `training/train_revision_epoch.py`，只允许1轮，不接收 `--max-steps`。

## 训练与覆盖实现

轮数模式使用 `num_train_epochs=1`、`max_steps=-1`，避免正的固定步数提前结束训练；正的 `max_steps` 会覆盖轮数设置。[Trainer官方说明](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments)

采用已有的标签、填充与梯度检查，保持原优化器及学习率配置。82个单条批次，每4批更新一次，共21次更新，最后一次只有2批。锁定版本运行时的实际累积长度与监督Token归一化数量会一并记录并核对。

样例索引只用于跟踪，在调用模型前移除；模型前向钩子会再次确认跟踪字段没有进入模型。每条记录对应一次已完成的前向和反向计算，优化器更新记录再确认其对应的梯度检查和未跳过的更新。数据预读及全量标签检查都不计入训练覆盖。梯度检查点内部的重算也不会被重复记为新样例。

训练产物：

- `epoch_report.json`：初始化方式、数据和代码身份、训练指标、保存重载。
- `coverage_report.json`：82条覆盖、原50条/新增32条统计、每条次数、最后一组更新。
- `training_trace.jsonl`：真实训练批次的ID、输入/标签指纹、监督Token数及归一化信息。
- `optimizer_trace.jsonl`：每次优化器更新所对应的实际批次及梯度检查。
- `run_config.json`、`trainer_state.json`、`train_metrics.json`、`adapter/`。

## 本地检查范围

实际锁定版本和随机两层小Qwen3已完成82条完整1轮NF4/BF16训练、21次梯度检查、原50/新增32条各一次覆盖、最后2条累积、适配器保存重载及单题生成功能检查。测试证据见 `local_validation.json`。

小模型只验证程序链路，不能代表8B回答质量、训练显存或耗时。本轮8B云端完整1轮及复评尚未执行。张家界测试集继续保留到最终评估。本轮不修改业务提示词，也不调用付费生成接口。
