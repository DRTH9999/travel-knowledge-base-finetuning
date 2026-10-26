# 50 条数据的 SFT + QLoRA 小步试跑

用户已确认云端 `DATA_CHECK_OK`、`TOKENIZER_CHECK_OK`、`MODEL_SMOKE_OK` 三项成功。现在验证真实优化器更新、答案 loss 标签、LoRA 保存和重载。默认 10 个优化器步，不把本次输出视为微调效果评测。

本目录沿用已选技术栈：Qwen3-8B、TRL 1.16.0、Transformers 5.20.2、PEFT 0.21.2、bitsandbytes。没有重新安装依赖或下载基础模型的命令。

## 上传与执行

将本机根目录的 `sft_training_smoke_v1.zip` 上传到 AutoDL 的 `/root/autodl-tmp/`。它只包含本目录的训练脚本和说明，使用云端已经解压好的 `sft_pilot_v1/data/pilot_v1` 数据。

在 AutoDL Bash 终端依次执行；若某条报错，先处理错误，不继续下一条。

```bash
source /root/autodl-tmp/travel-finetune/activate.sh
python -m zipfile -e \
  /root/autodl-tmp/sft_training_smoke_v1.zip \
  /root/autodl-tmp/travel-finetune/sft_pilot_v1
cd /root/autodl-tmp/travel-finetune/sft_pilot_v1
sha256sum -c training/checksums.sha256
python training/train_sft.py --validate-only
bash training/run_smoke.sh
```

`--validate-only` 只检查来源、数据及 loss 标签，成功时输出 `TRAINING_DATA_PREFLIGHT_OK`；不加载模型、不执行训练。

`run_smoke.sh` 激活已有 venv，实时显示日志并保存到 `logs/train-smoke-日期时间.log`。Python 脚本先复核数据和云端 Tokenizer，再执行训练。整个脚本成功才会输出 `TRAIN_SMOKE_OK`，失败以非零状态退出并保留日志；不是让用户手动执行这个成功标记。

## 本次试跑设置

| 项目 | 默认值 |
| --- | --- |
| 数据 | 已生成的 50 条预分词数据，全部作为试跑训练数据 |
| 优化器步数 | 10；脚本只允许 1–20 步 |
| 每次前向/反向的 batch size | 1 |
| 梯度累积 | 4；单 GPU 有效 batch size 为 4 |
| 学习率 | 0.0001，constant；仅为链路短测初始值 |
| LoRA | r=16、alpha=32、dropout=0.05，覆盖 all-linear |
| 基础模型 | NF4、双重量化、BF16 计算、SDPA |
| 最大长度 | 2048；当前最长 833 tokens，超长样本会报错，不自动截断 |
| 显存控制 | 梯度检查点开启，packing 关闭 |
| 训练目标 | 使用原有 labels，仅答案与结束 EOS 参与 loss |
| 保存 | 训练结束保存一份 LoRA 与 Tokenizer，不保存完整基础模型或优化器检查点 |

10 步指优化器更新 10 次；梯度累积为 4 时，每次更新前通常有 4 个微批次的前向和反向。loss 不要求每一步单调下降，但必须有限；脚本会检查每个优化器步具有有限且非零的 LoRA 梯度，并验证权重确实发生更新。

这里传入的是带 `labels` 的预分词数据，`skip_prepare_dataset=True` 保留标签，`completion_only_loss=False` 表示不再根据文本重新推断 completion mask；它不代表对整个输入计算 loss。脚本会实际检查 TRL collator 处理后的全部 50 条标签及补齐位置，发现输入未屏蔽就停止。

## 成功后查看什么

每次运行新建 `outputs/smoke_日期时间/`，已有非空输出目录不会被覆盖。

- `smoke_report.json`：最终状态、逐步 loss、梯度检查、实际权重更新、训练耗时、显存峰值和重载后的生成结果。
- `run_config.json`：实际试跑参数、训练参数数量和数据 collator 检查结果。
- `train_metrics.json`、`trainer_state.json`：训练指标与状态；这些不是独立验证集分数。
- `adapter/adapter_model.safetensors`、`adapter/adapter_config.json`：LoRA 权重与配置；同目录保存 Tokenizer。

保存后会先释放训练模型，再从本地基础模型目录重新加载模型与 LoRA，逐张量比较保存和重载的适配器权重，然后用第 005 条训练问题验证能生成回答。该回答只检查功能，不证明泛化能力或质量提升。

如果遇到显存不足，保留日志，使用新的输出目录把 LoRA 覆盖范围、序列长度或其他设置作为下一次实验单独调整；不要通过截断目标答案让脚本表面通过。当前 batch size 已是 1，减小梯度累积通常不能降低单个微批次的主要激活显存。

## 后续顺序

1. 取得本次实际 `TRAIN_SMOKE_OK` 和报告，确认训练、保存和重载链路可用。
2. 审核首批业务口径，按真实资料覆盖扩充训练数据，并独立构建厦门验证题、张家界测试题。
3. 在相同固定证据下记录未微调 Qwen3-8B 的基线；qwen-flash 对照按已确认的 API 预算执行。
4. 再制定正式训练轮数、检查点评选与质量验收标准，执行正式训练和同题评测。

50 条样本的 10 步试跑只验证链路。独立评测完成前，不将该适配器接入原旅游 RAG 替换现有答案模型。

## 官方依据

- [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [PEFT 量化与 QLoRA](https://huggingface.co/docs/peft/en/developer_guides/quantization)

本机验证范围会单独记录在 `local_validation.json`；云端 Qwen3-8B 的真实 NF4 训练成功以用户执行结果为准。
