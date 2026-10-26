# 首批旅游 RAG SFT 数据

本目录包含 50 条助手依据本地旅游文档编写的样例定义、构建脚本、独立格式检查和 Qwen3 Tokenizer 检查。只面向第一阶段的最终回答生成。没有调用付费生成 API，没有加载大模型权重，也没有启动训练。

**当前交付状态（2026-10-23）：50 条样例已导出到 `data/pilot_v1/`，构建、来源与引用结构校验、真实 Qwen3 Tokenizer 导出、loss 标签检查和 ZIP 打包均已在本机实际执行通过，输出 `PILOT_BUILD_AND_CHECK_OK`。** 可上传的数据包为根目录的 `sft_pilot_v1.zip`。助手已逐条对照答案与原文证据，业务审核状态仍为 `assistant_checked_user_pending`；尚未取得云端 Tokenizer 复核结果。以下表格中的文件均已生成。

本次检查结果：

- 50 条：单片段问答 15、多片段整合 12、多条件规划 8、历史追问 5、信息不足或部分可答 8、干扰筛选 2。
- 成都 13、杭州 12、三亚 13、云南 12；使用训练来源 20 份文档，92 次片段引用，对应 72 个不同的原文摘录。
- 42 条可答、4 条部分可答、4 条不可答；完全重复问题和答案均为 0，问题字符串相似度不低于 0.8 的样例对为 0。
- 预留验证和测试来源进入训练的数量为 0；原文整行重复检查发现 7 组重复，均在训练来源内部，未发现该检查口径下的跨集合重复；这不能证明不存在语义重叠。
- 使用本地 Transformers 4.57.6、tokenizers 0.22.2 和缓存中的 Qwen3-8B Tokenizer：每条共 422–833 tokens，监督区总计 3395 tokens，截断数量为 0；全部 prompt labels 为 `-100`，completion 末尾恰好保留一个 `<|im_end|>`。

可以重新执行 `data_pipeline/build_and_check.ps1` 复现构建、校验和打包；脚本不启动训练，各阶段失败即停止。

## 数据边界

- 成都、杭州、三亚、云南的 20 份文档作为首批训练来源。
- 厦门 5 份预留验证，张家界 5 份预留测试。此安排是首批数据的来源隔离，不代表完整评测设计已冻结；尚未生成正式验证集和测试集。
- 同一片段可用于不同问题或边界测试，50 条样例不等于 50 组独立知识。
- 文档缺少明确更新时间和权威来源，价格、房态、班次、天气等不能当作实时知识生成。
- 证据由助手按标题选择，是原文摘录；没有运行 RAG 检索来构造真实召回上下文。
- 历史对话全部合成。第 038 条故意放入错误的历史回答，训练根据当前资料纠正旧回答；没有改动原文伪造证据冲突。
- 状态为 `assistant_checked_user_pending`。格式检查不等于业务质量和语义蕴含检查；用户审核后再扩充。

## 三种导出，训练时选其中一种

数据目录：`E:\260515\知识库项目模型微调\data\pilot_v1`。

| 文件 | 内容和用途 |
| --- | --- |
| `review_samples.md` | 按题查看问题、历史、答案、来源全文片段、行号和引用 |
| `samples.raw.jsonl` | 保留任务、来源、原始问题、改写问题、历史、答案、逐段引用依据与审核状态 |
| `train.messages.jsonl` | 对话型 prompt/completion，一条 user 业务提示词与一条 assistant 答案；便于阅读和重新渲染 |
| `train.sft.jsonl` | 使用真实 Qwen3 模板，以 `enable_thinking=False` 预渲染的字符串 prompt/completion，供 TRL 的标准 prompt-completion 流程使用 |
| `train.tokenized.jsonl` | 与预渲染版等价的 input_ids、attention_mask 和 labels；仅答案及终止 EOS 参与 loss |
| `source_manifest.json`、`sources/` | 30 份来源的路径、目的地、类别、哈希和原文件快照，预留来源不会进入训练样例 |
| `prompt_template.txt` | 原项目 `ANSWER_PROMPT` 的字面值快照，未导入或运行业务项目 |
| `quality_report.json` | 数量、覆盖、引用、来源隔离、重复与局部数值异常检查，以及未解决的限制 |
| `tokenizer_report.json` | 本地真实 Tokenizer 的模板、版本、哈希、长度和 loss 标签检查 |
| `source_duplicate_report.json` | 原文中长度至少 30 字的非标题、非列表整行重复检查，不证明无语义泄漏 |
| `checksums.sha256` | 已生成文件的完整性清单 |

三个训练文件是同一批 50 条样例的不同表达，不能合并成 150 条。

业务输入遵循源码：`[编号][local][目的地][]` 后接证据正文；历史以 `[role]: 文本` 表达；实体逗号拼接；用户问题使用改写后的问题。与现有答案节点一样，整段业务提示词放入一个 user 消息。原始问题与证据角色只在审核元数据中保留，不向模型泄露“正确片段”标签。

非思考模式的 Qwen3 聊天模板可能在 prompt 末尾插入空的 `<think>...</think>` 预填段；这是模板自动生成的前缀，全部屏蔽 loss。目标 completion 只包含直接答案和一个结束 EOS，没有思维过程。预渲染文件已含聊天特殊标记，不应再包一层聊天模板或手工加 BOS/EOS。

初次小样本流程可采用 TRL 的 `completion_only_loss=True`、`packing=False`；`max_length` 至少覆盖 `tokenizer_report.json` 的最大长度。正式训练参数另行制定，本包不启动训练。前缀和答案不得直接截断，尤其不能截掉问题、目标答案或结束标记。

## 本地复现

构建脚本仅使用 Python 标准库。Tokenizer 检查使用已选技术栈中的 Transformers，不安装新的训练框架。

```powershell
Set-Location -LiteralPath 'E:\260515\知识库项目模型微调'
python data_pipeline/build_pilot.py
python data_pipeline/validate_pilot.py
& 'E:\260515\travel_knowledge_base\.venv\Scripts\python.exe' data_pipeline/tokenizer_export.py --model-dir '.cache/qwen3-8b-tokenizer'
python data_pipeline/validate_pilot.py --require-tokenized
```

原文和业务模板一旦冻结，如果后续改变，脚本会停止并要求新建数据版本，不能静默覆盖来源。修改首批样例后可在来源不变的情况下重新构建，但需要重新导出 Tokenizer、运行检查并重新审核。

本地 Tokenizer 缓存仅包含从 ModelScope 下载的 Qwen3-8B 配置和分词器文件，不是完整模型。云端依赖版本与本地可能不同；本地通过不代表云端已通过。

## 模型下载完成后的云端检查

把数据包解压到 `/root/autodl-tmp/travel-finetune/sft_pilot_v1` 后，在已经配置好的环境中执行以下检查。模型目录以实际下载位置为准。

```bash
source /root/autodl-tmp/travel-finetune/activate.sh
cd /root/autodl-tmp/travel-finetune/sft_pilot_v1
python data_pipeline/validate_pilot.py --require-tokenized
python data_pipeline/tokenizer_export.py \
  --model-dir /root/autodl-tmp/models/Qwen3-8B \
  --dataset-dir data/pilot_v1 \
  --check-only
```

第二条检查会按云端 Tokenizer 重新渲染并比对字符串、Token IDs、EOS 和标签边界，不加载模型权重、不占用训练显存。如果与导出不一致会报错，不会静默重写训练文件。只有看到实际输出 `TOKENIZER_CHECK_OK`，才算完成云端复核。本轮没有取得该云端结果。

先审核样例，再扩充到约 1000 条。只有 30 份短文档，扩充规模需要受独立事实、条件和实际召回场景约束，不能靠重复改写凑数。正式训练前还需要独立评测题、qwen-flash 与未微调 Qwen3-8B 的同题基线，以及真实召回失败样例。

格式依据：[TRL 数据格式](https://huggingface.co/docs/trl/en/dataset_formats)、[TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)、[Qwen3-8B 官方 Tokenizer 配置](https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json)。
