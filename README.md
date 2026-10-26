# 旅游知识库模型微调

基于 Qwen3-8B 的旅游 RAG 回答模型微调项目，包含证据约束 SFT 数据、QLoRA 小步训练与完整一轮实验、固定证据评估、结果审核，以及 OpenAI 兼容模型服务代码。

本仓库对应“知识库项目模型微调”工作目录。原始旅游知识库应用位于另一个项目中，不包含在本仓库内。

## 当前记录

- 初始数据集 `pilot_v1` 有 50 条样本；修订数据集 `revision_v2` 保留原样本并新增 32 条，共 82 条。
- 训练资料覆盖成都、杭州、三亚、云南；厦门用于验证，张家界用于最终固定证据测试。
- 当前服务候选为完整一轮训练的 LoRA 适配器，使用原业务提示词、非思考模式及 NF4/BF16 配置。候选哈希和运行版本见 [`serving/candidate.json`](serving/candidate.json)。
- 张家界最终固定证据测试的生成、文件核验和助手审核已有记录，详见 [`结果审核报告`](reviews/final_test_epoch1_original512/结果审核报告.md)。该审核不是独立人工盲审或生产验收。
- 服务目录记录了本地协议、生命周期、客户端及 Tokenizer 验证。现有 [`本地验证记录`](serving/local_validation.json) 不包含真实 8B GPU 服务启动、远程连接或真实 RAG 联调通过的证明。

## 目录导航

| 路径 | 内容 |
| --- | --- |
| [`autodl_setup/`](autodl_setup/README.md) | AutoDL 环境准备、模型下载与推理冒烟检查 |
| [`data_pipeline/`](data_pipeline/README.md) | 初始 50 条数据的构建、校验与 Tokenizer 检查 |
| [`data/`](data/) | 训练、验证、测试数据及来源快照、模板和校验清单 |
| [`training/`](training/README.md) | 初始训练、修订训练和完整一轮实验入口 |
| [`revision_pipeline/`](revision_pipeline/README.md) | 32 条补充样本和修订数据流程 |
| [`epoch_pipeline/`](epoch_pipeline/README.md) | 完整一轮训练、样本覆盖核验与结果收集 |
| [`evaluation/`](evaluation/README.md) | 固定证据评估、生成和审核汇总工具 |
| [`prompt_pipeline/`](prompt_pipeline/README.md) | 固定模型下原提示词与候选提示词的对照实验 |
| [`reviews/`](reviews/) | 各轮审核、逐题对照和后续建议 |
| [`artifacts/`](artifacts/) | 已下载实验的文本记录、运行配置、预测和原始文件快照 |
| [`serving/`](serving/README.md) | Transformers、PEFT、bitsandbytes 与 FastAPI/Uvicorn 服务 |

完整实施背景见 [`微调项目实施规划`](微调项目实施规划.md) 和 [`模型服务与 RAG 联调实施规划`](模型服务与RAG联调实施规划.md)。历史规划和阶段文档保留各自当时的状态，应结合后续实验记录阅读。

## 使用方式

克隆仓库：

```bash
git clone https://github.com/DRTH9999/travel-knowledge-base-finetuning.git
cd travel-knowledge-base-finetuning
```

先阅读对应目录的 README，再按阶段使用数据、训练、评估或服务入口。现有脚本保留了原 AutoDL 实验目录和冻结依赖，不能把克隆仓库视为已具备 GPU 运行环境。基础模型、LoRA 权重和运行路径需要按对应说明准备。

服务配置与启动步骤见 [`serving/README.md`](serving/README.md)。服务支持普通和流式回答，云端预检、HTTP 检查和回放仍应在实际部署环境中执行。

## Git 上传范围

仓库保留源代码、数据、文档、评估报告、文本实验结果及校验清单。以下本地文件由 `.gitignore` 排除：

- Python 环境、依赖缓存、字节码和测试缓存。
- 基础模型与 LoRA 权重、适配器目录内重复的大型 Tokenizer 文件。
- ZIP、TAR 等传输压缩包及 `serving_runtime/` 中的重复服务解压目录。
- `.env`、密钥和本机编辑器配置。

这些文件仍保留在原本地目录；它们不会随 `git clone` 下载。`artifacts/` 中的历史校验清单可能引用被排除的权重、Tokenizer 或原始压缩包，完整复核这些项目需要另行准备相应文件。

`.gitattributes` 保留文件原始字节，避免 Git 自动转换换行符破坏已有 SHA256 校验清单。
