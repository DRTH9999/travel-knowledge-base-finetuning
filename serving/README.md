# 方案 A：Qwen3 微调模型服务

用户已选择 Transformers + PEFT + bitsandbytes，外层 FastAPI/Uvicorn。首版用于单用户服务兼容性验证，独立于原 RAG 工程。

已完成本地 CPU 协议与生命周期检查、真实 LangChain 客户端调用和 48 份真实 Qwen3 Tokenizer 输入核对。本地模拟生成结果不能证明 AutoDL 的 8B 模型、GPU、连接或真实 RAG 已通过；云端需按下列步骤验证。

## 一、上传与首次启动

将 `qwen3_model_service_a_v1.zip` 上传到 JupyterLab 的 `/root/autodl-tmp/`。首次解压使用独立目录 `/root/autodl-tmp/serving_v1`，原训练工程不移动。如果该目录已经存在且有正在运行的服务，先确认目录状态，不重复覆盖。

在云端终端执行：

```bash
cd /root/autodl-tmp
python -m zipfile -e qwen3_model_service_a_v1.zip .
cd /root/autodl-tmp/serving_v1
bash serving/setup_cloud.sh
bash serving/run_cloud.sh
```

`setup_cloud.sh` 会校验上传文件，核对原训练环境的 ML 版本，创建独立的 `.serving-venv` 并复用已验证的 ML 包。仅增加 HTTP 依赖，不重装 Torch、Transformers、PEFT 或 bitsandbytes。它还会生成私有 API key 文件，不在终端打印密钥。

安装后应先看到 `SERVING_PREFLIGHT_OK` 和 `SERVING_SETUP_OK`。预检查只检查候选、48 份 Tokenizer 输入、分片是否存在和小规模 CUDA/BF16 运算，不加载 8B 权重。模型权重由 `run_cloud.sh` 启动时真正加载。

看到 Uvicorn 的 `Application startup complete` 后，保持这个终端运行。服务默认监听 `127.0.0.1:8008`。停止时在该终端按 Ctrl+C，待进程退出后再启动。

默认读取既有文件：

| 项目 | 云端路径 |
| --- | --- |
| 已验证训练 Python | `/root/autodl-tmp/travel-finetune/.venv/bin/python` |
| 基础模型及原 Tokenizer | `/root/autodl-tmp/models/Qwen3-8B` |
| 完整 1 轮适配器 | `/root/autodl-tmp/travel-finetune/sft_pilot_v1/outputs/revision_v2_epoch1/adapter` |

路径确有变化时，可设置 `TRAVEL_BASE_PYTHON`、`TRAVEL_MODEL_DIR`、`TRAVEL_ADAPTER_DIR`。版本或哈希不符时程序会停止，不会自动选择其他模型、适配器或降级到 CPU。

## 二、云端普通与流式检查

另开一个 JupyterLab 终端执行：

```bash
cd /root/autodl-tmp/serving_v1
bash serving/check_cloud.sh
```

该命令会用真实 GPU 发起两次生成，比较普通回答与流式拼接结果，核对输入哈希、Token 用量、结束事件，并检查未授权、错误模型和不支持的生成参数。成功标记为 `SERVING_HTTP_SMOKE_OK`，报告保存到 `serving_runtime/http-smoke-*.json`。

通过后先回放 3 道既有验证题：

```bash
.serving-venv/bin/python -m serving.replay --split validation --limit 3
```

看到 `SERVING_HTTP_REPLAY_OK` 表示 HTTP 请求和输入核对完成，不表示业务质量通过。`SAME_TEXT` 表示与此前离线输出文本一致；`REVIEW_DIFFERENCE` 要保留原文供审核，不自动判断新答案好坏。

完整回放入口：

```bash
.serving-venv/bin/python -m serving.replay --split validation
.serving-venv/bin/python -m serving.replay --split test --ack-test-is-regression
```

测试题已经审核过，因此这里只用于服务回归，不当作新的独立测试。先检查前 3 题结果，再决定是否运行全部题目。

## 三、打包云端结果

完成预检查和 HTTP 检查后，在第二个终端执行：

```bash
cd /root/autodl-tmp/serving_v1
.serving-venv/bin/python -m serving.collect_results
```

生成 `/root/autodl-tmp/qwen3_service_a_v1_results.tar.gz`，可在 JupyterLab 文件列表下载。该包包含环境、候选、请求指标与检查/回放结果；不包含 API key、模型权重或 Python 环境。已有同名结果包时不会覆盖，可用 `--output` 指定新名称。

## 四、本机连接（云端检查通过后）

云端 `127.0.0.1` 指云服务器自己，本机不能直接用它访问云端服务。建议通过 AutoDL 提供的实际 SSH 信息建立端口转发；下列主机、端口需要换成控制台中的值：

```powershell
ssh -N -L 18008:127.0.0.1:8008 -p <SSH端口> root@<SSH主机>
```

保持转发终端运行，本机接口地址为 `http://127.0.0.1:18008/v1`。本机先准备私有目录，再通过另一个终端复制密钥文件：

```powershell
New-Item -ItemType Directory -Force -Path serving_runtime
scp -P <SSH端口> root@<SSH主机>:/root/autodl-tmp/serving_v1/serving_runtime/api.key serving_runtime/api.key
```

密钥只存本地文件，不需要贴到聊天或截图中。然后用已具备 httpx 的本机 Python 运行：

```powershell
python -m serving.smoke --base-url http://127.0.0.1:18008/v1
```

本项目用于检查的 Python 也可用：`.cache/training-check-env/Scripts/python.exe`。它不是用于加载本地 8B 模型，只执行 HTTP 客户端。本机检查通过后，才在原 RAG 中为最终回答节点加入独立配置。原 RAG 代码本轮未改动。

## 五、接口与首版边界

| 接口 | 行为 |
| --- | --- |
| `GET /healthz` | 进程存活状态，不等于模型可用 |
| `GET /readyz` | 模型是否可用、是否忙碌；异常或过期未退出的任务返回不可用 |
| `GET /v1/models` | 模型别名 `qwen3-travel-epoch1`，需要 Bearer key |
| `GET /v1/identity` | 实际候选哈希、运行版本与推理设置，需要 Bearer key |
| `POST /v1/chat/completions` | 普通或 SSE 流式回答，需要 Bearer key |

首版每个请求只接收一条文本 user 消息。完整业务规则、检索证据、历史和当前问题都在这条消息里，与既有训练/评测保持一致。暂不支持工具、多模态、任意多轮角色消息或采样。

生成固定非思考、贪心和单 beam。`temperature` 只能为 0 或省略；`max_tokens` 或 `max_completion_tokens` 可指定 1–512，不能同时提供。未知参数会明确报错，不默默忽略。

输入加请求输出预算上限暂为 2048 Token，与既有固定输入兼容性检查一致。超限返回 400，不静默截断。这个限制用于第一批服务验证，不能据此认定真实 RAG 长输入已可用；长输入分布、裁剪策略与扩大上下文要在后续单独验证。

一个进程只持有一份模型、一个活动生成任务。其他生成请求返回 429，HTTP 客户端可在任务结束后重试。健康查询仍可响应。首版没有并发批处理，也不增加 Uvicorn worker。

默认整个请求限时 120 秒；停止条件会在生成步之间响应取消。GPU 内核不能被 Python 强行中断，因此超时/断线时不会提前释放仍在工作的 GPU 槽位。超过宽限时间仍未退出则 readiness 变为 503，需检查进程；无法继续时停止并重启服务。CUDA 异常也会标记为不可用，不自动冒充恢复成功。

流式开始后发生错误，会输出 error 事件并结束，不能保证 HTTP 状态码还能改成错误码；客户端应检查流错误，不能仅凭 200 判断成功。`smoke.py` 会把流中的 error 视为失败；异常流处理已做本地模拟检查，真实 GPU 异常与恢复仍待云端验证。

## 六、文件与验证记录

- `candidate.json`：固定候选与原 Tokenizer 资产哈希、ML 版本。
- `compatibility.jsonl`：48 份已评估输入、预期 Token 哈希与此前回答；不是新训练数据。
- `local_validation.json`：本机检查范围、版本和结果，明确区分模拟接口与真实模型。
- `serving_runtime/preflight.json`、`server_identity.json`、`requests.jsonl`：云端核验与请求指标。
- `serving_runtime/api.key`：私有文件，打包工具不收集。

基础模型仅核对配置、Tokenizer 与分片存在性，没有重新计算全部基础权重哈希；适配器权重会实际重算 SHA256。依赖解析/安装、小规模 CUDA 和完整模型启动分别记录，不能合并称为生产验收。

本轮不调用 qwen-flash，不运行真实 RAG 的上游付费模型或联网检索，也未修改最终测试集、训练数据或提示词。
