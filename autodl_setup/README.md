# AutoDL：Qwen3-8B 训练环境准备

用户已选择：bitsandbytes + venv。官方源和阿里云镜像的 Torch 主文件都下载缓慢后，用户已同意 venv 使用 `--system-site-packages` 读取原镜像已有的包，复用已验证的 Torch。模型从魔搭社区下载，首阶段使用非思考模式。

当前交付是云端可执行的安装和检查文件。用户后续截图已确认 `ENV_CHECK_OK`、`SETUP_OK` 与完整 Qwen3-8B 的 `MODEL_SMOKE_OK`，并确认数据和 Tokenizer 检查成功。环境与短推理通过后，下一步见 [训练试跑说明](../training/README.md)，真实 Qwen3-8B 训练兼容性仍以云端试跑结果为准。

## 1. 已知云端环境

以下来自用户 2026-10-23 提供的终端输出：

| 项目 | 当前情况 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 D，实例提供约 48GB 显存 |
| NVIDIA 驱动 | 580.105.08 |
| 原环境 Python | 3.12.3，`/root/miniconda3/bin/python` |
| 原环境 PyTorch | 2.12.1+cu130；CUDA 运行时 13.0 |
| GPU 检查 | CUDA 可用、BF16 可用、计算能力 8.9、基本 CUDA 运算通过 |
| 数据盘 | `/root/autodl-tmp`，50GB，检查时基本空闲 |

当前默认方案使用已有 Python 3.12 创建允许读取基础环境包的 venv。新训练包仍安装在 venv 中，基础环境中的其他包也会可见；这不是完全隔离。安装步骤先验证 Torch 的版本和可用性，然后复用已有安装，不再显式下载 Torch wheel。基础环境本身不作为 pip 安装目标。

## 2. 目录安排

| 云端路径 | 用途 |
| --- | --- |
| `/root/autodl-tmp/autodl_setup` | 本安装包中的脚本和说明 |
| `/root/autodl-tmp/travel-finetune/.venv` | 训练 venv，可读取原镜像已有包 |
| `/root/autodl-tmp/travel-finetune/activate.sh` | 新终端的环境激活与缓存配置 |
| `/root/autodl-tmp/travel-finetune/cache` | 模型工具与数据缓存 |
| `/root/autodl-tmp/travel-finetune/tmp` | 安装和运行的临时文件 |
| `/root/autodl-tmp/travel-finetune/logs` | 环境报告、已安装版本和模型加载报告 |
| `/root/autodl-tmp/models/Qwen3-8B` | 从魔搭下载的模型快照 |

魔搭当前仓库有 5 个 safetensors 权重分片，合计约 16.38GB（15.26GiB），还需要环境、缓存和临时空间。安装使用 `--no-cache-dir` 减少 pip 额外缓存。实际余量用 `df -h /root/autodl-tmp` 检查；后续训练的检查点和导出空间另行评估。

## 3. 上传并安装

在 JupyterLab 左侧进入 `autodl-tmp` 文件夹，将本机的 `qwen3_autodl_env.zip` 上传到这里。

然后在 AutoDL 的 Bash 终端执行：

```bash
/root/miniconda3/bin/python -m zipfile -e /root/autodl-tmp/qwen3_autodl_env.zip /root/autodl-tmp
bash /root/autodl-tmp/autodl_setup/setup_env.sh
```

新版脚本默认转入 `reuse_base_torch.sh`：验证基础环境 Torch、启用 venv 的基础包可见性、安装缺少的固定版本依赖、运行 `pip check`，最后执行 `check_env.py`。失败时会退出；应先保留报错并处理，不继续下载或模型检查。脚本不使用 `--upgrade` 或 `--ignore-installed` 强制重装训练依赖。

如果云端仍是最初上传的旧脚本，可使用第 8 节的续装命令，无需重新上传压缩包。旧脚本中的完全隔离检查不适合当前共享包方案。

基础包版本：

| 包 | 版本 |
| --- | --- |
| torch | 2.12.1+cu130，默认复用原镜像已安装的版本 |
| transformers | 5.20.2 |
| trl | 1.16.0 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |
| datasets | 5.1.0 |
| bitsandbytes | 0.50.2 |
| modelscope | 1.40.2 |

这些是本轮选用的固定版本。已核对直接包之间声明的版本要求；间接依赖由 pip 解析，并在成功安装后写入 `logs/installed-requirements.txt`，同时用 `logs/venv-local-requirements.txt` 单独记录 venv 内新增的依赖。新版环境报告还记录 venv 配置、Torch 文件路径及各包来源目录。基础镜像和新包的实际组合仍需要检查，不把这些清单当作已经完成训练验证的环境锁定文件。

成功时依次看到：

```text
ENV_CHECK_OK
SETUP_OK
```

`ENV_CHECK_OK` 表示：当前 venv 中的公开 API 导入、CUDA/BF16 以及小型 NF4 层前向和输入梯度计算通过；此时还没有加载 Qwen3-8B。

## 4. 从魔搭下载模型

安装成功后，在当前终端执行：

```bash
source /root/autodl-tmp/travel-finetune/activate.sh
python -u /root/autodl-tmp/autodl_setup/download_model.py
```

下载的是 `Qwen/Qwen3-8B` 官方原始模型。后续由 bitsandbytes 在加载时进行 4-bit 量化。

下载脚本使用 ModelScope SDK 的 `snapshot_download` 和显式 `local_dir`。它会检查配置、Tokenizer 文件、索引中的权重分片是否存在且非空，并生成 `download_manifest.json`。文件齐全检查不等于所有权重的完整哈希核验；实际可加载性由下一步检查。

成功时看到：

```text
MODEL_DOWNLOAD_OK: /root/autodl-tmp/models/Qwen3-8B
```

默认请求 `master` 分支，分支名会变化。保留下载目录中的 SDK 元数据和原始模型文件，后续训练复用这个快照。脚本支持 `--revision`，在确定不可变版本后可以显式指定。

终端断开或下载中断后，重新激活环境，再运行同一下载命令，由 SDK 检查已有文件。恢复情况取决于 SDK 对完整文件与未完成文件的处理；不手动删除模型目录重来。

## 5. 模型加载与非思考模式检查

下载成功后执行：

```bash
source /root/autodl-tmp/travel-finetune/activate.sh
python -u /root/autodl-tmp/autodl_setup/smoke_model.py
```

这一步只读取本地模型文件，检查 NF4 + 双重量化 + BF16 计算加载，使用 PyTorch SDPA，并通过官方聊天模板设置 `enable_thinking=False`，生成最多 64 个 Token 的短回答。这里的配置仅用于环境检查，正式训练参数另行确定。

成功时看到 `MODEL_SMOKE_OK`，并保存 `logs/model-smoke.json`，包含加载耗时、短回答和显存峰值。这不是正式训练，也不能用来判断旅游问答质量。

## 6. 新开终端与问题定位

每次新开终端先执行：

```bash
source /root/autodl-tmp/travel-finetune/activate.sh
which python
```

应指向 `/root/autodl-tmp/travel-finetune/.venv/bin/python`。通过 `bash setup_env.sh` 启动的子进程无法替父终端永久激活 venv，因此安装结束后仍需执行上面的 `source`。

如果失败，保留从首个 `ERROR` 或 `Traceback` 到末尾的完整文本，同时检查：

```bash
source /root/autodl-tmp/travel-finetune/activate.sh
python -m pip check
df -h /root/autodl-tmp
nvidia-smi
```

安装包没有启动正式训练、修改原 RAG、选择推理服务框架或调用付费模型接口。

## 7. Torch 主文件下载慢时

用户截图中 CUDA 依赖下载约 7–14MB/s，但 Torch 的 532.7MB 主文件只有约 118KB/s。速度差异集中在主文件，不能据此认定整个实例带宽只有 118KB/s。

本次已核对阿里云镜像存在 `torch-2.12.1+cu130-cp312-cp312-manylinux_2_28_x86_64.whl`，HEAD 返回 200，文件大小为 532716184 字节，并从 PyTorch 官方索引取得该构建的 SHA256。未在用户云端实测镜像吞吐，不保证固定下载速度。

如果旧版安装脚本仍停在 `Downloading torch...`，可按一次 Ctrl+C，等回到终端提示符后执行：

```bash
source /root/autodl-tmp/travel-finetune/activate.sh
python -m pip install --no-cache-dir --timeout 120 --retries 5 \
  'https://mirrors.aliyun.com/pytorch-wheels/cu130/torch-2.12.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=4bafc356fbb622e2756179406825c3a56c17b401196435a1487c5b40c657706c' \
  --index-url https://mirrors.aliyun.com/pypi/simple/
```

完成后再执行原安装脚本：

```bash
bash /root/autodl-tmp/autodl_setup/setup_env.sh
```

已满足的相同版本依赖通常会被直接复用，原脚本可继续运行剩余安装和检查。原命令使用 `--no-cache-dir`，中断时仅下载到临时目录、尚未安装的文件可能需要重新下载；已安装包不会因为 Ctrl+C 而被删除。如果已进入 `Installing collected packages`，先让本次安装完成，不为换源而中断安装阶段。

上面的换源方案在用户实例实测仍然慢，因此当前默认已改成复用基础环境。若以后明确要创建完全隔离的 venv，可先设置 `FT_REUSE_BASE_PACKAGES=0` 进入独立安装分支，使用上述直链和阿里云 PyPI 源。这个分支会检查隔离配置，不会自动把已经共享基础包的 venv 改成完全隔离；两种配置应按需要另行准备。源覆盖参数仍为 `FT_TORCH_WHEEL_URL` 和 `FT_PYPI_INDEX_URL`，不修改全局 pip 配置。

AutoDL 内置学术加速的官方页面列举 GitHub 和 Hugging Face 等域名，没有列出 PyTorch 下载域名，因此不把 `source /etc/network_turbo` 当作此问题的确定解决方案。

## 8. 云端旧安装包的续装命令：复用原镜像 Torch

此方案已经得到用户确认。若终端还在下载 Torch，先按一次 Ctrl+C，等命令提示符返回。然后执行以下整段命令；它会保留现有 venv，启用基础环境包的可见性，先检查 Torch，再继续安装训练依赖。

```bash
bash <<'BASH'
set -euo pipefail
/root/miniconda3/bin/python -m venv --system-site-packages /root/autodl-tmp/travel-finetune/.venv
source /root/autodl-tmp/travel-finetune/activate.sh
python - <<'PY'
import importlib.metadata
import sys
import torch
assert sys.prefix != sys.base_prefix, 'The training venv must be active.'
assert importlib.metadata.version('torch') == '2.12.1+cu130', 'Unexpected Torch version.'
assert torch.cuda.is_available(), 'CUDA is unavailable.'
print('Python:', sys.executable)
print('Torch:', torch.__version__)
print('Torch path:', torch.__file__)
print('CUDA:', torch.version.cuda)
PY
python -m pip install --no-cache-dir --timeout 120 --retries 5 \
  -r /root/autodl-tmp/autodl_setup/requirements.txt \
  --index-url https://mirrors.aliyun.com/pypi/simple/
python -m pip check
python /root/autodl-tmp/autodl_setup/check_env.py --report /root/autodl-tmp/travel-finetune/logs/environment.json
python -m pip freeze --all > /root/autodl-tmp/travel-finetune/logs/installed-requirements.txt
python -m pip freeze --local > /root/autodl-tmp/travel-finetune/logs/venv-local-requirements.txt
printf '\nSETUP_OK\n'
BASH
```

看到 `ENV_CHECK_OK` 和 `SETUP_OK` 后按第 4 节继续下载模型。旧版 `setup_env.sh` 有 `include-system-site-packages = false` 的断言，本次续装直接使用上面的流程。

## 9. 官方依据

- [PyTorch cu130 安装包索引](https://download.pytorch.org/whl/cu130/torch/)
- [bitsandbytes 安装与 CUDA 构建范围](https://huggingface.co/docs/bitsandbytes/en/installation)
- [PEFT 量化训练](https://huggingface.co/docs/peft/en/developer_guides/quantization)
- [魔搭 Qwen3-8B 仓库](https://modelscope.cn/models/Qwen/Qwen3-8B)
- [ModelScope snapshot_download 接口源码](https://github.com/modelscope/modelscope/blob/master/modelscope/hub/snapshot_download.py)
- [Qwen3-8B 模板与非思考模式](https://huggingface.co/Qwen/Qwen3-8B)
- [Python venv](https://docs.python.org/3/library/venv.html)
- [阿里云 PyTorch cu130 镜像目录](https://mirrors.aliyun.com/pytorch-wheels/cu130/)
- [AutoDL 学术资源加速范围](https://api.autodl.com/docs/network_turbo/)
- [pip URL 哈希校验说明](https://pip.pypa.io/en/stable/topics/secure-installs/)
