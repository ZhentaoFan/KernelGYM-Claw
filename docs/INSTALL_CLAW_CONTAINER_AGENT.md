# Claw Container Agent 安装与启动

这份文档说明如何从 fork 拉起 DR.Kernel 8B 的 Claw Container Agent 训练。设计目标是每个 rollout sample 启动一个独立 Docker container，container 内运行 `claw`，通过 OpenAI-compatible API 请求训练侧 vLLM，并通过 container 内注册的 `evaluate_kernel` tool 主动调用 KernelGYM 获取反馈。

## 1. Clone 分支

```bash
git clone https://github.com/ZhentaoFan/KernelGYM-Claw.git
cd KernelGYM-Claw
git checkout ClawContainerAgent
```

## 2. 准备 Python 环境

默认脚本使用 repo 内的 `.venv/drkernel310`，也可以通过 `ENV_DIR` 指到已有 conda/env。

```bash
conda create -p "$PWD/.venv/drkernel310" python=3.10 -y
ENV_DIR="$PWD/.venv/drkernel310" bash scripts/bootstrap_drkernel310_local.sh
```

如果已有环境:

```bash
export ENV_DIR=/path/to/drkernel310
bash scripts/bootstrap_drkernel310_local.sh
```

## 3. 下载模型

```bash
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
mkdir -p models/hkust-nlp
"$PWD/.venv/drkernel310/bin/hf" download hkust-nlp/drkernel-8b-coldstart \
  --local-dir "$PWD/models/hkust-nlp/drkernel-8b-coldstart"
```

训练数据默认使用 Hugging Face dataset repo:

```bash
hkust-nlp/drkernel-rl-data
hkust-nlp/drkernel-validation-data
```

如果要改成本地 parquet 或其他 dataset 路径，可以在 launcher 里改 `TRAIN_DATASET` 和 `VALID_DATASET`，或新增自己的 wrapper script。

## 4. 构建 Claw Agent 镜像

这个步骤会把 `claw-code` clone 到 `${CLAW_CODE_SRC:-$PWD/third_party/claw-code}`，然后构建 `claw-agent-runtime:latest`。

```bash
bash scripts/build_claw_agent_image.sh
```

可覆盖项:

```bash
export CLAW_CODE_REPO=https://github.com/ultraworkers/claw-code
export CLAW_CODE_SRC="$PWD/third_party/claw-code"
export CLAW_AGENT_IMAGE=claw-agent-runtime:latest
```

## 5. 启动 KernelGYM Server 和 Worker

推荐 KernelGYM worker 使用独立 GPU，例如 3,4,5；训练和 rollout 使用 6,7。

```bash
export ENV_DIR="$PWD/.venv/drkernel310"
export CUDA_VISIBLE_DEVICES=3,4,5
export GPU_DEVICES='[0,1,2]'
mkdir -p logs/kernelgym
LOG=logs/kernelgym/kernelgym_server_$(date -u +%Y%m%dT%H%M%SZ).log
nohup setsid bash scripts/start_kernelgym_local.sh > "$LOG" 2>&1 < /dev/null &
echo "$LOG"
```

健康检查:

```bash
curl -sS http://127.0.0.1:10907/health
```

## 6. 启动 Claw Container Agent 训练

下面命令默认在 GPU 6,7 上跑训练和 rollout。

```bash
export ENV_DIR="$PWD/.venv/drkernel310"
export CUDA_VISIBLE_DEVICES=6,7
export ARNOLD_WORKER_GPU=2
export N_GPUS_PER_NODE=2
export SP_SIZE=2
export MODEL_PATH="$PWD/models/hkust-nlp/drkernel-8b-coldstart"
export HDFS_CHECKPOINT_PATH="$PWD/checkpoints/drkernel"
export CLAW_WORKSPACE_ROOT="$PWD/tmp/claw_drkernel_rollouts"
export KERNELGYM_SERVER_URL=http://127.0.0.1:10907
export CLAW_AGENT_IMAGE=claw-agent-runtime:latest
export CLAW_KERNELGYM_MAX_EVALS="${CLAW_KERNELGYM_MAX_EVALS:-3}"
export CLAW_KERNELGYM_NUM_PERF_TRIALS="${CLAW_KERNELGYM_NUM_PERF_TRIALS:-20}"
export N_VAL="${N_VAL:-2}"
export SAVE_FREQ="${SAVE_FREQ:-5}"

mkdir -p logs/drkernel_rl
LOG=logs/drkernel_rl/drkernel_claw_full_$(date -u +%Y%m%dT%H%M%SZ).log
nohup bash drkernel/kernel/scripts/rl/8b_claw_container_agent_local.sh > "$LOG" 2>&1 &
echo "$LOG"
```

## 7. 关键环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ENV_DIR` | `$PWD/.venv/drkernel310` | Python 环境路径。 |
| `MODEL_PATH` | `$PWD/models/hkust-nlp/drkernel-8b-coldstart` | 冷启动模型路径。 |
| `HDFS_CHECKPOINT_PATH` | `$PWD/checkpoints/drkernel` | checkpoint 输出目录。 |
| `KERNELGYM_SERVER_URL` | `http://127.0.0.1:10907` | reward evaluate server。 |
| `CLAW_AGENT_IMAGE` | `claw-agent-runtime:latest` | 每个 rollout container 使用的镜像。 |
| `CLAW_WORKSPACE_ROOT` | `$PWD/tmp/claw_drkernel_rollouts` | 每个 sample 的 host workspace 根目录。 |
| `CLAW_CONTAINER_NETWORK` | `host` | container 网络模式。 |
| `CLAW_MAX_COMPLETION_TOKENS` | `MAX_RESPONSE_LENGTH` 或 `8192` | proxy enforced completion token budget。 |
| `CLAW_CONCURRENCY` | `8` | 同时运行的 Claw container 数量。 |
| `CLAW_KERNELGYM_MAX_EVALS` | `3` | 每个 rollout container 内 Claw 最多主动调用 `evaluate_kernel` 的次数。 |
| `CLAW_KERNELGYM_NUM_PERF_TRIALS` | `20` | Claw tool-in-loop 反馈用的 performance trials；最终 trainer reward 仍使用 `NUM_PERF_TRIALS`。 |
| `CLAW_KERNELGYM_NUM_CORRECT_TRIALS` | `5` | Claw tool-in-loop 反馈用的 correctness trials。 |
| `N_VAL` | `2` | validation 每题采样数。 |
| `SAVE_FREQ` | `5` | checkpoint 保存频率。 |
| `MAX_ACTOR_CKPT_TO_KEEP` | `1` | 只保留最近 actor checkpoint。 |
| `MAX_CRITIC_CKPT_TO_KEEP` | `1` | 只保留最近 critic checkpoint。 |

## 8. 常用排查

看训练日志:

```bash
tail -f "$LOG"
```

看 Claw container:

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' | grep claw-drkernel
```

看 GPU:

```bash
nvidia-smi
```

看 KernelGYM 健康状态:

```bash
curl -sS http://127.0.0.1:10907/health
```

停止 Ray:

```bash
"$ENV_DIR/bin/ray" stop --force
```

停止 KernelGYM:

```bash
bash stop_all.sh
```

## 9. 设计边界

Claw 的多轮工具调用发生在 container 内部。每个 workspace 会生成一个本地 plugin `claw_plugins/kernelgym-evaluator`，并通过 `.claw/settings.json` 启用 `evaluate_kernel`。该 tool 从隐藏的 `kernelgym_context.json` 读取 reference/entry_point/uuid，把 Claw 提供的候选 `kernel_code` 提交到 `KERNELGYM_SERVER_URL/evaluate`，再把 compile/correctness/speedup 作为 tool_result 注入 Claw conversation。

Trainer 侧保持 `ENABLE_MULTI_TURN=False`，把 Claw session 渲染成一个 rollout sample 的 response、mask 和 metadata 再进入 PPO。最终训练 reward 仍由 DR.Kernel reward manager 对最终 response 再跑一次 KernelGYM evaluate 计算。

当前实现读取每个 workspace 下最新 `.claw/sessions/*.jsonl`。如果后续 Claw context compression 产生多个 session 文件，并且需要严格训练完整 pre-compression 轨迹，需要把 session loader 改成按时间合并多个 JSONL 或要求 Claw 导出完整 transcript artifact。
