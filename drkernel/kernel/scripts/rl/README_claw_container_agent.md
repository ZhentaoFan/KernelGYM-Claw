# DR.Kernel Claw Container Agent 集成说明

最后更新: 2026-04-15 UTC

这份文档记录 DR.Kernel 里接入 "container as an agent" 的改动。当前目标是让每个 rollout sample 启动一个独立 docker container，container 里运行专业 agent `claw`，`claw` 通过 OpenAI-compatible API 调训练侧 vLLM 生成 token。container entrypoint 会强制执行 ReAct 循环：`claw 生成候选代码 -> regex/solution.py 抽取 -> KernelGYM evaluate -> 把反馈作为下一轮 prompt 写回 claw`。结束后把 transcript 转回 VERL/DR.Kernel trainer 能训练的 `response_ids`、`response_mask`、reward 和 metrics。

## 当前运行方式

这个分支提供的是可复用 launcher，不绑定某一台机器上的历史 run。默认路径都可以通过环境变量覆盖，推荐先看同目录下的 [INSTALL_CLAW_CONTAINER_AGENT.md](INSTALL_CLAW_CONTAINER_AGENT.md)。

当前使用资源:

| 资源 | 用途 |
| --- | --- |
| GPU 2,3 | KernelGYM kernel worker/server |
| GPU 4,5,6,7 | DR.Kernel rollout + training |
| docker network | `host` |
| Claw workspace | `${CLAW_WORKSPACE_ROOT:-$ROOT_DIR/tmp/claw_drkernel_rollouts}` |

下次正常启动可以用:

```bash
cd /path/to/KernelGYM-Claw
mkdir -p logs/drkernel_rl
LOG=logs/drkernel_rl/drkernel_claw_full_$(date -u +%Y%m%dT%H%M%SZ).log
nohup bash drkernel/kernel/scripts/rl/8b_claw_container_agent_local.sh > "$LOG" 2>&1 &
echo "$LOG"
```

## 一句话设计

DR.Kernel trainer 不直接生成单段文本，而是把每个 rollout sample 交给 `ClawContainerAgentLoop`，由它启动一个 container。container 内的 `claw_react_loop.py` 多次调用 `claw prompt`，每轮自动抽取 `solution.py` 或最后一个 `class ModelNew` 代码块，调用 KernelGYM 得到 compile/correctness/speedup feedback，再把 feedback 塞进下一轮 prompt。最后把 `claw_react_transcript.jsonl` 重新渲染并 tokenize 成训练样本。

## 端到端流程

1. Launcher 设置 `ROLLOUT_MODE=async_agent`，并通过 Hydra 指定 `actor_rollout_ref.rollout.agent.default_agent_name=claw_container`。
2. `kernel_trainer.py` 在初始化 worker 时创建 `AgentLoopManager`。
3. `AgentLoopWorker` 收到一个 rollout sample 后，把它路由到 `ClawContainerAgentLoop`。
4. `ClawContainerAgentLoop` 为这个 sample 建 workspace，写 prompt 和 container entrypoint。
5. 宿主机执行 `docker run --rm --network host -v <workspace>:/workspace ... claw-agent-runtime:latest`。
6. container 内先启动 OpenAI-compatible proxy，再执行 `python3 /workspace/claw_react_loop.py`。
7. `claw_react_loop.py` 每轮调用 `claw --output-format json ... prompt <turn_prompt>`；Claw 需要模型输出时经由 proxy 请求训练侧 vLLM。
8. 每轮 Claw 结束后，entrypoint 优先读 `/workspace/solution.py`，否则从 assistant 文本最后一个 Python code block 抽取 `class ModelNew`。
9. entrypoint 调 `/workspace/kernelgym_evaluate.py` POST 到 `KERNELGYM_SERVER_URL/evaluate`，得到 compile/correctness/speedup/error feedback。
10. feedback 作为下一轮 user prompt 写回 Claw；同时写入 `claw_react_transcript.jsonl`，其中 assistant token 可训练、feedback token 不训练。
11. 最后一轮 eval 的 reward/metrics 写入 `react_summary.json`，host 侧直接回填 `AgentLoopOutput.reward_score`，避免混合 transcript 被 reward manager 错误截断。
12. Trainer 用这些 rollout batch 做 PPO/update/save/eval。

## 关键文件清单

| 文件 | 改动作用 |
| --- | --- |
| [../../../verl_patch/experimental/agent_loop/claw_container_agent.py](../../../verl_patch/experimental/agent_loop/claw_container_agent.py) | 新增 `ClawContainerAgentLoop`，负责每个 sample 一个 docker container、Claw 启动、proxy 转发、session 读取、轨迹 tokenize、返回 `AgentLoopOutput`。 |
| [../../../verl_patch/experimental/agent_loop/claw_react_loop.py](../../../verl_patch/experimental/agent_loop/claw_react_loop.py) | container 内强制 ReAct loop：调用 Claw、抽取候选代码、调用 KernelGYM、生成下一轮 feedback prompt、写 transcript/summary。 |
| [../../../verl_patch/experimental/agent_loop/kernelgym_evaluate.py](../../../verl_patch/experimental/agent_loop/kernelgym_evaluate.py) | container 内 KernelGYM evaluate 客户端；只依赖 Python 标准库，读取 host 写入的 hidden reference context。 |
| [../../../verl_patch/experimental/agent_loop/__init__.py](../../../verl_patch/experimental/agent_loop/__init__.py) | import `ClawContainerAgentLoop`，让 `@register("claw_container")` 生效。 |
| [../../../verl_patch/experimental/agent_loop/agent_loop.py](../../../verl_patch/experimental/agent_loop/agent_loop.py) | 增加 `default_agent_name` fallback，不需要改 parquet schema 也能强制全量样本走 `claw_container`；当 agent loop 已经给出 `reward_score` 时直接生成 `token_level_scores` 并透传 `reward_extra_info`。 |
| [../../kernel_trainer.py](../../kernel_trainer.py) | 在 `async_agent` 路径初始化 `AgentLoopManager`；补 `ensure_sample_loss_mask`；兼容缺失 rollout logprobs 的分支；在 reward loss mask 前保证 sample mask 存在。 |
| [../../workers/reward_manager/kernel_async.py](../../workers/reward_manager/kernel_async.py) | 让 reward manager 支持 `DataProto` 输入，逐样本 decode response 并复用原来的 KernelGYM evaluate 逻辑，同时收集 numeric extra_info。 |
| [8b_claw_container_agent_local.sh](8b_claw_container_agent_local.sh) | Claw container-agent 训练 launcher，设置 async agent、GPU 4-7、rollout/training 超参、checkpoint 保留策略和 Claw 相关 env。 |
| [claw_agent_loop_configs.yaml](claw_agent_loop_configs.yaml) | Agent loop registry，把 `claw_container` 映射到 `verl_patch.experimental.agent_loop.claw_container_agent.ClawContainerAgentLoop`。 |
| [claw_empty_tools.yaml](claw_empty_tools.yaml) | 空 tool config，占位用；真正的工具调用由 container 内 Claw 自己管理。 |
| [train_rl_common.sh](train_rl_common.sh) | 支持 `EXTRA_HYDRA_OVERRIDES`，让 Claw launcher 可以追加 agent-loop 相关 Hydra override。 |
| [../../../verl_patch/workers/config/rollout.py](../../../verl_patch/workers/config/rollout.py) | rollout config schema 增加 `default_agent_name` 字段。 |
| [../../../verl_patch/trainer/code/config/rollout/rollout.yaml](../../../verl_patch/trainer/code/config/rollout/rollout.yaml) | rollout yaml 默认值增加 `default_agent_name: single_turn_agent`，保持非 Claw 路径兼容。 |
| [../../../verl_patch/workers/code/rollout/vllm_rollout/vllm_async_engine.py](../../../verl_patch/workers/code/rollout/vllm_rollout/vllm_async_engine.py) | 兼容 `MathRewardManager` 可选导入，避免 DR.Kernel 环境里不需要的依赖导致 async vLLM 路径启动失败。 |
| [../../../verl_patch/workers/code/rollout/vllm_rollout/vllm_async_server.py](../../../verl_patch/workers/code/rollout/vllm_rollout/vllm_async_server.py) | 修复 ErrorResponse 状态码读取，使用 `generator.error.code`。 |

## ClawContainerAgentLoop 内部结构

核心文件: [../../../verl_patch/experimental/agent_loop/claw_container_agent.py](../../../verl_patch/experimental/agent_loop/claw_container_agent.py)

| 代码区域 | 作用 |
| --- | --- |
| env knobs | 读取 `CLAW_AGENT_IMAGE`、`CLAW_WORKSPACE_ROOT`、`CLAW_MAX_COMPLETION_TOKENS`、`CLAW_CONCURRENCY`、`CLAW_AGENT_TIMEOUT_SEC` 等环境变量。 |
| `OPENAI_COMPAT_PROXY_SCRIPT` | 写进 container 的本地 proxy。它接收 Claw 的 OpenAI-compatible 请求，限制剩余 completion token，再转发给训练侧 vLLM。 |
| `_container_entrypoint_script` | container 内入口脚本。启动 proxy，设置 `OPENAI_BASE_URL`，执行 `python3 /workspace/claw_react_loop.py`。 |
| `_load_session_messages` | 优先读取 `claw_react_transcript.jsonl`；如果没有，再回退到最新 `.claw/sessions/*.jsonl`。 |
| `_render_claw_turns` | 把 Claw session message 转成训练侧可 tokenize 的 role/content turn，包含 tool_use/tool_result 的文本化。 |
| `_run_claw_container_blocking` | host 侧实际执行 docker container，处理 timeout、kill、stdout/stderr 和 artifact 路径。 |
| `ClawContainerAgentLoop.run` | 单个 rollout sample 的总控：准备 prompt、启动 container、读取 artifacts、构造 response/tokens/masks/metadata。 |

## 关键超参

当前 launcher 默认值在 [8b_claw_container_agent_local.sh](8b_claw_container_agent_local.sh) 里。

| 参数 | 当前默认 | 说明 |
| --- | --- | --- |
| `ROLLOUT_MODE` | `async_agent` | 使用 AgentLoopManager，而不是普通 vLLM rollout。 |
| `TRAIN_BATCH_SIZE` | `16` | DR.Kernel 原始 8B local 风格 batch。 |
| `PPO_MINI_BATCH_SIZE` | `16` | 和 train batch 对齐。 |
| `ROLLOUT_N` | `16` | 每个 prompt 的训练 rollout 数。 |
| `N_VAL` | `2` | eval rollout 数。 |
| `MAX_RESPONSE_LENGTH` | `8192` | trainer 侧最大 response token。 |
| `CLAW_MAX_COMPLETION_TOKENS` | `8192` | container proxy 对单个 sample 的 completion token budget。 |
| `CLAW_CONCURRENCY` | `8` | 同时运行的 Claw container 数。 |
| `CLAW_REACT_MAX_TURNS` | `3` | container 内强制 ReAct turn 数；默认跟 `MAX_TURN` 对齐。 |
| `CLAW_REACT_STOP_ON_OK` | `false` | 是否在某轮 KernelGYM 已正确时提前停止；默认继续尝试优化。 |
| `SP_SIZE` | `4` | actor sequence parallel size。 |
| `ROLLOUT_GPU_MEMORY_UTIL` | `0.75` | vLLM GPU memory utilization。 |
| `VAL_BEFORE_TRAIN` | `True` | 训练前先跑 validation，所以启动早期会看到大量 eval containers。 |
| `ENABLE_MULTI_TURN` | `False` | 有意保持 False；多轮由 container 内 Claw 自己完成，trainer 不再额外扩展 multi-turn batch。 |
| `MAX_ACTOR_CKPT_TO_KEEP` | `1` | 只保留最近 actor ckpt，避免磁盘被 8B checkpoint 撑满。 |
| `MAX_CRITIC_CKPT_TO_KEEP` | `1` | 只保留最近 critic ckpt。 |

## Artifacts 从哪里来

每个 sample 都有一个 host workspace，默认在:

```bash
${CLAW_WORKSPACE_ROOT:-$ROOT_DIR/tmp/claw_drkernel_rollouts}/claw-drkernel-*
```

常见文件:

| artifact | 来源 | 训练侧用途 |
| --- | --- | --- |
| `prompt.txt` | host 写入 | 给 Claw 的任务输入。 |
| `entrypoint.sh` | host 写入 | container 内执行脚本。 |
| `openai_proxy.py` | host 写入 | container 内 proxy 代码。 |
| `claw_react_loop.py` | host 写入 | container 内强制多轮 ReAct 控制器。 |
| `kernelgym_evaluate.py` | host 写入 | container 内 KernelGYM evaluate 客户端。 |
| `kernelgym_context.json` | host 写入 | hidden reference、entry point、uuid 等 evaluate 所需上下文。 |
| `claw_react_transcript.jsonl` | entrypoint 写入 | 训练轨迹主来源；记录 user prompt、assistant output、KernelGYM feedback。 |
| `react_summary.json` | entrypoint 写入 | 最后一轮 eval、reward_score、reward_extra_info。 |
| `claw_result.json` | Claw CLI 输出 | 优先提取最终 message。 |
| `claw_stderr.log` | Claw CLI stderr | debug Claw 失败原因。 |
| `.claw/sessions/*.jsonl` | Claw runtime 写入 | fallback/debug 来源；正式训练优先用 `claw_react_transcript.jsonl`。 |
| `openai_proxy.log` | proxy 写入 | debug vLLM 请求、budget 和错误。 |

训练侧不是凭空构造 response，而是等 container 退出后读这些 artifacts。优先级是:

1. 优先用 `claw_react_transcript.jsonl` 作为训练 trajectory。
2. 用 `react_summary.json` 里的最后一轮 evaluation 作为 reward/metrics。
3. 用 `claw_result.json` 里的最终 message 当最终文本 fallback。
4. 如果没有最终 message，则回退到 transcript/session 里最后一条 assistant message。
5. 如果 session 也不可用，则用 failure reason 生成一个失败 response。

## Tool call 和 context compression

Claw 的 tool call 发生在 container 内部，DR.Kernel trainer 不直接执行这些工具。训练侧看到的是 Claw session JSONL 里记录下来的 tool_use/tool_result 轨迹。

如果 Claw 发生 context compression，风险点在 `.claw/sessions/*.jsonl` 的记录语义，而不是 trainer 本身。当前实现读取最新 session，并把能看到的 session messages 渲染成一条训练 response。也就是说:

| 情况 | 当前行为 |
| --- | --- |
| 普通 tool call | tool_use/tool_result 被序列化进 response 轨迹。 |
| 多个 session jsonl | 当前取最新 session 文件；如果旧 session 包含被压缩前的重要轨迹，可能不会完整训练到。 |
| compression summary 写入最新 session | 训练侧会训练 summary 之后可见的轨迹。 |
| compression 把旧 tool 细节完全折叠掉 | trainer 只能训练折叠后的文本，不能恢复被折叠的 token。 |

如果后面要严格支持 "一次 rollout 多个 trajectory"，建议把 session loader 改成按时间合并多个 JSONL，或让 Claw 额外导出完整 transcript artifact，再由 `_render_claw_turns` 统一 tokenize。

## 为什么保留 `ENABLE_MULTI_TURN=False`

这个点容易误解。Claw 在 container 内部可以多轮思考、调用工具、多次请求 vLLM；这属于 agent 内部轨迹。

DR.Kernel/VERL trainer 侧的 `ENABLE_MULTI_TURN=True` 是另一层机制，会把 trainer batch 扩成框架理解的 multi-turn rollout。当前 Claw 集成已经把整段 Claw session 回填成一个 sample 的 response/mask，因此 trainer 侧继续开 multi-turn 会改变 batch/turn 结构，容易和 Claw 自己的轨迹语义打架。

所以当前选择是:

```bash
ENABLE_MULTI_TURN=False
```

含义不是 "Claw 不多轮"，而是 "多轮由 Claw/container 内部负责，trainer 只接收最终渲染后的单个 rollout sample"。

## 排查命令

看当前 formal run 日志:

```bash
LOG=${LOG:-logs/drkernel_rl/drkernel_claw_full_latest.log}
tail -f "$LOG"
```

确认训练主进程:

```bash
ps -ef | grep -E 'main_kernel|8b_claw_container_agent_local|ray::TaskRunner' | grep -v grep
```

看 Claw container:

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' | grep claw-drkernel
```

看 GPU:

```bash
nvidia-smi
```

检查最近错误:

```bash
LOG=${LOG:-logs/drkernel_rl/drkernel_claw_full_latest.log}
grep -Ei 'Traceback|Error executing|RuntimeError|CUDA Error|out of memory|AssertionError|KeyError|ValueError' "$LOG" | tail -80
```

检查 KernelGYM server:

```bash
curl -s http://127.0.0.1:10907/health
```

注意: `/health` 里的 `total_processed` 不一定能准确反映训练是否卡住，优先看 trainer log 里是否还有 `ClawContainer ... done`、reward/eval/train step，以及 GPU 4-7 是否持续有 vLLM/worker 负载。

## 已知风险

| 风险 | 现状 |
| --- | --- |
| 磁盘 | 8B checkpoint 很大，已经设置 actor/critic ckpt keep=1。 |
| 训练前 eval 很久 | `VAL_BEFORE_TRAIN=True` 且 `N_VAL=2`，启动后先看到 eval container 是正常的。 |
| response 接近 8192 | proxy 会限制 completion budget，但 Claw session 渲染后再 tokenize 可能接近或略受 chat template 影响，需要继续观察 `response_ids` 长度。 |
| tool 轨迹完整性 | 普通 tool_use/tool_result 会进 session；如果 context compression 拆成多个 session，当前只读最新 session。 |
| reward manager 输入 | 已兼容 `DataProto`，但如果 response 不是 KernelGYM 期望格式，reward 仍可能为 0。 |

## 训练结构框架

```
8b_claw_container_agent_local.sh
  |
  |  ROLLOUT_MODE=async_agent
  |  EXTRA_HYDRA_OVERRIDES: default_agent_name=claw_container, tool_config_path, torch_compile=false ...
  |
  v
kernel/main_kernel.py  -->  kernel_trainer.py  RayKernelTrainer
  |
  |  init_workers():
  |    mode == "async_agent"  -->  AgentLoopManager(config, worker_group)
  |                                   |
  |                                   +-- AsyncvLLMServer x4 (FastAPI /v1/chat/completions, 每 GPU 一个)
  |                                   |     bound to http://<host>:<port>, free_cache_engine=True
  |                                   |
  |                                   +-- AgentLoopWorker x N (Ray actor, num_workers=CLAW_CONCURRENCY)
  |                                         |
  |                                         +-- Hydra instantiate("claw_container")
  |                                               --> ClawContainerAgentLoop
  |
  v
fit() training loop:
  |
  |  for each step:
  |
  |  [1] ROLLOUT  ---------------------------------------------------------------
  |  |
  |  |  AgentLoopManager.generate_sequences(gen_batch)
  |  |    |
  |  |    +-- wake_up()  -->  AsyncvLLMServer.wake_up()  (恢复 vLLM KV cache)
  |  |    |
  |  |    +-- prompts.repeat(rollout_n)  -->  chunk across AgentLoopWorkers
  |  |    |
  |  |    +-- ray.get([ worker.generate_sequences.remote(chunk) ... ])
  |  |    |     |
  |  |    |     v  (inside each AgentLoopWorker, async)
  |  |    |     for sample in chunk:
  |  |    |       ClawContainerAgentLoop.run(sampling_params, **kwargs)
  |  |    |         |
  |  |    |         |  [a] tokenizer.apply_chat_template(messages) --> prompt_ids
  |  |    |         |
  |  |    |         |  [b] _resolve_upstream()
  |  |    |         |       server_handles[0].get_server_address.remote()
  |  |    |         |       --> "http://<host>:<port>"  (AsyncvLLMServer FastAPI)
  |  |    |         |
  |  |    |         |  [c] _prepare_workspace()
  |  |    |         |       写入 prompt.txt, openai_compat_proxy.py, TASK.md
  |  |    |         |
  |  |    |         |  [d] docker run --rm --network host  (受 CLAW_CONCURRENCY semaphore 控制)
  |  |    |         |       -v <workspace>:/workspace
  |  |    |         |       claw-agent-runtime:latest
  |  |    |         |       bash -lc "...entrypoint..."
  |  |    |         |         |
  |  |    |         |         |  container 内部:
  |  |    |         |         |    openai_compat_proxy.py  (budget cap, 转发)
  |  |    |         |         |       listen 127.0.0.1:<random_port>
  |  |    |         |         |       upstream = http://<host>:<vllm_port>
  |  |    |         |         |       每个 /chat/completions 请求:
  |  |    |         |         |         patch max_tokens <= remaining_budget
  |  |    |         |         |         forward to vLLM AsyncvLLMServer
  |  |    |         |         |         track consumed tokens
  |  |    |         |         |         return 429 when budget exhausted
  |  |    |         |         |
  |  |    |         |         |    claw --model hkust-nlp/drkernel-8b-coldstart
  |  |    |         |         |         --output-format json
  |  |    |         |         |         --permission-mode danger-full-access
  |  |    |         |         |         prompt "$PROMPT"
  |  |    |         |         |         |
  |  |    |         |         |         |  claw 内部多轮:
  |  |    |         |         |         |    思考 -> tool_call -> tool_result -> 思考 -> ...
  |  |    |         |         |         |    每次 LLM 请求 -> proxy -> vLLM -> 生成 token
  |  |    |         |         |         |
  |  |    |         |         |         v
  |  |    |         |         |    写出: claw_result.json, .claw/sessions/*.jsonl
  |  |    |         |         |
  |  |    |         |         v  container exit
  |  |    |         |
  |  |    |         |  [e] _load_session_messages(workspace)
  |  |    |         |       读取 .claw/sessions/*.jsonl 最新 session
  |  |    |         |
  |  |    |         |  [f] _render_claw_turns(session_messages, fallback)
  |  |    |         |       session -> [(role, content), ...] 列表
  |  |    |         |       assistant turn -> trainable (mask=1)
  |  |    |         |       tool turn       -> non-trainable (mask=0)
  |  |    |         |
  |  |    |         |  [g] _tokenize_turn_delta() per turn
  |  |    |         |       增量 tokenize: apply_chat_template(prev+new) - apply_chat_template(prev)
  |  |    |         |       --> response_ids, response_mask
  |  |    |         |
  |  |    |         v
  |  |    |         return AgentLoopOutput(prompt_ids, response_ids, response_mask, ...)
  |  |    |
  |  |    +-- _postprocess(outputs)
  |  |    |     pad prompt/response to max_length
  |  |    |     build attention_mask, position_ids
  |  |    |     --> DataProto(batch=TensorDict, non_tensor_batch)
  |  |    |
  |  |    +-- 补齐 trainer 所需字段:
  |  |    |     forward uid/data_source/reward_model from input batch
  |  |    |     token_level_scores = zeros  (reward 由 trainer 后续计算或 bypass)
  |  |    |     reward_extra_info = [{}]
  |  |    |     loss_mask = response_mask.clone()
  |  |    |
  |  |    +-- sleep()  -->  AsyncvLLMServer.sleep()  (释放 vLLM KV cache, 给训练腾 GPU 内存)
  |  |    |
  |  |    v
  |  |  gen_batch_output: DataProto
  |  |
  |  +-------------------------------------------------------------------
  |
  |  [2] REWARD  ----------------------------------------------------------------
  |  |
  |  |  if "token_level_scores" in batch:
  |  |    跳过 reward_fn 调用  (当前 agent loop 返回 zeros)
  |  |  else:
  |  |    reward_fn(batch)  -->  AsyncKernelRewardManager
  |  |      decode response_ids -> text
  |  |      extract kernel code
  |  |      POST /evaluate to KernelGYM (http://127.0.0.1:10907)
  |  |      --> reward_tensor, extra_info (speedup, correctness, compilation...)
  |  |
  |  +-------------------------------------------------------------------
  |
  |  [3] LOG PROB RECOMPUTATION  ------------------------------------------------
  |  |
  |  |  old_log_prob = actor_rollout_wg.compute_log_prob(batch)
  |  |    FSDP forward pass (teacher-forcing) on response tokens
  |  |    param_offload=True: 参数在 CPU, 逐层流式到 GPU
  |  |    SP_SIZE=4: Ulysses sequence parallel across 4 GPUs
  |  |
  |  |  ref_log_prob = ref_policy_wg.compute_ref_log_prob(batch)
  |  |    同上, 但用 ref model weights (frozen)
  |  |
  |  +-------------------------------------------------------------------
  |
  |  [4] ADVANTAGE COMPUTATION  -------------------------------------------------
  |  |
  |  |  compute_rollout_correction_and_add_to_batch()
  |  |    如果有 rollout_log_probs: 计算 IS weights, rejection sampling
  |  |    如果没有 (当前 claw 路径): 直接返回 batch, {}, {}
  |  |
  |  |  compute_multi_turn_advantage()  或  compute_advantage()
  |  |    token_level_rewards = token_level_scores (from reward)
  |  |    advantages = TRLOO / GRPO / GAE
  |  |    需要: response_mask, loss_mask, old_log_probs, ref_log_probs
  |  |
  |  +-------------------------------------------------------------------
  |
  |  [5] TRAINING STEP  ---------------------------------------------------------
  |  |
  |  |  actor_rollout_wg.update_actor(batch)
  |  |    FSDP forward + backward + optimizer step
  |  |    policy_loss = clipped PPO objective
  |  |    grad_clip, entropy_coeff, loss_scale_factor
  |  |    optimizer_offload=True: Adam states on CPU
  |  |
  |  +-------------------------------------------------------------------
  |
  |  [6] LOGGING & CHECKPOINT  --------------------------------------------------
  |  |
  |  |  wandb.log(metrics)  (offline mode)
  |  |  if step % save_freq == 0: save checkpoint
  |  |  if step % test_freq == 0: run validation (same claw container flow)
  |  |
  |  +-------------------------------------------------------------------
  |
  v  next step


GPU 内存时序:

  rollout phase:     vLLM loaded (~60GB/GPU), FSDP model (~4GB/GPU)
                     claw containers 通过 HTTP 请求 vLLM 生成
  sleep():           vLLM KV cache freed
  log_prob phase:    FSDP model on GPU (~4GB/GPU), optimizer on CPU
                     forward pass: ~4GB activations/GPU
  training phase:    FSDP model + gradients (~8GB/GPU), optimizer streams from CPU
  wake_up():         vLLM KV cache re-allocated


数据流:

  parquet row (raw_prompt, ground_truth, entry_point, uuid)
    |
    v
  apply_chat_template --> prompt_ids (int list, ~1000 tokens)
    |
    v
  docker container: claw generates Triton kernel code via vLLM
    |
    v
  .claw/sessions/*.jsonl (assistant text + tool_use + tool_result)
    |
    v
  _render_claw_turns --> [(role, content), ...]
    |
    v
  _tokenize_turn_delta --> response_ids (int list, <=MAX_RESPONSE_LENGTH)
                           response_mask (1=assistant, 0=tool)
                           loss_mask = response_mask
    |
    v
  _postprocess --> DataProto(prompts, responses, response_mask, loss_mask,
                             attention_mask, position_ids, input_ids)
    |
    v
  reward_fn --> token_level_scores (reward at last token)
    |
    v
  compute_log_prob --> old_log_probs
  compute_ref_log_prob --> ref_log_probs
    |
    v
  compute_advantage --> advantages, returns
    |
    v
  update_actor --> policy_loss, grad_norm
    |
    v
  next rollout step
```

## ClawContainerAgentLoop 内部设计与结构

核心文件: [claw_container_agent.py](../../../verl_patch/experimental/agent_loop/claw_container_agent.py)

整个文件按功能分为 5 个区域，自上而下：

### 1. In-container OpenAI Proxy

> 源码: [`claw_container_agent.py` L67-241](../../../verl_patch/experimental/agent_loop/claw_container_agent.py#L67-L241)

```
┌─────────────────────────────────────────────────────────────┐
│  OPENAI_COMPAT_PROXY_SCRIPT  (纯 Python，零外部依赖)        │
│                                                             │
│  运行位置: container 内部 127.0.0.1:<random_port>           │
│  上游目标: 宿主机 vLLM AsyncvLLMServer (--network host)     │
│                                                             │
│  核心职责:                                                  │
│    1. 接收 claw CLI 的 /chat/completions 请求               │
│    2. patch_chat_completions_request():  (L100)             │
│       - 读取 remaining_budget  (L86)                        │
│       - 把 max_tokens 压到 min(requested, remaining)        │
│       - budget <= 0 时直接返回 429 budget_exhausted          │
│    3. ProxyHandler._forward()  (L187) 转发给 vLLM           │
│    4. extract_completion_tokens()  (L132) 扣减 budget       │
│    5. /health 端点供 entrypoint 做就绪检查  (L155-163)       │
│                                                             │
│  线程安全: _budget_lock (L83) 保护 _remaining_budget        │
│  日志输出: -> /workspace/openai_proxy.log                   │
└─────────────────────────────────────────────────────────────┘
```

为什么不直接让 claw 打 vLLM？因为需要一个 budget enforcement 层：
- claw 自身的 `max_tokens` 请求值可能很大（如 64000）
- proxy 把它压到剩余 budget（如 8192），防止单个 sample 吃掉过多 token
- 当 budget 耗尽时返回 429，claw 会优雅退出

### 2. Container Entrypoint

> 源码: [`_container_entrypoint_script()` L244-268](../../../verl_patch/experimental/agent_loop/claw_container_agent.py#L244-L268)

```bash
# _container_entrypoint_script() 生成的 shell 脚本:

set -euo pipefail
mkdir -p /workspace/home

# 1) 后台启动 proxy
python3 /workspace/openai_compat_proxy.py > /workspace/openai_proxy.log 2>&1 &
PROXY_PID=$!

# 2) 等待 proxy 就绪（最多 8 秒）
for _ in $(seq 1 80); do
  curl -sf http://127.0.0.1:${OPENAI_PROXY_PORT}/health && break
  sleep 0.1
done

# 3) 设置 claw 使用 proxy 作为 LLM backend
export OPENAI_BASE_URL=http://127.0.0.1:${OPENAI_PROXY_PORT}/v1

# 4) 执行 claw
PROMPT=$(cat /workspace/prompt.txt)
claw --output-format json \
     --permission-mode danger-full-access \
     --dangerously-skip-permissions \
     --model "$CLAW_MODEL" \
     prompt "$PROMPT" \
     > /workspace/claw_result.json \
     2> /workspace/claw_stderr.log

# 5) 清理
kill $PROXY_PID
exit $CLAW_EXIT
```

`--permission-mode danger-full-access` 让 claw 可以在 container 内自由执行 bash/read/write 等工具。
`--output-format json` 使 claw_result.json 包含结构化的 message、usage、tool_uses 等字段。

### 3. Artifact 解析与 Session 转录

> 源码: [`_ClawArtifacts` L277](../../../verl_patch/experimental/agent_loop/claw_container_agent.py#L277) / [`_load_session_messages` L296](../../../verl_patch/experimental/agent_loop/claw_container_agent.py#L296) / [`_render_claw_turns` L336](../../../verl_patch/experimental/agent_loop/claw_container_agent.py#L336)

```
_ClawArtifacts (dataclass, L277)
  ├── exit_code: int
  ├── workspace_dir: Path
  ├── result_json: dict | None     ← claw_result.json 解析
  ├── session_messages: list[dict] ← .claw/sessions/*.jsonl 解析
  ├── stderr_text: str
  └── failure_reason: str | None

_latest_session_file(workspace)  (L286)
  └── 遍历 .claw/sessions/**/*.jsonl，按 mtime 取最新

_load_session_messages(workspace)  (L296)
  └── 逐行读 JSONL，过滤 type=="message"，提取 message dict
      每条 message 结构: {role, blocks: [{type, text?, name?, input?, output?}]}

_render_claw_turns(session_messages, fallback_text) -> [(role, content), ...]  (L336)
  ├── system message   → 跳过  (L351)
  ├── 第一个 user       → 跳过（已在 prompt_ids 里）  (L355-357)
  ├── 后续 user         → ("user", text)          [罕见]
  ├── assistant text    → ("assistant", text)      [trainable, mask=1]  (L361-377)
  ├── assistant tool_use→ ("assistant", "<tool_call>{json}</tool_call>")  [trainable, mask=1]  (L367-373)
  ├── tool tool_result  → ("tool", "<tool_result name=... error=...>output</tool_result>")  [non-trainable, mask=0]  (L379-390)
  └── 如果 fallback_text 不在已有 assistant 文本中 → 追加一个 ("assistant", fallback)  (L394-396)
```

trainable/non-trainable 的分界：
- assistant 的思考和 tool_call 都是模型生成的 → mask=1，参与 policy gradient
- tool_result 是环境返回的 → mask=0，不参与梯度但作为 context 保留在 response_ids 里

### 4. Docker Container Runner

> 源码: [`_run_claw_container_blocking()` L408-458](../../../verl_patch/experimental/agent_loop/claw_container_agent.py#L408-L458)

```
_run_claw_container_blocking(image, name, workspace_dir, env, network, timeout_sec)  (L408)
  │
  ├── 构建 docker 命令  (L421-437)
  │     docker run --rm --network host -v workspace:/workspace
  │       -e OPENAI_UPSTREAM_BASE_URL=http://<vllm_host>:<port>
  │       -e OPENAI_PROXY_PORT=<random 20000-40000>
  │       -e OPENAI_PROXY_COMPLETION_BUDGET_TOKENS=<budget>
  │       -e CLAW_MODEL=hkust-nlp/drkernel-8b-coldstart
  │       claw-agent-runtime:latest
  │       bash -lc "...entrypoint..."
  │
  ├── subprocess.Popen + proc.wait(timeout)  (L443-445)
  │     正常退出 → return (exit_code, "")
  │
  └── TimeoutExpired:  (L446-453)
        docker rm -f <name>   ← 从外部强杀，防止 orphan container
        proc.wait(30)
        return (124, "container timed out")
```

使用 docker CLI（`subprocess.Popen`）而不是 Docker Engine API，避免额外 python 依赖。
`--rm` 确保 container 退出后自动清理。
`--network host` 让 container 内 proxy 能通过 127.0.0.1 访问宿主机 vLLM。

### 5. ClawContainerAgentLoop 主类

> 源码: [`ClawContainerAgentLoop` L469-733](../../../verl_patch/experimental/agent_loop/claw_container_agent.py#L469-L733)

```
@register("claw_container")  (L468)
class ClawContainerAgentLoop(AgentLoopBase):  (L469)

  类级状态（所有 sample 共享）:
  ├── _semaphore: asyncio.Semaphore(CLAW_CONCURRENCY)   (L479, L507)  ← 全局并发控制
  ├── _cached_upstream: str                              (L480)       ← vLLM 地址缓存
  ├── _upstream_lock: asyncio.Lock                       (L481, L508) ← 首次解析的并发保护
  ├── tokenizer, processor                               (L489-490)   ← 来自 trainer
  ├── prompt_length, response_length                     (L491-492)   ← 来自 config
  └── system_prompt_ids                                  (L501-506)   ← chat template 的空前缀 token

  init_class(config, tokenizer, processor):  (L484)
  ├── 读取所有 CLAW_* 环境变量  (L510-518)
  ├── 预计算 system_prompt_ids（增量 tokenize 用）  (L500-506)
  └── 创建 _semaphore 和 _upstream_lock  (L507-508)

  run(sampling_params, **kwargs) -> AgentLoopOutput:  (L636)
  │
  │  Step 1: 构建 prompt_ids  (L641-648)
  │  ├── messages = kwargs["raw_prompt"]  ← 从 parquet 来的 chat messages
  │  └── tokenizer.apply_chat_template(messages, add_generation_prompt=True)  (L643-646)
  │      → prompt_ids (list[int])
  │
  │  Step 2: 解析 vLLM upstream + 准备 workspace  (L651-656)
  │  ├── _resolve_upstream()  (L526-543)
  │  │     首次调用: server_handles[0].get_server_address.remote()  (L535)
  │  │     后续调用: 用 _cached_upstream  (L530-531)
  │  │     返回: "http://<host>:<port>"  (不带 /v1)  (L540)
  │  ├── _build_task_prompt_text(messages)  (L545) → 纯文本 task
  │  └── _prepare_workspace(task_prompt)  (L560)
  │        写入 prompt.txt, TASK.md, openai_compat_proxy.py  (L562-570)
  │
  │  Step 3: 执行 claw container（受 semaphore 控制）  (L660-664)
  │  ├── async with _semaphore:  (L662)
  │  │     _run_claw(workspace, upstream, sample_index)  (L589)
  │  │       → asyncio.to_thread(_run_claw_container_blocking, ...)  (L600)
  │  │       → 读取 claw_result.json, claw_stderr.log, session JSONL  (L609-622)
  │  └── → _ClawArtifacts  (L626-633)
  │
  │  Step 4: 提取 final message  (L668-679)
  │  ├── 优先: claw_result.json["message"]  (L669-670)
  │  ├── 次选: session 最后一条 assistant message  (L671-676)
  │  └── 兜底: failure_reason 或 "Agent did not return a final answer."  (L678-679)
  │
  │  Step 5: 渲染 + 增量 tokenize  (L682-706)
  │  ├── _render_claw_turns(session_messages, fallback)  (L683)
  │  │     → [(role, content), ...]  见上面 §3
  │  │
  │  ├── for each turn:  (L688-704)
  │  │     _tokenize_turn_delta(turns_so_far, new_turn)  (L573, L689-690)
  │  │       prev_ids = chat_template(turns_so_far)  (L577-579)
  │  │       next_ids = chat_template(turns_so_far + [new_turn])  (L580-582)
  │  │       delta = next_ids[len(prev_ids):]  (L585-586) ← 只取新增 token
  │  │
  │  │     response_ids.extend(delta[:remaining])  (L699)
  │  │     mask_bit = 1 if role=="assistant" else 0  (L700)
  │  │     response_mask.extend([mask_bit] * len(delta))  (L701)
  │  │
  │  └── 如果 response_ids 为空 → 填一个 pad token (mask=0)  (L713-717)
  │
  │  补齐 metrics:  (L710-711)
  │  ├── metrics["generate_sequences"] = claw_container_run 耗时
  │  └── metrics["tool_calls"] = 0.0
  │
  └── return AgentLoopOutput(  (L719-733)
        prompt_ids      = prompt_ids,
        response_ids    = response_ids[:response_length],  (L721)
        response_mask   = response_mask[:response_length],  (L722)
        response_logprobs = None,         ← proxy 不透传 per-token logprob  (L723)
        num_turns       = len(turns) + 1,  (L725)
        metrics         = metrics,  (L726)
        extra_fields    = {  (L727-732)
          "claw_workspace":     str(workspace_dir),
          "claw_exit_code":     exit_code,
          "claw_failure":       failure_reason,
          "claw_final_message": final_message,
        }
      )
```

### 增量 Tokenize 的设计考虑

> 源码: [`_tokenize_turn_delta()` L573-587](../../../verl_patch/experimental/agent_loop/claw_container_agent.py#L573-L587)

为什么不一次性 tokenize 整个 response 文本，而是按 turn 做增量？

```
问题: chat template 在每条 message 周围加 special tokens（如 <|im_start|>assistant\n...<|im_end|>）
     如果把所有 turn 拼成 raw text 再 tokenize，会丢失 role 边界的 special tokens。
     如果整体调 apply_chat_template，无法区分哪些 token 是 assistant（trainable）、哪些是 tool（non-trainable）。

方案: 增量 tokenize  (L573-587)
     对每个 turn:
       prev_ids = apply_chat_template(已处理的 turns)  (L577-579)
       next_ids = apply_chat_template(已处理的 turns + 当前 turn)  (L580-582)
       delta_ids = next_ids[len(prev_ids):]  (L586)
     这样 delta_ids 精确对应当前 turn 的 token（含 role special tokens），可以正确标记 mask。

优化: fast-path 检查  (L585)
     if next_ids[:len(prev_ids)] == prev_ids:
       delta = next_ids[len(prev_ids):]  ← O(1) slice
     else:
       fallback: next_ids[len(system_prompt_ids):]  (L587) ← chat template 非单调时的保底
```

### 并发模型

> 源码: [`_semaphore` L479/L507](../../../verl_patch/experimental/agent_loop/claw_container_agent.py#L479) / [`async with _semaphore` L662](../../../verl_patch/experimental/agent_loop/claw_container_agent.py#L662) / [`asyncio.to_thread` L600](../../../verl_patch/experimental/agent_loop/claw_container_agent.py#L600)

```
AgentLoopWorker (Ray actor)
  └── generate_sequences(batch_chunk)
        └── asyncio.gather(
              _run_agent_loop(sample_0),    ← 每个 sample 一个 async task
              _run_agent_loop(sample_1),
            )
              └── ClawContainerAgentLoop.run()  (L636)
                    └── async with _semaphore:  (L662)  ← 全局并发门控
                          _run_claw()  (L589)
                            └── asyncio.to_thread(  (L600)
                                  _run_claw_container_blocking  (L408)
                                )  ← 不阻塞 event loop

全局并发控制:
  _semaphore = asyncio.Semaphore(CLAW_CONCURRENCY)  (L507)
  所有 AgentLoopWorker 的 ClawContainerAgentLoop 实例共享同一个 class-level semaphore (L479)。
  如果 CLAW_CONCURRENCY=8，最多同时 8 个 docker container 在跑。

  注意: semaphore 是 asyncio 级别的，在同一个 event loop 内有效。
  如果有多个 AgentLoopWorker Ray actor（每个是独立进程），每个 actor 有自己的 event loop，
  所以实际并发上限 = num_workers × CLAW_CONCURRENCY。
  例如 num_workers=2, CLAW_CONCURRENCY=8 → 最多 16 个同时容器。
```

### 错误处理策略

| 场景 | 处理 |
| --- | --- |
| docker CLI 不存在 | `RuntimeError`，立即失败 |
| container 超时 | `docker rm -f` 强杀，返回 exit_code=124 |
| claw 内部出错 | 读 claw_stderr.log 作为 failure_reason，response 为 fallback text |
| proxy 上游 vLLM 不可达 | proxy 返回 502，claw 重试或失败 |
| budget 耗尽 | proxy 返回 429，claw 停止生成 |
| session JSONL 不存在 | session_messages=[]，触发 fallback response |
| response_ids 为空 | 填一个 pad token (mask=0)，trainer 可以正常处理 |

### 与 slime 原始实现的差异

| 方面 | slime (generate_with_claw_agent.py) | DR.Kernel (claw_container_agent.py) |
| --- | --- | --- |
| 运行环境 | 在 slime-retool container 内 | 在宿主机 drkernel310 conda env 内 |
| 网络模式 | `NetworkMode: container:slime-retool` | `--network host` |
| Docker API | 手写 HTTP client 直连 docker.sock | `subprocess.Popen(["docker", "run", ...])` |
| vLLM upstream | SGLang router 或直连 SGLang worker | `AsyncvLLMServer.get_server_address.remote()` |
| 输出格式 | `slime.utils.types.Sample` | `AgentLoopOutput` (VERL protocol) |
| 转录渲染 | XML tags (`<assistant>`, `<tool_call>`) | Qwen3 chat template (增量 tokenize) |
| loss mask | 手动标记 trainable segments | `response_mask`: assistant=1, tool=0 |
| reward | slime 自带 math_dapo_compute_score | DR.Kernel kernel_async → KernelGYM /evaluate |
| logprobs | 无 (proxy 不透传) | 无 (同左，trainer 做 recomputation) |

## 回滚思路

如果要回到默认 DR.Kernel RL:

1. 不使用 [8b_claw_container_agent_local.sh](8b_claw_container_agent_local.sh)，改回原始 launcher。
2. 移除 `EXTRA_HYDRA_OVERRIDES` 里对 `agent_loop_config_path`、`default_agent_name`、`claw_empty_tools.yaml` 的覆盖。
3. 保留 `default_agent_name` schema 改动也不影响默认路径，因为默认值是 `single_turn_agent`。
4. 如果要完全清理，再移除 `claw_container_agent.py`、`claw_agent_loop_configs.yaml`、`claw_empty_tools.yaml` 和 `__init__.py` 里的 import。
