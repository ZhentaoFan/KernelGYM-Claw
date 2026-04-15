#!/usr/bin/env bash
# Variant of 8b_trloo_mrs_pr_prs_local.sh that routes every rollout sample
# through the `claw_container` agent loop (ClawContainerAgentLoop). Each
# sample spawns a sibling docker container running `claw`, which talks to
# the training-side vLLM via an in-container OpenAI-compat proxy.
#
# Before running:
#   - KernelGYM server should already be up on ${KERNELGYM_SERVER_URL} (GPU 2,3)
#   - `claw-agent-runtime:latest` docker image must be present on the host
#   - Host must have docker CLI reachable and the current user able to run docker
#   - GPUs 4,5,6,7 should be free (we use them for training + rollout as usual)
#
# Smoke-test tips:
#   - Set CLAW_AGENT_IMAGE / CLAW_MAX_COMPLETION_TOKENS / CLAW_CONCURRENCY via env
#   - Set ROLLOUT_N=1 and TRAIN_BATCH_SIZE=4 for a very fast sanity pass
#   - CLAW_AGENT_TIMEOUT_SEC caps per-sample wallclock
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ENV_DIR="${ENV_DIR:-${ROOT_DIR}/.venv/drkernel310}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}/drkernel:${ROOT_DIR}:${ROOT_DIR}/drkernel/verl:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MERLIN_JOB_ID="${MERLIN_JOB_ID:-local-drkernel-claw}"
export ARNOLD_MONITOR_TRIAL_ID="${ARNOLD_MONITOR_TRIAL_ID:-local}"
export GIT_COMMIT_URL="${GIT_COMMIT_URL:-local}"

export ARNOLD_WORKER_NUM="${ARNOLD_WORKER_NUM:-1}"
export ARNOLD_WORKER_GPU="${ARNOLD_WORKER_GPU:-4}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export KERNELGYM_SERVER_URL="${KERNELGYM_SERVER_URL:-http://127.0.0.1:10907}"
export HDFS_DATA_PATH="${HDFS_DATA_PATH:-${ROOT_DIR}/drkernel/data}"
export HDFS_CHECKPOINT_PATH="${HDFS_CHECKPOINT_PATH:-${ROOT_DIR}/checkpoints/drkernel}"

# ---- claw-container agent env (consumed by ClawContainerAgentLoop) -----------
export CLAW_AGENT_IMAGE="${CLAW_AGENT_IMAGE:-claw-agent-runtime:latest}"
export CLAW_WORKSPACE_ROOT="${CLAW_WORKSPACE_ROOT:-${ROOT_DIR}/tmp/claw_drkernel_rollouts}"
export CLAW_DOCKER_SOCKET="${CLAW_DOCKER_SOCKET:-/var/run/docker.sock}"
export CLAW_CONTAINER_NETWORK="${CLAW_CONTAINER_NETWORK:-host}"
export CLAW_MAX_COMPLETION_TOKENS="${CLAW_MAX_COMPLETION_TOKENS:-${MAX_RESPONSE_LENGTH:-8192}}"
export CLAW_CONCURRENCY="${CLAW_CONCURRENCY:-8}"
export CLAW_AGENT_TIMEOUT_SEC="${CLAW_AGENT_TIMEOUT_SEC:-900}"
export CLAW_MODEL_NAME="${CLAW_MODEL_NAME:-hkust-nlp/drkernel-8b-coldstart}"
mkdir -p "${CLAW_WORKSPACE_ROOT}"

cd "${ROOT_DIR}/drkernel"

TRAIN_DATASET=("hkust-nlp/drkernel-rl-data")
VALID_DATASET=("hkust-nlp/drkernel-validation-data")
MODEL_NAME="${MODEL_NAME:-drkernel-8b-coldstart}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/hkust-nlp/drkernel-8b-coldstart}"

RUN_NAME="${RUN_NAME:-drkernel-8b-claw-container}"
REWARD_MANAGER=kernel_async
REWARD_FUNC_NAME="calculate_reward_speedup"
ROLLOUT_MODE=async_agent

ALGORITHM="trloo"

SPEEDUP_REWARD_UPPER_BOUND=3.0
SPEEDUP_REWARD_LOWER_BOUND=0.0

ROLLOUT_RS="geometric"
ROLLOUT_TOKEN_VETO_THRESHOLD=1e-4
ROLLOUT_RS_KWARGS="{lower:0.999,upper:1.001}"

COVERAGE_RS="turn"
COVERAGE_RS_THRESHOLD=0.3
COVERAGE_RS_FACTOR=0.1
COVERAGE_RS_KEY="time_coverage"

COVERAGE_REWARD_TYPE="time_coverage"
COVERAGE_REWARD_WEIGHT=0.5
COVERAGE_REWARD_ENABLE=True

REWARD_TASK_TIMEOUT=300
REWARD_TIMEOUT=1800
REWARD_ACQUIRE_TIMEOUT=2400
REWARD_MAX_CONCURRENT=32
REWARD_MAX_RETRIES=3
REWARD_PRINT_STATUS=True
NUM_PERF_TRIALS=100
REWARD_TASK_TIMEOUT_CLIENT=2400

VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
IS_GET_LAST_TURN=True

# The container-agent (claw) owns multi-turn internally; the VERL trainer should
# NOT try to expand/repeat the batch for multi-turn. The container entrypoint
# now forces its own ReAct loop: claw -> extract code -> KernelGYM feedback ->
# claw follow-up prompt. The agent loop still returns one flat trajectory per
# sample, with feedback tokens masked out of policy loss.
ENABLE_MULTI_TURN=False
MAX_TURN=3
export CLAW_REACT_MAX_TURNS="${CLAW_REACT_MAX_TURNS:-${MAX_TURN}}"
export CLAW_REACT_STOP_ON_OK="${CLAW_REACT_STOP_ON_OK:-false}"
N_VAL="${N_VAL:-2}"
ACTOR_OPTIMIZER_OFFLOAD=True
ACTOR_PARAMETER_OFFLOAD=True
LEARNING_RATE=1e-6

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"

AUTOMATIC_OVERSAMPLING=False
REJECTION_SAMPLE=True

PPO_MICRO_TOKEN=null
CLIP_RATIO=0.2_0.28
ENTROPY_CLIP_RATE=0.0
GRAD_CLIP=1.0
VLLM_IS_THRESHOLD=2.0
EXTREME_RISK_PROB_THRESHOLD=null
KL_LOSS_COEF=0.0
ENTROPY_COEFFIENT=0.0
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
MIN_P=0.0
TOP_P=1.0
TOP_K=-1
ROLLOUT_N="${ROLLOUT_N:-16}"
KL_COEF=0.0
TOTAL_EPOCHS=1000
ROLLOUT_GPU_MEMORY_UTIL="${ROLLOUT_GPU_MEMORY_UTIL:-0.75}"

SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ=10
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
SP_SIZE="${SP_SIZE:-4}"
APPLY_CHAT_TEMPLATE=True
FREE_CACHE_ENGINE=True
ENFORCE_EAGER=False
NNODES="${ARNOLD_WORKER_NUM}"
GPUS_PER_NODE="${ARNOLD_WORKER_GPU}"

if [[ "$GPUS_PER_NODE" =~ ^[0-9]+$ && "$SP_SIZE" =~ ^[0-9]+$ && "$ROLLOUT_N" =~ ^[0-9]+$ && "$PPO_MINI_BATCH_SIZE" =~ ^[0-9]+$ && "$TRAIN_BATCH_SIZE" =~ ^[0-9]+$ ]]; then
  if (( SP_SIZE <= 0 || GPUS_PER_NODE % SP_SIZE != 0 )); then
    echo "[claw launcher] ERROR: SP_SIZE=${SP_SIZE} must be a positive divisor of GPUS_PER_NODE=${GPUS_PER_NODE}" >&2
    exit 1
  fi

  DP_SIZE=$((GPUS_PER_NODE / SP_SIZE))
  MIN_PPO_MINI_BATCH_SIZE=$(((DP_SIZE + ROLLOUT_N - 1) / ROLLOUT_N))

  if (( PPO_MINI_BATCH_SIZE < MIN_PPO_MINI_BATCH_SIZE )); then
    echo "[claw launcher] Adjusting PPO_MINI_BATCH_SIZE ${PPO_MINI_BATCH_SIZE} -> ${MIN_PPO_MINI_BATCH_SIZE} so verl FSDP normalization stays positive (gpus=${GPUS_PER_NODE}, sp=${SP_SIZE}, rollout_n=${ROLLOUT_N})."
    PPO_MINI_BATCH_SIZE="${MIN_PPO_MINI_BATCH_SIZE}"
  fi

  if (( TRAIN_BATCH_SIZE < DP_SIZE )); then
    echo "[claw launcher] Adjusting TRAIN_BATCH_SIZE ${TRAIN_BATCH_SIZE} -> ${DP_SIZE} to keep the smoke batch aligned with ${DP_SIZE} FSDP data-parallel ranks."
    TRAIN_BATCH_SIZE="${DP_SIZE}"
  fi
else
  echo "[claw launcher] Skipping FSDP batch-size guard for non-integer config: GPUS_PER_NODE=${GPUS_PER_NODE}, SP_SIZE=${SP_SIZE}, ROLLOUT_N=${ROLLOUT_N}, PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}, TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
fi

# For claw-container, response_length should match CLAW_MAX_COMPLETION_TOKENS
# to avoid padding overhead and config inconsistency.
MAX_PROMPT_LENGTH=10240
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-${CLAW_MAX_COMPLETION_TOKENS:-8192}}"
PROMPT_OVERSAMPLING_FACTOR=1.0
SAMPLE_OVERSAMPLING_FACTOR=1.0
SAMPLE_SELECTION_STRATEGY=efficiency_stochastic
MAX_SKIP_STEPS=5

# ---- claw-container routing ------------------------------------------------
# 1) Point verl's agent-loop registry at our YAML (adds the claw_container entry).
# 2) Force every sample to use that agent via default_agent_name.
# train_rl_common.sh run_training appends ${EXTRA_HYDRA_OVERRIDES:-} at the end
# of the python argv, so we just export it here.
CLAW_AGENT_LOOP_CONFIG="${CLAW_AGENT_LOOP_CONFIG:-${SCRIPT_DIR}/claw_agent_loop_configs.yaml}"
CLAW_EMPTY_TOOLS="${SCRIPT_DIR}/claw_empty_tools.yaml"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
MAX_CRITIC_CKPT_TO_KEEP="${MAX_CRITIC_CKPT_TO_KEEP:-1}"
export EXTRA_HYDRA_OVERRIDES="actor_rollout_ref.rollout.agent.agent_loop_config_path=${CLAW_AGENT_LOOP_CONFIG} actor_rollout_ref.rollout.agent.default_agent_name=claw_container actor_rollout_ref.rollout.agent.num_workers=${CLAW_CONCURRENCY:-8} actor_rollout_ref.rollout.multi_turn.tool_config_path=${CLAW_EMPTY_TOOLS} actor_rollout_ref.actor.use_torch_compile=false actor_rollout_ref.actor.fsdp_config.optimizer_offload=true actor_rollout_ref.ref.use_torch_compile=false trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP} trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP}"

source "${SCRIPT_DIR}/train_rl_common.sh"
main "$@"
