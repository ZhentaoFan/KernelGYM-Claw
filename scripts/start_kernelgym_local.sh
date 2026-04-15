#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-/home/ubuntu/z84318463/envs/drkernel310}"
LOG_DIR="${LOG_DIR:-logs/drkernel_local}"
ENV_FILE="${ROOT_DIR}/.env"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5}"
export GPU_DEVICES="${GPU_DEVICES:-[0,1,2]}"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/drkernel:${ROOT_DIR}/drkernel/verl:${PYTHONPATH:-}"

mkdir -p "${ROOT_DIR}/${LOG_DIR}"

cd "${ROOT_DIR}"
if [ -f "${ENV_FILE}" ]; then
  if grep -q '^GPU_DEVICES=' "${ENV_FILE}"; then
    sed -i "s|^GPU_DEVICES=.*|GPU_DEVICES=${GPU_DEVICES}|" "${ENV_FILE}"
  else
    printf '\nGPU_DEVICES=%s\n' "${GPU_DEVICES}" >> "${ENV_FILE}"
  fi
fi
exec ./start_all_with_monitor.sh --log-dir "${LOG_DIR}"
