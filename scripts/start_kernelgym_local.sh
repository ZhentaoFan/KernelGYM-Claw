#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-${ROOT_DIR}/.venv/drkernel310}"
LOG_DIR="${LOG_DIR:-logs/drkernel_local}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/drkernel:${ROOT_DIR}/drkernel/verl:${PYTHONPATH:-}"

mkdir -p "${ROOT_DIR}/${LOG_DIR}"

cd "${ROOT_DIR}"
exec ./start_all_with_monitor.sh --log-dir "${LOG_DIR}"
