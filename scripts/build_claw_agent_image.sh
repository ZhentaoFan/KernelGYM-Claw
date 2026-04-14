#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLAW_CODE_SRC="${CLAW_CODE_SRC:-${ROOT_DIR}/third_party/claw-code}"
CLAW_CODE_REPO="${CLAW_CODE_REPO:-https://github.com/ultraworkers/claw-code}"
CLAW_AGENT_IMAGE="${CLAW_AGENT_IMAGE:-claw-agent-runtime:latest}"

if [ ! -d "${CLAW_CODE_SRC}/.git" ]; then
    mkdir -p "$(dirname -- "${CLAW_CODE_SRC}")"
    git clone "${CLAW_CODE_REPO}" "${CLAW_CODE_SRC}"
fi

docker build \
    -t "${CLAW_AGENT_IMAGE}" \
    -f "${ROOT_DIR}/docker/claw_agent_runtime.Containerfile" \
    "${CLAW_CODE_SRC}"
