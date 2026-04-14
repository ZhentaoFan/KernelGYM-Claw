#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-${ROOT_DIR}/.venv/drkernel310}"
PYTHON="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

if [ ! -x "${PYTHON}" ]; then
  echo "Missing Python env at ${ENV_DIR}. Create it first, for example:"
  echo "  conda create -p ${ENV_DIR} python=3.10 -y"
  exit 1
fi

cd "${ROOT_DIR}/drkernel"
git submodule update --init

"${PIP}" install -e "${ROOT_DIR}/drkernel/verl" --no-build-isolation --no-deps

"${PIP}" install --no-cache-dir "ray==2.47.1"

"${PIP}" install --no-cache-dir \
  "vllm==0.10.2" "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" \
  tensordict torchdata "transformers[hf_xet]==4.56.0" accelerate datasets peft hf-transfer \
  "numpy<2.0.0" "pyarrow>=15.0.0" pandas codetiming hydra-core pylatexenc \
  qwen-vl-utils dill pybind11 liger-kernel mathruler decord torchcodec \
  pytest yapf py-spy pre-commit ruff uv pipx sandbox-fusion logfire gradio \
  huggingface_hub "protobuf==3.20" "wandb==0.16.6"

"${PIP}" install --no-cache-dir -r "${ROOT_DIR}/requirements.txt" pydantic-settings

ABI_FLAG="${ABI_FLAG:-FALSE}"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abi${ABI_FLAG}-cp310-cp310-linux_x86_64.whl"
FLASH_ATTN_WHEEL="${ROOT_DIR}/drkernel/$(basename "${FLASH_ATTN_URL}")"

if [ ! -f "${FLASH_ATTN_WHEEL}" ]; then
  wget -nv -O "${FLASH_ATTN_WHEEL}" "${FLASH_ATTN_URL}"
fi

"${PIP}" install --no-cache-dir "${FLASH_ATTN_WHEEL}"

"${PYTHON}" - <<'PY'
import importlib

mods = ["torch", "ray", "vllm", "verl", "kernel", "kernelgym", "huggingface_hub"]
for mod in mods:
    imported = importlib.import_module(mod)
    print(f"{mod}: ok")
PY
