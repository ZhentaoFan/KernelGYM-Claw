# Copyright 2026 claw-container-agent port.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This file ports slime's `generate_with_claw_agent` pattern to the VERL agent-loop
# framework used by DR.Kernel. Each sample spawns a sibling Docker container that
# runs the `claw` CLI talking to the training-side vLLM server via an in-container
# OpenAI-compat proxy. The final claw session transcript is re-tokenized through
# the trainer's chat template and returned as an AgentLoopOutput.
"""Claw-container agent loop for VERL.

Environment knobs (override per launcher):
    CLAW_AGENT_IMAGE                   docker image (default: claw-agent-runtime:latest)
    CLAW_WORKSPACE_ROOT                host dir for per-sample scratch workspaces
    CLAW_DOCKER_SOCKET                 path to docker socket (default: /var/run/docker.sock)
    CLAW_MODEL_NAME                    model name passed to `claw --model`
    CLAW_MAX_COMPLETION_TOKENS         per-sample completion-token budget (proxy cap)
    CLAW_CONTAINER_NETWORK             docker network mode (default: host)
    CLAW_AGENT_TIMEOUT_SEC             per-sample container wallclock timeout (default: 900)
    CLAW_CONCURRENCY                   max simultaneous containers (default: 16)
    CLAW_OPENAI_API_KEY                dummy API key (default: local-dev-token)
    CLAW_UPSTREAM_BASE_URL             explicit upstream override; else we query server_manager
"""

from __future__ import annotations

import asyncio
import http.client
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import ray
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from verl_patch.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# In-container OpenAI-compat proxy script (identical contract to slime's one):
# - Listens on 127.0.0.1:$OPENAI_PROXY_PORT
# - Forwards /chat/completions to $OPENAI_UPSTREAM_BASE_URL
# - Caps max_tokens to remaining per-sample budget
# - Tracks completion_tokens and returns 429 once the budget is exhausted
# ---------------------------------------------------------------------------
OPENAI_COMPAT_PROXY_SCRIPT = r"""#!/usr/bin/env python3
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


UPSTREAM_BASE_URL = os.environ["OPENAI_UPSTREAM_BASE_URL"].rstrip("/")
LISTEN_HOST = os.environ.get("OPENAI_PROXY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ["OPENAI_PROXY_PORT"])
REQUEST_TIMEOUT_SEC = float(os.environ.get("OPENAI_PROXY_TIMEOUT_SEC", "900"))
COMPLETION_BUDGET_TOKENS = int(os.environ.get("OPENAI_PROXY_COMPLETION_BUDGET_TOKENS", "0"))
_remaining_completion_budget = COMPLETION_BUDGET_TOKENS if COMPLETION_BUDGET_TOKENS > 0 else None
_budget_lock = threading.Lock()


def remaining_completion_budget():
    with _budget_lock:
        return _remaining_completion_budget


def consume_completion_budget(tokens):
    global _remaining_completion_budget
    if _remaining_completion_budget is None:
        return
    decrement = max(0, int(tokens or 0))
    with _budget_lock:
        _remaining_completion_budget = max(0, _remaining_completion_budget - decrement)


def patch_chat_completions_request(request_body):
    if _remaining_completion_budget is None or not request_body:
        return request_body, None, False
    try:
        payload = json.loads(request_body.decode("utf-8"))
    except Exception:
        return request_body, None, False
    if not isinstance(payload, dict):
        return request_body, None, False

    requested_max_tokens = payload.get("max_tokens")
    if requested_max_tokens is None:
        requested_max_tokens = payload.get("max_completion_tokens")
    if requested_max_tokens is not None:
        try:
            requested_max_tokens = int(requested_max_tokens)
        except Exception:
            requested_max_tokens = None

    remaining = remaining_completion_budget()
    if remaining is None:
        return request_body, None, False
    if remaining <= 0:
        return request_body, {"remaining": 0, "requested": requested_max_tokens}, True

    effective = remaining if requested_max_tokens is None else min(requested_max_tokens, remaining)
    payload["max_tokens"] = effective
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = effective
    return json.dumps(payload).encode("utf-8"), {"remaining": remaining, "requested": requested_max_tokens, "capped_to": effective}, False


def extract_completion_tokens(raw_body):
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if isinstance(usage, dict):
        return usage.get("completion_tokens")
    return None


def log_line(msg):
    print(msg, file=sys.stderr, flush=True)


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "ClawProxy/1.0"

    def log_message(self, fmt, *args):
        log_line("proxy " + (fmt % args))

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            remaining = remaining_completion_budget()
            body = json.dumps({"status": "ok", "remaining_completion_tokens": remaining}).encode()
            self.wfile.write(body)
            return
        self._forward("GET", self.path, b"")

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else b""
        is_chat = self.path.rstrip("/").endswith("/chat/completions")
        exhausted = False
        meta = None
        if is_chat:
            body, meta, exhausted = patch_chat_completions_request(body)
            if meta is not None:
                log_line(f"chat_completions meta={meta}")
        if exhausted:
            err = {"error": {"type": "budget_exhausted", "message": "completion budget exhausted"}}
            payload = json.dumps(err).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._forward("POST", self.path, body)

    def _forward(self, method, path, body):
        url = UPSTREAM_BASE_URL + path
        headers = {}
        for key in ("Content-Type", "Authorization", "Accept"):
            if self.headers.get(key):
                headers[key] = self.headers[key]
        if body and "Content-Length" not in headers:
            headers["Content-Length"] = str(len(body))
        request = urllib.request.Request(url=url, data=body if body else None, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
                raw = response.read()
                status = response.status
                content_type = response.headers.get("Content-Type", "application/json")
                if path.rstrip("/").endswith("/chat/completions"):
                    consumed = extract_completion_tokens(raw)
                    if consumed is not None:
                        consume_completion_budget(consumed)
                        log_line(f"consumed={consumed} remaining={remaining_completion_budget()}")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        except urllib.error.HTTPError as e:
            err_body = e.read() if hasattr(e, "read") else b""
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json") if e.headers else "application/json")
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)
        except Exception as exc:
            err = {"error": {"type": "proxy_upstream_error", "message": str(exc)}}
            payload = json.dumps(err).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


def main():
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), ProxyHandler)
    log_line(f"listening on {LISTEN_HOST}:{LISTEN_PORT} upstream={UPSTREAM_BASE_URL} budget={COMPLETION_BUDGET_TOKENS}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
"""


def _container_entrypoint_script() -> str:
    return (
        "set -euo pipefail\n"
        "mkdir -p /workspace/home/.claw\n"
        "export CLAW_CONFIG_HOME=/workspace/home/.claw\n"
        "python3 /workspace/openai_compat_proxy.py > /workspace/openai_proxy.log 2>&1 &\n"
        "PROXY_PID=$!\n"
        "for _ in $(seq 1 80); do\n"
        "  if curl -sf http://127.0.0.1:${OPENAI_PROXY_PORT}/health >/dev/null; then break; fi\n"
        "  sleep 0.1\n"
        "done\n"
        "export OPENAI_BASE_URL=http://127.0.0.1:${OPENAI_PROXY_PORT}/v1\n"
        "set +e\n"
        "python3 /workspace/claw_react_loop.py > /workspace/claw_react_stdout.log 2> /workspace/claw_stderr.log\n"
        "CLAW_EXIT=$?\n"
        "chmod -R a+rwX /workspace >/dev/null 2>&1 || true\n"
        "kill $PROXY_PID >/dev/null 2>&1 || true\n"
        "exit $CLAW_EXIT\n"
    )


# ---------------------------------------------------------------------------
# Artifacts and transcript parsing
# ---------------------------------------------------------------------------


@dataclass
class _ClawArtifacts:
    exit_code: int
    workspace_dir: Path
    result_json: Optional[dict] = None
    react_summary: Optional[dict] = None
    session_messages: list[dict] = field(default_factory=list)
    stderr_text: str = ""
    failure_reason: Optional[str] = None


@dataclass
class _KernelGymContext:
    reference_code: str = ""
    entry_point: str = "Model"
    uuid: str = ""
    data_source: str = ""
    is_valid: bool = False


def _support_script_text(filename: str) -> str:
    return (Path(__file__).resolve().parent / filename).read_text(encoding="utf-8")


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _coerce_mapping(value.item())
        except Exception:
            pass
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.item()
        except Exception:
            pass
    return str(value)


def _infer_entry_point(reference_code: str, default: str = "Model") -> str:
    if not reference_code:
        return default
    marker = "class "
    start = reference_code.find(marker)
    if start < 0:
        return default
    start += len(marker)
    end = start
    while end < len(reference_code) and (reference_code[end].isalnum() or reference_code[end] == "_"):
        end += 1
    candidate = reference_code[start:end].strip()
    return candidate or default


def _latest_session_file(workspace_dir: Path) -> Optional[Path]:
    sessions_root = workspace_dir / ".claw" / "sessions"
    if not sessions_root.exists():
        return None
    session_files = list(sessions_root.rglob("*.jsonl"))
    if not session_files:
        return None
    return max(session_files, key=lambda p: p.stat().st_mtime)


def _load_jsonl_messages(path: Path) -> list[dict]:
    messages: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if record.get("type") == "message" and isinstance(record.get("message"), dict):
                messages.append(record["message"])
    return messages


def _load_session_messages(workspace_dir: Path) -> list[dict]:
    react_transcript = workspace_dir / "claw_react_transcript.jsonl"
    if react_transcript.exists():
        try:
            messages = _load_jsonl_messages(react_transcript)
            if messages:
                return messages
        except Exception:
            pass

    session_file = _latest_session_file(workspace_dir)
    if session_file is None:
        return []
    return _load_jsonl_messages(session_file)


def _block_text(block: dict) -> str:
    if block.get("type") == "text":
        return block.get("text", "") or ""
    return ""


def _message_plain_text(message: dict) -> str:
    parts = []
    for block in message.get("blocks", []) or []:
        parts.append(_block_text(block))
    return "\n".join(p for p in parts if p).strip()


def _trim(s: str, limit: int = 4096) -> str:
    if s is None:
        return ""
    if len(s) <= limit:
        return s
    return s[: limit].rstrip() + "\n...<truncated>"


def _render_claw_turns(session_messages: list[dict], fallback_text: str) -> list[tuple[str, str]]:
    """Flatten a claw session into an ordered list of (role, content) tuples suitable for
    re-feeding through the trainer's chat template. We map:
        - claw user (first one) -> dropped (already in prompt_ids)
        - claw user (subsequent)-> role=user (rare; claw rarely emits follow-up user messages)
        - claw assistant text / tool_use -> role=assistant (text only; tool_use serialized as JSON)
        - claw tool (tool_result)        -> role=tool
    """
    turns: list[tuple[str, str]] = []
    skipped_first_user = False
    joined_assistant_text: list[str] = []

    for message in session_messages:
        role = message.get("role")
        blocks = message.get("blocks", []) or []
        if role == "system":
            continue
        if role == "user":
            text = _message_plain_text(message)
            if not skipped_first_user:
                skipped_first_user = True
                continue
            if text:
                turns.append(("user", text))
            continue
        if role == "assistant":
            parts: list[str] = []
            for block in blocks:
                btype = block.get("type")
                if btype == "text" and block.get("text"):
                    parts.append(block["text"])
                elif btype == "tool_use":
                    payload = {
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input"),
                    }
                    parts.append("<tool_call>" + json.dumps(payload, ensure_ascii=False, sort_keys=True) + "</tool_call>")
            content = "\n".join(p for p in parts if p).strip()
            if content:
                turns.append(("assistant", content))
                joined_assistant_text.append(content)
            continue
        if role == "tool":
            pieces: list[str] = []
            for block in blocks:
                if block.get("type") != "tool_result":
                    continue
                name = block.get("tool_name", "tool")
                error = "true" if block.get("is_error") else "false"
                output = _trim(block.get("output", ""))
                pieces.append(f"<tool_result name=\"{name}\" error=\"{error}\">\n{output}\n</tool_result>")
            content = "\n".join(pieces).strip()
            if content:
                turns.append(("tool", content))
            continue

    # Ensure the fallback / final answer text is present in trainable tokens.
    joined = "\n".join(joined_assistant_text)
    if fallback_text and fallback_text.strip() and fallback_text.strip() not in joined:
        turns.append(("assistant", fallback_text.strip()))

    if not turns:
        turns.append(("assistant", fallback_text.strip() or "Agent did not produce a response."))
    return turns


# ---------------------------------------------------------------------------
# Docker container runner (CLI based, avoids extra python deps)
# ---------------------------------------------------------------------------


def _run_claw_container_blocking(
    *,
    image: str,
    name: str,
    workspace_dir: Path,
    env: dict,
    network: str,
    timeout_sec: float,
) -> tuple[int, str]:
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        raise RuntimeError("docker CLI not found on PATH; cannot run claw container agent")

    cmd = [
        docker_bin,
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        network,
        "-v",
        f"{workspace_dir}:/workspace",
        "-w",
        "/workspace",
    ]
    for k, v in env.items():
        cmd.extend(["-e", f"{k}={v}"])
    cmd.append(image)
    cmd.extend(["bash", "-lc", _container_entrypoint_script()])

    stdout_path = workspace_dir / "container_stdout.log"
    stderr_path = workspace_dir / "container_stderr.log"
    try:
        with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err)
        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            # Force-kill the container from outside the docker-run process to avoid orphaning.
            subprocess.run([docker_bin, "rm", "-f", name], check=False, capture_output=True)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            return 124, "container timed out"
        return int(proc.returncode), ""
    except Exception as exc:
        subprocess.run([docker_bin, "rm", "-f", name], check=False, capture_output=True)
        return 1, str(exc)


# ---------------------------------------------------------------------------
# Agent loop registration
# ---------------------------------------------------------------------------


_CONCURRENCY_DEFAULT = int(os.getenv("CLAW_CONCURRENCY", "16"))


@register("claw_container")
class ClawContainerAgentLoop(AgentLoopBase):
    """Runs each sample inside an isolated docker container that executes `claw`.

    The container talks to the training-side vLLM server through an in-container
    OpenAI-compat proxy that enforces a per-sample completion-token budget. The
    resulting claw session transcript is re-tokenized through the trainer's chat
    template and returned as an AgentLoopOutput.
    """

    _class_initialized = False
    _semaphore: Optional[asyncio.Semaphore] = None
    _cached_upstream: Optional[str] = None
    _upstream_lock: Optional[asyncio.Lock] = None

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        logger.info("ClawContainerAgentLoop: class-level init")
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.model_name = Path(str(config.actor_rollout_ref.model.path)).name
        # Pre-compute the bare-system-prompt tokenization so we can strip it when
        # emitting per-turn deltas (same trick used by ToolAgentLoop).
        # Qwen3's Jinja template requires every message to have a 'role' key,
        # so we use an empty system message instead of a bare empty dict.
        try:
            cls.system_prompt_ids = tokenizer.apply_chat_template(
                [{"role": "system", "content": ""}], add_generation_prompt=False, tokenize=True,
                **cls.apply_chat_template_kwargs
            )
        except Exception:
            cls.system_prompt_ids = []
        cls._semaphore = asyncio.Semaphore(int(os.getenv("CLAW_CONCURRENCY", str(_CONCURRENCY_DEFAULT))))
        cls._upstream_lock = asyncio.Lock()

        cls.image = os.getenv("CLAW_AGENT_IMAGE", "claw-agent-runtime:latest")
        default_workspace_root = Path.cwd() / "claw_agent_rollouts" / "drkernel"
        cls.workspace_root = Path(os.getenv("CLAW_WORKSPACE_ROOT", str(default_workspace_root)))
        cls.workspace_root.mkdir(parents=True, exist_ok=True)
        cls.container_network = os.getenv("CLAW_CONTAINER_NETWORK", "host")
        cls.agent_timeout_sec = float(os.getenv("CLAW_AGENT_TIMEOUT_SEC", "900"))
        cls.completion_budget_tokens = int(os.getenv("CLAW_MAX_COMPLETION_TOKENS", str(cls.response_length)))
        cls.claw_model = os.getenv("CLAW_MODEL_NAME", cls.model_name)
        cls.openai_api_key = os.getenv("CLAW_OPENAI_API_KEY", "local-dev-token")
        cls.explicit_upstream = os.getenv("CLAW_UPSTREAM_BASE_URL") or None
        cls.react_max_turns = int(os.getenv("CLAW_REACT_MAX_TURNS", str(cls.max_user_turns or 3)))
        cls.react_stop_on_ok = os.getenv("CLAW_REACT_STOP_ON_OK", "false")
        cls.kernelgym_server_url = os.getenv("KERNELGYM_SERVER_URL", "http://127.0.0.1:10907")
        cls.kernelgym_task_timeout = int(
            os.getenv("CLAW_KERNELGYM_TASK_TIMEOUT", str(getattr(config.reward_model, "task_timeout", 300)))
        )
        cls.kernelgym_task_timeout_client = int(
            os.getenv(
                "CLAW_KERNELGYM_TASK_TIMEOUT_CLIENT",
                str(getattr(config.reward_model, "task_timeout_in_client", 2400)),
            )
        )
        cls.kernelgym_num_correct_trials = int(
            os.getenv("CLAW_KERNELGYM_NUM_CORRECT_TRIALS", str(getattr(config.reward_model, "num_correct_trials", 5)))
        )
        cls.kernelgym_num_perf_trials = int(
            os.getenv("CLAW_KERNELGYM_NUM_PERF_TRIALS", str(getattr(config.reward_model, "num_perf_trials", 20)))
        )
        cls.kernelgym_reference_backend = os.getenv(
            "REFERENCE_BACKEND", os.getenv("KERNELGYM_REFERENCE_BACKEND", str(getattr(config.reward_model, "reference_backend", "pytorch")))
        )
        cls.kernelgym_speedup_upper = os.getenv(
            "SPEEDUP_REWARD_UPPER_BOUND", str(getattr(config.reward_model, "speedup_reward_upper_bound", 3.0))
        )
        cls.kernelgym_speedup_lower = os.getenv(
            "SPEEDUP_REWARD_LOWER_BOUND", str(getattr(config.reward_model, "speedup_reward_lower_bound", 0.0))
        )
        coverage_cfg = getattr(config.reward_model, "coverage_reward", {})
        cls.kernelgym_coverage_enable = os.getenv("COVERAGE_REWARD_ENABLE", str(getattr(coverage_cfg, "enable", False)))
        cls.kernelgym_coverage_weight = os.getenv("COVERAGE_REWARD_WEIGHT", str(getattr(coverage_cfg, "weight", 0.5)))
        cls.kernelgym_coverage_type = os.getenv("COVERAGE_REWARD_TYPE", str(getattr(coverage_cfg, "reward_type", "time_coverage")))
        cls.kernelgym_init_correct_weight = str(getattr(config.reward_model, "init_correct_weight", 0.5))
        cls.kernelgym_init_performance_weight = str(getattr(config.reward_model, "init_performance_weight", 0.5))
        cls.kernelgym_speedup_eps = str(getattr(config.reward_model, "speedup_eps", 0.01))
        try:
            cls.kernelgym_penalty_score = str(config.reward_model.reward_policy.penalties.penalty_score)
        except Exception:
            cls.kernelgym_penalty_score = "0.0"

        logger.info(
            "ClawContainerAgentLoop config image=%s workspace=%s network=%s budget=%d model=%s react_turns=%d kernelgym=%s upstream_override=%s",
            cls.image, cls.workspace_root, cls.container_network,
            cls.completion_budget_tokens, cls.claw_model, cls.react_max_turns,
            cls.kernelgym_server_url, cls.explicit_upstream,
        )

    async def _resolve_upstream(self) -> str:
        if self.explicit_upstream:
            return self.explicit_upstream.rstrip("/")
        async with self._upstream_lock:
            if type(self)._cached_upstream:
                return type(self)._cached_upstream
            server_handles = self.server_manager.server_handles
            if not server_handles:
                raise RuntimeError("ClawContainerAgentLoop: no AsyncvLLMServer handles available")
            addr = await server_handles[0].get_server_address.remote()
            # NOTE: do NOT append /v1 here. The in-container proxy concatenates
            # UPSTREAM_BASE_URL + request_path, and the claw CLI already sends
            # requests to /v1/chat/completions. Adding /v1 here would cause
            # double-prefix (/v1/v1/...) → 404.
            upstream = f"http://{addr}"
            type(self)._cached_upstream = upstream
            logger.info("ClawContainerAgentLoop: resolved vLLM upstream=%s", upstream)
            return upstream

    def _build_task_prompt_text(self, messages: list[dict]) -> str:
        parts: list[str] = []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
        task = "\n".join(p for p in parts if p).strip()
        return task

    def _build_kernelgym_context(self, kwargs: dict[str, Any]) -> _KernelGymContext:
        reward_model = _coerce_mapping(kwargs.get("reward_model"))
        extra_info = _coerce_mapping(kwargs.get("extra_info"))
        reference_code = (
            _string_or_empty(kwargs.get("ground_truth"))
            or _string_or_empty(reward_model.get("ground_truth"))
            or _string_or_empty(extra_info.get("ground_truth"))
        )
        entry_point = (
            _string_or_empty(kwargs.get("entry_point"))
            or _string_or_empty(extra_info.get("entry_point"))
            or _infer_entry_point(reference_code)
        )
        task_uuid = (
            _string_or_empty(kwargs.get("uuid"))
            or _string_or_empty(extra_info.get("uuid"))
            or _string_or_empty(extra_info.get("problem_id"))
            or _string_or_empty(kwargs.get("uid"))
            or uuid.uuid4().hex
        )
        data_source = _string_or_empty(kwargs.get("data_source")) or _string_or_empty(extra_info.get("data_source"))
        is_valid = str(data_source).lower().startswith(("val", "valid", "test"))
        return _KernelGymContext(
            reference_code=reference_code,
            entry_point=entry_point or "Model",
            uuid=task_uuid,
            data_source=data_source,
            is_valid=is_valid,
        )

    def _prepare_workspace(self, task_prompt: str, kernelgym_context: _KernelGymContext) -> Path:
        workspace_dir = Path(tempfile.mkdtemp(prefix="claw-drkernel-", dir=self.workspace_root))
        (workspace_dir / "prompt.txt").write_text(
            "You are Claw Code running inside a throwaway workspace container.\n"
            "You may inspect files and use tools such as bash, read, write, edit, grep, and glob.\n"
            "The rollout harness will run a ReAct loop for you: after each answer it extracts your "
            "candidate code, evaluates it with KernelGYM, then sends the feedback back as the next "
            "prompt. You do not need to emit OpenAI tool calls.\n"
            "On every turn, write the complete candidate to /workspace/solution.py when possible and "
            "include the same complete Python code block in your answer. The code must define "
            "class ModelNew. Repair correctness first, then optimize speed.\n\n"
            f"Task:\n{task_prompt}\n",
            encoding="utf-8",
        )
        (workspace_dir / "TASK.md").write_text(f"# Task\n\n{task_prompt}\n", encoding="utf-8")
        (workspace_dir / "openai_compat_proxy.py").write_text(OPENAI_COMPAT_PROXY_SCRIPT, encoding="utf-8")
        (workspace_dir / "claw_react_loop.py").write_text(_support_script_text("claw_react_loop.py"), encoding="utf-8")
        (workspace_dir / "kernelgym_evaluate.py").write_text(_support_script_text("kernelgym_evaluate.py"), encoding="utf-8")
        (workspace_dir / "claw_react_loop.py").chmod(0o755)
        (workspace_dir / "kernelgym_evaluate.py").chmod(0o755)
        (workspace_dir / "kernelgym_context.json").write_text(
            json.dumps(
                {
                    "reference_code": kernelgym_context.reference_code,
                    "entry_point": kernelgym_context.entry_point,
                    "uuid": kernelgym_context.uuid,
                    "data_source": kernelgym_context.data_source,
                    "is_valid": kernelgym_context.is_valid,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return workspace_dir

    def _tokenize_turn_delta(self, turns_so_far: list[tuple[str, str]], new_turn: tuple[str, str]) -> list[int]:
        """Tokenize only the tokens introduced by appending `new_turn` to `turns_so_far`."""
        prev_messages = [{"role": r, "content": c} for r, c in turns_so_far]
        next_messages = prev_messages + [{"role": new_turn[0], "content": new_turn[1]}]
        prev_ids = self.tokenizer.apply_chat_template(
            prev_messages, add_generation_prompt=False, tokenize=True, **self.apply_chat_template_kwargs
        ) if prev_messages else list(self.system_prompt_ids)
        next_ids = self.tokenizer.apply_chat_template(
            next_messages, add_generation_prompt=False, tokenize=True, **self.apply_chat_template_kwargs
        )
        # Append delta. Guard against pathological template changes by returning the full
        # next-turn suffix if the prefix isn't strictly contained.
        if len(next_ids) >= len(prev_ids) and next_ids[: len(prev_ids)] == prev_ids:
            return next_ids[len(prev_ids) :]
        return next_ids[len(self.system_prompt_ids) :]

    async def _run_claw(self, workspace_dir: Path, upstream_base_url: str, sample_index: int) -> _ClawArtifacts:
        proxy_port = 20000 + (uuid.uuid4().int % 20000)
        container_name = f"claw-drkernel-{sample_index}-{uuid.uuid4().hex[:8]}"
        env = {
            "OPENAI_UPSTREAM_BASE_URL": upstream_base_url,
            "OPENAI_API_KEY": self.openai_api_key,
            "OPENAI_PROXY_PORT": str(proxy_port),
            "OPENAI_PROXY_COMPLETION_BUDGET_TOKENS": str(self.completion_budget_tokens),
            "CLAW_MODEL": self.claw_model,
            "CLAW_REACT_MAX_TURNS": str(self.react_max_turns),
            "CLAW_REACT_STOP_ON_OK": str(self.react_stop_on_ok),
            "KERNELGYM_SERVER_URL": self.kernelgym_server_url,
            "KERNELGYM_CONTEXT_PATH": "/workspace/kernelgym_context.json",
            "KERNELGYM_WORKSPACE_DIR": "/workspace",
            "KERNELGYM_EVAL_COUNTER_PATH": "/workspace/kernelgym_eval_count.txt",
            "KERNELGYM_MAX_EVALS_PER_SAMPLE": str(self.react_max_turns),
            "KERNELGYM_TASK_TIMEOUT": str(self.kernelgym_task_timeout),
            "KERNELGYM_TASK_TIMEOUT_CLIENT": str(self.kernelgym_task_timeout_client),
            "KERNELGYM_NUM_CORRECT_TRIALS": str(self.kernelgym_num_correct_trials),
            "KERNELGYM_NUM_PERF_TRIALS": str(self.kernelgym_num_perf_trials),
            "KERNELGYM_REFERENCE_BACKEND": self.kernelgym_reference_backend,
            "KERNELGYM_SPEEDUP_REWARD_UPPER_BOUND": str(self.kernelgym_speedup_upper),
            "KERNELGYM_SPEEDUP_REWARD_LOWER_BOUND": str(self.kernelgym_speedup_lower),
            "KERNELGYM_COVERAGE_REWARD_ENABLE": str(self.kernelgym_coverage_enable),
            "KERNELGYM_COVERAGE_REWARD_WEIGHT": str(self.kernelgym_coverage_weight),
            "KERNELGYM_COVERAGE_REWARD_TYPE": str(self.kernelgym_coverage_type),
            "KERNELGYM_INIT_CORRECT_WEIGHT": str(self.kernelgym_init_correct_weight),
            "KERNELGYM_INIT_PERFORMANCE_WEIGHT": str(self.kernelgym_init_performance_weight),
            "KERNELGYM_SPEEDUP_EPS": str(self.kernelgym_speedup_eps),
            "KERNELGYM_REWARD_PENALTY_SCORE": str(self.kernelgym_penalty_score),
            "HOME": "/workspace/home",
        }
        exit_code, failure = await asyncio.to_thread(
            _run_claw_container_blocking,
            image=self.image,
            name=container_name,
            workspace_dir=workspace_dir,
            env=env,
            network=self.container_network,
            timeout_sec=self.agent_timeout_sec,
        )
        result_json = None
        result_path = workspace_dir / "claw_result.json"
        if result_path.exists():
            raw = result_path.read_text(encoding="utf-8").strip()
            if raw:
                try:
                    result_json = json.loads(raw)
                except Exception:
                    pass
        react_summary = None
        summary_path = workspace_dir / "react_summary.json"
        if summary_path.exists():
            raw = summary_path.read_text(encoding="utf-8").strip()
            if raw:
                try:
                    react_summary = json.loads(raw)
                except Exception:
                    pass
        stderr_text = ""
        stderr_path = workspace_dir / "claw_stderr.log"
        if stderr_path.exists():
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        session_messages = _load_session_messages(workspace_dir)
        failure_reason = None
        if exit_code != 0:
            failure_reason = failure or (stderr_text.strip() or f"claw exited with status {exit_code}")
        return _ClawArtifacts(
            exit_code=int(exit_code),
            workspace_dir=workspace_dir,
            result_json=result_json,
            react_summary=react_summary,
            session_messages=session_messages,
            stderr_text=stderr_text,
            failure_reason=failure_reason,
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        metrics: dict[str, float] = {}
        messages = list(kwargs["raw_prompt"])
        _dbg = lambda msg: print(f"[ClawContainer] {msg}", flush=True)

        # 1) Build prompt_ids using the same chat template as ToolAgentLoop / SingleTurnAgentLoop.
        _dbg("step 1: apply_chat_template for prompt_ids")
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        )
        _dbg(f"step 1 done: prompt_ids len={len(prompt_ids)}")

        # 2) Resolve an OpenAI-compatible upstream and build the task prompt.
        _dbg("step 2: resolve upstream")
        upstream_url = await self._resolve_upstream()
        task_prompt = self._build_task_prompt_text(messages)
        kernelgym_context = self._build_kernelgym_context(kwargs)
        workspace_dir = await self.loop.run_in_executor(None, self._prepare_workspace, task_prompt, kernelgym_context)
        _dbg(
            "step 2 done: upstream={} workspace={} entry_point={} has_ref={}".format(
                upstream_url, workspace_dir, kernelgym_context.entry_point, bool(kernelgym_context.reference_code)
            )
        )

        sample_index = int(kwargs.get("index", 0)) if kwargs.get("index") is not None else 0

        # 3) Run the claw container under a global concurrency cap.
        _dbg(f"step 3: run claw container sample_index={sample_index}")
        async with type(self)._semaphore:
            with simple_timer("claw_container_run", metrics):
                artifacts = await self._run_claw(workspace_dir, upstream_url, sample_index)
        _dbg(f"step 3 done: exit_code={artifacts.exit_code} sessions={len(artifacts.session_messages)}")

        # 4) Extract a final-answer fallback string.
        final_message = ""
        if artifacts.result_json and isinstance(artifacts.result_json.get("message"), str):
            final_message = artifacts.result_json["message"].strip()
        if not final_message:
            for message in reversed(artifacts.session_messages):
                if message.get("role") == "assistant":
                    candidate = _message_plain_text(message)
                    if candidate:
                        final_message = candidate
                        break
        if not final_message:
            final_message = artifacts.failure_reason or "Agent did not return a final answer."

        # 5) Render claw turns into (role, content) list; re-tokenize via chat template.
        _dbg("step 5: render claw turns")
        turns = _render_claw_turns(artifacts.session_messages, final_message)
        _dbg(f"step 5: {len(turns)} turns to tokenize")
        response_ids: list[int] = []
        response_mask: list[int] = []
        turns_so_far: list[tuple[str, str]] = []
        for turn in turns:
            delta_ids = await self.loop.run_in_executor(
                None, lambda t=turn, s=list(turns_so_far): self._tokenize_turn_delta(s, t)
            )
            if not delta_ids:
                turns_so_far.append(turn)
                continue
            remaining = self.response_length - len(response_ids)
            if remaining <= 0:
                break
            take = delta_ids[:remaining]
            response_ids.extend(take)
            mask_bit = 1 if turn[0] == "assistant" else 0
            response_mask.extend([mask_bit] * len(take))
            turns_so_far.append(turn)
            if len(response_ids) >= self.response_length:
                break

        _dbg(f"step 5 done: response_ids={len(response_ids)} response_mask={len(response_mask)}")

        # The upstream _performance_metrics expects generate_sequences and tool_calls
        # timing keys (used by ToolAgentLoop). Provide them so it doesn't KeyError.
        metrics.setdefault("generate_sequences", metrics.get("claw_container_run", 0.0))
        metrics.setdefault("tool_calls", 0.0)

        if not response_ids:
            # Degenerate case: nothing rolled out. Emit a dummy pad-0 response so the trainer
            # stays healthy; reward manager will score it as 0.
            response_ids = [self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0]
            response_mask = [0]

        reward_score = 0.0
        reward_extra_info: dict[str, Any] = {
            "correctness": False,
            "performance": 0.0,
            "is_speedup_positive": False,
            "is_decoy_kernel": False,
            "compilation": False,
            "success": False,
            "status": "missing_react_summary",
            "error": artifacts.failure_reason or "",
            "num_custom_kernel": 0.0,
            "num_total_kernels": 0.0,
            "num_coverage": 0.0,
            "time_coverage": 0.0,
            "react_turn": 0,
            "react_reward": 0.0,
        }
        if isinstance(artifacts.react_summary, dict):
            raw_reward = artifacts.react_summary.get("reward_score")
            if raw_reward is not None:
                try:
                    reward_score = float(raw_reward)
                except Exception:
                    reward_score = 0.0
            raw_extra = artifacts.react_summary.get("reward_extra_info")
            if isinstance(raw_extra, dict):
                reward_extra_info = raw_extra

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=None,
            reward_score=reward_score,
            multi_modal_data={},
            num_turns=len(turns) + 1,
            metrics=metrics,
            extra_fields={
                "claw_workspace": str(artifacts.workspace_dir),
                "claw_exit_code": artifacts.exit_code,
                "claw_failure": artifacts.failure_reason or "",
                "claw_final_message": final_message,
                "reward_extra_info": reward_extra_info,
            },
        )
