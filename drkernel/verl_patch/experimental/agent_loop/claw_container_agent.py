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
import re
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
import re
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
FORCE_FIRST_TOOL_NAME = os.environ.get("OPENAI_PROXY_FORCE_FIRST_TOOL_NAME", "").strip()
FIRST_REQUEST_MAX_TOKENS = int(os.environ.get("OPENAI_PROXY_FIRST_REQUEST_MAX_TOKENS", "0"))
FOLLOWUP_REQUEST_MAX_TOKENS = int(os.environ.get("OPENAI_PROXY_FOLLOWUP_REQUEST_MAX_TOKENS", "0"))
_remaining_completion_budget = COMPLETION_BUDGET_TOKENS if COMPLETION_BUDGET_TOKENS > 0 else None
_budget_lock = threading.Lock()
_chat_request_count = 0


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
    global _chat_request_count
    if _remaining_completion_budget is None or not request_body:
        return request_body, None, False
    try:
        payload = json.loads(request_body.decode("utf-8"))
    except Exception:
        return request_body, None, False
    if not isinstance(payload, dict):
        return request_body, None, False

    tools = payload.get("tools")
    tools_count = len(tools) if isinstance(tools, list) else 0
    original_tool_choice = payload.get("tool_choice")
    if FORCE_FIRST_TOOL_NAME and _chat_request_count == 0 and tools_count > 0:
        payload["tool_choice"] = {
            "type": "function",
            "function": {"name": FORCE_FIRST_TOOL_NAME},
        }

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
    request_cap = FIRST_REQUEST_MAX_TOKENS if _chat_request_count == 0 else FOLLOWUP_REQUEST_MAX_TOKENS
    if request_cap > 0:
        effective = min(effective, request_cap)
    payload["max_tokens"] = effective
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = effective
    _chat_request_count += 1
    return json.dumps(payload).encode("utf-8"), {
        "remaining": remaining,
        "requested": requested_max_tokens,
        "capped_to": effective,
        "tools_count": tools_count,
        "tool_choice": payload.get("tool_choice"),
        "original_tool_choice": original_tool_choice,
    }, False


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


def _extract_context_cap_from_error(err_body):
    try:
        text = err_body.decode("utf-8", errors="ignore")
    except Exception:
        return None
    match = re.search(
        r"maximum context length is\s+(\d+)\s+tokens and your request has\s+(\d+)\s+input tokens",
        text,
    )
    if not match:
        return None
    try:
        max_context = int(match.group(1))
        input_tokens = int(match.group(2))
    except Exception:
        return None
    return max(0, max_context - input_tokens)


def _cap_request_max_tokens(request_body, cap):
    if cap <= 0:
        return None
    try:
        payload = json.loads(request_body.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    requested = payload.get("max_tokens")
    if requested is None:
        requested = payload.get("max_completion_tokens")
    try:
        requested_int = int(requested) if requested is not None else None
    except Exception:
        requested_int = None
    if requested_int is not None and requested_int <= cap:
        return None
    payload["max_tokens"] = cap
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = cap
    return json.dumps(payload).encode("utf-8")


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

        def forward_once(request_body):
            local_headers = dict(headers)
            if request_body and "Content-Length" in local_headers:
                local_headers["Content-Length"] = str(len(request_body))
            request = urllib.request.Request(
                url=url,
                data=request_body if request_body else None,
                method=method,
                headers=local_headers,
            )
            return urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC)

        try:
            with forward_once(body) as response:
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
            is_chat = path.rstrip("/").endswith("/chat/completions")
            context_cap = _extract_context_cap_from_error(err_body) if is_chat and e.code == 400 else None
            retry_body = _cap_request_max_tokens(body, context_cap) if context_cap is not None else None
            if retry_body is not None:
                log_line(f"retrying chat_completions with context_cap={context_cap}")
                try:
                    with forward_once(retry_body) as response:
                        raw = response.read()
                        status = response.status
                        content_type = response.headers.get("Content-Type", "application/json")
                        consumed = extract_completion_tokens(raw)
                        if consumed is not None:
                            consume_completion_budget(consumed)
                            log_line(f"consumed={consumed} remaining={remaining_completion_budget()}")
                        self.send_response(status)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Content-Length", str(len(raw)))
                        self.end_headers()
                        self.wfile.write(raw)
                        return
                except urllib.error.HTTPError as retry_e:
                    err_body = retry_e.read() if hasattr(retry_e, "read") else b""
                    e = retry_e
                except Exception as retry_exc:
                    err = {"error": {"type": "proxy_upstream_error", "message": str(retry_exc)}}
                    payload = json.dumps(err).encode()
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
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


KERNELGYM_EVALUATE_TOOL_SCRIPT = r"""#!/usr/bin/env python3
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def _read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _as_int(value, default):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _clamp(value, low, high):
    try:
        return max(low, min(high, int(value)))
    except Exception:
        return low


def _clip_text(value, limit=1600):
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>..."


def _workspace_root():
    return Path(os.environ.get("KERNELGYM_WORKSPACE_DIR", "/workspace")).resolve()


def _read_candidate_from_path(raw_path):
    path_text = str(raw_path or "").strip()
    if not path_text:
        return "", "", None
    try:
        workspace = _workspace_root()
        candidate = Path(path_text)
        if not candidate.is_absolute():
            candidate = workspace / candidate
        candidate = candidate.resolve()
        if workspace != candidate and workspace not in candidate.parents:
            return "", path_text, "kernel_code_path must stay under /workspace"
        if not candidate.is_file():
            return "", path_text, f"kernel_code_path does not exist or is not a file: {path_text}"
        return candidate.read_text(encoding="utf-8", errors="replace"), str(candidate.relative_to(workspace)), None
    except Exception as exc:
        return "", path_text, str(exc)


def _next_eval_index(max_evals):
    counter_path = Path(os.environ.get("KERNELGYM_EVAL_COUNTER_PATH", "/workspace/kernelgym_eval_count.txt"))
    try:
        current = int(counter_path.read_text(encoding="utf-8").strip() or "0")
    except Exception:
        current = 0
    if max_evals > 0 and current >= max_evals:
        return None, current
    counter_path.write_text(str(current + 1), encoding="utf-8")
    return current + 1, current + 1


def _post_json(url, payload, timeout_sec):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        raw = response.read()
        if not raw:
            return {"status": "failed", "error_message": "empty response from KernelGYM"}
        return json.loads(raw.decode("utf-8"))


def main():
    try:
        tool_input = json.loads(sys.stdin.read() or "{}")
        if not isinstance(tool_input, dict):
            tool_input = {}
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"invalid JSON tool input: {exc}"}))
        return

    max_evals = _as_int(os.environ.get("KERNELGYM_MAX_EVALS_PER_SAMPLE"), 3)
    eval_index, used_count = _next_eval_index(max_evals)
    if eval_index is None:
        print(json.dumps({
            "ok": False,
            "status": "max_evals_exceeded",
            "max_evals_per_sample": max_evals,
            "guidance": "Stop calling evaluate_kernel and produce the best final kernel from prior feedback.",
        }))
        return

    context_path = os.environ.get("KERNELGYM_CONTEXT_PATH", "/workspace/kernelgym_context.json")
    context = _read_json(context_path, {})
    kernel_code = str(tool_input.get("kernel_code") or tool_input.get("code") or "").strip()
    kernel_code_path = ""
    if not kernel_code:
        kernel_code, kernel_code_path, path_error = _read_candidate_from_path(tool_input.get("kernel_code_path"))
        kernel_code = kernel_code.strip()
        if path_error:
            print(json.dumps({
                "ok": False,
                "status": "invalid_input",
                "error": path_error,
                "guidance": "Write the complete candidate to /workspace/solution.py, then call evaluate_kernel with kernel_code_path=\"solution.py\".",
            }))
            return
    reference_code = str(context.get("reference_code") or "").strip()
    context_entry_point = str(context.get("entry_point") or "Model").strip() or "Model"
    # The evaluator's entry_point names the hidden reference class (usually Model).
    # Models often guess "ModelNew"; ignore that and prefer the dataset context.
    entry_point = context_entry_point
    task_uuid = str(context.get("uuid") or uuid.uuid4().hex)

    if not kernel_code:
        print(json.dumps({
            "ok": False,
            "status": "invalid_input",
            "error": "kernel_code is required and must contain the complete candidate Python/Triton solution.",
        }))
        return
    if not reference_code:
        print(json.dumps({
            "ok": False,
            "status": "missing_reference",
            "error": "hidden reference_code was not available in kernelgym_context.json",
        }))
        return

    evaluated_code_path = ""
    try:
        workspace = _workspace_root()
        eval_dir = workspace / ".kernelgym_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        evaluated = eval_dir / f"eval_{eval_index}.py"
        evaluated.write_text(kernel_code, encoding="utf-8")
        evaluated_code_path = str(evaluated.relative_to(workspace))
    except Exception:
        evaluated_code_path = kernel_code_path

    server_url = os.environ.get("KERNELGYM_SERVER_URL", "http://127.0.0.1:10907").rstrip("/")
    default_task_timeout = _clamp(_as_int(os.environ.get("KERNELGYM_TASK_TIMEOUT"), 300), 10, 3600)
    task_timeout = _clamp(
        _as_int(tool_input.get("timeout"), default_task_timeout),
        10,
        default_task_timeout,
    )
    client_timeout = _as_int(os.environ.get("KERNELGYM_TASK_TIMEOUT_CLIENT"), max(task_timeout + 60, task_timeout))
    default_correct_trials = _clamp(_as_int(os.environ.get("KERNELGYM_NUM_CORRECT_TRIALS"), 5), 1, 20)
    num_correct_trials = _as_int(
        tool_input.get("num_correct_trials"),
        default_correct_trials,
    )
    num_correct_trials = _clamp(num_correct_trials, 1, default_correct_trials)
    default_perf_trials = _clamp(_as_int(os.environ.get("KERNELGYM_NUM_PERF_TRIALS"), 20), 1, 1000)
    num_perf_trials = _as_int(
        tool_input.get("num_perf_trials"),
        default_perf_trials,
    )
    num_perf_trials = _clamp(num_perf_trials, 1, default_perf_trials)
    force_refresh = _as_bool(tool_input.get("force_refresh"), True)
    reference_backend = os.environ.get("KERNELGYM_REFERENCE_BACKEND") or None
    enable_profiling = _as_bool(os.environ.get("KERNELGYM_ENABLE_PROFILING"), True)
    enable_triton_detection = _as_bool(os.environ.get("KERNELGYM_ENABLE_TRITON_DETECTION"), True)

    task_id = str(tool_input.get("task_id") or f"claw_eval_{task_uuid}_{eval_index}_{uuid.uuid4().hex[:8]}")
    payload = {
        "task_id": task_id[:100],
        "reference_code": reference_code,
        "kernel_code": kernel_code,
        "toolkit": "kernelbench",
        "backend_adapter": "kernelbench",
        "backend": "triton",
        "entry_point": entry_point,
        "workflow": "kernelbench",
        "num_correct_trials": num_correct_trials,
        "num_perf_trials": num_perf_trials,
        "timeout": task_timeout,
        "priority": "normal",
        "uuid": task_uuid,
        "use_reference_cache": False,
        "is_valid": _as_bool(context.get("is_valid"), False),
        "force_refresh": force_refresh,
        "verbose_errors": True,
        "enable_profiling": enable_profiling,
        "enable_triton_detection": enable_triton_detection,
        "reference_backend": reference_backend,
    }

    started = time.time()
    try:
        result = _post_json(f"{server_url}/evaluate", payload, client_timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        print(json.dumps({
            "ok": False,
            "status": "http_error",
            "task_id": task_id,
            "http_status": exc.code,
            "error": _clip_text(body or str(exc)),
            "eval_index": eval_index,
            "max_evals_per_sample": max_evals,
        }))
        return
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "status": "request_failed",
            "task_id": task_id,
            "error": _clip_text(str(exc)),
            "eval_index": eval_index,
            "max_evals_per_sample": max_evals,
        }))
        return

    status = result.get("status")
    compiled = bool(result.get("compiled", False))
    correctness = bool(result.get("correctness", False))
    decoy_kernel = bool(result.get("decoy_kernel", False))
    speedup = result.get("speedup", 0.0)
    error_message = result.get("error_message") or result.get("error") or ""

    if decoy_kernel:
        guidance = "KernelGYM detected a decoy kernel. Use a real Triton/CUDA implementation and avoid delegating to torch ops."
    elif not compiled:
        guidance = "Compilation failed. Fix syntax, imports, class name, and Triton launch/signature issues before optimizing."
    elif not correctness:
        guidance = "The candidate compiled but failed correctness. Match Model.forward semantics exactly before chasing speed."
    else:
        guidance = "The candidate is correct. Improve performance only if you can preserve correctness."

    summary = {
        "ok": status == "completed" and compiled and correctness and not decoy_kernel,
        "status": status,
        "task_id": result.get("task_id", task_id),
        "eval_index": eval_index,
        "max_evals_per_sample": max_evals,
        "compiled": compiled,
        "correctness": correctness,
        "decoy_kernel": decoy_kernel,
        "speedup": speedup,
        "reference_runtime": result.get("reference_runtime"),
        "kernel_runtime": result.get("kernel_runtime"),
        "processing_time_sec": round(time.time() - started, 3),
        "error": _clip_text(error_message),
        "guidance": guidance,
        "kernel_code_chars": len(kernel_code),
    }
    if evaluated_code_path:
        summary["evaluated_code_path"] = evaluated_code_path
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        summary["time_coverage"] = metadata.get("time_coverage")
        summary["num_coverage"] = metadata.get("num_coverage")
        summary["gpu_name"] = metadata.get("gpu_name")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
"""


KERNELGYM_PLUGIN_MANIFEST = {
    "name": "kernelgym-evaluator",
    "version": "0.1.0",
    "description": "Evaluate candidate DR.Kernel solutions against KernelGYM from inside a Claw rollout container.",
    "tools": [
        {
            "name": "evaluate_kernel",
            "description": (
                "Evaluate a complete candidate kernel solution with KernelGYM. "
                "Pass the full Python code for class ModelNew in kernel_code. "
                "The reference implementation is supplied out-of-band by the training harness."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kernel_code": {
                        "type": "string",
                        "description": "Complete candidate Python/Triton solution containing class ModelNew. Prefer kernel_code_path for long code.",
                    },
                    "kernel_code_path": {
                        "type": "string",
                        "description": (
                            "Path under /workspace to a Python file containing the complete candidate solution. "
                            "Preferred workflow: write solution.py, then pass kernel_code_path=\"solution.py\"."
                        ),
                    },
                    "entry_point": {
                        "type": "string",
                        "description": "Reference class name, usually Model. Omit unless the task says otherwise.",
                    },
                    "num_correct_trials": {
                        "type": "integer",
                        "description": "Optional override for correctness trials.",
                    },
                    "num_perf_trials": {
                        "type": "integer",
                        "description": "Optional override for performance trials.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional server-side timeout in seconds.",
                    },
                    "force_refresh": {
                        "type": "boolean",
                        "description": "Whether to bypass KernelGYM cached task results.",
                    },
                },
                "anyOf": [
                    {"required": ["kernel_code"]},
                    {"required": ["kernel_code_path"]},
                ],
                "additionalProperties": False,
            },
            "command": "./tools/evaluate_kernel.py",
            "requiredPermission": "danger-full-access",
        }
    ],
}


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
        "PROMPT=$(cat /workspace/prompt.txt)\n"
        "set +e\n"
        "claw \\\n"
        "  --output-format json \\\n"
        "  --permission-mode danger-full-access \\\n"
        "  --dangerously-skip-permissions \\\n"
        "  --model \"$CLAW_MODEL\" \\\n"
        "  prompt \"$PROMPT\" \\\n"
        "  > /workspace/claw_result.json \\\n"
        "  2> /workspace/claw_stderr.log\n"
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


def _load_session_messages(workspace_dir: Path) -> list[dict]:
    session_file = _latest_session_file(workspace_dir)
    if session_file is None:
        return []
    messages: list[dict] = []
    with session_file.open("r", encoding="utf-8") as handle:
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


def _block_text(block: dict) -> str:
    if block.get("type") == "text":
        return block.get("text", "") or ""
    return ""


def _message_plain_text(message: dict) -> str:
    parts = []
    for block in message.get("blocks", []) or []:
        parts.append(_block_text(block))
    return "\n".join(p for p in parts if p).strip()


def _read_workspace_text(workspace_dir: Optional[Path], raw_path: Any) -> str:
    if workspace_dir is None:
        return ""
    path_text = str(raw_path or "").strip()
    if not path_text:
        return ""
    try:
        workspace = workspace_dir.resolve()
        candidate = Path(path_text)
        if not candidate.is_absolute():
            candidate = workspace / candidate
        candidate = candidate.resolve()
        if workspace != candidate and workspace not in candidate.parents:
            return ""
        if not candidate.is_file():
            return ""
        return candidate.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


def _candidate_code_from_tool_input(tool_input: dict[str, Any], workspace_dir: Optional[Path]) -> str:
    code = str(tool_input.get("kernel_code") or tool_input.get("code") or "").strip()
    if code and "ModelNew" in code:
        return code
    code = _read_workspace_text(workspace_dir, tool_input.get("kernel_code_path"))
    if code and "ModelNew" in code:
        return code
    return ""


def _has_evaluate_kernel_call(session_messages: list[dict], workspace_dir: Optional[Path] = None) -> bool:
    return bool(_extract_last_evaluate_kernel_code(session_messages, workspace_dir))


def _extract_last_evaluate_kernel_code(session_messages: list[dict], workspace_dir: Optional[Path] = None) -> str:
    """Return the latest candidate code passed to evaluate_kernel, if it is parseable.

    Claw's final prose sometimes summarizes the tool feedback without repeating the
    code. DR.Kernel's reward path extracts the last markdown code block from the
    rendered response, so we append this candidate as a code block as a safety net.
    """
    for message in reversed(session_messages):
        if message.get("role") != "assistant":
            continue
        for block in reversed(message.get("blocks", []) or []):
            if block.get("type") != "tool_use" or block.get("name") != "evaluate_kernel":
                continue
            tool_input = block.get("input")
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except Exception:
                    continue
            if not isinstance(tool_input, dict):
                continue
            code = _candidate_code_from_tool_input(tool_input, workspace_dir)
            if code:
                return code
    return ""


def _extract_last_modelnew_code_block(text: str) -> str:
    if not text:
        return ""
    code_blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in reversed(code_blocks):
        code = block.strip()
        if "class ModelNew" in code:
            return code
    return ""


def _append_session_messages(workspace_dir: Path, messages: list[dict[str, Any]]) -> None:
    session_file = _latest_session_file(workspace_dir)
    if session_file is None:
        session_file = workspace_dir / ".claw" / "sessions" / "posthoc" / f"session-{int(time.time() * 1000)}-0.jsonl"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        session_file.write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "session_id": session_file.stem,
                    "created_at_ms": int(time.time() * 1000),
                    "updated_at_ms": int(time.time() * 1000),
                    "version": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    with session_file.open("a", encoding="utf-8") as handle:
        for message in messages:
            handle.write(json.dumps({"type": "message", "message": message}, ensure_ascii=False) + "\n")


def _parse_tool_input(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _trim(s: str, limit: int = 4096) -> str:
    if s is None:
        return ""
    if len(s) <= limit:
        return s
    return s[: limit].rstrip() + "\n...<truncated>"


def _render_claw_turns(
    session_messages: list[dict],
    fallback_text: str,
    workspace_dir: Optional[Path] = None,
) -> list[tuple[str, str]]:
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
                    name = block.get("name")
                    tool_input = _parse_tool_input(block.get("input"))
                    kernel_code = ""
                    if name == "evaluate_kernel":
                        kernel_code = _candidate_code_from_tool_input(tool_input, workspace_dir)
                        if kernel_code:
                            tool_input = dict(tool_input)
                            tool_input.pop("kernel_code", None)
                            tool_input.pop("code", None)
                            tool_input["kernel_code_chars"] = len(kernel_code)
                    payload = {
                        "id": block.get("id"),
                        "name": name,
                        "input": tool_input if tool_input else block.get("input"),
                    }
                    rendered = "<tool_call>" + json.dumps(payload, ensure_ascii=False, sort_keys=True) + "</tool_call>"
                    if kernel_code and "ModelNew" in kernel_code:
                        rendered += "\n<evaluated_kernel_code>\n```python\n" + kernel_code + "\n```\n</evaluated_kernel_code>"
                    parts.append(rendered)
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
        cls.workspace_root = Path(os.getenv("CLAW_WORKSPACE_ROOT", "/home/ubuntu/z84318463/claw_agent_rollouts/drkernel"))
        cls.workspace_root.mkdir(parents=True, exist_ok=True)
        cls.container_network = os.getenv("CLAW_CONTAINER_NETWORK", "host")
        cls.agent_timeout_sec = float(os.getenv("CLAW_AGENT_TIMEOUT_SEC", "900"))
        cls.completion_budget_tokens = int(os.getenv("CLAW_MAX_COMPLETION_TOKENS", str(cls.response_length)))
        cls.claw_model = os.getenv("CLAW_MODEL_NAME", cls.model_name)
        cls.openai_api_key = os.getenv("CLAW_OPENAI_API_KEY", "local-dev-token")
        cls.explicit_upstream = os.getenv("CLAW_UPSTREAM_BASE_URL") or None
        cls.kernelgym_server_url = os.getenv("KERNELGYM_SERVER_URL", "http://127.0.0.1:10907")
        cls.kernelgym_max_evals = int(os.getenv("CLAW_KERNELGYM_MAX_EVALS", "3"))
        cls.kernelgym_task_timeout = int(
            os.getenv("CLAW_KERNELGYM_TASK_TIMEOUT", os.getenv("REWARD_TASK_TIMEOUT", "300"))
        )
        cls.kernelgym_task_timeout_client = int(
            os.getenv("CLAW_KERNELGYM_TASK_TIMEOUT_CLIENT", os.getenv("REWARD_TASK_TIMEOUT_CLIENT", "2400"))
        )
        cls.kernelgym_num_correct_trials = int(os.getenv("CLAW_KERNELGYM_NUM_CORRECT_TRIALS", "5"))
        cls.kernelgym_num_perf_trials = int(
            os.getenv("CLAW_KERNELGYM_NUM_PERF_TRIALS", os.getenv("NUM_PERF_TRIALS", "20"))
        )
        cls.kernelgym_reference_backend = os.getenv("REFERENCE_BACKEND", os.getenv("KERNELGYM_REFERENCE_BACKEND", "pytorch"))

        logger.info(
            "ClawContainerAgentLoop config image=%s workspace=%s network=%s budget=%d model=%s upstream_override=%s kernelgym=%s max_evals=%d",
            cls.image, cls.workspace_root, cls.container_network,
            cls.completion_budget_tokens, cls.claw_model, cls.explicit_upstream,
            cls.kernelgym_server_url, cls.kernelgym_max_evals,
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

    def _write_kernelgym_plugin(self, workspace_dir: Path, context: _KernelGymContext) -> None:
        config_dir = workspace_dir / ".claw"
        plugin_root = workspace_dir / "claw_plugins" / "kernelgym-evaluator"
        tool_dir = plugin_root / "tools"
        manifest_dir = plugin_root / ".claude-plugin"
        config_dir.mkdir(parents=True, exist_ok=True)
        tool_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)

        settings = {
            "plugins": {
                "enabled": {"kernelgym-evaluator@external": True},
                "externalDirectories": ["./claw_plugins"],
            },
        }
        (config_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        (manifest_dir / "plugin.json").write_text(
            json.dumps(KERNELGYM_PLUGIN_MANIFEST, indent=2),
            encoding="utf-8",
        )
        tool_path = tool_dir / "evaluate_kernel.py"
        tool_path.write_text(KERNELGYM_EVALUATE_TOOL_SCRIPT, encoding="utf-8")
        tool_path.chmod(0o755)
        (workspace_dir / "kernelgym_context.json").write_text(
            json.dumps(
                {
                    "reference_code": context.reference_code,
                    "entry_point": context.entry_point,
                    "uuid": context.uuid,
                    "data_source": context.data_source,
                    "is_valid": context.is_valid,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _prepare_workspace(self, task_prompt: str, kernelgym_context: _KernelGymContext) -> Path:
        workspace_dir = Path(tempfile.mkdtemp(prefix="claw-drkernel-", dir=self.workspace_root))
        (workspace_dir / "prompt.txt").write_text(
            "You are Claw Code running inside a throwaway workspace container.\n"
            "You may inspect files and use tools such as bash, read, write, edit, grep, and glob.\n"
            "A tool named evaluate_kernel is available in this container. This tool submits a complete candidate "
            "kernel solution to KernelGYM and returns compile/correctness/speedup feedback. The evaluator has "
            "the hidden reference implementation out-of-band; do not ask for or print the reference code.\n"
            "Mandatory tool-use protocol: create a complete candidate file named solution.py, then call "
            "evaluate_kernel with kernel_code_path=\"solution.py\" before writing your final answer. Keep the "
            "analysis short before the first evaluate_kernel call. Never pass placeholders such as "
            "\"full code\", \"REPLACE_ME\", or partial imports as kernel_code; invalid tool calls waste your "
            "limited evaluations. If evaluate_kernel returns an error, edit solution.py using that feedback and "
            "call evaluate_kernel again before finalizing. A rollout that never calls evaluate_kernel is "
            "considered failed.\n"
            f"You may call evaluate_kernel at most {self.kernelgym_max_evals} times for this task. "
            "Each evaluated candidate must be complete Python code containing class ModelNew. "
            "Use feedback to repair correctness first, then optimize. Your final answer must contain the final "
            "complete solution code.\n\n"
            f"Task:\n{task_prompt}\n",
            encoding="utf-8",
        )
        (workspace_dir / "TASK.md").write_text(f"# Task\n\n{task_prompt}\n", encoding="utf-8")
        (workspace_dir / "openai_compat_proxy.py").write_text(OPENAI_COMPAT_PROXY_SCRIPT, encoding="utf-8")
        self._write_kernelgym_plugin(workspace_dir, kernelgym_context)
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
            "OPENAI_PROXY_FIRST_REQUEST_MAX_TOKENS": os.getenv("CLAW_FIRST_REQUEST_MAX_TOKENS", "2048"),
            "OPENAI_PROXY_FOLLOWUP_REQUEST_MAX_TOKENS": os.getenv("CLAW_FOLLOWUP_REQUEST_MAX_TOKENS", "3072"),
            "CLAW_MODEL": self.claw_model,
            "HOME": "/workspace/home",
            "CLAW_CONFIG_HOME": "/workspace/home/.claw",
            "KERNELGYM_SERVER_URL": self.kernelgym_server_url,
            "KERNELGYM_CONTEXT_PATH": "/workspace/kernelgym_context.json",
            "KERNELGYM_MAX_EVALS_PER_SAMPLE": str(self.kernelgym_max_evals),
            "KERNELGYM_TASK_TIMEOUT": str(self.kernelgym_task_timeout),
            "KERNELGYM_TASK_TIMEOUT_CLIENT": str(self.kernelgym_task_timeout_client),
            "KERNELGYM_NUM_CORRECT_TRIALS": str(self.kernelgym_num_correct_trials),
            "KERNELGYM_NUM_PERF_TRIALS": str(self.kernelgym_num_perf_trials),
            "KERNELGYM_REFERENCE_BACKEND": self.kernelgym_reference_backend,
            "OPENAI_PROXY_FORCE_FIRST_TOOL_NAME": os.getenv("CLAW_FORCE_FIRST_TOOL_NAME", ""),
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
            session_messages=session_messages,
            stderr_text=stderr_text,
            failure_reason=failure_reason,
        )

    def _fallback_evaluate_kernel(self, workspace_dir: Path, final_message: str) -> list[dict[str, Any]]:
        """Evaluate the final code block when the model ignored the registered tool.

        This keeps the training trajectory self-contained: the synthetic tool call
        is appended to Claw's session JSONL, so the trainer can learn the desired
        evaluate_kernel call pattern even before the cold-start model reliably
        invokes tools by itself.
        """
        code = _extract_last_modelnew_code_block(final_message)
        if not code:
            return []

        solution_path = workspace_dir / "solution.py"
        solution_path.write_text(code, encoding="utf-8")
        tool_path = workspace_dir / "claw_plugins" / "kernelgym-evaluator" / "tools" / "evaluate_kernel.py"
        if not tool_path.exists():
            return []

        env = os.environ.copy()
        env.update(
            {
                "KERNELGYM_SERVER_URL": self.kernelgym_server_url,
                "KERNELGYM_CONTEXT_PATH": str(workspace_dir / "kernelgym_context.json"),
                "KERNELGYM_WORKSPACE_DIR": str(workspace_dir),
                "KERNELGYM_EVAL_COUNTER_PATH": str(workspace_dir / "kernelgym_eval_count.txt"),
                "KERNELGYM_MAX_EVALS_PER_SAMPLE": str(self.kernelgym_max_evals),
                "KERNELGYM_TASK_TIMEOUT": str(self.kernelgym_task_timeout),
                "KERNELGYM_TASK_TIMEOUT_CLIENT": str(self.kernelgym_task_timeout_client),
                "KERNELGYM_NUM_CORRECT_TRIALS": str(self.kernelgym_num_correct_trials),
                "KERNELGYM_NUM_PERF_TRIALS": str(self.kernelgym_num_perf_trials),
                "KERNELGYM_REFERENCE_BACKEND": self.kernelgym_reference_backend,
            }
        )
        tool_input = {"kernel_code_path": "solution.py"}
        tool_call_id = f"posthoc-evaluate-{uuid.uuid4().hex[:8]}"
        try:
            proc = subprocess.run(
                [str(tool_path)],
                input=json.dumps(tool_input),
                text=True,
                capture_output=True,
                cwd=str(workspace_dir),
                env=env,
                timeout=max(30, self.kernelgym_task_timeout_client + 30),
            )
            output = (proc.stdout or "").strip()
            if not output:
                output = json.dumps(
                    {
                        "ok": False,
                        "status": "posthoc_evaluate_failed",
                        "returncode": proc.returncode,
                        "stderr": _trim(proc.stderr or "", 1600),
                    },
                    ensure_ascii=False,
                )
        except Exception as exc:
            output = json.dumps(
                {"ok": False, "status": "posthoc_evaluate_exception", "error": _trim(str(exc), 1600)},
                ensure_ascii=False,
            )

        fallback_messages = [
            {
                "role": "assistant",
                "blocks": [
                    {
                        "type": "tool_use",
                        "id": tool_call_id,
                        "name": "evaluate_kernel",
                        "input": json.dumps(tool_input),
                    }
                ],
            },
            {
                "role": "tool",
                "blocks": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "tool_name": "evaluate_kernel",
                        "is_error": False,
                        "output": output,
                    }
                ],
            },
        ]
        try:
            _append_session_messages(workspace_dir, fallback_messages)
        except Exception as exc:
            logger.warning("failed to append posthoc evaluate messages to session: %s", exc)
        (workspace_dir / "claw_posthoc_evaluate.log").write_text(output + "\n", encoding="utf-8")
        return fallback_messages

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
        workspace_dir = await self.loop.run_in_executor(
            None, self._prepare_workspace, task_prompt, kernelgym_context
        )
        _dbg(
            "step 2 done: upstream="
            f"{upstream_url} workspace={workspace_dir} "
            f"entry_point={kernelgym_context.entry_point} has_ref={bool(kernelgym_context.reference_code)}"
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
        if not _has_evaluate_kernel_call(artifacts.session_messages, artifacts.workspace_dir):
            fallback_messages = await self.loop.run_in_executor(
                None, self._fallback_evaluate_kernel, artifacts.workspace_dir, final_message
            )
            if fallback_messages:
                artifacts.session_messages.extend(fallback_messages)
        evaluated_kernel_code = _extract_last_evaluate_kernel_code(artifacts.session_messages, artifacts.workspace_dir)
        if evaluated_kernel_code:
            final_message = (
                final_message.rstrip()
                + "\n\nFinal candidate submitted to evaluate_kernel:\n"
                + "```python\n"
                + evaluated_kernel_code
                + "\n```"
            )

        # 5) Render claw turns into (role, content) list; re-tokenize via chat template.
        _dbg("step 5: render claw turns")
        turns = _render_claw_turns(artifacts.session_messages, final_message, artifacts.workspace_dir)
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

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=None,
            multi_modal_data={},
            num_turns=len(turns) + 1,
            metrics=metrics,
            extra_fields={
                "claw_workspace": str(artifacts.workspace_dir),
                "claw_exit_code": artifacts.exit_code,
                "claw_failure": artifacts.failure_reason or "",
                "claw_final_message": final_message,
            },
        )
