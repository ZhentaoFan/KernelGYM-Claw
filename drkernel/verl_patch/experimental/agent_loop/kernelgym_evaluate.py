#!/usr/bin/env python3
"""Evaluate one candidate kernel against KernelGYM from a Claw rollout workspace.

This script intentionally uses only the Python standard library because it runs
inside the lightweight Claw container. The host agent loop writes
kernelgym_context.json with hidden reference code before launching the container.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def _read_json(path: str | Path, default):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _as_int(value, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _clamp(value, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except Exception:
        return low


def _clip_text(value, limit: int = 1600) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>..."


def _workspace_root() -> Path:
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


def _next_eval_index(max_evals: int):
    counter_path = Path(os.environ.get("KERNELGYM_EVAL_COUNTER_PATH", "/workspace/kernelgym_eval_count.txt"))
    try:
        current = int(counter_path.read_text(encoding="utf-8").strip() or "0")
    except Exception:
        current = 0
    if max_evals > 0 and current >= max_evals:
        return None, current
    counter_path.write_text(str(current + 1), encoding="utf-8")
    return current + 1, current + 1


def _post_json(url: str, payload: dict, timeout_sec: int):
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


def main() -> None:
    try:
        tool_input = json.loads(sys.stdin.read() or "{}")
        if not isinstance(tool_input, dict):
            tool_input = {}
    except Exception as exc:
        print(json.dumps({"ok": False, "status": "invalid_input", "error": f"invalid JSON input: {exc}"}))
        return

    max_evals = _as_int(os.environ.get("KERNELGYM_MAX_EVALS_PER_SAMPLE"), 3)
    eval_index, _used_count = _next_eval_index(max_evals)
    if eval_index is None:
        print(json.dumps({
            "ok": False,
            "status": "max_evals_exceeded",
            "max_evals_per_sample": max_evals,
            "guidance": "Stop evaluating and produce the best final kernel from prior feedback.",
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
                "guidance": "Write complete candidate code to /workspace/solution.py before evaluation.",
            }))
            return

    reference_code = str(context.get("reference_code") or "").strip()
    entry_point = str(context.get("entry_point") or "Model").strip() or "Model"
    task_uuid = str(context.get("uuid") or uuid.uuid4().hex)

    if not kernel_code:
        print(json.dumps({
            "ok": False,
            "status": "invalid_input",
            "error": "kernel_code is required and must contain complete candidate Python/Triton code.",
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
    task_timeout = _clamp(_as_int(tool_input.get("timeout"), default_task_timeout), 10, default_task_timeout)
    client_timeout = _as_int(os.environ.get("KERNELGYM_TASK_TIMEOUT_CLIENT"), max(task_timeout + 60, task_timeout))
    default_correct_trials = _clamp(_as_int(os.environ.get("KERNELGYM_NUM_CORRECT_TRIALS"), 5), 1, 20)
    num_correct_trials = _clamp(_as_int(tool_input.get("num_correct_trials"), default_correct_trials), 1, default_correct_trials)
    default_perf_trials = _clamp(_as_int(os.environ.get("KERNELGYM_NUM_PERF_TRIALS"), 20), 1, 1000)
    num_perf_trials = _clamp(_as_int(tool_input.get("num_perf_trials"), default_perf_trials), 1, default_perf_trials)
    force_refresh = _as_bool(tool_input.get("force_refresh"), True)
    reference_backend = os.environ.get("KERNELGYM_REFERENCE_BACKEND") or None
    enable_profiling = _as_bool(os.environ.get("KERNELGYM_ENABLE_PROFILING"), True)
    enable_triton_detection = _as_bool(os.environ.get("KERNELGYM_ENABLE_TRITON_DETECTION"), True)

    task_id = str(tool_input.get("task_id") or f"claw_react_{task_uuid}_{eval_index}_{uuid.uuid4().hex[:8]}")
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
