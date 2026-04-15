#!/usr/bin/env python3
"""Container-internal ReAct loop for Claw Code rollouts.

The cold-start DR.Kernel model is not expected to emit tool calls. This harness
therefore forces the interaction pattern:

    claw prompt -> extract solution.py / code block -> KernelGYM evaluate
    -> write feedback into the next claw prompt -> repeat

The script writes claw_react_transcript.jsonl in the same message shape consumed
by claw_container_agent.py, plus react_summary.json for reward/metrics.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


WORKSPACE = Path(os.environ.get("CLAW_WORKSPACE_DIR", "/workspace")).resolve()
TRANSCRIPT_PATH = WORKSPACE / "claw_react_transcript.jsonl"
SUMMARY_PATH = WORKSPACE / "react_summary.json"
LOG_PATH = WORKSPACE / "claw_react_loop.log"


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _as_float(value, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}"
    print(line, file=sys.stderr, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _clip(text: Any, limit: int = 4000) -> str:
    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "\n...<truncated>..."


def _append_message(role: str, text: str) -> None:
    message = {"role": role, "blocks": [{"type": "text", "text": text or ""}]}
    with TRANSCRIPT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "message", "message": message}, ensure_ascii=False) + "\n")


def _read_json(path: Path, default):
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return default
        return json.loads(raw)
    except Exception:
        return default


def _extract_result_message(path: Path) -> str:
    payload = _read_json(path, {})
    if not isinstance(payload, dict):
        return ""
    for key in ("message", "content", "response", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    result = payload.get("result")
    if isinstance(result, dict):
        for key in ("message", "content", "response", "text"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _latest_session_file() -> Path | None:
    sessions_root = WORKSPACE / ".claw" / "sessions"
    if not sessions_root.exists():
        return None
    files = list(sessions_root.rglob("*.jsonl"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _latest_assistant_text_from_session() -> str:
    session_file = _latest_session_file()
    if session_file is None:
        return ""
    latest = ""
    try:
        with session_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                message = record.get("message") if isinstance(record, dict) else None
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                parts = []
                for block in message.get("blocks", []) or []:
                    if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                        parts.append(block["text"])
                if parts:
                    latest = "\n".join(parts).strip()
    except Exception:
        return ""
    return latest


def _extract_code_from_text(text: str) -> str:
    if not text:
        return ""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in reversed(blocks):
        code = block.strip()
        if "class ModelNew" in code:
            return code
    marker = "class ModelNew"
    idx = text.rfind(marker)
    if idx >= 0:
        prefix = text.rfind("\nimport ", 0, idx)
        if prefix < 0:
            prefix = text.rfind("\nfrom ", 0, idx)
        start = max(prefix + 1, 0)
        return text[start:].strip()
    return ""


def _candidate_code_from_workspace_or_text(text: str) -> tuple[str, str]:
    solution_path = WORKSPACE / "solution.py"
    if solution_path.is_file():
        code = solution_path.read_text(encoding="utf-8", errors="replace").strip()
        if "class ModelNew" in code:
            return code, "solution.py"
    code = _extract_code_from_text(text)
    if code:
        solution_path.write_text(code + "\n", encoding="utf-8")
        return code, "solution.py"
    return "", ""


def _run_claw(prompt: str, turn: int) -> tuple[int, str]:
    result_path = WORKSPACE / f"claw_turn_{turn}.json"
    stderr_path = WORKSPACE / f"claw_turn_{turn}_stderr.log"
    cmd = [
        "claw",
        "--output-format",
        "json",
        "--permission-mode",
        "danger-full-access",
        "--dangerously-skip-permissions",
        "--model",
        os.environ["CLAW_MODEL"],
        "prompt",
        prompt,
    ]
    _log(f"turn={turn} running claw")
    with result_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True)
    text = _extract_result_message(result_path) or _latest_assistant_text_from_session()
    if proc.returncode != 0:
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
        text = text or f"claw exited with status {proc.returncode}\n{_clip(stderr, 2000)}"
    _log(f"turn={turn} claw_exit={proc.returncode} assistant_chars={len(text)}")
    return int(proc.returncode), text


def _evaluate_solution(turn: int, code_path: str) -> dict[str, Any]:
    payload = {
        "kernel_code_path": code_path,
        "task_id": f"{os.environ.get('KERNELGYM_TASK_UUID', 'claw')}_turn_{turn}_{int(time.time() * 1000)}",
        "force_refresh": True,
    }
    cmd = ["python3", str(WORKSPACE / "kernelgym_evaluate.py")]
    _log(f"turn={turn} evaluating path={code_path}")
    proc = subprocess.run(
        cmd,
        input=json.dumps(payload),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    raw = proc.stdout.strip()
    if proc.stderr.strip():
        (WORKSPACE / f"kernelgym_eval_turn_{turn}_stderr.log").write_text(proc.stderr, encoding="utf-8")
    try:
        result = json.loads(raw) if raw else {}
    except Exception:
        result = {"ok": False, "status": "invalid_evaluator_output", "error": _clip(raw or proc.stderr, 1600)}
    result["react_turn"] = turn
    result["evaluator_exit_code"] = int(proc.returncode)
    (WORKSPACE / f"kernelgym_eval_turn_{turn}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _log(
        "turn={turn} eval status={status} compiled={compiled} correctness={correctness} speedup={speedup}".format(
            turn=turn,
            status=result.get("status"),
            compiled=result.get("compiled"),
            correctness=result.get("correctness"),
            speedup=result.get("speedup"),
        )
    )
    return result


def _invalid_code_result(turn: int) -> dict[str, Any]:
    result = {
        "ok": False,
        "status": "no_candidate_code",
        "compiled": False,
        "correctness": False,
        "speedup": 0.0,
        "error": "No complete code containing class ModelNew was found in assistant text or /workspace/solution.py.",
        "guidance": "Return a complete Python code block and/or write /workspace/solution.py with class ModelNew.",
        "react_turn": turn,
    }
    (WORKSPACE / f"kernelgym_eval_turn_{turn}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _feedback_text(turn: int, result: dict[str, Any]) -> str:
    compact = {
        "turn": turn,
        "status": result.get("status"),
        "compiled": result.get("compiled"),
        "correctness": result.get("correctness"),
        "decoy_kernel": result.get("decoy_kernel"),
        "speedup": result.get("speedup"),
        "reference_runtime": result.get("reference_runtime"),
        "kernel_runtime": result.get("kernel_runtime"),
        "time_coverage": result.get("time_coverage"),
        "num_coverage": result.get("num_coverage"),
        "error": _clip(result.get("error") or result.get("error_message"), 1800),
        "guidance": result.get("guidance"),
    }
    return (
        f"KernelGYM feedback for turn {turn}:\n"
        f"```json\n{json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True)}\n```\n"
        "Use this feedback in the next turn. Repair correctness/compilation first, then optimize speed. "
        "Keep the final answer as a complete Python code block containing class ModelNew."
    )


def _compute_reward(result: dict[str, Any]) -> float:
    penalty_score = _as_float(os.environ.get("KERNELGYM_REWARD_PENALTY_SCORE"), 0.0)
    if result.get("status") != "completed" or result.get("decoy_kernel"):
        return penalty_score
    correctness = 1.0 if result.get("correctness") else 0.0
    speedup = _as_float(result.get("speedup"), 0.0)
    lower = _as_float(os.environ.get("KERNELGYM_SPEEDUP_REWARD_LOWER_BOUND"), 0.0)
    upper = _as_float(os.environ.get("KERNELGYM_SPEEDUP_REWARD_UPPER_BOUND"), 3.0)
    reward_speedup = min(speedup, upper)
    if reward_speedup < lower:
        reward_speedup = 0.0
    correct_weight = _as_float(os.environ.get("KERNELGYM_INIT_CORRECT_WEIGHT"), 0.5)
    perf_weight = _as_float(os.environ.get("KERNELGYM_INIT_PERFORMANCE_WEIGHT"), 0.5)
    reward = correct_weight * correctness + perf_weight * reward_speedup
    if correctness and _as_bool(os.environ.get("KERNELGYM_COVERAGE_REWARD_ENABLE"), False):
        coverage_key = os.environ.get("KERNELGYM_COVERAGE_REWARD_TYPE", "time_coverage")
        coverage = _as_float(result.get(coverage_key), 0.0)
        reward += _as_float(os.environ.get("KERNELGYM_COVERAGE_REWARD_WEIGHT"), 0.5) * coverage
    return float(reward)


def _reward_extra_info(result: dict[str, Any], reward_score: float) -> dict[str, Any]:
    return {
        "correctness": bool(result.get("correctness", False)),
        "performance": _as_float(result.get("speedup"), 0.0),
        "is_speedup_positive": _as_float(result.get("speedup"), 0.0) >= 1.0 + _as_float(os.environ.get("KERNELGYM_SPEEDUP_EPS"), 0.01),
        "is_decoy_kernel": bool(result.get("decoy_kernel", False)),
        "compilation": bool(result.get("compiled", False)),
        "success": bool(result.get("compiled", False) and result.get("correctness", False)),
        "status": result.get("status", "unknown"),
        "error": result.get("error") or result.get("error_message") or "",
        "num_custom_kernel": _as_float(result.get("num_custom_kernel"), 0.0),
        "num_total_kernels": _as_float(result.get("num_total_kernels"), 0.0),
        "num_coverage": _as_float(result.get("num_coverage"), 0.0),
        "time_coverage": _as_float(result.get("time_coverage"), 0.0),
        "react_turn": int(result.get("react_turn") or 0),
        "react_reward": reward_score,
    }


def _initial_turn_prompt(base_prompt: str) -> str:
    return (
        base_prompt.rstrip()
        + "\n\n"
        "Important rollout protocol:\n"
        "- You do not need to call an evaluation tool yourself.\n"
        "- Produce a complete candidate solution as a Python code block containing class ModelNew.\n"
        "- Also write the same complete code to /workspace/solution.py if you can.\n"
        "- After your response, the harness will automatically extract/evaluate the code with KernelGYM.\n"
        "- Keep explanations short; prioritize complete, runnable code.\n"
    )


def _next_turn_prompt(base_prompt: str, turn: int, feedback: str) -> str:
    current = ""
    solution = WORKSPACE / "solution.py"
    if solution.is_file():
        current = _clip(solution.read_text(encoding="utf-8", errors="replace"), 6000)
    return (
        f"You are continuing the same DR.Kernel task in turn {turn}.\n\n"
        f"Original task:\n{_clip(base_prompt, 5000)}\n\n"
        f"{feedback}\n\n"
        "Current /workspace/solution.py is:\n"
        f"```python\n{current}\n```\n\n"
        "Revise /workspace/solution.py and respond with the complete improved Python code block containing class ModelNew. "
        "Do not omit imports or helper kernels."
    )


def main() -> int:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if TRANSCRIPT_PATH.exists():
        TRANSCRIPT_PATH.unlink()

    base_prompt = (WORKSPACE / "prompt.txt").read_text(encoding="utf-8", errors="replace")
    max_turns = max(1, _as_int(os.environ.get("CLAW_REACT_MAX_TURNS"), 3))
    stop_on_ok = _as_bool(os.environ.get("CLAW_REACT_STOP_ON_OK"), False)
    _append_message("user", base_prompt)

    last_eval: dict[str, Any] = {}
    last_assistant = ""
    feedback = ""
    exit_code = 0
    turns = []

    for turn in range(1, max_turns + 1):
        prompt = _initial_turn_prompt(base_prompt) if turn == 1 else _next_turn_prompt(base_prompt, turn, feedback)
        claw_exit, assistant_text = _run_claw(prompt, turn)
        if claw_exit != 0:
            last_assistant = assistant_text
            _append_message("assistant", assistant_text)
            exit_code = claw_exit
            last_eval = {
                "ok": False,
                "status": "claw_failed",
                "compiled": False,
                "correctness": False,
                "speedup": 0.0,
                "error": assistant_text,
                "react_turn": turn,
            }
            break

        code, code_path = _candidate_code_from_workspace_or_text(assistant_text)
        assistant_record = assistant_text
        if code and code not in assistant_record:
            assistant_record = (
                assistant_record.rstrip()
                + "\n\nCandidate written to solution.py:\n"
                + f"```python\n{code}\n```"
            ).strip()
        last_assistant = assistant_record
        _append_message("assistant", assistant_record)
        if code:
            last_eval = _evaluate_solution(turn, code_path)
        else:
            last_eval = _invalid_code_result(turn)

        turns.append({"turn": turn, "assistant_chars": len(assistant_record), "eval": last_eval})
        should_stop = bool(stop_on_ok and last_eval.get("ok"))
        should_continue = turn < max_turns and not should_stop
        if should_continue:
            feedback = _feedback_text(turn, last_eval)
            _append_message("user", feedback)
        if should_stop:
            break

    reward_score = _compute_reward(last_eval or {})
    summary = {
        "message": last_assistant,
        "turns": turns,
        "last_evaluation": last_eval,
        "reward_score": reward_score,
        "reward_extra_info": _reward_extra_info(last_eval or {}, reward_score),
        "max_turns": max_turns,
        "stop_on_ok": stop_on_ok,
        "exit_code": exit_code,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (WORKSPACE / "claw_result.json").write_text(
        json.dumps({"message": last_assistant, "react_summary": summary}, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _log(f"done exit={exit_code} reward={reward_score} turns={len(turns)}")
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
