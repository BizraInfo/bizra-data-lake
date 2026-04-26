"""Flywheel Runner — execution flywheel kernel v0.1.

Combines pre-action guard + priority recommendation into a single advisory
evaluation over a JSON context. Prints a JSON result or a human-readable
summary. Never executes destructive actions, never calls external services,
never mutates runtime.

CLI:
    python3 -m tools.execution_flywheel.flywheel_runner --context context.json
    python3 -m tools.execution_flywheel.flywheel_runner --context - --explain-summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .pattern_registry import load_patterns
from .pre_action_guard import evaluate
from .priority_engine import recommend_priority
from .schemas import ActionContext, FlywheelResult, Pattern


def _to_action_context(data: dict[str, Any]) -> ActionContext:
    return ActionContext(
        action_type=str(data.get("action_type", "unknown")),
        target_files=[str(x) for x in (data.get("target_files") or [])],
        triggers_detected=[str(x) for x in (data.get("triggers_detected") or [])],
        metadata=dict(data.get("metadata") or {}),
    )


def run_flywheel(context: dict[str, Any], patterns: list[Pattern]) -> FlywheelResult:
    if not isinstance(context, dict):
        raise ValueError("flywheel context must be a dict")

    action_ctx = _to_action_context(context)
    guard = evaluate(action_ctx, patterns)

    priority_ctx = context.get("priority_context") or {}
    if not isinstance(priority_ctx, dict):
        raise ValueError("flywheel context.priority_context must be a dict")
    priority = recommend_priority(priority_ctx)

    explanations = [
        f"guard.decision={guard.decision}: {guard.reason}",
        f"priority.priority={priority.priority}: {priority.reason}",
    ]
    if guard.matched_patterns:
        explanations.append("matched_patterns=" + ",".join(guard.matched_patterns))
    if priority.evidence:
        explanations.append("priority.evidence=" + "; ".join(priority.evidence))
    return FlywheelResult(guard=guard, priority=priority, explanations=explanations)


def run_from_json(payload: str, patterns: list[Pattern]) -> FlywheelResult:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("flywheel context payload must be a JSON object")
    return run_flywheel(data, patterns)


def _explain_summary(result: FlywheelResult) -> str:
    lines = [
        "BIZRA Autonomous Flywheel Kernel v0.1 — advisory summary",
        "",
        f"Guard   : {result.guard.decision}",
        f"Priority: {result.priority.priority} (confidence {result.priority.confidence:.2f})",
        "",
        "Reasoning (observable evidence only):",
    ]
    lines.extend(f"  - {line}" for line in result.explanations)
    lines.append("")
    lines.append("This is an advisory recommendation. Operator decides the action.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="BIZRA Autonomous Flywheel Kernel runner")
    parser.add_argument(
        "--patterns",
        default=str(Path(__file__).parent / "patterns.yaml"),
        help="Path to patterns.yaml (default: packaged registry)",
    )
    parser.add_argument(
        "--context",
        required=True,
        help="Path to JSON context file, or '-' for stdin",
    )
    parser.add_argument(
        "--explain-summary",
        action="store_true",
        help="Emit a human-readable advisory summary instead of JSON",
    )
    args = parser.parse_args()
    patterns = load_patterns(args.patterns)
    payload = sys.stdin.read() if args.context == "-" else Path(args.context).read_text(encoding="utf-8")
    result = run_from_json(payload, patterns)
    if args.explain_summary:
        print(_explain_summary(result))
    else:
        print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
