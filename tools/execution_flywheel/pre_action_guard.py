"""Pre-Action Guard — execution flywheel kernel v0.1.

Takes an ActionContext, evaluates matching Patterns, and returns a
GuardDecision.  Stateless. Stdlib-only. Never mutates files. Never calls
git or GitHub.

Decision rules (strongest candidate wins; precedence ABORT > REVALIDATE >
NEEDS_OPERATOR_CONFIRMATION > PROCEED):

  1. No pattern triggers match                            → PROCEED
  2. metadata.fix_already_present is True                 → candidate ABORT
  3. metadata.reviewed_sha differs from metadata.head_sha → candidate REVALIDATE
  4. Any matched pattern declares default_decision        → candidate (its value)
  5. Critical-severity match without other candidates     → NEEDS_OPERATOR_CONFIRMATION
  6. Otherwise                                            → PROCEED
"""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import ActionContext, GuardDecision, Pattern


_DECISION_PRECEDENCE = {
    "ABORT": 3,
    "REVALIDATE": 2,
    "NEEDS_OPERATOR_CONFIRMATION": 1,
    "PROCEED": 0,
}


def evaluate(context: ActionContext, patterns: list[Pattern]) -> GuardDecision:
    matched = [p for p in patterns if p.matches(context.triggers_detected)]
    if not matched:
        return GuardDecision(
            decision="PROCEED",
            reason="No pattern triggers matched the action context.",
            matched_patterns=[],
        )

    matched_ids = [p.pattern_id for p in matched]
    meta = context.metadata or {}
    candidates: list[tuple[str, str]] = []

    if meta.get("fix_already_present") is True:
        candidates.append(
            (
                "ABORT",
                "Requested change is already present upstream. "
                "Applying the edit would fork the fix and diverge from the reviewed commit.",
            )
        )

    reviewed = meta.get("reviewed_sha")
    head = meta.get("head_sha")
    if reviewed and head and reviewed != head:
        candidates.append(
            (
                "REVALIDATE",
                f"PR head ({head}) has commits past the reviewed SHA ({reviewed}). "
                "Inspect origin before editing.",
            )
        )

    for p in matched:
        if p.default_decision:
            candidates.append(
                (
                    p.default_decision,
                    f"pattern {p.pattern_id} (severity={p.severity}) "
                    f"defaults to {p.default_decision}.",
                )
            )

    if not candidates and any(p.severity == "critical" for p in matched):
        candidates.append(
            (
                "NEEDS_OPERATOR_CONFIRMATION",
                f"Matched {len(matched_ids)} critical pattern(s) but metadata is "
                "insufficient to distinguish ABORT from REVALIDATE. Operator must "
                "confirm before edit.",
            )
        )

    if not candidates:
        return GuardDecision(
            decision="PROCEED",
            reason=f"Matched patterns {matched_ids} are non-critical; no halt condition.",
            matched_patterns=matched_ids,
        )

    best_decision, best_reason = max(
        candidates, key=lambda c: _DECISION_PRECEDENCE.get(c[0], -1)
    )
    return GuardDecision(
        decision=best_decision,
        reason=best_reason,
        matched_patterns=matched_ids,
    )


def evaluate_json(payload: str, patterns: list[Pattern]) -> GuardDecision:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Action context payload must be a JSON object")
    ctx = ActionContext(
        action_type=str(data.get("action_type", "unknown")),
        target_files=[str(x) for x in (data.get("target_files") or [])],
        triggers_detected=[str(x) for x in (data.get("triggers_detected") or [])],
        metadata=dict(data.get("metadata") or {}),
    )
    return evaluate(ctx, patterns)


def main() -> None:
    import argparse
    import sys

    from .pattern_registry import load_patterns

    parser = argparse.ArgumentParser(description="Pre-action guard evaluator")
    parser.add_argument(
        "--patterns",
        default=str(Path(__file__).parent / "patterns.yaml"),
        help="Path to patterns.yaml (default: packaged registry)",
    )
    parser.add_argument(
        "--context",
        required=True,
        help="Path to JSON action-context file, or '-' to read from stdin",
    )
    args = parser.parse_args()
    patterns = load_patterns(args.patterns)
    payload = sys.stdin.read() if args.context == "-" else Path(args.context).read_text(encoding="utf-8")
    decision = evaluate_json(payload, patterns)
    print(json.dumps(decision.to_dict(), indent=2))


if __name__ == "__main__":
    main()
