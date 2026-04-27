"""Adaptive Priority Engine — execution flywheel kernel v0.1.

Given a dict describing observable system state (audit / CI / security /
public claims), recommend the next bottleneck to work. Advisory only.
Stdlib-only. No I/O beyond stdin/argparse in the CLI wrapper.

Priority lattice (first match wins; see ADAPTIVE_PRIORITY_ENGINE_SPEC.md):

  1. SECURITY            — secret_findings > 0 or rotation_required
  2. RUNTIME_HARDENING   — runtime_defaults_insecure
  3. CI_BASELINE         — main_branch_red or ci_failing_count > 0
  4. SUPPLY_CHAIN        — dependency_vulnerabilities > 0 or sbom_stale
  5. PUBLIC_CLAIMS       — public_claims_risky (secret gate cleared)
  6. NODE0_ACTIVATION    — node0_activation_blocked_rows > 0
  7. STOP_AND_LAND       — all observable axes clean
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import PrioritySignal


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _bool(value: Any) -> bool:
    return bool(value)


def recommend_priority(context: dict[str, Any]) -> PrioritySignal:
    if not isinstance(context, dict):
        raise ValueError("priority context must be a dict")

    secrets = _int(context.get("secret_findings"))
    rotation = _bool(context.get("rotation_required"))
    runtime_insecure = _bool(context.get("runtime_defaults_insecure"))
    main_red = _bool(context.get("main_branch_red"))
    ci_failing = _int(context.get("ci_failing_count"))
    dep_vulns = _int(context.get("dependency_vulnerabilities"))
    sbom_stale = _bool(context.get("sbom_stale"))
    claims_risky = _bool(context.get("public_claims_risky"))
    node0_blocked = _int(context.get("node0_activation_blocked_rows"))

    if secrets > 0 or rotation:
        evidence = []
        if secrets > 0:
            evidence.append(f"secret_findings={secrets}")
        if rotation:
            evidence.append("rotation_required=True")
        return PrioritySignal(
            priority="SECURITY",
            reason="Secret hygiene must be closed before other streams advance.",
            confidence=0.95,
            evidence=evidence,
        )

    if runtime_insecure:
        return PrioritySignal(
            priority="RUNTIME_HARDENING",
            reason="Dev-default credential fallbacks in runtime must be closed before use.",
            confidence=0.9,
            evidence=["runtime_defaults_insecure=True"],
        )

    if main_red or ci_failing > 0:
        evidence = []
        if main_red:
            evidence.append("main_branch_red=True")
        if ci_failing > 0:
            evidence.append(f"ci_failing_count={ci_failing}")
        return PrioritySignal(
            priority="CI_BASELINE",
            reason="Trunk CI is not green; unblock baseline before new feature work.",
            confidence=0.85,
            evidence=evidence,
        )

    if dep_vulns > 0 or sbom_stale:
        evidence = []
        if dep_vulns > 0:
            evidence.append(f"dependency_vulnerabilities={dep_vulns}")
        if sbom_stale:
            evidence.append("sbom_stale=True")
        return PrioritySignal(
            priority="SUPPLY_CHAIN",
            reason="Dependency vulnerabilities or stale SBOM require immediate attention.",
            confidence=0.85,
            evidence=evidence,
        )

    if claims_risky:
        return PrioritySignal(
            priority="PUBLIC_CLAIMS",
            reason=(
                "Secret gate is closed; public claim discipline is now the next "
                "bottleneck (prohibited / proof-required items present)."
            ),
            confidence=0.85,
            evidence=[f"secret_findings={secrets}", "public_claims_risky=True"],
        )

    if node0_blocked > 0:
        return PrioritySignal(
            priority="NODE0_ACTIVATION",
            reason="Node0 closure scoreboard has blocked rows awaiting action.",
            confidence=0.75,
            evidence=[f"node0_activation_blocked_rows={node0_blocked}"],
        )

    return PrioritySignal(
        priority="STOP_AND_LAND",
        reason="No active bottleneck detected on observable axes; land the plane.",
        confidence=0.7,
        evidence=["all observable axes clean"],
    )


def recommend_from_json(payload: str) -> PrioritySignal:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("priority context payload must be a JSON object")
    return recommend_priority(data)


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Adaptive priority recommendation engine")
    parser.add_argument(
        "--context",
        required=True,
        help="Path to JSON priority-context file, or '-' for stdin",
    )
    args = parser.parse_args()
    payload = sys.stdin.read() if args.context == "-" else Path(args.context).read_text(encoding="utf-8")
    signal = recommend_from_json(payload)
    print(json.dumps(signal.to_dict(), indent=2))


if __name__ == "__main__":
    main()
