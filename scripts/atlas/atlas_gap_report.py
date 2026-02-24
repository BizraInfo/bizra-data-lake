#!/usr/bin/env python3
"""Generate deterministic Atlas capability alignment reports.

This script treats `docs/atlas_alignment/atlas_capability_matrix.yaml` as the
canonical map (JSON-formatted YAML), then emits a machine-readable report used
by CI and release gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional


ALLOWED_STATUSES = {"implemented", "partial", "missing"}

# ── Quality Tier Definitions ────────────────────────────────────

TIER_ORDER = ["seed", "sprout", "growing", "rooted", "flourishing"]

TIER_CAPABILITIES: dict[str, list[str]] = {
    "seed": ["chat", "teach"],
    "sprout": ["memory_recall", "bootstrap_reflexes"],
    "growing": ["reflex_compilation", "tool_call"],
    "rooted": ["desktop_actions", "file_op", "token_economy"],
    "flourishing": ["browser_nav", "full_action_bus", "agent_as_service"],
}

TIER_UNLOCK_CRITERIA: dict[str, str] = {
    "seed": "Default starting tier",
    "sprout": "1+ atoms taught",
    "growing": "25+ atoms and 10+ messages",
    "rooted": "100+ atoms or synthesis complete",
    "flourishing": "200+ atoms, multi-provider, 3+ elite reflexes",
}

# Maps Atlas capability priority levels to user tiers.
# P0 = core infrastructure available from the start.
# P1 = available at Growing tier (reflex compilation + tool calls).
# P2 = available at Rooted tier (desktop actions + token economy).
# P3+ = available at Flourishing tier (full action bus + agent-as-service).
PRIORITY_TO_TIER: dict[str, str] = {
    "P0": "seed",
    "P1": "growing",
    "P2": "rooted",
    "P3": "flourishing",
}


def _compute_tier_output(tier_key: str) -> dict[str, Any]:
    """Return capabilities unlocked/locked for a given user tier."""
    if tier_key not in TIER_ORDER:
        raise ValueError(f"Unknown tier: {tier_key}")

    tier_idx = TIER_ORDER.index(tier_key)
    unlocked: list[str] = []
    locked: list[str] = []

    for i, tk in enumerate(TIER_ORDER):
        caps = TIER_CAPABILITIES.get(tk, [])
        if i <= tier_idx:
            unlocked.extend(caps)
        else:
            locked.extend(caps)

    next_tier = TIER_ORDER[tier_idx + 1] if tier_idx < len(TIER_ORDER) - 1 else None
    unlock_criteria = (
        TIER_UNLOCK_CRITERIA.get(next_tier, "") if next_tier else "Max tier reached"
    )

    # Build priority mapping for this tier
    available_priorities: list[str] = []
    for priority, mapped_tier in sorted(PRIORITY_TO_TIER.items()):
        if TIER_ORDER.index(mapped_tier) <= tier_idx:
            available_priorities.append(priority)

    return {
        "tier": tier_key,
        "capabilities_unlocked": unlocked,
        "capabilities_locked": locked,
        "next_tier": next_tier,
        "unlock_criteria": unlock_criteria,
        "available_priorities": available_priorities,
    }


def user_tier_report(tier_name: str) -> dict[str, Any]:
    """Public API: return which capabilities are unlocked for a given tier.

    Parameters
    ----------
    tier_name:
        One of ``"seed"``, ``"sprout"``, ``"growing"``, ``"rooted"``,
        ``"flourishing"``.

    Returns
    -------
    dict
        Keys: ``tier``, ``capabilities_unlocked``, ``capabilities_locked``,
        ``next_tier``, ``unlock_criteria``, ``available_priorities``.

    Raises
    ------
    ValueError
        If *tier_name* is not a recognised tier key.
    """
    return _compute_tier_output(tier_name)


def _load_matrix(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)

    if not isinstance(payload, dict):
        raise ValueError("Matrix root must be an object")

    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list):
        raise ValueError("Matrix must include a 'capabilities' list")

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(capabilities):
        if not isinstance(item, dict):
            raise ValueError(f"Capability entry {idx} must be an object")

        capability = str(item.get("capability", "")).strip()
        status = str(item.get("status", "")).strip().lower()
        owner = str(item.get("owner", "")).strip()
        target_phase = str(item.get("target_phase", "")).strip()
        evidence = item.get("evidence", [])

        if not capability:
            raise ValueError(f"Capability entry {idx} missing 'capability'")
        if status not in ALLOWED_STATUSES:
            raise ValueError(
                f"Capability '{capability}' has invalid status '{status}'"
            )
        if not isinstance(evidence, list):
            raise ValueError(f"Capability '{capability}' evidence must be a list")

        normalized.append(
            {
                "capability": capability,
                "status": status,
                "owner": owner,
                "target_phase": target_phase,
                "evidence": [str(x) for x in evidence],
            }
        )

    payload["capabilities"] = sorted(normalized, key=lambda row: row["capability"])
    return payload


def _load_runtime_status(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Runtime status must be a JSON object")
    return payload


def _extract_pat_sat_chain(status_payload: dict[str, Any]) -> dict[str, Any]:
    chain = (
        status_payload.get("pat_sat", {})
        .get("negotiation_receipt_chain", {})
    )
    if not isinstance(chain, dict):
        chain = {}
    raw_total = chain.get("total_negotiation_receipts", 0)
    try:
        total_receipts = int(raw_total or 0)
    except (TypeError, ValueError):
        total_receipts = 0
    return {
        "verified_end_to_end": bool(chain.get("verified_end_to_end", False)),
        "chain_valid": chain.get("chain_valid"),
        "total_negotiation_receipts": total_receipts,
        "latest_sequence": chain.get("latest_sequence"),
        "latest_entry_hash": chain.get("latest_entry_hash"),
        "latest_receipt_id": chain.get("latest_receipt_id"),
    }


def _build_report(
    matrix: dict[str, Any],
    matrix_path: Path,
    runtime_status: Optional[dict[str, Any]] = None,
    runtime_status_path: Optional[Path] = None,
) -> dict[str, Any]:
    capabilities = [dict(row) for row in matrix["capabilities"]]

    status_counts = {status: 0 for status in sorted(ALLOWED_STATUSES)}
    for row in capabilities:
        status_counts[row["status"]] += 1

    p0_regressions = [
        row
        for row in capabilities
        if row.get("target_phase") == "P0" and row["status"] != "implemented"
    ]

    matrix_bytes = matrix_path.read_bytes()
    matrix_sha256 = hashlib.sha256(matrix_bytes).hexdigest()
    pat_sat_runtime = _extract_pat_sat_chain(runtime_status or {})

    for row in capabilities:
        if row.get("capability") == "PAT-SAT negotiation protocol":
            row["runtime_verification"] = {
                "status": (
                    "verified"
                    if pat_sat_runtime["verified_end_to_end"]
                    else "unverified"
                ),
                "source": "runtime_status",
                "receipt_chain": pat_sat_runtime,
            }
            break

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "matrix_schema_version": matrix.get("schema_version", "unknown"),
        "matrix_source": matrix.get("source", "unknown"),
        "matrix_path": str(matrix_path),
        "matrix_sha256": matrix_sha256,
        "capability_count": len(capabilities),
        "status_counts": status_counts,
        "p0_regression_count": len(p0_regressions),
        "p0_regressions": p0_regressions,
        "overall_status": "pass" if not p0_regressions else "fail",
        "pat_sat_receipt_chain_verified": bool(
            pat_sat_runtime["verified_end_to_end"]
        ),
        "pat_sat_receipt_chain": pat_sat_runtime,
        "runtime_status_path": (
            str(runtime_status_path) if runtime_status_path is not None else None
        ),
        "capabilities": capabilities,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Atlas alignment report")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("docs/atlas_alignment/atlas_capability_matrix.yaml"),
        help="Path to capability matrix (JSON-formatted YAML)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/atlas/atlas_gap_report.json"),
        help="Output report path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print report JSON to stdout",
    )
    parser.add_argument(
        "--runtime-status",
        type=Path,
        default=None,
        help=(
            "Optional runtime status JSON (for end-to-end capability verification "
            "signals such as PAT↔SAT receipt-chain health)"
        ),
    )
    parser.add_argument(
        "--fail-on-p0-regression",
        action="store_true",
        help="Return non-zero exit if any P0 capability is not implemented",
    )
    parser.add_argument(
        "--user-tier",
        choices=["seed", "sprout", "growing", "rooted", "flourishing"],
        default=None,
        help="Output capabilities unlocked at the specified user tier",
    )
    args = parser.parse_args()

    # ── User tier mode: standalone JSON output, no matrix needed ──
    if args.user_tier is not None:
        tier_report = _compute_tier_output(args.user_tier)
        print(json.dumps(tier_report, indent=2))
        return 0

    matrix = _load_matrix(args.matrix)
    runtime_status = (
        _load_runtime_status(args.runtime_status)
        if args.runtime_status is not None
        else None
    )
    report = _build_report(
        matrix,
        args.matrix,
        runtime_status=runtime_status,
        runtime_status_path=args.runtime_status,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))

    if args.fail_on_p0_regression and report["p0_regression_count"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
