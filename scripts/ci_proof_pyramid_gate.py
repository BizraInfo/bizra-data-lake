#!/usr/bin/env python3
"""Proof Pyramid Quality Gate — validates constitutional proof chain integrity.

Runs as the final aggregation step in the Proof Pyramid Gate workflow.
Merges evidence fragments from all 6 PP sub-gates, validates that every
gate passed, writes a structured evidence JSON, and exits 1 if any gate
failed.

Usage:
    python scripts/ci_proof_pyramid_gate.py \\
        --evidence-dir evidence-fragments \\
        --gate-results \\
            pp001=success \\
            pp002=success \\
            pp003=success \\
            pp004=failure \\
            pp005=success \\
            pp006=success \\
        --output evidence/proof_pyramid_evidence.json

Exit codes:
    0  All gates passed (or were skipped)
    1  One or more gates failed
    2  Argument error

Constitutional thresholds come from ``core.integration.constants``, the
Python authoritative source aligned with Rust ``bizra-core``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.integration.constants import (
    ADL_GINI_THRESHOLD as ADL_GINI_MAX,
    IHSAN_THRESHOLD,
    MAX_HARM_SCORE,
    MIN_CONFIDENCE,
    SNR_THRESHOLD,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants — imported from the Python/Rust-aligned authoritative source.
# ─────────────────────────────────────────────────────────────────────────────

# Gate registry: gate_id → human name
GATE_REGISTRY: dict[str, str] = {
    "pp001": "PP-001 Receipt Chain Integrity",
    "pp002": "PP-002 Sippar Encoding Verification",
    "pp003": "PP-003 SMT-LIB2 Syntax Gate",
    "pp004": "PP-004 Fate-Binding Z3 Proofs",
    "pp005": "PP-005 Mission → ProofSpace Bridge",
    "pp006": "PP-006 E2E Proof Pyramid (Layer 0→5)",
}

# Result values that are treated as passing
PASSING_RESULTS: frozenset[str] = frozenset({"success", "skipped"})
# Result values that are treated as failing
FAILING_RESULTS: frozenset[str] = frozenset({"failure", "cancelled"})


# ─────────────────────────────────────────────────────────────────────────────
# Evidence merging
# ─────────────────────────────────────────────────────────────────────────────


def load_evidence_fragments(evidence_dir: pathlib.Path) -> dict[str, Any]:
    """Load all JSON evidence fragments from the given directory tree."""
    fragments: dict[str, Any] = {}
    if not evidence_dir.exists():
        print(
            f"[INFO] Evidence directory not found: {evidence_dir} — using empty fragments"
        )
        return fragments
    for json_file in sorted(evidence_dir.rglob("*.json")):
        try:
            data = json.loads(json_file.read_text())
            gate_id = data.get("gate", json_file.stem).lower().replace("-", "")
            fragments[gate_id] = data
            print(f"[INFO] Loaded evidence fragment: {json_file} (gate={gate_id})")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Could not load evidence fragment {json_file}: {exc}")
    return fragments


# ─────────────────────────────────────────────────────────────────────────────
# Gate result parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_gate_results(gate_results_raw: list[str]) -> dict[str, str]:
    """Parse `key=value` gate result strings into a dict.

    Example input: ["pp001=success", "pp002=failure"]
    Example output: {"pp001": "success", "pp002": "failure"}
    """
    parsed: dict[str, str] = {}
    for item in gate_results_raw:
        if "=" not in item:
            print(
                f"[WARN] Ignoring malformed gate result (expected key=value): {item!r}"
            )
            continue
        key, _, value = item.partition("=")
        parsed[key.strip().lower()] = value.strip().lower()
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Gate validation
# ─────────────────────────────────────────────────────────────────────────────


def validate_gates(
    gate_results: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Validate all 6 PP gates.

    Returns:
        gate_details   — List of per-gate result dicts
        failed_gates   — Gate IDs that failed
        unknown_gates  — Gate IDs with unknown results
    """
    gate_details: list[dict[str, Any]] = []
    failed_gates: list[str] = []
    unknown_gates: list[str] = []

    for gate_id, gate_name in GATE_REGISTRY.items():
        result = gate_results.get(gate_id, "unknown")
        passed = result in PASSING_RESULTS
        failed = result in FAILING_RESULTS

        if failed:
            failed_gates.append(gate_id)
        elif result not in PASSING_RESULTS:
            unknown_gates.append(gate_id)

        gate_details.append(
            {
                "gate_id": gate_id.upper().replace("PP0", "PP-0"),
                "name": gate_name,
                "result": result,
                "passed": passed,
                "failed": failed,
            }
        )

    return gate_details, failed_gates, unknown_gates


# ─────────────────────────────────────────────────────────────────────────────
# Evidence assembly
# ─────────────────────────────────────────────────────────────────────────────


def assemble_evidence(
    gate_results: dict[str, str],
    gate_details: list[dict[str, Any]],
    failed_gates: list[str],
    fragments: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the complete proof pyramid evidence bundle."""
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    sha = os.environ.get("GITHUB_SHA", "unknown")
    ref = os.environ.get("GITHUB_REF", "unknown")
    repository = os.environ.get("GITHUB_REPOSITORY", "unknown")
    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")

    overall_pass = len(failed_gates) == 0

    return {
        "schema_version": "1.0.0",
        "bundle_type": "proof_pyramid_evidence",
        "overall_pass": overall_pass,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "run": {
            "run_id": run_id,
            "sha": sha,
            "ref": ref,
            "repository": repository,
            "url": f"{server_url}/{repository}/actions/runs/{run_id}",
        },
        "constitutional_thresholds": {
            "IHSAN_THRESHOLD": IHSAN_THRESHOLD,
            "SNR_THRESHOLD": SNR_THRESHOLD,
            "ADL_GINI_MAX": ADL_GINI_MAX,
            "MAX_HARM_SCORE": MAX_HARM_SCORE,
            "MIN_CONFIDENCE": MIN_CONFIDENCE,
        },
        "gates": gate_details,
        "summary": {
            "total": len(GATE_REGISTRY),
            "passed": sum(1 for g in gate_details if g["passed"]),
            "failed": len(failed_gates),
            "failed_gate_ids": failed_gates,
        },
        "fragments": fragments,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def print_report(evidence: dict[str, Any]) -> None:
    """Print a human-readable gate report to stdout."""
    summary = evidence["summary"]
    overall = evidence["overall_pass"]

    print()
    print("━" * 70)
    print("  PROOF PYRAMID QUALITY GATE REPORT")
    print("━" * 70)
    print(f"  SHA:       {evidence['run']['sha']}")
    print(f"  Ref:       {evidence['run']['ref']}")
    print(f"  Run ID:    {evidence['run']['run_id']}")
    print(f"  Timestamp: {evidence['timestamp']}")
    print()
    print(f"  Gates: {summary['passed']}/{summary['total']} passed")
    print()

    for gate in evidence["gates"]:
        icon = "✅" if gate["passed"] else ("❌" if gate["failed"] else "❓")
        result_str = gate["result"].upper().ljust(10)
        print(f"  {icon}  {gate['gate_id'].ljust(8)}  {result_str}  {gate['name']}")

    print()
    print("━" * 70)
    if overall:
        print("  ✅ ALL PROOF PYRAMID GATES PASSED")
    else:
        print(f"  ❌ GATE FAILURE — {len(summary['failed_gate_ids'])} gate(s) failed:")
        for gate_id in summary["failed_gate_ids"]:
            gate_name = GATE_REGISTRY.get(gate_id, gate_id)
            print(f"       • {gate_id.upper()}: {gate_name}")
        print()
        print("  Constitutional constraint violated.")
        print(f"  IHSAN_THRESHOLD={IHSAN_THRESHOLD}, SNR_THRESHOLD={SNR_THRESHOLD}")
    print("━" * 70)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Proof Pyramid Quality Gate — validates all 6 PP gates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--evidence-dir",
        type=pathlib.Path,
        default=pathlib.Path("evidence-fragments"),
        help="Directory containing PP evidence JSON fragments (default: evidence-fragments/)",
    )
    parser.add_argument(
        "--gate-results",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="Gate results as key=value pairs, e.g. pp001=success pp002=failure",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("evidence/proof_pyramid_evidence.json"),
        help="Output path for the merged evidence JSON (default: evidence/proof_pyramid_evidence.json)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail on unknown gate results (not just explicit failures)",
    )

    args = parser.parse_args(argv)

    # Parse gate results from CLI
    gate_results = parse_gate_results(args.gate_results)

    if not gate_results:
        print(
            "[WARN] No gate results provided via --gate-results. "
            "All gates will be marked as 'unknown'."
        )

    # Load evidence fragments
    fragments = load_evidence_fragments(args.evidence_dir)

    # Validate gates
    gate_details, failed_gates, unknown_gates = validate_gates(gate_results)

    # If --strict, treat unknown as failure
    if args.strict and unknown_gates:
        print(
            f"[ERROR] --strict mode: {len(unknown_gates)} gate(s) have unknown results: "
            f"{unknown_gates}"
        )
        failed_gates = failed_gates + unknown_gates

    # Assemble evidence bundle
    evidence = assemble_evidence(gate_results, gate_details, failed_gates, fragments)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2))
    print(f"[INFO] Evidence written to: {args.output}")

    # Print report
    print_report(evidence)

    # Set GitHub Actions output
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(
                f"proof_pyramid_passed={'true' if evidence['overall_pass'] else 'false'}\n"
            )
            f.write(f"gates_passed={evidence['summary']['passed']}\n")
            f.write(f"gates_failed={evidence['summary']['failed']}\n")

    return 0 if evidence["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
