#!/usr/bin/env python3
"""Build a unified release-readiness bundle from program + proof artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ci_delivery_program_gate import (
    DEFAULT_PROGRAM_PATH,
    validate_delivery_program,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_release_readiness_bundle(
    *,
    program: dict[str, Any],
    program_issues: list[str],
    canonical_report: dict[str, Any],
    membrane_report: dict[str, Any],
    boundary_report: dict[str, Any],
) -> dict[str, Any]:
    scorecard_counts = Counter(row["status"] for row in program.get("scorecard", []))
    workstream_counts = Counter(ws["priority"] for ws in program.get("workstreams", []))

    planes = {
        "delivery_program": {
            "passed": not program_issues,
            "issues": program_issues,
        },
        "canonical_e2e": {
            "passed": bool(
                canonical_report.get("gate_verdict", {}).get("passed", False)
            ),
            "failed_metrics": list(
                canonical_report.get("gate_verdict", {}).get("failed_metrics", [])
            ),
        },
        "membrane_tax": {
            "passed": bool(
                membrane_report.get("gate_verdict", {}).get("passed", False)
            ),
            "failed_metrics": list(
                membrane_report.get("gate_verdict", {}).get("failed_metrics", [])
            ),
            "clamped_negative_metrics": membrane_report.get("benchmark_sanity", {}).get(
                "clamped_negative_metrics", {}
            ),
        },
        "boundary_quality": {
            "passed": bool(
                boundary_report.get("gate_verdict", {}).get("passed", False)
            ),
            "failed_checks": sorted(
                name
                for name, passed in boundary_report.get("gate_verdict", {})
                .get("checks", {})
                .items()
                if not passed
            ),
        },
    }
    failed_planes = [
        plane_name for plane_name, plane in planes.items() if not plane["passed"]
    ]

    signal = boundary_report.get("boundary_signal", {})
    canonical_results = canonical_report.get("benchmark_results", {})
    membrane_tax = membrane_report.get("membrane_tax", {})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "program": {
            "id": program["program_id"],
            "version": program["version"],
            "status": program["status"],
            "north_star": program["north_star"],
            "top_next_step": dict(program["top_next_step"]),
            "scorecard_status_counts": dict(sorted(scorecard_counts.items())),
            "workstream_priority_counts": dict(sorted(workstream_counts.items())),
        },
        "planes": planes,
        "overall_verdict": {
            "passed": not failed_planes,
            "failed_planes": failed_planes,
        },
        "runtime_signals": {
            "full_spine_ms": float(canonical_results.get("full_spine_ms", 0.0) or 0.0),
            "node0_breathe_ms": float(
                canonical_results.get("node0_breathe_ms", 0.0) or 0.0
            ),
            "eventbus_emission_ms": float(
                canonical_results.get("eventbus_emission_ms", 0.0) or 0.0
            ),
            "governance_tax_ratio": float(
                membrane_tax.get("governance_tax_ratio", 0.0) or 0.0
            ),
            "rss_growth_mb": float(membrane_tax.get("rss_growth_mb", 0.0) or 0.0),
            "boundary_quality_multiplier": float(
                signal.get("boundary_quality_multiplier", 1.0) or 1.0
            ),
            "boundary_error_receipts": int(
                signal.get("boundary_error_receipts", 0) or 0
            ),
            "boundary_retries": int(signal.get("boundary_retries", 0) or 0),
            "boundary_degradations": int(signal.get("boundary_degradations", 0) or 0),
        },
    }


def render_release_readiness_markdown(bundle: dict[str, Any]) -> str:
    program = bundle["program"]
    runtime = bundle["runtime_signals"]
    planes = bundle["planes"]
    verdict = bundle["overall_verdict"]

    lines = [
        "# BIZRA Release Readiness Bundle",
        "",
        f"- Program: `{program['id']}`",
        f"- Version: `{program['version']}`",
        f"- Overall Verdict: `{'PASS' if verdict['passed'] else 'FAIL'}`",
        "",
        "## Planes",
        "",
        "| Plane | Verdict | Notes |",
        "| --- | --- | --- |",
    ]

    lines.append(
        f"| `delivery_program` | `{'PASS' if planes['delivery_program']['passed'] else 'FAIL'}` | "
        f"{len(planes['delivery_program']['issues'])} issue(s) |"
    )
    lines.append(
        f"| `canonical_e2e` | `{'PASS' if planes['canonical_e2e']['passed'] else 'FAIL'}` | "
        f"{len(planes['canonical_e2e']['failed_metrics'])} failed metric(s) |"
    )
    lines.append(
        f"| `membrane_tax` | `{'PASS' if planes['membrane_tax']['passed'] else 'FAIL'}` | "
        f"{len(planes['membrane_tax']['clamped_negative_metrics'])} clamped metric(s) |"
    )
    lines.append(
        f"| `boundary_quality` | `{'PASS' if planes['boundary_quality']['passed'] else 'FAIL'}` | "
        f"{len(planes['boundary_quality']['failed_checks'])} failed check(s) |"
    )

    lines.extend(
        [
            "",
            "## Runtime Signals",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| `full_spine_ms` | `{runtime['full_spine_ms']:.2f}` |",
            f"| `node0_breathe_ms` | `{runtime['node0_breathe_ms']:.2f}` |",
            f"| `eventbus_emission_ms` | `{runtime['eventbus_emission_ms']:.2f}` |",
            f"| `governance_tax_ratio` | `{runtime['governance_tax_ratio']:.4f}` |",
            f"| `rss_growth_mb` | `{runtime['rss_growth_mb']:.2f}` |",
            f"| `boundary_quality_multiplier` | `{runtime['boundary_quality_multiplier']:.4f}` |",
            f"| `boundary_error_receipts` | `{runtime['boundary_error_receipts']}` |",
            f"| `boundary_retries` | `{runtime['boundary_retries']}` |",
            f"| `boundary_degradations` | `{runtime['boundary_degradations']}` |",
            "",
            "## Program Posture",
            "",
            "| Status | Count |",
            "| --- | ---: |",
        ]
    )
    for status, count in program["scorecard_status_counts"].items():
        lines.append(f"| `{status}` | `{count}` |")

    lines.extend(
        [
            "",
            "## Top Next Step",
            "",
            f"- `{program['top_next_step']['id']}`: {program['top_next_step']['title']}",
            f"- Reason: {program['top_next_step']['reason']}",
        ]
    )

    if planes["delivery_program"]["issues"]:
        lines.extend(
            [
                "",
                "## Delivery Program Issues",
                "",
            ]
        )
        for issue in planes["delivery_program"]["issues"]:
            lines.append(f"- {issue}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a unified release-readiness bundle."
    )
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM_PATH)
    parser.add_argument("--canonical-report", type=Path, required=True)
    parser.add_argument("--membrane-report", type=Path, required=True)
    parser.add_argument("--boundary-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    program = _load_json(args.program)
    program_issues = validate_delivery_program(args.program)
    bundle = build_release_readiness_bundle(
        program=program,
        program_issues=program_issues,
        canonical_report=_load_json(args.canonical_report),
        membrane_report=_load_json(args.membrane_report),
        boundary_report=_load_json(args.boundary_report),
    )
    markdown = render_release_readiness_markdown(bundle)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
