#!/usr/bin/env python3
"""Render canonical benchmark + membrane-tax results into a CI summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_summary(
    *,
    canonical_report: dict[str, Any],
    membrane_report: dict[str, Any],
) -> str:
    canonical_results = canonical_report["benchmark_results"]
    canonical_gate = canonical_report["gate_verdict"]
    membrane_tax = membrane_report["membrane_tax"]
    membrane_gate = membrane_report["gate_verdict"]
    clamped = membrane_report.get("benchmark_sanity", {}).get(
        "clamped_negative_metrics", {}
    )

    lines = [
        "## Performance Proof Plane",
        "",
        f"- Canonical E2E: {'PASS' if canonical_gate['passed'] else 'FAIL'}",
        f"- Membrane Tax: {'PASS' if membrane_gate['passed'] else 'FAIL'}",
        "",
        "### Canonical E2E",
        "",
        "| Metric | Value | Gate |",
        "|---|---:|---:|",
    ]

    for metric, check in canonical_gate["checks"].items():
        lines.append(
            f"| {metric} | {float(canonical_results.get(metric, 0.0)):.2f} | {float(check['gate']):.2f} |"
        )

    lines.extend(
        [
            "",
            "### Membrane Tax",
            "",
            "| Metric | Value | Gate |",
            "|---|---:|---:|",
            f"| governance_tax_ms | {float(membrane_tax['governance_tax_ms']):.2f} | {float(membrane_gate['checks']['governance_tax_ms']['gate']):.2f} |",
            f"| governance_tax_ratio | {float(membrane_tax['governance_tax_ratio']):.4f} | {float(membrane_gate['checks']['governance_tax_ratio']['gate']):.4f} |",
            f"| rss_growth_mb | {float(membrane_tax['rss_growth_mb']):.2f} | {float(membrane_gate['checks']['rss_growth_mb']['gate']):.2f} |",
            "",
        ]
    )

    if clamped:
        lines.extend(
            [
                "### Sanity Findings",
                "",
                f"- Clamped negative metrics: {json.dumps(clamped, sort_keys=True)}",
            ]
        )
    else:
        lines.extend(
            [
                "### Sanity Findings",
                "",
                "- No clamped negative metrics detected.",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render benchmark summary markdown.")
    parser.add_argument("--canonical-report", type=Path, required=True)
    parser.add_argument("--membrane-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = render_summary(
        canonical_report=_load_json(args.canonical_report),
        membrane_report=_load_json(args.membrane_report),
    )
    args.output.write_text(summary, encoding="utf-8")
    print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
