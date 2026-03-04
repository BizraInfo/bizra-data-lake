"""
Phase65 KPI packer.

Combines lifecycle summary + blueprint gate report into a single KPI snapshot
artifact and optional GitHub Actions outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tier(snapshot: dict[str, Any]) -> str:
    if (
        snapshot["gate_passed"]
        and snapshot["signed_receipts"]
        and snapshot["speedup_system1_vs_system2"] >= 8.0
        and snapshot["avg_ihsan"] >= 0.75
        and snapshot["avg_latency_ms"] <= 2200.0
    ):
        return "elite-operational"
    if snapshot["gate_passed"]:
        return "operational"
    return "degraded"


def build_kpi_snapshot(
    summary_payload: dict[str, Any],
    gate_payload: dict[str, Any],
) -> dict[str, Any]:
    summary = summary_payload.get("summary", summary_payload)
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "final_state": summary.get("final_state"),
        "gate_passed": bool(gate_payload.get("gate_passed")),
        "snr_score": float(gate_payload.get("snr_score", 0.0)),
        "actions_total": int(summary.get("actions_total", 0)),
        "system1_ratio": float(summary.get("system1_ratio", 0.0)),
        "avg_ihsan": float(summary.get("avg_ihsan", 0.0)),
        "avg_latency_ms": float(summary.get("avg_latency_ms", 0.0)),
        "speedup_system1_vs_system2": float(
            summary.get("speedup_system1_vs_system2", 0.0)
        ),
        "impt_balance": float(summary.get("impt_balance", 0.0)),
        "ledger_chain_valid": bool(summary.get("ledger_chain_valid", False)),
        "signed_receipts": bool(summary.get("signed_receipts", False)),
    }
    snapshot["tier"] = _tier(snapshot)
    return snapshot


def render_markdown(snapshot: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase65 KPI Snapshot",
            "",
            "| KPI | Value |",
            "|-----|-------|",
            f"| Timestamp (UTC) | {snapshot['timestamp_utc']} |",
            f"| Tier | {snapshot['tier']} |",
            f"| Gate Passed | {snapshot['gate_passed']} |",
            f"| SNR Score | {snapshot['snr_score']:.4f} |",
            f"| Final State | {snapshot['final_state']} |",
            f"| Actions Total | {snapshot['actions_total']} |",
            f"| System-1 Ratio | {snapshot['system1_ratio']:.3f} |",
            f"| Avg Ihsan | {snapshot['avg_ihsan']:.3f} |",
            f"| Avg Latency (ms) | {snapshot['avg_latency_ms']:.2f} |",
            f"| Speedup S1/S2 | {snapshot['speedup_system1_vs_system2']:.2f} |",
            f"| IMPT Balance | {snapshot['impt_balance']:.3f} |",
            f"| Ledger Chain Valid | {snapshot['ledger_chain_valid']} |",
            f"| Signed Receipts | {snapshot['signed_receipts']} |",
        ]
    )


def _emit_github_outputs(snapshot: dict[str, Any], output_path: Path) -> None:
    lines = [
        f"gate_passed={'true' if snapshot['gate_passed'] else 'false'}",
        f"snr_score={snapshot['snr_score']:.4f}",
        f"avg_ihsan={snapshot['avg_ihsan']:.4f}",
        f"avg_latency_ms={snapshot['avg_latency_ms']:.2f}",
        f"speedup={snapshot['speedup_system1_vs_system2']:.2f}",
        f"signed_receipts={'true' if snapshot['signed_receipts'] else 'false'}",
        f"tier={snapshot['tier']}",
    ]
    with output_path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Phase65 KPI snapshot artifacts."
    )
    parser.add_argument(
        "--summary", type=Path, required=True, help="Lifecycle summary JSON path."
    )
    parser.add_argument(
        "--gate-report", type=Path, required=True, help="Phase65 gate report JSON path."
    )
    parser.add_argument(
        "--out-json", type=Path, required=True, help="Output KPI JSON path."
    )
    parser.add_argument(
        "--out-md", type=Path, default=None, help="Optional output markdown path."
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="Optional GITHUB_OUTPUT file path.",
    )
    args = parser.parse_args()

    summary_payload = _load_json(args.summary)
    gate_payload = _load_json(args.gate_report)
    snapshot = build_kpi_snapshot(summary_payload, gate_payload)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(snapshot), encoding="utf-8")
    if args.github_output is not None:
        _emit_github_outputs(snapshot, args.github_output)

    print(json.dumps(snapshot, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
