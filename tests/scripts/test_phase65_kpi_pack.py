from __future__ import annotations

import json
from pathlib import Path

from scripts.ops.phase65_kpi_pack import build_kpi_snapshot, render_markdown


def test_build_kpi_snapshot_elite_tier() -> None:
    summary = {
        "summary": {
            "final_state": "FLOURISHING",
            "actions_total": 10,
            "system1_ratio": 0.9,
            "avg_ihsan": 0.86,
            "avg_latency_ms": 1200.0,
            "speedup_system1_vs_system2": 9.1,
            "impt_balance": 150.0,
            "ledger_chain_valid": True,
            "signed_receipts": True,
        }
    }
    gate = {"gate_passed": True, "snr_score": 0.96}
    snapshot = build_kpi_snapshot(summary, gate)

    assert snapshot["tier"] == "elite-operational"
    assert snapshot["gate_passed"] is True
    assert snapshot["signed_receipts"] is True


def test_build_kpi_snapshot_degraded_tier_when_gate_fails() -> None:
    summary = {
        "summary": {
            "final_state": "LEARNING",
            "actions_total": 2,
            "system1_ratio": 0.0,
            "avg_ihsan": 0.3,
            "avg_latency_ms": 5000.0,
            "speedup_system1_vs_system2": 1.0,
            "impt_balance": 0.0,
            "ledger_chain_valid": False,
            "signed_receipts": False,
        }
    }
    gate = {"gate_passed": False, "snr_score": 0.5}
    snapshot = build_kpi_snapshot(summary, gate)
    assert snapshot["tier"] == "degraded"


def test_render_markdown_contains_key_rows(tmp_path: Path) -> None:
    summary = {
        "summary": {
            "final_state": "FLOURISHING",
            "actions_total": 6,
            "system1_ratio": 0.167,
            "avg_ihsan": 0.792,
            "avg_latency_ms": 1994.17,
            "speedup_system1_vs_system2": 8.21,
            "impt_balance": 135.852,
            "ledger_chain_valid": True,
            "signed_receipts": True,
        }
    }
    gate = {"gate_passed": True, "snr_score": 1.0}
    snapshot = build_kpi_snapshot(summary, gate)
    md = render_markdown(snapshot)

    assert "| Tier |" in md
    assert "| Signed Receipts | True |" in md
    assert "| SNR Score | 1.0000 |" in md

    output = tmp_path / "snapshot.json"
    output.write_text(json.dumps(snapshot), encoding="utf-8")
    assert json.loads(output.read_text(encoding="utf-8"))["final_state"] == "FLOURISHING"
