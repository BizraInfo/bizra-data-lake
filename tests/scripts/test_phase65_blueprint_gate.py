from __future__ import annotations

import json
from pathlib import Path

from scripts.ops.phase65_blueprint_gate import evaluate


def _load_cfg_via_yaml() -> dict:
    import yaml

    return yaml.safe_load(
        Path("config/phase65_masterpiece_roadmap.yaml").read_text(encoding="utf-8")
    )


def test_phase65_gate_passes_for_valid_summary() -> None:
    cfg = _load_cfg_via_yaml()
    summary = {
        "final_state": "FLOURISHING",
        "ledger_chain_valid": True,
        "signed_receipts": True,
        "avg_ihsan": 0.82,
        "speedup_system1_vs_system2": 8.3,
        "avg_latency_ms": 1994.17,
        "impt_balance": 42.0,
    }
    report = evaluate(summary, cfg)

    assert report["gate_passed"] is True
    assert report["snr_score"] >= cfg["quality_gates"]["scoring"]["min_snr_score"]
    assert all(check["passed"] for check in report["checks"])


def test_phase65_gate_fails_for_bad_summary() -> None:
    cfg = _load_cfg_via_yaml()
    summary = {
        "final_state": "LEARNING",
        "ledger_chain_valid": False,
        "signed_receipts": False,
        "avg_ihsan": 0.40,
        "speedup_system1_vs_system2": 1.5,
        "avg_latency_ms": 9000.0,
        "impt_balance": -2.0,
    }
    report = evaluate(summary, cfg)

    assert report["gate_passed"] is False
    assert report["hard_fail"] is True
    assert any(check["passed"] is False for check in report["checks"])


def test_phase65_gate_supports_summary_wrapped_payload(tmp_path: Path) -> None:
    cfg = _load_cfg_via_yaml()
    payload = {
        "summary": {
            "final_state": "FLOURISHING",
            "ledger_chain_valid": True,
            "signed_receipts": True,
            "avg_ihsan": 0.81,
            "speedup_system1_vs_system2": 8.0,
            "avg_latency_ms": 2000.0,
            "impt_balance": 10.0,
        }
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(report_path.read_text(encoding="utf-8"))["summary"]

    report = evaluate(loaded, cfg)
    assert report["gate_passed"] is True


def test_phase65_gate_checks_signed_receipts_from_ledger(tmp_path: Path) -> None:
    cfg = _load_cfg_via_yaml()
    ledger = tmp_path / "ledger.jsonl"
    good_entry = {
        "seq": 1,
        "receipt": {"signature": "aa", "signer_pubkey": "bb"},
        "prev_hash": "0" * 64,
        "entry_hash": "1" * 64,
        "ts": "2026-01-01T00:00:00Z",
    }
    ledger.write_text(json.dumps(good_entry) + "\n", encoding="utf-8")
    payload = {
        "summary": {
            "final_state": "FLOURISHING",
            "ledger_chain_valid": True,
            "signed_receipts": False,
            "avg_ihsan": 0.81,
            "speedup_system1_vs_system2": 8.0,
            "avg_latency_ms": 2000.0,
            "impt_balance": 10.0,
        },
        "artifacts": {"ledger_path": str(ledger)},
    }
    report = evaluate(payload["summary"], cfg, payload=payload)
    assert report["gate_passed"] is True
