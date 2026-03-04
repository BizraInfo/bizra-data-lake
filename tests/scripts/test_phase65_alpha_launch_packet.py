from __future__ import annotations

import json
from pathlib import Path

from scripts.ops.phase65_alpha_launch_packet import build_alpha_launch_packet


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _base_inputs(tmp_path: Path) -> tuple[dict, dict, dict, dict, Path, Path, Path]:
    summary_payload = {
        "summary": {
            "final_state": "FLOURISHING",
            "ledger_chain_valid": True,
            "signed_receipts": True,
            "avg_ihsan": 0.82,
            "speedup_system1_vs_system2": 8.3,
            "avg_latency_ms": 1800.0,
        }
    }
    gate_payload = {"gate_passed": True, "snr_score": 0.96}
    kpi_payload = {"tier": "elite-operational"}
    cfg = {
        "quality_gates": {
            "required": {
                "final_state": "FLOURISHING",
                "ledger_chain_valid": True,
                "min_avg_ihsan": 0.75,
                "min_speedup_system1_vs_system2": 8.0,
                "max_avg_latency_ms": 2200.0,
            },
            "scoring": {"min_snr_score": 0.9},
        }
    }
    summary_path = _write_json(tmp_path / "summary.json", summary_payload)
    gate_path = _write_json(tmp_path / "gate.json", gate_payload)
    kpi_path = _write_json(tmp_path / "kpi.json", kpi_payload)
    return summary_payload, gate_payload, kpi_payload, cfg, summary_path, gate_path, kpi_path


def test_alpha_packet_conditional_go_when_manual_pending(tmp_path: Path) -> None:
    summary_payload, gate_payload, kpi_payload, cfg, summary_path, gate_path, kpi_path = _base_inputs(
        tmp_path
    )
    packet = build_alpha_launch_packet(
        summary_payload=summary_payload,
        gate_payload=gate_payload,
        kpi_payload=kpi_payload,
        cfg=cfg,
        summary_path=summary_path,
        gate_path=gate_path,
        kpi_path=kpi_path,
        alpha_users_target=100,
        require_tier="elite-operational",
        strict_manual=False,
        manual_values=None,
        signer_private_key_hex="",
    )
    assert packet["decision"] == "CONDITIONAL_GO"
    assert packet["automated"]["pass"] is True
    assert packet["manual"]["counts"]["pending"] > 0
    assert packet["signature"]["signed"] is False


def test_alpha_packet_go_with_manual_checks_and_signature(tmp_path: Path) -> None:
    summary_payload, gate_payload, kpi_payload, cfg, summary_path, gate_path, kpi_path = _base_inputs(
        tmp_path
    )
    manual = {
        "website_updated": True,
        "unified_installer_ready": True,
        "onboarding_lifecycle_ready": True,
        "urp_active": True,
        "identity_activated": True,
        "pat_minted": True,
        "sat_minted": True,
        "local_filesystem_automation_ready": True,
        "web_autonomy_ready": True,
    }
    packet = build_alpha_launch_packet(
        summary_payload=summary_payload,
        gate_payload=gate_payload,
        kpi_payload=kpi_payload,
        cfg=cfg,
        summary_path=summary_path,
        gate_path=gate_path,
        kpi_path=kpi_path,
        alpha_users_target=100,
        require_tier="elite-operational",
        strict_manual=True,
        manual_values=manual,
        signer_private_key_hex="11" * 32,
    )
    assert packet["decision"] == "GO"
    assert packet["launch_ready"] is True
    assert packet["signature"]["signed"] is True
    assert isinstance(packet["signature"]["value"], str)
    assert len(packet["signature"]["value"]) > 0


def test_alpha_packet_no_go_when_automated_fails(tmp_path: Path) -> None:
    summary_payload, gate_payload, kpi_payload, cfg, summary_path, gate_path, kpi_path = _base_inputs(
        tmp_path
    )
    gate_payload["gate_passed"] = False
    packet = build_alpha_launch_packet(
        summary_payload=summary_payload,
        gate_payload=gate_payload,
        kpi_payload=kpi_payload,
        cfg=cfg,
        summary_path=summary_path,
        gate_path=gate_path,
        kpi_path=kpi_path,
        alpha_users_target=100,
        require_tier="elite-operational",
        strict_manual=False,
        manual_values={},
        signer_private_key_hex="",
    )
    assert packet["decision"] == "NO_GO"
    assert "auto:gate_passed" in packet["blockers"]


def test_alpha_packet_no_go_when_strict_manual_and_pending(tmp_path: Path) -> None:
    summary_payload, gate_payload, kpi_payload, cfg, summary_path, gate_path, kpi_path = _base_inputs(
        tmp_path
    )
    packet = build_alpha_launch_packet(
        summary_payload=summary_payload,
        gate_payload=gate_payload,
        kpi_payload=kpi_payload,
        cfg=cfg,
        summary_path=summary_path,
        gate_path=gate_path,
        kpi_path=kpi_path,
        alpha_users_target=100,
        require_tier="elite-operational",
        strict_manual=True,
        manual_values={"website_updated": True},
        signer_private_key_hex="",
    )
    assert packet["decision"] == "NO_GO"
    assert any(b.startswith("manual_pending:") for b in packet["blockers"])
