from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.ops.autonomous_engine_gate import (
    evaluate_gate,
    load_gate_config,
    run_gate,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_autonomous_engine_gate_passes_high_signal(tmp_path: Path) -> None:
    cfg_file = tmp_path / "autonomous.yaml"
    _write_yaml(
        cfg_file,
        {
            "episodes_profile": "high_signal",
            "episodes_count": 6,
            "require_compiled": True,
            "require_chain_valid": True,
        },
    )
    cfg = load_gate_config(cfg_file)
    report = run_gate(cfg)
    assert report["gate_passed"] is True
    assert report["metrics"]["compiled"] is True
    assert report["metrics"]["chain_valid"] is True
    assert report["metrics"]["score"] >= report["thresholds"]["min_score"]


def test_autonomous_engine_gate_fails_on_manual_constraints() -> None:
    cfg = load_gate_config(Path("/nonexistent.yaml"))
    prompt_artifact = {
        "snr": {"normalized": 0.35},
        "graph_of_thought": {"nodes": [], "edges": []},
        "snr_tuning_actions": ["increase specificity"],
    }
    rlvr_report = {
        "summary": {
            "snr": {"normalized": 0.20},
            "qualified_rate": 0.10,
            "compiled": False,
            "chain_valid": False,
        },
        "decision": {"action": "RECALIBRATE_POLICY"},
    }
    report = evaluate_gate(cfg, prompt_artifact, rlvr_report)
    assert report["gate_passed"] is False
    assert report["constraints"]["min_score"] is False
    assert report["constraints"]["compiled"] is False
    assert report["constraints"]["chain_valid"] is False
    assert report["autonomous_next_step"]["priority"] == "P1"


def test_autonomous_engine_gate_output_contract(tmp_path: Path) -> None:
    cfg_file = tmp_path / "autonomous.yaml"
    _write_yaml(cfg_file, {"episodes_profile": "high_signal", "episodes_count": 5})
    cfg = load_gate_config(cfg_file)
    report = run_gate(cfg)
    assert "graph_of_thought" in report
    assert "standing_on_giants_protocol" in report
    node_ids = {n["id"] for n in report["graph_of_thought"]["nodes"]}
    assert "prompt_engine" in node_ids
    assert "rlvr_loop" in node_ids
    assert "release_decision" in node_ids

    # Report is JSON serializable for artifact publishing.
    encoded = json.dumps(report)
    assert encoded
