from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import learning_loop_closure_gate as llcg


def test_build_report_passes_when_loop_closes() -> None:
    cfg = llcg.load_config(Path("config/learning_loop_closure_gate.json"))
    scenario = {
        "training_executed": True,
        "training_result": {
            "final_ihsan_score": 0.99,
            "final_loss": 0.05,
            "total_steps": 3,
        },
        "eligible_candidates": [{"pattern_id": "abc", "avg_snr": 0.92}],
        "compiled_candidates": [{"pattern_id": "abc", "avg_snr": 0.92}],
        "event_types": [
            "CANDIDATE_ACCEPTED",
            "TRAINING_COMPLETED",
            "REFLEX_COMPILED",
        ],
        "metrics": {
            "candidates_accepted": 3,
            "total_observations": 3,
            "training_runs": 1,
            "avg_training_ihsan": 0.99,
            "reflexes_compiled": 1,
            "reflex_cache_size": 1,
            "avg_candidate_snr": 0.92,
        },
    }

    report = llcg.build_report(cfg, scenario)

    assert report["gate_passed"] is True
    assert report["closure_status"] == "CLOSED"
    assert report["constraints"]["compiled_reflexes"] is True
    assert report["metrics"]["compiled_reflexes"] == 1
    assert report["receipt"]["receipt_id"].startswith("llcg-")


def test_build_report_fails_when_reflex_not_compiled() -> None:
    cfg = llcg.load_config(Path("config/learning_loop_closure_gate.json"))
    scenario = {
        "training_executed": True,
        "training_result": {
            "final_ihsan_score": 0.99,
            "final_loss": 0.05,
            "total_steps": 3,
        },
        "eligible_candidates": [{"pattern_id": "abc", "avg_snr": 0.92}],
        "compiled_candidates": [],
        "event_types": ["CANDIDATE_ACCEPTED", "TRAINING_COMPLETED"],
        "metrics": {
            "candidates_accepted": 3,
            "total_observations": 3,
            "training_runs": 1,
            "avg_training_ihsan": 0.99,
            "reflexes_compiled": 0,
            "reflex_cache_size": 0,
            "avg_candidate_snr": 0.92,
        },
    }

    report = llcg.build_report(cfg, scenario)

    assert report["gate_passed"] is False
    assert report["closure_status"] == "DEGRADED"
    assert report["constraints"]["compiled_reflexes"] is False
    assert report["constraints"]["required_events"] is False


def test_run_learning_loop_closure_gate_writes_reports(tmp_path: Path) -> None:
    report_path = tmp_path / "learning_loop_closure_gate.json"
    markdown_path = tmp_path / "learning_loop_closure_gate.md"
    github_output = tmp_path / "github_output.txt"

    report = llcg.run_learning_loop_closure_gate(
        config_path=Path("config/learning_loop_closure_gate.json"),
        report_path=report_path,
        markdown_report_path=markdown_path,
        github_output=github_output,
    )

    assert report["gate_passed"] is True
    assert report["closure_status"] == "CLOSED"
    assert report_path.exists()
    assert markdown_path.exists()
    assert github_output.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["compiled_reflexes"] >= 1
    assert "Closure status" in markdown_path.read_text(encoding="utf-8")
    outputs = github_output.read_text(encoding="utf-8")
    assert "learning_loop_closure_passed=true" in outputs
