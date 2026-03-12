from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import canonical_empirical_validation as cev


def _proof_plane(label: str, passed: bool) -> dict[str, object]:
    return {
        "label": label,
        "targets": [f"tests/{label}.py"],
        "pytest_args": ["-q", f"tests/{label}.py"],
        "exit_code": 0 if passed else 1,
        "passed": passed,
    }


def test_build_report_yields_canonical_status_on_full_pass() -> None:
    cfg = cev.load_config(Path("config/canonical_empirical_validation.json"))
    empirical_report = {
        "passed": 10,
        "failed": 0,
        "total": 10,
        "pass_rate": 1.0,
        "proof_hash": "abc123",
        "results_file": "/tmp/validation_results.json",
        "raw_data_file": "/tmp/raw_data.json",
    }
    proof_planes = {
        "flagship_metabolism": _proof_plane("flagship_metabolism", True),
        "receipt_contract": _proof_plane("receipt_contract", True),
        "sovereignty_pipeline": _proof_plane("sovereignty_pipeline", True),
    }

    report = cev.build_report(cfg, empirical_report, proof_planes)

    assert report["gate_passed"] is True
    assert report["canonical_status"] == "CANONICAL"
    assert report["metrics"]["score"] >= cfg.min_score
    assert report["constraints"]["empirical_suite"] is True
    assert report["constraints"]["flagship_metabolism"] is True
    assert report["constraints"]["receipt_contract"] is True
    assert report["constraints"]["sovereignty_pipeline"] is True


def test_build_report_fails_when_required_plane_fails() -> None:
    cfg = cev.load_config(Path("config/canonical_empirical_validation.json"))
    empirical_report = {
        "passed": 10,
        "failed": 0,
        "total": 10,
        "pass_rate": 1.0,
        "proof_hash": "abc123",
        "results_file": "/tmp/validation_results.json",
        "raw_data_file": "/tmp/raw_data.json",
    }
    proof_planes = {
        "flagship_metabolism": _proof_plane("flagship_metabolism", True),
        "receipt_contract": _proof_plane("receipt_contract", False),
        "sovereignty_pipeline": _proof_plane("sovereignty_pipeline", True),
    }

    report = cev.build_report(cfg, empirical_report, proof_planes)

    assert report["gate_passed"] is False
    assert report["canonical_status"] == "DEGRADED"
    assert report["constraints"]["receipt_contract"] is False
    assert report["autonomous_next_step"]["priority"] == "P1"


def test_run_canonical_empirical_validation_writes_reports(
    monkeypatch, tmp_path: Path
) -> None:
    config = cev.load_config(Path("config/canonical_empirical_validation.json"))

    monkeypatch.setattr(
        cev,
        "load_config",
        lambda _: cev.CanonicalEmpiricalConfig(
            empirical_results_dir=tmp_path / "empirical",
            native_proof_planes=config.native_proof_planes,
            pytest_targets=config.pytest_targets,
            score_weights=config.score_weights,
            min_score=config.min_score,
            min_empirical_pass_rate=config.min_empirical_pass_rate,
            required_proof_planes=config.required_proof_planes,
            giants_protocol=config.giants_protocol,
            program=config.program,
        ),
    )
    monkeypatch.setattr(
        cev,
        "_run_empirical_suite",
        lambda _: {
            "passed": 10,
            "failed": 0,
            "total": 10,
            "pass_rate": 1.0,
            "proof_hash": "abc123",
            "results_file": str(tmp_path / "empirical" / "validation_results.json"),
            "raw_data_file": str(tmp_path / "empirical" / "raw_data.json"),
        },
    )
    monkeypatch.setattr(
        cev,
        "_run_pytest_targets",
        lambda label, targets: _proof_plane(label, True),
    )
    monkeypatch.setattr(
        cev,
        "_run_native_proof",
        lambda label: _proof_plane(label, True),
    )

    report_path = tmp_path / "canonical_empirical_validation.json"
    markdown_path = tmp_path / "canonical_empirical_validation.md"
    github_output = tmp_path / "github_output.txt"

    report = cev.run_canonical_empirical_validation(
        config_path=Path("config/canonical_empirical_validation.json"),
        report_path=report_path,
        markdown_report_path=markdown_path,
        github_output=github_output,
    )

    assert report["gate_passed"] is True
    assert report_path.exists()
    assert markdown_path.exists()
    assert github_output.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["canonical_status"] == "CANONICAL"
    assert "Composite Score" in markdown_path.read_text(encoding="utf-8")
    outputs = github_output.read_text(encoding="utf-8")
    assert "canonical_empirical_passed=true" in outputs
