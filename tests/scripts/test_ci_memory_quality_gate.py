from __future__ import annotations

from pathlib import Path

from scripts.ci_memory_quality_gate import GateThresholds, run_memory_quality_gate


def test_memory_quality_gate_passes_and_writes_report(tmp_path: Path) -> None:
    report_path = tmp_path / "artifacts" / "memory_quality_gate.json"

    exit_code, report = run_memory_quality_gate(report_out=report_path)

    assert exit_code == 0
    assert report["gate"]["passed"] is True
    assert report["convergence"]["migration"]["total_imported"] >= 5
    assert report["stats"]["index_health"]["status"] == "healthy"
    assert report["rebuild"]["indexed_vectors"] >= 8
    assert report["search_latency"]["result_count"] > 0
    assert report_path.exists()


def test_memory_quality_gate_fails_when_import_floor_is_impossible(
    tmp_path: Path,
) -> None:
    thresholds = GateThresholds(min_imported_records=999)

    exit_code, report = run_memory_quality_gate(
        report_out=tmp_path / "artifacts" / "memory_quality_gate.json",
        thresholds=thresholds,
    )

    assert exit_code == 1
    assert report["gate"]["passed"] is False
    assert any(
        reason.startswith("imported_records=")
        for reason in report["gate"]["reasons"]
    )
