from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import runtime_canon_lock_gate as rclg


def test_build_report_passes_on_repo_truth() -> None:
    cfg = rclg.load_config(Path("config/runtime_canon_lock_gate.json"))

    report = rclg.build_report(cfg)

    assert report["gate_passed"] is True
    assert report["status"] == "LOCKED"
    assert report["score"] == 1.0
    assert report["metrics"]["checks_total"] >= 6
    assert report["receipt"]["receipt_id"].startswith("rclg-")


def test_build_report_fails_when_required_pattern_missing(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "core/sovereign").mkdir(parents=True, exist_ok=True)
    (repo_root / "tests/integration").mkdir(parents=True, exist_ok=True)
    (repo_root / "tests/core/sovereign").mkdir(parents=True, exist_ok=True)

    (repo_root / "core/sovereign/api.py").write_text(
        "runtime_receipt = await runtime_mission(description)\n",
        encoding="utf-8",
    )
    (repo_root / "core/sovereign/__main__.py").write_text(
        "receipt = await runtime.mission(description, source=source, context={})\n",
        encoding="utf-8",
    )
    (repo_root / "tests/integration/test_plan_endpoint.py").write_text(
        "def test_plan_canonical_mode_never_calls_mission_orchestrator():\n    pass\n",
        encoding="utf-8",
    )
    (repo_root / "tests/core/sovereign/test_main_cli.py").write_text(
        "def test_run_mission_uses_runtime_mission_and_emits_canonical_fields():\n    pass\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "runtime_canon_lock_gate.json"
    config_path.write_text(
        json.dumps(
            {
                "program": {"id": "runtime_canon_lock_gate"},
                "repo_root": str(repo_root),
                "files": {
                    "api": "core/sovereign/api.py",
                    "cli": "core/sovereign/__main__.py",
                    "plan_tests": "tests/integration/test_plan_endpoint.py",
                    "cli_tests": "tests/core/sovereign/test_main_cli.py",
                },
                "checks": {
                    "api_canonical_runtime_authority": [
                        "if canonical_mode_enabled and not runtime_has_canonical_authority:"
                    ],
                    "api_noncanonical_shim_explicit": [
                        "if not runtime_has_canonical_authority:"
                    ],
                    "api_runtime_reflex_lineage": [
                        "def _runtime_reflex_lineage_payload("
                    ],
                    "cli_runtime_authority": [
                        "receipt = await runtime.mission(description, source=source, context={})"
                    ],
                    "plan_tests_cover_canonical_authority": [
                        "test_plan_canonical_mode_system1_cache_hit_flows_through_runtime"
                    ],
                    "cli_tests_cover_runtime_authority": [
                        "test_run_mission_uses_runtime_mission_and_emits_canonical_fields"
                    ],
                },
                "thresholds": {"min_score": 1.0},
                "giants_protocol": [],
            }
        ),
        encoding="utf-8",
    )

    cfg = rclg.load_config(config_path)
    report = rclg.build_report(cfg)

    assert report["gate_passed"] is False
    assert report["status"] == "DEGRADED"
    failed = [item for item in report["checks"] if not item["passed"]]
    assert failed


def test_run_runtime_canon_lock_gate_writes_reports(tmp_path: Path) -> None:
    report_path = tmp_path / "runtime_canon_lock_gate.json"
    markdown_path = tmp_path / "runtime_canon_lock_gate.md"
    github_output = tmp_path / "github_output.txt"

    report = rclg.run_runtime_canon_lock_gate(
        config_path=Path("config/runtime_canon_lock_gate.json"),
        report_path=report_path,
        markdown_report_path=markdown_path,
        github_output=github_output,
    )

    assert report["gate_passed"] is True
    assert report["status"] == "LOCKED"
    assert report_path.exists()
    assert markdown_path.exists()
    assert github_output.exists()
    assert "runtime_canon_lock_passed=true" in github_output.read_text(encoding="utf-8")
