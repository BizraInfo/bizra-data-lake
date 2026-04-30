"""Node0 readiness API tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from starlette.testclient import TestClient

from core.sovereign.api import create_fastapi_app


class _Node0Stub:
    def health(self) -> dict[str, object]:
        return {
            "booted": True,
            "node_id": "node0-test",
            "chain_hash": "a" * 64,
            "total_breaths": 2,
        }


class _AgentDBStatsStub:
    def stats(self) -> dict[str, object]:
        return {"total_records": 3}


def _runtime(state_dir: Path, *, node0: object | None = None) -> MagicMock:
    runtime = MagicMock()
    runtime.config = SimpleNamespace(state_dir=state_dir)
    runtime.status.return_value = {
        "health": {"status": "healthy", "critical_subsystems": {}},
        "identity": {"version": "test"},
        "state": {"running": True},
        "autonomous": {"running": False},
    }
    runtime._constitutional_wallets = []
    runtime._last_tick_result = None
    runtime._node0 = node0
    runtime._agent_db = _AgentDBStatsStub()
    return runtime


def _write_spearpoint_summary(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "campaign_summary.json").write_text(
        json.dumps(
            {
                "run_id": "run-123",
                "mode": "strict",
                "status": "success",
                "timestamp_utc": "2026-04-29T23:31:57Z",
                "targets": [
                    {
                        "target": "swe_bench_verified",
                        "baseline_score": 0.861901,
                        "final_score": 0.876901,
                        "gates": {
                            "reproducibility": {"passed": True},
                            "integrity": {"passed": True},
                            "budget": {"passed": True},
                            "submission": {"passed": True},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_node0_readiness_reports_green_when_booted_and_spearpoint_passed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
    monkeypatch.setenv(
        "BIZRA_USERSTORE_MASTER_SECRET",
        "test-node0-readiness-master-secret",
    )
    campaign_dir = tmp_path / "spearpoint"
    monkeypatch.setenv("BIZRA_SPEARPOINT_CAMPAIGN_DIR", str(campaign_dir))
    _write_spearpoint_summary(campaign_dir)

    app = create_fastapi_app(_runtime(tmp_path, node0=_Node0Stub()))
    client = TestClient(app)

    response = client.get("/v1/node0/readiness")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "green"
    assert data["next_action"] == "submit mission"
    assert data["product_shell"]["available"] is True
    assert data["proof_surface"]["available"] is True
    assert data["boot_service"]["status"] == "booted"
    assert data["boot_service"]["node_id"] == "node0-test"
    assert data["memory_import"]["available"] is True
    assert data["memory_import"]["status"] == "ready"
    assert data["memory_import"]["mode"] == "single_user_provided_record"
    assert data["memory_import"]["requires_consent"] is True
    assert data["voice_input"]["available"] is True
    assert data["voice_input"]["status"] == "browser_required"
    assert data["voice_input"]["mode"] == "browser_speech_recognition"
    assert data["voice_input"]["requires_user_gesture"] is True
    assert data["voice_input"]["stores_audio"] is False
    assert data["voice_input"]["auto_submit"] is False
    assert data["desktop_browser_action"]["available"] is True
    assert data["desktop_browser_action"]["status"] == "preview_only"
    assert data["desktop_browser_action"]["mode"] == "client_handoff_only"
    assert data["desktop_browser_action"]["server_executes"] is False
    assert data["desktop_browser_action"]["requires_user_confirmation"] is True
    assert data["desktop_browser_action"]["allowed_actions"] == [
        "open_url",
        "copy_text",
    ]
    assert data["spearpoint"]["status"] == "pass"
    assert data["spearpoint"]["run_id"] == "run-123"
    assert data["spearpoint"]["official_submission"] is False
    assert data["spearpoint"]["classification"] == "internal_strict_harness"


def test_node0_readiness_reports_missing_spearpoint_without_fabricating_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
    monkeypatch.setenv(
        "BIZRA_USERSTORE_MASTER_SECRET",
        "test-node0-readiness-master-secret",
    )
    monkeypatch.setenv("BIZRA_SPEARPOINT_CAMPAIGN_DIR", str(tmp_path / "missing"))

    app = create_fastapi_app(_runtime(tmp_path, node0=None))
    client = TestClient(app)

    response = client.get("/v1/node0/readiness")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "yellow"
    assert data["next_action"] == "start Node0 boot service"
    assert data["boot_service"]["status"] == "unavailable"
    assert data["spearpoint"]["status"] == "unknown"
    assert data["spearpoint"]["artifact_status"] == "missing"
