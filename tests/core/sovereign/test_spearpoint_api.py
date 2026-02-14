"""
Spearpoint API Integration Tests — Phase 20 Last-Mile Wiring
=============================================================

Tests that the /v1/spearpoint/* endpoints correctly route to the
SpearpointOrchestrator wired into runtime_core.py.

Standing on Giants: Boyd (OODA), Goldratt (Theory of Constraints)
"""

from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from core.sovereign.api import create_fastapi_app
from core.sovereign.runtime_types import RuntimeMetrics


def _mock_runtime(with_orchestrator: bool = True) -> MagicMock:
    """Create a mock runtime with optional spearpoint orchestrator."""
    runtime = MagicMock()
    runtime.metrics = RuntimeMetrics(
        queries_processed=0,
        queries_succeeded=0,
        current_snr_score=0.95,
        current_ihsan_score=0.96,
        avg_query_time_ms=50.0,
    )
    runtime.status.return_value = {
        "health": {"status": "healthy"},
        "identity": {"version": "test"},
        "state": {"running": True},
        "autonomous": {"running": False},
    }

    if with_orchestrator:
        from core.spearpoint.config import MissionType
        from core.spearpoint.orchestrator import MissionResult

        # reproduce returns MissionResult
        reproduce_result = MissionResult(
            mission_id="mission_test123",
            mission_type=MissionType.REPRODUCE,
            success=True,
            evaluation_results=[{"verdict": "ACCEPTED", "score": 0.97}],
            elapsed_ms=42.5,
        )
        runtime._spearpoint_orchestrator = MagicMock()
        runtime._spearpoint_orchestrator.reproduce.return_value = reproduce_result

        # improve returns MissionResult
        improve_result = MissionResult(
            mission_id="mission_improve456",
            mission_type=MissionType.IMPROVE,
            success=True,
            research_results=[{"hypothesis": "H1", "outcome": "approved"}],
            elapsed_ms=120.0,
        )
        runtime._spearpoint_orchestrator.improve.return_value = improve_result

        # stats
        runtime._spearpoint_orchestrator.get_statistics.return_value = {
            "total_missions": 5,
            "successful_missions": 4,
        }
        runtime._spearpoint_orchestrator.get_mission_history.return_value = [
            {"mission_id": "m1", "success": True}
        ]
    else:
        runtime._spearpoint_orchestrator = None

    return runtime


class TestSpearpointReproduce:
    """POST /v1/spearpoint/reproduce"""

    def test_reproduce_returns_200(self):
        """Reproduce endpoint returns result from orchestrator."""
        runtime = _mock_runtime(with_orchestrator=True)
        app = create_fastapi_app(runtime)
        client = TestClient(app)

        resp = client.post(
            "/v1/spearpoint/reproduce",
            json={"claim": "System latency < 100ms"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mission_id"] == "mission_test123"
        assert data["success"] is True
        assert data["mission_type"] == "reproduce"

    def test_reproduce_passes_all_fields(self):
        """All request fields are forwarded to orchestrator.reproduce()."""
        runtime = _mock_runtime(with_orchestrator=True)
        app = create_fastapi_app(runtime)
        client = TestClient(app)

        client.post(
            "/v1/spearpoint/reproduce",
            json={
                "claim": "BLAKE3 is faster",
                "proposed_change": "Switch hashing",
                "prompt": "test prompt",
                "response": "test response",
                "metrics": {"latency_ms": 42},
            },
        )

        runtime._spearpoint_orchestrator.reproduce.assert_called_once_with(
            claim="BLAKE3 is faster",
            proposed_change="Switch hashing",
            prompt="test prompt",
            response="test response",
            metrics={"latency_ms": 42},
        )

    def test_reproduce_503_when_orchestrator_missing(self):
        """Returns 503 when orchestrator not wired."""
        runtime = _mock_runtime(with_orchestrator=False)
        app = create_fastapi_app(runtime)
        client = TestClient(app)

        resp = client.post(
            "/v1/spearpoint/reproduce",
            json={"claim": "anything"},
        )
        assert resp.status_code == 503
        assert "not available" in resp.json()["error"]


class TestSpearpointImprove:
    """POST /v1/spearpoint/improve"""

    def test_improve_returns_200(self):
        """Improve endpoint returns result from orchestrator."""
        runtime = _mock_runtime(with_orchestrator=True)
        app = create_fastapi_app(runtime)
        client = TestClient(app)

        resp = client.post(
            "/v1/spearpoint/improve",
            json={"observation": {"metric": "latency", "value": 150}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mission_id"] == "mission_improve456"
        assert data["success"] is True
        assert data["mission_type"] == "improve"

    def test_improve_with_defaults(self):
        """Improve works with empty body (all defaults)."""
        runtime = _mock_runtime(with_orchestrator=True)
        app = create_fastapi_app(runtime)
        client = TestClient(app)

        resp = client.post("/v1/spearpoint/improve", json={})
        assert resp.status_code == 200

        runtime._spearpoint_orchestrator.improve.assert_called_once_with(
            observation=None,
            top_k=3,
        )

    def test_improve_503_when_orchestrator_missing(self):
        """Returns 503 when orchestrator not wired."""
        runtime = _mock_runtime(with_orchestrator=False)
        app = create_fastapi_app(runtime)
        client = TestClient(app)

        resp = client.post("/v1/spearpoint/improve", json={})
        assert resp.status_code == 503


class TestSpearpointStats:
    """GET /v1/spearpoint/stats"""

    def test_stats_returns_200(self):
        """Stats endpoint returns orchestrator statistics."""
        runtime = _mock_runtime(with_orchestrator=True)
        app = create_fastapi_app(runtime)
        client = TestClient(app)

        resp = client.get("/v1/spearpoint/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["statistics"]["total_missions"] == 5
        assert data["statistics"]["successful_missions"] == 4
        assert len(data["recent_missions"]) == 1

    def test_stats_503_when_orchestrator_missing(self):
        """Returns 503 when orchestrator not wired."""
        runtime = _mock_runtime(with_orchestrator=False)
        app = create_fastapi_app(runtime)
        client = TestClient(app)

        resp = client.get("/v1/spearpoint/stats")
        assert resp.status_code == 503


class TestRuntimeCoreWiring:
    """Verify SpearpointOrchestrator is wired into runtime_core.py."""

    def test_runtime_has_spearpoint_orchestrator_attribute(self):
        """runtime_core.py declares _spearpoint_orchestrator attribute."""
        import inspect
        from core.sovereign.runtime_core import SovereignRuntime

        source = inspect.getsource(SovereignRuntime.__init__)
        assert "_spearpoint_orchestrator" in source

    def test_runtime_has_init_spearpoint_orchestrator_method(self):
        """runtime_core.py has _init_spearpoint_orchestrator method."""
        from core.sovereign.runtime_core import SovereignRuntime

        assert hasattr(SovereignRuntime, "_init_spearpoint_orchestrator")
        assert callable(getattr(SovereignRuntime, "_init_spearpoint_orchestrator"))
