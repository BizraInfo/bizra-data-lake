"""API exposure contract tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from core.sovereign.api import create_fastapi_app
from core.sovereign.api_exposure_policy import (
    RouteExposure,
    get_api_route_policy,
    summarize_api_exposure,
    validate_api_exposure_policy,
)


def _runtime(state_dir: Path) -> MagicMock:
    runtime = MagicMock()
    runtime.config = SimpleNamespace(state_dir=state_dir)
    runtime.metrics = MagicMock(to_prometheus=lambda include_help=False: "")
    runtime.status.return_value = {
        "health": {
            "status": "healthy",
            "strict_gate": {"enabled": False, "passed": True},
        },
        "identity": {"version": "test"},
        "state": {"running": True},
        "autonomous": {"running": False},
        "pat_sat": {
            "negotiation_receipt_chain": {
                "verified_end_to_end": False,
                "chain_valid": None,
                "total_negotiation_receipts": 0,
                "latest_sequence": None,
                "latest_entry_hash": None,
                "latest_receipt_id": None,
            }
        },
    }
    runtime.query = AsyncMock(
        return_value=SimpleNamespace(
            query_id="q-test",
            success=True,
            response="ok",
            snr_score=0.9,
            ihsan_score=0.9,
            processing_time_ms=12.0,
            graph_hash=None,
        )
    )
    runtime._orchestrator = None
    runtime._node_signer = None
    runtime._evidence_ledger = None
    return runtime


def test_api_exposure_policy_covers_every_live_v1_route(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "BIZRA_USERSTORE_MASTER_SECRET",
        "test-api-exposure-contract-master-secret",
    )
    app = create_fastapi_app(_runtime(tmp_path))

    report = validate_api_exposure_policy(app)

    assert report.ok, report.format_issues()


def test_selected_route_exposure_decisions_remain_stable() -> None:
    cases = [
        ("/v1/query", "POST", RouteExposure.AUTHENTICATED),
        ("/v1/stream", "WEBSOCKET", RouteExposure.AUTHENTICATED),
        ("/v1/memory/stats", "GET", RouteExposure.AUTHENTICATED),
        ("/v1/auth/register", "POST", RouteExposure.BOOTSTRAP_PUBLIC),
        ("/v1/status", "GET", RouteExposure.PUBLIC),
        ("/v1/verify/receipt", "POST", RouteExposure.PUBLIC),
        ("/v1/token/verify", "GET", RouteExposure.PUBLIC),
    ]

    for path, verb, expected in cases:
        policy = get_api_route_policy(path, verb)
        assert (
            policy.exposure == expected
        ), f"{verb} {path} exposure changed unexpectedly"


def test_api_exposure_summary_is_fully_accounted_for(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "BIZRA_USERSTORE_MASTER_SECRET",
        "test-api-exposure-contract-master-secret",
    )
    app = create_fastapi_app(_runtime(tmp_path))

    summary = summarize_api_exposure(app)

    assert sum(summary.values()) == 63
    assert summary[RouteExposure.PUBLIC] == 24
    assert summary[RouteExposure.BOOTSTRAP_PUBLIC] == 3
    assert summary[RouteExposure.AUTHENTICATED] == 36
