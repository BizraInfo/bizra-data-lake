"""Regression tests for Sovereign API metrics endpoints."""

from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from core.sovereign.api import SovereignAPIServer, create_fastapi_app
from core.sovereign.runtime_types import RuntimeMetrics


def _runtime_with_metrics() -> MagicMock:
    runtime = MagicMock()
    runtime.metrics = RuntimeMetrics(
        queries_processed=10,
        queries_succeeded=8,
        current_snr_score=0.91,
        current_ihsan_score=0.96,
        avg_query_time_ms=123.4,
    )
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
    return runtime


@pytest.mark.asyncio
async def test_async_server_metrics_uses_runtime_metrics_fields() -> None:
    runtime = _runtime_with_metrics()
    server = SovereignAPIServer(runtime)
    resp = await server._handle_metrics()
    body = resp.split("\r\n\r\n", 1)[1] if "\r\n\r\n" in resp else resp

    assert body == runtime.metrics.to_prometheus(include_help=True)


def test_fastapi_metrics_uses_runtime_metrics_fields() -> None:
    runtime = _runtime_with_metrics()
    app = create_fastapi_app(runtime)
    client = TestClient(app)

    resp = client.get("/v1/metrics")
    assert resp.status_code == 200
    assert resp.text == runtime.metrics.to_prometheus(include_help=False)


def test_fastapi_health_exposes_pat_sat_receipt_chain_summary() -> None:
    runtime = _runtime_with_metrics()
    runtime.status.return_value["pat_sat"]["negotiation_receipt_chain"] = {
        "verified_end_to_end": True,
        "chain_valid": True,
        "total_negotiation_receipts": 3,
        "latest_sequence": 42,
        "latest_entry_hash": "f" * 64,
        "latest_receipt_id": "a" * 32,
    }
    app = create_fastapi_app(runtime)
    client = TestClient(app)

    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "pat_sat_receipt_chain" in body
    assert body["pat_sat_receipt_chain"]["verified_end_to_end"] is True
    assert body["pat_sat_receipt_chain"]["chain_valid"] is True
    assert body["pat_sat_receipt_chain"]["total_negotiation_receipts"] == 3
    assert body["pat_sat_receipt_chain"]["latest_sequence"] == 42
    assert body["pat_sat_receipt_chain"]["latest_entry_hash"] == "f" * 64
