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
        "health": {"status": "healthy"},
        "identity": {"version": "test"},
        "state": {"running": True},
        "autonomous": {"running": False},
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
