"""Typed error-taxonomy coverage for sovereign query boundaries."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from core.errors import BridgeError, InferenceError
from core.sovereign.api import (
    QueryRequestModel,
    SovereignAPIServer,
    create_fastapi_app,
)


def _runtime(tmp_path: Path) -> MagicMock:
    runtime = MagicMock()
    runtime.config = SimpleNamespace(state_dir=tmp_path / "state")
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
    runtime.query = AsyncMock()
    runtime._evidence_ledger = None
    return runtime


def _parse_asyncio_response(response: str) -> tuple[int, dict[str, object]]:
    head, body = response.split("\r\n\r\n", 1)
    status = int(head.split()[1])
    return status, json.loads(body)


def _query_endpoint(app):
    route = next(r for r in app.routes if getattr(r, "path", "") == "/v1/query")
    return route.endpoint


class _URL:
    def __init__(self, path: str = "/v1/query"):
        self.path = path


class _Request:
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}
        self.url = _URL()


@pytest.mark.asyncio
async def test_async_query_boundary_returns_typed_bridge_receipt(
    tmp_path: Path,
) -> None:
    from core.node0.heartbeat import Node0Heartbeat

    runtime = _runtime(tmp_path)
    runtime._node0 = Node0Heartbeat(
        data_dir=tmp_path / "node0-async", node_id="api-raw"
    )
    runtime._node0.boot()
    runtime.query.side_effect = BridgeError(
        "got_bridge",
        "offline",
        context={"component": "test"},
    )
    server = SovereignAPIServer(runtime=runtime)

    status, body = _parse_asyncio_response(
        await server._handle_query(json.dumps({"query": "hello"}).encode("utf-8"))
    )

    assert status == 503
    assert body["error_type"] == "BridgeError"
    assert body["boundary"] == "BRIDGE"
    assert body["severity"] == "DEGRADE"
    assert body["retryable"] is True
    assert body["correlation_id"].startswith("api-")
    assert "trace" not in body
    assert runtime._node0.health()["total_boundary_error_receipts"] == 1

    path = (
        tmp_path / "node0-async" / "audit" / "canonical_boundary_error_receipts.jsonl"
    )
    assert path.exists()


@pytest.mark.asyncio
async def test_async_query_boundary_wraps_legacy_exception(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.query.side_effect = RuntimeError("boom")
    server = SovereignAPIServer(runtime=runtime)

    status, body = _parse_asyncio_response(
        await server._handle_query(json.dumps({"query": "hello"}).encode("utf-8"))
    )

    assert status == 500
    assert body["error_type"] == "MembraneError"
    assert body["boundary"] == "MEMBRANE"
    assert body["context"]["route"] == "/v1/query"
    assert body["message"] == "Internal server error"
    assert body["correlation_id"].startswith("api-")
    assert "trace" not in body


@pytest.mark.asyncio
async def test_async_query_boundary_logs_trace_without_exposing_client_trace(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = _runtime(tmp_path)
    runtime.query.side_effect = RuntimeError("internal stack detail")
    server = SovereignAPIServer(runtime=runtime)

    with caplog.at_level(logging.ERROR, logger="sovereign.api"):
        status, body = _parse_asyncio_response(
            await server._handle_query(json.dumps({"query": "hello"}).encode("utf-8"))
        )

    assert status == 500
    assert body["message"] == "Internal server error"
    assert "internal stack detail" not in json.dumps(body)
    assert "trace" not in body
    assert any("correlation_id=api-" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_async_query_validation_error_is_sanitized(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    server = SovereignAPIServer(runtime=runtime)

    status, body = _parse_asyncio_response(
        await server._handle_query(json.dumps({"query": ""}).encode("utf-8"))
    )

    assert status == 400
    assert body["error_type"] == "ValidationBoundaryError"
    assert body["message"] == "Query required"
    assert body["retryable"] is False
    assert body["correlation_id"].startswith("api-")
    assert "trace" not in body
    runtime.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_query_schema_type_error_is_validation_boundary(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    server = SovereignAPIServer(runtime=runtime)

    status, body = _parse_asyncio_response(
        await server._handle_query(
            json.dumps({"query": "hello", "context": "not-a-dict"}).encode("utf-8")
        )
    )

    assert status == 400
    assert body["error_type"] == "ValidationBoundaryError"
    assert body["message"] == "context must be an object"
    assert body["correlation_id"].startswith("api-")
    assert "trace" not in body
    runtime.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_fastapi_query_boundary_returns_typed_inference_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from core.node0.heartbeat import Node0Heartbeat

    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
    runtime = _runtime(tmp_path)
    runtime._node0 = Node0Heartbeat(
        data_dir=tmp_path / "node0-fastapi",
        node_id="api-fastapi",
    )
    runtime._node0.boot()
    runtime.query.side_effect = InferenceError("planner", "timeout")
    app = create_fastapi_app(runtime)
    endpoint = _query_endpoint(app)

    response = await endpoint(QueryRequestModel(query="hello"), _Request())

    assert isinstance(response, JSONResponse)
    assert response.status_code == 502
    body = json.loads(response.body.decode("utf-8"))
    assert body["error_type"] == "InferenceError"
    assert body["boundary"] == "INFERENCE"
    assert body["severity"] == "RETRY"
    assert body["retryable"] is True
    assert body["correlation_id"].startswith("api-")
    assert "trace" not in body
    assert runtime._node0.health()["total_boundary_error_receipts"] == 1


@pytest.mark.asyncio
async def test_fastapi_query_boundary_sanitizes_unknown_exception(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
    runtime = _runtime(tmp_path)
    runtime.query.side_effect = OSError("database path leaked")
    app = create_fastapi_app(runtime)
    endpoint = _query_endpoint(app)

    response = await endpoint(QueryRequestModel(query="hello"), _Request())

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    body = json.loads(response.body.decode("utf-8"))
    assert body["error_type"] == "MembraneError"
    assert body["message"] == "Internal server error"
    assert body["correlation_id"].startswith("api-")
    assert "database path leaked" not in json.dumps(body)
    assert "trace" not in body


@pytest.mark.asyncio
async def test_fastapi_query_validation_error_is_sanitized(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
    runtime = _runtime(tmp_path)
    app = create_fastapi_app(runtime)
    endpoint = _query_endpoint(app)

    response = await endpoint(QueryRequestModel(query=""), _Request())

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    body = json.loads(response.body.decode("utf-8"))
    assert body["error_type"] == "ValidationBoundaryError"
    assert body["message"] == "Query required"
    assert body["retryable"] is False
    assert body["correlation_id"].startswith("api-")
    assert "trace" not in body
    runtime.query.assert_not_awaited()


def test_fastapi_framework_validation_error_uses_boundary_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
    runtime = _runtime(tmp_path)
    app = create_fastapi_app(runtime)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/v1/query", json={})

    assert response.status_code == 422
    body = response.json()
    assert body["error_type"] == "ValidationBoundaryError"
    assert body["message"] == "Request validation failed"
    assert body["correlation_id"].startswith("api-")
    assert "detail" not in body
    assert "trace" not in body
    runtime.query.assert_not_awaited()
