"""Fail-closed auth tests for /v1/query."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.responses import JSONResponse

from core.sovereign.api import (
    OrchestrateRequestModel,
    QueryRequestModel,
    ReceiptVerifyModel,
    create_fastapi_app,
)


def _runtime(tmp_path) -> MagicMock:
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
    runtime.compute_poi_epoch = MagicMock(return_value={"ok": True})
    runtime.finalize_sat_epoch = MagicMock(return_value={"ok": True})
    runtime._orchestrator = None
    runtime._node_signer = None
    runtime._evidence_ledger = None
    return runtime


def _query_endpoint(app):
    route = next(r for r in app.routes if getattr(r, "path", "") == "/v1/query")
    return route.endpoint


def _endpoint(app, path: str, method: str = "POST"):
    route = next(
        r
        for r in app.routes
        if getattr(r, "path", "") == path and method in getattr(r, "methods", set())
    )
    return route.endpoint


def _websocket_endpoint(app):
    route = next(r for r in app.routes if getattr(r, "path", "") == "/v1/stream")
    return route.endpoint


class _Request:
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}


@pytest.mark.asyncio
async def test_query_denies_missing_credentials_by_default(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("BIZRA_AUTH_ALLOW_ANONYMOUS", raising=False)
    runtime = _runtime(tmp_path)
    app = create_fastapi_app(runtime)
    endpoint = _query_endpoint(app)

    resp = await endpoint(QueryRequestModel(query="hello"), _Request())

    # 401: auth layer online but credentials missing
    # 503: auth layer unavailable at startup
    assert isinstance(resp, JSONResponse)
    assert resp.status_code in {401, 503}
    runtime.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_allows_anonymous_only_when_opted_in(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
    runtime = _runtime(tmp_path)
    app = create_fastapi_app(runtime)
    endpoint = _query_endpoint(app)

    resp = await endpoint(QueryRequestModel(query="hello"), _Request())

    assert isinstance(resp, dict)
    runtime.query.assert_awaited_once()
    assert runtime.query.await_args.kwargs["user_id"] == ""
    assert "user_id" not in resp


@pytest.mark.asyncio
async def test_query_returns_503_when_auth_layer_boot_fails(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("BIZRA_AUTH_ALLOW_ANONYMOUS", raising=False)

    import core.auth.jwt_auth as jwt_auth_module

    class BrokenJWTAuth:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("forced auth init failure")

    monkeypatch.setattr(jwt_auth_module, "JWTAuth", BrokenJWTAuth)

    runtime = _runtime(tmp_path)
    app = create_fastapi_app(runtime)
    endpoint = _query_endpoint(app)

    resp = await endpoint(QueryRequestModel(query="hello"), _Request())

    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 503
    assert (
        json.loads(resp.body.decode("utf-8"))["error"]
        == "Authentication service unavailable"
    )
    runtime.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_mutating_epochs_deny_missing_credentials_by_default(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("BIZRA_AUTH_ALLOW_ANONYMOUS", raising=False)
    runtime = _runtime(tmp_path)
    app = create_fastapi_app(runtime)

    poi_endpoint = _endpoint(app, "/v1/poi/epoch")
    sat_endpoint = _endpoint(app, "/v1/sat/epoch")

    poi_resp = await poi_endpoint(_Request())
    sat_resp = await sat_endpoint(_Request())

    assert isinstance(poi_resp, JSONResponse)
    assert isinstance(sat_resp, JSONResponse)
    assert poi_resp.status_code in {401, 503}
    assert sat_resp.status_code in {401, 503}
    runtime.compute_poi_epoch.assert_not_called()
    runtime.finalize_sat_epoch.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrate_denies_missing_credentials_by_default(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("BIZRA_AUTH_ALLOW_ANONYMOUS", raising=False)
    runtime = _runtime(tmp_path)
    app = create_fastapi_app(runtime)

    endpoint = _endpoint(app, "/v1/orchestrate")
    resp = await endpoint(OrchestrateRequestModel(task="draft plan"), _Request())

    assert isinstance(resp, JSONResponse)
    assert resp.status_code in {401, 503}


@pytest.mark.asyncio
async def test_websocket_denies_missing_credentials_by_default(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("BIZRA_AUTH_ALLOW_ANONYMOUS", raising=False)
    runtime = _runtime(tmp_path)
    app = create_fastapi_app(runtime)
    endpoint = _websocket_endpoint(app)

    class _FakeWebSocket:
        def __init__(self):
            self.headers = {}
            self.accepted = False
            self.closed: tuple[int, str] | None = None

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = (code, reason)

        async def accept(self) -> None:
            self.accepted = True

        async def send_json(self, _: dict[str, object]) -> None:
            pass

        async def receive_json(self) -> dict[str, object]:
            raise RuntimeError("ws should be rejected before receive loop")

    ws = _FakeWebSocket()
    await endpoint(ws)

    assert ws.accepted is False
    assert ws.closed is not None
    assert ws.closed[0] in {1013, 4401}


@pytest.mark.asyncio
async def test_verify_receipt_rejects_tampered_signature(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
    runtime = _runtime(tmp_path)

    from core.proof_engine.receipt import Receipt, ReceiptStatus, SimpleSigner

    signer = SimpleSigner(b"test-secret")
    runtime._node_signer = signer

    receipt = Receipt(
        receipt_id="rcpt-test",
        status=ReceiptStatus.ACCEPTED,
        query_digest=b"\x11" * 32,
        policy_digest=b"\x22" * 32,
        payload_digest=b"\x33" * 32,
        snr=0.9,
        ihsan_score=0.95,
        gate_passed="commit",
    ).sign_with(signer)

    app = create_fastapi_app(runtime)
    endpoint = _endpoint(app, "/v1/verify/receipt")

    good = await endpoint(ReceiptVerifyModel(receipt=receipt.to_dict()))
    assert isinstance(good, dict)
    assert good["decision"] == "APPROVED"

    tampered = receipt.to_dict()
    tampered["query_digest"] = "aa" * 32
    bad = await endpoint(ReceiptVerifyModel(receipt=tampered))
    assert isinstance(bad, dict)
    assert bad["decision"] == "REJECTED"
    assert "SIGNATURE_INVALID" in bad["reason_codes"]
