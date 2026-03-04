from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tools.mcp import mcp_gateway


class _DummyRequest:
    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        host: str = "127.0.0.1",
        method: str = "POST",
        path: str = "/mcp",
    ) -> None:
        self.headers = headers or {}
        self.client = SimpleNamespace(host=host)
        self.method = method
        self.url = SimpleNamespace(path=path)


def test_extract_request_token_uses_bearer_then_api_key() -> None:
    req = _DummyRequest(
        headers={"authorization": "Bearer token-1", "x-api-key": "token-2"}
    )
    assert mcp_gateway._extract_request_token(req) == "token-1"

    req = _DummyRequest(headers={"x-api-key": "token-2"})
    assert mcp_gateway._extract_request_token(req) == "token-2"


def test_authorize_request_requires_configured_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BIZRA_MCP_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("BIZRA_BRIDGE_TOKEN", raising=False)
    monkeypatch.delenv("BIZRA_MCP_ALLOW_ANONYMOUS", raising=False)
    monkeypatch.delenv("BIZRA_MCP_ALLOW_REMOTE", raising=False)

    req = _DummyRequest()
    with pytest.raises(HTTPException, match="not configured") as exc:
        mcp_gateway._authorize_request(req)
    assert exc.value.status_code == 503


def test_authorize_request_rejects_remote_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BIZRA_MCP_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.delenv("BIZRA_MCP_ALLOW_ANONYMOUS", raising=False)
    monkeypatch.delenv("BIZRA_MCP_ALLOW_REMOTE", raising=False)

    req = _DummyRequest(
        host="10.1.2.3",
        headers={"authorization": "Bearer gateway-secret"},
    )
    with pytest.raises(HTTPException, match="Remote MCP access denied") as exc:
        mcp_gateway._authorize_request(req)
    assert exc.value.status_code == 403


def test_authorize_request_rejects_invalid_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BIZRA_MCP_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.delenv("BIZRA_MCP_ALLOW_ANONYMOUS", raising=False)
    monkeypatch.delenv("BIZRA_MCP_ALLOW_REMOTE", raising=False)

    req = _DummyRequest(headers={"authorization": "Bearer wrong"})
    with pytest.raises(HTTPException, match="Authentication required") as exc:
        mcp_gateway._authorize_request(req)
    assert exc.value.status_code == 401


def test_authorize_request_accepts_valid_local_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BIZRA_MCP_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.delenv("BIZRA_MCP_ALLOW_ANONYMOUS", raising=False)
    monkeypatch.delenv("BIZRA_MCP_ALLOW_REMOTE", raising=False)

    req = _DummyRequest(headers={"authorization": "Bearer gateway-secret"})
    mcp_gateway._authorize_request(req)


def test_authorize_request_allows_anonymous_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BIZRA_MCP_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("BIZRA_BRIDGE_TOKEN", raising=False)
    monkeypatch.setenv("BIZRA_MCP_ALLOW_ANONYMOUS", "true")

    req = _DummyRequest(headers={})
    mcp_gateway._authorize_request(req)
