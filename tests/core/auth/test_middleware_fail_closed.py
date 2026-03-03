from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import core.auth.middleware as middleware_module


@pytest.fixture(autouse=True)
def _reset_global_middleware():
    previous = middleware_module._global_middleware
    middleware_module._global_middleware = None
    yield
    middleware_module._global_middleware = previous


@pytest.mark.asyncio
async def test_uninitialized_middleware_returns_503(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BIZRA_AUTH_ALLOW_ANONYMOUS", raising=False)

    with pytest.raises(HTTPException) as exc_info:
        await middleware_module.get_current_user(authorization=None, x_api_key=None)

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_uninitialized_middleware_allows_anon_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")

    result = await middleware_module.get_current_user(
        authorization=None,
        x_api_key=None,
    )

    assert result is None


@pytest.mark.asyncio
async def test_initialized_middleware_rejects_invalid_credentials():
    middleware_module._global_middleware = SimpleNamespace(
        authenticate=lambda authorization, api_key: None
    )

    with pytest.raises(HTTPException) as exc_info:
        await middleware_module.get_current_user(
            authorization="Bearer invalid",
            x_api_key=None,
        )

    assert exc_info.value.status_code == 401
