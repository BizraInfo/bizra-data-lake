from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

_SERVICE_ROOT = (
    Path(__file__).resolve().parents[2]
    / ".tmp_prod_artifacts_v2"
    / "services"
    / "node_gateway"
)
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

from app import routers  # noqa: E402
from app.node.mission_bridge import MissionPlan  # noqa: E402


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(routers.router)
    return app


@pytest.mark.asyncio
async def test_plan_uses_reflex_cache_hit(monkeypatch):
    monkeypatch.setenv("BIZRA_API_KEY", "test-api-key")
    monkeypatch.setattr(routers.cache, "get", lambda _macro: ["cached-step"])

    calls = {"bridge": 0}

    async def _unexpected_bridge(_text, _context, macro_state):  # pragma: no cover
        calls["bridge"] += 1
        return MissionPlan(macro_state=macro_state, steps=["x"], snr=0.8, poi_score=0.8)

    monkeypatch.setattr(routers.mission_bridge, "run", _unexpected_bridge)

    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/plan",
            json={"text": "research llm routing", "context": {}},
            headers={"x-bizra-api-key": "test-api-key"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["steps"] == ["cached-step"]
    assert 0.0 <= body["snr"] <= 1.0
    assert calls["bridge"] == 0


@pytest.mark.asyncio
async def test_plan_cache_miss_runs_mission_bridge_and_persists(monkeypatch):
    monkeypatch.setenv("BIZRA_API_KEY", "test-api-key")
    monkeypatch.setattr(routers.cache, "get", lambda _macro: None)

    persisted: dict[str, list[str]] = {}

    def _put(macro_state: str, steps: list[str]) -> None:
        persisted[macro_state] = steps

    monkeypatch.setattr(routers.cache, "put", _put)

    async def _bridge(_text, _context, macro_state):
        return MissionPlan(
            macro_state=macro_state,
            steps=["Run browser channel (ok)", "Emit evidence receipt deadbeef"],
            snr=0.91,
            poi_score=0.93,
        )

    monkeypatch.setattr(routers.mission_bridge, "run", _bridge)

    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/plan",
            json={"text": "research browser bridge", "context": {}},
            headers={"x-bizra-api-key": "test-api-key"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["steps"][0].startswith("Run browser channel")
    assert body["snr"] == 0.91
    assert body["poi_score"] == 0.93
    assert body["macro_state"] in persisted


@pytest.mark.asyncio
async def test_plan_requires_api_key(monkeypatch):
    monkeypatch.setenv("BIZRA_API_KEY", "test-api-key")
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/plan", json={"text": "hello", "context": {}})
    assert response.status_code == 401
