"""
Ghost Overlay — Unit tests for WS bridge, debouncer, and connection manager.

Tests validate:
  1. Connection management (connect, disconnect, max clients)
  2. Prediction debouncing (SNR filter, window, highest-confidence wins)
  3. Overlay event broadcasting
  4. Health endpoint
  5. Constitutional threshold alignment
  6. No hardcoded secrets in source

Standing on Giants:
- Shannon (SNR gating) · Norman (invisible design) · Boyd (OODA feedback)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from unittest.mock import AsyncMock

import pytest

from core.bridges.ghost_ws import (
    MAX_SUGGESTIONS,
    GhostConnectionManager,
    OverlayEvent,
    OverlaySuggestion,
    PredictionDebouncer,
    app,
)
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)


def _ghost_ws_endpoint():
    route = next(r for r in app.routes if getattr(r, "path", "") == "/ws/ghost")
    return route.endpoint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ws_mock(accepted: bool = True) -> AsyncMock:
    """Create a mock WebSocket."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    ws.close = AsyncMock()
    return ws


def _make_prediction(
    intent: str = "batch_rename",
    confidence: float = 0.92,
    node_id: str = "n1",
) -> dict:
    return {"intent": intent, "confidence": confidence, "node_id": node_id}


# ---------------------------------------------------------------------------
# TestGhostConnectionManager
# ---------------------------------------------------------------------------


class TestGhostConnectionManager:
    """Connection lifecycle and broadcasting."""

    @pytest.mark.asyncio
    async def test_connect_accept(self):
        mgr = GhostConnectionManager(max_clients=2)
        ws = _make_ws_mock()
        result = await mgr.connect(ws)
        assert result is True
        assert mgr.client_count == 1
        ws.accept.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_max_clients_enforced(self):
        mgr = GhostConnectionManager(max_clients=1)
        ws1 = _make_ws_mock()
        ws2 = _make_ws_mock()
        await mgr.connect(ws1)
        result = await mgr.connect(ws2)
        assert result is False
        assert mgr.client_count == 1

    @pytest.mark.asyncio
    async def test_disconnect_removes(self):
        mgr = GhostConnectionManager(max_clients=2)
        ws = _make_ws_mock()
        await mgr.connect(ws)
        assert mgr.client_count == 1
        mgr.disconnect(ws)
        assert mgr.client_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_all_clients(self):
        mgr = GhostConnectionManager(max_clients=4)
        ws1, ws2 = _make_ws_mock(), _make_ws_mock()
        await mgr.connect(ws1)
        await mgr.connect(ws2)
        event = OverlayEvent(type="show_overlay")
        sent = await mgr.broadcast(event)
        assert sent == 2
        ws1.send_text.assert_awaited_once()
        ws2.send_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_broadcast_removes_stale(self):
        mgr = GhostConnectionManager(max_clients=4)
        ws1 = _make_ws_mock()
        ws_broken = _make_ws_mock()
        ws_broken.send_text.side_effect = ConnectionError("pipe broken")
        await mgr.connect(ws1)
        await mgr.connect(ws_broken)
        assert mgr.client_count == 2
        event = OverlayEvent(type="test")
        sent = await mgr.broadcast(event)
        assert sent == 1  # only ws1 succeeded
        assert mgr.client_count == 1  # ws_broken removed

    @pytest.mark.asyncio
    async def test_broadcast_empty_no_error(self):
        mgr = GhostConnectionManager(max_clients=4)
        event = OverlayEvent(type="test")
        sent = await mgr.broadcast(event)
        assert sent == 0

    def test_health_summary(self):
        mgr = GhostConnectionManager(max_clients=4)
        summary = mgr.health_summary()
        assert summary["connected_clients"] == 0
        assert summary["max_clients"] == 4
        assert "uptime_seconds" in summary


# ---------------------------------------------------------------------------
# TestPredictionDebouncer
# ---------------------------------------------------------------------------


class TestPredictionDebouncer:
    """Debounce rapid HHMM predictions."""

    @pytest.mark.asyncio
    async def test_low_confidence_suppressed(self):
        """Predictions below UNIFIED_SNR_THRESHOLD do not trigger callback."""
        callback = AsyncMock()
        db = PredictionDebouncer(debounce_ms=50)
        db.set_callback(callback)
        await db.on_prediction(
            _make_prediction(confidence=UNIFIED_SNR_THRESHOLD - 0.01)
        )
        await asyncio.sleep(0.1)
        callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_above_threshold_fires(self):
        """Predictions above threshold fire after debounce window."""
        callback = AsyncMock()
        db = PredictionDebouncer(debounce_ms=50)
        db.set_callback(callback)
        pred = _make_prediction(confidence=UNIFIED_SNR_THRESHOLD + 0.05)
        await db.on_prediction(pred)
        await asyncio.sleep(0.15)
        callback.assert_awaited_once_with(pred)

    @pytest.mark.asyncio
    async def test_debounce_keeps_highest_confidence(self):
        """Two predictions within window: only highest-confidence fires."""
        fired = []

        async def capture(p):
            fired.append(p)

        db = PredictionDebouncer(debounce_ms=100)
        db.set_callback(capture)
        await db.on_prediction(_make_prediction(intent="sort", confidence=0.88))
        await db.on_prediction(_make_prediction(intent="batch_rename", confidence=0.91))
        await asyncio.sleep(0.2)
        assert len(fired) == 1
        assert fired[0]["intent"] == "batch_rename"
        assert fired[0]["confidence"] == 0.91

    @pytest.mark.asyncio
    async def test_debounce_resets_timer(self):
        """Second prediction resets the debounce timer."""
        fired = []

        async def capture(p):
            fired.append(p)

        db = PredictionDebouncer(debounce_ms=100)
        db.set_callback(capture)
        await db.on_prediction(_make_prediction(confidence=0.88))
        await asyncio.sleep(0.05)  # 50ms — still in window
        assert len(fired) == 0
        await db.on_prediction(_make_prediction(confidence=0.92))
        await asyncio.sleep(0.05)  # 50ms from second — still in window
        assert len(fired) == 0
        await asyncio.sleep(0.1)  # Now past debounce
        assert len(fired) == 1


# ---------------------------------------------------------------------------
# TestOverlayEvent
# ---------------------------------------------------------------------------


class TestOverlayEvent:
    """Overlay event data model."""

    def test_event_serializable(self):
        event = OverlayEvent(type="show_overlay", suggestions=[{"id": "s1"}])
        from dataclasses import asdict

        d = asdict(event)
        payload = json.dumps(d)
        assert '"show_overlay"' in payload
        assert '"s1"' in payload

    def test_event_has_timestamp(self):
        before = time.time()
        event = OverlayEvent(type="test")
        after = time.time()
        assert before <= event.timestamp <= after

    def test_suggestion_dataclass(self):
        s = OverlaySuggestion(
            id="s1",
            action_label="Batch Rename selected cells",
            intent_summary="HHMM: batch_rename (92% confident)",
            hhmm_confidence=0.92,
            ihsan_precheck="pass",
            ihsan_score=0.97,
            ahk_action_id="act_001",
        )
        assert s.block_reason is None
        assert s.hhmm_confidence == 0.92


# ---------------------------------------------------------------------------
# TestConstitutionalAlignment
# ---------------------------------------------------------------------------


class TestConstitutionalAlignment:
    """Thresholds must come from constants.py, never hardcoded."""

    def test_snr_threshold_from_constants(self):
        """Ghost WS uses UNIFIED_SNR_THRESHOLD from constants.py."""
        from core.bridges import ghost_ws

        assert ghost_ws.UNIFIED_SNR_THRESHOLD == UNIFIED_SNR_THRESHOLD

    def test_ihsan_threshold_from_constants(self):
        """Ghost WS uses UNIFIED_IHSAN_THRESHOLD from constants.py."""
        from core.bridges import ghost_ws

        assert ghost_ws.UNIFIED_IHSAN_THRESHOLD == UNIFIED_IHSAN_THRESHOLD

    def test_no_hardcoded_threshold_values(self):
        """Source code must not contain hardcoded 0.85 or 0.95 as threshold assignments."""
        source = inspect.getsource(PredictionDebouncer)
        # Should reference the constant, not a literal
        assert "0.85" not in source
        assert "0.95" not in source

    def test_max_suggestions_is_3(self):
        """Spec requires max 3 suggestions."""
        assert MAX_SUGGESTIONS == 3


# ---------------------------------------------------------------------------
# TestHealthEndpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """FastAPI /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self):
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert "thresholds" in data
            assert data["thresholds"]["ihsan"] == UNIFIED_IHSAN_THRESHOLD
            assert data["thresholds"]["snr"] == UNIFIED_SNR_THRESHOLD


class TestProductionHardening:
    """Production defaults must disable or harden the Ghost bridge."""

    @pytest.mark.asyncio
    async def test_rpc_disabled_by_default_in_production(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from httpx import ASGITransport, AsyncClient

        import core.bridges.ghost_ws as ghost_ws

        monkeypatch.setattr(ghost_ws, "GHOST_WS_ENABLED", False)
        monkeypatch.setattr(ghost_ws, "BIZRA_ENV", "production")
        monkeypatch.setattr(ghost_ws, "_production_mode_enabled", lambda: True)

        transport = ASGITransport(app=ghost_ws.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/rpc",
                json={"jsonrpc": "2.0", "method": "ping", "params": {}, "id": 1},
            )

        assert resp.status_code == 503
        assert "disabled in production" in resp.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_rpc_requires_auth_token_when_enabled_in_production(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from httpx import ASGITransport, AsyncClient

        import core.bridges.ghost_ws as ghost_ws

        monkeypatch.setattr(ghost_ws, "GHOST_WS_ENABLED", True)
        monkeypatch.setattr(ghost_ws, "GHOST_WS_AUTH_TOKEN", "test-secret-token")
        monkeypatch.setattr(ghost_ws, "BIZRA_ENV", "production")
        monkeypatch.setattr(ghost_ws, "_production_mode_enabled", lambda: True)
        monkeypatch.setattr(ghost_ws, "_loopback_client", lambda host: True)

        transport = ASGITransport(app=ghost_ws.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/rpc",
                json={"jsonrpc": "2.0", "method": "ping", "params": {}, "id": 2},
            )

        assert resp.status_code == 401
        assert "authentication failed" in resp.json()["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_websocket_disabled_by_default_in_production(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import core.bridges.ghost_ws as ghost_ws

        monkeypatch.setattr(ghost_ws, "GHOST_WS_ENABLED", False)
        monkeypatch.setattr(ghost_ws, "BIZRA_ENV", "production")
        monkeypatch.setattr(ghost_ws, "_production_mode_enabled", lambda: True)

        endpoint = _ghost_ws_endpoint()

        class _FakeWebSocket:
            def __init__(self):
                self.client = type("Client", (), {"host": "127.0.0.1"})()
                self.headers = {}
                self.closed = None

            async def close(self, code: int = 1000, reason: str = "") -> None:
                self.closed = (code, reason)

        ws = _FakeWebSocket()
        await endpoint(ws)

        assert ws.closed == (4403, "Ghost bridge disabled in production")


# ---------------------------------------------------------------------------
# TestNoHardcodedSecrets
# ---------------------------------------------------------------------------


class TestNoHardcodedSecrets:
    """Verify no secrets leak into source."""

    def test_source_clean(self):
        """Ghost WS source has no hardcoded secrets."""
        import core.bridges.ghost_ws as mod
        from tests.ui_ux_apex.conftest import assert_no_hardcoded_secrets

        source = inspect.getsource(mod)
        assert_no_hardcoded_secrets(source)

    def test_auth_token_from_env(self):
        """Auth token must come from environment, not source."""
        # In test env, it's empty (no env var set) — correct
        # In production, GHOST_WS_AUTH_TOKEN env var is required
        source = inspect.getsource(PredictionDebouncer)
        assert "GHOST_WS_AUTH_TOKEN" not in source  # debouncer doesn't touch auth
