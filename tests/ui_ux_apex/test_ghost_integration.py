"""
Ghost Overlay Integration Tests — Event Bus → Daemon → WS Bridge chain.

Tests the full integration path:
  EventBus.emit("proactive.prediction") → GhostOverlayDaemon → ghost_ws broadcast

Standing on Giants:
- Boyd (OODA loop verification) · Shannon (SNR threshold gating) · Lamport (single truth)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.integration.constants import (
    UNIFIED_SNR_THRESHOLD,
)
from core.sovereign.event_bus import Event, get_event_bus
from core.sovereign.ghost_overlay_daemon import (
    TOPIC_OPPORTUNITY,
    TOPIC_OVERLAY_DISPATCH,
    TOPIC_OVERLAY_GESTURE,
    TOPIC_PREDICTION,
    GhostOverlayDaemon,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus():
    """Get a fresh event bus (singleton, but reset state)."""
    bus = get_event_bus()
    # Clear subscribers for test isolation
    bus._subscribers.clear()
    bus._wildcard_subscribers.clear()
    bus._event_count = 0
    bus._processed_count = 0
    return bus


@pytest.fixture
def daemon():
    """Create a GhostOverlayDaemon with short debounce for testing."""
    return GhostOverlayDaemon(debounce_ms=50)


@pytest.fixture
def mock_gate():
    """Mock ConstitutionalGate that approves by default."""
    gate = MagicMock()
    result = MagicMock()
    result.approved = True
    result.ihsan_score = 0.97
    result.reason = None
    gate.check = AsyncMock(return_value=result)
    return gate


@pytest.fixture
def gated_daemon(mock_gate):
    """Daemon with a ConstitutionalGate configured."""
    return GhostOverlayDaemon(
        constitutional_gate=mock_gate,
        debounce_ms=50,
    )


# ---------------------------------------------------------------------------
# TestDaemonLifecycle
# ---------------------------------------------------------------------------


class TestDaemonLifecycle:
    """Start/stop and event bus subscription."""

    @pytest.mark.asyncio
    async def test_start_subscribes_to_topics(self, daemon, event_bus):
        await daemon.start()
        assert daemon._running is True
        # Verify event bus has handlers for our topics
        assert len(event_bus._subscribers.get(TOPIC_PREDICTION, set())) == 1
        assert len(event_bus._subscribers.get(TOPIC_OPPORTUNITY, set())) == 1
        assert len(event_bus._subscribers.get(TOPIC_OVERLAY_GESTURE, set())) == 1
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self, daemon, event_bus):
        await daemon.start()
        await daemon.stop()
        assert daemon._running is False
        assert len(event_bus._subscribers.get(TOPIC_PREDICTION, set())) == 0

    @pytest.mark.asyncio
    async def test_stats_initial(self, daemon):
        stats = daemon.stats
        assert stats["running"] is False
        assert stats["events_received"] == 0
        assert stats["events_emitted"] == 0


# ---------------------------------------------------------------------------
# TestPredictionFlow
# ---------------------------------------------------------------------------


class TestPredictionFlow:
    """Event bus prediction → daemon → overlay event chain."""

    @pytest.mark.asyncio
    async def test_prediction_below_threshold_no_overlay(self, daemon):
        """Predictions below SNR threshold do not trigger overlay."""
        await daemon.start()
        event = Event(
            topic=TOPIC_PREDICTION,
            payload={
                "intent": "batch_rename",
                "confidence": UNIFIED_SNR_THRESHOLD - 0.01,
            },
            source="test",
        )
        await daemon._on_prediction_event(event)
        await asyncio.sleep(0.1)
        assert daemon._events_received == 1
        assert daemon.overlay_visible is False
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_prediction_above_threshold_emits_overlay(self, daemon):
        """Predictions above threshold trigger overlay after debounce."""
        await daemon.start()

        with patch(
            "core.sovereign.ghost_overlay_daemon.emit_overlay_event",
            new_callable=AsyncMock,
            return_value=1,
        ) as mock_emit:
            event = Event(
                topic=TOPIC_PREDICTION,
                payload={
                    "intent": "batch_rename",
                    "confidence": UNIFIED_SNR_THRESHOLD + 0.05,
                },
                source="test",
            )
            await daemon._on_prediction_event(event)
            await asyncio.sleep(0.15)

            mock_emit.assert_awaited_once()
            call_args = mock_emit.call_args[0][0]
            assert call_args.type == "show_overlay"
            assert len(call_args.suggestions) >= 1
            assert daemon.overlay_visible is True

        await daemon.stop()

    @pytest.mark.asyncio
    async def test_duplicate_overlay_suppressed(self, daemon):
        """Second prediction while overlay visible does not create new overlay."""
        await daemon.start()
        daemon._overlay_visible = True

        with patch(
            "core.sovereign.ghost_overlay_daemon.emit_overlay_event",
            new_callable=AsyncMock,
        ) as mock_emit:
            event = Event(
                topic=TOPIC_PREDICTION,
                payload={"intent": "sort", "confidence": 0.92},
                source="test",
            )
            await daemon._on_prediction_event(event)
            await asyncio.sleep(0.15)
            mock_emit.assert_not_awaited()

        await daemon.stop()


# ---------------------------------------------------------------------------
# TestConstitutionalGateIntegration
# ---------------------------------------------------------------------------


class TestConstitutionalGateIntegration:
    """Suggestions must pass through ConstitutionalGate."""

    @pytest.mark.asyncio
    async def test_gate_called_per_suggestion(self, gated_daemon, mock_gate):
        """Each suggestion passes through ConstitutionalGate.check()."""
        await gated_daemon.start()

        with patch(
            "core.sovereign.ghost_overlay_daemon.emit_overlay_event",
            new_callable=AsyncMock,
            return_value=1,
        ):
            event = Event(
                topic=TOPIC_PREDICTION,
                payload={"intent": "batch_rename", "confidence": 0.92},
                source="test",
            )
            await gated_daemon._on_prediction_event(event)
            await asyncio.sleep(0.15)

            mock_gate.check.assert_awaited()

        await gated_daemon.stop()

    @pytest.mark.asyncio
    async def test_blocked_suggestion_marked(self, mock_gate):
        """When gate rejects, suggestion shows ihsan_precheck='blocked'."""
        mock_gate.check.return_value.approved = False
        mock_gate.check.return_value.reason = "Below threshold"
        daemon = GhostOverlayDaemon(constitutional_gate=mock_gate, debounce_ms=50)
        await daemon.start()

        with patch(
            "core.sovereign.ghost_overlay_daemon.emit_overlay_event",
            new_callable=AsyncMock,
            return_value=1,
        ) as mock_emit:
            event = Event(
                topic=TOPIC_PREDICTION,
                payload={"intent": "auto_fill", "confidence": 0.91},
                source="test",
            )
            await daemon._on_prediction_event(event)
            await asyncio.sleep(0.15)

            call_args = mock_emit.call_args[0][0]
            suggestion = call_args.suggestions[0]
            assert suggestion["ihsan_precheck"] == "blocked"
            assert suggestion["block_reason"] == "Below threshold"

        await daemon.stop()


# ---------------------------------------------------------------------------
# TestOpportunityFlow
# ---------------------------------------------------------------------------


class TestOpportunityFlow:
    """Muraqabah opportunities routed to overlay."""

    @pytest.mark.asyncio
    async def test_opportunity_above_threshold_processed(self, daemon):
        """Opportunities with high confidence are fed to the debouncer."""
        await daemon.start()

        with patch(
            "core.sovereign.ghost_overlay_daemon.emit_overlay_event",
            new_callable=AsyncMock,
            return_value=1,
        ):
            event = Event(
                topic=TOPIC_OPPORTUNITY,
                payload={
                    "type": "cost_optimization",
                    "confidence": 0.90,
                    "context": {"target": "cloud_spend"},
                },
                source="muraqabah",
            )
            await daemon._on_opportunity_event(event)
            await asyncio.sleep(0.15)
            assert daemon._events_received == 1

        await daemon.stop()

    @pytest.mark.asyncio
    async def test_opportunity_below_threshold_suppressed(self, daemon):
        """Opportunities below SNR threshold are dropped."""
        await daemon.start()
        event = Event(
            topic=TOPIC_OPPORTUNITY,
            payload={"type": "minor", "confidence": 0.50},
            source="muraqabah",
        )
        await daemon._on_opportunity_event(event)
        await asyncio.sleep(0.1)
        assert daemon.overlay_visible is False
        await daemon.stop()


# ---------------------------------------------------------------------------
# TestGestureFlow
# ---------------------------------------------------------------------------


class TestGestureFlow:
    """Sovereign gestures from overlay UI."""

    @pytest.mark.asyncio
    async def test_dismiss_gesture_hides_overlay(self, daemon):
        """Dismiss gesture sets overlay_visible=False and emits dismiss event."""
        await daemon.start()
        daemon._overlay_visible = True

        with patch(
            "core.sovereign.ghost_overlay_daemon.emit_overlay_event",
            new_callable=AsyncMock,
        ) as mock_emit:
            event = Event(
                topic=TOPIC_OVERLAY_GESTURE,
                payload={"gesture": "dismiss"},
                source="overlay_ui",
            )
            await daemon._on_gesture_event(event)
            assert daemon.overlay_visible is False
            mock_emit.assert_awaited_once()
            assert mock_emit.call_args[0][0].type == "dismiss_overlay"

        await daemon.stop()

    @pytest.mark.asyncio
    async def test_solidify_gesture_dispatches_action(self, daemon, event_bus):
        """Solidify gesture emits action dispatch event to the bus."""
        await daemon.start()
        daemon._overlay_visible = True

        dispatched = []

        async def capture_dispatch(event: Event):
            dispatched.append(event)

        event_bus.subscribe(TOPIC_OVERLAY_DISPATCH, capture_dispatch)

        with patch(
            "core.sovereign.ghost_overlay_daemon.emit_overlay_event",
            new_callable=AsyncMock,
        ):
            event = Event(
                topic=TOPIC_OVERLAY_GESTURE,
                payload={"gesture": "solidify", "action_id": "act_rename_abc123"},
                source="overlay_ui",
            )
            await daemon._on_gesture_event(event)

            # Process the event bus queue
            while not event_bus._event_queue.empty():
                _, _, queued = await event_bus._event_queue.get()
                await event_bus._process_event(queued)

            assert len(dispatched) == 1
            assert dispatched[0].payload["action_id"] == "act_rename_abc123"
            assert dispatched[0].payload["channel"] == "Ahk"
            assert daemon.overlay_visible is False

        await daemon.stop()


# ---------------------------------------------------------------------------
# TestIntentMapping
# ---------------------------------------------------------------------------


class TestIntentMapping:
    """Intent → action label mapping."""

    def test_known_intents(self, daemon):
        assert daemon._intent_to_label("batch_rename") == "Batch rename selected files"
        assert daemon._intent_to_label("merge_region") == "Merge selected region"
        assert daemon._intent_to_label("auto_fill") == "Auto-fill detected fields"

    def test_unknown_intent_formatted(self, daemon):
        label = daemon._intent_to_label("custom_workflow")
        assert "Custom Workflow" in label

    def test_candidate_generation(self, daemon):
        candidates = daemon._generate_candidates("batch_rename", 0.92, {})
        assert len(candidates) == 1
        assert candidates[0].hhmm_confidence == 0.92
        assert "batch_rename" in candidates[0].intent_summary
        assert candidates[0].ahk_action_id.startswith("act_batch_rename_")


# ---------------------------------------------------------------------------
# TestTopicConstants
# ---------------------------------------------------------------------------


class TestTopicConstants:
    """Event topics are consistent and non-overlapping."""

    def test_topics_unique(self):
        topics = {
            TOPIC_PREDICTION,
            TOPIC_OPPORTUNITY,
            TOPIC_OVERLAY_GESTURE,
            TOPIC_OVERLAY_DISPATCH,
        }
        assert len(topics) == 4  # All unique

    def test_topics_namespaced(self):
        assert TOPIC_PREDICTION.startswith("proactive.")
        assert TOPIC_OPPORTUNITY.startswith("muraqabah.")
        assert TOPIC_OVERLAY_GESTURE.startswith("ghost.")
        assert TOPIC_OVERLAY_DISPATCH.startswith("ghost.")


# ---------------------------------------------------------------------------
# TestRpcProxyAuthInjection
# ---------------------------------------------------------------------------


class TestRpcProxyAuthInjection:
    """Verify /rpc proxy injects auth into msg['headers'] (not params._auth).

    Integration gap fixed: desktop_bridge._validate_auth() reads msg['headers']
    with X-BIZRA-TOKEN, X-BIZRA-TS, X-BIZRA-NONCE.  The old code injected into
    params['_auth'] which is never read by the bridge — every request would
    receive AUTH_MISSING_HEADERS.
    """

    @pytest.mark.asyncio
    async def test_auth_injected_into_headers_with_all_fields(self):
        """Token from X-BIZRA-TOKEN HTTP header lands in body['headers'] with TS + NONCE."""
        import asyncio as _asyncio
        import json as _json
        import time as _time
        from unittest.mock import AsyncMock, MagicMock, patch

        from httpx import ASGITransport, AsyncClient

        from core.bridges.ghost_ws import app

        captured: dict = {}

        mock_reader = AsyncMock()
        mock_reader.readline = AsyncMock(
            return_value=b'{"jsonrpc":"2.0","result":{"ok":true},"id":1}\n'
        )
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        def capture_write(data: bytes) -> None:
            captured["body"] = _json.loads(data.rstrip(b"\n"))

        mock_writer.write = capture_write

        with patch.object(
            _asyncio,
            "open_connection",
            new_callable=AsyncMock,
            return_value=(mock_reader, mock_writer),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                before_ms = int(_time.time() * 1000)
                resp = await client.post(
                    "/rpc",
                    json={"jsonrpc": "2.0", "method": "ping", "params": {}, "id": 1},
                    headers={"X-BIZRA-TOKEN": "test-secret-token"},
                )
                after_ms = int(_time.time() * 1000)

        assert resp.status_code == 200
        body = captured.get("body", {})

        # Auth must be in msg["headers"], the structure _validate_auth() reads
        assert "headers" in body, "auth must be injected into body['headers']"
        assert body["headers"]["X-BIZRA-TOKEN"] == "test-secret-token"

        assert "X-BIZRA-TS" in body["headers"], "X-BIZRA-TS must be present"
        ts_ms = int(body["headers"]["X-BIZRA-TS"])
        assert (
            before_ms <= ts_ms <= after_ms + 100
        ), "X-BIZRA-TS must be current ms timestamp"

        assert "X-BIZRA-NONCE" in body["headers"], "X-BIZRA-NONCE must be present"
        import uuid as _uuid

        _uuid.UUID(body["headers"]["X-BIZRA-NONCE"])  # must be valid UUID4

        # Old broken path: params["_auth"] must NOT be set
        assert (
            body.get("params", {}).get("_auth") is None
        ), "Old auth path must not be used"

    @pytest.mark.asyncio
    async def test_no_token_no_headers_injected(self):
        """When no X-BIZRA-TOKEN header is sent, body['headers'] must NOT be injected."""
        import asyncio as _asyncio
        import json as _json
        from unittest.mock import AsyncMock, MagicMock, patch

        from httpx import ASGITransport, AsyncClient

        from core.bridges.ghost_ws import app

        captured: dict = {}

        mock_reader = AsyncMock()
        mock_reader.readline = AsyncMock(
            return_value=b'{"jsonrpc":"2.0","result":{},"id":2}\n'
        )
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        def capture_write(data: bytes) -> None:
            captured["body"] = _json.loads(data.rstrip(b"\n"))

        mock_writer.write = capture_write

        with patch.object(
            _asyncio,
            "open_connection",
            new_callable=AsyncMock,
            return_value=(mock_reader, mock_writer),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/rpc",
                    json={"jsonrpc": "2.0", "method": "ping", "params": {}, "id": 2},
                    # No X-BIZRA-TOKEN header — proxy must not fabricate auth
                )

        assert resp.status_code == 200
        body = captured.get("body", {})
        # Without token, no headers dict should be injected
        assert not body.get(
            "headers"
        ), "Proxy must not inject auth headers without a token"
