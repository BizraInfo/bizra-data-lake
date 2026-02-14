"""
Tests for the BIZRA Channel Adapter Protocol and Telegram Gateway.

Covers:
    - ChannelAdapter abstract protocol
    - GatewayMetrics tracking
    - TelegramAdapter message routing
    - Command handling (/start, /help, /status, /agents)
    - Session management
    - Webhook update processing
    - Error handling and graceful degradation
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.pat.bridge import ChannelType
from core.pat.channels import (
    ChannelAdapter,
    GatewayMetrics,
    GatewayState,
    QueryFn,
)


# ─── Concrete test adapter ────────────────────────────────────────


class MockAdapter(ChannelAdapter):
    """Concrete adapter for testing the abstract protocol."""

    def __init__(self, query_fn: QueryFn, node_id: str = "BIZRA-TEST0001"):
        super().__init__(
            channel_type=ChannelType.INTERNAL,
            query_fn=query_fn,
            node_id=node_id,
        )
        self.sent_messages: List[Dict[str, Any]] = []

    async def start(self) -> None:
        self._state = GatewayState.RUNNING

    async def stop(self) -> None:
        self._state = GatewayState.STOPPED

    async def send_response(
        self, platform_user_id: str, content: str, **kwargs: Any
    ) -> bool:
        self.sent_messages.append({
            "user_id": platform_user_id,
            "content": content,
            **kwargs,
        })
        return True


# ─── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def echo_query_fn():
    """Query function that echoes input."""
    async def fn(content: str, context: Optional[Dict] = None) -> str:
        return f"Response: {content}"
    return fn


@pytest.fixture
def failing_query_fn():
    """Query function that always fails."""
    async def fn(content: str, context: Optional[Dict] = None) -> str:
        raise RuntimeError("Query engine offline")
    return fn


@pytest.fixture
def adapter(echo_query_fn):
    return MockAdapter(query_fn=echo_query_fn)


# ─── GatewayMetrics ───────────────────────────────────────────────


class TestGatewayMetrics:
    def test_initial_state(self):
        m = GatewayMetrics()
        assert m.messages_received == 0
        assert m.messages_sent == 0
        assert m.errors == 0
        assert m.unique_users == 0

    def test_to_dict(self):
        m = GatewayMetrics(messages_received=5, unique_users=2)
        d = m.to_dict()
        assert d["messages_received"] == 5
        assert d["unique_users"] == 2


# ─── ChannelAdapter Protocol ──────────────────────────────────────


class TestChannelAdapter:
    @pytest.mark.asyncio
    async def test_start_stop(self, adapter):
        assert adapter.state == GatewayState.STOPPED
        await adapter.start()
        assert adapter.state == GatewayState.RUNNING
        await adapter.stop()
        assert adapter.state == GatewayState.STOPPED

    @pytest.mark.asyncio
    async def test_handle_incoming_routes_to_query(self, adapter):
        response = await adapter.handle_incoming("user1", "Hello BIZRA")
        assert response == "Response: Hello BIZRA"

    @pytest.mark.asyncio
    async def test_metrics_track_messages(self, adapter):
        await adapter.handle_incoming("user1", "msg1")
        await adapter.handle_incoming("user1", "msg2")
        await adapter.handle_incoming("user2", "msg3")

        assert adapter.metrics.messages_received == 3
        assert adapter.metrics.messages_sent == 3
        assert adapter.metrics.unique_users == 2

    @pytest.mark.asyncio
    async def test_session_tracking(self, adapter):
        await adapter.handle_incoming("user1", "first")
        await adapter.handle_incoming("user1", "second")

        session = adapter._sessions.get("user1")
        assert session is not None
        assert session.message_count == 2
        assert session.channel == ChannelType.INTERNAL

    @pytest.mark.asyncio
    async def test_error_handling(self, failing_query_fn):
        adapter = MockAdapter(query_fn=failing_query_fn)
        response = await adapter.handle_incoming("user1", "test")

        assert "error" in response.lower()
        assert adapter.metrics.errors == 1

    @pytest.mark.asyncio
    async def test_context_includes_channel_info(self, adapter):
        """Verify context dict passed to query_fn."""
        received_context = {}

        async def capture_fn(content, context=None):
            nonlocal received_context
            received_context = context or {}
            return "ok"

        adapter._query_fn = capture_fn
        await adapter.handle_incoming("user42", "test", metadata={"platform": "test"})

        assert received_context["channel"] == "internal"
        assert received_context["user_id"] == "user42"
        assert "session_id" in received_context

    def test_get_status(self, adapter):
        status = adapter.get_status()
        assert status["channel"] == "internal"
        assert status["state"] == "stopped"
        assert status["node_id"] == "BIZRA-TEST0001"
        assert "metrics" in status


# ─── TelegramAdapter ──────────────────────────────────────────────


class TestTelegramAdapter:
    def test_requires_token(self):
        """Must provide token or env var."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="token required"):
                from core.pat.adapters.telegram import TelegramAdapter
                TelegramAdapter(token="")

    def test_accepts_token(self):
        from core.pat.adapters.telegram import TelegramAdapter

        async def dummy_fn(c, ctx=None):
            return "ok"

        adapter = TelegramAdapter(token="test:TOKEN", query_fn=dummy_fn)
        assert adapter._token == "test:TOKEN"
        assert adapter.channel_type == ChannelType.TELEGRAM

    def test_env_var_token(self):
        from core.pat.adapters.telegram import TelegramAdapter

        with patch.dict("os.environ", {"BIZRA_TELEGRAM_TOKEN": "env:TOKEN"}):
            adapter = TelegramAdapter(query_fn=AsyncMock(return_value="ok"))
            assert adapter._token == "env:TOKEN"

    @pytest.mark.asyncio
    async def test_process_update_text(self):
        from core.pat.adapters.telegram import TelegramAdapter

        responses = []

        async def mock_query(content, context=None):
            return f"Answer: {content}"

        adapter = TelegramAdapter(token="test:TOKEN", query_fn=mock_query)
        adapter._started_at = time.time()

        # Mock send_response
        adapter.send_response = AsyncMock(return_value=True)

        update = {
            "update_id": 1,
            "message": {
                "message_id": 42,
                "chat": {"id": 12345, "type": "private"},
                "from": {"first_name": "Test", "username": "testuser"},
                "text": "What is BIZRA?",
            },
        }

        await adapter._process_update(update)

        adapter.send_response.assert_called_once()
        call_args = adapter.send_response.call_args
        assert call_args[1]["platform_user_id"] == "12345"
        assert "Answer: What is BIZRA?" in call_args[1]["content"]

    @pytest.mark.asyncio
    async def test_start_command(self):
        from core.pat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(token="test:TOKEN", query_fn=AsyncMock())
        adapter._started_at = time.time()

        response = await adapter._handle_command(
            "/start", "12345", {"first_name": "Alice"}
        )

        assert "Welcome to BIZRA" in response
        assert "Alice" in response
        assert "sovereign" in response.lower()

    @pytest.mark.asyncio
    async def test_help_command(self):
        from core.pat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(token="test:TOKEN", query_fn=AsyncMock())

        response = await adapter._handle_command("/help", "12345", {})
        assert "/status" in response
        assert "/agents" in response

    @pytest.mark.asyncio
    async def test_status_command(self):
        from core.pat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(
            token="test:TOKEN",
            query_fn=AsyncMock(),
            node_id="BIZRA-TESTNODE",
        )
        adapter._started_at = time.time()
        adapter._bot_info = {"username": "testbot"}
        adapter._state = GatewayState.RUNNING

        response = await adapter._cmd_status("12345")
        assert "BIZRA-TESTNODE" in response
        assert "@testbot" in response

    @pytest.mark.asyncio
    async def test_webhook_update(self):
        from core.pat.adapters.telegram import TelegramAdapter

        async def mock_query(content, context=None):
            return f"Sovereign: {content}"

        adapter = TelegramAdapter(token="test:TOKEN", query_fn=mock_query)
        adapter._started_at = time.time()
        adapter.send_response = AsyncMock(return_value=True)

        update = {
            "message": {
                "message_id": 1,
                "chat": {"id": 999, "type": "private"},
                "from": {"first_name": "Bob"},
                "text": "Hello",
            },
        }

        result = await adapter.handle_webhook_update(update)
        assert result == "Sovereign: Hello"
        adapter.send_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_update_no_text(self):
        from core.pat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(token="test:TOKEN", query_fn=AsyncMock())

        result = await adapter.handle_webhook_update({"message": {"chat": {"id": 1}}})
        assert result is None

    def test_get_status_includes_telegram_info(self):
        from core.pat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(token="test:TOKEN", query_fn=AsyncMock())
        adapter._bot_info = {"username": "mybot"}

        status = adapter.get_status()
        assert status["bot_username"] == "mybot"
        assert status["mode"] == "polling"
        assert status["channel"] == "telegram"

    @pytest.mark.asyncio
    async def test_unknown_command_routes_as_query(self):
        from core.pat.adapters.telegram import TelegramAdapter

        async def mock_query(content, context=None):
            return f"Queried: {content}"

        adapter = TelegramAdapter(token="test:TOKEN", query_fn=mock_query)
        adapter._started_at = time.time()

        response = await adapter._handle_command(
            "/weather", "12345", {"first_name": "User"}
        )
        assert "Queried: weather" in response

    @pytest.mark.asyncio
    async def test_command_strips_bot_mention(self):
        from core.pat.adapters.telegram import TelegramAdapter

        adapter = TelegramAdapter(token="test:TOKEN", query_fn=AsyncMock())
        adapter._started_at = time.time()

        response = await adapter._handle_command(
            "/help@mybotname", "12345", {}
        )
        assert "/status" in response
