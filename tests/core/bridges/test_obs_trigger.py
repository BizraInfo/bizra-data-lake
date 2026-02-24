"""Tests for core.bridges.obs_trigger — OBS WebSocket Trigger."""

from __future__ import annotations

import pytest

from core.bridges.obs_trigger import OBSTrigger


class TestOBSTrigger:
    """OBS trigger initialization and graceful degradation."""

    def test_initialization(self) -> None:
        trigger = OBSTrigger()
        assert trigger.host == "127.0.0.1"
        assert trigger.port == 4455
        assert trigger._connected is False
        assert trigger._ws is None

    def test_initialization_custom(self) -> None:
        trigger = OBSTrigger(host="10.0.0.1", port=5555, password="secret")
        assert trigger.host == "10.0.0.1"
        assert trigger.port == 5555

    @pytest.mark.asyncio
    async def test_connect_returns_false_when_obs_not_running(self) -> None:
        """connect() should return False when no OBS instance is reachable."""
        trigger = OBSTrigger(port=19999)  # unlikely to have OBS on this port
        result = await trigger.connect()
        assert result is False
        assert trigger._connected is False

    @pytest.mark.asyncio
    async def test_start_recording_returns_false_when_not_connected(self) -> None:
        trigger = OBSTrigger()
        result = await trigger.start_recording()
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_recording_returns_false_when_not_connected(self) -> None:
        trigger = OBSTrigger()
        result = await trigger.stop_recording()
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_is_safe_when_not_connected(self) -> None:
        trigger = OBSTrigger()
        # Should not raise
        await trigger.disconnect()
        assert trigger._connected is False
