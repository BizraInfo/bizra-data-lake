from __future__ import annotations

import pytest

from scripts.moneyshot_demo import run_demo


@pytest.mark.asyncio
async def test_moneyshot_demo_mock_runs_end_to_end() -> None:
    result = await run_demo(mock=True)

    assert result["mock"] is True
    assert result["channels_total"] >= 1
    assert 0.0 <= result["reward"] <= 1.0
    assert len(result["receipt_hash"]) == 32


@pytest.mark.asyncio
async def test_moneyshot_demo_single_channel_voice() -> None:
    result = await run_demo(mock=True, channel="voice")

    assert result["channel"] == "voice"
    assert result["channels_total"] == 1


@pytest.mark.asyncio
async def test_moneyshot_demo_single_channel_browser() -> None:
    result = await run_demo(mock=True, channel="browser")

    assert result["channel"] == "browser"
    assert result["channels_total"] == 1
