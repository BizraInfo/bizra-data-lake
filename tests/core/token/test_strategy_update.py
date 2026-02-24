from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from core.token.strategy import AgentStrategy, load_strategy, persist_strategy, update_strategy


@dataclass
class _FakeEntry:
    content: str
    created_at: datetime


class _FakeMemory:
    def __init__(self) -> None:
        self._entries: list[_FakeEntry] = []
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True

    async def encode(self, content: str, memory_type, source: str, importance: float):
        del memory_type, source, importance
        self._entries.append(_FakeEntry(content=content, created_at=datetime.utcnow()))
        return None

    async def retrieve(self, query: str, memory_type, top_k: int, min_score: float):
        del memory_type, top_k, min_score
        return [entry for entry in self._entries if query in entry.content]


def test_update_strategy_bounds_temperature_and_tokens() -> None:
    base = AgentStrategy(agent_id="researcher", temperature=0.9, max_tokens=1900, ema_reward=0.8)
    updated = update_strategy(base, reward=0.95, mission_context={"complexity": 0.9})

    assert 0.3 <= updated.temperature <= 1.0
    assert 300 <= updated.max_tokens <= 2000
    assert updated.missions_seen == base.missions_seen + 1


def test_update_strategy_can_enable_rlm() -> None:
    base = AgentStrategy(agent_id="analyst", ema_reward=0.75, use_rlm=False)
    updated = update_strategy(base, reward=0.9, mission_context={"complexity": 0.8})
    assert updated.use_rlm is True


@pytest.mark.asyncio
async def test_persist_and_load_strategy_with_dict_store() -> None:
    memory: dict[str, str] = {}
    strategy = AgentStrategy(agent_id="creator", temperature=0.55, max_tokens=777, use_rlm=True)

    await persist_strategy(memory, "creator", strategy)
    loaded = await load_strategy(memory, "creator")

    assert loaded.agent_id == "creator"
    assert loaded.temperature == strategy.temperature
    assert loaded.max_tokens == strategy.max_tokens
    assert loaded.use_rlm is True


@pytest.mark.asyncio
async def test_persist_and_load_strategy_with_memory_interface() -> None:
    memory = _FakeMemory()
    strategy = AgentStrategy(agent_id="strategist", temperature=0.61, max_tokens=888, use_rlm=True)

    await persist_strategy(memory, "strategist", strategy)

    # Add an older value and verify latest wins by timestamp sorting.
    memory._entries.append(
        _FakeEntry(
            content='agent_strategy::strategist:{"agent_id":"strategist","temperature":0.3,"max_tokens":300}',
            created_at=datetime.utcnow() - timedelta(days=1),
        )
    )

    loaded = await load_strategy(memory, "strategist")
    assert loaded.temperature == pytest.approx(0.61)
    assert loaded.max_tokens == 888


@pytest.mark.asyncio
async def test_load_strategy_returns_default_when_missing() -> None:
    loaded = await load_strategy({}, "guardian")
    assert loaded.agent_id == "guardian"
    assert loaded.temperature == pytest.approx(0.7)
    assert loaded.max_tokens == 1200
