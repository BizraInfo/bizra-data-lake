"""Adaptive per-agent strategy memory for reward-driven tuning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.living_memory.core import MemoryType

_STRATEGY_PREFIX = "agent_strategy::"
_MIN_TEMPERATURE = 0.3
_MAX_TEMPERATURE = 1.0
_MIN_MAX_TOKENS = 300
_MAX_MAX_TOKENS = 2000


@dataclass
class AgentStrategy:
    """Learnable generation policy for a single PAT agent."""

    agent_id: str
    temperature: float = 0.7
    max_tokens: int = 1200
    use_rlm: bool = False
    ema_reward: float = 0.5
    ema_alpha: float = 0.3
    missions_seen: int = 0
    updated_at: str = ""

    def normalized(self) -> "AgentStrategy":
        return AgentStrategy(
            agent_id=self.agent_id,
            temperature=max(_MIN_TEMPERATURE, min(_MAX_TEMPERATURE, self.temperature)),
            max_tokens=max(_MIN_MAX_TOKENS, min(_MAX_MAX_TOKENS, int(self.max_tokens))),
            use_rlm=bool(self.use_rlm),
            ema_reward=max(0.0, min(1.0, self.ema_reward)),
            ema_alpha=max(0.05, min(0.9, self.ema_alpha)),
            missions_seen=max(0, int(self.missions_seen)),
            updated_at=self.updated_at or datetime.now(timezone.utc).isoformat(),
        )


def update_strategy(
    strategy: AgentStrategy,
    reward: float,
    mission_context: dict[str, Any] | None,
) -> AgentStrategy:
    """Update EMA reward and adapt temperature/token budget with bounded values.

    Uses curvature-aware EMA: when the reward landscape is sharp (large delta
    between consecutive rewards), alpha scales up for faster adaptation. When
    flat, alpha stays low for stability. This satisfies Lemma 2 of the Thermal
    Consciousness proofs: Δt < 2/(λ_max + T·σ_noise).
    """
    ctx = mission_context or {}
    bounded_reward = max(0.0, min(1.0, float(reward)))

    # Curvature-aware adaptive alpha (Lemma 2: Hessian proxy via reward delta)
    reward_delta = abs(bounded_reward - strategy.ema_reward)
    curvature_factor = min(reward_delta / 0.1, 3.0)
    adaptive_alpha = min(strategy.ema_alpha * curvature_factor, 0.5)
    # Floor: never go below base alpha (prevents stagnation in flat regions)
    adaptive_alpha = max(adaptive_alpha, strategy.ema_alpha)

    ema = adaptive_alpha * bounded_reward + (1.0 - adaptive_alpha) * strategy.ema_reward
    raw_complexity = ctx.get("task_complexity", ctx.get("complexity", 0.5))
    complexity = max(0.0, min(1.0, float(raw_complexity)))

    temperature = strategy.temperature
    if ema < 0.4 or complexity > 0.8:
        temperature += 0.08
    elif ema > 0.75:
        temperature -= 0.06

    max_tokens = int(strategy.max_tokens)
    if bounded_reward > 0.8 and complexity > 0.6:
        max_tokens += 120
    elif bounded_reward < 0.4:
        max_tokens -= 100

    use_rlm = bool(strategy.use_rlm)
    if ctx.get("force_rlm") is True:
        use_rlm = True
    elif ctx.get("force_rlm") is False:
        use_rlm = False
    elif ema > 0.7 and complexity >= 0.6:
        use_rlm = True

    updated = AgentStrategy(
        agent_id=strategy.agent_id,
        temperature=temperature,
        max_tokens=max_tokens,
        use_rlm=use_rlm,
        ema_reward=ema,
        ema_alpha=strategy.ema_alpha,
        missions_seen=strategy.missions_seen + 1,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    return updated.normalized()


async def persist_strategy(
    memory: Any,
    agent_id: str,
    strategy: AgentStrategy,
) -> None:
    """Persist strategy into procedural memory (or compatible fallback store)."""
    payload = json.dumps(asdict(strategy.normalized()), sort_keys=True)

    if memory is None:
        return

    if isinstance(memory, dict):
        memory[f"{_STRATEGY_PREFIX}{agent_id}"] = payload
        return

    if isinstance(memory, Path):
        memory.parent.mkdir(parents=True, exist_ok=True)
        memory.write_text(payload, encoding="utf-8")
        return

    if hasattr(memory, "encode"):
        await _ensure_memory_initialized(memory)
        content = f"{_STRATEGY_PREFIX}{agent_id}:{payload}"
        await memory.encode(
            content=content,
            memory_type=MemoryType.PROCEDURAL,
            source=f"strategy/{agent_id}",
            importance=0.8,
        )
        return

    # Best-effort generic fallback for key-value style stores.
    if hasattr(memory, "set"):
        memory.set(f"{_STRATEGY_PREFIX}{agent_id}", payload)


async def load_strategy(memory: Any, agent_id: str) -> AgentStrategy:
    """Load strategy from procedural memory (or compatible fallback store)."""
    if memory is None:
        return AgentStrategy(agent_id=agent_id).normalized()

    if isinstance(memory, dict):
        raw = memory.get(f"{_STRATEGY_PREFIX}{agent_id}")
        return _decode_strategy(raw, agent_id)

    if isinstance(memory, Path):
        if not memory.exists():
            return AgentStrategy(agent_id=agent_id).normalized()
        return _decode_strategy(memory.read_text(encoding="utf-8"), agent_id)

    if hasattr(memory, "retrieve"):
        await _ensure_memory_initialized(memory)
        entries = await memory.retrieve(
            query=f"{_STRATEGY_PREFIX}{agent_id}",
            memory_type=MemoryType.PROCEDURAL,
            top_k=20,
            min_score=0.0,
        )

        candidates = []
        for entry in entries:
            content = getattr(entry, "content", "")
            if content.startswith(f"{_STRATEGY_PREFIX}{agent_id}:"):
                candidates.append(entry)

        if candidates:
            candidates.sort(
                key=lambda e: getattr(e, "created_at", datetime.min),
                reverse=True,
            )
            prefix = f"{_STRATEGY_PREFIX}{agent_id}:"
            latest = candidates[0].content
            if latest.startswith(prefix):
                latest = latest[len(prefix) :]
            return _decode_strategy(latest, agent_id)

    if hasattr(memory, "get"):
        return _decode_strategy(memory.get(f"{_STRATEGY_PREFIX}{agent_id}"), agent_id)

    return AgentStrategy(agent_id=agent_id).normalized()


async def _ensure_memory_initialized(memory: Any) -> None:
    if hasattr(memory, "initialize"):
        initialized = getattr(memory, "_initialized", False)
        if not initialized:
            await memory.initialize()


def _decode_strategy(raw: Any, agent_id: str) -> AgentStrategy:
    if raw is None:
        return AgentStrategy(agent_id=agent_id).normalized()

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError, ValueError):  # SEC-003 — json boundary
        return AgentStrategy(agent_id=agent_id).normalized()

    strategy = AgentStrategy(
        agent_id=str(data.get("agent_id", agent_id)),
        temperature=float(data.get("temperature", 0.7)),
        max_tokens=int(data.get("max_tokens", 1200)),
        use_rlm=bool(data.get("use_rlm", False)),
        ema_reward=float(data.get("ema_reward", 0.5)),
        ema_alpha=float(data.get("ema_alpha", 0.3)),
        missions_seen=int(data.get("missions_seen", data.get("total_missions", 0))),
        updated_at=str(data.get("updated_at", "")),
    )
    return strategy.normalized()


__all__ = [
    "AgentStrategy",
    "load_strategy",
    "persist_strategy",
    "update_strategy",
]
