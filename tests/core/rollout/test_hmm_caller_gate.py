"""Tests for core.rollout.hmm_gate — HMM single-caller isolation gate.

Standing on Giants: Rabiner (HMM, 1989) · Lamport (distributed state, 1978)
"""

from __future__ import annotations

import os
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

import pytest

from core.rollout.hmm_gate import HMMCallerGate

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

_PHASE46_KEYS = [
    "BIZRA_PHASE46_HMM_CALLER_MODE",
    "BIZRA_PHASE46_HMM_ALLOWED_CALLER",
]


@pytest.fixture(autouse=True)
def _clean_env():
    """Strip HMM env vars before each test."""
    saved = {k: os.environ.pop(k, None) for k in _PHASE46_KEYS}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


class FakeHMMEngine:
    """Minimal fake HMM engine for testing gate behaviour."""

    def __init__(self) -> None:
        self.observations: list[str] = []
        self._next_prediction = "predicted_symbol"

    def observe(self, symbol: str) -> str:
        self.observations.append(symbol)
        return f"observed:{symbol}"

    def predict_next(self) -> str:
        return self._next_prediction


@pytest.fixture()
def engine() -> FakeHMMEngine:
    return FakeHMMEngine()


def _make_gate(
    engine: Any,
    mode: str = "single",
    allowed: str = "mcp",
) -> HMMCallerGate:
    """Build a gate with explicit mode and allowed caller via env."""
    env = {}
    env["BIZRA_PHASE46_HMM_CALLER_MODE"] = mode
    env["BIZRA_PHASE46_HMM_ALLOWED_CALLER"] = allowed
    with patch.dict(os.environ, env):
        return HMMCallerGate(engine)


# ------------------------------------------------------------------
# Single mode tests
# ------------------------------------------------------------------


class TestSingleMode:
    """In 'single' mode only the allowed caller can observe."""

    def test_single_mode_accepts_allowed_caller(self, engine: FakeHMMEngine) -> None:
        gate = _make_gate(engine, mode="single", allowed="mcp")
        result = gate.observe("search", "mcp")
        assert result == "observed:search"
        assert engine.observations == ["search"]

    def test_single_mode_drops_non_allowed_caller(self, engine: FakeHMMEngine) -> None:
        gate = _make_gate(engine, mode="single", allowed="mcp")
        result = gate.observe("search", "proactive")
        assert result is None
        assert engine.observations == []


# ------------------------------------------------------------------
# Multi mode tests
# ------------------------------------------------------------------


class TestMultiMode:
    """In 'multi' mode all callers can observe."""

    def test_multi_mode_accepts_all(self, engine: FakeHMMEngine) -> None:
        gate = _make_gate(engine, mode="multi", allowed="mcp")
        r1 = gate.observe("search", "mcp")
        r2 = gate.observe("edit", "proactive")
        r3 = gate.observe("click", "unknown_agent")
        assert r1 is not None
        assert r2 is not None
        assert r3 is not None
        assert len(engine.observations) == 3


# ------------------------------------------------------------------
# Disabled mode tests
# ------------------------------------------------------------------


class TestDisabledMode:
    """In 'disabled' mode no callers can observe."""

    def test_disabled_mode_drops_all(self, engine: FakeHMMEngine) -> None:
        gate = _make_gate(engine, mode="disabled", allowed="mcp")
        r1 = gate.observe("search", "mcp")
        r2 = gate.observe("search", "proactive")
        assert r1 is None
        assert r2 is None
        assert engine.observations == []


# ------------------------------------------------------------------
# Predict (always allowed)
# ------------------------------------------------------------------


class TestPredict:
    """predict() is always allowed regardless of mode."""

    @pytest.mark.parametrize("mode", ["single", "multi", "disabled"])
    def test_predict_always_allowed(self, engine: FakeHMMEngine, mode: str) -> None:
        gate = _make_gate(engine, mode=mode, allowed="mcp")
        result = gate.predict("blocked_caller")
        assert result == "predicted_symbol"

    def test_predict_with_none_engine(self) -> None:
        """predict returns None when engine is None."""
        with patch.dict(os.environ, {"BIZRA_PHASE46_HMM_CALLER_MODE": "single"}):
            gate = HMMCallerGate(None)
        result = gate.predict("anyone")
        assert result is None

    def test_predict_handles_engine_exception(self) -> None:
        """predict returns None when engine.predict_next raises."""

        class BrokenEngine:
            def predict_next(self) -> None:
                raise RuntimeError("boom")

        gate = _make_gate(BrokenEngine(), mode="single", allowed="mcp")
        result = gate.predict("mcp")
        assert result is None


# ------------------------------------------------------------------
# Stats / telemetry
# ------------------------------------------------------------------


class TestStats:
    """Stats track accepted, dropped counts and timestamps."""

    def test_stats_track_accepted_and_dropped(self, engine: FakeHMMEngine) -> None:
        gate = _make_gate(engine, mode="single", allowed="mcp")
        gate.observe("s1", "mcp")
        gate.observe("s2", "mcp")
        gate.observe("s3", "proactive")

        stats = gate.stats
        assert stats["accepted_count"] == 2
        assert stats["dropped_count"] == 1

    def test_stats_track_dropped_callers_by_identity(
        self, engine: FakeHMMEngine
    ) -> None:
        gate = _make_gate(engine, mode="single", allowed="mcp")
        gate.observe("s1", "proactive")
        gate.observe("s2", "proactive")
        gate.observe("s3", "unknown_agent")

        stats = gate.stats
        assert stats["dropped_callers"]["proactive"] == 2
        assert stats["dropped_callers"]["unknown_agent"] == 1

    def test_stats_include_timestamps(self, engine: FakeHMMEngine) -> None:
        gate = _make_gate(engine, mode="single", allowed="mcp")
        gate.observe("s1", "mcp")
        gate.observe("s2", "proactive")

        stats = gate.stats
        # Accepted timestamp must be set.
        assert stats["last_accepted"] is not None
        # Dropped timestamp must be set.
        assert stats["last_dropped"] is not None
        # Both should be valid ISO 8601 strings.
        datetime.fromisoformat(stats["last_accepted"])
        datetime.fromisoformat(stats["last_dropped"])

    def test_stats_mode_and_allowed(self, engine: FakeHMMEngine) -> None:
        gate = _make_gate(engine, mode="single", allowed="mcp")
        stats = gate.stats
        assert stats["mode"] == "single"
        assert stats["allowed_caller"] == "mcp"
