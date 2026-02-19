"""Tests for Phase 46 GoT bridge integration in apex_engine._explore_thoughts.

Validates three code paths inside ``ApexSovereignEngine._explore_thoughts``:
1. Phase 46 GoT Bridge path (env var enabled + bridge importable)
2. Canonical got_engine.reason() path (env var disabled)
3. Fallback when the bridge raises an exception

All tests are fully mocked -- no live LLM, FAISS, or network access required.

Standing on Giants: Besta (GoT, 2024) . Shannon (1948)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Required dict contract keys returned by _explore_thoughts
# ---------------------------------------------------------------------------
# These keys are present in ALL code paths (including the error fallback).
_REQUIRED_KEYS = frozenset({
    "conclusion",
    "snr_score",
    "passes_threshold",
    "explored_nodes",
    "depth_reached",
    "best_path",
})

# These additional keys are present in the success paths (bridge, canonical,
# and no-reason fallback) but NOT in the exception-recovery path.
_SUCCESS_EXTRA_KEYS = frozenset({"ihsan_score"})


# ---------------------------------------------------------------------------
# Fake GoTBridgeResult used when the bridge path is active
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _FakeBridgeResult:
    answer: str = "bridge answer"
    snr_score: float = 0.93
    converged: bool = True
    hypotheses_explored: int = 7
    reasoning_depth: int = 4
    convergence_path: tuple = ("s1", "s2")


# ---------------------------------------------------------------------------
# Minimal ``self``-like object for calling _explore_thoughts directly
# ---------------------------------------------------------------------------
def _make_engine_self(got_engine: Any = None, got_max_depth: int = 5) -> MagicMock:
    """Build a minimal mock that looks enough like ApexSovereignEngine
    for ``_explore_thoughts`` to work when invoked as an unbound call."""
    mock = MagicMock()
    mock.got_engine = got_engine or MagicMock()
    mock.config = MagicMock()
    mock.config.got_max_depth = got_max_depth
    return mock


# We import the *class* so we can call _explore_thoughts as a bound method
# on our minimal mock.
from core.sovereign.apex_engine import ApexSovereignEngine


# =========================================================================
# 1. Bridge enabled -- GoTBridge preferred
# =========================================================================

class TestBridgeEnabled:
    """When BIZRA_PHASE46_GOT_BRIDGE_ENABLED=true AND GoTBridge is importable,
    the bridge path should be taken and canonical got_engine.reason() should
    NOT be called."""

    @patch.dict(os.environ, {"BIZRA_PHASE46_GOT_BRIDGE_ENABLED": "true"})
    async def test_bridge_path_used(self):
        fake_bridge_instance = MagicMock()
        fake_bridge_instance.reason = AsyncMock(return_value=_FakeBridgeResult())

        fake_bridge_cls = MagicMock(return_value=fake_bridge_instance)

        got_engine = MagicMock()
        got_engine.reason = AsyncMock()

        engine_self = _make_engine_self(got_engine=got_engine)

        with patch(
            "core.sovereign.apex_engine.GoTBridge",
            fake_bridge_cls,
            create=True,
        ), patch.dict(
            "sys.modules",
            {"core.reasoning.got_bridge": MagicMock(GoTBridge=fake_bridge_cls)},
        ):
            result = await ApexSovereignEngine._explore_thoughts(
                engine_self, "test query", {}, 0.85
            )

        assert result["conclusion"] == "bridge answer"
        assert result["snr_score"] == pytest.approx(0.93)
        assert result["passes_threshold"] is True
        got_engine.reason.assert_not_awaited()

    @patch.dict(os.environ, {"BIZRA_PHASE46_GOT_BRIDGE_ENABLED": "1"})
    async def test_bridge_result_mapped_correctly(self):
        bridge_result = _FakeBridgeResult(
            answer="mapped",
            snr_score=0.95,
            converged=True,
            hypotheses_explored=10,
            reasoning_depth=5,
            convergence_path=("a", "b", "c"),
        )
        fake_bridge_instance = MagicMock()
        fake_bridge_instance.reason = AsyncMock(return_value=bridge_result)
        fake_bridge_cls = MagicMock(return_value=fake_bridge_instance)

        engine_self = _make_engine_self()

        with patch.dict(
            "sys.modules",
            {"core.reasoning.got_bridge": MagicMock(GoTBridge=fake_bridge_cls)},
        ):
            result = await ApexSovereignEngine._explore_thoughts(
                engine_self, "q", {}, 0.80
            )

        assert result["explored_nodes"] == 10
        assert result["depth_reached"] == 5
        assert list(result["best_path"]) == ["a", "b", "c"]


# =========================================================================
# 2. Bridge disabled -- canonical got_engine.reason() used
# =========================================================================

class TestBridgeDisabled:
    """When the env var is unset (or '0'), the canonical GoT path is used."""

    @patch.dict(os.environ, {"BIZRA_PHASE46_GOT_BRIDGE_ENABLED": "0"})
    async def test_canonical_path_used(self):
        got_engine = MagicMock()
        got_engine.reason = AsyncMock(return_value={
            "conclusion": "canonical answer",
            "snr_score": 0.88,
            "ihsan_score": 0.90,
            "passes_threshold": True,
            "graph_stats": {"nodes_created": 4},
            "depth_reached": 3,
            "thoughts": ["t1", "t2"],
        })

        engine_self = _make_engine_self(got_engine=got_engine, got_max_depth=5)

        result = await ApexSovereignEngine._explore_thoughts(
            engine_self, "canonical query", {}, 0.80
        )

        got_engine.reason.assert_awaited_once()
        assert result["conclusion"] == "canonical answer"
        assert result["snr_score"] == pytest.approx(0.88)

    @patch.dict(os.environ, {}, clear=False)
    async def test_env_var_missing_uses_canonical(self):
        """When BIZRA_PHASE46_GOT_BRIDGE_ENABLED is not present at all."""
        # Remove it if present
        os.environ.pop("BIZRA_PHASE46_GOT_BRIDGE_ENABLED", None)

        got_engine = MagicMock()
        got_engine.reason = AsyncMock(return_value={
            "conclusion": "fallback canonical",
            "snr_score": 0.87,
            "ihsan_score": 0.89,
            "passes_threshold": True,
            "graph_stats": {"nodes_created": 2},
            "depth_reached": 2,
            "thoughts": ["h1"],
        })

        engine_self = _make_engine_self(got_engine=got_engine)
        result = await ApexSovereignEngine._explore_thoughts(
            engine_self, "q", {}, 0.80
        )
        assert result["conclusion"] == "fallback canonical"


# =========================================================================
# 3. Bridge fails -- falls back to canonical path
# =========================================================================

class TestBridgeFailsFallback:
    """When the bridge import or execution raises, the method must fall
    through to the canonical got_engine.reason() path."""

    @patch.dict(os.environ, {"BIZRA_PHASE46_GOT_BRIDGE_ENABLED": "true"})
    async def test_import_error_fallback(self):
        """If core.reasoning.got_bridge cannot be imported, canonical path is used."""
        got_engine = MagicMock()
        got_engine.reason = AsyncMock(return_value={
            "conclusion": "fallback used",
            "snr_score": 0.86,
            "ihsan_score": 0.88,
            "passes_threshold": True,
            "graph_stats": {"nodes_created": 1},
            "depth_reached": 1,
            "thoughts": [],
        })

        engine_self = _make_engine_self(got_engine=got_engine)

        # Ensure importing GoTBridge raises ImportError
        import sys
        saved = sys.modules.get("core.reasoning.got_bridge")
        sys.modules["core.reasoning.got_bridge"] = None  # type: ignore[assignment]
        try:
            result = await ApexSovereignEngine._explore_thoughts(
                engine_self, "q", {}, 0.80
            )
        finally:
            if saved is not None:
                sys.modules["core.reasoning.got_bridge"] = saved
            else:
                sys.modules.pop("core.reasoning.got_bridge", None)

        got_engine.reason.assert_awaited_once()
        assert result["conclusion"] == "fallback used"

    @patch.dict(os.environ, {"BIZRA_PHASE46_GOT_BRIDGE_ENABLED": "true"})
    async def test_bridge_runtime_error_fallback(self):
        """If the bridge raises at runtime, canonical path is used."""
        fake_bridge_instance = MagicMock()
        fake_bridge_instance.reason = AsyncMock(
            side_effect=RuntimeError("bridge exploded")
        )
        fake_bridge_cls = MagicMock(return_value=fake_bridge_instance)

        got_engine = MagicMock()
        got_engine.reason = AsyncMock(return_value={
            "conclusion": "recovered",
            "snr_score": 0.85,
            "ihsan_score": 0.87,
            "passes_threshold": True,
            "graph_stats": {"nodes_created": 2},
            "depth_reached": 2,
            "thoughts": ["r1"],
        })

        engine_self = _make_engine_self(got_engine=got_engine)

        with patch.dict(
            "sys.modules",
            {"core.reasoning.got_bridge": MagicMock(GoTBridge=fake_bridge_cls)},
        ):
            result = await ApexSovereignEngine._explore_thoughts(
                engine_self, "q", {}, 0.80
            )

        got_engine.reason.assert_awaited_once()
        assert result["conclusion"] == "recovered"


# =========================================================================
# 4. Dict contract -- all required keys present
# =========================================================================

class TestDictContract:
    """Regardless of code path, the returned dict must contain all required keys."""

    @patch.dict(os.environ, {"BIZRA_PHASE46_GOT_BRIDGE_ENABLED": "true"})
    async def test_bridge_path_contract(self):
        fake_bridge_instance = MagicMock()
        fake_bridge_instance.reason = AsyncMock(return_value=_FakeBridgeResult())
        fake_bridge_cls = MagicMock(return_value=fake_bridge_instance)

        engine_self = _make_engine_self()

        with patch.dict(
            "sys.modules",
            {"core.reasoning.got_bridge": MagicMock(GoTBridge=fake_bridge_cls)},
        ):
            result = await ApexSovereignEngine._explore_thoughts(
                engine_self, "q", {}, 0.80
            )

        all_expected = _REQUIRED_KEYS | _SUCCESS_EXTRA_KEYS
        assert all_expected.issubset(result.keys()), (
            f"Missing keys: {all_expected - result.keys()}"
        )

    @patch.dict(os.environ, {"BIZRA_PHASE46_GOT_BRIDGE_ENABLED": "0"})
    async def test_canonical_path_contract(self):
        got_engine = MagicMock()
        got_engine.reason = AsyncMock(return_value={
            "conclusion": "c",
            "snr_score": 0.88,
            "ihsan_score": 0.90,
            "passes_threshold": True,
            "graph_stats": {"nodes_created": 3},
            "depth_reached": 2,
            "thoughts": [],
        })

        engine_self = _make_engine_self(got_engine=got_engine)
        result = await ApexSovereignEngine._explore_thoughts(
            engine_self, "q", {}, 0.80
        )

        all_expected = _REQUIRED_KEYS | _SUCCESS_EXTRA_KEYS
        assert all_expected.issubset(result.keys()), (
            f"Missing keys: {all_expected - result.keys()}"
        )

    async def test_fallback_path_contract(self):
        """When got_engine has no .reason method, the fallback dict is returned."""
        got_engine = MagicMock(spec=[])  # no methods at all

        engine_self = _make_engine_self(got_engine=got_engine)
        # Remove env var so bridge is not attempted
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BIZRA_PHASE46_GOT_BRIDGE_ENABLED", None)
            result = await ApexSovereignEngine._explore_thoughts(
                engine_self, "q", {}, 0.80
            )

        all_expected = _REQUIRED_KEYS | _SUCCESS_EXTRA_KEYS
        assert all_expected.issubset(result.keys()), (
            f"Missing keys: {all_expected - result.keys()}"
        )

    async def test_error_path_contract(self):
        """When got_engine.reason() raises, the error fallback dict is returned."""
        got_engine = MagicMock()
        got_engine.reason = AsyncMock(side_effect=RuntimeError("exploded"))

        engine_self = _make_engine_self(got_engine=got_engine)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BIZRA_PHASE46_GOT_BRIDGE_ENABLED", None)
            result = await ApexSovereignEngine._explore_thoughts(
                engine_self, "q", {}, 0.80
            )

        assert _REQUIRED_KEYS.issubset(result.keys()), (
            f"Missing keys: {_REQUIRED_KEYS - result.keys()}"
        )
        assert result["passes_threshold"] is False
