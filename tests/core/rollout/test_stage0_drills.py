"""Stage 0 integration drills for Phase 46 rollout infrastructure.

Each drill exercises one failure scenario end-to-end: breach detection,
rollback trigger, receipt generation, and kill switch enforcement.

Standing on Giants: Nygard (Release It!, 2007) · Fowler (canary, 2010)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from core.rollout.canary import CanaryRouter
from core.rollout.hmm_gate import HMMCallerGate
from core.rollout.metrics import Phase46Metrics
from core.rollout.rollback import RollbackEngine

# ------------------------------------------------------------------
# Env cleanup
# ------------------------------------------------------------------

_PHASE46_KEYS = [
    "BIZRA_PHASE46_SEARCH_ENABLED",
    "BIZRA_PHASE46_SEARCH_PERCENT",
    "BIZRA_PHASE46_GOT_BRIDGE_ENABLED",
    "BIZRA_PHASE46_GOT_BRIDGE_PERCENT",
    "BIZRA_PHASE46_HMM_ENABLED",
    "BIZRA_PHASE46_HMM_PERCENT",
    "BIZRA_PHASE46_CANARY_SALT",
    "BIZRA_PHASE46_HMM_CALLER_MODE",
    "BIZRA_PHASE46_HMM_ALLOWED_CALLER",
]


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {k: os.environ.pop(k, None) for k in _PHASE46_KEYS}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class FakeHMMEngine:
    def __init__(self) -> None:
        self.observations: list[str] = []

    def observe(self, symbol: str) -> str:
        self.observations.append(symbol)
        return f"observed:{symbol}"

    def predict_next(self) -> str:
        return "predicted"


# ------------------------------------------------------------------
# Drill: search_error_rate breach triggers rollback receipt
# ------------------------------------------------------------------


class TestDrillSearchErrorBreach:
    """Drill: search error breach triggers rollback receipt."""

    def test_search_error_breach_triggers_rollback(self, tmp_path: Path) -> None:
        receipt_dir = tmp_path / "receipts"
        engine = RollbackEngine(receipt_dir=str(receipt_dir))

        with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_PERCENT": "50"}):
            engine.evaluate("search_error_rate", breached=True)
            receipt = engine.evaluate("search_error_rate", breached=True)

        assert receipt is not None
        assert receipt.trigger == "search_error_rate"
        assert receipt.component == "search"
        assert receipt.action == "percent_zero"

        # Receipt file on disk
        files = list(receipt_dir.glob("rollback_*.json"))
        assert len(files) == 1


# ------------------------------------------------------------------
# Drill: GoT fallback breach triggers rollback receipt
# ------------------------------------------------------------------


class TestDrillGoTFallbackBreach:
    """Drill: GoT fallback breach triggers rollback receipt."""

    def test_got_fallback_breach_triggers_rollback(self, tmp_path: Path) -> None:
        receipt_dir = tmp_path / "receipts"
        engine = RollbackEngine(receipt_dir=str(receipt_dir))

        with patch.dict(os.environ, {"BIZRA_PHASE46_GOT_BRIDGE_PERCENT": "30"}):
            engine.evaluate("got_fallback_rate", breached=True)
            receipt = engine.evaluate("got_fallback_rate", breached=True)

        assert receipt is not None
        assert receipt.trigger == "got_fallback_rate"
        assert receipt.component == "got_bridge"


# ------------------------------------------------------------------
# Drill: HMM confidence breach triggers rollback receipt
# ------------------------------------------------------------------


class TestDrillHMMConfidenceBreach:
    """Drill: HMM confidence breach triggers rollback receipt."""

    def test_hmm_confidence_breach_triggers_rollback(self, tmp_path: Path) -> None:
        receipt_dir = tmp_path / "receipts"
        engine = RollbackEngine(receipt_dir=str(receipt_dir))

        with patch.dict(os.environ, {"BIZRA_PHASE46_HMM_PERCENT": "20"}):
            engine.evaluate("hmm_confidence", breached=True)
            receipt = engine.evaluate("hmm_confidence", breached=True)

        assert receipt is not None
        assert receipt.trigger == "hmm_confidence"
        assert receipt.component == "hmm"


# ------------------------------------------------------------------
# Drill: hard kill sets all booleans to "0"
# ------------------------------------------------------------------


class TestDrillHardKill:
    """Drill: hard kill sets all booleans to '0'."""

    def test_hard_kill_sets_all_to_zero(self, tmp_path: Path) -> None:
        receipt_dir = tmp_path / "receipts"
        engine = RollbackEngine(receipt_dir=str(receipt_dir))

        # All percents already 0 => hard kill
        with patch.dict(
            os.environ,
            {
                "BIZRA_PHASE46_SEARCH_PERCENT": "0",
                "BIZRA_PHASE46_GOT_BRIDGE_PERCENT": "0",
                "BIZRA_PHASE46_HMM_PERCENT": "0",
            },
        ):
            engine.evaluate("search_error_rate", breached=True)
            receipt = engine.evaluate("search_error_rate", breached=True)
            assert receipt is not None
            assert receipt.action == "hard_kill"
            assert os.environ.get("BIZRA_PHASE46_SEARCH_ENABLED") == "0"
            assert os.environ.get("BIZRA_PHASE46_GOT_BRIDGE_ENABLED") == "0"
            assert os.environ.get("BIZRA_PHASE46_HMM_ENABLED") == "0"


# ------------------------------------------------------------------
# Drill: receipt is valid JSON with required fields
# ------------------------------------------------------------------


class TestDrillReceiptValidity:
    """Drill: receipt is valid JSON with required fields."""

    def test_receipt_valid_json(self, tmp_path: Path) -> None:
        receipt_dir = tmp_path / "receipts"
        engine = RollbackEngine(receipt_dir=str(receipt_dir))

        with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_PERCENT": "10"}):
            engine.evaluate("search_error_rate", breached=True)
            engine.evaluate("search_error_rate", breached=True)

        files = list(receipt_dir.glob("rollback_*.json"))
        assert len(files) == 1

        data = json.loads(files[0].read_text())
        required_fields = {
            "timestamp",
            "trigger",
            "breach_count",
            "component",
            "action",
            "previous_config",
            "metrics_snapshot",
        }
        assert required_fields.issubset(set(data.keys()))


# ------------------------------------------------------------------
# Drill: rollback of one component leaves others unchanged
# ------------------------------------------------------------------


class TestDrillIsolatedRollback:
    """Drill: rollback of one component leaves others unchanged."""

    def test_rollback_one_leaves_others(self, tmp_path: Path) -> None:
        receipt_dir = tmp_path / "receipts"
        engine = RollbackEngine(receipt_dir=str(receipt_dir))

        with patch.dict(
            os.environ,
            {
                "BIZRA_PHASE46_SEARCH_PERCENT": "50",
                "BIZRA_PHASE46_GOT_BRIDGE_PERCENT": "40",
                "BIZRA_PHASE46_HMM_PERCENT": "30",
            },
        ):
            engine.evaluate("search_error_rate", breached=True)
            receipt = engine.evaluate("search_error_rate", breached=True)
            assert receipt is not None
            assert receipt.component == "search"
            # Search set to 0
            assert os.environ.get("BIZRA_PHASE46_SEARCH_PERCENT") == "0"
            # GoT and HMM unchanged
            assert os.environ.get("BIZRA_PHASE46_GOT_BRIDGE_PERCENT") == "40"
            assert os.environ.get("BIZRA_PHASE46_HMM_PERCENT") == "30"


# ------------------------------------------------------------------
# Kill switch takes precedence over percent in canary router
# ------------------------------------------------------------------


class TestDrillKillSwitchPrecedence:
    """Kill switch takes precedence over percent in canary router."""

    def test_kill_switch_overrides_percent(self) -> None:
        router = CanaryRouter(salt="drill-salt")
        with patch.dict(
            os.environ,
            {"BIZRA_PHASE46_SEARCH_ENABLED": "0"},
        ):
            # 50% would normally route some requests, but kill switch = "0"
            assert router.should_route("search", "any-key", percent=50) is False


# ------------------------------------------------------------------
# HMM gate blocks non-allowed caller
# ------------------------------------------------------------------


class TestDrillHMMGateBlocking:
    """HMM gate blocks non-allowed caller."""

    def test_hmm_gate_blocks(self) -> None:
        engine = FakeHMMEngine()
        with patch.dict(
            os.environ,
            {
                "BIZRA_PHASE46_HMM_CALLER_MODE": "single",
                "BIZRA_PHASE46_HMM_ALLOWED_CALLER": "mcp",
            },
        ):
            gate = HMMCallerGate(engine)

        result = gate.observe("search", "proactive")
        assert result is None
        assert engine.observations == []


# ------------------------------------------------------------------
# Metrics entropy computation correct
# ------------------------------------------------------------------


class TestDrillMetricsEntropy:
    """Metrics entropy computation correct."""

    def test_entropy_uniform_4_symbols(self) -> None:
        m = Phase46Metrics()
        for sym in ["a", "b", "c", "d"]:
            for _ in range(250):
                m.record_hmm_observation(sym)
        expected = math.log2(4)
        assert abs(m.observation_entropy() - expected) < 0.01


# ------------------------------------------------------------------
# Combined drill: all pass = gate passes
# ------------------------------------------------------------------


class TestDrillCombinedPass:
    """Combined drill: all components pass when properly configured."""

    def test_all_pass_gate_passes(self, tmp_path: Path) -> None:
        # 1. Canary routes at 100%
        router = CanaryRouter(salt="combined-drill")
        assert router.should_route("search", "key-1", percent=100) is True

        # 2. HMM gate allows correct caller
        hmm_engine = FakeHMMEngine()
        with patch.dict(
            os.environ,
            {
                "BIZRA_PHASE46_HMM_CALLER_MODE": "single",
                "BIZRA_PHASE46_HMM_ALLOWED_CALLER": "mcp",
            },
        ):
            gate = HMMCallerGate(hmm_engine)
        result = gate.observe("search", "mcp")
        assert result is not None

        # 3. Metrics record cleanly
        metrics = Phase46Metrics()
        metrics.inc("search_requests")
        metrics.inc("search_hits")
        assert metrics.compute_hit_rate() == 1.0

        # 4. No rollback (single clean evaluation)
        rb = RollbackEngine(receipt_dir=str(tmp_path / "receipts"), metrics=metrics)
        receipt = rb.evaluate("search_error_rate", breached=False)
        assert receipt is None

        # All four subsystems passed.
