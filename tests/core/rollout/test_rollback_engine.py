"""Tests for core.rollout.rollback — strict rollback automation.

Standing on Giants: Nygard (Release It!, 2007) · Fowler (canary, 2010)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from core.rollout.metrics import Phase46Metrics
from core.rollout.rollback import RollbackEngine, RollbackReceipt


# ------------------------------------------------------------------
# Env vars to clean between tests
# ------------------------------------------------------------------

_PHASE46_KEYS = [
    "BIZRA_PHASE46_SEARCH_ENABLED",
    "BIZRA_PHASE46_SEARCH_PERCENT",
    "BIZRA_PHASE46_GOT_BRIDGE_ENABLED",
    "BIZRA_PHASE46_GOT_BRIDGE_PERCENT",
    "BIZRA_PHASE46_HMM_ENABLED",
    "BIZRA_PHASE46_HMM_PERCENT",
]


@pytest.fixture(autouse=True)
def _clean_env():
    """Strip Phase46 env vars before and restore after each test."""
    saved = {k: os.environ.pop(k, None) for k in _PHASE46_KEYS}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


@pytest.fixture()
def engine(tmp_path: Path) -> RollbackEngine:
    """RollbackEngine with receipts in a temp directory."""
    return RollbackEngine(receipt_dir=str(tmp_path / "receipts"))


@pytest.fixture()
def engine_with_metrics(tmp_path: Path) -> RollbackEngine:
    """RollbackEngine with a real Phase46Metrics instance."""
    m = Phase46Metrics()
    m.inc("search_requests", 100)
    return RollbackEngine(receipt_dir=str(tmp_path / "receipts"), metrics=m)


# ------------------------------------------------------------------
# Breach window logic
# ------------------------------------------------------------------


class TestBreachWindows:
    """Two consecutive breaches trigger rollback; clean window resets."""

    def test_single_breach_does_not_trigger(self, engine: RollbackEngine) -> None:
        result = engine.evaluate("search_error_rate", breached=True)
        assert result is None

    def test_two_consecutive_breaches_trigger_rollback(
        self, engine: RollbackEngine
    ) -> None:
        with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_PERCENT": "50"}):
            engine.evaluate("search_error_rate", breached=True)
            result = engine.evaluate("search_error_rate", breached=True)
        assert result is not None
        assert isinstance(result, RollbackReceipt)
        assert result.trigger == "search_error_rate"

    def test_clean_window_resets_breach_counter(
        self, engine: RollbackEngine
    ) -> None:
        engine.evaluate("search_error_rate", breached=True)
        # Clean window resets counter
        engine.evaluate("search_error_rate", breached=False)
        # Next breach is the first again
        result = engine.evaluate("search_error_rate", breached=True)
        assert result is None

    def test_unknown_metric_returns_none(self, engine: RollbackEngine) -> None:
        result = engine.evaluate("nonexistent_metric", breached=True)
        assert result is None


# ------------------------------------------------------------------
# Rollback actions
# ------------------------------------------------------------------


class TestRollbackActions:
    """Rollback sets percent to 0 or performs hard kill."""

    def test_rollback_sets_percent_to_zero(self, engine: RollbackEngine) -> None:
        with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_PERCENT": "50"}):
            engine.evaluate("search_error_rate", breached=True)
            receipt = engine.evaluate("search_error_rate", breached=True)
            assert receipt is not None
            assert receipt.action == "percent_zero"
            assert os.environ.get("BIZRA_PHASE46_SEARCH_PERCENT") == "0"

    def test_hard_kill_when_all_percents_zero(
        self, engine: RollbackEngine
    ) -> None:
        """When all percents are already 0, rollback escalates to hard kill."""
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
            assert receipt.component == "all"
            # All ENABLED flags should be "0"
            assert os.environ.get("BIZRA_PHASE46_SEARCH_ENABLED") == "0"
            assert os.environ.get("BIZRA_PHASE46_GOT_BRIDGE_ENABLED") == "0"
            assert os.environ.get("BIZRA_PHASE46_HMM_ENABLED") == "0"


# ------------------------------------------------------------------
# Rollback order
# ------------------------------------------------------------------


class TestRollbackOrder:
    """Reverse activation order: HMM -> GoT -> Search -> hard kill."""

    def test_hmm_first_for_cross_cutting_trigger(
        self, engine: RollbackEngine
    ) -> None:
        """Cross-cutting trigger (latency) rolls back HMM first if active."""
        with patch.dict(
            os.environ,
            {
                "BIZRA_PHASE46_SEARCH_PERCENT": "50",
                "BIZRA_PHASE46_GOT_BRIDGE_PERCENT": "50",
                "BIZRA_PHASE46_HMM_PERCENT": "50",
            },
        ):
            engine.evaluate("latency_regression", breached=True)
            receipt = engine.evaluate("latency_regression", breached=True)
        assert receipt is not None
        assert receipt.component == "hmm"

    def test_component_specific_trigger_targets_that_component(
        self, engine: RollbackEngine
    ) -> None:
        """got_fallback_rate targets got_bridge first."""
        with patch.dict(
            os.environ,
            {
                "BIZRA_PHASE46_SEARCH_PERCENT": "50",
                "BIZRA_PHASE46_GOT_BRIDGE_PERCENT": "50",
                "BIZRA_PHASE46_HMM_PERCENT": "50",
            },
        ):
            engine.evaluate("got_fallback_rate", breached=True)
            receipt = engine.evaluate("got_fallback_rate", breached=True)
        assert receipt is not None
        assert receipt.component == "got_bridge"

    def test_hmm_confidence_targets_hmm(self, engine: RollbackEngine) -> None:
        with patch.dict(os.environ, {"BIZRA_PHASE46_HMM_PERCENT": "30"}):
            engine.evaluate("hmm_confidence", breached=True)
            receipt = engine.evaluate("hmm_confidence", breached=True)
        assert receipt is not None
        assert receipt.component == "hmm"


# ------------------------------------------------------------------
# Receipt persistence
# ------------------------------------------------------------------


class TestReceiptPersistence:
    """Rollback receipts are persisted as valid JSON."""

    def test_receipt_persisted_as_valid_json(
        self, engine: RollbackEngine, tmp_path: Path
    ) -> None:
        receipt_dir = tmp_path / "receipts"
        with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_PERCENT": "50"}):
            engine.evaluate("search_error_rate", breached=True)
            receipt = engine.evaluate("search_error_rate", breached=True)

        assert receipt is not None
        files = list(receipt_dir.glob("rollback_*.json"))
        assert len(files) == 1

        data = json.loads(files[0].read_text())
        assert "timestamp" in data
        assert "trigger" in data
        assert "breach_count" in data
        assert "component" in data
        assert "action" in data
        assert "previous_config" in data
        assert "metrics_snapshot" in data

    def test_receipt_contains_previous_config(
        self, engine: RollbackEngine
    ) -> None:
        with patch.dict(
            os.environ,
            {
                "BIZRA_PHASE46_SEARCH_PERCENT": "40",
                "BIZRA_PHASE46_HMM_PERCENT": "30",
            },
        ):
            engine.evaluate("search_error_rate", breached=True)
            receipt = engine.evaluate("search_error_rate", breached=True)
        assert receipt is not None
        assert receipt.previous_config["BIZRA_PHASE46_SEARCH_PERCENT"] == "40"
        assert receipt.previous_config["BIZRA_PHASE46_HMM_PERCENT"] == "30"

    def test_receipt_with_metrics_snapshot(
        self, engine_with_metrics: RollbackEngine
    ) -> None:
        with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_PERCENT": "50"}):
            engine_with_metrics.evaluate("search_error_rate", breached=True)
            receipt = engine_with_metrics.evaluate("search_error_rate", breached=True)
        assert receipt is not None
        assert "counters" in receipt.metrics_snapshot


# ------------------------------------------------------------------
# Status property
# ------------------------------------------------------------------


class TestStatus:
    """status property reports breach window state."""

    def test_status_reports_breach_windows(self, engine: RollbackEngine) -> None:
        engine.evaluate("search_error_rate", breached=True)
        status = engine.status
        assert "breach_windows" in status
        assert "search_error_rate" in status["breach_windows"]
        assert status["breach_windows"]["search_error_rate"]["consecutive"] == 1
        assert status["breach_windows"]["search_error_rate"]["last_breached"] is True

    def test_status_rollback_not_in_progress(self, engine: RollbackEngine) -> None:
        assert engine.status["rollback_in_progress"] is False

    def test_status_receipts_dir(self, engine: RollbackEngine, tmp_path: Path) -> None:
        assert "receipts_dir" in engine.status
