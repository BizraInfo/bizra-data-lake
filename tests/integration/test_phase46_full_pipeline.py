"""End-to-end Phase 46 pipeline integration test.

Validates that CanaryRouter, Phase46Metrics, HMMCallerGate, and
RollbackEngine function as a coherent pipeline — from request routing
through metrics collection to automatic rollback.

Standing on Giants: Fowler (canary, 2010) · Shannon (metrics, 1948) ·
Nygard (Release It!, 2007) · Rabiner (HMM, 1989)

Artifact: tests/integration/test_phase46_full_pipeline.py
"""

import os
import json
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest

from core.rollout.canary import CanaryRouter
from core.rollout.metrics import Phase46Metrics
from core.rollout.hmm_gate import HMMCallerGate
from core.rollout.rollback import RollbackEngine, RollbackReceipt


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Clear all Phase 46 env vars before each test."""
    for key in (
        "BIZRA_PHASE46_SEARCH_ENABLED",
        "BIZRA_PHASE46_SEARCH_PERCENT",
        "BIZRA_PHASE46_GOT_BRIDGE_ENABLED",
        "BIZRA_PHASE46_GOT_BRIDGE_PERCENT",
        "BIZRA_PHASE46_HMM_ENABLED",
        "BIZRA_PHASE46_HMM_PERCENT",
        "BIZRA_PHASE46_CANARY_SALT",
        "BIZRA_PHASE46_HMM_CALLER_MODE",
        "BIZRA_PHASE46_HMM_ALLOWED_CALLER",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def metrics():
    return Phase46Metrics()


@pytest.fixture
def canary():
    return CanaryRouter(salt="test-pipeline-salt")


@pytest.fixture
def receipt_dir():
    d = tempfile.mkdtemp(prefix="rollback_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mock_hmm():
    engine = MagicMock()
    engine.observe.return_value = MagicMock(
        prediction_confidence=0.87,
        predicted_next="search",
    )
    engine.predict_next.return_value = MagicMock(
        prediction_confidence=0.87,
        predicted_next="search",
    )
    return engine


# ======================================================================
# Stage 1: Canary → Metrics pipeline
# ======================================================================


class TestCanaryToMetrics:
    """Route requests through canary, record metrics for routed ones."""

    def test_100pct_routes_all_and_records_metrics(
        self, monkeypatch, canary, metrics
    ):
        """At 100% canary, every request is routed and counted."""
        monkeypatch.setenv("BIZRA_PHASE46_SEARCH_ENABLED", "1")

        routed_count = 0
        for i in range(50):
            if canary.should_route("search", f"query-{i}", percent=100):
                routed_count += 1
                metrics.inc("search_requests")
                metrics.record_latency("search", 12.5 + i * 0.1)

        assert routed_count == 50
        assert metrics.get_counter("search_requests") == 50

        snap = metrics.snapshot()
        assert snap["search"]["latency_p50_ms"] > 0
        assert snap["search"]["latency_p95_ms"] >= snap["search"]["latency_p50_ms"]

    def test_0pct_routes_none(self, canary, metrics):
        """At 0% canary, no requests routed — metrics stay at zero."""
        for i in range(50):
            if canary.should_route("search", f"query-{i}", percent=0):
                metrics.inc("search_requests")

        assert metrics.get_counter("search_requests") == 0

    def test_kill_switch_overrides_percent(
        self, monkeypatch, canary, metrics
    ):
        """Kill switch OFF disables routing even at 100%."""
        monkeypatch.setenv("BIZRA_PHASE46_SEARCH_ENABLED", "0")
        monkeypatch.setenv("BIZRA_PHASE46_SEARCH_PERCENT", "100")

        for i in range(20):
            if canary.should_route("search", f"query-{i}"):
                metrics.inc("search_requests")

        assert metrics.get_counter("search_requests") == 0

    def test_partial_canary_deterministic(self, canary, metrics):
        """50% canary is deterministic — same key always routes the same."""
        results_a = [
            canary.should_route("search", f"q-{i}", percent=50)
            for i in range(100)
        ]
        results_b = [
            canary.should_route("search", f"q-{i}", percent=50)
            for i in range(100)
        ]
        assert results_a == results_b

        routed = sum(1 for r in results_a if r)
        # With 100 keys at 50%, expect roughly 40-60 routed
        assert 20 < routed < 80

    def test_hit_rate_tracking(self, canary, metrics):
        """Hit rate = search_hits / search_requests."""
        for i in range(10):
            metrics.inc("search_requests")
            if i < 7:
                metrics.inc("search_hits")

        assert abs(metrics.compute_hit_rate() - 0.7) < 0.01


# ======================================================================
# Stage 2: HMM gate → Metrics pipeline
# ======================================================================


class TestHMMGateToMetrics:
    """HMM caller gate feeds observations through to metrics."""

    def test_single_mode_accepts_allowed_caller(
        self, monkeypatch, mock_hmm, metrics
    ):
        monkeypatch.setenv("BIZRA_PHASE46_HMM_CALLER_MODE", "single")
        monkeypatch.setenv("BIZRA_PHASE46_HMM_ALLOWED_CALLER", "mcp")
        gate = HMMCallerGate(mock_hmm)

        result = gate.observe("search", "mcp")
        assert result is not None
        metrics.inc("hmm_requests")
        metrics.record_hmm_confidence(result.prediction_confidence)
        metrics.record_hmm_observation("search")

        assert metrics.get_counter("hmm_requests") == 1
        snap = metrics.snapshot()
        assert snap["hmm"]["confidence_p50"] == pytest.approx(0.87, abs=0.01)
        assert snap["hmm"]["observation_entropy"] == 0.0  # single symbol

    def test_single_mode_drops_wrong_caller(
        self, monkeypatch, mock_hmm, metrics
    ):
        monkeypatch.setenv("BIZRA_PHASE46_HMM_CALLER_MODE", "single")
        monkeypatch.setenv("BIZRA_PHASE46_HMM_ALLOWED_CALLER", "mcp")
        gate = HMMCallerGate(mock_hmm)

        result = gate.observe("search", "proactive")
        assert result is None
        assert gate.stats["dropped_count"] == 1
        assert gate.stats["dropped_callers"]["proactive"] == 1

    def test_disabled_mode_drops_all(
        self, monkeypatch, mock_hmm
    ):
        monkeypatch.setenv("BIZRA_PHASE46_HMM_CALLER_MODE", "disabled")
        gate = HMMCallerGate(mock_hmm)

        result = gate.observe("search", "mcp")
        assert result is None
        assert gate.stats["dropped_count"] == 1

    def test_predict_always_allowed(
        self, monkeypatch, mock_hmm
    ):
        """Predict is read-only — allowed even in single mode for wrong caller."""
        monkeypatch.setenv("BIZRA_PHASE46_HMM_CALLER_MODE", "single")
        monkeypatch.setenv("BIZRA_PHASE46_HMM_ALLOWED_CALLER", "mcp")
        gate = HMMCallerGate(mock_hmm)

        result = gate.predict("proactive")
        assert result is not None
        assert result.predicted_next == "search"

    def test_multi_mode_accepts_all(
        self, monkeypatch, mock_hmm, metrics
    ):
        monkeypatch.setenv("BIZRA_PHASE46_HMM_CALLER_MODE", "multi")
        gate = HMMCallerGate(mock_hmm)

        for caller in ("mcp", "proactive", "apex"):
            result = gate.observe("edit", caller)
            assert result is not None
            metrics.inc("hmm_requests")

        assert metrics.get_counter("hmm_requests") == 3
        assert gate.stats["accepted_count"] == 3
        assert gate.stats["dropped_count"] == 0

    def test_entropy_with_multiple_symbols(self, metrics):
        """Shannon entropy increases with symbol diversity."""
        # Single symbol → entropy = 0
        metrics.record_hmm_observation("search")
        metrics.record_hmm_observation("search")
        e1 = metrics.observation_entropy()
        assert e1 == 0.0

        # Two equally frequent symbols → entropy = 1.0
        metrics2 = Phase46Metrics()
        metrics2.record_hmm_observation("search")
        metrics2.record_hmm_observation("edit")
        e2 = metrics2.observation_entropy()
        assert abs(e2 - 1.0) < 0.01


# ======================================================================
# Stage 3: Metrics → Rollback pipeline
# ======================================================================


class TestMetricsToRollback:
    """Metrics feed into rollback evaluation — breaches trigger rollback."""

    def test_single_breach_no_rollback(self, metrics, receipt_dir):
        """One breach is not enough — need 2 consecutive."""
        engine = RollbackEngine(receipt_dir=receipt_dir, metrics=metrics)

        result = engine.evaluate("search_error_rate", breached=True)
        assert result is None

    def test_two_consecutive_breaches_trigger_rollback(
        self, monkeypatch, metrics, receipt_dir
    ):
        """Two consecutive breaches on same metric → rollback."""
        monkeypatch.setenv("BIZRA_PHASE46_SEARCH_PERCENT", "50")
        engine = RollbackEngine(receipt_dir=receipt_dir, metrics=metrics)

        # First breach
        r1 = engine.evaluate("search_error_rate", breached=True)
        assert r1 is None

        # Second breach
        r2 = engine.evaluate("search_error_rate", breached=True)
        assert r2 is not None
        assert isinstance(r2, RollbackReceipt)
        assert r2.trigger == "search_error_rate"
        assert r2.component == "search"
        assert r2.action == "percent_zero"
        assert r2.breach_count >= 2

        # Env var was zeroed
        assert os.environ.get("BIZRA_PHASE46_SEARCH_PERCENT") == "0"

    def test_clean_window_resets_breach_count(
        self, monkeypatch, metrics, receipt_dir
    ):
        """A clean evaluation resets the counter — no rollback on next breach."""
        monkeypatch.setenv("BIZRA_PHASE46_SEARCH_PERCENT", "50")
        engine = RollbackEngine(receipt_dir=receipt_dir, metrics=metrics)

        engine.evaluate("search_error_rate", breached=True)  # count=1
        engine.evaluate("search_error_rate", breached=False)  # reset to 0
        result = engine.evaluate("search_error_rate", breached=True)  # count=1
        assert result is None  # Still only 1

    def test_rollback_reverse_order_hmm_first(
        self, monkeypatch, metrics, receipt_dir
    ):
        """When all components are active, HMM rolls back first."""
        monkeypatch.setenv("BIZRA_PHASE46_SEARCH_PERCENT", "50")
        monkeypatch.setenv("BIZRA_PHASE46_GOT_BRIDGE_PERCENT", "30")
        monkeypatch.setenv("BIZRA_PHASE46_HMM_PERCENT", "20")
        engine = RollbackEngine(receipt_dir=receipt_dir, metrics=metrics)

        # latency_regression is cross-cutting → rolls back HMM first
        engine.evaluate("latency_regression", breached=True)
        receipt = engine.evaluate("latency_regression", breached=True)

        assert receipt is not None
        assert receipt.component == "hmm"
        assert os.environ.get("BIZRA_PHASE46_HMM_PERCENT") == "0"
        # search and got_bridge untouched
        assert os.environ.get("BIZRA_PHASE46_SEARCH_PERCENT") == "50"
        assert os.environ.get("BIZRA_PHASE46_GOT_BRIDGE_PERCENT") == "30"

    def test_rollback_receipt_persisted(
        self, monkeypatch, metrics, receipt_dir
    ):
        """Rollback receipt is written as JSON file."""
        monkeypatch.setenv("BIZRA_PHASE46_SEARCH_PERCENT", "50")
        engine = RollbackEngine(receipt_dir=receipt_dir, metrics=metrics)

        engine.evaluate("search_error_rate", breached=True)
        engine.evaluate("search_error_rate", breached=True)

        files = os.listdir(receipt_dir)
        assert len(files) == 1
        assert files[0].startswith("rollback_")
        assert files[0].endswith(".json")

        with open(os.path.join(receipt_dir, files[0])) as f:
            data = json.load(f)
        assert data["trigger"] == "search_error_rate"
        assert data["component"] == "search"

    def test_rollback_includes_metrics_snapshot(
        self, monkeypatch, metrics, receipt_dir
    ):
        """Rollback receipt captures the metrics snapshot at rollback time."""
        monkeypatch.setenv("BIZRA_PHASE46_SEARCH_PERCENT", "50")

        metrics.inc("search_requests", 100)
        metrics.inc("search_hits", 42)
        metrics.record_latency("search", 55.0)

        engine = RollbackEngine(receipt_dir=receipt_dir, metrics=metrics)
        engine.evaluate("search_error_rate", breached=True)
        receipt = engine.evaluate("search_error_rate", breached=True)

        assert receipt is not None
        snap = receipt.metrics_snapshot
        assert snap["counters"]["search_requests"] == 100
        assert snap["counters"]["search_hits"] == 42
        assert snap["search"]["latency_p50_ms"] == 55.0


# ======================================================================
# Stage 4: Full pipeline — Canary → HMM Gate → Metrics → Rollback
# ======================================================================


class TestFullPipeline:
    """End-to-end: canary routing feeds HMM observations, metrics collect,
    breach threshold triggers rollback, verify post-rollback state."""

    def test_full_lifecycle(
        self, monkeypatch, canary, mock_hmm, receipt_dir
    ):
        """Complete Phase 46 lifecycle: route → observe → metrics → rollback."""
        # Setup: all components at 100% via PERCENT (not ENABLED).
        # ENABLED is NOT set — this lets percent routing be the active gate.
        # Rollback zeros PERCENT, which then blocks routing.
        monkeypatch.setenv("BIZRA_PHASE46_SEARCH_PERCENT", "100")
        monkeypatch.setenv("BIZRA_PHASE46_GOT_BRIDGE_PERCENT", "100")
        monkeypatch.setenv("BIZRA_PHASE46_HMM_PERCENT", "100")
        monkeypatch.setenv("BIZRA_PHASE46_HMM_CALLER_MODE", "single")
        monkeypatch.setenv("BIZRA_PHASE46_HMM_ALLOWED_CALLER", "mcp")

        metrics = Phase46Metrics()
        hmm_gate = HMMCallerGate(mock_hmm)
        rollback = RollbackEngine(receipt_dir=receipt_dir, metrics=metrics)

        # Phase A: Normal operation — canary routes, metrics collect
        for i in range(20):
            query = f"user-query-{i}"

            if canary.should_route("search", query):
                metrics.inc("search_requests")
                metrics.record_latency("search", 15.0 + i * 0.5)
                if i < 15:
                    metrics.inc("search_hits")

            if canary.should_route("hmm", query):
                result = hmm_gate.observe("search", "mcp")
                if result:
                    metrics.inc("hmm_requests")
                    metrics.record_hmm_confidence(result.prediction_confidence)
                    metrics.record_hmm_observation("search")

        # Verify normal state
        assert metrics.get_counter("search_requests") == 20
        assert metrics.get_counter("search_hits") == 15
        assert metrics.get_counter("hmm_requests") == 20
        assert metrics.compute_hit_rate() == pytest.approx(0.75, abs=0.01)

        snap = metrics.snapshot()
        assert snap["hmm"]["confidence_p50"] > 0
        assert snap["search"]["latency_p50_ms"] > 0
        assert snap["uptime_seconds"] >= 0

        # Phase B: Simulate degradation — 2 breaches on search_error_rate
        r1 = rollback.evaluate("search_error_rate", breached=True)
        assert r1 is None  # First breach — no rollback yet

        r2 = rollback.evaluate("search_error_rate", breached=True)
        assert r2 is not None  # Second breach — rollback triggered

        # Phase C: Verify rollback state
        assert r2.component == "search"
        assert r2.action == "percent_zero"
        assert os.environ.get("BIZRA_PHASE46_SEARCH_PERCENT") == "0"

        # Other components still active
        assert os.environ.get("BIZRA_PHASE46_GOT_BRIDGE_PERCENT") == "100"
        assert os.environ.get("BIZRA_PHASE46_HMM_PERCENT") == "100"

        # Receipt was persisted
        receipts = os.listdir(receipt_dir)
        assert len(receipts) == 1

        # Phase D: Canary now blocks search (percent=0)
        # Reload canary to pick up new env state
        canary_post = CanaryRouter(salt="test-pipeline-salt")
        assert not canary_post.should_route("search", "new-query")

        # HMM and GoT still route
        assert canary_post.should_route("got_bridge", "new-query", percent=100)
        assert canary_post.should_route("hmm", "new-query", percent=100)

    def test_hard_kill_when_all_zeroed(
        self, monkeypatch, metrics, receipt_dir
    ):
        """When all components are at 0%, rollback escalates to hard kill."""
        # All percents at 0 — nothing to roll back individually
        monkeypatch.setenv("BIZRA_PHASE46_SEARCH_PERCENT", "0")
        monkeypatch.setenv("BIZRA_PHASE46_GOT_BRIDGE_PERCENT", "0")
        monkeypatch.setenv("BIZRA_PHASE46_HMM_PERCENT", "0")

        rollback = RollbackEngine(receipt_dir=receipt_dir, metrics=metrics)
        rollback.evaluate("resonance_snr", breached=True)
        receipt = rollback.evaluate("resonance_snr", breached=True)

        assert receipt is not None
        assert receipt.component == "all"
        assert receipt.action == "hard_kill"

        # All ENABLED flags forced to "0"
        for key in (
            "BIZRA_PHASE46_SEARCH_ENABLED",
            "BIZRA_PHASE46_GOT_BRIDGE_ENABLED",
            "BIZRA_PHASE46_HMM_ENABLED",
        ):
            assert os.environ.get(key) == "0"

    def test_canary_salt_isolation(self, metrics):
        """Different salts produce different routing decisions at 50%."""
        # No ENABLED flag set — percent routing is the active gate.
        router_a = CanaryRouter(salt="salt-alpha")
        router_b = CanaryRouter(salt="salt-beta")

        results_a = [
            router_a.should_route("search", f"q-{i}", percent=50)
            for i in range(100)
        ]
        results_b = [
            router_b.should_route("search", f"q-{i}", percent=50)
            for i in range(100)
        ]

        # Different salts should produce different routing (not identical)
        assert results_a != results_b

    def test_metrics_snapshot_structure(self, metrics):
        """Verify complete snapshot structure for Prometheus exposition."""
        metrics.inc("search_requests", 10)
        metrics.inc("search_hits", 7)
        metrics.inc("resonance_requests", 5)
        metrics.inc("hmm_requests", 3)
        metrics.inc("gateway_requests", 20)
        metrics.record_latency("search", 25.0)
        metrics.record_snr(0.92)
        metrics.record_hmm_confidence(0.88)
        metrics.record_hmm_observation("search")
        metrics.record_hmm_observation("edit")

        snap = metrics.snapshot()

        # Counters
        assert snap["counters"]["search_requests"] == 10
        assert snap["counters"]["search_hits"] == 7
        assert snap["counters"]["gateway_requests"] == 20

        # Search latency
        assert "latency_p50_ms" in snap["search"]
        assert "latency_p95_ms" in snap["search"]
        assert "hit_rate" in snap["search"]
        assert snap["search"]["hit_rate"] == pytest.approx(0.7, abs=0.01)

        # Resonance
        assert "combined_snr_p50" in snap["resonance"]
        assert snap["resonance"]["combined_snr_p50"] == pytest.approx(0.92, abs=0.01)

        # HMM
        assert "confidence_p50" in snap["hmm"]
        assert snap["hmm"]["confidence_p50"] == pytest.approx(0.88, abs=0.01)
        assert snap["hmm"]["observation_entropy"] > 0  # 2 symbols → entropy > 0

        # GoT bridge
        assert "convergence_pass_rate" in snap["got_bridge"]
        assert "fallback_rate" in snap["got_bridge"]

        # Uptime
        assert snap["uptime_seconds"] >= 0

    def test_rollback_engine_status(self, metrics, receipt_dir):
        """Rollback engine exposes current breach window state."""
        engine = RollbackEngine(receipt_dir=receipt_dir, metrics=metrics)

        status = engine.status
        assert "rollback_in_progress" in status
        assert status["rollback_in_progress"] is False
        assert "breach_windows" in status
        assert "search_error_rate" in status["breach_windows"]
        assert "got_fallback_rate" in status["breach_windows"]
        assert "hmm_confidence" in status["breach_windows"]
        assert "resonance_snr" in status["breach_windows"]
        assert "latency_regression" in status["breach_windows"]

    def test_canary_get_active_percents(self, monkeypatch, canary):
        """Active percents reflect current env state."""
        monkeypatch.setenv("BIZRA_PHASE46_SEARCH_PERCENT", "75")
        monkeypatch.setenv("BIZRA_PHASE46_GOT_BRIDGE_PERCENT", "30")
        monkeypatch.setenv("BIZRA_PHASE46_HMM_PERCENT", "10")

        percents = canary.get_active_percents()
        assert percents["search"] == 75
        assert percents["got_bridge"] == 30
        assert percents["hmm"] == 10


# ======================================================================
# Stage 5: Canary Stage 1 — SEARCH_PERCENT=10 activation verification
# ======================================================================


class TestCanaryStage1Activation:
    """Verify the Stage 1 production configuration: SEARCH_ENABLED=1,
    SEARCH_PERCENT=10, GoT/HMM disabled. Validates ~10% routing,
    clean rollback windows, and correct metrics flow."""

    @pytest.fixture(autouse=True)
    def stage1_env(self, monkeypatch):
        """Set the exact Stage 1 env vars from .env.example.

        SEARCH_ENABLED is deliberately UNSET — percent routing controls search.
        GoT/HMM kill switches are OFF (hard-disabled).
        """
        monkeypatch.delenv("BIZRA_PHASE46_SEARCH_ENABLED", raising=False)
        monkeypatch.setenv("BIZRA_PHASE46_SEARCH_PERCENT", "10")
        monkeypatch.setenv("BIZRA_PHASE46_GOT_BRIDGE_ENABLED", "0")
        monkeypatch.setenv("BIZRA_PHASE46_GOT_BRIDGE_PERCENT", "0")
        monkeypatch.setenv("BIZRA_PHASE46_HMM_ENABLED", "0")
        monkeypatch.setenv("BIZRA_PHASE46_HMM_PERCENT", "0")

    def test_search_routes_approximately_10_percent(self):
        """With SEARCH_PERCENT=10, roughly 10% of 1000 keys should route."""
        router = CanaryRouter(salt="stage1-verify")
        routed = sum(
            router.should_route("search", f"query-{i}")
            for i in range(1000)
        )
        # Expect ~100 routed (10%). Allow 5-15% range for hash variance.
        assert 50 <= routed <= 150, f"Expected ~100 routed, got {routed}"

    def test_search_routing_is_deterministic(self):
        """Same keys produce identical routing decisions across runs."""
        router = CanaryRouter(salt="stage1-verify")
        run1 = [router.should_route("search", f"q-{i}") for i in range(200)]
        run2 = [router.should_route("search", f"q-{i}") for i in range(200)]
        assert run1 == run2

    def test_got_bridge_fully_blocked(self):
        """GoT bridge kill switch (ENABLED=0) blocks all routing."""
        router = CanaryRouter()
        routed = sum(
            router.should_route("got_bridge", f"q-{i}")
            for i in range(100)
        )
        assert routed == 0

    def test_hmm_fully_blocked(self):
        """HMM kill switch (ENABLED=0) blocks all routing."""
        router = CanaryRouter()
        routed = sum(
            router.should_route("hmm", f"q-{i}")
            for i in range(100)
        )
        assert routed == 0

    def test_active_percents_reflect_stage1(self):
        """get_active_percents() returns the Stage 1 configuration."""
        router = CanaryRouter()
        percents = router.get_active_percents()
        assert percents["search"] == 10
        assert percents["got_bridge"] == 0
        assert percents["hmm"] == 0

    def test_metrics_accumulate_for_routed_requests(self):
        """Only routed requests increment search_requests counter."""
        router = CanaryRouter(salt="stage1-metrics")
        metrics = Phase46Metrics()

        for i in range(100):
            if router.should_route("search", f"query-{i}"):
                metrics.inc("search_requests")
                metrics.record_latency("search", 20.0 + i * 0.1)
                metrics.inc("search_hits")

        # Should have ~10 requests (10%)
        count = metrics.get_counter("search_requests")
        assert 3 <= count <= 20, f"Expected ~10, got {count}"
        assert metrics.get_counter("search_hits") == count

        snap = metrics.snapshot()
        if count > 0:
            assert snap["search"]["latency_p50_ms"] > 0
            assert snap["search"]["hit_rate"] == pytest.approx(1.0, abs=0.01)

    def test_rollback_stays_clean_with_no_errors(self):
        """Under normal operation (no errors), rollback windows stay clean."""
        metrics = Phase46Metrics()
        receipt_dir = tempfile.mkdtemp(prefix="stage1_rollback_")

        try:
            rollback = RollbackEngine(receipt_dir=receipt_dir, metrics=metrics)

            # Simulate 50 clean requests
            for _ in range(50):
                metrics.inc("search_requests")
                metrics.record_latency("search", 18.0)
                metrics.inc("search_hits")

            # Evaluate with no breaches
            rollback.evaluate("search_error_rate", breached=False)
            rollback.evaluate("search_error_rate", breached=False)

            status = rollback.status
            for window_name, window in status["breach_windows"].items():
                assert window["consecutive"] == 0, (
                    f"{window_name} has consecutive={window['consecutive']}"
                )
            assert status["rollback_in_progress"] is False

            # No receipts written
            assert len(os.listdir(receipt_dir)) == 0
        finally:
            shutil.rmtree(receipt_dir, ignore_errors=True)

    def test_stage1_rollback_preserves_got_hmm(self, monkeypatch):
        """If search rolls back, GoT/HMM kill switches remain at 0 (disabled)."""
        receipt_dir = tempfile.mkdtemp(prefix="stage1_preserve_")
        metrics = Phase46Metrics()

        try:
            rollback = RollbackEngine(receipt_dir=receipt_dir, metrics=metrics)

            # Trigger search rollback
            rollback.evaluate("search_error_rate", breached=True)
            receipt = rollback.evaluate("search_error_rate", breached=True)

            assert receipt is not None
            assert receipt.component == "search"
            assert os.environ.get("BIZRA_PHASE46_SEARCH_PERCENT") == "0"

            # GoT and HMM kill switches still blocking
            assert os.environ.get("BIZRA_PHASE46_GOT_BRIDGE_ENABLED") == "0"
            assert os.environ.get("BIZRA_PHASE46_HMM_ENABLED") == "0"
        finally:
            shutil.rmtree(receipt_dir, ignore_errors=True)
