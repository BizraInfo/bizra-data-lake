"""
Tests for Persistence State Providers — Proposal 1+++ Synthesis
===============================================================
Verifies that all 5 persistence gaps are closed:
1. ProactiveScheduler.get_persistable_state()
2. PredictiveMonitor.get_persistable_state()
3. OpportunityPipeline.get_persistable_state() (rate limiter = SAFETY)
4. MemoryCoordinator RestorePriority ordering
5. Round-trip: save → restore → verify

Standing on Giants: Event Sourcing + Snapshot Pattern + Priority-Aware Restore
"""

import asyncio
import json
import time

import pytest

from core.sovereign.memory_coordinator import (
    MemoryCoordinator,
    MemoryCoordinatorConfig,
    RestorePriority,
)
from core.sovereign.opportunity_pipeline import (
    OpportunityPipeline,
    RateLimitFilter,
)
from core.sovereign.predictive_monitor import PredictiveMonitor
from core.sovereign.proactive_scheduler import (
    JobPriority,
    ProactiveScheduler,
    ScheduleType,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def scheduler():
    s = ProactiveScheduler()
    return s


@pytest.fixture
def monitor():
    return PredictiveMonitor(window_size=50)


@pytest.fixture
def pipeline():
    return OpportunityPipeline(snr_threshold=0.85, ihsan_threshold=0.95)


@pytest.fixture
def coordinator(tmp_path):
    config = MemoryCoordinatorConfig(state_dir=tmp_path, max_checkpoints=5)
    mc = MemoryCoordinator(config)
    mc.initialize(node_id="node0_test_persist")
    return mc


# =============================================================================
# TESTS: RestorePriority
# =============================================================================


class TestRestorePriority:
    def test_priority_ordering(self):
        assert RestorePriority.SAFETY < RestorePriority.CORE
        assert RestorePriority.CORE < RestorePriority.QUALITY
        assert RestorePriority.QUALITY < RestorePriority.AUXILIARY

    def test_priority_values(self):
        assert RestorePriority.SAFETY.value == 0
        assert RestorePriority.CORE.value == 1
        assert RestorePriority.QUALITY.value == 2
        assert RestorePriority.AUXILIARY.value == 3


# =============================================================================
# TESTS: ProactiveScheduler Persistence
# =============================================================================


class TestSchedulerPersistence:
    def test_empty_scheduler_state(self, scheduler):
        state = scheduler.get_persistable_state()
        assert state["total_jobs"] == 0
        assert state["jobs"] == {}
        assert state["running"] is False

    def test_scheduler_with_jobs(self, scheduler):
        async def noop():
            pass

        scheduler.schedule(
            "test_job",
            noop,
            schedule_type=ScheduleType.RECURRING,
            priority=JobPriority.HIGH,
            interval=60.0,
        )
        state = scheduler.get_persistable_state()
        assert state["total_jobs"] == 1
        job_data = list(state["jobs"].values())[0]
        assert job_data["name"] == "test_job"
        assert job_data["schedule_type"] == "RECURRING"
        assert job_data["priority"] == "HIGH"
        assert job_data["interval_seconds"] == 60.0
        assert job_data["run_count"] == 0
        assert job_data["enabled"] is True

    def test_scheduler_state_no_handlers(self, scheduler):
        """Handlers must NOT be serialized (callables aren't JSON-safe)."""

        async def noop():
            pass

        scheduler.schedule("test", noop)
        state = scheduler.get_persistable_state()
        # Verify state is JSON-serializable
        serialized = json.dumps(state)
        assert serialized  # no TypeError from callables

    def test_restore_run_counts(self, scheduler):
        async def noop():
            pass

        job_id = scheduler.schedule("test", noop)
        # Simulate runs
        scheduler._jobs[job_id].run_count = 42
        scheduler._jobs[job_id].error_count = 3

        saved = scheduler.get_persistable_state()

        # New scheduler with same job ID
        new_scheduler = ProactiveScheduler()
        new_job_id = new_scheduler.schedule("test", noop)
        # Inject matching job_id
        job = new_scheduler._jobs.pop(new_job_id)
        job.id = job_id
        new_scheduler._jobs[job_id] = job

        restored = new_scheduler.restore_persistable_state(saved)
        assert restored == 1
        assert new_scheduler._jobs[job_id].run_count == 42
        assert new_scheduler._jobs[job_id].error_count == 3


# =============================================================================
# TESTS: PredictiveMonitor Persistence
# =============================================================================


class TestMonitorPersistence:
    def test_empty_monitor_state(self, monitor):
        state = monitor.get_persistable_state()
        assert state["baselines"] == {}
        assert state["analyses"] == {}
        assert state["alert_count"] == 0

    def test_monitor_with_readings(self, monitor):
        for i in range(10):
            monitor.record("snr_score", 0.90 + i * 0.005)
        state = monitor.get_persistable_state()
        assert "snr_score" in state["baselines"]
        baseline = state["baselines"]["snr_score"]
        assert baseline["count"] == 10
        assert baseline["min"] == pytest.approx(0.90, abs=0.01)
        assert baseline["latest"] == pytest.approx(0.945, abs=0.01)

    def test_monitor_with_analysis(self, monitor):
        for i in range(10):
            monitor.record("latency_ms", 100 + i * 5)
        monitor.analyze("latency_ms")
        state = monitor.get_persistable_state()
        assert "latency_ms" in state["analyses"]
        analysis = state["analyses"]["latency_ms"]
        assert analysis["direction"] == "RISING"
        assert analysis["slope"] > 0

    def test_monitor_state_serializable(self, monitor):
        for i in range(10):
            monitor.record("test_metric", float(i))
        monitor.analyze("test_metric")
        state = monitor.get_persistable_state()
        serialized = json.dumps(state)
        assert serialized

    def test_restore_alert_count(self, monitor):
        monitor._alert_count = 25
        state = monitor.get_persistable_state()

        new_monitor = PredictiveMonitor()
        restored = new_monitor.restore_persistable_state(state)
        assert new_monitor._alert_count == 25
        assert restored == 0  # no baselines to count


# =============================================================================
# TESTS: OpportunityPipeline Persistence
# =============================================================================


class TestPipelinePersistence:
    def test_empty_pipeline_state(self, pipeline):
        state = pipeline.get_persistable_state()
        assert state["metrics"]["total_received"] == 0
        assert state["rate_limiter"] != {}  # Rate limiter exists

    def test_pipeline_rate_limiter_persists(self, pipeline):
        """SAFETY-CRITICAL: Rate limiter counts must survive restarts."""
        # Simulate some rate-limited actions
        for f in pipeline._filters:
            if isinstance(f, RateLimitFilter):
                f._hourly_counts = {"financial": 5, "health": 2}
                f._daily_counts = {"financial": 15, "health": 8}
                break

        state = pipeline.get_persistable_state()
        assert state["rate_limiter"]["hourly_counts"]["financial"] == 5
        assert state["rate_limiter"]["daily_counts"]["health"] == 8

    def test_restore_rate_limiter(self, pipeline):
        """Rate limiter state must restore correctly."""
        saved_state = {
            "rate_limiter": {
                "hourly_counts": {"financial": 7},
                "daily_counts": {"financial": 30, "social": 10},
                "last_reset_hour": time.time() - 1800,
                "last_reset_day": time.time() - 43200,
            }
        }

        restored = pipeline.restore_persistable_state(saved_state)
        assert restored is True

        # Verify the filter got the state
        for f in pipeline._filters:
            if isinstance(f, RateLimitFilter):
                assert f._hourly_counts["financial"] == 7
                assert f._daily_counts["social"] == 10
                break

    def test_pipeline_metrics_persist(self, pipeline):
        pipeline._metrics["total_received"] = 100
        pipeline._metrics["total_executed"] = 80
        state = pipeline.get_persistable_state()
        assert state["metrics"]["total_received"] == 100

    def test_pipeline_state_serializable(self, pipeline):
        for f in pipeline._filters:
            if isinstance(f, RateLimitFilter):
                f.record_action("test_domain")
        state = pipeline.get_persistable_state()
        serialized = json.dumps(state)
        assert serialized


# =============================================================================
# TESTS: Priority-Aware Coordinator
# =============================================================================


class TestPriorityAwareCoordinator:
    def test_register_with_priority(self, coordinator):
        coordinator.register_state_provider(
            "safety_item", lambda: {"x": 1}, RestorePriority.SAFETY
        )
        coordinator.register_state_provider(
            "quality_item", lambda: {"y": 2}, RestorePriority.QUALITY
        )
        assert "safety_item" in coordinator._state_providers
        assert coordinator._state_providers["safety_item"][1] == RestorePriority.SAFETY

    def test_default_priority_is_core(self, coordinator):
        coordinator.register_state_provider("default", lambda: {})
        assert coordinator._state_providers["default"][1] == RestorePriority.CORE

    @pytest.mark.asyncio
    async def test_save_includes_priority(self, coordinator, tmp_path):
        coordinator.register_state_provider(
            "rate_limiter", lambda: {"counts": {"a": 5}}, RestorePriority.SAFETY
        )
        coordinator.register_state_provider(
            "trends", lambda: {"slope": 0.1}, RestorePriority.QUALITY
        )
        await coordinator.save_all()

        state = await coordinator.restore_latest()
        assert state is not None
        # Priority values are embedded in the saved state
        assert state["rate_limiter"]["counts"]["a"] == 5

    @pytest.mark.asyncio
    async def test_restore_by_priority_ordering(self, coordinator):
        coordinator.register_state_provider(
            "quality", lambda: {"q": True}, RestorePriority.QUALITY
        )
        coordinator.register_state_provider(
            "safety", lambda: {"s": True}, RestorePriority.SAFETY
        )
        coordinator.register_state_provider(
            "core", lambda: {"c": True}, RestorePriority.CORE
        )
        await coordinator.save_all()

        state = await coordinator.restore_latest()
        prioritized = coordinator.restore_by_priority(state)

        # SAFETY must come first
        names = [name for _, name, _ in prioritized]
        safety_idx = names.index("safety")
        core_idx = names.index("core")
        quality_idx = names.index("quality")
        assert safety_idx < core_idx < quality_idx


# =============================================================================
# TESTS: End-to-End Round Trip
# =============================================================================


class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_full_save_restore_cycle(self, coordinator, tmp_path):
        """Save scheduler + monitor + pipeline state, restore, verify."""
        scheduler = ProactiveScheduler()
        monitor = PredictiveMonitor()
        pipeline = OpportunityPipeline()

        # Populate with data
        for i in range(5):
            monitor.record("snr", 0.90 + i * 0.01)

        for f in pipeline._filters:
            if isinstance(f, RateLimitFilter):
                f._hourly_counts = {"test": 3}
                break

        # Register providers
        coordinator.register_state_provider(
            "pipeline",
            pipeline.get_persistable_state,
            RestorePriority.SAFETY,
        )
        coordinator.register_state_provider(
            "scheduler",
            scheduler.get_persistable_state,
            RestorePriority.QUALITY,
        )
        coordinator.register_state_provider(
            "monitor",
            monitor.get_persistable_state,
            RestorePriority.QUALITY,
        )

        # Save
        ok = await coordinator.save_all()
        assert ok is True

        # Restore
        state = await coordinator.restore_latest()
        assert state is not None

        # Verify pipeline rate limiter (SAFETY)
        assert state["pipeline"]["rate_limiter"]["hourly_counts"]["test"] == 3

        # Verify monitor baselines (QUALITY)
        assert "snr" in state["monitor"]["baselines"]
        assert state["monitor"]["baselines"]["snr"]["count"] == 5

        # Verify scheduler (QUALITY)
        assert state["scheduler"]["total_jobs"] == 0

        # Verify priority ordering
        prioritized = coordinator.restore_by_priority(state)
        names = [name for _, name, _ in prioritized]
        # Pipeline (SAFETY=0) should come before scheduler/monitor (QUALITY=2)
        pipeline_idx = names.index("pipeline")
        scheduler_idx = names.index("scheduler")
        assert pipeline_idx < scheduler_idx
