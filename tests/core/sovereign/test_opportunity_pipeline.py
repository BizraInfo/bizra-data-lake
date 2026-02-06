"""
Tests for Opportunity Pipeline — BIZRA Proactive Sovereign Entity
==================================================================
Validates the nervous system connecting Muraqabah to execution.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from core.sovereign.opportunity_pipeline import (
    ConstitutionalFilter,
    DaughterTestFilter,
    FilterResult,
    IhsanFilter,
    OpportunityPipeline,
    OpportunityStatus,
    PipelineOpportunity,
    PipelineStage,
    RateLimitFilter,
    SNRFilter,
    connect_background_agents_to_pipeline,
    connect_muraqabah_to_pipeline,
    create_opportunity_pipeline,
)
from core.sovereign.autonomy_matrix import AutonomyLevel
from core.sovereign.event_bus import Event, EventBus


# =============================================================================
# DATA STRUCTURE TESTS
# =============================================================================

class TestPipelineOpportunity:
    """Tests for PipelineOpportunity dataclass."""

    def test_opportunity_creation(self):
        """Test creating a pipeline opportunity."""
        opp = PipelineOpportunity(
            id="opp-123",
            domain="financial",
            description="Test opportunity",
            source="test_sensor",
            detected_at=time.time(),
            snr_score=0.9,
            ihsan_score=0.95,
        )
        assert opp.id == "opp-123"
        assert opp.domain == "financial"
        assert opp.snr_score == 0.9
        assert opp.stage == PipelineStage.DETECTION
        assert opp.status == OpportunityStatus.PENDING

    def test_advance_stage(self):
        """Test stage advancement tracking."""
        opp = PipelineOpportunity(
            id="opp-456",
            domain="health",
            description="Health check",
            source="health_sensor",
            detected_at=time.time(),
        )

        opp.advance_stage(PipelineStage.ENRICHMENT, "Starting enrichment")
        assert opp.stage == PipelineStage.ENRICHMENT
        assert len(opp.stage_history) == 1
        assert opp.stage_history[0][0] == PipelineStage.DETECTION

    def test_to_dict(self):
        """Test serialization."""
        opp = PipelineOpportunity(
            id="opp-789",
            domain="cognitive",
            description="Learning opportunity",
            source="cognitive_sensor",
            detected_at=time.time(),
            snr_score=0.85,
            autonomy_level=AutonomyLevel.AUTOLOW,
        )

        data = opp.to_dict()
        assert data["id"] == "opp-789"
        assert data["domain"] == "cognitive"
        assert data["snr_score"] == 0.85
        assert data["autonomy_level"] == "AUTOLOW"


# =============================================================================
# FILTER TESTS
# =============================================================================

class TestSNRFilter:
    """Tests for SNR constitutional filter."""

    @pytest.mark.asyncio
    async def test_snr_filter_pass(self):
        """Test SNR filter passes high-quality opportunities."""
        f = SNRFilter(min_snr=0.85)
        opp = PipelineOpportunity(
            id="test",
            domain="test",
            description="Test",
            source="test",
            detected_at=time.time(),
            snr_score=0.9,
        )

        result = await f.check(opp)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_snr_filter_reject(self):
        """Test SNR filter rejects low-quality opportunities."""
        f = SNRFilter(min_snr=0.85)
        opp = PipelineOpportunity(
            id="test",
            domain="test",
            description="Test",
            source="test",
            detected_at=time.time(),
            snr_score=0.7,
        )

        result = await f.check(opp)
        assert result.passed is False
        assert "0.7" in result.reason


class TestIhsanFilter:
    """Tests for Ihsan constitutional filter."""

    @pytest.mark.asyncio
    async def test_ihsan_filter_observer_level(self):
        """Test Ihsan filter at OBSERVER level (no threshold)."""
        f = IhsanFilter()
        opp = PipelineOpportunity(
            id="test",
            domain="test",
            description="Test",
            source="test",
            detected_at=time.time(),
            ihsan_score=0.5,
            autonomy_level=AutonomyLevel.OBSERVER,
        )

        result = await f.check(opp)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_ihsan_filter_autolow_level(self):
        """Test Ihsan filter at AUTOLOW level (0.97 threshold)."""
        f = IhsanFilter()

        # Should pass
        opp_pass = PipelineOpportunity(
            id="test1",
            domain="test",
            description="Test",
            source="test",
            detected_at=time.time(),
            ihsan_score=0.98,
            autonomy_level=AutonomyLevel.AUTOLOW,
        )
        result = await f.check(opp_pass)
        assert result.passed is True

        # Should fail
        opp_fail = PipelineOpportunity(
            id="test2",
            domain="test",
            description="Test",
            source="test",
            detected_at=time.time(),
            ihsan_score=0.96,
            autonomy_level=AutonomyLevel.AUTOLOW,
        )
        result = await f.check(opp_fail)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_ihsan_filter_sovereign_level(self):
        """Test Ihsan filter at SOVEREIGN level (1.0 threshold)."""
        f = IhsanFilter()

        # Should fail (perfect score required)
        opp = PipelineOpportunity(
            id="test",
            domain="test",
            description="Test",
            source="test",
            detected_at=time.time(),
            ihsan_score=0.99,
            autonomy_level=AutonomyLevel.SOVEREIGN,
        )
        result = await f.check(opp)
        assert result.passed is False


class TestDaughterTestFilter:
    """Tests for Daughter Test constitutional filter."""

    @pytest.mark.asyncio
    async def test_daughter_test_sensitive_domain(self):
        """Test Daughter Test blocks sensitive domains at high autonomy."""
        f = DaughterTestFilter()

        # Health domain at AUTOMEDIUM should fail
        opp = PipelineOpportunity(
            id="test",
            domain="health",
            description="Schedule surgery",
            source="test",
            detected_at=time.time(),
            autonomy_level=AutonomyLevel.AUTOMEDIUM,
        )
        result = await f.check(opp)
        assert result.passed is False
        assert "Daughter Test" in result.reason

    @pytest.mark.asyncio
    async def test_daughter_test_sensitive_keywords(self):
        """Test Daughter Test blocks sensitive keywords."""
        f = DaughterTestFilter()

        # Delete keyword at AUTOMEDIUM should fail
        opp = PipelineOpportunity(
            id="test",
            domain="files",
            description="Delete all old backups",
            source="test",
            detected_at=time.time(),
            autonomy_level=AutonomyLevel.AUTOMEDIUM,
        )
        result = await f.check(opp)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_daughter_test_safe_action(self):
        """Test Daughter Test passes safe actions."""
        f = DaughterTestFilter()

        opp = PipelineOpportunity(
            id="test",
            domain="cognitive",
            description="Suggest learning resource",
            source="test",
            detected_at=time.time(),
            autonomy_level=AutonomyLevel.AUTOMEDIUM,
        )
        result = await f.check(opp)
        assert result.passed is True


class TestRateLimitFilter:
    """Tests for rate limit filter."""

    @pytest.mark.asyncio
    async def test_rate_limit_within_limits(self):
        """Test rate limit allows actions within limits."""
        f = RateLimitFilter(max_per_hour=10, max_per_day=50)

        opp = PipelineOpportunity(
            id="test",
            domain="files",
            description="Test",
            source="test",
            detected_at=time.time(),
        )

        result = await f.check(opp)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        """Test rate limit blocks when exceeded."""
        f = RateLimitFilter(max_per_hour=2, max_per_day=5)

        # Record some actions
        f.record_action("files")
        f.record_action("files")

        opp = PipelineOpportunity(
            id="test",
            domain="files",
            description="Test",
            source="test",
            detected_at=time.time(),
        )

        result = await f.check(opp)
        assert result.passed is False
        assert "Hourly limit" in result.reason


# =============================================================================
# PIPELINE TESTS
# =============================================================================

class TestOpportunityPipeline:
    """Tests for OpportunityPipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = OpportunityPipeline(snr_threshold=0.85, ihsan_threshold=0.95)

        assert pipeline.snr_threshold == 0.85
        assert pipeline.ihsan_threshold == 0.95
        assert not pipeline._running
        assert len(pipeline._filters) == 4  # SNR, Ihsan, DaughterTest, RateLimit

    @pytest.mark.asyncio
    async def test_pipeline_start_stop(self):
        """Test pipeline lifecycle."""
        pipeline = OpportunityPipeline()

        await pipeline.start()
        assert pipeline._running is True

        await pipeline.stop()
        assert pipeline._running is False

    @pytest.mark.asyncio
    async def test_pipeline_submit_opportunity(self):
        """Test submitting an opportunity."""
        pipeline = OpportunityPipeline()
        await pipeline.start()

        try:
            opp_id = await pipeline.submit_from_muraqabah(
                domain="cognitive",
                description="Learning opportunity",
                source="test_sensor",
                snr_score=0.9,
                urgency=0.7,
            )

            assert opp_id.startswith("opp-")

            stats = pipeline.stats()
            assert stats["total_received"] == 1
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_filter_low_snr(self):
        """Test that pipeline filters low SNR opportunities."""
        pipeline = OpportunityPipeline(snr_threshold=0.85)
        await pipeline.start()

        try:
            # Submit low SNR opportunity
            opp = PipelineOpportunity(
                id="low-snr",
                domain="test",
                description="Low quality",
                source="test",
                detected_at=time.time(),
                snr_score=0.5,  # Below threshold
            )
            await pipeline.submit(opp)

            # Wait for processing
            await asyncio.sleep(0.5)

            stats = pipeline.stats()
            assert stats["total_filtered"] >= 1
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_approval_queue(self):
        """Test that SUGGESTER level opportunities go to approval queue."""
        pipeline = OpportunityPipeline()
        await pipeline.start()

        try:
            opp = PipelineOpportunity(
                id="need-approval",
                domain="financial",
                description="Investment suggestion",
                source="test",
                detected_at=time.time(),
                snr_score=0.95,
                ihsan_score=0.98,
                autonomy_level=AutonomyLevel.SUGGESTER,
            )
            await pipeline.submit(opp)

            # Wait for processing
            await asyncio.sleep(0.5)

            pending = pipeline.get_pending_approvals()
            assert len(pending) >= 1
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_approve_opportunity(self):
        """Test approving a pending opportunity."""
        pipeline = OpportunityPipeline()
        await pipeline.start()

        try:
            opp = PipelineOpportunity(
                id="approve-me",
                domain="test",
                description="Test approval",
                source="test",
                detected_at=time.time(),
                snr_score=0.95,
                ihsan_score=0.98,
                autonomy_level=AutonomyLevel.SUGGESTER,
            )
            await pipeline.submit(opp)

            # Wait for it to hit approval queue
            await asyncio.sleep(0.5)

            # Approve it
            success = await pipeline.approve("approve-me")
            assert success is True

            stats = pipeline.stats()
            assert stats["total_approved"] >= 1
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_reject_opportunity(self):
        """Test rejecting a pending opportunity."""
        pipeline = OpportunityPipeline()
        await pipeline.start()

        try:
            opp = PipelineOpportunity(
                id="reject-me",
                domain="test",
                description="Test rejection",
                source="test",
                detected_at=time.time(),
                snr_score=0.95,
                ihsan_score=0.98,
                autonomy_level=AutonomyLevel.SUGGESTER,
            )
            await pipeline.submit(opp)

            # Wait for it to hit approval queue
            await asyncio.sleep(0.5)

            # Reject it
            success = await pipeline.reject("reject-me", "User declined")
            assert success is True

            stats = pipeline.stats()
            assert stats["total_rejected"] >= 1
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_auto_execute_autolow(self):
        """Test that AUTOLOW opportunities are auto-executed."""
        pipeline = OpportunityPipeline()

        # Track execution
        executed = []
        async def track_execution(opp):
            executed.append(opp.id)
            return {"success": True}

        pipeline.set_execution_callback(track_execution)
        await pipeline.start()

        try:
            opp = PipelineOpportunity(
                id="auto-exec",
                domain="cognitive",
                description="Auto execute test",
                source="test",
                detected_at=time.time(),
                snr_score=0.95,
                ihsan_score=0.98,
                autonomy_level=AutonomyLevel.AUTOLOW,
            )
            await pipeline.submit(opp)

            # Wait for processing
            await asyncio.sleep(0.5)

            assert "auto-exec" in executed
            stats = pipeline.stats()
            assert stats["total_executed"] >= 1
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_callbacks(self):
        """Test callback registration and invocation."""
        pipeline = OpportunityPipeline()

        enrichment_called = False
        planning_called = False

        async def enrichment_cb(opp):
            nonlocal enrichment_called
            enrichment_called = True
            return {"context": {"enriched": True}}

        async def planning_cb(opp):
            nonlocal planning_called
            planning_called = True
            return {"plan": {"steps": []}, "agent": "test_agent"}

        pipeline.set_enrichment_callback(enrichment_cb)
        pipeline.set_planning_callback(planning_cb)
        await pipeline.start()

        try:
            opp = PipelineOpportunity(
                id="callback-test",
                domain="test",
                description="Test callbacks",
                source="test",
                detected_at=time.time(),
                snr_score=0.95,
                ihsan_score=0.98,
                autonomy_level=AutonomyLevel.AUTOLOW,
            )
            await pipeline.submit(opp)

            # Wait for processing
            await asyncio.sleep(0.5)

            assert enrichment_called
            assert planning_called
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_with_event_bus(self):
        """Test pipeline publishes events to EventBus."""
        event_bus = EventBus()
        pipeline = OpportunityPipeline(event_bus=event_bus)

        received_events = []

        async def event_handler(event):
            received_events.append(event.topic)

        event_bus.subscribe("pipeline.started", event_handler)
        event_bus.subscribe("pipeline.opportunity.received", event_handler)

        # Start event bus processing loop in background
        bus_task = asyncio.create_task(event_bus.start())

        await pipeline.start()

        try:
            opp = PipelineOpportunity(
                id="event-test",
                domain="test",
                description="Test events",
                source="test",
                detected_at=time.time(),
                snr_score=0.95,
            )
            await pipeline.submit(opp)

            # Wait for events to propagate
            await asyncio.sleep(0.5)

            assert "pipeline.started" in received_events
            assert "pipeline.opportunity.received" in received_events
        finally:
            await pipeline.stop()
            event_bus.stop()
            bus_task.cancel()
            try:
                await bus_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_pipeline_stats(self):
        """Test pipeline statistics."""
        pipeline = OpportunityPipeline()

        stats = pipeline.stats()

        assert "total_received" in stats
        assert "total_filtered" in stats
        assert "total_approved" in stats
        assert "total_rejected" in stats
        assert "total_executed" in stats
        assert "queue_size" in stats
        assert "active_count" in stats
        assert "by_domain" in stats
        assert "by_autonomy" in stats


# =============================================================================
# FACTORY AND INTEGRATION TESTS
# =============================================================================

class TestPipelineFactory:
    """Tests for pipeline factory function."""

    @pytest.mark.asyncio
    async def test_create_opportunity_pipeline(self):
        """Test factory function."""
        pipeline = await create_opportunity_pipeline(
            snr_threshold=0.9,
            ihsan_threshold=0.96,
            auto_start=True,
        )

        try:
            assert isinstance(pipeline, OpportunityPipeline)
            assert pipeline._running is True
            assert pipeline.snr_threshold == 0.9
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_create_pipeline_no_auto_start(self):
        """Test factory function without auto-start."""
        pipeline = await create_opportunity_pipeline(auto_start=False)

        assert pipeline._running is False

        await pipeline.start()
        assert pipeline._running is True
        await pipeline.stop()


class TestPipelineConnectors:
    """Tests for pipeline connector functions."""

    def test_connect_muraqabah_placeholder(self):
        """Test muraqabah connector function exists."""
        # Just verify the function is callable
        assert callable(connect_muraqabah_to_pipeline)

    def test_connect_background_agents_placeholder(self):
        """Test background agents connector function exists."""
        assert callable(connect_background_agents_to_pipeline)


# =============================================================================
# END-TO-END TESTS
# =============================================================================

class TestPipelineE2E:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self):
        """Test complete opportunity flow through pipeline."""
        event_bus = EventBus()
        pipeline = OpportunityPipeline(event_bus=event_bus)

        # Track all stages
        stages_seen = []

        async def track_events(event):
            stages_seen.append(event.topic)

        event_bus.subscribe("pipeline.opportunity.completed", track_events)

        async def execute(opp):
            return {"success": True, "message": "Executed"}

        pipeline.set_execution_callback(execute)
        await pipeline.start()

        try:
            # Submit high-quality AUTOLOW opportunity
            opp = PipelineOpportunity(
                id="e2e-test",
                domain="cognitive",
                description="E2E test opportunity",
                source="e2e_test",
                detected_at=time.time(),
                snr_score=0.95,
                ihsan_score=0.98,
                urgency=0.8,
                autonomy_level=AutonomyLevel.AUTOLOW,
            )
            await pipeline.submit(opp)

            # Wait for full processing
            await asyncio.sleep(1.0)

            # Verify completion
            stats = pipeline.stats()
            assert stats["total_received"] == 1
            assert stats["total_executed"] == 1

            # Verify completed opportunity
            completed = pipeline.get_completed(limit=10)
            assert len(completed) >= 1
            assert completed[-1]["id"] == "e2e-test"
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_concurrency(self):
        """Test pipeline handles concurrent opportunities."""
        pipeline = OpportunityPipeline(max_concurrent=5)

        execution_count = 0

        async def slow_execute(opp):
            nonlocal execution_count
            await asyncio.sleep(0.1)
            execution_count += 1
            return {"success": True}

        pipeline.set_execution_callback(slow_execute)
        await pipeline.start()

        try:
            # Submit multiple opportunities
            for i in range(10):
                opp = PipelineOpportunity(
                    id=f"concurrent-{i}",
                    domain="test",
                    description=f"Concurrent test {i}",
                    source="test",
                    detected_at=time.time(),
                    snr_score=0.95,
                    ihsan_score=0.98,
                    autonomy_level=AutonomyLevel.AUTOLOW,
                )
                await pipeline.submit(opp)

            # Wait for all to complete
            await asyncio.sleep(2.0)

            assert execution_count == 10
        finally:
            await pipeline.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
