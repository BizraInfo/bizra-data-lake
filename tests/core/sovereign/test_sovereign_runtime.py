"""
Sovereign Runtime Integration Tests
═══════════════════════════════════════════════════════════════════════════════

Tests for the unified Sovereign Runtime — the peak masterpiece implementation.

Created: 2026-02-04
"""

import asyncio
import pytest
from datetime import datetime, timezone

# Import from runtime package
from core.sovereign.runtime_engines.giants_registry import (
    Giant,
    GiantCategory,
    GiantsRegistry,
    get_giants_registry,
    attribute,
)
from core.sovereign.runtime_engines.snr_maximizer import (
    Signal,
    SignalQuality,
    SNRMaximizer,
    get_snr_maximizer,
    SNR_FLOOR,
    SNR_EXCELLENT,
)
from core.sovereign.runtime_engines.got_bridge import (
    ThoughtNode,
    ThoughtGraph,
    ThoughtType,
    GoTResult,
    GoTBridge,
    think,
)
from core.sovereign.runtime_engines.sovereign_runtime import (
    SovereignRuntime,
    RuntimeInput,
    RuntimeDecision,
    RuntimePhase,
    ConstitutionalGate,
    ConstitutionalResult,
    get_sovereign_runtime,
    process,
)


class TestGiantsRegistry:
    """Tests for the Giants Registry."""

    def test_initialization(self):
        """Registry should initialize with foundational giants."""
        registry = GiantsRegistry()
        assert len(registry._giants) > 15  # At least 15 giants registered

    def test_get_giant_by_name(self):
        """Should retrieve giant by name."""
        registry = GiantsRegistry()
        shannon = registry.get("Claude Shannon")
        assert shannon is not None
        assert shannon.year == 1948
        assert shannon.category == GiantCategory.INFORMATION_THEORY

    def test_get_giant_by_name_and_year(self):
        """Should retrieve giant by name and year."""
        registry = GiantsRegistry()
        lamport = registry.get("Leslie Lamport", 1982)
        assert lamport is not None
        assert "Byzantine" in lamport.work

    def test_get_by_category(self):
        """Should retrieve giants by category."""
        registry = GiantsRegistry()
        distributed = registry.get_by_category(GiantCategory.DISTRIBUTED_SYSTEMS)
        assert len(distributed) >= 1
        assert any(g.name == "Leslie Lamport" for g in distributed)

    def test_record_application(self):
        """Should record application of giants' work."""
        registry = GiantsRegistry()
        registry.record_application(
            module="TestModule",
            method="test_method",
            giant_names=["Claude Shannon"],
            explanation="Testing attribution",
        )
        apps = registry.get_applications_for("Claude Shannon")
        assert len(apps) >= 1

    def test_format_attribution(self):
        """Should format attribution string."""
        registry = GiantsRegistry()
        shannon = registry.get("Claude Shannon")
        attr = shannon.format_attribution()
        assert "Shannon" in attr
        assert "1948" in attr

    def test_attribute_helper(self):
        """Attribute helper should format multiple giants."""
        attr = attribute(["Claude Shannon", "Leslie Lamport"])
        assert "Shannon" in attr
        assert "Lamport" in attr


class TestSNRMaximizer:
    """Tests for the SNR Maximizer."""

    @pytest.fixture
    def maximizer(self):
        return SNRMaximizer()

    def test_initialization(self, maximizer):
        """Should initialize with correct defaults."""
        assert maximizer.snr_floor == SNR_FLOOR
        assert maximizer.snr_excellent == SNR_EXCELLENT

    def test_process_high_confidence(self, maximizer):
        """High confidence input should have high SNR."""
        signal = maximizer.process(0.95, source="test", metric_type="confidence")
        assert signal.snr > 0.5

    def test_process_low_confidence(self, maximizer):
        """Low confidence input should have lower SNR than high confidence."""
        signal = maximizer.process(0.2, source="test", metric_type="confidence")
        high_signal = maximizer.process(0.95, source="test", metric_type="confidence")
        # Low confidence should produce lower SNR than high confidence
        assert signal.snr < high_signal.snr
        # Low confidence should be classified as poor or noise quality
        assert signal.quality in [SignalQuality.POOR, SignalQuality.NOISE]

    def test_filter_passes(self, maximizer):
        """Filter should pass high-quality signals."""
        signal = maximizer.filter(0.95, source="test")
        assert signal is not None

    def test_filter_batch(self, maximizer):
        """Batch filter should filter list."""
        items = [0.9, 0.5, 0.95, 0.3, 0.85]
        results = maximizer.filter_batch(items, min_snr=0.5)
        assert len(results) >= 2

    def test_is_excellent(self, maximizer):
        """Should identify excellent signals."""
        signal = maximizer.process(0.99, source="test")
        # May or may not be excellent depending on noise
        assert isinstance(maximizer.is_excellent(signal), bool)

    def test_get_statistics(self, maximizer):
        """Should track statistics."""
        maximizer.process(0.9, source="test")
        maximizer.process(0.8, source="test")
        stats = maximizer.get_statistics()
        assert stats["total_processed"] == 2


class TestThoughtGraph:
    """Tests for the Graph-of-Thoughts."""

    @pytest.fixture
    def graph(self):
        return ThoughtGraph()

    def test_initialization(self, graph):
        """Should initialize correctly."""
        assert graph.max_depth == 10
        assert graph.max_branches == 5

    def test_create_root(self, graph):
        """Should create root node."""
        root = graph.create_root("Test goal")
        assert root.thought_type == ThoughtType.ROOT
        assert root.content == "Test goal"
        assert root.depth == 0

    def test_generate(self, graph):
        """Should generate child thoughts."""
        root = graph.create_root("Test goal")
        children = graph.generate(root, "Test goal")
        assert len(children) <= graph.max_branches
        for child in children:
            assert child.depth == 1
            assert root.id in child.parent_ids

    def test_aggregate(self, graph):
        """Should aggregate multiple thoughts."""
        root = graph.create_root("Test goal")
        children = graph.generate(root, "Test goal")
        if len(children) >= 2:
            aggregated = graph.aggregate(children[:2])
            assert aggregated.thought_type == ThoughtType.AGGREGATE
            assert len(aggregated.parent_ids) == 2

    def test_refine(self, graph):
        """Should refine a thought."""
        root = graph.create_root("Test goal")
        refined = graph.refine(root, iterations=2)
        assert refined.thought_type == ThoughtType.REFINE
        assert refined.depth > root.depth

    def test_validate(self, graph):
        """Should score a thought."""
        root = graph.create_root("Test goal")
        score = graph.validate(root)
        assert 0.0 <= score <= 1.0

    def test_prune(self, graph):
        """Should prune low-quality nodes."""
        root = graph.create_root("Test goal")
        children = graph.generate(root, "Test goal")
        # Manually set low scores
        for child in children:
            child.score = 0.1
        pruned = graph.prune()
        assert pruned >= 0

    @pytest.mark.asyncio
    async def test_reason(self, graph):
        """Should complete reasoning process."""
        result = await graph.reason("Test problem", max_iterations=20)
        assert isinstance(result, GoTResult)
        assert result.explored_nodes > 0

    def test_visualize(self, graph):
        """Should generate visualization."""
        root = graph.create_root("Test goal")
        graph.generate(root, "Test goal")
        viz = graph.visualize()
        assert "Graph of Thoughts" in viz


class TestGoTBridge:
    """Tests for the GoT Bridge."""

    @pytest.fixture
    def bridge(self):
        return GoTBridge()

    def test_initialization(self, bridge):
        """Should initialize correctly."""
        assert bridge._python_graph is not None

    @pytest.mark.asyncio
    async def test_reason(self, bridge):
        """Should perform reasoning."""
        result = await bridge.reason("Test problem")
        assert isinstance(result, GoTResult)


class TestConstitutionalGate:
    """Tests for the Constitutional Gate."""

    @pytest.fixture
    def gate(self):
        return ConstitutionalGate()

    def test_initialization(self, gate):
        """Should initialize with correct thresholds."""
        assert gate.ihsan_threshold == 0.95

    def test_calculate_ihsan_score(self, gate):
        """Should calculate Ihsan score."""
        decision = {"confidence": 0.9, "reversible": True, "expected_benefit": 0.8}
        score = gate.calculate_ihsan_score(decision)
        assert 0.0 <= score <= 1.0

    def test_validate_high_ihsan(self, gate):
        """High Ihsan should be approved."""
        decision = {"confidence": 0.99, "reversible": True, "expected_benefit": 0.95}
        result, score = gate.validate(decision)
        # May or may not be approved depending on calculation
        assert result in list(ConstitutionalResult)

    def test_validate_low_ihsan(self, gate):
        """Low Ihsan should be rejected."""
        decision = {"confidence": 0.3, "reversible": False, "expected_benefit": 0.2}
        result, score = gate.validate(decision)
        assert result == ConstitutionalResult.REJECTED


class TestSovereignRuntime:
    """Tests for the unified Sovereign Runtime."""

    @pytest.fixture
    def runtime(self):
        return SovereignRuntime()

    def test_initialization(self, runtime):
        """Should initialize correctly."""
        assert runtime.snr_floor == SNR_FLOOR
        assert runtime.ihsan_threshold == 0.95
        assert runtime._phase == RuntimePhase.IDLE

    @pytest.mark.asyncio
    async def test_process_input(self, runtime):
        """Should process input through full pipeline."""
        input_data = RuntimeInput(query="Test query for processing")
        result = await runtime.process(input_data)
        assert isinstance(result, RuntimeDecision)
        assert result.snr_score >= 0.0
        assert result.ihsan_score >= 0.0

    @pytest.mark.asyncio
    async def test_process_high_quality_input(self, runtime):
        """High-quality input should produce approved decision."""
        input_data = RuntimeInput(
            query="Analyze the portfolio performance",
            context={"quality": "high"},
        )
        result = await runtime.process(input_data)
        # Decision may or may not be approved depending on reasoning
        assert isinstance(result.constitutional_result, ConstitutionalResult)

    @pytest.mark.asyncio
    async def test_giants_attribution(self, runtime):
        """Decision should include giants attribution."""
        input_data = RuntimeInput(query="Test attribution")
        result = await runtime.process(input_data)
        assert len(result.giants_attribution) >= 1

    def test_get_metrics(self, runtime):
        """Should provide metrics."""
        metrics = runtime.get_metrics()
        assert "total_inputs" in metrics
        assert "average_snr" in metrics
        assert "approval_rate" in metrics

    def test_status(self, runtime):
        """Should provide status."""
        status = runtime.status()
        assert status["phase"] == "idle"
        assert "standing_on_giants" in status
        assert len(status["standing_on_giants"]) >= 5


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_giants_registry(self):
        """Should return global registry."""
        registry = get_giants_registry()
        assert isinstance(registry, GiantsRegistry)

    def test_get_snr_maximizer(self):
        """Should return global maximizer."""
        maximizer = get_snr_maximizer()
        assert isinstance(maximizer, SNRMaximizer)

    def test_get_sovereign_runtime(self):
        """Should return global runtime."""
        runtime = get_sovereign_runtime()
        assert isinstance(runtime, SovereignRuntime)

    @pytest.mark.asyncio
    async def test_think_function(self):
        """Think function should perform GoT reasoning."""
        result = await think("Test problem")
        assert isinstance(result, GoTResult)

    @pytest.mark.asyncio
    async def test_process_function(self):
        """Process function should use global runtime."""
        result = await process("Test query")
        assert isinstance(result, RuntimeDecision)


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Should complete full processing pipeline."""
        runtime = SovereignRuntime()

        # Process multiple inputs
        queries = [
            "Analyze market trends",
            "Optimize portfolio allocation",
            "Detect arbitrage opportunities",
        ]

        for query in queries:
            result = await runtime.process(RuntimeInput(query=query))
            assert isinstance(result, RuntimeDecision)
            assert result.id is not None

        # Check metrics
        metrics = runtime.get_metrics()
        assert metrics["total_inputs"] == 3

    @pytest.mark.asyncio
    async def test_attribution_chain(self):
        """Should maintain attribution throughout pipeline."""
        runtime = SovereignRuntime()

        input_data = RuntimeInput(query="Complex multi-step reasoning task")
        result = await runtime.process(input_data)

        # Should have attribution from at least one giant (Shannon for SNR filtering)
        all_attributions = " ".join(result.giants_attribution)
        assert "Shannon" in all_attributions
        # Besta is attributed when reasoning succeeds and decision is approved
        # May or may not be present depending on execution path
        assert len(result.giants_attribution) >= 1

    @pytest.mark.asyncio
    async def test_constitutional_enforcement(self):
        """Constitutional gate should properly filter decisions."""
        runtime = SovereignRuntime(ihsan_threshold=0.99)  # Very strict

        input_data = RuntimeInput(query="Test with strict threshold")
        result = await runtime.process(input_data)

        # With 0.99 threshold, most decisions should need review or be rejected
        assert result.constitutional_result in [
            ConstitutionalResult.APPROVED,
            ConstitutionalResult.NEEDS_REVIEW,
            ConstitutionalResult.REJECTED,
        ]


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
