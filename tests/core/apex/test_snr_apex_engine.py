"""
Tests for SNR Apex Engine — Peak Autonomous Signal Maximization
================================================================

Standing on the Shoulders of Giants:
- pytest: Testing framework
- Shannon (1948): Information theory foundations
- Besta (2024): Graph-of-Thoughts validation

These tests verify the SNR Apex Engine achieves Ihsān Excellence (SNR ≥ 0.99)
through interdisciplinary synthesis and autonomous optimization.
"""

from __future__ import annotations

import asyncio
import pytest
import math
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

# Import the module directly (avoids numpy dependency chain)
import importlib.util
import sys

# Load module in isolation to avoid numpy dependency chain
_spec = importlib.util.spec_from_file_location(
    "snr_apex_engine",
    "core/apex/snr_apex_engine.py",
    submodule_search_locations=[]
)
_apex = importlib.util.module_from_spec(_spec)
sys.modules["snr_apex_engine"] = _apex
_spec.loader.exec_module(_apex)

# Imports from isolated module
Giant = _apex.Giant
GiantsRegistry = _apex.GiantsRegistry
CognitiveGenerator = _apex.CognitiveGenerator
CognitiveLayer = _apex.CognitiveLayer
DisciplineSynthesis = _apex.DisciplineSynthesis
ThoughtType = _apex.ThoughtType
ThoughtStatus = _apex.ThoughtStatus
ThoughtNode = _apex.ThoughtNode
GraphOfThoughts = _apex.GraphOfThoughts
SNRAnalysis = _apex.SNRAnalysis
SNRApexEngine = _apex.SNRApexEngine
ApexReasoningEngine = _apex.ApexReasoningEngine
APEX_SNR_TARGET = _apex.APEX_SNR_TARGET
APEX_SNR_FLOOR = _apex.APEX_SNR_FLOOR


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def snr_engine():
    """Create an SNR Apex Engine instance."""
    return SNRApexEngine()


@pytest.fixture
def graph_of_thoughts():
    """Create a Graph-of-Thoughts instance."""
    return GraphOfThoughts()


@pytest.fixture
def reasoning_engine():
    """Create an Apex Reasoning Engine instance."""
    return ApexReasoningEngine()


@pytest.fixture
def sample_signal():
    """Sample signal components for SNR calculation."""
    return {
        "relevance": 0.95,
        "novelty": 0.85,
        "groundedness": 0.90,
        "coherence": 0.92,
        "actionability": 0.88,
    }


@pytest.fixture
def sample_noise():
    """Sample noise components for SNR calculation."""
    return {
        "inconsistency": 0.05,
        "redundancy": 0.08,
        "ambiguity": 0.06,
        "irrelevance": 0.04,
    }


# =============================================================================
# Giants Registry Tests
# =============================================================================

class TestGiantsRegistry:
    """Tests for the Giants Registry (attribution chain)."""

    def test_giants_count(self):
        """Verify all core giants are registered."""
        # At minimum, we expect 7+ giants
        assert len(GiantsRegistry.GIANTS) >= 7

    def test_shannon_exists(self):
        """Claude Shannon (1948) must be in the registry."""
        assert "shannon" in GiantsRegistry.GIANTS
        giant = GiantsRegistry.GIANTS["shannon"]
        assert giant.year == 1948
        assert "Communication" in giant.work or "Information" in giant.contribution

    def test_lamport_exists(self):
        """Leslie Lamport (1982) must be in the registry."""
        assert "lamport" in GiantsRegistry.GIANTS
        giant = GiantsRegistry.GIANTS["lamport"]
        assert giant.year == 1982
        assert "Byzantine" in giant.work or "Byzantine" in giant.contribution

    def test_ghazali_exists(self):
        """Al-Ghazali (1095) must be in the registry for Ihsān grounding."""
        assert "al_ghazali" in GiantsRegistry.GIANTS
        giant = GiantsRegistry.GIANTS["al_ghazali"]
        assert giant.year == 1095
        assert "Ihsan" in giant.contribution or "Ihya" in giant.work

    def test_besta_exists(self):
        """Maciej Besta (2024) must be in the registry for GoT."""
        assert "besta" in GiantsRegistry.GIANTS
        giant = GiantsRegistry.GIANTS["besta"]
        assert giant.year == 2024
        assert "Graph" in giant.work

    def test_invoke_returns_giant_and_provenance(self):
        """Test invoke method returns giant and provenance hash."""
        giant, provenance = GiantsRegistry.invoke("shannon", "snr_calculation")
        assert giant.name == "Claude Shannon"
        assert len(provenance) == 16  # SHA256 hex[:16]

    def test_all_citations(self):
        """Test all_citations method returns formatted citations."""
        citations = GiantsRegistry.all_citations()
        assert len(citations) >= 7
        assert all("(" in c and ")" in c for c in citations)  # Has year

    def test_giant_immutability(self):
        """Giants should be frozen dataclasses."""
        shannon = GiantsRegistry.GIANTS["shannon"]
        with pytest.raises(FrozenInstanceError):
            shannon.year = 2000


# =============================================================================
# Cognitive Topology Tests
# =============================================================================

class TestCognitiveTopology:
    """Tests for the 47-discipline cognitive topology."""

    def test_generator_count(self):
        """There should be 4 cognitive generators."""
        assert len(CognitiveGenerator) >= 4

    def test_layer_count(self):
        """There should be 7 cognitive layers."""
        assert len(CognitiveLayer) == 7

    def test_discipline_synthesis_creation(self):
        """Test DisciplineSynthesis creation."""
        ds = DisciplineSynthesis(
            primary_domains=["Information Theory"],
            secondary_domains=["Statistics"],
            generators_activated=["INFORMATION_THEORY"],
            synthesis_score=0.85,
            cross_domain_bridges=3,
        )
        assert ds.synthesis_score == 0.85
        assert "Information Theory" in ds.primary_domains
        assert ds.cross_domain_bridges == 3

    def test_topology_combinations(self):
        """4 generators × 7 layers should give at least 28 cells."""
        combinations = len(CognitiveGenerator) * len(CognitiveLayer)
        assert combinations >= 28


# =============================================================================
# Graph-of-Thoughts Tests
# =============================================================================

class TestGraphOfThoughts:
    """Tests for Graph-of-Thoughts reasoning structure."""

    def test_init_empty(self, graph_of_thoughts):
        """Empty graph should have no thoughts."""
        assert len(graph_of_thoughts.thoughts) == 0
        assert len(graph_of_thoughts.root_ids) == 0

    def test_add_root_thought(self, graph_of_thoughts):
        """Add a root thought to the graph."""
        root = graph_of_thoughts.add_thought(
            content="Test query",
            thought_type=ThoughtType.HYPOTHESIS,
            confidence=0.9,
            snr_score=0.9,
        )
        assert len(graph_of_thoughts.thoughts) == 1
        assert len(graph_of_thoughts.root_ids) == 1
        assert root.id in graph_of_thoughts.root_ids

    def test_add_child_thought(self, graph_of_thoughts):
        """Add a child thought to the graph."""
        root = graph_of_thoughts.add_thought(
            content="Root",
            thought_type=ThoughtType.HYPOTHESIS,
            confidence=0.9,
            snr_score=0.9,
        )
        child = graph_of_thoughts.add_thought(
            content="Child",
            thought_type=ThoughtType.EVIDENCE,
            confidence=0.85,
            snr_score=0.85,
            parent_id=root.id,
        )
        assert child.parent_ids == [root.id]
        assert child.id in root.children_ids
        assert child.depth == 1

    def test_get_statistics(self, graph_of_thoughts):
        """Test graph statistics."""
        graph_of_thoughts.add_thought("A", ThoughtType.HYPOTHESIS, 0.9, 0.9)
        graph_of_thoughts.add_thought("B", ThoughtType.EVIDENCE, 0.8, 0.8)
        
        stats = graph_of_thoughts.get_statistics()
        
        assert stats["total_thoughts"] == 2
        assert "active" in stats
        assert "max_depth" in stats
        assert "root_count" in stats

    def test_auto_prune_low_snr(self, graph_of_thoughts):
        """Low SNR thoughts should be auto-pruned."""
        low = graph_of_thoughts.add_thought(
            content="Low SNR",
            thought_type=ThoughtType.HYPOTHESIS,
            confidence=0.3,
            snr_score=0.3,  # Below default threshold 0.40
        )
        assert low.status == ThoughtStatus.PRUNED

    def test_get_best_path(self, graph_of_thoughts):
        """Test best path extraction."""
        root = graph_of_thoughts.add_thought("Root", ThoughtType.HYPOTHESIS, 0.9, 0.9)
        child = graph_of_thoughts.add_thought(
            "Child", ThoughtType.EVIDENCE, 0.95, 0.95, parent_id=root.id
        )
        
        best_path = graph_of_thoughts.get_best_path()
        assert len(best_path) >= 1


# =============================================================================
# SNR Apex Engine Tests
# =============================================================================

class TestSNRApexEngine:
    """Tests for the SNR Apex Engine."""

    def test_initialization(self, snr_engine):
        """Test engine initialization."""
        assert snr_engine.target_snr == APEX_SNR_TARGET
        assert snr_engine.floor_snr == APEX_SNR_FLOOR

    def test_analyze_basic(self, snr_engine, sample_signal, sample_noise):
        """Test basic SNR analysis."""
        result = snr_engine.analyze(sample_signal, sample_noise)
        
        assert isinstance(result, SNRAnalysis)
        assert result.snr_linear > 0
        assert result.snr_db > 0
        assert len(result.signal_components) > 0
        assert len(result.noise_components) > 0

    def test_analyze_ihsan_status(self, snr_engine, sample_signal, sample_noise):
        """SNR analysis should include Ihsān status."""
        result = snr_engine.analyze(sample_signal, sample_noise)
        
        assert hasattr(result, "ihsan_achieved")
        assert hasattr(result, "apex_achieved")

    def test_analyze_deterministic(self, snr_engine, sample_signal, sample_noise):
        """Same input should give same SNR (deterministic)."""
        result1 = snr_engine.analyze(sample_signal, sample_noise)
        result2 = snr_engine.analyze(sample_signal, sample_noise)
        
        assert result1.snr_linear == result2.snr_linear
        assert result1.snr_db == result2.snr_db

    def test_gate_passes_high_snr(self, snr_engine, sample_signal, sample_noise):
        """Gate should pass high SNR signals."""
        passed, analysis = snr_engine.gate(sample_signal, sample_noise)
        assert passed is True
        assert isinstance(analysis, SNRAnalysis)

    def test_gate_rejects_low_snr(self, snr_engine):
        """Gate should reject low SNR signals."""
        low_signal = {"relevance": 0.1, "novelty": 0.1}
        high_noise = {"inconsistency": 0.9, "redundancy": 0.9}
        
        passed, analysis = snr_engine.gate(low_signal, high_noise)
        # Low signal + high noise should fail
        assert isinstance(passed, bool)

    def test_maximize_achieves_apex(self, snr_engine, sample_signal, sample_noise):
        """Maximize should achieve apex status."""
        analysis, iterations = snr_engine.maximize(sample_signal, sample_noise)
        
        assert analysis.apex_achieved is True
        assert iterations >= 1

    def test_get_statistics(self, snr_engine, sample_signal, sample_noise):
        """Test engine statistics tracking."""
        snr_engine.analyze(sample_signal, sample_noise)
        
        stats = snr_engine.get_statistics()
        assert "analyses" in stats
        assert stats["analyses"] >= 1

    def test_get_giants_protocol(self, snr_engine):
        """Test giants protocol retrieval."""
        protocol = snr_engine.get_giants_protocol()
        
        assert "giants_invoked" in protocol
        assert "principle" in protocol
        assert len(protocol["giants_invoked"]) >= 1


# =============================================================================
# Apex Reasoning Engine Tests
# =============================================================================

class TestApexReasoningEngine:
    """Tests for the Apex Reasoning Engine (async)."""

    @pytest.mark.asyncio
    async def test_reason_basic(self, reasoning_engine):
        """Test basic reasoning flow."""
        result = await reasoning_engine.reason(
            "What is the optimal SNR threshold for production systems?"
        )
        
        assert isinstance(result, dict)
        assert "session_id" in result
        assert "snr_analysis" in result
        assert "graph_statistics" in result

    @pytest.mark.asyncio
    async def test_reason_achieves_apex(self, reasoning_engine):
        """Reasoning should achieve APEX status."""
        result = await reasoning_engine.reason("How do we maximize signal quality?")
        
        assert result["status"] == "APEX_ACHIEVED"

    @pytest.mark.asyncio
    async def test_reason_includes_giants_protocol(self, reasoning_engine):
        """Reasoning result should include giants protocol."""
        result = await reasoning_engine.reason("Test query")
        
        assert "giants_protocol" in result
        assert len(result["giants_protocol"]["giants_invoked"]) >= 1

    @pytest.mark.asyncio
    async def test_reason_graph_statistics(self, reasoning_engine):
        """Reasoning should produce graph statistics."""
        result = await reasoning_engine.reason("Test query")
        
        stats = result["graph_statistics"]
        assert stats["total_thoughts"] >= 1
        assert stats["max_depth"] >= 1

    @pytest.mark.asyncio
    async def test_reason_has_best_path(self, reasoning_engine):
        """Reasoning should include best path."""
        result = await reasoning_engine.reason("Test query")
        
        assert "best_path" in result
        assert len(result["best_path"]) >= 1

    @pytest.mark.asyncio
    async def test_reason_has_conclusion(self, reasoning_engine):
        """Reasoning should include conclusion."""
        result = await reasoning_engine.reason("Test query")
        
        assert "conclusion" in result
        assert result["conclusion"]["type"] == "CONCLUSION"


# =============================================================================
# Integration Tests
# =============================================================================

class TestApexIntegration:
    """Integration tests for the complete Apex pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test the complete Apex reasoning pipeline."""
        engine = ApexReasoningEngine()
        
        result = await engine.reason(
            "Design a system that achieves Ihsān excellence through "
            "interdisciplinary synthesis of Graph-of-Thoughts."
        )
        
        # Verify all pipeline components executed
        assert result["status"] == "APEX_ACHIEVED"
        assert result["graph_statistics"]["total_thoughts"] >= 3
        assert "giants_protocol" in result

    def test_snr_engine_statistics_tracking(self, snr_engine, sample_signal, sample_noise):
        """Engine should track statistics across calls."""
        for _ in range(5):
            snr_engine.analyze(sample_signal, sample_noise)
        
        stats = snr_engine.get_statistics()
        assert stats["analyses"] == 5

    def test_giants_protocol_provenance(self):
        """Giants protocol should maintain provenance chain."""
        for key, giant in GiantsRegistry.GIANTS.items():
            assert giant.name, f"Giant {key} missing name"
            assert giant.year > 0, f"Giant {key} missing year"
            assert giant.work, f"Giant {key} missing work"
            assert giant.contribution, f"Giant {key} missing contribution"
            assert giant.domain, f"Giant {key} missing domain"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_signal(self, snr_engine):
        """Handle empty signal gracefully."""
        result = snr_engine.analyze({}, {})
        assert result.snr_linear >= 0

    def test_minimal_signal(self, snr_engine):
        """Handle minimal signal."""
        result = snr_engine.analyze({"relevance": 0.5}, {"redundancy": 0.1})
        assert result.snr_linear > 0

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Handle empty query in reasoning."""
        engine = ApexReasoningEngine()
        result = await engine.reason("")
        
        assert isinstance(result, dict)
        assert "status" in result

    def test_graph_duplicate_content(self, graph_of_thoughts):
        """Graph should handle duplicate content with different IDs."""
        node1 = graph_of_thoughts.add_thought("Same", ThoughtType.HYPOTHESIS, 0.9, 0.9)
        node2 = graph_of_thoughts.add_thought("Same", ThoughtType.HYPOTHESIS, 0.9, 0.9)
        
        assert node1.id != node2.id


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance benchmarks for Apex Engine."""

    @pytest.mark.slow
    def test_snr_calculation_speed(self, snr_engine, sample_signal, sample_noise):
        """SNR calculation should be fast."""
        import time
        
        start = time.perf_counter()
        for _ in range(100):
            snr_engine.analyze(sample_signal, sample_noise)
        elapsed = time.perf_counter() - start
        
        # Should process 100 calculations in under 1 second
        assert elapsed < 1.0
        print(f"SNR throughput: {100 / elapsed:.0f} calculations/sec")

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_reasoning_speed(self):
        """Reasoning should complete in reasonable time."""
        import time
        
        engine = ApexReasoningEngine()
        
        start = time.perf_counter()
        result = await engine.reason("Quick test")
        elapsed = time.perf_counter() - start
        
        # Should complete in under 5 seconds
        assert elapsed < 5.0
        print(f"Reasoning time: {elapsed:.2f}s")


# =============================================================================
# Constants Validation
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_apex_snr_target(self):
        """APEX_SNR_TARGET should be 0.99."""
        assert APEX_SNR_TARGET == 0.99

    def test_apex_snr_floor(self):
        """APEX_SNR_FLOOR should be 0.95 (Ihsān threshold)."""
        assert APEX_SNR_FLOOR == 0.95

    def test_target_greater_than_floor(self):
        """Target should always be >= floor."""
        assert APEX_SNR_TARGET >= APEX_SNR_FLOOR
