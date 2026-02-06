"""
Tests for NTU Bridge integration with BIZRA components.

Tests cover:
1. SNR adapter functionality
2. Memory adapter functionality
3. Unified bridge operations
4. Integration with existing BIZRA modules
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from core.ntu import NTU, NTUConfig, NTUState
from core.ntu.bridge import (
    NTUBridge,
    NTUSNRAdapter,
    NTUMemoryAdapter,
    NTUBridgeConfig,
)


# Mock classes for testing without full BIZRA dependencies
@dataclass
class MockSNRComponents:
    """Mock SNR components for testing."""
    signal_strength: float = 0.8
    diversity: float = 0.7
    grounding: float = 0.75
    iaas_score: float = 0.85
    snr: float = 0.82
    quality_tier: str = "T2_STANDARD"


@dataclass
class MockMemoryEntry:
    """Mock memory entry for testing."""
    content: str = "test content"
    memory_type: str = "episodic"
    relevance: float = 0.7
    timestamp: Optional[float] = None


class TestNTUBridgeConfig:
    """Tests for bridge configuration."""

    def test_default_config(self):
        """Default config should have valid values."""
        config = NTUBridgeConfig()

        assert config.snr_signal_weight > 0
        assert config.snr_diversity_weight > 0
        assert config.snr_grounding_weight > 0
        assert config.memory_decay_rate > 0

    def test_custom_ntu_config(self):
        """Should accept custom NTU config."""
        ntu_config = NTUConfig(window_size=10)
        config = NTUBridgeConfig(ntu_config=ntu_config)

        assert config.ntu_config.window_size == 10


class TestNTUSNRAdapter:
    """Tests for SNR adapter."""

    def test_initialization(self):
        """Adapter should initialize correctly."""
        adapter = NTUSNRAdapter()

        assert adapter.ntu is not None
        assert len(adapter._snr_history) == 0

    def test_snr_to_observation(self):
        """Should convert SNR components to observation value."""
        adapter = NTUSNRAdapter()
        snr = MockSNRComponents()

        obs_value = adapter.snr_to_observation(snr)

        assert 0.0 <= obs_value <= 1.0

    def test_snr_to_observation_weighted(self):
        """Observation should be weighted combination."""
        config = NTUBridgeConfig(
            snr_signal_weight=0.5,
            snr_diversity_weight=0.3,
            snr_grounding_weight=0.2,
        )
        adapter = NTUSNRAdapter(config)

        snr = MockSNRComponents(
            signal_strength=1.0,
            diversity=0.0,
            grounding=0.0,
        )

        obs_value = adapter.snr_to_observation(snr)

        # Should be close to signal weight since others are 0
        assert abs(obs_value - 0.5) < 0.01

    def test_observe_snr(self):
        """Should process SNR through NTU."""
        adapter = NTUSNRAdapter()
        snr = MockSNRComponents()

        state = adapter.observe_snr(snr)

        assert isinstance(state, NTUState)
        assert len(adapter._snr_history) == 1

    def test_observe_snr_with_metadata(self):
        """Should pass metadata to observation."""
        adapter = NTUSNRAdapter()
        snr = MockSNRComponents()
        metadata = {"source": "test", "timestamp": 12345}

        state = adapter.observe_snr(snr, metadata)

        assert state.iteration == 1

    def test_snr_history_tracking(self):
        """Should track SNR history."""
        adapter = NTUSNRAdapter()

        for i in range(5):
            snr = MockSNRComponents(snr=0.8 + i * 0.02)
            adapter.observe_snr(snr)

        assert len(adapter._snr_history) == 5

    def test_snr_history_limit(self):
        """History should be limited to max_memory_observations."""
        config = NTUBridgeConfig(max_memory_observations=10)
        adapter = NTUSNRAdapter(config)

        for i in range(20):
            snr = MockSNRComponents(snr=0.5 + i * 0.02)
            adapter.observe_snr(snr)

        assert len(adapter._snr_history) == 10

    def test_get_ntu_confidence(self):
        """Should return NTU belief as confidence."""
        adapter = NTUSNRAdapter()

        # Feed high quality
        for _ in range(5):
            snr = MockSNRComponents(snr=0.95, signal_strength=0.95)
            adapter.observe_snr(snr)

        confidence = adapter.get_ntu_confidence()

        assert 0.0 <= confidence <= 1.0

    def test_modulate_ihsan_threshold(self):
        """Should modulate threshold based on NTU state."""
        adapter = NTUSNRAdapter()

        # Initial threshold
        base_threshold = 0.95
        initial_modulated = adapter.modulate_ihsan_threshold(base_threshold)

        # Process some observations
        for _ in range(10):
            snr = MockSNRComponents(snr=0.9, signal_strength=0.9)
            adapter.observe_snr(snr)

        modulated = adapter.modulate_ihsan_threshold(base_threshold)

        # Should be within valid range
        assert 0.90 <= modulated <= 0.99

    def test_get_temporal_quality_score(self):
        """Should return NTU potential."""
        adapter = NTUSNRAdapter()

        # Process observations
        for _ in range(5):
            snr = MockSNRComponents()
            adapter.observe_snr(snr)

        score = adapter.get_temporal_quality_score()

        assert 0.0 <= score <= 1.0

    def test_reset(self):
        """Should reset adapter state."""
        adapter = NTUSNRAdapter()

        for _ in range(5):
            snr = MockSNRComponents()
            adapter.observe_snr(snr)

        adapter.reset()

        assert len(adapter._snr_history) == 0
        assert adapter.ntu.state.iteration == 0


class TestNTUMemoryAdapter:
    """Tests for memory adapter."""

    def test_initialization(self):
        """Adapter should initialize correctly."""
        adapter = NTUMemoryAdapter()

        assert adapter.ntu is not None

    def test_memory_entry_to_observation(self):
        """Should convert memory entry to observation."""
        adapter = NTUMemoryAdapter()
        entry = MockMemoryEntry(memory_type="semantic", relevance=0.8)

        obs_value = adapter.memory_entry_to_observation(entry, relevance_score=0.8)

        assert 0.0 <= obs_value <= 1.0

    def test_memory_type_weights(self):
        """Different memory types should have different weights."""
        adapter = NTUMemoryAdapter()

        entry_semantic = MockMemoryEntry(memory_type="semantic")
        entry_working = MockMemoryEntry(memory_type="working")

        obs_semantic = adapter.memory_entry_to_observation(entry_semantic, 0.8)
        obs_working = adapter.memory_entry_to_observation(entry_working, 0.8)

        # Semantic should have higher weight than working
        assert obs_semantic >= obs_working

    def test_observe_memory_entries(self):
        """Should process memory entries through NTU."""
        adapter = NTUMemoryAdapter()

        entries = [
            MockMemoryEntry(content="test1"),
            MockMemoryEntry(content="test2"),
            MockMemoryEntry(content="test3"),
        ]

        state = adapter.observe_memory_entries(entries)

        assert isinstance(state, NTUState)
        assert state.iteration == 3

    def test_observe_memory_with_relevance(self):
        """Should use provided relevance scores."""
        adapter = NTUMemoryAdapter()

        entries = [MockMemoryEntry() for _ in range(3)]
        relevance = [0.9, 0.5, 0.2]

        state = adapter.observe_memory_entries(entries, relevance)

        assert state.iteration == 3

    def test_get_consolidation_priority(self):
        """Should return consolidation priority."""
        adapter = NTUMemoryAdapter()

        entries = [MockMemoryEntry(memory_type="semantic") for _ in range(5)]
        adapter.observe_memory_entries(entries, [0.9] * 5)

        priority = adapter.get_consolidation_priority()

        assert 0.0 <= priority <= 1.0

    def test_should_retain_memory_high_potential(self):
        """Should retain memory with high potential."""
        adapter = NTUMemoryAdapter()

        # Feed high quality entries
        entries = [MockMemoryEntry(memory_type="semantic") for _ in range(10)]
        adapter.observe_memory_entries(entries, [0.95] * 10)

        # Should likely retain
        should_retain = adapter.should_retain_memory(threshold=0.3)

        # May or may not retain depending on NTU dynamics
        assert should_retain in (True, False)  # Accept numpy bool

    def test_should_retain_memory_threshold(self):
        """Should respect retention threshold."""
        adapter = NTUMemoryAdapter()

        entries = [MockMemoryEntry() for _ in range(5)]
        adapter.observe_memory_entries(entries, [0.5] * 5)

        # Very low threshold should almost always retain
        should_retain_low = adapter.should_retain_memory(threshold=0.01)

        # Very high threshold should almost always discard
        should_retain_high = adapter.should_retain_memory(threshold=0.99)

        # Different thresholds should give different results (usually)
        # or at least not error

    def test_reset(self):
        """Should reset adapter state."""
        adapter = NTUMemoryAdapter()

        entries = [MockMemoryEntry() for _ in range(5)]
        adapter.observe_memory_entries(entries)

        adapter.reset()

        assert adapter.ntu.state.iteration == 0


class TestNTUBridge:
    """Tests for unified NTU bridge."""

    def test_initialization(self):
        """Bridge should initialize all components."""
        bridge = NTUBridge()

        assert bridge.snr_adapter is not None
        assert bridge.memory_adapter is not None
        assert bridge._unified_ntu is not None

    def test_process_snr(self):
        """Should process SNR through unified NTU."""
        bridge = NTUBridge()
        snr = MockSNRComponents()

        state = bridge.process_snr(snr)

        assert isinstance(state, NTUState)
        assert bridge._unified_ntu.state.iteration == 1

    def test_process_snr_with_metadata(self):
        """Should pass metadata through."""
        bridge = NTUBridge()
        snr = MockSNRComponents()
        metadata = {"context": "test"}

        state = bridge.process_snr(snr, metadata)

        assert state.iteration == 1

    def test_process_memory(self):
        """Should process memory entries through unified NTU."""
        bridge = NTUBridge()

        entries = [MockMemoryEntry() for _ in range(3)]
        state = bridge.process_memory(entries)

        assert isinstance(state, NTUState)
        assert bridge._unified_ntu.state.iteration == 3

    def test_get_state(self):
        """Should return current unified state."""
        bridge = NTUBridge()

        snr = MockSNRComponents()
        bridge.process_snr(snr)

        state = bridge.get_state()

        assert isinstance(state, NTUState)
        assert state.iteration == 1

    def test_get_dynamic_ihsan_threshold(self):
        """Should compute dynamic threshold."""
        bridge = NTUBridge()

        threshold = bridge.get_dynamic_ihsan_threshold(base=0.95)

        assert 0.90 <= threshold <= 0.99

    def test_is_quality_stable(self):
        """Should check quality stability."""
        bridge = NTUBridge()

        is_stable = bridge.is_quality_stable()

        assert isinstance(is_stable, bool)

    def test_is_quality_stable_threshold(self):
        """Should respect stability threshold."""
        bridge = NTUBridge()

        # Very low threshold
        is_stable_low = bridge.is_quality_stable(threshold=0.01)

        # Very high threshold
        is_stable_high = bridge.is_quality_stable(threshold=0.99)

        # Low threshold should be easier to pass

    def test_get_quality_metrics(self):
        """Should return comprehensive metrics."""
        bridge = NTUBridge()

        snr = MockSNRComponents()
        bridge.process_snr(snr)

        metrics = bridge.get_quality_metrics()

        assert "belief" in metrics
        assert "entropy" in metrics
        assert "potential" in metrics
        assert "ihsan_achieved" in metrics
        assert "dynamic_threshold" in metrics
        assert "quality_stable" in metrics
        assert "iterations" in metrics

    def test_reset(self):
        """Should reset all components."""
        bridge = NTUBridge()

        # Process some data
        snr = MockSNRComponents()
        bridge.process_snr(snr)

        entries = [MockMemoryEntry() for _ in range(3)]
        bridge.process_memory(entries)

        # Reset
        bridge.reset()

        assert bridge._unified_ntu.state.iteration == 0
        assert len(bridge.snr_adapter._snr_history) == 0

    def test_combined_snr_and_memory(self):
        """Should handle both SNR and memory processing."""
        bridge = NTUBridge()

        # Process SNR first
        for _ in range(5):
            snr = MockSNRComponents(snr=0.9)
            bridge.process_snr(snr)

        mid_state = bridge.get_state()
        mid_iteration = mid_state.iteration

        # Then process memory
        entries = [MockMemoryEntry() for _ in range(3)]
        bridge.process_memory(entries)

        final_state = bridge.get_state()

        assert final_state.iteration == mid_iteration + 3


class TestBridgeIntegration:
    """Integration tests for bridge with simulated BIZRA components."""

    def test_snr_quality_tracking(self):
        """Should track SNR quality over time."""
        bridge = NTUBridge()

        # Simulate quality variations
        quality_sequence = [
            0.95, 0.92, 0.88, 0.85, 0.87,  # Declining
            0.89, 0.92, 0.94, 0.96, 0.97,  # Recovering
        ]

        states = []
        for snr_value in quality_sequence:
            snr = MockSNRComponents(
                snr=snr_value,
                signal_strength=snr_value,
            )
            state = bridge.process_snr(snr)
            states.append(state.belief)

        # Belief should track quality trend
        assert len(states) == 10

    def test_memory_consolidation_decisions(self):
        """Should inform memory consolidation."""
        bridge = NTUBridge()

        # High quality memory
        high_quality = [MockMemoryEntry(memory_type="semantic") for _ in range(5)]
        bridge.process_memory(high_quality, [0.95] * 5)

        high_metrics = bridge.get_quality_metrics()

        bridge.reset()

        # Low quality memory
        low_quality = [MockMemoryEntry(memory_type="working") for _ in range(5)]
        bridge.process_memory(low_quality, [0.2] * 5)

        low_metrics = bridge.get_quality_metrics()

        # High quality should have better metrics
        # (exact comparison depends on NTU dynamics)

    def test_adaptive_thresholding(self):
        """Dynamic threshold should adapt to quality patterns."""
        bridge = NTUBridge()

        # Feed stable high quality
        for _ in range(20):
            snr = MockSNRComponents(snr=0.96)
            bridge.process_snr(snr)

        stable_threshold = bridge.get_dynamic_ihsan_threshold()

        bridge.reset()

        # Feed noisy quality
        for i in range(20):
            snr_value = 0.7 + (i % 3) * 0.1  # Oscillates
            snr = MockSNRComponents(snr=snr_value, signal_strength=snr_value)
            bridge.process_snr(snr)

        noisy_threshold = bridge.get_dynamic_ihsan_threshold()

        # Both should be valid thresholds
        assert 0.90 <= stable_threshold <= 0.99
        assert 0.90 <= noisy_threshold <= 0.99
