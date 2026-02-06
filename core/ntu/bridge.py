"""
NTU Bridge — Integration with BIZRA Core Architecture

This module bridges the NeuroTemporal Unit (NTU) with existing BIZRA
components, implementing the Reduction Theorem mapping:

- NTU.belief ↔ IhsanGate (SNR quality enforcement)
- NTU.memory ↔ LivingMemoryCore (sliding window → episodic memory)
- NTU.temporal_consistency ↔ SNRCalculatorV2.diversity (temporal coherence)
- NTU.neural_prior ↔ embedding-based priors

This provides bidirectional data flow:
1. NTU → BIZRA: Pattern detection informs quality gates
2. BIZRA → NTU: SNR/memory data feeds NTU observations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging

import numpy as np

from .ntu import NTU, NTUConfig, NTUState, Observation

# Avoid circular imports
if TYPE_CHECKING:
    from core.iaas.snr_v2 import SNRComponentsV2, SNRCalculatorV2
    from core.living_memory import LivingMemoryCore, MemoryEntry

logger = logging.getLogger(__name__)


@dataclass
class NTUBridgeConfig:
    """Configuration for NTU bridge."""
    # NTU configuration
    ntu_config: Optional[NTUConfig] = None

    # SNR mapping weights
    snr_signal_weight: float = 0.4
    snr_diversity_weight: float = 0.3
    snr_grounding_weight: float = 0.3

    # Memory adapter settings
    memory_decay_rate: float = 0.95  # Per-step decay for memory relevance
    max_memory_observations: int = 100


class NTUSNRAdapter:
    """
    Adapter connecting NTU to SNR v2 system.

    Provides bidirectional mapping:
    - SNR → NTU: Convert SNR components to NTU observations
    - NTU → SNR: Use NTU belief to modulate SNR confidence

    This implements the reduction:
    - NTU.belief ≈ SNRComponentsV2.snr (quality metric)
    - NTU.entropy ≈ 1 - SNRComponentsV2.diversity
    """

    def __init__(self, config: Optional[NTUBridgeConfig] = None):
        """Initialize SNR adapter."""
        self.config = config or NTUBridgeConfig()
        self.ntu = NTU(self.config.ntu_config or NTUConfig())

        # History for temporal analysis
        self._snr_history: List[float] = []

    def snr_to_observation(self, snr_components: "SNRComponentsV2") -> float:
        """
        Convert SNR components to a single NTU observation value.

        The observation value is a weighted combination of SNR components,
        normalized to [0, 1].

        Args:
            snr_components: SNR v2 components

        Returns:
            Observation value in [0, 1]
        """
        # Extract components
        signal = snr_components.signal_strength
        diversity = snr_components.diversity
        grounding = snr_components.grounding

        # Weighted combination
        w_s = self.config.snr_signal_weight
        w_d = self.config.snr_diversity_weight
        w_g = self.config.snr_grounding_weight

        observation = w_s * signal + w_d * diversity + w_g * grounding

        # Normalize to [0, 1]
        observation = max(0.0, min(1.0, observation))

        return observation

    def observe_snr(
        self,
        snr_components: "SNRComponentsV2",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NTUState:
        """
        Process SNR components through NTU.

        Args:
            snr_components: SNR v2 calculation result
            metadata: Optional context

        Returns:
            Updated NTU state
        """
        obs_value = self.snr_to_observation(snr_components)

        # Enrich metadata with SNR details
        obs_metadata = metadata or {}
        obs_metadata.update({
            "source": "snr_v2",
            "snr": snr_components.snr,
            "quality_tier": snr_components.quality_tier,
        })

        # Track history
        self._snr_history.append(snr_components.snr)
        if len(self._snr_history) > self.config.max_memory_observations:
            self._snr_history = self._snr_history[-self.config.max_memory_observations:]

        state = self.ntu.observe(obs_value, obs_metadata)

        logger.debug(
            f"SNR → NTU: snr={snr_components.snr:.3f} → obs={obs_value:.3f}, "
            f"belief={state.belief:.3f}"
        )

        return state

    def get_ntu_confidence(self) -> float:
        """
        Get NTU belief as a confidence score for SNR operations.

        This can be used to modulate SNR thresholds or gate decisions.

        Returns:
            NTU belief in [0, 1]
        """
        return self.ntu.state.belief

    def modulate_ihsan_threshold(self, base_threshold: float = 0.95) -> float:
        """
        Compute dynamic Ihsan threshold based on NTU state.

        When NTU detects high-quality temporal patterns (high belief),
        we can be more confident in results. When entropy is high,
        we may want to be more conservative.

        Args:
            base_threshold: Base Ihsan threshold

        Returns:
            Modulated threshold
        """
        state = self.ntu.state

        # If NTU has high confidence and low entropy, we can trust results more
        # If NTU has low confidence or high entropy, be more conservative
        confidence_factor = state.belief * (1.0 - state.entropy)

        # Small adjustment range: ±0.02
        adjustment = 0.02 * (confidence_factor - 0.5)

        return max(0.90, min(0.99, base_threshold - adjustment))

    def get_temporal_quality_score(self) -> float:
        """
        Get temporal quality score from NTU potential.

        Potential measures predictive capacity - high potential indicates
        stable, predictable quality over time.

        Returns:
            Temporal quality score in [0, 1]
        """
        return self.ntu.state.potential

    def reset(self) -> None:
        """Reset adapter state."""
        self.ntu.reset()
        self._snr_history.clear()


class NTUMemoryAdapter:
    """
    Adapter connecting NTU to Living Memory system.

    Provides bidirectional mapping:
    - Memory → NTU: Convert memory entries to observations
    - NTU → Memory: Use NTU state to inform memory consolidation

    This implements the reduction:
    - NTU.memory (deque) ↔ LivingMemoryCore episodic memory
    """

    def __init__(self, config: Optional[NTUBridgeConfig] = None):
        """Initialize memory adapter."""
        self.config = config or NTUBridgeConfig()
        self.ntu = NTU(self.config.ntu_config or NTUConfig())

    def memory_entry_to_observation(
        self,
        entry: "MemoryEntry",
        relevance_score: float = 0.5,
    ) -> float:
        """
        Convert a memory entry to NTU observation value.

        Args:
            entry: Memory entry from LivingMemoryCore
            relevance_score: Relevance to current context

        Returns:
            Observation value in [0, 1]
        """
        # Base value from relevance
        base_value = relevance_score

        # Modulate by memory type priority
        type_weights = {
            "episodic": 0.8,
            "semantic": 1.0,
            "procedural": 0.9,
            "working": 0.7,
            "prospective": 0.6,
        }

        # Get weight (default 0.8 if unknown type)
        memory_type = getattr(entry, 'memory_type', 'episodic')
        type_weight = type_weights.get(memory_type, 0.8)

        observation = base_value * type_weight

        return max(0.0, min(1.0, observation))

    def observe_memory_entries(
        self,
        entries: List["MemoryEntry"],
        relevance_scores: Optional[List[float]] = None,
    ) -> NTUState:
        """
        Process multiple memory entries through NTU.

        Args:
            entries: Memory entries to process
            relevance_scores: Relevance scores for each entry

        Returns:
            Updated NTU state
        """
        if relevance_scores is None:
            relevance_scores = [0.5] * len(entries)

        for entry, relevance in zip(entries, relevance_scores):
            obs_value = self.memory_entry_to_observation(entry, relevance)

            metadata = {
                "source": "living_memory",
                "memory_type": getattr(entry, 'memory_type', 'unknown'),
                "relevance": relevance,
            }

            self.ntu.observe(obs_value, metadata)

        return self.ntu.state

    def get_consolidation_priority(self) -> float:
        """
        Get memory consolidation priority based on NTU state.

        High belief + low entropy = important pattern → high consolidation priority
        Low belief + high entropy = noise → low consolidation priority

        Returns:
            Consolidation priority in [0, 1]
        """
        state = self.ntu.state
        priority = state.belief * (1.0 - 0.5 * state.entropy)
        return max(0.0, min(1.0, priority))

    def should_retain_memory(self, threshold: float = 0.5) -> bool:
        """
        Determine if current pattern warrants memory retention.

        Args:
            threshold: Minimum potential for retention

        Returns:
            True if memory should be retained
        """
        return self.ntu.state.potential >= threshold

    def reset(self) -> None:
        """Reset adapter state."""
        self.ntu.reset()


class NTUBridge:
    """
    Unified bridge connecting NTU to all BIZRA components.

    This is the main integration point providing:
    1. SNR quality assessment with temporal consistency
    2. Memory integration with pattern-based consolidation
    3. Ihsan gate modulation based on temporal patterns

    Usage:
        bridge = NTUBridge()

        # Process SNR through NTU
        state = bridge.process_snr(snr_components)

        # Get modulated Ihsan threshold
        threshold = bridge.get_dynamic_ihsan_threshold()

        # Check if quality pattern is stable
        is_stable = bridge.is_quality_stable()
    """

    def __init__(self, config: Optional[NTUBridgeConfig] = None):
        """Initialize unified bridge."""
        self.config = config or NTUBridgeConfig()

        # Initialize sub-adapters
        self.snr_adapter = NTUSNRAdapter(self.config)
        self.memory_adapter = NTUMemoryAdapter(self.config)

        # Shared NTU for unified state
        self._unified_ntu = NTU(self.config.ntu_config or NTUConfig())

    def process_snr(
        self,
        snr_components: "SNRComponentsV2",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NTUState:
        """
        Process SNR components through unified NTU.

        Args:
            snr_components: SNR calculation result
            metadata: Optional context

        Returns:
            Updated NTU state
        """
        obs_value = self.snr_adapter.snr_to_observation(snr_components)

        obs_metadata = metadata or {}
        obs_metadata["source"] = "snr_v2"
        obs_metadata["snr"] = snr_components.snr

        return self._unified_ntu.observe(obs_value, obs_metadata)

    def process_memory(
        self,
        entries: List["MemoryEntry"],
        relevance_scores: Optional[List[float]] = None,
    ) -> NTUState:
        """
        Process memory entries through unified NTU.

        Args:
            entries: Memory entries to process
            relevance_scores: Relevance scores

        Returns:
            Updated NTU state
        """
        if relevance_scores is None:
            relevance_scores = [0.5] * len(entries)

        for entry, relevance in zip(entries, relevance_scores):
            obs_value = self.memory_adapter.memory_entry_to_observation(entry, relevance)

            metadata = {
                "source": "living_memory",
                "memory_type": getattr(entry, 'memory_type', 'unknown'),
            }

            self._unified_ntu.observe(obs_value, metadata)

        return self._unified_ntu.state

    def get_state(self) -> NTUState:
        """Get current unified NTU state."""
        return self._unified_ntu.state

    def get_dynamic_ihsan_threshold(self, base: float = 0.95) -> float:
        """
        Get dynamically modulated Ihsan threshold.

        Args:
            base: Base threshold

        Returns:
            Modulated threshold
        """
        state = self._unified_ntu.state
        confidence_factor = state.belief * (1.0 - state.entropy)
        adjustment = 0.02 * (confidence_factor - 0.5)
        return max(0.90, min(0.99, base - adjustment))

    def is_quality_stable(self, threshold: float = 0.7) -> bool:
        """
        Check if quality pattern is temporally stable.

        Stability = high potential (predictable pattern)

        Args:
            threshold: Minimum potential for stability

        Returns:
            True if quality is stable
        """
        return self._unified_ntu.state.potential >= threshold

    def get_quality_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive quality metrics from NTU.

        Returns:
            Dict with belief, entropy, potential, and derived metrics
        """
        state = self._unified_ntu.state
        return {
            "belief": state.belief,
            "entropy": state.entropy,
            "potential": state.potential,
            "ihsan_achieved": state.ihsan_achieved,
            "dynamic_threshold": self.get_dynamic_ihsan_threshold(),
            "quality_stable": self.is_quality_stable(),
            "iterations": state.iteration,
        }

    def reset(self) -> None:
        """Reset all adapters and unified NTU."""
        self._unified_ntu.reset()
        self.snr_adapter.reset()
        self.memory_adapter.reset()
