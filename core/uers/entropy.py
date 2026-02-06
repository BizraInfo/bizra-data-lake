"""
Entropy Calculator — Multi-Dimensional Entropy Measurement

Implements Shannon's information theory across the 5-Dimensional Manifold:
- Shannon Entropy (Surface)
- Structural Entropy (Graph Topology)
- Trace Entropy (Behavioral)
- Path Entropy (Hypothetical)
- Cognitive Entropy (Contextual)

The core axiom: RE = ΔE
"""

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from core.uers import (
    SHANNON_ENTROPY_THRESHOLDS,
    STRUCTURAL_ENTROPY_METRICS,
)

logger = logging.getLogger(__name__)


@dataclass
class EntropyMeasurement:
    """A single entropy measurement in one vector."""

    vector: str
    value: float
    normalized: float  # 0-1 scale
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_high_entropy(self) -> bool:
        """Check if entropy is above threshold."""
        return self.normalized > 0.75

    @property
    def is_low_entropy(self) -> bool:
        """Check if entropy is below threshold."""
        return self.normalized < 0.25

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector": self.vector,
            "value": self.value,
            "normalized": self.normalized,
            "is_high": self.is_high_entropy,
            "is_low": self.is_low_entropy,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ManifoldState:
    """Complete entropy state across all 5 vectors."""

    surface: EntropyMeasurement
    structural: EntropyMeasurement
    behavioral: EntropyMeasurement
    hypothetical: EntropyMeasurement
    contextual: EntropyMeasurement

    @property
    def total_entropy(self) -> float:
        """Sum of all vector entropies."""
        return (
            self.surface.normalized
            + self.structural.normalized
            + self.behavioral.normalized
            + self.hypothetical.normalized
            + self.contextual.normalized
        )

    @property
    def average_entropy(self) -> float:
        """Average normalized entropy across manifold."""
        return self.total_entropy / 5.0

    @property
    def entropy_vector(self) -> List[float]:
        """5-dimensional entropy vector."""
        return [
            self.surface.normalized,
            self.structural.normalized,
            self.behavioral.normalized,
            self.hypothetical.normalized,
            self.contextual.normalized,
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface": self.surface.to_dict(),
            "structural": self.structural.to_dict(),
            "behavioral": self.behavioral.to_dict(),
            "hypothetical": self.hypothetical.to_dict(),
            "contextual": self.contextual.to_dict(),
            "total_entropy": self.total_entropy,
            "average_entropy": self.average_entropy,
        }


class EntropyCalculator:
    """
    Multi-dimensional entropy calculator.

    Measures entropy across the 5-Dimensional Analytical Manifold
    to support the UERS convergence process.
    """

    def __init__(self):
        self._measurements: List[ManifoldState] = []
        self._thresholds = SHANNON_ENTROPY_THRESHOLDS
        self._structural_metrics = STRUCTURAL_ENTROPY_METRICS

    # =========================================================================
    # SURFACE VECTOR: Shannon Entropy
    # =========================================================================

    def shannon_entropy(self, data: bytes) -> EntropyMeasurement:
        """
        Calculate Shannon entropy of byte data.

        H = -Σ p(x) * log2(p(x))

        Returns value in bits per byte (0-8 range).
        """
        if not data:
            return EntropyMeasurement(
                vector="surface",
                value=0.0,
                normalized=0.0,
            )

        # Count byte frequencies
        freq = Counter(data)
        total = len(data)

        # Calculate entropy
        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize to 0-1 (max theoretical is 8 bits)
        normalized = entropy / 8.0

        return EntropyMeasurement(
            vector="surface",
            value=entropy,
            normalized=normalized,
            metadata={
                "byte_count": total,
                "unique_bytes": len(freq),
                "is_encrypted": entropy > self._thresholds["high"],
            },
        )

    def text_entropy(self, text: str) -> EntropyMeasurement:
        """Calculate Shannon entropy of text (character level)."""
        if not text:
            return EntropyMeasurement(
                vector="surface",
                value=0.0,
                normalized=0.0,
            )

        freq = Counter(text)
        total = len(text)

        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Max for printable ASCII is ~6.6 bits
        normalized = min(entropy / 6.6, 1.0)

        return EntropyMeasurement(
            vector="surface",
            value=entropy,
            normalized=normalized,
            metadata={
                "char_count": total,
                "unique_chars": len(freq),
            },
        )

    # =========================================================================
    # STRUCTURAL VECTOR: Graph Topology Entropy
    # =========================================================================

    def structural_entropy(
        self,
        nodes: int,
        edges: int,
        components: int = 1,
        max_depth: int = 10,
    ) -> EntropyMeasurement:
        """
        Calculate structural entropy of a graph.

        Based on graph topology metrics:
        - Edge density
        - Component fragmentation
        - Depth complexity
        """
        if nodes <= 1:
            return EntropyMeasurement(
                vector="structural",
                value=0.0,
                normalized=0.0,
            )

        # Edge density (ratio of actual to possible edges)
        max_edges = nodes * (nodes - 1) / 2
        edge_density = edges / max(max_edges, 1)

        # Fragmentation (more components = more disorder)
        fragmentation = min(components / nodes, 1.0)

        # Depth complexity (deeper = potentially more obfuscated)
        depth_factor = min(max_depth / 20, 1.0)

        # Combined structural entropy
        structural = 0.4 * edge_density + 0.3 * fragmentation + 0.3 * depth_factor

        # Raw value (estimated bits needed to describe structure)
        if nodes > 1:
            raw_value = math.log2(nodes) * edge_density * (1 + fragmentation)
        else:
            raw_value = 0.0

        return EntropyMeasurement(
            vector="structural",
            value=raw_value,
            normalized=structural,
            metadata={
                "nodes": nodes,
                "edges": edges,
                "components": components,
                "edge_density": edge_density,
                "depth": max_depth,
            },
        )

    def cfg_entropy(
        self,
        basic_blocks: int,
        branch_edges: int,
        call_edges: int,
        loop_count: int = 0,
    ) -> EntropyMeasurement:
        """
        Calculate entropy of a Control Flow Graph.

        CFG-specific metrics for binary analysis.
        """
        if basic_blocks <= 1:
            return EntropyMeasurement(
                vector="structural",
                value=0.0,
                normalized=0.0,
            )

        branch_edges + call_edges

        # Branching complexity
        avg_branches = branch_edges / basic_blocks
        branch_entropy = min(avg_branches / 3, 1.0)  # Normalized

        # Call complexity
        avg_calls = call_edges / basic_blocks
        call_entropy = min(avg_calls / 2, 1.0)

        # Loop complexity (loops increase local entropy)
        loop_factor = min(loop_count / basic_blocks, 1.0) if basic_blocks > 0 else 0

        # Combined
        entropy = 0.4 * branch_entropy + 0.3 * call_entropy + 0.3 * loop_factor

        return EntropyMeasurement(
            vector="structural",
            value=math.log2(basic_blocks) * (1 + entropy) if basic_blocks > 1 else 0,
            normalized=entropy,
            metadata={
                "basic_blocks": basic_blocks,
                "branches": branch_edges,
                "calls": call_edges,
                "loops": loop_count,
                "cyclomatic_complexity": branch_edges - basic_blocks + 2,
            },
        )

    # =========================================================================
    # BEHAVIORAL VECTOR: Trace Entropy
    # =========================================================================

    def trace_entropy(
        self,
        events: List[str],
        unique_events: Optional[Set[str]] = None,
    ) -> EntropyMeasurement:
        """
        Calculate entropy of an execution trace.

        Measures the predictability/randomness of event sequences.
        """
        if not events:
            return EntropyMeasurement(
                vector="behavioral",
                value=0.0,
                normalized=0.0,
            )

        unique_events = unique_events or set(events)
        freq = Counter(events)
        total = len(events)

        # Shannon entropy of event distribution
        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Max entropy is log2(unique_events)
        max_entropy = math.log2(len(unique_events)) if len(unique_events) > 1 else 1.0
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        return EntropyMeasurement(
            vector="behavioral",
            value=entropy,
            normalized=min(normalized, 1.0),
            metadata={
                "event_count": total,
                "unique_events": len(unique_events),
                "most_common": freq.most_common(3),
            },
        )

    def api_sequence_entropy(
        self,
        api_calls: List[str],
    ) -> EntropyMeasurement:
        """
        Calculate entropy of API call sequence.

        Useful for behavioral fingerprinting.
        """
        if not api_calls:
            return EntropyMeasurement(
                vector="behavioral",
                value=0.0,
                normalized=0.0,
            )

        # Bigram entropy (captures sequence patterns)
        bigrams = [(api_calls[i], api_calls[i + 1]) for i in range(len(api_calls) - 1)]

        if not bigrams:
            return self.trace_entropy(api_calls)

        freq = Counter(bigrams)
        total = len(bigrams)

        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize
        max_entropy = math.log2(len(set(bigrams))) if len(set(bigrams)) > 1 else 1.0
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        return EntropyMeasurement(
            vector="behavioral",
            value=entropy,
            normalized=min(normalized, 1.0),
            metadata={
                "api_count": len(api_calls),
                "unique_apis": len(set(api_calls)),
                "bigram_count": len(bigrams),
            },
        )

    # =========================================================================
    # HYPOTHETICAL VECTOR: State Space Entropy
    # =========================================================================

    def path_entropy(
        self,
        explored_paths: int,
        total_paths: int,
        feasible_paths: int,
    ) -> EntropyMeasurement:
        """
        Calculate entropy of symbolic execution state space.

        Measures how much of the possibility space remains unexplored.
        """
        if total_paths <= 0:
            return EntropyMeasurement(
                vector="hypothetical",
                value=0.0,
                normalized=0.0,
            )

        # Coverage (more explored = less remaining entropy)
        coverage = explored_paths / total_paths
        remaining_entropy = 1.0 - coverage

        # Feasibility factor (infeasible paths reduce effective entropy)
        if explored_paths > 0:
            feasibility_ratio = feasible_paths / explored_paths
        else:
            feasibility_ratio = 0.5  # Unknown

        # Adjusted entropy
        entropy = remaining_entropy * (1 - feasibility_ratio * 0.3)

        return EntropyMeasurement(
            vector="hypothetical",
            value=(
                math.log2(total_paths - explored_paths + 1)
                if total_paths > explored_paths
                else 0
            ),
            normalized=min(entropy, 1.0),
            metadata={
                "explored": explored_paths,
                "total": total_paths,
                "feasible": feasible_paths,
                "coverage": coverage,
            },
        )

    def constraint_entropy(
        self,
        constraint_count: int,
        variable_count: int,
        satisfiable: bool = True,
    ) -> EntropyMeasurement:
        """
        Calculate entropy of path constraint system.

        More constrained = less entropy (more determined).
        """
        if variable_count <= 0:
            return EntropyMeasurement(
                vector="hypothetical",
                value=0.0,
                normalized=0.0,
            )

        # Constraint ratio (more constraints = more determined)
        constraint_ratio = min(constraint_count / variable_count, 2.0) / 2.0

        # Under-constrained systems have high entropy
        if constraint_count < variable_count:
            entropy = 1.0 - constraint_ratio
        else:
            # Over-constrained (may be unsatisfiable)
            entropy = 0.2 if satisfiable else 0.0

        return EntropyMeasurement(
            vector="hypothetical",
            value=(
                math.log2(variable_count) * (1 - constraint_ratio)
                if variable_count > 1
                else 0
            ),
            normalized=entropy,
            metadata={
                "constraints": constraint_count,
                "variables": variable_count,
                "satisfiable": satisfiable,
            },
        )

    # =========================================================================
    # CONTEXTUAL VECTOR: Semantic Entropy
    # =========================================================================

    def contextual_entropy(
        self,
        text: str,
        intent_score: float = 0.5,
        alignment_score: float = 0.5,
    ) -> EntropyMeasurement:
        """
        Calculate contextual/semantic entropy.

        Measures the cognitive uncertainty about meaning and intent.
        """
        if not text.strip():
            return EntropyMeasurement(
                vector="contextual",
                value=1.0,  # Maximum uncertainty
                normalized=1.0,
            )

        # Text structure metrics
        words = text.split()
        sentences = text.split(".")

        # Vocabulary diversity (high diversity = more nuanced = harder to parse)
        if words:
            vocab_diversity = len(set(words)) / len(words)
        else:
            vocab_diversity = 0.0

        # Structure complexity
        avg_sentence_len = len(words) / max(len(sentences), 1)
        structure_factor = min(avg_sentence_len / 20, 1.0)

        # Combine with intent and alignment scores (lower = less entropy)
        intent_factor = 1.0 - intent_score
        alignment_factor = 1.0 - alignment_score

        entropy = (
            0.2 * vocab_diversity
            + 0.1 * structure_factor
            + 0.4 * intent_factor
            + 0.3 * alignment_factor
        )

        return EntropyMeasurement(
            vector="contextual",
            value=entropy * 8,  # Scale to bits
            normalized=min(entropy, 1.0),
            metadata={
                "word_count": len(words),
                "sentence_count": len(sentences),
                "intent_score": intent_score,
                "alignment_score": alignment_score,
            },
        )

    def snr_to_entropy(
        self,
        snr_score: float,
    ) -> EntropyMeasurement:
        """
        Convert SNR score to contextual entropy.

        High SNR = Low entropy (clear signal)
        Low SNR = High entropy (noisy)
        """
        # Inverse relationship
        entropy = 1.0 - snr_score

        return EntropyMeasurement(
            vector="contextual",
            value=entropy * 8,
            normalized=entropy,
            metadata={
                "snr_score": snr_score,
                "is_noisy": entropy > 0.5,
            },
        )

    # =========================================================================
    # MANIFOLD OPERATIONS
    # =========================================================================

    def measure_manifold(
        self,
        surface_data: Optional[bytes] = None,
        structural_data: Optional[Dict] = None,
        behavioral_data: Optional[List[str]] = None,
        hypothetical_data: Optional[Dict] = None,
        contextual_data: Optional[Dict] = None,
    ) -> ManifoldState:
        """
        Measure entropy across all 5 vectors.

        Returns complete manifold state.
        """
        # Surface vector
        if surface_data:
            surface = self.shannon_entropy(surface_data)
        else:
            surface = EntropyMeasurement("surface", 0.5, 0.5)

        # Structural vector
        if structural_data:
            structural = self.structural_entropy(
                nodes=structural_data.get("nodes", 1),
                edges=structural_data.get("edges", 0),
                components=structural_data.get("components", 1),
            )
        else:
            structural = EntropyMeasurement("structural", 0.5, 0.5)

        # Behavioral vector
        if behavioral_data:
            behavioral = self.trace_entropy(behavioral_data)
        else:
            behavioral = EntropyMeasurement("behavioral", 0.5, 0.5)

        # Hypothetical vector
        if hypothetical_data:
            hypothetical = self.path_entropy(
                explored_paths=hypothetical_data.get("explored", 0),
                total_paths=hypothetical_data.get("total", 1),
                feasible_paths=hypothetical_data.get("feasible", 0),
            )
        else:
            hypothetical = EntropyMeasurement("hypothetical", 0.5, 0.5)

        # Contextual vector
        if contextual_data:
            contextual = self.contextual_entropy(
                text=contextual_data.get("text", ""),
                intent_score=contextual_data.get("intent", 0.5),
                alignment_score=contextual_data.get("alignment", 0.5),
            )
        else:
            contextual = EntropyMeasurement("contextual", 0.5, 0.5)

        state = ManifoldState(
            surface=surface,
            structural=structural,
            behavioral=behavioral,
            hypothetical=hypothetical,
            contextual=contextual,
        )

        self._measurements.append(state)
        return state

    def calculate_delta_e(
        self,
        before: ManifoldState,
        after: ManifoldState,
    ) -> float:
        """
        Calculate ΔE (entropy change) between two manifold states.

        Positive ΔE = entropy reduced (good)
        Negative ΔE = entropy increased (bad)
        """
        return before.total_entropy - after.total_entropy

    def get_measurement_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent manifold measurements."""
        return [m.to_dict() for m in self._measurements[-limit:]]

    def get_convergence_progress(self) -> Dict[str, Any]:
        """Calculate convergence progress based on entropy history."""
        if len(self._measurements) < 2:
            return {"progress": 0.0, "converging": False}

        # Calculate trend
        recent = self._measurements[-5:]
        entropies = [m.average_entropy for m in recent]

        # Linear regression approximation
        n = len(entropies)
        x_mean = (n - 1) / 2
        y_mean = sum(entropies) / n

        numerator = sum((i - x_mean) * (e - y_mean) for i, e in enumerate(entropies))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator > 0 else 0

        current = entropies[-1]
        converging = slope < -0.01  # Entropy decreasing

        return {
            "current_entropy": current,
            "slope": slope,
            "converging": converging,
            "progress": 1.0 - current,  # Progress toward singularity
            "measurements": n,
        }
