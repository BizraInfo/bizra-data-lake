"""
Analytical Manifold — 5-Dimensional Vector Space

The complete analytical framework for UERS:
- Surface: Static byte-level analysis
- Structural: Graph topology analysis
- Behavioral: Dynamic execution analysis
- Hypothetical: Symbolic state space
- Contextual: Semantic intent analysis

Each vector provides a unique perspective on the target system.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.uers import ANALYTICAL_VECTORS
from core.uers.entropy import EntropyCalculator, EntropyMeasurement, ManifoldState

logger = logging.getLogger(__name__)


class VectorType(str, Enum):
    """The 5 analytical vectors."""

    SURFACE = "surface"
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    HYPOTHETICAL = "hypothetical"
    CONTEXTUAL = "contextual"


class ProbeResult(str, Enum):
    """Result of a cross-vector probe."""

    SUCCESS = "success"  # Information gained
    FAILURE = "failure"  # No information gained
    BLOCKED = "blocked"  # Probe blocked (e.g., anti-debug)
    PENDING = "pending"  # Probe in progress


@dataclass
class VectorState:
    """State of a single analytical vector."""

    vector_type: VectorType
    entropy: EntropyMeasurement
    artifacts: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_resolved(self) -> bool:
        """Vector is resolved when entropy is low."""
        return self.entropy.normalized < 0.1

    @property
    def needs_attention(self) -> bool:
        """Vector needs work when entropy is high."""
        return self.entropy.normalized > 0.7

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.vector_type.value,
            "entropy": self.entropy.to_dict(),
            "artifacts_count": len(self.artifacts),
            "confidence": self.confidence,
            "is_resolved": self.is_resolved,
            "needs_attention": self.needs_attention,
        }


@dataclass
class Probe:
    """A cross-dimensional probe operation."""

    id: str
    source_vector: VectorType
    target_vector: VectorType
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: ProbeResult = ProbeResult.PENDING
    delta_e: float = 0.0
    output: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source_vector.value,
            "target": self.target_vector.value,
            "operation": self.operation,
            "result": self.result.value,
            "delta_e": self.delta_e,
            "timestamp": self.timestamp.isoformat(),
        }


class AnalyticalManifold:
    """
    The 5-Dimensional Analytical Manifold.

    Manages the state of all vectors and coordinates
    cross-dimensional probing for entropy reduction.
    """

    def __init__(self):
        self._entropy_calc = EntropyCalculator()
        self._vectors: Dict[VectorType, VectorState] = {}
        self._probes: List[Probe] = []
        self._probe_count = 0

        # Initialize empty vectors
        for vtype in VectorType:
            self._vectors[vtype] = VectorState(
                vector_type=vtype,
                entropy=EntropyMeasurement(vtype.value, 1.0, 1.0),
            )

        # Probe registry (source -> target -> operation)
        self._probe_registry: Dict[Tuple[VectorType, VectorType], List[str]] = {}
        self._register_standard_probes()

    def _register_standard_probes(self) -> None:
        """Register standard cross-vector probe operations."""
        # Surface → Structural
        self._probe_registry[(VectorType.SURFACE, VectorType.STRUCTURAL)] = [
            "extract_cfg_from_bytes",
            "identify_basic_blocks",
            "map_function_boundaries",
        ]

        # Surface → Behavioral
        self._probe_registry[(VectorType.SURFACE, VectorType.BEHAVIORAL)] = [
            "extract_strings_for_breakpoints",
            "identify_api_imports",
            "locate_entry_points",
        ]

        # Structural → Behavioral
        self._probe_registry[(VectorType.STRUCTURAL, VectorType.BEHAVIORAL)] = [
            "trace_cfg_paths",
            "instrument_basic_blocks",
            "monitor_function_calls",
        ]

        # Behavioral → Hypothetical
        self._probe_registry[(VectorType.BEHAVIORAL, VectorType.HYPOTHETICAL)] = [
            "seed_from_traces",
            "extract_input_patterns",
            "identify_magic_checks",
        ]

        # Hypothetical → Behavioral (Hybrid Analysis)
        self._probe_registry[(VectorType.HYPOTHETICAL, VectorType.BEHAVIORAL)] = [
            "solve_constraint_for_input",
            "generate_test_case",
            "bypass_magic_number",
        ]

        # Contextual → All
        for target in VectorType:
            if target != VectorType.CONTEXTUAL:
                self._probe_registry[(VectorType.CONTEXTUAL, target)] = [
                    f"semantic_filter_{target.value}",
                    f"intent_guide_{target.value}",
                ]

    # =========================================================================
    # VECTOR UPDATES
    # =========================================================================

    def update_surface(
        self,
        data: bytes,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> VectorState:
        """Update surface vector with new binary data."""
        entropy = self._entropy_calc.shannon_entropy(data)

        state = VectorState(
            vector_type=VectorType.SURFACE,
            entropy=entropy,
            artifacts=artifacts or {},
            confidence=1.0 - entropy.normalized,
        )

        self._vectors[VectorType.SURFACE] = state
        logger.info(f"Surface vector updated: entropy={entropy.normalized:.3f}")
        return state

    def update_structural(
        self,
        nodes: int,
        edges: int,
        components: int = 1,
        cfg_data: Optional[Dict[str, Any]] = None,
    ) -> VectorState:
        """Update structural vector with graph data."""
        entropy = self._entropy_calc.structural_entropy(nodes, edges, components)

        state = VectorState(
            vector_type=VectorType.STRUCTURAL,
            entropy=entropy,
            artifacts=cfg_data or {"nodes": nodes, "edges": edges},
            confidence=1.0 - entropy.normalized,
        )

        self._vectors[VectorType.STRUCTURAL] = state
        logger.info(f"Structural vector updated: entropy={entropy.normalized:.3f}")
        return state

    def update_behavioral(
        self,
        events: List[str],
        traces: Optional[Dict[str, Any]] = None,
    ) -> VectorState:
        """Update behavioral vector with execution trace."""
        entropy = self._entropy_calc.trace_entropy(events)

        state = VectorState(
            vector_type=VectorType.BEHAVIORAL,
            entropy=entropy,
            artifacts=traces or {"events": events[:100]},  # Limit stored events
            confidence=1.0 - entropy.normalized,
        )

        self._vectors[VectorType.BEHAVIORAL] = state
        logger.info(f"Behavioral vector updated: entropy={entropy.normalized:.3f}")
        return state

    def update_hypothetical(
        self,
        explored_paths: int,
        total_paths: int,
        feasible_paths: int,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> VectorState:
        """Update hypothetical vector with symbolic execution state."""
        entropy = self._entropy_calc.path_entropy(
            explored_paths, total_paths, feasible_paths
        )

        state = VectorState(
            vector_type=VectorType.HYPOTHETICAL,
            entropy=entropy,
            artifacts=constraints
            or {
                "explored": explored_paths,
                "total": total_paths,
                "feasible": feasible_paths,
            },
            confidence=1.0 - entropy.normalized,
        )

        self._vectors[VectorType.HYPOTHETICAL] = state
        logger.info(f"Hypothetical vector updated: entropy={entropy.normalized:.3f}")
        return state

    def update_contextual(
        self,
        text: str,
        intent_score: float = 0.5,
        alignment_score: float = 0.5,
        context: Optional[Dict[str, Any]] = None,
    ) -> VectorState:
        """Update contextual vector with semantic data."""
        entropy = self._entropy_calc.contextual_entropy(
            text, intent_score, alignment_score
        )

        state = VectorState(
            vector_type=VectorType.CONTEXTUAL,
            entropy=entropy,
            artifacts=context
            or {
                "intent": intent_score,
                "alignment": alignment_score,
            },
            confidence=1.0 - entropy.normalized,
        )

        self._vectors[VectorType.CONTEXTUAL] = state
        logger.info(f"Contextual vector updated: entropy={entropy.normalized:.3f}")
        return state

    # =========================================================================
    # PROBING OPERATIONS
    # =========================================================================

    def create_probe(
        self,
        source: VectorType,
        target: VectorType,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Probe:
        """Create a cross-vector probe."""
        self._probe_count += 1
        probe = Probe(
            id=f"probe_{self._probe_count:05d}",
            source_vector=source,
            target_vector=target,
            operation=operation,
            parameters=parameters or {},
        )
        self._probes.append(probe)
        return probe

    def execute_probe(
        self,
        probe: Probe,
        probe_fn: Optional[Callable[[Probe], Dict[str, Any]]] = None,
    ) -> Probe:
        """
        Execute a probe and measure entropy change.

        If probe_fn is provided, it should return the probe output.
        Otherwise, a simulated probe is performed.
        """
        # Get before state
        before_entropy = self._vectors[probe.target_vector].entropy.normalized

        # Execute probe
        if probe_fn:
            try:
                probe.output = probe_fn(probe)
                probe.result = ProbeResult.SUCCESS
            except Exception as e:
                probe.output = {"error": str(e)}
                probe.result = ProbeResult.FAILURE
        else:
            # Simulated probe (for testing)
            probe.output = {"simulated": True}
            probe.result = ProbeResult.SUCCESS

        # Get after state (in real implementation, this would update the vector)
        after_entropy = self._vectors[probe.target_vector].entropy.normalized

        # Calculate ΔE
        probe.delta_e = before_entropy - after_entropy

        logger.info(
            f"Probe {probe.id} [{probe.source_vector.value}→{probe.target_vector.value}]: "
            f"ΔE={probe.delta_e:.4f}"
        )

        return probe

    def suggest_probes(self) -> List[Tuple[VectorType, VectorType, str]]:
        """
        Suggest the most promising probes based on current state.

        Prioritizes probes that are likely to reduce the most entropy.
        """
        suggestions = []

        # Find the highest-entropy vector that isn't contextual
        sorted_vectors = sorted(
            [(v.entropy.normalized, v.vector_type) for v in self._vectors.values()],
            reverse=True,
        )

        for entropy, target in sorted_vectors:
            if entropy < 0.3:
                continue  # Already low entropy

            # Find potential source vectors
            for source in VectorType:
                if source == target:
                    continue

                key = (source, target)
                if key in self._probe_registry:
                    for op in self._probe_registry[key][:1]:  # Top operation
                        suggestions.append((source, target, op))

            if len(suggestions) >= 3:
                break

        return suggestions

    # =========================================================================
    # MANIFOLD STATE
    # =========================================================================

    def get_manifold_state(self) -> ManifoldState:
        """Get the current state of the entire manifold."""
        return ManifoldState(
            surface=self._vectors[VectorType.SURFACE].entropy,
            structural=self._vectors[VectorType.STRUCTURAL].entropy,
            behavioral=self._vectors[VectorType.BEHAVIORAL].entropy,
            hypothetical=self._vectors[VectorType.HYPOTHETICAL].entropy,
            contextual=self._vectors[VectorType.CONTEXTUAL].entropy,
        )

    def get_total_entropy(self) -> float:
        """Get total entropy across all vectors."""
        return sum(v.entropy.normalized for v in self._vectors.values())

    def get_average_entropy(self) -> float:
        """Get average entropy across all vectors."""
        return self.get_total_entropy() / len(self._vectors)

    def get_entropy_vector(self) -> List[float]:
        """Get the 5D entropy vector."""
        return [
            self._vectors[VectorType.SURFACE].entropy.normalized,
            self._vectors[VectorType.STRUCTURAL].entropy.normalized,
            self._vectors[VectorType.BEHAVIORAL].entropy.normalized,
            self._vectors[VectorType.HYPOTHETICAL].entropy.normalized,
            self._vectors[VectorType.CONTEXTUAL].entropy.normalized,
        ]

    def is_converged(self, threshold: float = 0.1) -> bool:
        """Check if manifold has converged (all vectors below threshold)."""
        return all(v.entropy.normalized < threshold for v in self._vectors.values())

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get manifold statistics."""
        return {
            "total_entropy": self.get_total_entropy(),
            "average_entropy": self.get_average_entropy(),
            "entropy_vector": self.get_entropy_vector(),
            "is_converged": self.is_converged(),
            "vectors": {
                v.vector_type.value: v.to_dict() for v in self._vectors.values()
            },
            "probe_count": len(self._probes),
            "successful_probes": sum(
                1 for p in self._probes if p.result == ProbeResult.SUCCESS
            ),
            "total_delta_e": sum(p.delta_e for p in self._probes),
        }

    def get_probe_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent probe history."""
        return [p.to_dict() for p in self._probes[-limit:]]

    def get_vector_info(self, vector_type: VectorType) -> Dict[str, Any]:
        """Get detailed information about a specific vector."""
        state = self._vectors[vector_type]
        config = ANALYTICAL_VECTORS.get(vector_type.value, {})

        return {
            "state": state.to_dict(),
            "config": config,
            "available_probes": [
                {"target": k[1].value, "operations": v}
                for k, v in self._probe_registry.items()
                if k[0] == vector_type
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete manifold state."""
        return {
            "stats": self.get_stats(),
            "recent_probes": self.get_probe_history(5),
        }
