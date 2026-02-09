"""
SARE-UERS Bridge — Integration Layer

Connects the Sovereign Autonomous Reasoning Engine (SARE) with
the Universal Entropy Reduction Singularity (UERS) framework.

Giants Protocol Integration:
- Shannon → Surface Vector
- Lamport → Structural Vector
- Vaswani → Contextual Vector
- Besta → Hypothetical Vector
- Anthropic → Contextual Vector
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import SARE components
from core.autonomous import SARE_VERSION
from core.autonomous.engine import ReasoningResult, SovereignReasoningEngine
from core.autonomous.giants import GiantsProtocol
from core.autonomous.loop import LoopExecution
from core.autonomous.nodes import ReasoningGraph
from core.uers import UERS_GIANTS_MAPPING
from core.uers.convergence import ConvergenceLoop, ConvergenceResult
from core.uers.entropy import EntropyCalculator, EntropyMeasurement, ManifoldState
from core.uers.impact import ImpactOracle, ImpactType
from core.uers.vectors import VectorType

logger = logging.getLogger(__name__)


@dataclass
class UnifiedReasoningResult:
    """Combined SARE + UERS result."""

    # SARE metrics
    sare_result: ReasoningResult
    snr_score: float
    ihsan_score: float

    # UERS metrics
    entropy_state: ManifoldState
    total_delta_e: float
    convergence_achieved: bool

    # Combined metrics
    quality_score: float
    singularity_distance: float

    # Metadata
    giants_invoked: List[str] = field(default_factory=list)
    vectors_resolved: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sare": self.sare_result.to_dict(),
            "entropy": self.entropy_state.to_dict(),
            "total_delta_e": self.total_delta_e,
            "convergence_achieved": self.convergence_achieved,
            "quality_score": self.quality_score,
            "singularity_distance": self.singularity_distance,
            "giants_invoked": self.giants_invoked,
            "vectors_resolved": self.vectors_resolved,
        }


class SAREUERSBridge:
    """
    Bridge between SARE and UERS frameworks.

    Enables unified reasoning that combines:
    - SARE's Graph-of-Thoughts and Giants Protocol
    - UERS's 5-Dimensional Manifold and Convergence Loop

    The bridge translates between the two frameworks, allowing
    each to enhance the other's capabilities.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        max_loops: int = 3,
        max_convergence_iterations: int = 50,
    ):
        # SARE components
        self._sare_engine = SovereignReasoningEngine(
            llm_fn=llm_fn,
            max_loops=max_loops,
        )
        self._giants = GiantsProtocol()

        # UERS components
        self._convergence = ConvergenceLoop(
            max_iterations=max_convergence_iterations,
        )
        self._oracle = ImpactOracle()
        self._entropy_calc = EntropyCalculator()

        # Integration state
        self._unified_results: List[UnifiedReasoningResult] = []

    # =========================================================================
    # GIANT → VECTOR MAPPING
    # =========================================================================

    def _giant_to_vector(self, giant: str) -> VectorType:
        """Map a giant to its primary vector."""
        mapping = UERS_GIANTS_MAPPING.get(giant, "contextual")
        return VectorType(mapping)

    def _vector_to_giants(self, vector: VectorType) -> List[str]:
        """Get all giants that contribute to a vector."""
        return [
            giant for giant, vec in UERS_GIANTS_MAPPING.items() if vec == vector.value
        ]

    # =========================================================================
    # ENTROPY TRANSLATION
    # =========================================================================

    def snr_to_entropy(self, snr_score: float) -> float:
        """Convert SNR score to entropy (inverse relationship)."""
        return 1.0 - snr_score

    def entropy_to_snr(self, entropy: float) -> float:
        """Convert entropy to SNR score."""
        return 1.0 - entropy

    def graph_to_structural_entropy(self, graph: ReasoningGraph) -> EntropyMeasurement:
        """Convert SARE reasoning graph to structural entropy."""
        stats = graph.get_stats()

        return self._entropy_calc.structural_entropy(
            nodes=stats["total_nodes"],
            edges=len([n for n in graph._nodes.values() if n.children]),
            components=stats.get("root_nodes", 1),
            max_depth=(
                max(n.depth for n in graph._nodes.values()) if graph._nodes else 0
            ),
        )

    def loop_to_behavioral_entropy(
        self, execution: LoopExecution
    ) -> EntropyMeasurement:
        """Convert SARE loop execution to behavioral entropy."""
        events = [p.phase.value for p in execution.phases]
        return self._entropy_calc.trace_entropy(events)

    # =========================================================================
    # UNIFIED REASONING
    # =========================================================================

    async def unified_reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> UnifiedReasoningResult:
        """
        Execute unified reasoning combining SARE and UERS.

        1. SARE performs Graph-of-Thoughts reasoning
        2. UERS measures entropy across all vectors
        3. Results are synthesized into unified output
        """
        context = context or {}

        # Phase 1: SARE Reasoning
        logger.info(f"Starting SARE reasoning: {query[:50]}...")
        sare_result = await self._sare_engine.reason(query, context)

        # Phase 2: Extract entropy measurements from SARE result
        graph = self._sare_engine._loop.get_graph()
        execution = sare_result.loop_execution

        # Surface entropy (from response text)
        surface_entropy = self._entropy_calc.text_entropy(sare_result.response)

        # Structural entropy (from reasoning graph)
        if graph:
            structural_entropy = self.graph_to_structural_entropy(graph)
        else:
            structural_entropy = EntropyMeasurement("structural", 0.5, 0.5)

        # Behavioral entropy (from loop execution)
        if execution:
            behavioral_entropy = self.loop_to_behavioral_entropy(execution)
        else:
            behavioral_entropy = EntropyMeasurement("behavioral", 0.5, 0.5)

        # Hypothetical entropy (from path coverage)
        if graph:
            stats = graph.get_stats()
            hypothetical_entropy = self._entropy_calc.path_entropy(
                explored_paths=stats.get("total_nodes", 1),
                total_paths=stats.get("total_nodes", 1) * 2,  # Estimate
                feasible_paths=stats.get("active_nodes", 1),
            )
        else:
            hypothetical_entropy = EntropyMeasurement("hypothetical", 0.5, 0.5)

        # Contextual entropy (from SNR/Ihsan scores)
        contextual_entropy = self._entropy_calc.snr_to_entropy(sare_result.snr_score)

        # Construct manifold state
        manifold_state = ManifoldState(
            surface=surface_entropy,
            structural=structural_entropy,
            behavioral=behavioral_entropy,
            hypothetical=hypothetical_entropy,
            contextual=contextual_entropy,
        )

        # Phase 3: Calculate convergence metrics
        total_entropy = manifold_state.total_entropy
        singularity_distance = manifold_state.average_entropy

        # Delta E is inverse of remaining entropy
        total_delta_e = 5.0 - total_entropy  # Max entropy is 5 (1 per vector)

        # Convergence achieved if below threshold
        convergence_achieved = singularity_distance < 0.1

        # Quality score (geometric mean of SNR and 1-entropy)
        import math

        quality_score = math.sqrt(sare_result.snr_score * (1 - singularity_distance))

        # Collect giants invoked
        giants_invoked = sare_result.giants_invoked or []

        # Collect resolved vectors
        vectors_resolved = [
            v
            for v in [
                "surface",
                "structural",
                "behavioral",
                "hypothetical",
                "contextual",
            ]
            if manifold_state.__getattribute__(v).normalized < 0.3
        ]

        result = UnifiedReasoningResult(
            sare_result=sare_result,
            snr_score=sare_result.snr_score,
            ihsan_score=sare_result.ihsan_score,
            entropy_state=manifold_state,
            total_delta_e=total_delta_e,
            convergence_achieved=convergence_achieved,
            quality_score=quality_score,
            singularity_distance=singularity_distance,
            giants_invoked=giants_invoked,
            vectors_resolved=vectors_resolved,
        )

        self._unified_results.append(result)

        logger.info(
            f"Unified reasoning complete: "
            f"quality={quality_score:.3f}, "
            f"ΔE={total_delta_e:.3f}, "
            f"singularity_dist={singularity_distance:.3f}"
        )

        return result

    # =========================================================================
    # CONVERGENCE WITH SARE
    # =========================================================================

    async def converge_with_sare(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ConvergenceResult, List[UnifiedReasoningResult]]:
        """
        Run convergence loop using SARE for hypothesis generation.

        Iteratively refines understanding until singularity is achieved.
        """
        context = context or {}
        results = []

        # Custom hypothesis generator using SARE
        async def sare_hypothesis_generator(manifold):
            # Use SARE to reason about the current state
            state_description = f"""
            Current entropy state:
            - Surface: {manifold._vectors[VectorType.SURFACE].entropy.normalized:.3f}
            - Structural: {manifold._vectors[VectorType.STRUCTURAL].entropy.normalized:.3f}
            - Behavioral: {manifold._vectors[VectorType.BEHAVIORAL].entropy.normalized:.3f}
            - Hypothetical: {manifold._vectors[VectorType.HYPOTHETICAL].entropy.normalized:.3f}
            - Contextual: {manifold._vectors[VectorType.CONTEXTUAL].entropy.normalized:.3f}

            Original query: {query}

            Generate hypotheses to reduce entropy.
            """

            sare_result = await self._sare_engine.reason(state_description, context)

            # Convert SARE reasoning to UERS hypotheses
            from core.uers.convergence import Hypothesis

            hypotheses: list = []

            suggestions = manifold.suggest_probes()
            for source, target, operation in suggestions:
                hypotheses.append(
                    Hypothesis(
                        id=f"hyp_sare_{len(hypotheses)}",
                        description=f"SARE-guided: {operation}",
                        source_vector=source,
                        target_vector=target,
                        confidence=sare_result.snr_score,
                        predicted_delta_e=sare_result.quality_score * 0.1,
                        probe_operations=[operation],
                    )
                )

            return hypotheses

        # Run unified reasoning for initial state
        initial_result = await self.unified_reason(query, context)
        results.append(initial_result)

        # Configure convergence with SARE integration
        # Note: In production, we'd integrate more deeply
        convergence_result = await self._convergence.converge(
            contextual_data={
                "text": query,
                "intent": initial_result.snr_score,
                "alignment": initial_result.ihsan_score,
            }
        )

        return convergence_result, results

    # =========================================================================
    # IMPACT VERIFICATION
    # =========================================================================

    async def verify_impact(
        self,
        agent_id: str,
        result: UnifiedReasoningResult,
    ) -> Dict[str, Any]:
        """
        Verify the impact of a unified reasoning result.

        Submits to the Impact Oracle for Proof-of-Impact verification.
        """
        # Submit claim
        claim = self._oracle.submit_claim(
            agent_id=agent_id,
            impact_type=ImpactType.ENTROPY_REDUCTION,
            description=f"Unified reasoning: {result.sare_result.query[:100]}",
            claimed_delta_e=result.total_delta_e,
            evidence={
                "snr_score": result.snr_score,
                "ihsan_score": result.ihsan_score,
                "quality_score": result.quality_score,
                "convergence_achieved": result.convergence_achieved,
                "giants_invoked": result.giants_invoked,
                "vectors_resolved": result.vectors_resolved,
            },
        )

        # Verify claim
        verdict = await self._oracle.verify_claim(
            claim=claim,
            before_state=None,  # Would need to track before/after
            after_state=result.entropy_state,
        )

        return {
            "claim": claim.to_dict(),
            "verdict": verdict.to_dict(),
            "rewarded": verdict.is_rewarded,
        }

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        if not self._unified_results:
            return {
                "total_results": 0,
                "sare_version": SARE_VERSION,
            }

        return {
            "total_results": len(self._unified_results),
            "avg_quality_score": sum(r.quality_score for r in self._unified_results)
            / len(self._unified_results),
            "avg_delta_e": sum(r.total_delta_e for r in self._unified_results)
            / len(self._unified_results),
            "convergence_rate": sum(
                1 for r in self._unified_results if r.convergence_achieved
            )
            / len(self._unified_results),
            "sare_stats": self._sare_engine.get_stats(),
            "convergence_stats": self._convergence.get_stats(),
            "oracle_stats": self._oracle.get_stats(),
            "sare_version": SARE_VERSION,
        }

    def get_recent_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent unified results."""
        return [r.to_dict() for r in self._unified_results[-limit:]]
