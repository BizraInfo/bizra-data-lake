"""
Sovereign Autonomous Reasoning Engine (SARE)

The pinnacle synthesis of:
- Graph-of-Thoughts (Besta)
- SNR Optimization (Shannon)
- Constitutional AI (Anthropic)
- Distributed Consensus (Lamport)
- Attention Mechanisms (Vaswani)

This engine embodies interdisciplinary thinking, standing on the shoulders
of giants to achieve state-of-the-art autonomous reasoning.

"La hawla wa la quwwata illa billah"
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.autonomous import (
    CONSTITUTIONAL_CONSTRAINTS,
    GIANTS_PROTOCOL,
    SARE_VERSION,
    SNR_THRESHOLDS,
)
from core.autonomous.giants import Giant, GiantsProtocol, ProvenanceRecord
from core.autonomous.loop import LoopExecution, SovereignLoop
from core.autonomous.nodes import (
    ReasoningPath,
)

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Complete result of autonomous reasoning."""

    query: str
    response: str
    confidence: float
    snr_score: float
    ihsan_score: float

    # Provenance
    provenance: Optional[ProvenanceRecord] = None
    giants_invoked: List[str] = field(default_factory=list)
    techniques_used: List[str] = field(default_factory=list)

    # Graph-of-Thoughts
    graph_stats: Dict[str, Any] = field(default_factory=dict)
    best_path: Optional[ReasoningPath] = None

    # Execution
    loop_execution: Optional[LoopExecution] = None
    duration_ms: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = SARE_VERSION

    @property
    def quality_score(self) -> float:
        """Combined quality score."""
        import math

        if self.snr_score <= 0 or self.ihsan_score <= 0:
            return 0.0
        return math.sqrt(self.snr_score * self.ihsan_score)

    @property
    def is_constitutional(self) -> bool:
        """Check if result meets constitutional thresholds."""
        return (
            self.ihsan_score >= CONSTITUTIONAL_CONSTRAINTS["ihsan_threshold"]
            and self.snr_score >= SNR_THRESHOLDS["action"]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query[:100],
            "response": self.response[:500],
            "confidence": self.confidence,
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
            "quality_score": self.quality_score,
            "is_constitutional": self.is_constitutional,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "giants_invoked": self.giants_invoked,
            "techniques_used": self.techniques_used,
            "graph_stats": self.graph_stats,
            "duration_ms": self.duration_ms,
            "version": self.version,
        }


class SovereignReasoningEngine:
    """
    The Sovereign Autonomous Reasoning Engine.

    Pinnacle synthesis of interdisciplinary approaches:

    1. OBSERVATION (Shannon)
       - Information-theoretic input processing
       - SNR maximization from first principles

    2. ORIENTATION (Vaswani)
       - Attention-weighted context establishment
       - Relevant information extraction

    3. REASONING (Besta)
       - Graph-of-Thoughts non-linear inference
       - Backtracking on quality degradation
       - Multi-path synthesis

    4. VALIDATION (Anthropic)
       - Constitutional constraint enforcement
       - Ihsān excellence verification

    5. CONSENSUS (Lamport)
       - Distributed verification
       - Formal provenance chains

    Usage:
        engine = SovereignReasoningEngine()
        result = await engine.reason("What is the meaning of existence?")
        print(result.response)
        print(f"Quality: {result.quality_score:.3f}")
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        max_loops: int = 3,
        max_backtrack: int = 5,
        strict_mode: bool = True,
    ):
        """
        Initialize the Sovereign Reasoning Engine.

        Args:
            llm_fn: Optional LLM function for inference
            max_loops: Maximum reasoning iterations
            max_backtrack: Maximum backtracking attempts
            strict_mode: If True, fail on constitutional violations
        """
        self.llm_fn = llm_fn
        self.max_loops = max_loops
        self.max_backtrack = max_backtrack
        self.strict_mode = strict_mode

        # Core components
        self._giants = GiantsProtocol()
        self._loop = SovereignLoop(
            llm_fn=llm_fn,
            max_loops=max_loops,
            max_backtrack=max_backtrack,
        )

        # State
        self._results: List[ReasoningResult] = []
        self._total_queries = 0
        self._total_duration_ms = 0.0

        # Constitutional constraints
        self._constitutional = CONSTITUTIONAL_CONSTRAINTS
        self._snr_thresholds = SNR_THRESHOLDS

        logger.info(f"SARE v{SARE_VERSION} initialized")
        logger.info(f"Giants Protocol: {list(GIANTS_PROTOCOL.keys())}")

    # =========================================================================
    # CORE REASONING
    # =========================================================================

    async def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """
        Execute autonomous reasoning on a query.

        This is the main entry point for the engine.

        Args:
            query: The input query or content to reason about
            context: Optional context dictionary

        Returns:
            ReasoningResult with response and provenance
        """
        start_time = time.time()
        context = context or {}
        self._total_queries += 1

        logger.info(f"SARE reasoning: {query[:50]}...")

        # Execute sovereign loop
        execution = await self._loop.execute(query, context)

        # Extract results
        response = execution.output_content
        snr = execution.final_snr
        ihsan = execution.final_ihsan

        # Get graph stats
        graph = self._loop.get_graph()
        graph_stats = graph.get_stats() if graph else {}

        # Collect giants and techniques
        giants_invoked = list(
            set(
                [p.node_id.split("_")[0] if p.node_id else "" for p in execution.phases]
            )
        )
        techniques_used = list(
            set(
                [
                    p.metadata.get("technique", "") if p.metadata else ""
                    for p in execution.phases
                ]
            )
        )

        duration = (time.time() - start_time) * 1000
        self._total_duration_ms += duration

        # Create result
        result = ReasoningResult(
            query=query,
            response=response,
            confidence=min(snr, ihsan),
            snr_score=snr,
            ihsan_score=ihsan,
            provenance=execution.provenance,
            giants_invoked=[g for g in giants_invoked if g],
            techniques_used=[t for t in techniques_used if t],
            graph_stats=graph_stats,
            best_path=execution.best_path,
            loop_execution=execution,
            duration_ms=duration,
        )

        # Constitutional check
        if self.strict_mode and not result.is_constitutional:
            logger.warning(
                f"Result below constitutional threshold: "
                f"SNR={snr:.3f}, Ihsān={ihsan:.3f}"
            )
            result.response = self._generate_fallback_response(query, result)

        self._results.append(result)

        logger.info(
            f"SARE completed: SNR={snr:.3f}, Ihsān={ihsan:.3f}, "
            f"Duration={duration:.1f}ms"
        )

        return result

    def _generate_fallback_response(
        self,
        query: str,
        result: ReasoningResult,
    ) -> str:
        """Generate a safe fallback response when constitutional thresholds not met."""
        return (
            f"I apologize, but I cannot provide a fully confident response to this query. "
            f"My reasoning achieved SNR={result.snr_score:.3f} and Ihsān={result.ihsan_score:.3f}, "
            f"which is below my quality threshold of {self._constitutional['ihsan_threshold']:.2f}. "
            f"Please consider rephrasing or providing more context."
        )

    # =========================================================================
    # SPECIALIZED REASONING MODES
    # =========================================================================

    async def analyze(
        self,
        content: str,
        analysis_type: str = "comprehensive",
    ) -> ReasoningResult:
        """
        Analytical reasoning mode.

        Optimized for deep analysis with emphasis on:
        - Pattern extraction
        - Causal reasoning
        - Multi-perspective synthesis
        """
        context = {
            "mode": "analysis",
            "analysis_type": analysis_type,
            "emphasis": ["patterns", "causality", "perspectives"],
        }

        augmented_query = f"Analyze the following content comprehensively:\n\n{content}"

        return await self.reason(augmented_query, context)

    async def synthesize(
        self,
        sources: List[str],
        synthesis_goal: str = "integrate",
    ) -> ReasoningResult:
        """
        Synthesis reasoning mode.

        Optimized for integrating multiple sources with emphasis on:
        - Cross-source patterns
        - Coherent integration
        - Novel insights
        """
        context = {
            "mode": "synthesis",
            "synthesis_goal": synthesis_goal,
            "source_count": len(sources),
        }

        sources_text = "\n\n---\n\n".join(sources[:10])  # Limit sources
        augmented_query = (
            f"Synthesize these sources into a coherent understanding:\n\n{sources_text}"
        )

        return await self.reason(augmented_query, context)

    async def evaluate(
        self,
        content: str,
        criteria: Optional[List[str]] = None,
    ) -> ReasoningResult:
        """
        Evaluative reasoning mode.

        Optimized for assessment with emphasis on:
        - Criteria-based judgment
        - Quality scoring
        - Improvement recommendations
        """
        criteria = criteria or ["quality", "clarity", "completeness", "accuracy"]

        context = {
            "mode": "evaluation",
            "criteria": criteria,
        }

        criteria_text = ", ".join(criteria)
        augmented_query = f"Evaluate the following content against these criteria ({criteria_text}):\n\n{content}"

        return await self.reason(augmented_query, context)

    async def create(
        self,
        specification: str,
        output_type: str = "text",
    ) -> ReasoningResult:
        """
        Creative reasoning mode.

        Optimized for generation with emphasis on:
        - Originality
        - Coherence
        - Quality
        """
        context = {
            "mode": "creation",
            "output_type": output_type,
        }

        augmented_query = (
            f"Create {output_type} based on this specification:\n\n{specification}"
        )

        return await self.reason(augmented_query, context)

    # =========================================================================
    # GIANTS PROTOCOL ACCESS
    # =========================================================================

    def invoke_giant(
        self,
        giant: str,
        technique: str,
        *args,
        **kwargs,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Directly invoke a technique from a giant.

        Args:
            giant: Giant name (shannon, lamport, vaswani, besta, anthropic)
            technique: Technique name
            *args, **kwargs: Technique arguments

        Returns:
            (result, inheritance_record)
        """
        giant_enum = Giant(giant)
        result, inheritance = self._giants.invoke(
            giant_enum, technique, *args, **kwargs
        )
        return result, inheritance.to_dict()

    def get_giant_info(self, giant: str) -> Dict[str, Any]:
        """Get information about a giant."""
        return self._giants.get_giant_info(Giant(giant))

    def list_giants(self) -> List[Dict[str, Any]]:
        """List all giants and their contributions."""
        return [
            {
                "name": info["name"],
                "work": info["work"],
                "contribution": info["contribution"],
                "application": info["application"],
            }
            for info in GIANTS_PROTOCOL.values()
        ]

    def list_techniques(self, giant: Optional[str] = None) -> List[str]:
        """List available techniques."""
        if giant:
            return self._giants.list_techniques(Giant(giant))
        return self._giants.list_techniques()

    # =========================================================================
    # STATISTICS AND INTROSPECTION
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        if not self._results:
            return {
                "version": SARE_VERSION,
                "total_queries": 0,
                "constitutional_thresholds": self._constitutional,
            }

        return {
            "version": SARE_VERSION,
            "total_queries": self._total_queries,
            "total_duration_ms": self._total_duration_ms,
            "avg_duration_ms": self._total_duration_ms / self._total_queries,
            "avg_snr": sum(r.snr_score for r in self._results) / len(self._results),
            "avg_ihsan": sum(r.ihsan_score for r in self._results) / len(self._results),
            "avg_quality": sum(r.quality_score for r in self._results)
            / len(self._results),
            "constitutional_compliance": sum(
                1 for r in self._results if r.is_constitutional
            )
            / len(self._results),
            "loop_stats": self._loop.get_stats(),
            "constitutional_thresholds": self._constitutional,
        }

    def get_recent_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent reasoning results."""
        return [r.to_dict() for r in self._results[-limit:]]

    def get_provenance_chain(self) -> List[Dict[str, Any]]:
        """Get the complete provenance chain."""
        return self._giants.get_provenance_chain()

    def get_constitutional_status(self) -> Dict[str, Any]:
        """Get current constitutional compliance status."""
        if not self._results:
            return {"status": "no_results", "compliance": 1.0}

        recent = self._results[-10:]
        compliance = sum(1 for r in recent if r.is_constitutional) / len(recent)

        return {
            "status": (
                "compliant"
                if compliance >= 0.9
                else "warning" if compliance >= 0.7 else "critical"
            ),
            "compliance": compliance,
            "recent_results": len(recent),
            "constitutional_count": sum(1 for r in recent if r.is_constitutional),
            "thresholds": self._constitutional,
        }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def reset(self) -> None:
        """Reset engine state."""
        self._results = []
        self._total_queries = 0
        self._total_duration_ms = 0.0
        logger.info("SARE state reset")

    def get_version(self) -> str:
        """Get engine version."""
        return SARE_VERSION

    def __repr__(self) -> str:
        return (
            f"SovereignReasoningEngine("
            f"version={SARE_VERSION}, "
            f"queries={self._total_queries}, "
            f"strict_mode={self.strict_mode})"
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_sovereign_engine(
    llm_fn: Optional[Callable[[str], str]] = None,
    **kwargs,
) -> SovereignReasoningEngine:
    """
    Factory function to create a Sovereign Reasoning Engine.

    This is the recommended way to instantiate the engine.

    Args:
        llm_fn: Optional LLM function for inference
        **kwargs: Additional configuration

    Returns:
        Configured SovereignReasoningEngine
    """
    return SovereignReasoningEngine(llm_fn=llm_fn, **kwargs)
