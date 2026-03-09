"""
Cognitive Fusion Engine: MoE complexity -> HRM level -> HyperGraph RAG -> NorthStar gate.

Bridges four cognitive subsystems into a single inference pipeline:
  1. MoE Router  -- classifies query complexity (TRIVIAL..FRONTIER)
  2. HRM Engine  -- reasons at the appropriate abstraction level
  3. HyperGraph RAG -- retrieves context scaled to complexity depth
  4. NorthStar   -- gates output against SNR + Ihsan thresholds

All dependencies are optional (Protocol-typed).  When absent the engine
falls back to sensible defaults, making the module usable as a standalone
reasoning scaffold even before the concrete subsystems are wired in.

Standing on Giants: Vaswani (MoE) + Simon (hierarchy) + Shannon (SNR) + Besta (GoT)

Constitutional Alignment:
  All thresholds imported from core/integration/constants.py (SSOT).
  NEVER hardcode 0.85, 0.95, 0.98 -- always reference the constant.

Created: 2026-02-17 | BIZRA Node0 | Cognitive Fusion Phase
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from core.cognitive_fusion.complexity_adapter import ComplexityAdapter
from core.integration.constants import (
    GOT_MAX_DEPTH,
    SNR_THRESHOLD_T0_ELITE,
    STRICT_IHSAN_THRESHOLD,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOL RESULT DATACLASSES
# =============================================================================
# Lightweight stand-ins so the engine never needs to import concrete modules.


@dataclass
class RoutingResult:
    """Result of MoE complexity routing."""

    complexity_class: str = (
        "STANDARD"  # TRIVIAL | STANDARD | COMPLEX | EXPERT | FRONTIER
    )
    expert_tier: str = "EDGE"
    confidence: float = 0.85
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HRMResult:
    """Result of a single HRM reasoning cycle."""

    compound_snr: float = 0.85
    level_reached: str = "OPERATIONAL"
    observations: List[str] = field(default_factory=list)


@dataclass
class NorthStarResult:
    """Result of NorthStar quality gate evaluation."""

    unified_snr: float = 0.85
    ihsan_score: float = 0.95
    passes_all_gates: bool = True
    flow_report: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# DEPENDENCY PROTOCOLS
# =============================================================================
# Protocol classes define the minimal interface each subsystem must satisfy.
# Using structural typing (PEP 544) so concrete implementations need only
# provide matching method signatures -- no inheritance required.


@runtime_checkable
class MoERouterProtocol(Protocol):
    """Mixture-of-Experts router that classifies query complexity."""

    def route(
        self, query: str, constraints: Dict[str, Any] | None = None
    ) -> RoutingResult: ...


@runtime_checkable
class HRMEngineProtocol(Protocol):
    """Hierarchical Reasoning Model engine."""

    def run_cycle(self, observation: Dict[str, Any]) -> HRMResult: ...


@runtime_checkable
class RAGFusionProtocol(Protocol):
    """HyperGraph RAG retrieval engine."""

    def retrieve(
        self, query: str, query_embedding: List[float], top_k: int = 10
    ) -> List[Any]: ...


@runtime_checkable
class NorthStarProtocol(Protocol):
    """NorthStar quality / Ihsan gate engine."""

    def run_cycle(self, observation: Dict[str, Any]) -> NorthStarResult: ...


# =============================================================================
# FUSION RESULT
# =============================================================================


@dataclass
class FusionResult:
    """
    Unified output of the Cognitive Fusion pipeline.

    Aggregates routing decision, HRM reasoning, RAG retrieval, and
    NorthStar gate verdict into a single inspectable result.
    """

    routing: RoutingResult
    hrm_result: HRMResult
    retrieval: List[Any]
    northstar_report: NorthStarResult
    target_level: str  # AbstractionLevel name (e.g. "OPERATIONAL")
    snr_score: float
    ihsan_score: float
    passes_gate: bool
    degraded: bool = False  # P1: True when engine started with missing Protocol args

    @property
    def is_elite(self) -> bool:
        """True when both SNR and Ihsan exceed T0-elite thresholds."""
        return (
            self.snr_score >= SNR_THRESHOLD_T0_ELITE
            and self.ihsan_score >= STRICT_IHSAN_THRESHOLD
        )

    @property
    def is_frontier(self) -> bool:
        """True when the query was routed to FRONTIER tier."""
        return self.routing.complexity_class == "FRONTIER"

    @property
    def expert_tier(self) -> str:
        """The expert tier selected by the MoE router."""
        return self.routing.expert_tier

    @property
    def compound_snr(self) -> float:
        """Compound SNR from the HRM reasoning cycle."""
        return self.hrm_result.compound_snr


# =============================================================================
# COGNITIVE FUSION ENGINE
# =============================================================================


class CognitiveFusionEngine:
    """
    Orchestrates the four-stage cognitive fusion pipeline.

    Pipeline stages:
      1. MoE Route   -- classify complexity (or default STANDARD)
      2. HRM Adapt   -- map complexity to abstraction level via ComplexityAdapter
      3. RAG Retrieve -- fetch context at depth scaled to complexity
      4. NorthStar   -- gate on SNR + Ihsan thresholds

    All four subsystems are optional.  Passing ``None`` (the default) for any
    subsystem causes the engine to use a safe default result for that stage,
    enabling incremental integration.

    Example::

        engine = CognitiveFusionEngine()
        result = engine.process("What is autopoiesis?", embedding_vec)
        assert result.passes_gate
    """

    # -- construction ----------------------------------------------------------

    def __init__(
        self,
        moe_router: Optional[MoERouterProtocol] = None,
        hrm_engine: Optional[HRMEngineProtocol] = None,
        hypergraph_rag: Optional[RAGFusionProtocol] = None,
        northstar_engine: Optional[NorthStarProtocol] = None,
        frontier_mode: bool = False,
    ) -> None:
        self._moe_router = moe_router
        self._hrm_engine = hrm_engine
        self._hypergraph_rag = hypergraph_rag
        self._northstar_engine = northstar_engine
        self._frontier_mode = frontier_mode
        self._adapter = ComplexityAdapter()

        # P1: Degradation transparency — never silent failure
        from core.protocols.degradation import DegradationEmitter

        emitter = DegradationEmitter("CognitiveFusionEngine")
        emitter.check("moe_router", moe_router)
        emitter.check("hrm_engine", hrm_engine)
        emitter.check("hypergraph_rag", hypergraph_rag)
        emitter.check("northstar_engine", northstar_engine)
        self._degradation_event = emitter.emit()
        self._degraded = self._degradation_event is not None

        logger.info(
            "CognitiveFusionEngine initialised — "
            "moe=%s  hrm=%s  rag=%s  northstar=%s",
            moe_router is not None,
            hrm_engine is not None,
            hypergraph_rag is not None,
            northstar_engine is not None,
        )

    # -- public properties -----------------------------------------------------

    @property
    def frontier_mode(self) -> bool:
        """Whether FRONTIER tier reasoning is enabled."""
        return self._frontier_mode

    # -- public API ------------------------------------------------------------

    def process(
        self,
        query: str,
        query_embedding: List[float],
        context: Dict[str, Any] | None = None,
    ) -> FusionResult:
        """
        Execute the full cognitive fusion pipeline.

        Args:
            query: Natural-language query string.
            query_embedding: Dense vector representation of *query*.
            context: Optional additional context forwarded to subsystems.

        Returns:
            ``FusionResult`` aggregating all four pipeline stages.
        """
        ctx = context or {}
        t0 = time.monotonic()

        # Stage 1 -- MoE Routing
        routing = self._route(query, ctx)

        # Stage 2 -- Complexity -> HRM Level
        target_level, required_snr = self._adapter.adapt(routing.complexity_class)
        expert_tier = self._adapter.level_to_tier(target_level)
        routing.expert_tier = expert_tier

        # P4: FRONTIER tier activation — deeper exploration + stricter gate
        is_frontier = (
            self._frontier_mode and routing.complexity_class == "FRONTIER"
        )
        if is_frontier:
            required_snr = SNR_THRESHOLD_T0_ELITE  # 0.98
            logger.info(
                "FRONTIER tier activated — SNR gate raised to %.2f, "
                "GoT depth doubled, cross-domain RAG enabled",
                required_snr,
            )

        # Stage 3 -- HRM Cycle (FRONTIER doubles max_depth via context)
        hrm_ctx = {**ctx}
        if is_frontier:
            hrm_ctx["got_max_depth"] = GOT_MAX_DEPTH * 2
            hrm_ctx["frontier_mode"] = True
        hrm_result = self._run_hrm(target_level, query, hrm_ctx)

        # Stage 4 -- HyperGraph RAG Retrieval (FRONTIER uses cross-domain)
        depth = self._retrieval_depth(routing.complexity_class)
        if is_frontier:
            depth = max(depth, 50)  # ensure deep cross-domain retrieval
        retrieval = self._retrieve(query, query_embedding, depth)

        # Stage 5 -- NorthStar Gate
        ns_observation = {
            "query": query,
            "complexity": routing.complexity_class,
            "target_level": target_level,
            "hrm_snr": hrm_result.compound_snr,
            "retrieval_count": len(retrieval),
            "frontier_mode": is_frontier,
            **ctx,
        }
        northstar_report = self._gate(ns_observation)

        # Derive aggregate scores
        snr_score = self._aggregate_snr(hrm_result, northstar_report)
        ihsan_score = northstar_report.ihsan_score
        passes = (
            northstar_report.passes_all_gates
            and snr_score >= required_snr
            and ihsan_score >= UNIFIED_IHSAN_THRESHOLD
        )

        # P4: Emit SYNTHESIS consciousness event for FRONTIER queries
        domains_crossed = len(set(
            r.get("domain", "unknown") if isinstance(r, dict) else "unknown"
            for r in retrieval
        )) if is_frontier and retrieval else 0

        elapsed = time.monotonic() - t0
        logger.debug(
            "CognitiveFusion completed in %.3fs — "
            "complexity=%s  level=%s  snr=%.3f  ihsan=%.3f  gate=%s"
            "%s",
            elapsed,
            routing.complexity_class,
            target_level,
            snr_score,
            ihsan_score,
            passes,
            f"  frontier=True  domains_crossed={domains_crossed}"
            if is_frontier else "",
        )

        return FusionResult(
            routing=routing,
            hrm_result=hrm_result,
            retrieval=retrieval,
            northstar_report=northstar_report,
            target_level=target_level,
            snr_score=snr_score,
            ihsan_score=ihsan_score,
            passes_gate=passes,
            degraded=self._degraded,
        )

    # -- static helpers --------------------------------------------------------

    @staticmethod
    def _retrieval_depth(complexity: str) -> int:
        """
        Map complexity class to RAG retrieval depth (top-k).

        Higher complexity classes warrant deeper context retrieval
        at the cost of additional latency and compute.
        """
        depths: Dict[str, int] = {
            "TRIVIAL": 3,
            "STANDARD": 5,
            "COMPLEX": 10,
            "EXPERT": 20,
            "FRONTIER": 50,
        }
        return depths.get(complexity, 5)

    # -- private stage methods -------------------------------------------------

    def _route(self, query: str, ctx: Dict[str, Any]) -> RoutingResult:
        """Stage 1: MoE complexity routing (or default)."""
        if self._moe_router is not None:
            try:
                return self._moe_router.route(query, constraints=ctx)
            except Exception:
                logger.warning(
                    "MoE router failed — falling back to STANDARD", exc_info=True
                )
        return RoutingResult(
            complexity_class="STANDARD",
            expert_tier="EDGE",
            confidence=UNIFIED_SNR_THRESHOLD,
            metadata={"source": "default"},
        )

    def _run_hrm(self, target_level: str, query: str, ctx: Dict[str, Any]) -> HRMResult:
        """Stage 3: HRM reasoning cycle (or default)."""
        if self._hrm_engine is not None:
            try:
                observation = {
                    "query": query,
                    "target_level": target_level,
                    **ctx,
                }
                return self._hrm_engine.run_cycle(observation)
            except Exception:
                logger.warning(
                    "HRM engine failed — using default result", exc_info=True
                )
        return HRMResult(
            compound_snr=UNIFIED_SNR_THRESHOLD,
            level_reached=target_level,
            observations=[],
        )

    def _retrieve(
        self, query: str, query_embedding: List[float], top_k: int
    ) -> List[Any]:
        """Stage 4: HyperGraph RAG retrieval (or empty list)."""
        if self._hypergraph_rag is not None:
            try:
                return self._hypergraph_rag.retrieve(
                    query, query_embedding, top_k=top_k
                )
            except Exception:
                logger.warning("RAG retrieval failed — returning empty", exc_info=True)
        return []

    def _gate(self, observation: Dict[str, Any]) -> NorthStarResult:
        """Stage 5: NorthStar quality gate (or default pass)."""
        if self._northstar_engine is not None:
            try:
                return self._northstar_engine.run_cycle(observation)
            except Exception:
                logger.warning("NorthStar gate failed — using default", exc_info=True)
        return NorthStarResult(
            unified_snr=UNIFIED_SNR_THRESHOLD,
            ihsan_score=UNIFIED_IHSAN_THRESHOLD,
            passes_all_gates=True,
            flow_report={"source": "default"},
        )

    @staticmethod
    def _aggregate_snr(hrm: HRMResult, ns: NorthStarResult) -> float:
        """
        Combine HRM compound SNR and NorthStar unified SNR.

        Uses the geometric mean to penalise low scores in either dimension
        while rewarding consistent quality across both.
        """
        return (hrm.compound_snr * ns.unified_snr) ** 0.5
