"""
SpearPoint Pipeline — The Cockpit
===================================
Unified orchestrator that consolidates the 7 fire-and-forget post-query
operations + SAT health check into a single, observable, fault-tolerant pipeline.

Each step runs with independent error handling so one failure never
blocks another. The pipeline returns a SpearPointResult with per-step
status, enabling observability and diagnostics.

Standing on Giants:
- Lamport (1978): Fail-closed with per-step isolation
- Shannon (1948): SNR gating determines which steps execute
- Nakamoto (2008): Evidence receipt → hash-chained ledger
- Besta (2024): Graph artifacts as first-class proof objects
- Tulving (1972): Episodic memory encoding

Pipeline stages (ordered by dependency):
  1. graph_artifact   — Store GoT graph for retrieval
  2. evidence_receipt  — Emit hash-chained receipt to Evidence Ledger
  3. record_impact     — Record sovereignty impact (UERSScore)
  4. poi_contribution  — Register PoI contribution (requires SNR score)
  5. living_memory     — Encode experience into Living Memory
  6. experience_ledger — Auto-commit episode to SEL on SNR_OK
  7. judgment_observe  — SJE Phase A telemetry observation
  8. sat_health        — SAT ecosystem Gini health check (observational)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from core.proof_engine.canonical import hex_digest

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Outcome of a single pipeline step."""

    name: str
    success: bool
    duration_ms: float = 0.0
    detail: str = ""
    error: Optional[str] = None


@dataclass
class SpearPointResult:
    """Aggregate outcome of the full spearpoint pipeline."""

    steps: list[StepResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    all_passed: bool = True

    @property
    def failed_steps(self) -> list[str]:
        return [s.name for s in self.steps if not s.success]

    @property
    def step_summary(self) -> dict[str, bool]:
        return {s.name: s.success for s in self.steps}

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "steps": [
                {
                    "name": s.name,
                    "success": s.success,
                    "duration_ms": round(s.duration_ms, 2),
                    "detail": s.detail,
                    **({"error": s.error} if s.error else {}),
                }
                for s in self.steps
            ],
        }


class SpearPointPipeline:
    """Unified post-query orchestrator — the cockpit.

    Consolidates 7 fire-and-forget operations + SAT health check into one observable pipeline.
    Each step is independently error-isolated. Failure in one step never
    blocks or corrupts another.

    Usage:
        pipeline = SpearPointPipeline(
            evidence_ledger=ledger,
            graph_reasoner=got,
            living_memory=memory,
            experience_ledger=sel,
            poi_orchestrator=poi,
            judgment_telemetry=sje,
            impact_tracker=impact,
            config=runtime_config,
        )
        spearpoint_result = await pipeline.execute(result, query)
    """

    def __init__(
        self,
        *,
        evidence_ledger: Any = None,
        graph_reasoner: Any = None,
        graph_artifacts: Optional[dict[str, Any]] = None,
        living_memory: Any = None,
        experience_ledger: Any = None,
        poi_orchestrator: Any = None,
        judgment_telemetry: Any = None,
        impact_tracker: Any = None,
        sat_controller: Any = None,
        config: Any = None,
        snr_trace_ref: Optional[list] = None,
    ):
        self._evidence_ledger = evidence_ledger
        self._graph_reasoner = graph_reasoner
        self._graph_artifacts = graph_artifacts if graph_artifacts is not None else {}
        self._living_memory = living_memory
        self._experience_ledger = experience_ledger
        self._poi_orchestrator = poi_orchestrator
        self._judgment_telemetry = judgment_telemetry
        self._impact_tracker = impact_tracker
        self._sat_controller = sat_controller
        self._config = config
        # Mutable reference to the runtime's _last_snr_trace slot
        self._snr_trace_ref = snr_trace_ref

    async def execute(self, result: Any, query: Any) -> SpearPointResult:
        """Run the full spearpoint pipeline.

        Each step is isolated — one failure never prevents the next.
        Returns SpearPointResult with per-step diagnostics.
        """
        pipeline_start = time.perf_counter()
        sp_result = SpearPointResult()

        # Step 1: Store graph artifact
        sp_result.steps.append(self._step_graph_artifact(result, query))

        # Step 2: Emit evidence receipt
        sp_result.steps.append(self._step_evidence_receipt(result, query))

        # Step 3: Record impact (sovereignty progression)
        sp_result.steps.append(self._step_record_impact(result))

        # Step 4: Register PoI contribution
        sp_result.steps.append(self._step_poi_contribution(result, query))

        # Step 5: Encode living memory
        sp_result.steps.append(await self._step_living_memory(result, query))

        # Step 6: Commit experience to SEL
        sp_result.steps.append(self._step_experience_ledger(result, query))

        # Step 7: Judgment telemetry
        sp_result.steps.append(self._step_judgment_observe(result))

        # Step 8: SAT ecosystem health check (observational)
        sp_result.steps.append(self._step_sat_health_check(result))

        # Finalize
        sp_result.total_duration_ms = (time.perf_counter() - pipeline_start) * 1000
        sp_result.all_passed = all(s.success for s in sp_result.steps)

        if not sp_result.all_passed:
            logger.info(
                f"SpearPoint pipeline: {len(sp_result.failed_steps)} step(s) "
                f"degraded: {sp_result.failed_steps}"
            )

        return sp_result

    # ------------------------------------------------------------------
    # Step implementations (each independently error-isolated)
    # ------------------------------------------------------------------

    def _step_graph_artifact(self, result: Any, query: Any) -> StepResult:
        """Step 1: Store GoT graph artifact for retrieval."""
        t0 = time.perf_counter()
        try:
            if not self._graph_reasoner:
                return StepResult(
                    name="graph_artifact",
                    success=True,
                    duration_ms=_elapsed(t0),
                    detail="skipped (no graph reasoner)",
                )
            to_artifact = getattr(self._graph_reasoner, "to_artifact", None)
            if to_artifact is None or not result.graph_hash:
                return StepResult(
                    name="graph_artifact",
                    success=True,
                    duration_ms=_elapsed(t0),
                    detail="skipped (no artifact method or graph_hash)",
                )
            artifact = to_artifact(build_id=query.id)
            self._graph_artifacts[query.id] = artifact
            # Bound cache to prevent unbounded memory growth
            if len(self._graph_artifacts) > 100:
                oldest = next(iter(self._graph_artifacts))
                del self._graph_artifacts[oldest]
            return StepResult(
                name="graph_artifact",
                success=True,
                duration_ms=_elapsed(t0),
                detail=f"stored (hash={result.graph_hash[:12]}...)",
            )
        except Exception as e:
            return StepResult(
                name="graph_artifact",
                success=False,
                duration_ms=_elapsed(t0),
                error=str(e),
            )

    def _step_evidence_receipt(self, result: Any, query: Any) -> StepResult:
        """Step 2: Emit hash-chained receipt into Evidence Ledger."""
        t0 = time.perf_counter()
        if self._evidence_ledger is None:
            return StepResult(
                name="evidence_receipt",
                success=True,
                duration_ms=_elapsed(t0),
                detail="skipped (no evidence ledger)",
            )
        try:
            from core.proof_engine.evidence_ledger import emit_receipt

            decision = "APPROVED"
            reason_codes: list = []
            status = "accepted"
            ihsan_threshold = getattr(self._config, "ihsan_threshold", 0.95)

            if not result.validation_passed:
                decision = "REJECTED"
                reason_codes.append("IHSAN_BELOW_THRESHOLD")
                status = "rejected"
            if result.snr_score < 0.85:
                if "SNR_BELOW_THRESHOLD" not in reason_codes:
                    reason_codes.append("SNR_BELOW_THRESHOLD")
                if decision == "APPROVED":
                    decision = "QUARANTINED"
                    status = "quarantined"

            query_digest = hex_digest(
                query.text.encode("utf-8")
            )  # SEC-001: BLAKE3 for Rust interop
            seal_digest = hex_digest(
                (result.response or "").encode("utf-8")
            )  # SEC-001: BLAKE3 for Rust interop

            node_id = getattr(self._config, "node_id", "BIZRA-00000000")

            emit_receipt(
                self._evidence_ledger,
                receipt_id=result.query_id.replace("-", "")[:32],
                node_id=node_id,
                policy_version="1.0.0",
                status=status,
                decision=decision,
                reason_codes=reason_codes,
                snr_score=result.snr_score,
                ihsan_score=result.ihsan_score,
                ihsan_threshold=ihsan_threshold,
                seal_digest=seal_digest,
                query_digest=query_digest,
                graph_hash=result.graph_hash,
                payload_digest=(
                    hex_digest("|".join(result.thoughts).encode("utf-8"))
                    if result.thoughts
                    else None
                ),  # SEC-001: BLAKE3 for Rust interop
                gate_passed="commit" if decision == "APPROVED" else "ihsan_gate",
                duration_ms=result.processing_time_ms,
                claim_tags=(
                    {
                        "measured": sum(
                            1 for v in result.claim_tags.values() if v == "measured"
                        ),
                        "design": sum(
                            1 for v in result.claim_tags.values() if v == "design"
                        ),
                        "implemented": sum(
                            1 for v in result.claim_tags.values() if v == "implemented"
                        ),
                        "target": sum(
                            1 for v in result.claim_tags.values() if v == "target"
                        ),
                    }
                    if result.claim_tags
                    else None
                ),
                snr_trace=(
                    self._snr_trace_ref[0]
                    if self._snr_trace_ref and self._snr_trace_ref[0]
                    else None
                ),
                critical_decision=True,
            )
            # Clear trace after emission
            if self._snr_trace_ref:
                self._snr_trace_ref[0] = None

            return StepResult(
                name="evidence_receipt",
                success=True,
                duration_ms=_elapsed(t0),
                detail=f"emitted ({decision})",
            )
        except Exception as e:
            return StepResult(
                name="evidence_receipt",
                success=False,
                duration_ms=_elapsed(t0),
                error=str(e),
            )

    def _step_record_impact(self, result: Any) -> StepResult:
        """Step 3: Record query impact for sovereignty progression."""
        t0 = time.perf_counter()
        if not self._impact_tracker:
            return StepResult(
                name="record_impact",
                success=True,
                duration_ms=_elapsed(t0),
                detail="skipped (no impact tracker)",
            )
        try:
            from core.pat.impact_tracker import UERSScore, compute_query_bloom

            bloom = compute_query_bloom(
                processing_time_ms=result.processing_time_ms,
                reasoning_depth=result.reasoning_depth,
                validated=getattr(result, "validation_passed", False),
            )

            uers = UERSScore(
                utility=min(1.0, len(result.response or "") / 500),
                efficiency=min(1.0, 1.0 - (result.processing_time_ms / 10000)),
                resilience=result.snr_score,
                sustainability=0.5,
                ethics=result.ihsan_score,
            )
            self._impact_tracker.record_query(bloom=bloom, uers=uers)
            return StepResult(
                name="record_impact",
                success=True,
                duration_ms=_elapsed(t0),
                detail=f"recorded (bloom={bloom:.3f})",
            )
        except Exception as e:
            return StepResult(
                name="record_impact",
                success=False,
                duration_ms=_elapsed(t0),
                error=str(e),
            )

    def _step_poi_contribution(self, result: Any, query: Any) -> StepResult:
        """Step 4: Register PoI contribution."""
        t0 = time.perf_counter()
        if self._poi_orchestrator is None:
            return StepResult(
                name="poi_contribution",
                success=True,
                duration_ms=_elapsed(t0),
                detail="skipped (no PoI orchestrator)",
            )
        if not result.success:
            return StepResult(
                name="poi_contribution",
                success=True,
                duration_ms=_elapsed(t0),
                detail="skipped (query not successful)",
            )
        try:
            from core.proof_engine.poi_engine import (
                ContributionMetadata,
                ContributionType,
            )

            node_id = getattr(self._config, "node_id", "BIZRA-00000000")
            content_hash = result.graph_hash or result.query_id
            metadata = ContributionMetadata(
                contributor_id=node_id,
                contribution_type=ContributionType.DATA,
                content_hash=content_hash,
                snr_score=result.snr_score,
                ihsan_score=result.ihsan_score,
                timestamp=datetime.now(),
            )
            self._poi_orchestrator.register_contribution(metadata)
            return StepResult(
                name="poi_contribution",
                success=True,
                duration_ms=_elapsed(t0),
                detail=f"registered (hash={content_hash[:12]}...)",
            )
        except Exception as e:
            return StepResult(
                name="poi_contribution",
                success=False,
                duration_ms=_elapsed(t0),
                error=str(e),
            )

    async def _step_living_memory(self, result: Any, query: Any) -> StepResult:
        """Step 5: Encode experience into Living Memory."""
        t0 = time.perf_counter()
        if self._living_memory is None or not result.success:
            return StepResult(
                name="living_memory",
                success=True,
                duration_ms=_elapsed(t0),
                detail="skipped (no memory or unsuccessful query)",
            )
        if not result.response:
            return StepResult(
                name="living_memory",
                success=True,
                duration_ms=_elapsed(t0),
                detail="skipped (empty response)",
            )
        try:
            from core.living_memory.core import MemoryType

            q_text = query.text[:500]
            r_text = (result.response or "")[:500]
            content = (
                f"Query: {q_text}\n"
                f"Response: {r_text}\n"
                f"SNR: {result.snr_score:.3f} | Ihsan: {result.ihsan_score:.3f}"
            )

            await self._living_memory.encode(
                content=content,
                memory_type=MemoryType.EPISODIC,
                source="query_pipeline",
                importance=result.ihsan_score,
                emotional_weight=max(result.snr_score, 0.5),
            )
            return StepResult(
                name="living_memory",
                success=True,
                duration_ms=_elapsed(t0),
                detail="encoded",
            )
        except Exception as e:
            return StepResult(
                name="living_memory",
                success=False,
                duration_ms=_elapsed(t0),
                error=str(e),
            )

    def _step_experience_ledger(self, result: Any, query: Any) -> StepResult:
        """Step 6: Auto-commit episode to SEL on SNR_OK."""
        t0 = time.perf_counter()
        if self._experience_ledger is None:
            return StepResult(
                name="experience_ledger",
                success=True,
                duration_ms=_elapsed(t0),
                detail="skipped (no experience ledger)",
            )
        if not result.success or not result.snr_ok:
            return StepResult(
                name="experience_ledger",
                success=True,
                duration_ms=_elapsed(t0),
                detail="skipped (query not successful or SNR below threshold)",
            )
        try:
            graph_hash = ""
            graph_node_count = 0
            if result.thoughts:
                graph_hash = hex_digest(
                    "|".join(result.thoughts).encode("utf-8")
                )  # SEC-001: BLAKE3 for Rust interop
                graph_node_count = len(result.thoughts)

            actions = []
            model_used = result.model_used
            if model_used:
                actions.append(
                    (
                        "inference",
                        f"LLM: {model_used}",
                        True,
                        int(result.processing_time_ms * 1000),
                    )
                )
            if result.snr_ok:
                actions.append(
                    (
                        "snr_gate",
                        f"SNR={result.snr_score:.3f}",
                        True,
                        0,
                    )
                )

            response_summary = (result.response or "")[:500] or None

            self._experience_ledger.commit(
                context=query.text[:500],
                graph_hash=graph_hash,
                graph_node_count=graph_node_count,
                actions=actions,
                snr_score=result.snr_score,
                ihsan_score=result.ihsan_score,
                snr_ok=result.snr_ok,
                response_summary=response_summary,
            )
            return StepResult(
                name="experience_ledger",
                success=True,
                duration_ms=_elapsed(t0),
                detail=f"committed (nodes={graph_node_count})",
            )
        except Exception as e:
            return StepResult(
                name="experience_ledger",
                success=False,
                duration_ms=_elapsed(t0),
                error=str(e),
            )

    def _step_judgment_observe(self, result: Any) -> StepResult:
        """Step 7: SJE Phase A telemetry observation."""
        t0 = time.perf_counter()
        if self._judgment_telemetry is None:
            return StepResult(
                name="judgment_observe",
                success=True,
                duration_ms=_elapsed(t0),
                detail="skipped (no judgment telemetry)",
            )
        try:
            from core.sovereign.judgment_telemetry import JudgmentVerdict

            if not result.success or (
                result.validated and not result.validation_passed
            ):
                verdict = JudgmentVerdict.FORBID
            elif not result.snr_ok:
                verdict = JudgmentVerdict.DEMOTE
            elif result.ihsan_score >= 0.95:
                verdict = JudgmentVerdict.PROMOTE
            else:
                verdict = JudgmentVerdict.NEUTRAL

            self._judgment_telemetry.observe(verdict)
            return StepResult(
                name="judgment_observe",
                success=True,
                duration_ms=_elapsed(t0),
                detail=f"observed ({verdict.name})",
            )
        except Exception as e:
            return StepResult(
                name="judgment_observe",
                success=False,
                duration_ms=_elapsed(t0),
                error=str(e),
            )

    def _step_sat_health_check(self, result: Any) -> StepResult:
        """Step 8: SAT ecosystem health check (observational, non-blocking).

        Checks URP Gini coefficient. If above threshold, logs a warning
        but does NOT fail the step — this is an observation for v1.
        """
        t0 = time.perf_counter()
        if self._sat_controller is None:
            return StepResult(
                name="sat_health",
                success=True,
                duration_ms=_elapsed(t0),
                detail="skipped (no SAT controller)",
            )
        try:
            snapshot = self._sat_controller.get_urp_snapshot()
            gini = snapshot.gini_coefficient
            threshold = getattr(self._sat_controller, "config", None)
            gini_threshold = threshold.gini_rebalance_threshold if threshold else 0.45

            if gini > gini_threshold:
                logger.warning(
                    f"SAT: Gini coefficient {gini:.3f} exceeds "
                    f"threshold {gini_threshold} — rebalancing recommended"
                )
                return StepResult(
                    name="sat_health",
                    success=True,  # Observational — always succeeds
                    duration_ms=_elapsed(t0),
                    detail=f"warning: gini={gini:.3f} > {gini_threshold}",
                )

            return StepResult(
                name="sat_health",
                success=True,
                duration_ms=_elapsed(t0),
                detail=f"healthy (gini={gini:.3f})",
            )
        except Exception as e:
            return StepResult(
                name="sat_health",
                success=False,
                duration_ms=_elapsed(t0),
                error=str(e),
            )


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------


def _elapsed(start: float) -> float:
    """Elapsed milliseconds since start."""
    return (time.perf_counter() - start) * 1000
