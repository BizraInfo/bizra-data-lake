"""
AutoResearcher — Hypothesis Generation with Evaluator Gate (spearpoint.improve)
================================================================================

Generates improvement hypotheses and gates every claim through AutoEvaluator.
Research NEVER touches CLEAR/guardrails directly — that is AutoEvaluator's
exclusive domain.

Flow:
  Observe system metrics -> Generate hypothesis -> Create experiment plan
  -> Call AutoEvaluator.evaluate() -> Check tier decision
  -> If tier permits: constitutional admission -> Emit receipt -> Return

NO circuit breaker logic here (that's recursive_loop's job).
NO direct CLEAR/guardrail calls (that's AutoEvaluator's exclusive domain).

Standing on Giants:
- Maturana & Varela (Autopoiesis)
- Deming (Continuous Improvement)
- Boyd (1995): OODA Loop
"""

from __future__ import annotations

import fcntl
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from core.autopoiesis.hypothesis_generator import (
    Hypothesis,
    HypothesisGenerator,
    SystemObservation,
    create_hypothesis_generator,
)
from core.proof_engine.evidence_ledger import emit_receipt

from .auto_evaluator import (
    AutoEvaluator,
    EvaluationResult,
    ExperimentDesign,
    Verdict,
)
from .config import SpearpointConfig

logger = logging.getLogger(__name__)

# Lazy import for Sci-Reasoning bridge (optional dependency)
_SCI_BRIDGE_TYPE = None
try:
    from core.bridges.sci_reasoning_bridge import SciReasoningBridge  # noqa: F401

    _SCI_BRIDGE_TYPE = SciReasoningBridge  # type: ignore[assignment]
except ImportError:
    pass


class ResearchOutcome(str, Enum):
    """Outcome of a research cycle."""

    APPROVED = "approved"  # Hypothesis validated and admitted
    REJECTED = "rejected"  # Hypothesis failed evaluation
    INCONCLUSIVE = "inconclusive"  # Needs more evidence
    GATED = "gated"  # Blocked by constitutional gate
    NO_HYPOTHESES = "no_hypotheses"  # No hypotheses generated


@dataclass
class ResearchResult:
    """Complete result of a research cycle."""

    research_id: str
    outcome: ResearchOutcome
    hypothesis: Optional[Hypothesis] = None
    evaluation: Optional[EvaluationResult] = None
    constitutional_status: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    elapsed_ms: float = 0.0
    mission_id: str = ""
    reason: str = ""
    receipt_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "research_id": self.research_id,
            "outcome": self.outcome.value,
            "hypothesis_id": self.hypothesis.id if self.hypothesis else None,
            "hypothesis_category": (
                self.hypothesis.category.value if self.hypothesis else None
            ),
            "evaluation_verdict": (
                self.evaluation.verdict.value if self.evaluation else None
            ),
            "credibility_score": (
                self.evaluation.credibility_score if self.evaluation else None
            ),
            "constitutional_status": self.constitutional_status,
            "timestamp": self.timestamp,
            "elapsed_ms": self.elapsed_ms,
            "mission_id": self.mission_id,
            "reason": self.reason,
            "receipt_hash": self.receipt_hash,
        }


class AutoResearcher:
    """
    Hypothesis generation with evaluator gate (spearpoint.improve).

    Generates improvement hypotheses and routes every claim through
    AutoEvaluator. No direct CLEAR/guardrail access.
    """

    def __init__(
        self,
        evaluator: AutoEvaluator,
        config: Optional[SpearpointConfig] = None,
        hypothesis_generator: Optional[HypothesisGenerator] = None,
        constitutional_gate: Optional[Any] = None,
        sci_reasoning_bridge: Optional[Any] = None,
    ):
        self.config = config or SpearpointConfig()
        self._evaluator = evaluator
        self._constitutional_gate = constitutional_gate

        # Wire hypothesis generator
        self._generator = hypothesis_generator or create_hypothesis_generator(
            memory_path=self.config.hypothesis_memory_path,
            ihsan_threshold=self.config.ihsan_threshold,
            snr_threshold=self.config.snr_threshold,
        )

        # Sci-Reasoning bridge (optional, enables pattern-aware research)
        self._sci_bridge = sci_reasoning_bridge

        # Statistics
        self._total_cycles = 0
        self._outcomes: dict[str, int] = {o.value: 0 for o in ResearchOutcome}

    def research(
        self,
        observation: Optional[SystemObservation] = None,
        mission_id: str = "",
        top_k: int = 3,
    ) -> list[ResearchResult]:
        """
        Run a research cycle: observe -> hypothesize -> evaluate -> admit.

        Args:
            observation: System state snapshot (defaults if None)
            mission_id: Parent mission ID for provenance
            top_k: Maximum hypotheses to evaluate per cycle

        Returns:
            List of ResearchResult for each hypothesis evaluated
        """
        import time

        start = time.perf_counter()
        results: list[ResearchResult] = []

        # Step 1: Observe (use provided or default observation)
        obs = observation or SystemObservation()

        # Step 2: Generate hypotheses
        hypotheses = self._generator.generate(obs)
        ranked = self._generator.rank_hypotheses(hypotheses, top_k=top_k)

        if not ranked:
            result = ResearchResult(
                research_id=f"res_{uuid.uuid4().hex[:12]}",
                outcome=ResearchOutcome.NO_HYPOTHESES,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                mission_id=mission_id,
                reason="No improvement hypotheses generated from observation",
            )
            results.append(self._finalize_result(result))
            self._outcomes[ResearchOutcome.NO_HYPOTHESES.value] += 1
            self._total_cycles += 1
            return results

        # Step 3: Evaluate each hypothesis through AutoEvaluator
        for hypothesis in ranked:
            result = self._evaluate_hypothesis(hypothesis, mission_id)
            results.append(result)

            # Learn from outcome
            success = result.outcome == ResearchOutcome.APPROVED
            self._generator.learn_from_outcome(hypothesis, success=success)

        self._total_cycles += 1
        return results

    def _evaluate_hypothesis(
        self,
        hypothesis: Hypothesis,
        mission_id: str,
    ) -> ResearchResult:
        """
        Evaluate a single hypothesis through the evaluator gate.

        1. Build experiment design from hypothesis
        2. Call AutoEvaluator.evaluate() (sole gateway)
        3. Check tier decision
        4. If tier permits: run constitutional admission
        5. Return result
        """
        import time

        start = time.perf_counter()
        research_id = f"res_{uuid.uuid4().hex[:12]}"

        # Build experiment design from hypothesis
        experiment = ExperimentDesign(
            claim=hypothesis.description,
            proposed_change="\n".join(hypothesis.implementation_plan),
            metrics_to_measure=list(hypothesis.predicted_improvement.keys()),
            acceptance_criteria={
                "ihsan": self.config.ihsan_threshold,
                "confidence": hypothesis.confidence,
            },
        )

        # Gate through AutoEvaluator (sole pathway to CLEAR/guardrails)
        evaluation = self._evaluator.evaluate(
            claim=hypothesis.description,
            proposed_change="\n".join(hypothesis.implementation_plan),
            experiment_design=experiment,
            mission_id=mission_id,
            response=hypothesis.description,
            metrics={
                "accuracy": hypothesis.confidence,
                "task_completion": 0.5,
                "goal_achievement": max(0.0, hypothesis.expected_value()),
            },
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Check evaluation verdict
        if evaluation.verdict == Verdict.REJECTED:
            self._outcomes[ResearchOutcome.REJECTED.value] += 1
            result = ResearchResult(
                research_id=research_id,
                outcome=ResearchOutcome.REJECTED,
                hypothesis=hypothesis,
                evaluation=evaluation,
                elapsed_ms=elapsed_ms,
                mission_id=mission_id,
                reason=f"Evaluation rejected: {evaluation.reason_codes}",
            )
            return self._finalize_result(result)

        if evaluation.verdict == Verdict.INCONCLUSIVE:
            self._outcomes[ResearchOutcome.INCONCLUSIVE.value] += 1
            result = ResearchResult(
                research_id=research_id,
                outcome=ResearchOutcome.INCONCLUSIVE,
                hypothesis=hypothesis,
                evaluation=evaluation,
                elapsed_ms=elapsed_ms,
                mission_id=mission_id,
                reason="Evaluation inconclusive: diagnostics-only tier",
            )
            return self._finalize_result(result)

        # Tier permits action — check constitutional gate if available
        constitutional_status = "not_configured"
        if self._constitutional_gate is not None:
            constitutional_status = self._run_constitutional_gate(
                hypothesis, evaluation
            )
            if constitutional_status != "admitted":
                self._outcomes[ResearchOutcome.GATED.value] += 1
                result = ResearchResult(
                    research_id=research_id,
                    outcome=ResearchOutcome.GATED,
                    hypothesis=hypothesis,
                    evaluation=evaluation,
                    constitutional_status=constitutional_status,
                    elapsed_ms=elapsed_ms,
                    mission_id=mission_id,
                    reason=f"Constitutional gate: {constitutional_status}",
                )
                return self._finalize_result(result)

        # Approved
        self._outcomes[ResearchOutcome.APPROVED.value] += 1
        result = ResearchResult(
            research_id=research_id,
            outcome=ResearchOutcome.APPROVED,
            hypothesis=hypothesis,
            evaluation=evaluation,
            constitutional_status=constitutional_status,
            elapsed_ms=elapsed_ms,
            mission_id=mission_id,
            reason="Hypothesis validated and admitted",
        )
        return self._finalize_result(result)

    def _run_constitutional_gate(
        self,
        hypothesis: Hypothesis,
        evaluation: EvaluationResult,
    ) -> str:
        """
        Run constitutional admission gate.

        Returns admission status string.
        """
        try:
            import asyncio

            gate = self._constitutional_gate
            candidate = hypothesis.description
            query = f"Hypothesis: {hypothesis.category.value} improvement"

            # Run async admission synchronously if needed
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Already in async context — create task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        gate.admit(candidate, query),
                    ).result(timeout=10)
            else:
                result = asyncio.run(gate.admit(candidate, query))

            return result.status.value

        except Exception as e:
            logger.warning(f"Constitutional gate error: {e}")
            return "error"

    def research_single(
        self,
        claim: str,
        mission_id: str = "",
    ) -> ResearchResult:
        """
        Evaluate a single specific claim (convenience method).

        Builds a hypothesis from the claim and evaluates it through
        the standard pipeline.
        """
        import time

        start = time.perf_counter()
        research_id = f"res_{uuid.uuid4().hex[:12]}"

        experiment = ExperimentDesign(
            claim=claim,
            proposed_change="Verify claim accuracy",
            metrics_to_measure=["accuracy", "ihsan"],
            acceptance_criteria={"ihsan": self.config.ihsan_threshold},
        )

        evaluation = self._evaluator.evaluate(
            claim=claim,
            experiment_design=experiment,
            mission_id=mission_id,
            response=claim,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        outcome_map = {
            Verdict.SUPPORTED: ResearchOutcome.APPROVED,
            Verdict.REJECTED: ResearchOutcome.REJECTED,
            Verdict.INCONCLUSIVE: ResearchOutcome.INCONCLUSIVE,
        }

        outcome = outcome_map[evaluation.verdict]
        self._outcomes[outcome.value] += 1
        self._total_cycles += 1

        result = ResearchResult(
            research_id=research_id,
            outcome=outcome,
            evaluation=evaluation,
            elapsed_ms=elapsed_ms,
            mission_id=mission_id,
            reason=f"Single claim evaluation: {evaluation.verdict.value}",
        )
        return self._finalize_result(result)

    def _finalize_result(self, result: ResearchResult) -> ResearchResult:
        """Attach tamper-evident receipt metadata to every research outcome."""
        result.receipt_hash = self._emit_research_receipt(result)
        return result

    def _emit_research_receipt(self, result: ResearchResult) -> str:
        """Emit a receipt for researcher-level outcomes (innovation audit trail)."""
        status_map = {
            ResearchOutcome.APPROVED: "accepted",
            ResearchOutcome.REJECTED: "rejected",
            ResearchOutcome.INCONCLUSIVE: "quarantined",
            ResearchOutcome.GATED: "rejected",
            ResearchOutcome.NO_HYPOTHESES: "quarantined",
        }
        decision_map = {
            ResearchOutcome.APPROVED: "APPROVED",
            ResearchOutcome.REJECTED: "REJECTED",
            ResearchOutcome.INCONCLUSIVE: "QUARANTINED",
            ResearchOutcome.GATED: "REJECTED",
            ResearchOutcome.NO_HYPOTHESES: "QUARANTINED",
        }

        reason_codes: list[str] = []
        if result.outcome == ResearchOutcome.REJECTED:
            reason_codes = (
                result.evaluation.reason_codes
                if result.evaluation and result.evaluation.reason_codes
                else ["HYPOTHESIS_REJECTED"]
            )
        elif result.outcome == ResearchOutcome.INCONCLUSIVE:
            reason_codes = (
                result.evaluation.reason_codes
                if result.evaluation and result.evaluation.reason_codes
                else ["EVIDENCE_DIAGNOSTICS_ONLY"]
            )
        elif result.outcome == ResearchOutcome.GATED:
            reason_codes = ["CONSTITUTIONAL_GATE_BLOCKED"]
        elif result.outcome == ResearchOutcome.NO_HYPOTHESES:
            reason_codes = ["NO_ACTIONABLE_HYPOTHESES"]

        lock_path = self.config.evidence_ledger_path.parent / ".ledger.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    entry = emit_receipt(
                        self._evaluator.ledger,
                        receipt_id=uuid.uuid4().hex,
                        node_id=f"spearpoint-{result.mission_id or 'standalone'}",
                        policy_version="1.0.0",
                        status=status_map[result.outcome],
                        decision=decision_map[result.outcome],
                        reason_codes=reason_codes,
                        snr_score=(
                            result.evaluation.clear_score if result.evaluation else 0.0
                        ),
                        ihsan_score=(
                            result.evaluation.ihsan_score if result.evaluation else 0.0
                        ),
                        ihsan_threshold=self.config.ihsan_threshold,
                        gate_passed=(
                            "research_gate"
                            if result.outcome == ResearchOutcome.APPROVED
                            else "research_gate_reject"
                        ),
                        duration_ms=result.elapsed_ms,
                        snr_trace={
                            "mission_id": result.mission_id,
                            "research_id": result.research_id,
                            "outcome": result.outcome.value,
                            "hypothesis_id": (
                                result.hypothesis.id if result.hypothesis else "none"
                            ),
                        },
                    )
                    return entry.entry_hash
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to emit research receipt {result.research_id}: {e}")
            return ""

    def research_with_pattern(
        self,
        pattern_id: str,
        claim_context: str = "",
        mission_id: str = "",
        top_k: int = 3,
        metrics: Optional[dict[str, float]] = None,
        ihsan_components: Optional["IhsanComponents"] = None,
    ) -> list[ResearchResult]:
        """
        Run pattern-aware research using Sci-Reasoning thinking patterns.

        Uses the 15 cognitive moves identified by Li et al. (2025) to
        seed hypothesis generation with proven innovation strategies.

        Standing on: Li et al. (Sci-Reasoning), Boyd (OODA Orient phase)

        Args:
            pattern_id: Sci-Reasoning pattern ID (e.g., "P01", "P02")
            claim_context: Additional context for the research claim
            mission_id: Parent mission ID for provenance
            top_k: Max hypotheses to seed from pattern exemplars
            metrics: Optional CLEAR-compatible metrics dict from MetricsProvider
            ihsan_components: Optional IhsanComponents from MetricsProvider

        Returns:
            List of ResearchResult, each backed by a signed receipt
        """
        import time

        start = time.perf_counter()

        if self._sci_bridge is None:
            try:
                from core.bridges.sci_reasoning_bridge import SciReasoningBridge

                self._sci_bridge = SciReasoningBridge()
                self._sci_bridge.load()
            except Exception as e:
                logger.warning(f"Sci-Reasoning bridge unavailable: {e}")
                result = ResearchResult(
                    research_id=f"res_{uuid.uuid4().hex[:12]}",
                    outcome=ResearchOutcome.NO_HYPOTHESES,
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                    mission_id=mission_id,
                    reason="Sci-Reasoning bridge not available",
                )
                return [self._finalize_result(result)]

        # Ensure bridge data loaded
        self._sci_bridge.ensure_loaded()

        # Import pattern types
        from core.bridges.sci_reasoning_patterns import PatternID as PID

        try:
            pid = PID(pattern_id)
        except ValueError:
            result = ResearchResult(
                research_id=f"res_{uuid.uuid4().hex[:12]}",
                outcome=ResearchOutcome.NO_HYPOTHESES,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                mission_id=mission_id,
                reason=f"Unknown pattern ID: {pattern_id}",
            )
            return [self._finalize_result(result)]

        # Get pattern seeds from bridge
        seeds = self._sci_bridge.seed_hypotheses(pid, top_k=top_k)
        pattern_obj = self._sci_bridge.taxonomy.get(pid)

        if not seeds or pattern_obj is None:
            result = ResearchResult(
                research_id=f"res_{uuid.uuid4().hex[:12]}",
                outcome=ResearchOutcome.NO_HYPOTHESES,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                mission_id=mission_id,
                reason=f"No exemplars found for pattern {pattern_id}",
            )
            return [self._finalize_result(result)]

        # Evaluate each seed through the evaluator gate
        results: list[ResearchResult] = []
        for seed in seeds:
            research_id = f"res_{uuid.uuid4().hex[:12]}"

            # Build claim from pattern cognitive move + exemplar
            claim = (
                f"[{pattern_obj.name}] {pattern_obj.cognitive_move}. "
                f"Exemplar: {seed['exemplar_title']} "
                f"({seed['exemplar_conference']} {seed['exemplar_year']}). "
            )
            if claim_context:
                claim += f"Context: {claim_context}"

            # Build experiment design with pattern metadata
            experiment = ExperimentDesign(
                claim=claim,
                proposed_change=seed.get("learnable_insight", "Apply pattern"),
                metrics_to_measure=["ihsan", "snr", "pattern_applicability"],
                acceptance_criteria={
                    "ihsan": self.config.ihsan_threshold,
                    "confidence": 0.7,
                },
            )

            # Gate through evaluator (sole pathway)
            # Use real metrics from MetricsProvider when available,
            # fall back to conservative defaults only as last resort.
            eval_metrics = (
                metrics
                if metrics is not None
                else {
                    "accuracy": 0.7,
                    "task_completion": 0.6,
                    "goal_achievement": 0.5,
                }
            )
            evaluation = self._evaluator.evaluate(
                claim=claim,
                proposed_change=seed.get("learnable_insight", ""),
                experiment_design=experiment,
                mission_id=mission_id,
                response=claim,
                metrics=eval_metrics,
                ihsan_components=ihsan_components,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000

            outcome_map = {
                Verdict.SUPPORTED: ResearchOutcome.APPROVED,
                Verdict.REJECTED: ResearchOutcome.REJECTED,
                Verdict.INCONCLUSIVE: ResearchOutcome.INCONCLUSIVE,
            }
            outcome = outcome_map[evaluation.verdict]
            self._outcomes[outcome.value] += 1

            result = ResearchResult(
                research_id=research_id,
                outcome=outcome,
                evaluation=evaluation,
                elapsed_ms=elapsed_ms,
                mission_id=mission_id,
                reason=(
                    f"Pattern {pattern_id} ({pattern_obj.name}): "
                    f"{evaluation.verdict.value}"
                ),
                receipt_hash="",
            )
            results.append(self._finalize_result(result))

        self._total_cycles += 1
        return results

    def get_statistics(self) -> dict[str, Any]:
        """Get researcher statistics."""
        return {
            "total_cycles": self._total_cycles,
            "outcomes": dict(self._outcomes),
            "generator_stats": self._generator.get_statistics(),
            "evaluator_stats": self._evaluator.get_statistics(),
            "sci_reasoning_available": self._sci_bridge is not None,
        }


__all__ = [
    "AutoResearcher",
    "ResearchResult",
    "ResearchOutcome",
]
