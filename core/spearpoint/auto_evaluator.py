"""
AutoEvaluator — The Single Truth Engine API (spearpoint.reproduce)
==================================================================

Only this module touches CLEAR + guardrails + Ihsan gate.
Everything else calls through it. No direct evaluation bypass path.

Flow:
  Parse claim -> Design experiment -> Run guardrails -> Execute CLEAR metrics
  -> Compute tier -> Apply tier behavior policy -> Emit signed receipt
  -> Return EvaluationResult

Every verdict (SUPPORTED, REJECTED, INCONCLUSIVE) emits a receipt.
INCONCLUSIVE is an explicit tier (diagnostics-only, no silent promotion).

Standing on Giants:
- HAL (2025): Holistic Agent Leaderboard
- Saltzer & Schroeder (1975): Fail-closed design
- Shannon (1948): Information-theoretic quality measurement
"""

from __future__ import annotations

import fcntl
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from core.benchmark.clear_framework import CLEARFramework
from core.benchmark.guardrails import GuardrailSuite
from core.proof_engine.evidence_ledger import EvidenceLedger, emit_receipt
from core.proof_engine.ihsan_gate import IhsanComponents, IhsanGate

from .config import SpearpointConfig, TierLevel, TierPolicy, resolve_tier

logger = logging.getLogger(__name__)


class Verdict(str, Enum):
    """Evaluation verdict."""

    SUPPORTED = "SUPPORTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class ExperimentDesign:
    """Design for evaluating a claim."""

    claim: str
    proposed_change: str
    metrics_to_measure: list[str]
    acceptance_criteria: dict[str, float]
    experiment_id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:12]}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "claim": self.claim,
            "proposed_change": self.proposed_change,
            "metrics_to_measure": self.metrics_to_measure,
            "acceptance_criteria": self.acceptance_criteria,
        }


@dataclass
class TierDecision:
    """Tier-based behavior decision."""

    tier: TierPolicy
    permitted_actions: list[str]
    restrictions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.to_dict(),
            "permitted_actions": self.permitted_actions,
            "restrictions": self.restrictions,
        }


@dataclass
class EvaluationResult:
    """Complete result of an evaluation."""

    evaluation_id: str
    verdict: Verdict
    credibility_score: float
    tier_decision: TierDecision
    clear_score: float
    ihsan_score: float
    guardrail_summary: dict[str, Any]
    experiment: ExperimentDesign
    receipt_hash: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    reason_codes: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    mission_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "verdict": self.verdict.value,
            "credibility_score": self.credibility_score,
            "tier_decision": self.tier_decision.to_dict(),
            "clear_score": self.clear_score,
            "ihsan_score": self.ihsan_score,
            "guardrail_summary": self.guardrail_summary,
            "experiment": self.experiment.to_dict(),
            "receipt_hash": self.receipt_hash,
            "timestamp": self.timestamp,
            "reason_codes": self.reason_codes,
            "elapsed_ms": self.elapsed_ms,
            "mission_id": self.mission_id,
        }


class AutoEvaluator:
    """
    The single truth engine API for spearpoint.reproduce.

    All evaluation traffic flows through this class. No other module
    may call CLEAR, guardrails, or Ihsan gate directly for spearpoint
    evaluation purposes.
    """

    def __init__(
        self,
        config: Optional[SpearpointConfig] = None,
        ledger: Optional[EvidenceLedger] = None,
    ):
        self.config = config or SpearpointConfig()
        self.config.ensure_dirs()

        # Wire existing evaluation components
        self._clear = CLEARFramework()
        self._guardrails = GuardrailSuite(
            max_cost_usd=self.config.max_cost_usd,
            max_tokens=self.config.max_tokens,
        )
        self._ihsan_gate = IhsanGate(threshold=self.config.ihsan_threshold)

        # Evidence ledger with single-writer safety
        self._ledger = ledger or EvidenceLedger(
            self.config.evidence_ledger_path,
            validate_on_append=True,
        )

        # Statistics
        self._total_evaluations = 0
        self._verdicts: dict[str, int] = {
            Verdict.SUPPORTED.value: 0,
            Verdict.REJECTED.value: 0,
            Verdict.INCONCLUSIVE.value: 0,
        }

    def evaluate(
        self,
        claim: str,
        proposed_change: str = "",
        experiment_design: Optional[ExperimentDesign] = None,
        mission_id: str = "",
        prompt: str = "",
        response: str = "",
        ihsan_components: Optional[IhsanComponents] = None,
        metrics: Optional[dict[str, float]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a claim through the full truth engine pipeline.

        Steps:
          1. Design experiment (or use provided)
          2. Run guardrails (fail-closed)
          3. Compute CLEAR metrics
          4. Evaluate Ihsan gate
          5. Compute credibility score
          6. Resolve tier and apply behavior policy
          7. Emit receipt (EVERY verdict)
          8. Return EvaluationResult

        Args:
            claim: The claim to evaluate
            proposed_change: Description of proposed change
            experiment_design: Optional pre-built experiment design
            mission_id: Parent mission ID for ledger provenance
            prompt: The prompt used (for guardrail checks)
            response: The response to evaluate
            ihsan_components: Pre-computed Ihsan components
            metrics: Pre-computed metrics dict

        Returns:
            EvaluationResult with verdict, tier, and receipt
        """
        start = time.perf_counter()
        eval_id = f"eval_{uuid.uuid4().hex[:12]}"
        reason_codes: list[str] = []

        # Step 1: Design experiment
        experiment = experiment_design or ExperimentDesign(
            claim=claim,
            proposed_change=proposed_change or "Verify claim accuracy",
            metrics_to_measure=["accuracy", "ihsan", "snr"],
            acceptance_criteria={
                "ihsan": self.config.ihsan_threshold,
                "snr": self.config.snr_threshold,
            },
        )

        # Use claim as response if none provided
        eval_text = response or claim

        # Step 2: Run guardrails (fail-closed)
        guardrail_results = self._guardrails.check_all(
            prompt=prompt or claim,
            response=eval_text,
            metrics=metrics or {"accuracy": 0.0, "cost_usd": 0.0, "tokens": 0},
        )
        guardrail_summary = self._guardrails.summarize(guardrail_results)

        if not guardrail_summary["all_passed"]:
            reason_codes.extend(
                f"GUARDRAIL_{g}" for g in guardrail_summary["failed_guardrails"]
            )

        # Step 3: Compute CLEAR metrics
        clear_score = self._compute_clear_score(eval_id, eval_text, metrics or {})

        # Step 4: Evaluate Ihsan gate
        components = ihsan_components or self._estimate_ihsan_components(
            eval_text, guardrail_summary
        )
        ihsan_result = self._ihsan_gate.evaluate(components)
        ihsan_score = ihsan_result.score

        if ihsan_result.decision == "REJECTED":
            reason_codes.extend(ihsan_result.reason_codes)

        # Step 5: Compute credibility score
        credibility = self._compute_credibility(
            clear_score, ihsan_score, guardrail_summary
        )

        # Step 6: Resolve tier and apply behavior policy
        tier = resolve_tier(credibility)
        tier_decision = self._apply_tier_policy(tier, credibility)

        # Step 7: Determine verdict
        verdict = self._determine_verdict(
            credibility, guardrail_summary, tier, reason_codes
        )

        # Ensure reason_codes non-empty for REJECTED/QUARANTINED receipts
        # (schema requires minItems:1 for non-APPROVED decisions)
        if verdict == Verdict.REJECTED and not reason_codes:
            reason_codes.append("EVIDENCE_INSUFFICIENT")
        elif verdict == Verdict.INCONCLUSIVE and not reason_codes:
            reason_codes.append("EVIDENCE_DIAGNOSTICS_ONLY")

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Step 8: Emit receipt (EVERY verdict, including failures)
        receipt_hash = self._emit_receipt(
            eval_id=eval_id,
            verdict=verdict,
            credibility=credibility,
            ihsan_score=ihsan_score,
            clear_score=clear_score,
            tier=tier,
            mission_id=mission_id,
            reason_codes=reason_codes,
            elapsed_ms=elapsed_ms,
        )

        # Update statistics
        self._total_evaluations += 1
        self._verdicts[verdict.value] += 1

        return EvaluationResult(
            evaluation_id=eval_id,
            verdict=verdict,
            credibility_score=credibility,
            tier_decision=tier_decision,
            clear_score=clear_score,
            ihsan_score=ihsan_score,
            guardrail_summary=guardrail_summary,
            experiment=experiment,
            receipt_hash=receipt_hash,
            reason_codes=reason_codes,
            elapsed_ms=elapsed_ms,
            mission_id=mission_id,
        )

    def _compute_clear_score(
        self,
        eval_id: str,
        text: str,
        metrics: dict[str, float],
    ) -> float:
        """Compute CLEAR score using the framework."""
        with self._clear.evaluate(eval_id, "spearpoint-evaluator") as ctx:
            ctx.record_efficacy(
                accuracy=metrics.get("accuracy", 0.5),
                task_completion=metrics.get("task_completion", 0.5),
                goal_achievement=metrics.get("goal_achievement", 0.5),
            )
            ctx.record_cost(
                input_tokens=int(metrics.get("input_tokens", 0)),
                output_tokens=max(len(text.split()), 1),
                cost_usd=metrics.get("cost_usd", 0.0),
            )
            ctx.record_assurance(
                safety_violations=int(metrics.get("safety_violations", 0)),
                hallucination_rate=metrics.get("hallucination_rate", 0.0),
                reproducibility=metrics.get("reproducibility", 0.5),
            )
            ctx.record_reliability(
                consistency=metrics.get("consistency", 0.5),
                runs_completed=int(metrics.get("runs_completed", 1)),
            )

        clear_metrics = self._clear.get_metrics(eval_id)
        if clear_metrics:
            return clear_metrics.compute_overall_score()
        return 0.0

    def _estimate_ihsan_components(
        self,
        text: str,
        guardrail_summary: dict[str, Any],
    ) -> IhsanComponents:
        """Estimate Ihsan components from text and guardrail results."""
        words = text.split()
        word_count = len(words)
        unique_ratio = len(set(w.lower() for w in words)) / max(word_count, 1)

        safety = (
            1.0
            if guardrail_summary["all_passed"]
            else max(0.5, 1.0 - 0.1 * guardrail_summary.get("failed", 0))
        )

        return IhsanComponents(
            correctness=min(unique_ratio * 1.2, 1.0),
            safety=safety,
            efficiency=min(1.0, max(0.3, 1.0 - word_count / 5000)),
            user_benefit=0.7 if word_count >= 10 else 0.3,
        )

    def _compute_credibility(
        self,
        clear_score: float,
        ihsan_score: float,
        guardrail_summary: dict[str, Any],
    ) -> float:
        """
        Compute overall credibility score.

        Weighted: 40% CLEAR + 40% Ihsan + 20% guardrails
        Fail-closed: any guardrail failure caps credibility below OPERATIONAL.
        """
        guardrail_score = (
            1.0
            if guardrail_summary["all_passed"]
            else max(0.0, 1.0 - 0.15 * guardrail_summary.get("failed", 1))
        )

        credibility = 0.40 * clear_score + 0.40 * ihsan_score + 0.20 * guardrail_score

        # Fail-closed: guardrail failure caps below operational threshold
        if not guardrail_summary["all_passed"]:
            credibility = min(credibility, self.config.ihsan_threshold - 0.01)

        return max(0.0, min(1.0, credibility))

    def _apply_tier_policy(
        self,
        tier: TierPolicy,
        credibility: float,
    ) -> TierDecision:
        """Apply tier behavior policy and compute permitted actions."""
        actions: list[str] = []
        restrictions: list[str] = []

        if tier.level == TierLevel.REJECT:
            restrictions = [
                "output_rejected",
                "must_provide_more_evidence",
                "no_recommendations",
            ]
        elif tier.diagnostics_only:
            actions = ["emit_diagnostics", "log_metrics"]
            restrictions = [
                "no_recommendations",
                "no_patch_proposals",
                "diagnostics_only",
            ]
        else:
            actions.append("emit_recommendations")
            if tier.may_recommend:
                actions.append("recommend_changes")
            if tier.may_propose_patch:
                actions.append("propose_patch_plan")
            if tier.requires_confirmation:
                restrictions.append("requires_human_confirmation")
            if tier.requires_provenance:
                restrictions.append("requires_provenance_chain")

        return TierDecision(
            tier=tier,
            permitted_actions=actions,
            restrictions=restrictions,
        )

    def _determine_verdict(
        self,
        credibility: float,
        guardrail_summary: dict[str, Any],
        tier: TierPolicy,
        reason_codes: list[str],
    ) -> Verdict:
        """
        Determine verdict from credibility and guardrail results.

        Fail-closed:
        - Any guardrail failure -> REJECTED
        - Credibility < 0.85 -> REJECTED
        - 0.85 <= credibility < ihsan_threshold -> INCONCLUSIVE
        - credibility >= ihsan_threshold -> SUPPORTED
        """
        if not guardrail_summary["all_passed"]:
            return Verdict.REJECTED

        if tier.level == TierLevel.REJECT:
            return Verdict.REJECTED

        if tier.level == TierLevel.DIAGNOSTICS:
            return Verdict.INCONCLUSIVE

        return Verdict.SUPPORTED

    def _emit_receipt(
        self,
        eval_id: str,
        verdict: Verdict,
        credibility: float,
        ihsan_score: float,
        clear_score: float,
        tier: TierPolicy,
        mission_id: str,
        reason_codes: list[str],
        elapsed_ms: float,
    ) -> str:
        """Emit a signed receipt to the evidence ledger with file lock safety."""
        decision_map = {
            Verdict.SUPPORTED: "APPROVED",
            Verdict.REJECTED: "REJECTED",
            Verdict.INCONCLUSIVE: "QUARANTINED",
        }

        status_map = {
            Verdict.SUPPORTED: "accepted",
            Verdict.REJECTED: "rejected",
            Verdict.INCONCLUSIVE: "quarantined",
        }

        try:
            # File-lock for single-writer safety
            lock_path = self.config.evidence_ledger_path.parent / ".ledger.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)

            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    # Receipt ID must be hex-only per schema
                    receipt_hex = uuid.uuid4().hex
                    entry = emit_receipt(
                        self._ledger,
                        receipt_id=receipt_hex,
                        node_id=f"spearpoint-{mission_id or 'standalone'}",
                        policy_version="1.0.0",
                        status=status_map[verdict],
                        decision=decision_map[verdict],
                        reason_codes=reason_codes or [],
                        snr_score=clear_score,
                        ihsan_score=ihsan_score,
                        ihsan_threshold=self.config.ihsan_threshold,
                        duration_ms=elapsed_ms,
                        snr_trace={
                            "tier": tier.level.value,
                            "credibility": credibility,
                            "mission_id": mission_id,
                        },
                    )
                    return entry.entry_hash
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)

        except Exception as e:
            logger.error(f"Failed to emit receipt for {eval_id}: {e}")
            return ""

    def get_statistics(self) -> dict[str, Any]:
        """Get evaluator statistics."""
        return {
            "total_evaluations": self._total_evaluations,
            "verdicts": dict(self._verdicts),
            "ledger_entries": self._ledger.count(),
        }

    @property
    def ledger(self) -> EvidenceLedger:
        """Access the evidence ledger (for chain verification)."""
        return self._ledger


__all__ = [
    "AutoEvaluator",
    "EvaluationResult",
    "ExperimentDesign",
    "TierDecision",
    "Verdict",
]
