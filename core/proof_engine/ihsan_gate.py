"""
Ihsan Gate — Fail-closed Excellence Constraint.

If ihsan_score < threshold, output is REJECTED with reason code
IHSAN_BELOW_THRESHOLD. Emits a receipt on both pass and fail.

Components:
- correctness: factual accuracy of the output
- safety: absence of harmful content
- efficiency: resource usage proportionality
- user_benefit: value delivered to the human

Standing on Giants:
- The concept of Ihsan (excellence as obligation) from Islamic ethics
- Constitutional AI (Anthropic, 2022): AI alignment through principles
- Shannon (1948): Quality as measurable, not narrative
- BIZRA Spearpoint PRD SP-005: "fail-closed excellence constraint"
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.proof_engine.reason_codes import ReasonCode


@dataclass
class IhsanComponents:
    """Individual components of the Ihsan excellence score."""

    correctness: float = 0.0
    safety: float = 0.0
    efficiency: float = 0.0
    user_benefit: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Serialize Ihsan component scores to dictionary."""
        return {
            "correctness": self.correctness,
            "safety": self.safety,
            "efficiency": self.efficiency,
            "user_benefit": self.user_benefit,
        }

    def composite_score(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute weighted composite score.

        Default weights: safety=0.35, correctness=0.30, user_benefit=0.20, efficiency=0.15
        Safety is weighted highest because harm is irreversible.
        """
        w = weights or {
            "correctness": 0.30,
            "safety": 0.35,
            "efficiency": 0.15,
            "user_benefit": 0.20,
        }
        return (
            w.get("correctness", 0.3) * self.correctness
            + w.get("safety", 0.35) * self.safety
            + w.get("efficiency", 0.15) * self.efficiency
            + w.get("user_benefit", 0.2) * self.user_benefit
        )


@dataclass
class IhsanResult:
    """Result of an Ihsan gate evaluation."""

    score: float
    threshold: float
    decision: str  # "APPROVED" | "REJECTED"
    components: IhsanComponents
    reason_codes: List[str] = field(default_factory=list)
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Schema-compatible dict for receipt embedding."""
        return {
            "score": self.score,
            "threshold": self.threshold,
            "decision": self.decision,
            "components": self.components.to_dict(),
            "version": self.version,
        }


class IhsanGate:
    """
    Fail-closed Ihsan excellence gate.

    Every output must pass the Ihsan threshold to be APPROVED.
    Below-threshold outputs are REJECTED with machine-readable reason codes.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.threshold = threshold
        self.weights = weights

    def evaluate(
        self,
        components: IhsanComponents,
    ) -> IhsanResult:
        """
        Evaluate Ihsan gate.

        Returns IhsanResult with APPROVED or REJECTED decision.
        Fail-closed: any error → REJECTED.
        """
        score = components.composite_score(self.weights)
        reason_codes: List[str] = []

        if score < self.threshold:
            reason_codes.append(ReasonCode.IHSAN_BELOW_THRESHOLD.value)

            # Identify which components are weak
            if components.safety < 0.90:
                reason_codes.append("SAFETY_COMPONENT_LOW")
            if components.correctness < 0.85:
                reason_codes.append("CORRECTNESS_COMPONENT_LOW")

        decision = "APPROVED" if score >= self.threshold else "REJECTED"

        return IhsanResult(
            score=score,
            threshold=self.threshold,
            decision=decision,
            components=components,
            reason_codes=reason_codes,
        )

    def ihsan_score(
        self,
        components: IhsanComponents,
    ) -> Dict[str, Any]:
        """
        Single authoritative Ihsan scorer — receipt-compatible output shape.

        Returns the canonical dict matching the receipt.ihsan schema:
        {
            "score": float [0,1],
            "threshold": float,
            "decision": "APPROVED"|"REJECTED",
            "components": {"correctness": ..., "safety": ..., ...},
            "version": str,
            "passed": bool,
            "reason_codes": [...],
        }
        """
        result = self.evaluate(components)
        return {
            "score": result.score,
            "threshold": result.threshold,
            "decision": result.decision,
            "components": result.components.to_dict(),
            "version": result.version,
            "passed": result.decision == "APPROVED",
            "reason_codes": result.reason_codes,
        }


class IhsanFloorWatchdog:
    """
    IHSAN_FLOOR invariant enforcer — runtime governance watchdog.

    Tracks consecutive Ihsan failures and triggers graceful degradation
    when the system's ethical score drops below a critical floor.

    Invariant: If consecutive_failures >= max_consecutive_failures,
    the runtime enters DEGRADED mode (no autonomous execution).

    Standing on: Lamport (fail-closed), BIZRA Constitutional Axiom.
    """

    IHSAN_FLOOR = 0.90  # Hard floor — below this is unacceptable

    def __init__(
        self,
        max_consecutive_failures: int = 3,
        floor: float = 0.90,
    ):
        self.max_consecutive_failures = max_consecutive_failures
        self.floor = floor
        self._consecutive_failures = 0
        self._total_evaluations = 0
        self._total_failures = 0
        self._degraded = False

    def record(self, ihsan_score: float) -> bool:
        """
        Record an Ihsan evaluation result.

        Returns True if the system is still healthy, False if degraded.
        """
        self._total_evaluations += 1

        if ihsan_score < self.floor:
            self._consecutive_failures += 1
            self._total_failures += 1
        else:
            self._consecutive_failures = 0

        if self._consecutive_failures >= self.max_consecutive_failures:
            self._degraded = True

        return not self._degraded

    def reset(self) -> None:
        """Reset after human intervention or recovery."""
        self._consecutive_failures = 0
        self._degraded = False

    @property
    def is_degraded(self) -> bool:
        """Whether the system has entered degraded mode due to repeated Ihsan failures."""
        return self._degraded

    @property
    def consecutive_failures(self) -> int:
        """Number of consecutive Ihsan evaluations below the floor."""
        return self._consecutive_failures

    def status(self) -> dict:
        """Return current watchdog state as a dictionary."""
        return {
            "degraded": self._degraded,
            "consecutive_failures": self._consecutive_failures,
            "max_consecutive_failures": self.max_consecutive_failures,
            "total_evaluations": self._total_evaluations,
            "total_failures": self._total_failures,
            "floor": self.floor,
        }


__all__ = [
    "IhsanGate",
    "IhsanResult",
    "IhsanComponents",
    "IhsanFloorWatchdog",
]
