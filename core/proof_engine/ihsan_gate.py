"""
Ihsan Gate — Fail-closed Excellence Constraint.

If ihsan_score < threshold, output is REJECTED with reason code
IHSAN_BELOW_THRESHOLD. Emits a receipt on both pass and fail.

Components:
- correctness: factual accuracy of the output
- safety: absence of harmful content
- efficiency: resource usage proportionality
- user_benefit: value delivered to the human
- auditability: traceability and reviewability of reasoning
- robustness: resilience under perturbation and stress

Standing on Giants:
- The concept of Ihsan (excellence as obligation) from Islamic ethics
- Constitutional AI (Anthropic, 2022): AI alignment through principles
- Shannon (1948): Quality as measurable, not narrative
- BIZRA Spearpoint PRD SP-005: "fail-closed excellence constraint"
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.integration.constants import IHSAN_WEIGHTS, UNIFIED_IHSAN_THRESHOLD
from core.proof_engine.reason_codes import ReasonCode


_GATE_COMPONENT_KEYS = (
    "correctness",
    "safety",
    "efficiency",
    "user_benefit",
    "auditability",
    "robustness",
)


def _canonical_component_weights(
    dimensions: Optional[list[str]] = None,
) -> Dict[str, float]:
    """
    Derive gate weights from canonical Ihsan weights.

    By default this gate supports 6 dimensions:
    correctness, safety, efficiency, user_benefit, auditability, robustness.
    If a subset of dimensions is passed, weights are projected and normalized
    over that subset to preserve backward compatibility.

    To avoid split-brain drift, those weights are projected from
    `core.integration.constants.IHSAN_WEIGHTS` and normalized.
    """
    dims = dimensions or list(_GATE_COMPONENT_KEYS)
    base = {k: float(IHSAN_WEIGHTS[k]) for k in dims if k in IHSAN_WEIGHTS}
    total = sum(base.values())
    if total <= 0.0:
        # Fail-safe fallback (equal weights) if constants are ever corrupted.
        n = max(len(base), 1)
        return {k: 1.0 / n for k in base}
    return {k: v / total for k, v in base.items()}


@dataclass
class IhsanComponents:
    """Individual components of the Ihsan excellence score."""

    correctness: float = 0.0
    safety: float = 0.0
    efficiency: float = 0.0
    user_benefit: float = 0.0
    auditability: float | None = None
    robustness: float | None = None

    def to_dict(self) -> Dict[str, float]:
        """Serialize Ihsan component scores to dictionary."""
        out: Dict[str, float] = {
            "correctness": self.correctness,
            "safety": self.safety,
            "efficiency": self.efficiency,
            "user_benefit": self.user_benefit,
        }
        if self.auditability is not None:
            out["auditability"] = self.auditability
        if self.robustness is not None:
            out["robustness"] = self.robustness
        return out

    def composite_score(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute weighted composite score.

        Default weights are derived from canonical SOT weights in
        `core.integration.constants.IHSAN_WEIGHTS` and normalized over
        the dimensions present in this component vector.
        """
        components = self.to_dict()
        active_dims = list(components.keys())

        if weights is None:
            w = _canonical_component_weights(active_dims)
        else:
            raw = {k: float(weights.get(k, 0.0)) for k in active_dims}
            total = sum(max(v, 0.0) for v in raw.values())
            if total > 0.0:
                w = {k: max(v, 0.0) / total for k, v in raw.items()}
            else:
                w = _canonical_component_weights(active_dims)

        return sum(w.get(k, 0.0) * v for k, v in components.items())


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
        threshold: float = UNIFIED_IHSAN_THRESHOLD,
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
            if components.auditability is not None and components.auditability < 0.80:
                reason_codes.append("AUDITABILITY_COMPONENT_LOW")
            if components.robustness is not None and components.robustness < 0.80:
                reason_codes.append("ROBUSTNESS_COMPONENT_LOW")

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
