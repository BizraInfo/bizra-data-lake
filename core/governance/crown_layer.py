"""
CROWN Layer — Three-Horizon Governance Invariant Enforcement
============================================================

The CROWN (Constitutional Rights Operating With Neutrality) layer
enforces system-wide invariants across three horizons:

  H0 (Ethical):      Ihsan floor, Gini ceiling, no riba/gharar
  H1 (Performance):  Latency SLA, SNR thresholds, timeout bounds
  H2 (Safety):       Reversibility, human override, audit trail

Every horizon can independently PASS, WARN, or HALT the system.
A HALT on ANY horizon stops all non-diagnostic operations.

Standing on Giants:
- Rawls (1971): Justice as foundational constraint (H0)
- Deming (1950): Statistical process control (H1)
- Lamport (1982): Safety and liveness properties (H2)
- Al-Ghazali (1095): Ihsan as excellence beyond compliance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from core.integration.constants import (
    ADL_GINI_THRESHOLD,
    UNIFIED_AGENT_TIMEOUT_MS,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)


class CROWNHorizon(Enum):
    """Three horizons of system governance."""

    H0_ETHICAL = "H0_ETHICAL"
    H1_PERFORMANCE = "H1_PERFORMANCE"
    H2_SAFETY = "H2_SAFETY"


class CROWNStatus(Enum):
    """Verdict for each horizon audit."""

    PASS = "PASS"
    WARN = "WARN"
    HALT = "HALT"


@dataclass
class HorizonVerdict:
    """Result of auditing a single horizon."""

    horizon: CROWNHorizon
    status: CROWNStatus
    details: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CROWNVerdict:
    """Aggregate result across all three horizons."""

    status: CROWNStatus  # Worst status across horizons
    horizons: list[HorizonVerdict]
    all_green: bool  # True only if all three are PASS

    @property
    def halted(self) -> bool:
        return self.status == CROWNStatus.HALT

    @property
    def warnings(self) -> list[HorizonVerdict]:
        return [h for h in self.horizons if h.status == CROWNStatus.WARN]


@dataclass
class SystemState:
    """Snapshot of system state for CROWN audit.

    Callers populate the fields they have available.
    Missing fields (None) are skipped in the audit.
    """

    ihsan_score: Optional[float] = None
    snr_score: Optional[float] = None
    gini_coefficient: Optional[float] = None
    latency_ms: Optional[float] = None
    has_riba: bool = False
    has_gharar: bool = False
    is_reversible: Optional[bool] = None
    human_override_available: Optional[bool] = None
    has_audit_trail: Optional[bool] = None


# Severity ordering for worst-of aggregation
_STATUS_SEVERITY: dict[CROWNStatus, int] = {
    CROWNStatus.PASS: 0,
    CROWNStatus.WARN: 1,
    CROWNStatus.HALT: 2,
}


class CROWNLayer:
    """Three-horizon governance auditor."""

    def __init__(
        self,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        snr_threshold: float = UNIFIED_SNR_THRESHOLD,
        gini_threshold: float = ADL_GINI_THRESHOLD,
        latency_bound_ms: float = UNIFIED_AGENT_TIMEOUT_MS,
    ) -> None:
        self._ihsan_threshold = ihsan_threshold
        self._snr_threshold = snr_threshold
        self._gini_threshold = gini_threshold
        self._latency_bound_ms = latency_bound_ms

    def render_verdict(self, state: SystemState) -> CROWNVerdict:
        """Audit system state across all three horizons."""
        h0 = self._audit_h0_ethical(state)
        h1 = self._audit_h1_performance(state)
        h2 = self._audit_h2_safety(state)

        horizons = [h0, h1, h2]
        worst = max(horizons, key=lambda h: _STATUS_SEVERITY[h.status])
        all_green = all(h.status == CROWNStatus.PASS for h in horizons)

        return CROWNVerdict(
            status=worst.status,
            horizons=horizons,
            all_green=all_green,
        )

    def _audit_h0_ethical(self, state: SystemState) -> HorizonVerdict:
        """H0: Ethical invariants -- riba, gharar, Ihsan floor, Gini ceiling."""
        # Hard HALT: riba or gharar detected
        if state.has_riba:
            return HorizonVerdict(
                horizon=CROWNHorizon.H0_ETHICAL,
                status=CROWNStatus.HALT,
                details="Riba (interest-based debt) detected",
                metrics={"has_riba": True},
            )
        if state.has_gharar:
            return HorizonVerdict(
                horizon=CROWNHorizon.H0_ETHICAL,
                status=CROWNStatus.HALT,
                details="Gharar (excessive uncertainty) detected",
                metrics={"has_gharar": True},
            )

        status = CROWNStatus.PASS
        details_parts: list[str] = []
        metrics: dict[str, Any] = {}

        # Ihsan floor check
        if state.ihsan_score is not None:
            metrics["ihsan_score"] = state.ihsan_score
            if state.ihsan_score < self._ihsan_threshold:
                status = CROWNStatus.HALT
                details_parts.append(
                    f"Ihsan {state.ihsan_score:.4f} < {self._ihsan_threshold}"
                )

        # Gini ceiling check
        if state.gini_coefficient is not None:
            metrics["gini_coefficient"] = state.gini_coefficient
            if state.gini_coefficient > self._gini_threshold:
                status = CROWNStatus.HALT
                details_parts.append(
                    f"Gini {state.gini_coefficient:.4f} > {self._gini_threshold}"
                )

        return HorizonVerdict(
            horizon=CROWNHorizon.H0_ETHICAL,
            status=status,
            details=(
                "; ".join(details_parts)
                if details_parts
                else "All ethical invariants hold"
            ),
            metrics=metrics,
        )

    def _audit_h1_performance(self, state: SystemState) -> HorizonVerdict:
        """H1: Performance invariants -- latency SLA, SNR thresholds."""
        metrics: dict[str, Any] = {}
        status = CROWNStatus.PASS
        details_parts: list[str] = []

        # Latency bound check
        if state.latency_ms is not None:
            metrics["latency_ms"] = state.latency_ms
            if state.latency_ms > self._latency_bound_ms:
                status = CROWNStatus.WARN
                details_parts.append(
                    f"Latency {state.latency_ms:.0f}ms > {self._latency_bound_ms:.0f}ms SLA"
                )

        # SNR threshold check
        if state.snr_score is not None:
            metrics["snr_score"] = state.snr_score
            if state.snr_score < self._snr_threshold:
                status = CROWNStatus.WARN
                details_parts.append(
                    f"SNR {state.snr_score:.4f} < {self._snr_threshold}"
                )

        return HorizonVerdict(
            horizon=CROWNHorizon.H1_PERFORMANCE,
            status=status,
            details=(
                "; ".join(details_parts)
                if details_parts
                else "All performance SLAs met"
            ),
            metrics=metrics,
        )

    def _audit_h2_safety(self, state: SystemState) -> HorizonVerdict:
        """H2: Safety invariants -- reversibility, human override, audit trail."""
        metrics: dict[str, Any] = {}
        status = CROWNStatus.PASS
        details_parts: list[str] = []

        # Reversibility check
        if state.is_reversible is not None:
            metrics["is_reversible"] = state.is_reversible
            if not state.is_reversible:
                status = CROWNStatus.HALT
                details_parts.append("Irreversible action without safety confirmation")

        # Human override availability
        if state.human_override_available is not None:
            metrics["human_override_available"] = state.human_override_available
            if not state.human_override_available:
                # WARN, not HALT -- human override should be available but
                # absence doesn't require immediate stop
                if status != CROWNStatus.HALT:
                    status = CROWNStatus.WARN
                details_parts.append("Human override not available")

        # Audit trail check
        if state.has_audit_trail is not None:
            metrics["has_audit_trail"] = state.has_audit_trail
            if not state.has_audit_trail:
                if status != CROWNStatus.HALT:
                    status = CROWNStatus.WARN
                details_parts.append("No audit trail for this operation")

        return HorizonVerdict(
            horizon=CROWNHorizon.H2_SAFETY,
            status=status,
            details=(
                "; ".join(details_parts)
                if details_parts
                else "All safety invariants hold"
            ),
            metrics=metrics,
        )
