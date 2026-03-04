"""
Thermodynamic Ihsan Gate.

Adds an optional Lyapunov-style stability check on top of thermodynamic
Ihsan scoring while keeping the existing IhsanGate as the authoritative
runtime contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.constitutional.energy_functions import (
    EnergyProfile,
    ThermodynamicEnergySuite,
)
from core.integration.constants import UNIFIED_IHSAN_THRESHOLD


@dataclass(frozen=True)
class ThermodynamicGateDecision:
    approved: bool
    reason: str
    threshold: float
    profile: EnergyProfile
    delta_energy: float | None = None
    lyapunov_bound: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "threshold": self.threshold,
            "temperature": self.profile.temperature,
            "composite_ihsan": self.profile.composite_ihsan,
            "total_energy": self.profile.total_energy,
            "delta_energy": self.delta_energy,
            "lyapunov_bound": self.lyapunov_bound,
            "energies": dict(self.profile.energies),
            "ihsan_dimensions": dict(self.profile.ihsan_dimensions),
        }


class ThermodynamicIhsanGate:
    """
    Thermodynamic evaluator with optional Lyapunov bound.

    Approval conditions:
    1. Composite Ihsan >= threshold
    2. If previous_energy is provided: delta_energy <= temperature * lyapunov_constant
    """

    def __init__(
        self,
        *,
        threshold: float = UNIFIED_IHSAN_THRESHOLD,
        lyapunov_constant: float = 2.0,
        energy_suite: ThermodynamicEnergySuite | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self.lyapunov_constant = float(lyapunov_constant)
        self.energy_suite = energy_suite or ThermodynamicEnergySuite()

    def evaluate(
        self,
        content: str,
        *,
        snr_score: float | None = None,
        query_text: str = "",
        context: Mapping[str, Any] | None = None,
        previous_energy: float | None = None,
        step: int | float = 0,
    ) -> ThermodynamicGateDecision:
        profile = self.energy_suite.compute(
            content=content,
            snr_score=snr_score,
            query_text=query_text,
            context=context,
            step=step,
        )

        delta_energy: float | None = None
        lyapunov_bound: float | None = None
        if previous_energy is not None:
            delta_energy = float(profile.total_energy - previous_energy)
            lyapunov_bound = profile.temperature * self.lyapunov_constant
            if delta_energy > lyapunov_bound:
                return ThermodynamicGateDecision(
                    approved=False,
                    reason=(
                        f"Lyapunov bound exceeded: ΔE={delta_energy:.4f} > "
                        f"T·C={lyapunov_bound:.4f}"
                    ),
                    threshold=self.threshold,
                    profile=profile,
                    delta_energy=delta_energy,
                    lyapunov_bound=lyapunov_bound,
                )

        if profile.composite_ihsan < self.threshold:
            return ThermodynamicGateDecision(
                approved=False,
                reason=(
                    f"Ihsan below threshold: {profile.composite_ihsan:.4f} < "
                    f"{self.threshold:.4f}"
                ),
                threshold=self.threshold,
                profile=profile,
                delta_energy=delta_energy,
                lyapunov_bound=lyapunov_bound,
            )

        return ThermodynamicGateDecision(
            approved=True,
            reason="APPROVED",
            threshold=self.threshold,
            profile=profile,
            delta_energy=delta_energy,
            lyapunov_bound=lyapunov_bound,
        )


__all__ = ["ThermodynamicGateDecision", "ThermodynamicIhsanGate"]
