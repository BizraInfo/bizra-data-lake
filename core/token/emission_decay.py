"""
Logistic Emission Decay — Gini-Gated Token Minting

Gates BLOOM/SEED minting by the current Gini coefficient using
a logistic decay function. As inequality rises, emission rate
drops exponentially, creating natural pressure toward equality.

Math:
    emission(t) = E_max / (1 + exp(k * (G(t) - G_target)))

Where:
    E_max = Maximum emission rate per epoch
    k = Steepness of the logistic curve
    G(t) = Current Gini coefficient
    G_target = Target Gini (0.35)

Properties:
    - At G = G_target: emission = E_max / 2
    - As G → 0: emission → E_max
    - As G → 1: emission → 0
    - Monotonically decreasing in G
    - Never exactly 0 (logistic asymptote)

Standing on Giants:
- Verhulst (1838): Logistic function
- Gini (1912): Inequality measurement
- Nakamoto (2008): Emission scheduling in proof-of-work
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

from core.integration.constants import ADL_GINI_THRESHOLD

logger = logging.getLogger(__name__)

# Default parameters aligned with ADL kernel constants
DEFAULT_EMISSION_MAX = 1000.0  # Maximum SEED emission per epoch
DEFAULT_GINI_TARGET = ADL_GINI_THRESHOLD  # Target Gini coefficient
DEFAULT_STEEPNESS = 20.0  # Logistic curve steepness


def compute_logistic_emission(
    gini: float,
    e_max: float = DEFAULT_EMISSION_MAX,
    g_target: float = DEFAULT_GINI_TARGET,
    steepness: float = DEFAULT_STEEPNESS,
) -> float:
    """
    Compute emission rate using logistic decay gated by Gini coefficient.

    Args:
        gini: Current Gini coefficient [0, 1]
        e_max: Maximum emission rate
        g_target: Target Gini threshold
        steepness: Logistic curve steepness (k)

    Returns:
        Emission rate [0, E_max]
    """
    exponent = steepness * (gini - g_target)

    # Clamp to avoid overflow
    if exponent > 500:
        return e_max / (1.0 + math.exp(500))
    if exponent < -500:
        return e_max

    return e_max / (1.0 + math.exp(exponent))


def _calculate_gini(values: List[float]) -> float:
    """
    Calculate Gini coefficient from a list of values.

    Uses the relative mean absolute difference formula:
    G = (sum |x_i - x_j|) / (2 * n * mean)
    """
    filtered = [v for v in values if v > 0]
    n = len(filtered)
    if n <= 1:
        return 0.0

    filtered.sort()
    total = sum(filtered)
    mean = total / n

    if mean <= 0:
        return 0.0

    # Efficient O(n log n) formula using sorted values
    numerator = sum((2 * i - n - 1) * x for i, x in enumerate(filtered, 1))
    gini = numerator / (n * total)

    return max(0.0, min(1.0, gini))


class LogisticEmissionGate:
    """
    Emission gate that applies logistic decay based on current Gini.

    Integrates with the TokenMinter to gate minting operations.
    When Gini is low (fair distribution), full emission is allowed.
    When Gini is high (unequal distribution), emission is throttled.
    """

    def __init__(
        self,
        e_max: float = DEFAULT_EMISSION_MAX,
        g_target: float = DEFAULT_GINI_TARGET,
        steepness: float = DEFAULT_STEEPNESS,
    ):
        self._e_max = e_max
        self._g_target = g_target
        self._steepness = steepness

    @property
    def e_max(self) -> float:
        return self._e_max

    @property
    def g_target(self) -> float:
        return self._g_target

    def compute_gated_emission(
        self,
        requested_amount: float,
        current_holdings: List[float],
    ) -> Dict[str, Any]:
        """
        Compute the gated emission amount based on current holdings.

        Args:
            requested_amount: Amount requested for minting
            current_holdings: List of all account balances

        Returns:
            Dict with gated amount, gini, and gate status
        """
        if not current_holdings:
            return {
                "requested_amount": requested_amount,
                "gated_amount": requested_amount,
                "emission_rate": self._e_max,
                "gini": 0.0,
                "gate_open": True,
            }

        gini = _calculate_gini(current_holdings)
        emission_rate = compute_logistic_emission(
            gini=gini,
            e_max=self._e_max,
            g_target=self._g_target,
            steepness=self._steepness,
        )

        # Scale the requested amount by the emission rate ratio
        rate_ratio = emission_rate / self._e_max
        gated_amount = requested_amount * rate_ratio

        return {
            "requested_amount": requested_amount,
            "gated_amount": gated_amount,
            "emission_rate": emission_rate,
            "gini": gini,
            "rate_ratio": rate_ratio,
            "gate_open": gini <= self._g_target,
        }


__all__ = [
    "compute_logistic_emission",
    "LogisticEmissionGate",
    "DEFAULT_EMISSION_MAX",
    "DEFAULT_GINI_TARGET",
    "DEFAULT_STEEPNESS",
]
