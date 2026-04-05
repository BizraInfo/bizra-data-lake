"""
Constitutional Gate Policy — Python mirror of bizra-core/src/gate_policy.rs.

Unified enforcement for all constitutional threshold checks (Ihsan, SNR, ADL).

The same threshold check (e.g., ``ihsan < 0.95``) previously triggered 5 different
behaviors across modules. This module provides ONE canonical decision function that
all enforcement surfaces should call.

Standing on Giants: Al-Ghazali (Ihsan as obligation) · Deming (quality at source)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class GatePolicy(Enum):
    """What happens when a constitutional threshold is violated.

    Ordered from most permissive to most restrictive:
    OBSERVE → FLAG → THROTTLE → REJECT
    """

    OBSERVE = "observe"
    FLAG = "flag"
    THROTTLE = "throttle"
    REJECT = "reject"


class GateAction(Enum):
    """The action taken after policy evaluation."""

    ALLOW = "allow"
    ALLOW_WITH_WARNING = "allow_with_warning"
    FLAGGED = "flagged"
    THROTTLED = "throttled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class GateVerdict:
    """The result of applying a gate policy to a score."""

    score: float
    threshold: float
    passed: bool
    policy: GatePolicy
    action: GateAction


def env_gate_policy() -> GatePolicy:
    """Resolve gate policy from ``BIZRA_ENV`` environment variable.

    - ``BIZRA_ENV=prod`` or ``BIZRA_ENV=production`` → REJECT
    - All other values or unset → OBSERVE
    """
    env = os.environ.get("BIZRA_ENV", "").lower()
    if env in ("prod", "production"):
        return GatePolicy.REJECT
    return GatePolicy.OBSERVE


def apply_gate(
    score: float,
    threshold: float,
    policy: GatePolicy | None = None,
) -> GateVerdict:
    """Apply a gate policy to a score/threshold pair.

    This is the ONE function all enforcement surfaces should call.

    If ``policy`` is None, resolves from ``BIZRA_ENV``.

    Args:
        score: The measured quality score.
        threshold: The constitutional floor.
        policy: Enforcement policy. Defaults to env-resolved.

    Returns:
        GateVerdict with the decision.
    """
    if policy is None:
        policy = env_gate_policy()

    passed = score >= threshold
    if passed:
        action = GateAction.ALLOW
    elif policy == GatePolicy.OBSERVE:
        action = GateAction.ALLOW_WITH_WARNING
    elif policy == GatePolicy.FLAG:
        action = GateAction.FLAGGED
    elif policy == GatePolicy.THROTTLE:
        action = GateAction.THROTTLED
    else:
        action = GateAction.REJECTED

    return GateVerdict(
        score=score,
        threshold=threshold,
        passed=passed,
        policy=policy,
        action=action,
    )


# ── Wire 5: Gate Maturation ──────────────────────────────────────────────────


@dataclass
class MaturationThresholds:
    """Cycle counts at which the policy auto-promotes.

    Standing on Giants: Deming (PDCA maturation) · Lamport (safety liveness)
    """

    observe_to_flag: int = 100
    flag_to_throttle: int = 500
    throttle_to_reject: int = 1000


class GateMaturationPolicy:
    """Auto-promoting gate policy that hardens with accumulated evidence.

    Starts at OBSERVE and promotes through the GatePolicy ladder:
      OBSERVE → FLAG → THROTTLE → REJECT

    Each tick() increments the cycle counter. Promotion is monotonic —
    a gate never softens once hardened.
    """

    def __init__(self, thresholds: MaturationThresholds | None = None) -> None:
        self._thresholds = thresholds or MaturationThresholds()
        self._cycle_count = 0
        self._current = GatePolicy.OBSERVE

    def tick(self) -> GatePolicy:
        """Record one cycle and auto-promote if a threshold is crossed."""
        self._cycle_count += 1
        if (
            self._current == GatePolicy.OBSERVE
            and self._cycle_count >= self._thresholds.observe_to_flag
        ):
            self._current = GatePolicy.FLAG
        elif (
            self._current == GatePolicy.FLAG
            and self._cycle_count >= self._thresholds.flag_to_throttle
        ):
            self._current = GatePolicy.THROTTLE
        elif (
            self._current == GatePolicy.THROTTLE
            and self._cycle_count >= self._thresholds.throttle_to_reject
        ):
            self._current = GatePolicy.REJECT
        return self._current

    @property
    def current(self) -> GatePolicy:
        """Current active policy."""
        return self._current

    @property
    def cycle_count(self) -> int:
        """Total cycles recorded."""
        return self._cycle_count

    @property
    def is_mature(self) -> bool:
        """Whether the gate has reached its terminal (REJECT) state."""
        return self._current == GatePolicy.REJECT
