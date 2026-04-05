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
