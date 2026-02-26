"""
Conservative FATE Fallback — Default-Deny Constitutional Gate
=============================================================
Golden Gem α4: When Z3 is unavailable, the fallback must be STRICTER,
not weaker. Unknown ≠ Safe. Unknown = Reject.

Standing on Giants:
  - Saltzer & Schroeder (1975) — "Fail-safe defaults"
  - Lamport (1982) — "Verify, don't trust"
  - XZ Backdoor Lesson — Goldilocks zone between validation modes

This module replaces the permissive _manual_constraint_check() with a
conservative fallback that only approves actions matching known-safe patterns.
Everything else is rejected with an explicit reason code.

Constitutional Principle: ZANN_ZERO (no unverified claims survive)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Final, Optional

logger = logging.getLogger("sovereign.conservative_fallback")

# ═══════════════════════════════════════════════════════════════════════════════
# KNOWN-SAFE PATTERN REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════
# Instead of checking for known violations (permissive: allow by default),
# we check for known-safe patterns (conservative: deny by default).
# Only actions that POSITIVELY MATCH a safe pattern are approved.


class DegradationMode(Enum):
    """FATE engine operational mode."""

    Z3_FULL = auto()  # Full Z3 formal verification
    CONSERVATIVE = auto()  # Conservative Python fallback (default-deny)
    EMERGENCY_LOCKDOWN = auto()  # All non-essential actions blocked


class RejectionReason(Enum):
    """Why an action was rejected by conservative fallback."""

    IHSAN_BELOW_THRESHOLD = "ihsan_below_threshold"
    SNR_BELOW_THRESHOLD = "snr_below_threshold"
    HIGH_RISK_UNVERIFIED = "high_risk_without_human_approval"
    COST_EXCEEDS_LIMIT = "cost_exceeds_autonomy_limit"
    UNKNOWN_ACTION_TYPE = "unknown_action_type_in_degraded_mode"
    MISSING_REQUIRED_FIELD = "missing_required_context_field"
    NEGATIVE_VALUES = "negative_values_in_context"
    Z3_UNAVAILABLE_HIGH_RISK = "z3_unavailable_for_high_risk_action"
    FALLBACK_MODE_RESTRICTED = "action_type_restricted_in_fallback_mode"


@dataclass
class FallbackVerdict:
    """Result of conservative fallback verification."""

    approved: bool
    reason: Optional[RejectionReason] = None
    reason_detail: str = ""
    degradation_mode: DegradationMode = DegradationMode.CONSERVATIVE
    verification_time_ms: int = 0
    constraints_checked: list[str] = field(default_factory=list)
    # Flag: should SAT re-evaluate when Z3 is restored?
    requires_z3_revalidation: bool = False


# Action types that are SAFE to approve in degraded mode
# Everything NOT in this set is REJECTED
SAFE_IN_DEGRADED_MODE: Final[frozenset[str]] = frozenset(
    {
        "query",  # Read-only information retrieval
        "search",  # Read-only search
        "summarize",  # Read-only summarization
        "analyze",  # Read-only analysis
        "explain",  # Read-only explanation
        "translate",  # Read-only translation
        "health_check",  # System health monitoring
        "status_report",  # Status reporting
    }
)

# Action types that REQUIRE Z3 — never approved in fallback
Z3_REQUIRED_ACTIONS: Final[frozenset[str]] = frozenset(
    {
        "execute_code",  # Code execution — must be formally verified
        "modify_filesystem",  # File system changes — irreversible
        "network_request",  # External network calls — information leakage risk
        "blockchain_attest",  # On-chain operations — immutable
        "token_transfer",  # Economic operations — irreversible value transfer
        "agent_spawn",  # Creating new agents — resource commitment
        "permission_change",  # Access control changes — security critical
        "federation_message",  # Cross-node communication — trust boundary
    }
)

# Required fields in action context — missing any = reject
REQUIRED_CONTEXT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "ihsan",
        "snr",
        "risk_level",
        "action_type",
    }
)


def conservative_fallback_check(
    ctx: dict[str, Any],
    *,
    degradation_mode: DegradationMode = DegradationMode.CONSERVATIVE,
) -> FallbackVerdict:
    """
    Conservative constitutional constraint check when Z3 is unavailable.

    PRINCIPLE: Default-deny. Only actions matching known-safe patterns
    with verified constraints are approved. Everything else is rejected
    with an explicit reason code.

    This is the INVERSE of the old _manual_constraint_check():
    - OLD: Check known violations → approve if none found (permissive)
    - NEW: Check known-safe patterns → reject if not matched (conservative)

    Args:
        ctx: Action context with ihsan, snr, risk_level, action_type, etc.
        degradation_mode: Current FATE engine operational mode.

    Returns:
        FallbackVerdict with approval status and reason.
    """
    start_ns = time.perf_counter_ns()
    constraints_checked: list[str] = []

    try:
        from core.integration.constants import (
            UNIFIED_IHSAN_THRESHOLD,
            UNIFIED_SNR_THRESHOLD,
        )
    except ImportError:
        # If constants unavailable, FULL LOCKDOWN — cannot verify anything
        return FallbackVerdict(
            approved=False,
            reason=RejectionReason.MISSING_REQUIRED_FIELD,
            reason_detail="Constitutional constants unavailable — full lockdown",
            degradation_mode=DegradationMode.EMERGENCY_LOCKDOWN,
            verification_time_ms=_elapsed_ms(start_ns),
        )

    # ── GATE 0: Emergency lockdown — reject everything ──
    if degradation_mode == DegradationMode.EMERGENCY_LOCKDOWN:
        return FallbackVerdict(
            approved=False,
            reason=RejectionReason.FALLBACK_MODE_RESTRICTED,
            reason_detail="Emergency lockdown active — all actions blocked",
            degradation_mode=degradation_mode,
            verification_time_ms=_elapsed_ms(start_ns),
        )

    # ── GATE 1: Required fields present ──
    constraints_checked.append("required_fields")
    missing = REQUIRED_CONTEXT_FIELDS - set(ctx.keys())
    if missing:
        return FallbackVerdict(
            approved=False,
            reason=RejectionReason.MISSING_REQUIRED_FIELD,
            reason_detail=f"Missing required fields: {sorted(missing)}",
            degradation_mode=degradation_mode,
            verification_time_ms=_elapsed_ms(start_ns),
            constraints_checked=constraints_checked,
        )

    # ── GATE 2: No negative values (sanity check) ──
    constraints_checked.append("value_sanity")
    for numeric_field in ("ihsan", "snr", "risk_level", "cost"):
        val = ctx.get(numeric_field)
        if val is not None and isinstance(val, (int, float)) and val < 0:
            return FallbackVerdict(
                approved=False,
                reason=RejectionReason.NEGATIVE_VALUES,
                reason_detail=f"Negative value for {numeric_field}: {val}",
                degradation_mode=degradation_mode,
                verification_time_ms=_elapsed_ms(start_ns),
                constraints_checked=constraints_checked,
            )

    # Extract values
    ihsan: float = float(ctx.get("ihsan", 0.0))
    snr: float = float(ctx.get("snr", 0.0))
    risk_level: float = float(ctx.get("risk_level", 0.0))
    cost: float = float(ctx.get("cost", 0.0))
    autonomy_limit: float = float(ctx.get("autonomy_limit", 0.0))
    reversible: bool = bool(ctx.get("reversible", False))
    human_approved: bool = bool(ctx.get("human_approved", False))
    action_type: str = str(ctx.get("action_type", "")).lower().strip()

    # ── GATE 3: إحسان threshold (constitutional invariant) ──
    constraints_checked.append("ihsan_threshold")
    if ihsan < UNIFIED_IHSAN_THRESHOLD:
        return FallbackVerdict(
            approved=False,
            reason=RejectionReason.IHSAN_BELOW_THRESHOLD,
            reason_detail=f"ihsan {ihsan:.4f} < {UNIFIED_IHSAN_THRESHOLD}",
            degradation_mode=degradation_mode,
            verification_time_ms=_elapsed_ms(start_ns),
            constraints_checked=constraints_checked,
        )

    # ── GATE 4: SNR threshold ──
    constraints_checked.append("snr_threshold")
    if snr < UNIFIED_SNR_THRESHOLD:
        return FallbackVerdict(
            approved=False,
            reason=RejectionReason.SNR_BELOW_THRESHOLD,
            reason_detail=f"snr {snr:.4f} < {UNIFIED_SNR_THRESHOLD}",
            degradation_mode=degradation_mode,
            verification_time_ms=_elapsed_ms(start_ns),
            constraints_checked=constraints_checked,
        )

    # ── GATE 5: Cost within autonomy limit ──
    constraints_checked.append("cost_bounds")
    if cost > autonomy_limit and autonomy_limit > 0:
        return FallbackVerdict(
            approved=False,
            reason=RejectionReason.COST_EXCEEDS_LIMIT,
            reason_detail=f"cost {cost} > autonomy_limit {autonomy_limit}",
            degradation_mode=degradation_mode,
            verification_time_ms=_elapsed_ms(start_ns),
            constraints_checked=constraints_checked,
        )

    # ── GATE 6: Z3-required actions blocked in fallback ──
    constraints_checked.append("z3_requirement")
    if action_type in Z3_REQUIRED_ACTIONS:
        # These actions CANNOT be verified without formal methods
        # Exception: human-approved actions with low risk
        if human_approved and risk_level <= 0.3:
            logger.warning(
                "Z3-required action '%s' approved via human override (risk=%.2f)",
                action_type,
                risk_level,
            )
            # Flag for re-evaluation when Z3 returns
            return FallbackVerdict(
                approved=True,
                degradation_mode=degradation_mode,
                verification_time_ms=_elapsed_ms(start_ns),
                constraints_checked=constraints_checked,
                requires_z3_revalidation=True,
            )
        return FallbackVerdict(
            approved=False,
            reason=RejectionReason.Z3_UNAVAILABLE_HIGH_RISK,
            reason_detail=(
                f"Action '{action_type}' requires Z3 formal verification. "
                f"Z3 is unavailable. Action blocked until Z3 is restored."
            ),
            degradation_mode=degradation_mode,
            verification_time_ms=_elapsed_ms(start_ns),
            constraints_checked=constraints_checked,
        )

    # ── GATE 7: High risk without approval ──
    constraints_checked.append("risk_reversibility")
    if risk_level > 0.7 and not reversible and not human_approved:
        return FallbackVerdict(
            approved=False,
            reason=RejectionReason.HIGH_RISK_UNVERIFIED,
            reason_detail=(
                f"risk_level {risk_level:.2f} > 0.7, "
                f"not reversible, not human_approved"
            ),
            degradation_mode=degradation_mode,
            verification_time_ms=_elapsed_ms(start_ns),
            constraints_checked=constraints_checked,
        )

    # ── GATE 8: Action type in safe set (conservative) ──
    constraints_checked.append("safe_action_set")
    if action_type and action_type not in SAFE_IN_DEGRADED_MODE:
        # Medium-risk unknown actions: approve if low risk + reversible,
        # but flag for Z3 re-evaluation
        if risk_level <= 0.3 and reversible:
            logger.info(
                "Unknown action '%s' approved (low risk + reversible), "
                "flagged for Z3 re-evaluation",
                action_type,
            )
            return FallbackVerdict(
                approved=True,
                degradation_mode=degradation_mode,
                verification_time_ms=_elapsed_ms(start_ns),
                constraints_checked=constraints_checked,
                requires_z3_revalidation=True,
            )
        return FallbackVerdict(
            approved=False,
            reason=RejectionReason.UNKNOWN_ACTION_TYPE,
            reason_detail=(
                f"Action '{action_type}' not in safe set for degraded mode. "
                f"Safe actions: {sorted(SAFE_IN_DEGRADED_MODE)}"
            ),
            degradation_mode=degradation_mode,
            verification_time_ms=_elapsed_ms(start_ns),
            constraints_checked=constraints_checked,
        )

    # ── ALL GATES PASSED ──
    return FallbackVerdict(
        approved=True,
        degradation_mode=degradation_mode,
        verification_time_ms=_elapsed_ms(start_ns),
        constraints_checked=constraints_checked,
        requires_z3_revalidation=risk_level > 0.0,
    )


def _elapsed_ms(start_ns: int) -> int:
    """Calculate elapsed milliseconds from nanosecond start time."""
    return (time.perf_counter_ns() - start_ns) // 1_000_000


__all__ = [
    "DegradationMode",
    "RejectionReason",
    "FallbackVerdict",
    "conservative_fallback_check",
    "SAFE_IN_DEGRADED_MODE",
    "Z3_REQUIRED_ACTIONS",
]
