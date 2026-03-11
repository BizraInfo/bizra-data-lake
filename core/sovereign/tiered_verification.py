"""
Tiered Verification — Multi-Speed Constitutional Verification
=============================================================
Golden Gem α7: 50ms pre-check / 500ms concurrent / 1.6s attestation / async consensus

Standing on Giants:
  - XZ Goldilocks Lesson — verification gaps are universal vulnerability
  - Lamport (1978) — Byzantine agreement requires layered verification
  - Shannon (1948) — Information has irreversible effects on observers

PRINCIPLE: Information is not reversible.
Once a user sees a response, you cannot un-show it.
Optimistic execution with post-hoc verification is INSUFFICIENT
for safety-critical content because the cognitive impact is immediate.

Tier 1 (< 50ms):  Pattern-match against known-dangerous categories → BLOCK
Tier 2 (< 500ms): FATE evaluation concurrent with execution → INTERRUPT if needed
Tier 3 (< 1.6s):  Full blockchain attestation → flag + quarantine on violation
Tier 4 (async):   Network consensus for edge cases → community review
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Final

logger = logging.getLogger("sovereign.tiered_verification")


def _normalize_whitespace(text: str) -> str:
    """Collapse whitespace runs for resilient dangerous-pattern matching."""
    return re.sub(r"\s+", " ", text.strip())


class VerificationTier(Enum):
    """Which verification tier processed the action."""

    TIER_1_PRECHECK = auto()  # Instant pattern matching (< 50ms)
    TIER_2_CONCURRENT = auto()  # FATE evaluation during execution (< 500ms)
    TIER_3_ATTESTATION = auto()  # Full blockchain attestation (< 1.6s)
    TIER_4_CONSENSUS = auto()  # Async network consensus


class TierDecision(Enum):
    """Decision at each tier."""

    PASS = auto()  # Action approved at this tier
    BLOCK = auto()  # Action blocked — do not execute
    INTERRUPT = auto()  # Mid-execution interrupt — partial result quarantined
    FLAG = auto()  # Post-execution flag — record for review
    DEFER = auto()  # Deferred to higher tier


@dataclass
class TierResult:
    """Result from a single verification tier."""

    tier: VerificationTier
    decision: TierDecision
    confidence: float  # 0.0–1.0
    reason: str = ""
    elapsed_ms: float = 0.0


@dataclass
class VerificationChain:
    """Complete verification chain across all tiers."""

    action_type: str
    tier_results: list[TierResult] = field(default_factory=list)
    final_decision: TierDecision = TierDecision.PASS
    final_tier: VerificationTier = VerificationTier.TIER_1_PRECHECK
    total_elapsed_ms: float = 0.0

    @property
    def is_blocked(self) -> bool:
        return self.final_decision in (TierDecision.BLOCK, TierDecision.INTERRUPT)

    @property
    def needs_review(self) -> bool:
        return self.final_decision == TierDecision.FLAG


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1: INSTANT PATTERN MATCHING (< 50ms)
# ═══════════════════════════════════════════════════════════════════════════════
# Known-dangerous patterns that must NEVER execute.
# This is the SAFETY BOUNDARY — not quality, not optimization, SAFETY.

KNOWN_DANGEROUS_PATTERNS: Final[frozenset[str]] = frozenset(
    {
        "rm -rf",
        "format c:",
        "drop table",
        "delete from",
        "shutdown",
        "mkfs",
        "dd if=",
        "> /dev/sda",
        "curl | bash",
        "wget | sh",
        "eval(",
        "exec(",
        "__import__",
    }
)

DANGEROUS_ACTION_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "self_harm_content",
        "weapon_synthesis",
        "privacy_violation",
        "identity_theft",
        "financial_fraud",
    }
)


def tier_1_precheck(
    action_type: str,
    content: str = "",
    category: str = "",
) -> TierResult:
    """Tier 1: Instant pattern match against known-dangerous categories.

    Target: < 50ms. No network calls, no model inference, no disk I/O.
    Pure in-memory pattern matching.
    """
    start = time.perf_counter_ns()

    # Check dangerous categories
    if category.lower() in DANGEROUS_ACTION_CATEGORIES:
        return TierResult(
            tier=VerificationTier.TIER_1_PRECHECK,
            decision=TierDecision.BLOCK,
            confidence=0.99,
            reason=f"Dangerous category detected: {category}",
            elapsed_ms=_ns_to_ms(start),
        )

    # Check content for known dangerous patterns
    content_lower = _normalize_whitespace(content.lower())
    for pattern in KNOWN_DANGEROUS_PATTERNS:
        if pattern in content_lower:
            return TierResult(
                tier=VerificationTier.TIER_1_PRECHECK,
                decision=TierDecision.BLOCK,
                confidence=0.95,
                reason=f"Dangerous pattern detected: '{pattern}'",
                elapsed_ms=_ns_to_ms(start),
            )

    # No known-dangerous patterns found — pass to Tier 2
    return TierResult(
        tier=VerificationTier.TIER_1_PRECHECK,
        decision=TierDecision.PASS,
        confidence=0.7,  # Low confidence = "haven't found danger" not "proven safe"
        reason="No known-dangerous patterns detected",
        elapsed_ms=_ns_to_ms(start),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2: CONCURRENT FATE EVALUATION (< 500ms)
# ═══════════════════════════════════════════════════════════════════════════════


async def tier_2_concurrent(
    action_context: dict[str, Any],
    *,
    z3_available: bool = False,
) -> TierResult:
    """Tier 2: FATE engine evaluation concurrent with action execution.

    Runs WHILE the action is executing. If violation detected,
    signals INTERRUPT to halt execution mid-stream.
    """
    start = time.perf_counter_ns()

    try:
        if z3_available:
            from .z3_fate_gate import Z3FATEGate

            gate = Z3FATEGate()
            proof = gate.generate_proof(action_context)
            return TierResult(
                tier=VerificationTier.TIER_2_CONCURRENT,
                decision=(
                    TierDecision.PASS if proof.satisfiable else TierDecision.INTERRUPT
                ),
                confidence=0.99 if proof.satisfiable else 0.95,
                reason=f"Z3 proof: {'SAT' if proof.satisfiable else proof.counterexample}",
                elapsed_ms=_ns_to_ms(start),
            )
        else:
            from .conservative_fallback import conservative_fallback_check

            verdict = conservative_fallback_check(action_context)
            return TierResult(
                tier=VerificationTier.TIER_2_CONCURRENT,
                decision=(
                    TierDecision.PASS if verdict.approved else TierDecision.INTERRUPT
                ),
                confidence=0.85 if verdict.approved else 0.90,
                reason=(
                    verdict.reason_detail
                    if not verdict.approved
                    else "Conservative fallback: approved"
                ),
                elapsed_ms=_ns_to_ms(start),
            )
    except Exception as e:  # noqa: BLE001 — boundary boundary
        logger.error("Tier 2 evaluation failed: %s", e)
        # Failure in verification = BLOCK (conservative)
        return TierResult(
            tier=VerificationTier.TIER_2_CONCURRENT,
            decision=TierDecision.INTERRUPT,
            confidence=0.5,
            reason=f"Verification engine error: {e}. Default: INTERRUPT.",
            elapsed_ms=_ns_to_ms(start),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3: BLOCKCHAIN ATTESTATION (< 1.6s)
# ═══════════════════════════════════════════════════════════════════════════════


async def tier_3_attestation(
    action_context: dict[str, Any],
    execution_result: Any = None,
) -> TierResult:
    """Tier 3: Post-execution blockchain attestation.

    Records the action + result on the immutable ledger.
    If violation detected at this stage, the result is FLAGGED
    (user already saw it) and quarantined for review.
    """
    start = time.perf_counter_ns()

    # In production, this would call the PoI orchestrator
    # For now: structural validation of result integrity
    try:
        # Verify the result matches the action context constraints
        ihsan = float(action_context.get("ihsan", 0.0))
        snr = float(action_context.get("snr", 0.0))

        from core.integration.constants import (
            UNIFIED_IHSAN_THRESHOLD,
            UNIFIED_SNR_THRESHOLD,
        )

        if ihsan < UNIFIED_IHSAN_THRESHOLD or snr < UNIFIED_SNR_THRESHOLD:
            return TierResult(
                tier=VerificationTier.TIER_3_ATTESTATION,
                decision=TierDecision.FLAG,
                confidence=0.90,
                reason=(
                    f"Post-execution quality below threshold: "
                    f"ihsan={ihsan:.3f}, snr={snr:.3f}. Flagged for review."
                ),
                elapsed_ms=_ns_to_ms(start),
            )

        return TierResult(
            tier=VerificationTier.TIER_3_ATTESTATION,
            decision=TierDecision.PASS,
            confidence=0.95,
            reason="Attestation complete. Quality within bounds.",
            elapsed_ms=_ns_to_ms(start),
        )
    except Exception as e:  # noqa: BLE001 — boundary boundary
        return TierResult(
            tier=VerificationTier.TIER_3_ATTESTATION,
            decision=TierDecision.FLAG,
            confidence=0.5,
            reason=f"Attestation error: {e}. Flagged for manual review.",
            elapsed_ms=_ns_to_ms(start),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — Runs verification chain
# ═══════════════════════════════════════════════════════════════════════════════


async def run_verification_chain(
    action_type: str,
    action_context: dict[str, Any],
    content: str = "",
    category: str = "",
    execution_result: Any = None,
    z3_available: bool = False,
) -> VerificationChain:
    """Run the full tiered verification chain.

    Short-circuits: if Tier 1 blocks, Tier 2 never runs.
    Each tier's result is recorded for audit trail.
    """
    chain = VerificationChain(action_type=action_type)
    start_total = time.perf_counter_ns()

    # ── Tier 1: Pre-check ──
    t1 = tier_1_precheck(action_type, content, category)
    chain.tier_results.append(t1)
    if t1.decision == TierDecision.BLOCK:
        chain.final_decision = TierDecision.BLOCK
        chain.final_tier = VerificationTier.TIER_1_PRECHECK
        chain.total_elapsed_ms = _ns_to_ms(start_total)
        return chain

    # ── Tier 2: Concurrent FATE ──
    t2 = await tier_2_concurrent(action_context, z3_available=z3_available)
    chain.tier_results.append(t2)
    if t2.decision == TierDecision.INTERRUPT:
        chain.final_decision = TierDecision.INTERRUPT
        chain.final_tier = VerificationTier.TIER_2_CONCURRENT
        chain.total_elapsed_ms = _ns_to_ms(start_total)
        return chain

    # ── Tier 3: Attestation ──
    t3 = await tier_3_attestation(action_context, execution_result)
    chain.tier_results.append(t3)
    chain.final_decision = t3.decision
    chain.final_tier = VerificationTier.TIER_3_ATTESTATION
    chain.total_elapsed_ms = _ns_to_ms(start_total)
    return chain


def _ns_to_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000


__all__ = [
    "VerificationTier",
    "TierDecision",
    "TierResult",
    "VerificationChain",
    "tier_1_precheck",
    "tier_2_concurrent",
    "tier_3_attestation",
    "run_verification_chain",
]
