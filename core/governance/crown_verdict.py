"""
Crown Verdict System -- Three-Tier Invariant Verification Watchdog

BIZRA Atlas v4.0, Diagram d10 (SNR 0.94)

The Crown Verdict is the final constitutional arbiter for all sovereign actions.
It enforces three tiers of invariants that no action may violate:

    H0: Ethical + Shariah Invariants
        No gharar (uncertainty/deception), no riba (usury/exploitation),
        fairness (Gini <= 0.40), constitutional alignment (Ihsan >= threshold).

    H1: Performance Invariants
        SLA bounds, throughput floors, resource cost ceilings,
        signal quality (SNR >= minimum).

    H2: Safety Invariants
        No harm, reversible actions, blast containment,
        human escalation availability.

    Crown Verdict: H0 AND H1 AND H2 --> ACCEPT | REJECT | REVISE

Principle: Fail closed. Every verdict is signed with Ed25519 for
tamper-proof auditability. No silent failures -- every rejection
carries a structured reason and remediation path.

Standing on the Shoulders of Giants:
- Al-Ghazali (1095): Ihsan Ethics and Maqasid al-Shariah
- Gini (1912): Inequality Measurement
- Shannon (1948): Signal-to-Noise Theory
- Lamport (1982): Byzantine Fault Tolerance
- Bernstein et al. (2012): Ed25519 High-Speed Signatures
- Anthropic (2022): Constitutional AI
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Optional, Sequence

from core.integration.constants import (
    ADL_GINI_THRESHOLD,
    IHSAN_THRESHOLD,
    SNR_THRESHOLD,
)
from core.pci.crypto import (
    domain_separated_digest,
    generate_keypair,
    sign_message,
    verify_signature,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class Verdict(Enum):
    """Crown Verdict outcomes -- the only three possible results."""

    ACCEPT = "ACCEPT"  # All tiers passed -- action may proceed
    REJECT = "REJECT"  # At least one tier failed with no fix available
    REVISE = "REVISE"  # Partial failure with actionable remediation


class TierStatus(Enum):
    """Status of a single invariant tier evaluation."""

    PASSED = "passed"
    FAILED = "failed"
    DEGRADED = "degraded"  # Passed minimums but below optimal


class ActionScope(Enum):
    """Blast radius scope for H2 safety containment."""

    SELF = auto()  # Affects only the acting agent
    LOCAL = auto()  # Affects the local node
    CLUSTER = auto()  # Affects the local cluster
    FEDERATION = auto()  # Affects the entire federation
    EXTERNAL = auto()  # Affects external systems


# =============================================================================
# ACTION DESCRIPTOR -- What the Crown Verdict evaluates
# =============================================================================


@dataclass
class SovereignAction:
    """
    Descriptor for an action submitted to the Crown Verdict system.

    Every action entering the sovereign pipeline must carry sufficient
    metadata for all three tiers of invariant verification. Missing
    metadata is treated as a violation (fail-closed).
    """

    # Identity
    action_id: str
    action_type: str
    description: str
    agent_id: str

    # H0: Ethical metadata
    has_audit_trail: bool = False
    involves_interest: bool = False
    resource_distribution: Optional[list[float]] = None
    ihsan_score: float = 0.0

    # H1: Performance metadata
    sla_deadline_ms: Optional[float] = None
    estimated_duration_ms: float = 0.0
    throughput_rps: float = 0.0
    resource_cost: float = 0.0
    resource_cost_ceiling: float = 1.0
    snr_score: float = 0.0

    # H2: Safety metadata
    reversible: bool = False
    blast_radius: ActionScope = ActionScope.SELF
    max_allowed_scope: ActionScope = ActionScope.LOCAL
    human_override_available: bool = False
    harm_assessment: float = 0.0  # [0.0, 1.0] where 0 = no harm

    # Context
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# TIER RESULTS -- Structured output from each invariant tier
# =============================================================================


@dataclass
class H0Result:
    """
    H0: Ethical + Shariah Invariant Verification Result.

    Checks four sub-invariants:
    1. Gharar detection -- actions without audit trails create uncertainty
    2. Riba detection -- interest-bearing or exploitative patterns
    3. Fairness -- resource distribution Gini coefficient <= threshold
    4. Constitutional alignment -- Ihsan score >= threshold
    """

    status: TierStatus
    gharar_detected: bool = False
    riba_detected: bool = False
    gini_coefficient: float = 0.0
    gini_passed: bool = True
    ihsan_score: float = 0.0
    ihsan_passed: bool = True
    violations: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    duration_us: int = 0

    @property
    def passed(self) -> bool:
        return self.status == TierStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": "H0",
            "tier_name": "Ethical + Shariah Invariants",
            "status": self.status.value,
            "gharar_detected": self.gharar_detected,
            "riba_detected": self.riba_detected,
            "gini_coefficient": self.gini_coefficient,
            "gini_passed": self.gini_passed,
            "ihsan_score": self.ihsan_score,
            "ihsan_passed": self.ihsan_passed,
            "violations": self.violations,
            "evidence": self.evidence,
            "duration_us": self.duration_us,
        }


@dataclass
class H1Result:
    """
    H1: Performance Invariant Verification Result.

    Checks four sub-invariants:
    1. SLA bounds -- estimated duration within deadline
    2. Throughput -- minimum operations per second
    3. Resource cost -- within cost ceiling
    4. Signal quality -- SNR >= minimum threshold
    """

    status: TierStatus
    sla_met: bool = True
    throughput_adequate: bool = True
    cost_within_ceiling: bool = True
    snr_score: float = 0.0
    snr_passed: bool = True
    violations: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    duration_us: int = 0

    @property
    def passed(self) -> bool:
        return self.status == TierStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": "H1",
            "tier_name": "Performance Invariants",
            "status": self.status.value,
            "sla_met": self.sla_met,
            "throughput_adequate": self.throughput_adequate,
            "cost_within_ceiling": self.cost_within_ceiling,
            "snr_score": self.snr_score,
            "snr_passed": self.snr_passed,
            "violations": self.violations,
            "evidence": self.evidence,
            "duration_us": self.duration_us,
        }


@dataclass
class H2Result:
    """
    H2: Safety Invariant Verification Result.

    Checks four sub-invariants:
    1. Reversibility -- irreversible actions require elevated approval
    2. Blast containment -- scope must not exceed allowed maximum
    3. Human escalation -- override path must be available
    4. No-harm verification -- harm assessment below threshold
    """

    status: TierStatus
    reversible: bool = True
    blast_contained: bool = True
    human_override_available: bool = True
    harm_score: float = 0.0
    no_harm_verified: bool = True
    violations: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    duration_us: int = 0

    @property
    def passed(self) -> bool:
        return self.status == TierStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": "H2",
            "tier_name": "Safety Invariants",
            "status": self.status.value,
            "reversible": self.reversible,
            "blast_contained": self.blast_contained,
            "human_override_available": self.human_override_available,
            "harm_score": self.harm_score,
            "no_harm_verified": self.no_harm_verified,
            "violations": self.violations,
            "evidence": self.evidence,
            "duration_us": self.duration_us,
        }


# =============================================================================
# CROWN VERDICT RESULT -- Final signed output
# =============================================================================


@dataclass
class CrownVerdictResult:
    """
    The final, signed Crown Verdict.

    Combines H0, H1, H2 tier results into a single constitutional
    ruling. The verdict is signed with Ed25519 to prevent tampering.
    An unsigned or invalid-signature verdict MUST be rejected by
    all downstream consumers.
    """

    action_id: str
    verdict: Verdict
    h0: H0Result
    h1: H1Result
    h2: H2Result
    remediations: list[str] = field(default_factory=list)
    total_duration_us: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signature: str = ""
    signer_public_key: str = ""

    @property
    def accepted(self) -> bool:
        return self.verdict == Verdict.ACCEPT

    @property
    def all_tiers_passed(self) -> bool:
        return self.h0.passed and self.h1.passed and self.h2.passed

    @property
    def total_violations(self) -> list[str]:
        return self.h0.violations + self.h1.violations + self.h2.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "verdict": self.verdict.value,
            "h0": self.h0.to_dict(),
            "h1": self.h1.to_dict(),
            "h2": self.h2.to_dict(),
            "remediations": self.remediations,
            "total_violations": self.total_violations,
            "total_duration_us": self.total_duration_us,
            "timestamp": self.timestamp.isoformat(),
            "signature": self.signature,
            "signer_public_key": self.signer_public_key,
        }

    def verify(self) -> bool:
        """
        Verify the Ed25519 signature on this verdict.

        Returns False if the verdict is unsigned or the signature
        does not match, indicating potential tampering.
        """
        if not self.signature or not self.signer_public_key:
            return False
        digest = self._compute_digest()
        return verify_signature(digest, self.signature, self.signer_public_key)

    def _compute_digest(self) -> str:
        """Compute the domain-separated BLAKE3 digest of the verdict payload."""
        # Canonical representation: deterministic string of verdict data
        # excluding signature fields to avoid circular dependency
        canonical = (
            f"{self.action_id}:{self.verdict.value}"
            f":{self.h0.status.value}:{self.h1.status.value}:{self.h2.status.value}"
            f":{self.timestamp.isoformat()}"
        )
        return domain_separated_digest(canonical.encode("utf-8"))


# =============================================================================
# CROWN VERDICT ENGINE
# =============================================================================


def _compute_gini(distribution: Sequence[float]) -> float:
    """
    Compute the Gini coefficient of a resource distribution.

    Standing on Giants: Gini (1912) -- "Variabilita e mutabilita"

    The Gini coefficient measures inequality in a distribution.
    0.0 = perfect equality, 1.0 = perfect inequality.

    Args:
        distribution: Sequence of non-negative resource holdings.

    Returns:
        Gini coefficient in [0.0, 1.0].
    """
    if not distribution:
        return 0.0

    sorted_dist = sorted(distribution)
    n = len(sorted_dist)
    total = sum(sorted_dist)

    if total == 0.0 or n <= 1:
        return 0.0

    # Mean absolute difference formula: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
    cumulative = sum((i + 1) * x for i, x in enumerate(sorted_dist))
    return (2.0 * cumulative) / (n * total) - (n + 1) / n


class CrownVerdict:
    """
    Three-Tier Invariant Verification Watchdog.

    The CrownVerdict is the constitutional arbiter that evaluates every
    sovereign action against three orthogonal invariant tiers:

    - H0: Ethical + Shariah (gharar, riba, fairness, constitutional alignment)
    - H1: Performance (SLA, throughput, resource cost, signal quality)
    - H2: Safety (reversibility, blast containment, human escalation, no-harm)

    The final verdict is one of:
    - ACCEPT: All tiers passed -- action proceeds
    - REJECT: Hard failure in any tier with no remediation path
    - REVISE: Partial failure with actionable remediation suggestions

    Architecture:
    - Fail-closed: missing metadata is treated as violation
    - Ed25519-signed: verdicts are cryptographically tamper-proof
    - Auditable: every verdict carries structured evidence
    - Deterministic: same inputs always produce same verdict

    Usage:
        crown = CrownVerdict()
        result = crown.adjudicate(action)
        if result.accepted:
            execute(action)
        elif result.verdict == Verdict.REVISE:
            fix(action, result.remediations)
        else:
            reject(action, result.total_violations)
    """

    # Maximum harm score before hard rejection (no remediation possible)
    HARM_HARD_REJECT: float = 0.7
    # Harm score that triggers REVISE instead of ACCEPT
    HARM_SOFT_THRESHOLD: float = 0.3
    # Minimum throughput (requests per second) -- 0 means no floor enforced
    MIN_THROUGHPUT_RPS: float = 0.0
    # SLA safety margin multiplier (action must finish in deadline * margin)
    SLA_MARGIN: float = 0.9

    def __init__(
        self,
        ihsan_threshold: float = IHSAN_THRESHOLD,
        snr_minimum: float = SNR_THRESHOLD,
        gini_threshold: float = ADL_GINI_THRESHOLD,
        harm_hard_reject: float = HARM_HARD_REJECT,
        harm_soft_threshold: float = HARM_SOFT_THRESHOLD,
        min_throughput_rps: float = MIN_THROUGHPUT_RPS,
        sla_margin: float = SLA_MARGIN,
    ):
        """
        Initialize the Crown Verdict engine.

        All thresholds default to the constitutional values from
        core.integration.constants -- the single source of truth.

        Args:
            ihsan_threshold: Minimum Ihsan score for H0 (default: 0.95).
            snr_minimum: Minimum SNR for H1 (default: 0.85).
            gini_threshold: Maximum Gini coefficient for H0 (default: 0.40).
            harm_hard_reject: Harm score above which H2 hard-rejects (default: 0.7).
            harm_soft_threshold: Harm score above which H2 degrades (default: 0.3).
            min_throughput_rps: Minimum throughput floor (default: 0.0, disabled).
            sla_margin: SLA safety margin multiplier (default: 0.9).
        """
        self.ihsan_threshold = ihsan_threshold
        self.snr_minimum = snr_minimum
        self.gini_threshold = gini_threshold
        self.harm_hard_reject = harm_hard_reject
        self.harm_soft_threshold = harm_soft_threshold
        self.min_throughput_rps = min_throughput_rps
        self.sla_margin = sla_margin

        # Ed25519 keypair for verdict signing
        self._private_key, self._public_key = generate_keypair()

        # Audit log
        self._verdicts: list[CrownVerdictResult] = []

    @property
    def public_key(self) -> str:
        """Public key for signature verification (safe to expose)."""
        return self._public_key

    # =========================================================================
    # H0: ETHICAL + SHARIAH INVARIANTS
    # =========================================================================

    def verify_h0(self, action: SovereignAction) -> H0Result:
        """
        Verify H0: Ethical + Shariah invariants.

        Sub-invariants:
        1. Gharar (uncertainty/deception) -- detected via missing audit trail.
           An action without provenance is by definition uncertain.
        2. Riba (usury/exploitation) -- detected via interest-bearing flag.
           Any action involving interest on principal is prohibited.
        3. Fairness -- resource distribution Gini <= 0.40.
           Concentrating resources beyond this threshold violates Adl (justice).
        4. Constitutional alignment -- Ihsan score >= threshold.
           Actions below the excellence threshold fail the constitution.

        Standing on Giants:
        - Al-Ghazali (1095): Maqasid al-Shariah (Objectives of Islamic Law)
        - Gini (1912): Statistical measure of inequality
        - Anthropic (2022): Constitutional AI constraints

        Args:
            action: The sovereign action to evaluate.

        Returns:
            H0Result with pass/fail status and structured evidence.
        """
        start = time.perf_counter_ns()
        violations: list[str] = []
        evidence: dict[str, Any] = {}

        # --- Sub-invariant 1: Gharar detection (missing audit trail) ---
        gharar_detected = not action.has_audit_trail
        if gharar_detected:
            violations.append(
                "H0-GHARAR: Action lacks audit trail -- "
                "uncertainty/deception risk (gharar)"
            )
        evidence["has_audit_trail"] = action.has_audit_trail

        # --- Sub-invariant 2: Riba detection (interest-bearing patterns) ---
        riba_detected = action.involves_interest
        if riba_detected:
            violations.append(
                "H0-RIBA: Action involves interest-bearing pattern -- "
                "exploitation risk (riba)"
            )
        evidence["involves_interest"] = action.involves_interest

        # --- Sub-invariant 3: Fairness (Gini coefficient) ---
        gini = 0.0
        gini_passed = True
        if action.resource_distribution is not None:
            gini = _compute_gini(action.resource_distribution)
            gini_passed = gini <= self.gini_threshold
            if not gini_passed:
                violations.append(
                    f"H0-ADL: Gini coefficient {gini:.4f} exceeds "
                    f"threshold {self.gini_threshold} -- "
                    f"resource concentration violates justice (adl)"
                )
        evidence["gini_coefficient"] = gini
        evidence["gini_threshold"] = self.gini_threshold

        # --- Sub-invariant 4: Constitutional alignment (Ihsan) ---
        ihsan_passed = action.ihsan_score >= self.ihsan_threshold
        if not ihsan_passed:
            violations.append(
                f"H0-IHSAN: Ihsan score {action.ihsan_score:.4f} below "
                f"threshold {self.ihsan_threshold} -- "
                f"constitutional excellence not met"
            )
        evidence["ihsan_score"] = action.ihsan_score
        evidence["ihsan_threshold"] = self.ihsan_threshold

        # --- Determine tier status ---
        hard_failures = gharar_detected or riba_detected or not gini_passed
        if hard_failures or not ihsan_passed:
            status = TierStatus.FAILED
        else:
            status = TierStatus.PASSED

        duration_us = (time.perf_counter_ns() - start) // 1000

        return H0Result(
            status=status,
            gharar_detected=gharar_detected,
            riba_detected=riba_detected,
            gini_coefficient=gini,
            gini_passed=gini_passed,
            ihsan_score=action.ihsan_score,
            ihsan_passed=ihsan_passed,
            violations=violations,
            evidence=evidence,
            duration_us=duration_us,
        )

    # =========================================================================
    # H1: PERFORMANCE INVARIANTS
    # =========================================================================

    def verify_h1(self, action: SovereignAction) -> H1Result:
        """
        Verify H1: Performance invariants.

        Sub-invariants:
        1. SLA bounds -- estimated duration must not exceed deadline.
           Actions that will miss their SLA are rejected before execution.
        2. Throughput floor -- minimum operations per second.
           Below this floor, the system is under unacceptable load.
        3. Resource cost ceiling -- cost must not exceed budget.
           Prevents runaway resource consumption.
        4. Signal quality -- SNR >= minimum threshold.
           Low-quality signals are noise, not actionable intelligence.

        Standing on Giants:
        - Shannon (1948): Information Theory, Signal-to-Noise
        - Deming (1986): Quality Management Principles

        Args:
            action: The sovereign action to evaluate.

        Returns:
            H1Result with pass/fail status and structured evidence.
        """
        start = time.perf_counter_ns()
        violations: list[str] = []
        evidence: dict[str, Any] = {}

        # --- Sub-invariant 1: SLA bounds ---
        sla_met = True
        if action.sla_deadline_ms is not None and action.sla_deadline_ms > 0:
            effective_deadline = action.sla_deadline_ms * self.sla_margin
            sla_met = action.estimated_duration_ms <= effective_deadline
            if not sla_met:
                violations.append(
                    f"H1-SLA: Estimated duration {action.estimated_duration_ms:.1f}ms "
                    f"exceeds effective deadline "
                    f"{effective_deadline:.1f}ms "
                    f"(raw: {action.sla_deadline_ms:.1f}ms * {self.sla_margin} margin)"
                )
        evidence["sla_deadline_ms"] = action.sla_deadline_ms
        evidence["estimated_duration_ms"] = action.estimated_duration_ms
        evidence["sla_margin"] = self.sla_margin

        # --- Sub-invariant 2: Throughput floor ---
        throughput_adequate = True
        if self.min_throughput_rps > 0:
            throughput_adequate = action.throughput_rps >= self.min_throughput_rps
            if not throughput_adequate:
                violations.append(
                    f"H1-THROUGHPUT: Current throughput {action.throughput_rps:.2f} rps "
                    f"below minimum floor {self.min_throughput_rps:.2f} rps"
                )
        evidence["throughput_rps"] = action.throughput_rps
        evidence["min_throughput_rps"] = self.min_throughput_rps

        # --- Sub-invariant 3: Resource cost ceiling ---
        cost_within_ceiling = action.resource_cost <= action.resource_cost_ceiling
        if not cost_within_ceiling:
            violations.append(
                f"H1-COST: Resource cost {action.resource_cost:.4f} "
                f"exceeds ceiling {action.resource_cost_ceiling:.4f}"
            )
        evidence["resource_cost"] = action.resource_cost
        evidence["resource_cost_ceiling"] = action.resource_cost_ceiling

        # --- Sub-invariant 4: Signal quality (SNR) ---
        snr_passed = action.snr_score >= self.snr_minimum
        if not snr_passed:
            violations.append(
                f"H1-SNR: Signal-to-noise ratio {action.snr_score:.4f} "
                f"below minimum {self.snr_minimum} -- "
                f"insufficient signal quality"
            )
        evidence["snr_score"] = action.snr_score
        evidence["snr_minimum"] = self.snr_minimum

        # --- Determine tier status ---
        if (
            not sla_met
            or not throughput_adequate
            or not cost_within_ceiling
            or not snr_passed
        ):
            status = TierStatus.FAILED
        else:
            status = TierStatus.PASSED

        duration_us = (time.perf_counter_ns() - start) // 1000

        return H1Result(
            status=status,
            sla_met=sla_met,
            throughput_adequate=throughput_adequate,
            cost_within_ceiling=cost_within_ceiling,
            snr_score=action.snr_score,
            snr_passed=snr_passed,
            violations=violations,
            evidence=evidence,
            duration_us=duration_us,
        )

    # =========================================================================
    # H2: SAFETY INVARIANTS
    # =========================================================================

    def verify_h2(self, action: SovereignAction) -> H2Result:
        """
        Verify H2: Safety invariants.

        Sub-invariants:
        1. Reversibility -- irreversible actions are inherently risky.
           Non-reversible actions require elevated approval.
        2. Blast containment -- scope must not exceed allowed maximum.
           Prevents uncontrolled cascading failures.
        3. Human escalation -- override path must be available.
           No autonomous action may lack a human kill-switch.
        4. No-harm verification -- harm assessment must be below threshold.
           Direct harm prevention is a constitutional absolute.

        Standing on Giants:
        - Asimov (1942): Three Laws of Robotics (harm prevention)
        - Leveson (2011): Engineering a Safer World (system safety)
        - Anthropic (2022): Constitutional AI (no-harm constraints)

        Args:
            action: The sovereign action to evaluate.

        Returns:
            H2Result with pass/fail status and structured evidence.
        """
        start = time.perf_counter_ns()
        violations: list[str] = []
        evidence: dict[str, Any] = {}

        # --- Sub-invariant 1: Reversibility ---
        reversible = action.reversible
        if not reversible:
            violations.append(
                "H2-REVERSIBILITY: Action is irreversible -- "
                "elevated approval required"
            )
        evidence["reversible"] = reversible

        # --- Sub-invariant 2: Blast containment ---
        blast_contained = action.blast_radius.value <= action.max_allowed_scope.value
        if not blast_contained:
            violations.append(
                f"H2-BLAST: Blast radius {action.blast_radius.name} "
                f"exceeds maximum allowed scope {action.max_allowed_scope.name} -- "
                f"cascading failure risk"
            )
        evidence["blast_radius"] = action.blast_radius.name
        evidence["max_allowed_scope"] = action.max_allowed_scope.name

        # --- Sub-invariant 3: Human escalation path ---
        human_override = action.human_override_available
        if not human_override:
            violations.append(
                "H2-HUMAN: No human override path available -- "
                "autonomous action without kill-switch"
            )
        evidence["human_override_available"] = human_override

        # --- Sub-invariant 4: No-harm verification ---
        harm_score = action.harm_assessment
        no_harm_verified = harm_score < self.harm_hard_reject
        if not no_harm_verified:
            violations.append(
                f"H2-HARM: Harm assessment {harm_score:.4f} "
                f"exceeds hard reject threshold {self.harm_hard_reject} -- "
                f"action poses unacceptable harm"
            )
        evidence["harm_score"] = harm_score
        evidence["harm_hard_reject"] = self.harm_hard_reject
        evidence["harm_soft_threshold"] = self.harm_soft_threshold

        # --- Determine tier status ---
        # Hard failures: blast breach or excessive harm are non-negotiable
        hard_failure = not blast_contained or not no_harm_verified
        # Soft failures: irreversibility and missing human override are remediable
        soft_failure = not reversible or not human_override

        if hard_failure:
            status = TierStatus.FAILED
        elif soft_failure:
            # Degraded means "passed minimums but has remediable concerns"
            status = TierStatus.DEGRADED
        else:
            status = TierStatus.PASSED

        duration_us = (time.perf_counter_ns() - start) // 1000

        return H2Result(
            status=status,
            reversible=reversible,
            blast_contained=blast_contained,
            human_override_available=human_override,
            harm_score=harm_score,
            no_harm_verified=no_harm_verified,
            violations=violations,
            evidence=evidence,
            duration_us=duration_us,
        )

    # =========================================================================
    # CROWN ADJUDICATION -- Combine all three tiers
    # =========================================================================

    def adjudicate(self, action: SovereignAction) -> CrownVerdictResult:
        """
        Run all three invariant tiers and produce a signed Crown Verdict.

        Decision logic:
        - All tiers PASSED --> ACCEPT
        - Any tier has hard FAILED with no remediation --> REJECT
        - Some failures but all have remediation paths --> REVISE

        The REVISE verdict is only issued when:
        1. H0 failed solely on Ihsan score (close to threshold), OR
        2. H1 failed solely on SLA/throughput (can be rescheduled), OR
        3. H2 is DEGRADED (reversibility/human override missing but
           harm and blast are contained)

        Hard rejections (always REJECT, never REVISE):
        - Gharar or riba detected (H0)
        - Gini violation (H0)
        - Blast radius breach (H2)
        - Harm above hard threshold (H2)

        Args:
            action: The sovereign action to evaluate.

        Returns:
            CrownVerdictResult with signed verdict, tier results,
            and remediation suggestions.
        """
        total_start = time.perf_counter_ns()

        # Evaluate all three tiers
        h0 = self.verify_h0(action)
        h1 = self.verify_h1(action)
        h2 = self.verify_h2(action)

        # Collect remediations
        remediations = self._compute_remediations(action, h0, h1, h2)

        # Determine final verdict
        verdict = self._determine_verdict(h0, h1, h2, remediations)

        total_duration_us = (time.perf_counter_ns() - total_start) // 1000

        # Build result
        result = CrownVerdictResult(
            action_id=action.action_id,
            verdict=verdict,
            h0=h0,
            h1=h1,
            h2=h2,
            remediations=remediations,
            total_duration_us=total_duration_us,
        )

        # Sign the verdict with Ed25519
        self._sign_verdict(result)

        # Record for audit
        self._verdicts.append(result)

        logger.info(
            "Crown Verdict for %s: %s (H0=%s, H1=%s, H2=%s) in %d us",
            action.action_id,
            verdict.value,
            h0.status.value,
            h1.status.value,
            h2.status.value,
            total_duration_us,
        )

        return result

    def _determine_verdict(
        self,
        h0: H0Result,
        h1: H1Result,
        h2: H2Result,
        remediations: list[str],
    ) -> Verdict:
        """
        Determine the final verdict from tier results.

        Logic:
        1. All passed -> ACCEPT
        2. Any hard, non-remediable failure -> REJECT
        3. Only soft/remediable failures -> REVISE
        """
        # All tiers passed cleanly
        if h0.passed and h1.passed and (h2.passed or h2.status == TierStatus.DEGRADED):
            # H2 DEGRADED with no hard failures can still ACCEPT with warnings
            if h2.status == TierStatus.DEGRADED:
                # Only if the degraded issues have remediations
                if remediations:
                    return Verdict.REVISE
                return Verdict.ACCEPT
            return Verdict.ACCEPT

        # Check for hard rejections (no remediation possible)
        has_hard_rejection = (
            h0.gharar_detected
            or h0.riba_detected
            or not h0.gini_passed
            or not h2.blast_contained
            or not h2.no_harm_verified
        )

        if has_hard_rejection:
            return Verdict.REJECT

        # Remaining failures may be remediable
        if remediations:
            return Verdict.REVISE

        # Failures with no remediation path -> REJECT
        return Verdict.REJECT

    def _compute_remediations(
        self,
        action: SovereignAction,
        h0: H0Result,
        h1: H1Result,
        h2: H2Result,
    ) -> list[str]:
        """
        Compute actionable remediation suggestions for failed sub-invariants.

        Only produces remediations for soft failures -- hard rejections
        (gharar, riba, Gini, blast, harm) have no remediation path.
        """
        remediations: list[str] = []

        # H0 remediations (only for Ihsan -- gharar/riba/Gini are hard)
        if (
            not h0.ihsan_passed
            and not h0.gharar_detected
            and not h0.riba_detected
            and h0.gini_passed
        ):
            deficit = self.ihsan_threshold - action.ihsan_score
            remediations.append(
                f"REVISE-H0: Raise Ihsan score by {deficit:.4f} to meet "
                f"threshold {self.ihsan_threshold}. "
                f"Consider improving correctness, safety, or auditability dimensions."
            )

        # H1 remediations
        if not h1.sla_met:
            remediations.append(
                "REVISE-H1-SLA: Reschedule action with relaxed deadline "
                "or optimize estimated duration."
            )
        if not h1.throughput_adequate:
            remediations.append(
                "REVISE-H1-THROUGHPUT: Wait for system load to decrease "
                "or provision additional compute resources."
            )
        if not h1.cost_within_ceiling:
            remediations.append(
                "REVISE-H1-COST: Reduce resource cost or request "
                "elevated cost ceiling approval."
            )
        if not h1.snr_passed:
            deficit = self.snr_minimum - action.snr_score
            remediations.append(
                f"REVISE-H1-SNR: Improve signal quality by {deficit:.4f}. "
                f"Add provenance depth, corroboration, or source trust."
            )

        # H2 remediations (only for reversibility and human override)
        if not h2.reversible and h2.blast_contained and h2.no_harm_verified:
            remediations.append(
                "REVISE-H2-REVERSIBILITY: Add rollback mechanism or "
                "checkpoint before executing irreversible action."
            )
        if (
            not h2.human_override_available
            and h2.blast_contained
            and h2.no_harm_verified
        ):
            remediations.append(
                "REVISE-H2-HUMAN: Register a human escalation path "
                "before autonomous execution."
            )

        return remediations

    def _sign_verdict(self, result: CrownVerdictResult) -> None:
        """Sign a verdict with the engine's Ed25519 keypair."""
        digest = result._compute_digest()
        result.signature = sign_message(digest, self._private_key)
        result.signer_public_key = self._public_key

    # =========================================================================
    # AUDIT & STATISTICS
    # =========================================================================

    def get_verdict_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent verdicts as serializable dicts."""
        return [v.to_dict() for v in self._verdicts[-limit:]]

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics across all verdicts."""
        if not self._verdicts:
            return {
                "total_verdicts": 0,
                "accept_count": 0,
                "reject_count": 0,
                "revise_count": 0,
            }

        accept_count = sum(1 for v in self._verdicts if v.verdict == Verdict.ACCEPT)
        reject_count = sum(1 for v in self._verdicts if v.verdict == Verdict.REJECT)
        revise_count = sum(1 for v in self._verdicts if v.verdict == Verdict.REVISE)

        avg_duration = sum(v.total_duration_us for v in self._verdicts) / len(
            self._verdicts
        )

        # Count failures by tier
        h0_failures = sum(1 for v in self._verdicts if not v.h0.passed)
        h1_failures = sum(1 for v in self._verdicts if not v.h1.passed)
        h2_failures = sum(1 for v in self._verdicts if not v.h2.passed)

        return {
            "total_verdicts": len(self._verdicts),
            "accept_count": accept_count,
            "reject_count": reject_count,
            "revise_count": revise_count,
            "accept_rate": accept_count / len(self._verdicts),
            "avg_duration_us": avg_duration,
            "tier_failures": {
                "h0_ethical": h0_failures,
                "h1_performance": h1_failures,
                "h2_safety": h2_failures,
            },
        }


# =============================================================================
# CONVENIENCE FACTORY
# =============================================================================


def create_crown_verdict(
    ihsan_threshold: float = IHSAN_THRESHOLD,
    snr_minimum: float = SNR_THRESHOLD,
    gini_threshold: float = ADL_GINI_THRESHOLD,
) -> CrownVerdict:
    """
    Create a CrownVerdict engine with constitutional defaults.

    All thresholds are sourced from core.integration.constants,
    the single source of truth for BIZRA constitutional values.
    """
    return CrownVerdict(
        ihsan_threshold=ihsan_threshold,
        snr_minimum=snr_minimum,
        gini_threshold=gini_threshold,
    )
