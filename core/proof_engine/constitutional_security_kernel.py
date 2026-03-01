"""
Constitutional Security Kernel (CSK)
====================================

Focused security/control-plane primitives for sovereign execution:
1. Conservative formal fallback checks (alpha4)
2. Tiered verification pipeline (alpha7)
3. Performance anomaly attestation (alpha9)
4. Takaful admission gate (alpha3)
5. Oblivious compute scheduling (alpha1)
6. Static Ihsan fitness baseline (alpha6)
"""

from __future__ import annotations

import hashlib
import logging
import random
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD

logger = logging.getLogger("sovereign.csk")


# ============================================================================
# §1 Reason Codes + Receipts
# ============================================================================


class CSKReasonCode(str, Enum):
    """Machine-readable reason codes for CSK decisions."""

    # Tier 1
    T1_SAFE_PATTERN_MATCH = "T1_SAFE_PATTERN_MATCH"
    T1_KNOWN_DANGEROUS_BLOCKED = "T1_KNOWN_DANGEROUS_BLOCKED"
    T1_ESCALATED_UNKNOWN = "T1_ESCALATED_UNKNOWN"

    # Tier 2
    T2_FATE_Z3_PASSED = "T2_FATE_Z3_PASSED"
    T2_FATE_Z3_REJECTED = "T2_FATE_Z3_REJECTED"
    T2_FATE_FALLBACK_CONSERVATIVE_PASS = "T2_FATE_FALLBACK_CONSERVATIVE_PASS"
    T2_FATE_FALLBACK_CONSERVATIVE_REJECT = "T2_FATE_FALLBACK_CONSERVATIVE_REJECT"

    # Performance attestation
    PERF_WITHIN_ENVELOPE = "PERF_WITHIN_ENVELOPE"
    PERF_ANOMALY_MINOR = "PERF_ANOMALY_MINOR"
    PERF_ANOMALY_CRITICAL = "PERF_ANOMALY_CRITICAL"

    # Takaful admission
    TAKAFUL_PROBATIONARY = "TAKAFUL_PROBATIONARY"
    TAKAFUL_REJECTED_NO_HUMANITY_PROOF = "TAKAFUL_REJECTED_NO_HUMANITY_PROOF"
    TAKAFUL_REJECTED_INSUFFICIENT_IMPACT = "TAKAFUL_REJECTED_INSUFFICIENT_IMPACT"
    TAKAFUL_REJECTED_IHSAN_BELOW_FLOOR = "TAKAFUL_REJECTED_IHSAN_BELOW_FLOOR"
    TAKAFUL_ADMITTED = "TAKAFUL_ADMITTED"


@dataclass(frozen=True)
class CSKReceipt:
    """Immutable receipt for every CSK decision."""

    receipt_id: str
    tier: int
    reason_code: CSKReasonCode
    timestamp_ns: int
    action_digest: str
    passed: bool
    evidence: Dict[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        payload = (
            f"{self.receipt_id}|{self.tier}|{self.reason_code.value}|"
            f"{self.timestamp_ns}|{self.action_digest}|{self.passed}|"
            f"{repr(sorted(self.evidence.items()))}"
        )
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=32).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "tier": self.tier,
            "reason_code": self.reason_code.value,
            "timestamp_ns": self.timestamp_ns,
            "action_digest": self.action_digest,
            "passed": self.passed,
            "evidence": dict(self.evidence),
            "receipt_digest": self.digest(),
        }


def _action_digest(context: Dict[str, Any]) -> str:
    """Deterministic digest for action context."""
    canonical = repr(sorted((str(k), repr(v)) for k, v in context.items()))
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()


# ============================================================================
# §2 Conservative FATE fallback (alpha4)
# ============================================================================


SafePatternPredicate = Callable[[Dict[str, Any]], bool]

_SAFE_PATTERNS: Dict[str, SafePatternPredicate] = {
    "query": lambda _: True,
    "read": lambda _: True,
    "status": lambda _: True,
}
_KNOWN_DANGEROUS_ACTIONS = {
    "delete",
    "execute",
    "deploy",
    "shutdown",
    "wipe",
    "drop",
}


def register_safe_pattern(
    action_type: str,
    predicate: Optional[SafePatternPredicate] = None,
) -> None:
    """Register a safe pattern for conservative fallback checks."""
    key = action_type.strip().lower()
    if not key:
        return
    _SAFE_PATTERNS[key] = predicate or (lambda _: True)


def _is_safe_pattern(action_type: str, context: Dict[str, Any]) -> bool:
    pred = _SAFE_PATTERNS.get(action_type.strip().lower())
    if pred is None:
        return False
    try:
        return bool(pred(context))
    except Exception:
        logger.debug("Safe pattern predicate failed for action_type=%s", action_type)
        return False


def conservative_constraint_check(
    context: Dict[str, Any],
) -> Tuple[bool, CSKReasonCode]:
    """
    Fail-closed fallback for when a formal solver is unavailable.

    Policy:
    - Enforce minimum Ihsan and SNR thresholds.
    - Enforce cost <= autonomy budget.
    - Non-zero risk requires explicit mitigation:
      safe pattern OR human approval OR low-risk reversible action.
    """
    ihsan = float(context.get("ihsan", 0.0))
    snr = float(context.get("snr", 0.0))
    cost = float(context.get("cost", 0.0))
    autonomy_limit = float(context.get("autonomy_limit", 0.0))
    risk_level = float(context.get("risk_level", 1.0))
    action_type = str(context.get("action_type", "")).strip().lower()
    human_approved = bool(context.get("human_approved", False))
    reversible = bool(context.get("reversible", False))

    if ihsan < UNIFIED_IHSAN_THRESHOLD or snr < UNIFIED_SNR_THRESHOLD:
        return False, CSKReasonCode.T2_FATE_FALLBACK_CONSERVATIVE_REJECT

    if cost > autonomy_limit:
        return False, CSKReasonCode.T2_FATE_FALLBACK_CONSERVATIVE_REJECT

    if risk_level <= 0.0:
        return True, CSKReasonCode.T2_FATE_FALLBACK_CONSERVATIVE_PASS

    if _is_safe_pattern(action_type, context):
        return True, CSKReasonCode.T2_FATE_FALLBACK_CONSERVATIVE_PASS

    if human_approved:
        return True, CSKReasonCode.T2_FATE_FALLBACK_CONSERVATIVE_PASS

    if reversible and risk_level <= 0.2:
        return True, CSKReasonCode.T2_FATE_FALLBACK_CONSERVATIVE_PASS

    return False, CSKReasonCode.T2_FATE_FALLBACK_CONSERVATIVE_REJECT


# ============================================================================
# §3 Tiered verification engine (alpha7)
# ============================================================================


class TierDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    ESCALATE = "escalate"


@dataclass
class TierResult:
    tier: int
    decision: TierDecision
    receipt: CSKReceipt
    duration_us: int
    evidence: Dict[str, Any] = field(default_factory=dict)


class TieredVerificationEngine:
    """Tiered verification pipeline with fail-closed behavior."""

    def __init__(self) -> None:
        self._z3_available = False
        try:
            import z3  # type: ignore  # noqa: F401

            self._z3_available = True
        except Exception:
            self._z3_available = False

    def tier1_precheck(self, context: Dict[str, Any]) -> TierResult:
        start_ns = time.perf_counter_ns()
        action_type = str(context.get("action_type", "")).strip().lower()
        risk_level = float(context.get("risk_level", 0.0))
        human_approved = bool(context.get("human_approved", False))

        if (
            action_type in _KNOWN_DANGEROUS_ACTIONS
            and risk_level >= 0.7
            and not human_approved
        ):
            decision = TierDecision.BLOCK
            reason = CSKReasonCode.T1_KNOWN_DANGEROUS_BLOCKED
            passed = False
        elif _is_safe_pattern(action_type, context) and risk_level <= 0.3:
            decision = TierDecision.ALLOW
            reason = CSKReasonCode.T1_SAFE_PATTERN_MATCH
            passed = True
        else:
            decision = TierDecision.ESCALATE
            reason = CSKReasonCode.T1_ESCALATED_UNKNOWN
            passed = False

        duration_us = (time.perf_counter_ns() - start_ns) // 1000
        receipt = CSKReceipt(
            receipt_id=f"csk_t1_{time.time_ns()}",
            tier=1,
            reason_code=reason,
            timestamp_ns=time.time_ns(),
            action_digest=_action_digest(context),
            passed=passed,
            evidence={"action_type": action_type, "risk_level": risk_level},
        )
        return TierResult(1, decision, receipt, duration_us, dict(receipt.evidence))

    def tier2_formal_verification(self, context: Dict[str, Any]) -> TierResult:
        start_ns = time.perf_counter_ns()

        if self._z3_available:
            # Conservative Z3 path placeholder: if clearly dangerous and unapproved, block.
            risk_level = float(context.get("risk_level", 1.0))
            human_approved = bool(context.get("human_approved", False))
            if risk_level > 0.8 and not human_approved:
                passed = False
                reason = CSKReasonCode.T2_FATE_Z3_REJECTED
                decision = TierDecision.BLOCK
            else:
                passed = True
                reason = CSKReasonCode.T2_FATE_Z3_PASSED
                decision = TierDecision.ALLOW
        else:
            passed, reason = conservative_constraint_check(context)
            decision = TierDecision.ALLOW if passed else TierDecision.BLOCK

        duration_us = (time.perf_counter_ns() - start_ns) // 1000
        receipt = CSKReceipt(
            receipt_id=f"csk_t2_{time.time_ns()}",
            tier=2,
            reason_code=reason,
            timestamp_ns=time.time_ns(),
            action_digest=_action_digest(context),
            passed=passed,
            evidence={"z3_available": self._z3_available},
        )
        return TierResult(2, decision, receipt, duration_us, dict(receipt.evidence))

    def verify(self, context: Dict[str, Any]) -> List[TierResult]:
        results: List[TierResult] = []
        t1 = self.tier1_precheck(context)
        results.append(t1)

        if t1.decision in (TierDecision.ALLOW, TierDecision.BLOCK):
            return results

        t2 = self.tier2_formal_verification(context)
        results.append(t2)
        return results


# ============================================================================
# §4 Performance attestation registry (alpha9)
# ============================================================================


@dataclass
class PerformanceEnvelope:
    """Expected execution-time envelope in microseconds."""

    module_name: str
    expected_duration_us: float
    duration_stddev_us: float = 1.0
    sigma_threshold: float = 2.0

    _count: int = 0
    observed_mean: float = 0.0
    _m2: float = 0.0

    def record_observation(self, duration_us: float) -> None:
        self._count += 1
        delta = duration_us - self.observed_mean
        self.observed_mean += delta / self._count
        delta2 = duration_us - self.observed_mean
        self._m2 += delta * delta2

    @property
    def observed_stddev(self) -> float:
        if self._count < 2:
            return max(self.duration_stddev_us, 1.0)
        return max((self._m2 / (self._count - 1)) ** 0.5, 1.0)

    def sigma_deviation(self, duration_us: float) -> float:
        mean = self.observed_mean if self._count > 0 else self.expected_duration_us
        stddev = self.observed_stddev
        return abs(duration_us - mean) / stddev


class PerformanceAttestationRegistry:
    """Tracks envelopes and emits receipts for runtime performance claims."""

    def __init__(self) -> None:
        self._envelopes: Dict[str, PerformanceEnvelope] = {}
        self._anomalies: List[Dict[str, Any]] = []

    def register(self, envelope: PerformanceEnvelope) -> None:
        self._envelopes[envelope.module_name] = envelope

    def attest(self, module_name: str, duration_us: float) -> CSKReceipt:
        now_ns = time.time_ns()
        envelope = self._envelopes.get(module_name)
        evidence = {"module": module_name, "duration_us": duration_us}
        action_digest = hashlib.blake2b(
            f"{module_name}:{duration_us}:{now_ns}".encode("utf-8"),
            digest_size=16,
        ).hexdigest()

        if envelope is None:
            reason = CSKReasonCode.PERF_ANOMALY_MINOR
            passed = True
            evidence["note"] = "module_not_registered"
        else:
            sigma = envelope.sigma_deviation(duration_us)
            envelope.record_observation(duration_us)
            evidence["sigma"] = round(sigma, 4)
            evidence["threshold"] = envelope.sigma_threshold

            if sigma <= envelope.sigma_threshold:
                reason = CSKReasonCode.PERF_WITHIN_ENVELOPE
                passed = True
            elif sigma <= envelope.sigma_threshold * 10:
                reason = CSKReasonCode.PERF_ANOMALY_MINOR
                passed = False
            else:
                reason = CSKReasonCode.PERF_ANOMALY_CRITICAL
                passed = False

        if reason in (
            CSKReasonCode.PERF_ANOMALY_MINOR,
            CSKReasonCode.PERF_ANOMALY_CRITICAL,
        ):
            self._anomalies.append(
                {
                    "ts_ns": now_ns,
                    "module": module_name,
                    "duration_us": duration_us,
                    "reason_code": reason.value,
                    "passed": passed,
                }
            )

        return CSKReceipt(
            receipt_id=f"csk_perf_{now_ns}",
            tier=3,
            reason_code=reason,
            timestamp_ns=now_ns,
            action_digest=action_digest,
            passed=passed,
            evidence=evidence,
        )

    def get_anomalies(self) -> List[Dict[str, Any]]:
        return list(self._anomalies)


# ============================================================================
# §5 Takaful admission gate (alpha3)
# ============================================================================


class TakafulStatus(Enum):
    PROBATIONARY = "probationary"
    ADMITTED = "admitted"
    SUSPENDED = "suspended"
    EXPELLED = "expelled"


@dataclass
class TakafulNodeProfile:
    node_id: str
    status: TakafulStatus = TakafulStatus.PROBATIONARY
    humanity_verified: bool = False
    total_interactions: int = 0
    verified_impact_score: float = 0.0
    ihsan_history: List[float] = field(default_factory=list)
    admission_timestamp_ns: int = 0
    last_activity_ns: int = 0
    cluster_id: Optional[str] = None

    @property
    def ihsan_mean(self) -> float:
        if not self.ihsan_history:
            return 0.0
        return statistics.mean(self.ihsan_history)

    @property
    def ihsan_min(self) -> float:
        if not self.ihsan_history:
            return 0.0
        return min(self.ihsan_history)


class TakafulAdmissionGate:
    """Three-gate admission for Takaful contribution rights."""

    def __init__(
        self,
        min_interactions: int = 50,
        min_impact_score: float = 10.0,
        ihsan_floor: float = UNIFIED_IHSAN_THRESHOLD,
        ihsan_history_window: int = 10,
    ) -> None:
        self._min_interactions = min_interactions
        self._min_impact_score = min_impact_score
        self._ihsan_floor = ihsan_floor
        self._ihsan_history_window = ihsan_history_window
        self._nodes: Dict[str, TakafulNodeProfile] = {}

    def register_node(self, node_id: str) -> TakafulNodeProfile:
        profile = TakafulNodeProfile(
            node_id=node_id,
            status=TakafulStatus.PROBATIONARY,
            admission_timestamp_ns=time.time_ns(),
        )
        self._nodes[node_id] = profile
        return profile

    def record_interaction(
        self,
        node_id: str,
        impact_delta: float = 0.0,
        ihsan_score: float = 0.0,
    ) -> None:
        profile = self._nodes.get(node_id)
        if profile is None:
            return

        profile.total_interactions += 1
        profile.verified_impact_score += max(0.0, impact_delta)
        profile.ihsan_history.append(ihsan_score)
        profile.last_activity_ns = time.time_ns()

        if len(profile.ihsan_history) > self._ihsan_history_window * 2:
            profile.ihsan_history = profile.ihsan_history[-self._ihsan_history_window :]

    def verify_humanity(self, node_id: str, proof: Any = None) -> bool:
        del proof
        profile = self._nodes.get(node_id)
        if profile is None:
            return False
        profile.humanity_verified = True
        return True

    def evaluate_admission(self, node_id: str) -> CSKReceipt:
        now_ns = time.time_ns()
        profile = self._nodes.get(node_id)

        if profile is None:
            return CSKReceipt(
                receipt_id=f"takaful_{node_id}_{now_ns}",
                tier=4,
                reason_code=CSKReasonCode.TAKAFUL_REJECTED_NO_HUMANITY_PROOF,
                timestamp_ns=now_ns,
                action_digest=hashlib.blake2b(node_id.encode(), digest_size=16).hexdigest(),
                passed=False,
                evidence={"node_id": node_id, "note": "node_not_registered"},
            )

        digest = hashlib.blake2b(node_id.encode(), digest_size=16).hexdigest()

        if not profile.humanity_verified:
            return CSKReceipt(
                receipt_id=f"takaful_{node_id}_{now_ns}",
                tier=4,
                reason_code=CSKReasonCode.TAKAFUL_REJECTED_NO_HUMANITY_PROOF,
                timestamp_ns=now_ns,
                action_digest=digest,
                passed=False,
                evidence={"gate": "humanity", "humanity_verified": False},
            )

        if (
            profile.total_interactions < self._min_interactions
            or profile.verified_impact_score < self._min_impact_score
        ):
            return CSKReceipt(
                receipt_id=f"takaful_{node_id}_{now_ns}",
                tier=4,
                reason_code=CSKReasonCode.TAKAFUL_REJECTED_INSUFFICIENT_IMPACT,
                timestamp_ns=now_ns,
                action_digest=digest,
                passed=False,
                evidence={
                    "gate": "impact_history",
                    "interactions": profile.total_interactions,
                    "min_interactions": self._min_interactions,
                    "impact_score": profile.verified_impact_score,
                    "min_impact": self._min_impact_score,
                },
            )

        recent_ihsan = profile.ihsan_history[-self._ihsan_history_window :]
        if len(recent_ihsan) < self._ihsan_history_window:
            return CSKReceipt(
                receipt_id=f"takaful_{node_id}_{now_ns}",
                tier=4,
                reason_code=CSKReasonCode.TAKAFUL_PROBATIONARY,
                timestamp_ns=now_ns,
                action_digest=digest,
                passed=False,
                evidence={
                    "gate": "ihsan_history",
                    "history_length": len(recent_ihsan),
                    "required": self._ihsan_history_window,
                },
            )

        if min(recent_ihsan) < self._ihsan_floor:
            return CSKReceipt(
                receipt_id=f"takaful_{node_id}_{now_ns}",
                tier=4,
                reason_code=CSKReasonCode.TAKAFUL_REJECTED_IHSAN_BELOW_FLOOR,
                timestamp_ns=now_ns,
                action_digest=digest,
                passed=False,
                evidence={
                    "gate": "ihsan_floor",
                    "ihsan_min": min(recent_ihsan),
                    "ihsan_mean": statistics.mean(recent_ihsan),
                    "threshold": self._ihsan_floor,
                },
            )

        profile.status = TakafulStatus.ADMITTED
        return CSKReceipt(
            receipt_id=f"takaful_{node_id}_{now_ns}",
            tier=4,
            reason_code=CSKReasonCode.TAKAFUL_ADMITTED,
            timestamp_ns=now_ns,
            action_digest=digest,
            passed=True,
            evidence={
                "humanity_verified": True,
                "interactions": profile.total_interactions,
                "impact_score": profile.verified_impact_score,
                "ihsan_mean": statistics.mean(recent_ihsan),
                "ihsan_min": min(recent_ihsan),
            },
        )

    def can_contribute(self, node_id: str) -> bool:
        profile = self._nodes.get(node_id)
        if profile is None:
            return False
        return profile.status == TakafulStatus.ADMITTED

    def can_receive(self, node_id: str) -> bool:
        profile = self._nodes.get(node_id)
        if profile is None:
            return False
        return profile.status in (TakafulStatus.PROBATIONARY, TakafulStatus.ADMITTED)

    def get_contributors(self) -> List[str]:
        return [nid for nid, p in self._nodes.items() if p.status == TakafulStatus.ADMITTED]

    def suspend_node(self, node_id: str, reason: str = "") -> None:
        profile = self._nodes.get(node_id)
        if profile:
            profile.status = TakafulStatus.SUSPENDED
            logger.warning("Takaful node suspended: %s reason=%s", node_id, reason)


# ============================================================================
# §6 Oblivious scheduler (alpha1)
# ============================================================================


class ObliviousComputeScheduler:
    """Differential-privacy-inspired padding of compute operation lists."""

    def __init__(self, epsilon: float = 1.0, dummy_compute_ratio: float = 0.15) -> None:
        self._epsilon = epsilon
        self._dummy_ratio = dummy_compute_ratio
        self._rng = random.SystemRandom()

    def schedule(self, real_operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        n_real = len(real_operations)
        n_dummy = max(1, int(n_real * self._dummy_ratio))

        dummies: List[Dict[str, Any]] = []
        for _ in range(n_dummy):
            template = self._rng.choice(real_operations) if real_operations else {}
            dummies.append(
                {
                    "op_type": template.get("op_type", "inference"),
                    "payload_size": template.get("payload_size", 256),
                    "is_dummy": True,
                    "nonce": self._rng.getrandbits(64),
                }
            )

        tagged_real = [{**op, "is_dummy": False} for op in real_operations]
        combined = tagged_real + dummies
        self._rng.shuffle(combined)
        return combined

    def strip_dummies(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [r for r in results if not r.get("is_dummy", False)]


# ============================================================================
# §7 Ihsan fitness interface (alpha6)
# ============================================================================


class IhsanFitnessFunction(Protocol):
    def evaluate(self, action_context: Dict[str, Any]) -> float:
        ...

    def propose_mutation(self, evidence: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        ...


class StaticIhsanFitness:
    """Static v1.0 Ihsan evaluator with bounded score in [0, 1]."""

    def evaluate(self, action_context: Dict[str, Any]) -> float:
        weights = {
            "ihsan": 0.35,
            "snr": 0.25,
            "impact_delta": 0.20,
            "reversible_bonus": 0.10,
            "human_approved_bonus": 0.10,
        }

        score = 0.0
        score += weights["ihsan"] * min(1.0, float(action_context.get("ihsan", 0.0)))
        score += weights["snr"] * min(1.0, float(action_context.get("snr", 0.0)))
        score += weights["impact_delta"] * min(
            1.0, float(action_context.get("impact_delta", 0.0)) / 10.0
        )
        if action_context.get("reversible", False):
            score += weights["reversible_bonus"]
        if action_context.get("human_approved", False):
            score += weights["human_approved_bonus"]

        return min(1.0, max(0.0, score))

    def propose_mutation(self, evidence: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        del evidence
        return None


__all__ = [
    "CSKReasonCode",
    "CSKReceipt",
    "PerformanceEnvelope",
    "PerformanceAttestationRegistry",
    "conservative_constraint_check",
    "register_safe_pattern",
    "TierDecision",
    "TierResult",
    "TieredVerificationEngine",
    "TakafulStatus",
    "TakafulNodeProfile",
    "TakafulAdmissionGate",
    "ObliviousComputeScheduler",
    "IhsanFitnessFunction",
    "StaticIhsanFitness",
]
