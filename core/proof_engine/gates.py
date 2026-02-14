"""
6-Gate Chain — Fail-Closed Execution Pipeline

The gates form a chain where each must pass for execution to proceed.
Failure at any gate produces a signed rejection receipt.

Gate Chain:
1. SchemaGate     → Input validation (structure, types, bounds)
2. ProvenanceGate → Source verification (origin, trust, history)
3. SNRGate        → Signal-to-noise threshold (quality filter)
4. ConstraintGate → Z3 + Ihsān constraints (formal verification)
5. SafetyGate     → Constitutional safety check (harm prevention)
6. CommitGate     → Final commit gate (resource allocation)

Principle: Fail fast, fail closed, always produce receipt.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.proof_engine.canonical import CanonPolicy, CanonQuery
from core.proof_engine.receipt import (
    Metrics,
    Receipt,
    ReceiptBuilder,
    SovereignSigner,
)
from core.proof_engine.snr import SNREngine, SNRInput, SNRPolicy, SNRTrace


class GateStatus(Enum):
    """Gate execution status."""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class GateResult:
    """Result of a single gate evaluation."""

    gate_name: str
    status: GateStatus
    duration_us: int = 0
    reason: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def passed(self) -> bool:
        return self.status == GateStatus.PASSED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "status": self.status.value,
            "duration_us": self.duration_us,
            "reason": self.reason,
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class GateChainResult:
    """Result of full gate chain evaluation."""

    query: CanonQuery
    policy: CanonPolicy
    gate_results: List[GateResult]
    final_status: GateStatus
    last_gate_passed: str
    rejection_reason: Optional[str] = None

    # Metrics
    total_duration_us: int = 0
    snr: float = 0.0
    ihsan_score: float = 0.0
    snr_trace: Optional[SNRTrace] = None

    # Timestamp
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def passed(self) -> bool:
        return self.final_status == GateStatus.PASSED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_digest": self.query.hex_digest(),
            "policy_digest": self.policy.hex_digest(),
            "gate_results": [g.to_dict() for g in self.gate_results],
            "final_status": self.final_status.value,
            "last_gate_passed": self.last_gate_passed,
            "rejection_reason": self.rejection_reason,
            "total_duration_us": self.total_duration_us,
            "snr": self.snr,
            "ihsan_score": self.ihsan_score,
            "timestamp": self.timestamp.isoformat(),
        }


class Gate:
    """
    Base class for gates.

    Each gate evaluates a query against a policy and returns a result.
    """

    def __init__(self, name: str):
        self.name = name

    def evaluate(
        self,
        query: CanonQuery,
        policy: CanonPolicy,
        context: Dict[str, Any],
    ) -> GateResult:
        """
        Evaluate the gate.

        Must be overridden by subclasses.
        """
        raise NotImplementedError


class SchemaGate(Gate):
    """
    Gate 1: Schema Validation

    Validates input structure, types, and bounds.
    """

    def __init__(
        self,
        max_payload_size: int = 1_000_000,  # 1MB
        max_intent_length: int = 10_000,
        required_fields: Optional[List[str]] = None,
    ):
        super().__init__("schema")
        self.max_payload_size = max_payload_size
        self.max_intent_length = max_intent_length
        self.required_fields = required_fields or ["user_id", "intent"]

    def evaluate(
        self,
        query: CanonQuery,
        policy: CanonPolicy,
        context: Dict[str, Any],
    ) -> GateResult:
        start = time.perf_counter_ns()
        evidence: Dict[str, Any] = {}

        # Check required fields
        for field in self.required_fields:
            value = getattr(query, field, None)
            if not value:
                return GateResult(
                    gate_name=self.name,
                    status=GateStatus.FAILED,
                    duration_us=(time.perf_counter_ns() - start) // 1000,
                    reason=f"Missing required field: {field}",
                    evidence={"missing_field": field},
                )

        # Check intent length
        if len(query.intent) > self.max_intent_length:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.FAILED,
                duration_us=(time.perf_counter_ns() - start) // 1000,
                reason=f"Intent exceeds max length: {len(query.intent)} > {self.max_intent_length}",
                evidence={"intent_length": len(query.intent)},
            )

        # Check payload size
        payload_bytes = len(query.canonical_bytes())
        if payload_bytes > self.max_payload_size:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.FAILED,
                duration_us=(time.perf_counter_ns() - start) // 1000,
                reason=f"Payload exceeds max size: {payload_bytes} > {self.max_payload_size}",
                evidence={"payload_bytes": payload_bytes},
            )

        evidence = {
            "intent_length": len(query.intent),
            "payload_bytes": payload_bytes,
            "fields_validated": self.required_fields,
        }

        return GateResult(
            gate_name=self.name,
            status=GateStatus.PASSED,
            duration_us=(time.perf_counter_ns() - start) // 1000,
            evidence=evidence,
        )


class ProvenanceGate(Gate):
    """
    Gate 2: Provenance Verification

    Validates source origin, trust level, and history.
    """

    def __init__(
        self,
        trusted_sources: Optional[List[str]] = None,
        min_trust_score: float = 0.5,
        require_signature: bool = False,
    ):
        super().__init__("provenance")
        self.trusted_sources = set(trusted_sources or [])
        self.min_trust_score = min_trust_score
        self.require_signature = require_signature

    def evaluate(
        self,
        query: CanonQuery,
        policy: CanonPolicy,
        context: Dict[str, Any],
    ) -> GateResult:
        start = time.perf_counter_ns()
        evidence: Dict[str, Any] = {}

        # Check source trust
        source = query.user_state
        trust_score = context.get("trust_score", 0.5)

        if self.trusted_sources and source not in self.trusted_sources:
            # Unknown source, check trust score
            if trust_score < self.min_trust_score:
                return GateResult(
                    gate_name=self.name,
                    status=GateStatus.FAILED,
                    duration_us=(time.perf_counter_ns() - start) // 1000,
                    reason=f"Untrusted source with low trust score: {trust_score:.2f}",
                    evidence={"source": source, "trust_score": trust_score},
                )

        # Check signature if required
        if self.require_signature:
            signature = context.get("signature")
            if not signature:
                return GateResult(
                    gate_name=self.name,
                    status=GateStatus.FAILED,
                    duration_us=(time.perf_counter_ns() - start) // 1000,
                    reason="Missing required signature",
                    evidence={"signature_required": True},
                )

        evidence = {
            "source": source,
            "trust_score": trust_score,
            "is_trusted_source": source in self.trusted_sources,
        }

        return GateResult(
            gate_name=self.name,
            status=GateStatus.PASSED,
            duration_us=(time.perf_counter_ns() - start) // 1000,
            evidence=evidence,
        )


class SNRGate(Gate):
    """
    Gate 3: Signal-to-Noise Ratio

    Enforces minimum SNR threshold for quality filtering.
    """

    def __init__(
        self,
        snr_engine: Optional[SNREngine] = None,
        snr_policy: Optional[SNRPolicy] = None,
    ):
        super().__init__("snr")
        self.snr_engine = snr_engine or SNREngine(snr_policy)

    def evaluate(
        self,
        query: CanonQuery,
        policy: CanonPolicy,
        context: Dict[str, Any],
    ) -> GateResult:
        start = time.perf_counter_ns()

        # Extract SNR inputs from context
        snr_input = SNRInput(
            provenance_depth=context.get("provenance_depth", 1),
            corroboration_count=context.get("corroboration_count", 0),
            source_trust_score=context.get("trust_score", 0.5),
            z3_satisfiable=context.get("z3_satisfiable", True),
            ihsan_score=context.get("ihsan_score", 0.95),
            constraint_violations=context.get("constraint_violations", 0),
            contradiction_count=context.get("contradiction_count", 0),
            conflicting_sources=context.get("conflicting_sources", 0),
            prediction_accuracy=context.get("prediction_accuracy", 0.5),
            context_fit_score=context.get("context_fit_score", 0.5),
            unverifiable_claims=context.get("unverifiable_claims", 0),
            missing_citations=context.get("missing_citations", 0),
        )

        # Compute SNR
        passed, snr, trace = self.snr_engine.check_threshold(snr_input)

        if not passed:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.FAILED,
                duration_us=(time.perf_counter_ns() - start) // 1000,
                reason=f"SNR below threshold: {snr:.3f} < {self.snr_engine.policy.snr_min}",
                evidence={
                    "snr": snr,
                    "threshold": self.snr_engine.policy.snr_min,
                    "trace": trace.to_dict(),
                },
            )

        return GateResult(
            gate_name=self.name,
            status=GateStatus.PASSED,
            duration_us=(time.perf_counter_ns() - start) // 1000,
            evidence={
                "snr": snr,
                "threshold": self.snr_engine.policy.snr_min,
                "signal_mass": trace.signal_mass,
                "noise_mass": trace.noise_mass,
            },
        )


class ConstraintGate(Gate):
    """
    Gate 4: Constraint Verification

    Validates Z3 satisfiability and Ihsān constraints.
    """

    def __init__(
        self,
        ihsan_threshold: float = 0.95,
        z3_timeout_ms: int = 1000,
        constraint_validator: Optional[
            Callable[[CanonQuery, CanonPolicy], Tuple[bool, str]]
        ] = None,
    ):
        super().__init__("constraint")
        self.ihsan_threshold = ihsan_threshold
        self.z3_timeout_ms = z3_timeout_ms
        self.constraint_validator = constraint_validator

    def evaluate(
        self,
        query: CanonQuery,
        policy: CanonPolicy,
        context: Dict[str, Any],
    ) -> GateResult:
        start = time.perf_counter_ns()

        # Check Ihsān score
        ihsan_score = context.get("ihsan_score", 1.0)
        if ihsan_score < self.ihsan_threshold:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.FAILED,
                duration_us=(time.perf_counter_ns() - start) // 1000,
                reason=f"Ihsān score below threshold: {ihsan_score:.3f} < {self.ihsan_threshold}",
                evidence={
                    "ihsan_score": ihsan_score,
                    "threshold": self.ihsan_threshold,
                },
            )

        # Check Z3 satisfiability
        # CRITICAL-3 FIX: Default to False (fail-closed), not True.
        # Z3 satisfiability must be COMPUTED, not ASSUMED.
        # Standing on: ZANN_ZERO ("no assumptions")
        z3_satisfiable = context.get("z3_satisfiable", False)
        if not z3_satisfiable:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.FAILED,
                duration_us=(time.perf_counter_ns() - start) // 1000,
                reason="Z3 constraints unsatisfiable",
                evidence={"z3_satisfiable": False},
            )

        # Run custom validator if provided
        if self.constraint_validator:
            valid, error = self.constraint_validator(query, policy)
            if not valid:
                return GateResult(
                    gate_name=self.name,
                    status=GateStatus.FAILED,
                    duration_us=(time.perf_counter_ns() - start) // 1000,
                    reason=f"Constraint validation failed: {error}",
                    evidence={"validator_error": error},
                )

        return GateResult(
            gate_name=self.name,
            status=GateStatus.PASSED,
            duration_us=(time.perf_counter_ns() - start) // 1000,
            evidence={
                "ihsan_score": ihsan_score,
                "z3_satisfiable": z3_satisfiable,
            },
        )


class SafetyGate(Gate):
    """
    Gate 5: Constitutional Safety

    Enforces harm prevention and safety constraints.
    """

    def __init__(
        self,
        safety_checker: Optional[Callable[[str], Tuple[bool, str]]] = None,
        blocked_patterns: Optional[List[str]] = None,
        max_risk_score: float = 0.3,
    ):
        super().__init__("safety")
        self.safety_checker = safety_checker
        self.blocked_patterns = blocked_patterns or []
        self.max_risk_score = max_risk_score

    def evaluate(
        self,
        query: CanonQuery,
        policy: CanonPolicy,
        context: Dict[str, Any],
    ) -> GateResult:
        start = time.perf_counter_ns()
        intent_lower = query.intent.lower()

        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.lower() in intent_lower:
                return GateResult(
                    gate_name=self.name,
                    status=GateStatus.FAILED,
                    duration_us=(time.perf_counter_ns() - start) // 1000,
                    reason=f"Blocked pattern detected: {pattern}",
                    evidence={"blocked_pattern": pattern},
                )

        # Check risk score
        risk_score = context.get("risk_score", 0.0)
        if risk_score > self.max_risk_score:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.FAILED,
                duration_us=(time.perf_counter_ns() - start) // 1000,
                reason=f"Risk score exceeds threshold: {risk_score:.3f} > {self.max_risk_score}",
                evidence={"risk_score": risk_score, "threshold": self.max_risk_score},
            )

        # Run custom safety checker
        if self.safety_checker:
            safe, reason = self.safety_checker(query.intent)
            if not safe:
                return GateResult(
                    gate_name=self.name,
                    status=GateStatus.FAILED,
                    duration_us=(time.perf_counter_ns() - start) // 1000,
                    reason=f"Safety check failed: {reason}",
                    evidence={"safety_reason": reason},
                )

        return GateResult(
            gate_name=self.name,
            status=GateStatus.PASSED,
            duration_us=(time.perf_counter_ns() - start) // 1000,
            evidence={
                "risk_score": risk_score,
                "blocked_patterns_checked": len(self.blocked_patterns),
            },
        )


class CommitGate(Gate):
    """
    Gate 6: Final Commit

    Allocates resources and commits to execution.
    """

    def __init__(
        self,
        resource_checker: Optional[Callable[[], Tuple[bool, str]]] = None,
        max_concurrent_ops: int = 100,
    ):
        super().__init__("commit")
        self.resource_checker = resource_checker
        self.max_concurrent_ops = max_concurrent_ops
        self._current_ops = 0

    def evaluate(
        self,
        query: CanonQuery,
        policy: CanonPolicy,
        context: Dict[str, Any],
    ) -> GateResult:
        start = time.perf_counter_ns()

        # Check concurrent operations
        if self._current_ops >= self.max_concurrent_ops:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.FAILED,
                duration_us=(time.perf_counter_ns() - start) // 1000,
                reason=f"Max concurrent operations reached: {self._current_ops}/{self.max_concurrent_ops}",
                evidence={
                    "current_ops": self._current_ops,
                    "max_ops": self.max_concurrent_ops,
                },
            )

        # Check resources if checker provided
        if self.resource_checker:
            available, reason = self.resource_checker()
            if not available:
                return GateResult(
                    gate_name=self.name,
                    status=GateStatus.FAILED,
                    duration_us=(time.perf_counter_ns() - start) // 1000,
                    reason=f"Resources unavailable: {reason}",
                    evidence={"resource_reason": reason},
                )

        # Commit
        self._current_ops += 1

        return GateResult(
            gate_name=self.name,
            status=GateStatus.PASSED,
            duration_us=(time.perf_counter_ns() - start) // 1000,
            evidence={
                "current_ops": self._current_ops,
                "commit_id": query.hex_digest()[:16],
            },
        )

    def release(self):
        """Release a committed operation."""
        if self._current_ops > 0:
            self._current_ops -= 1


class GateChain:
    """
    The 6-gate execution chain.

    Evaluates a query through all gates in order.
    Any failure produces a signed rejection receipt.
    """

    def __init__(
        self,
        signer: SovereignSigner,
        gates: Optional[List[Gate]] = None,
        snr_policy: Optional[SNRPolicy] = None,
    ):
        self.signer = signer
        self.snr_engine = SNREngine(snr_policy)
        self.receipt_builder = ReceiptBuilder(signer)

        # Default gates
        self.gates = gates or [
            SchemaGate(),
            ProvenanceGate(),
            SNRGate(self.snr_engine),
            ConstraintGate(),
            SafetyGate(),
            CommitGate(),
        ]

        self._evaluations: List[GateChainResult] = []

    def evaluate(
        self,
        query: CanonQuery,
        policy: CanonPolicy,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GateChainResult, Receipt]:
        """
        Evaluate query through all gates.

        Returns (chain_result, receipt).
        """
        context = context or {}
        gate_results: List[GateResult] = []
        total_start = time.perf_counter_ns()

        last_gate_passed = "none"
        final_status = GateStatus.PENDING
        rejection_reason = None
        snr = 0.0
        ihsan_score = context.get("ihsan_score", 0.95)
        snr_trace = None

        for gate in self.gates:
            result = gate.evaluate(query, policy, context)
            gate_results.append(result)

            if result.passed:
                last_gate_passed = gate.name

                # Extract SNR info if this was SNR gate
                if gate.name == "snr" and "snr" in result.evidence:
                    snr = result.evidence["snr"]
                    if "trace" in result.evidence:
                        snr_trace = SNRTrace(**result.evidence["trace"])
            else:
                # Gate failed - stop chain
                final_status = GateStatus.FAILED
                rejection_reason = result.reason
                break

        # All gates passed
        if final_status == GateStatus.PENDING:
            final_status = GateStatus.PASSED

        total_duration_us = (time.perf_counter_ns() - total_start) // 1000

        chain_result = GateChainResult(
            query=query,
            policy=policy,
            gate_results=gate_results,
            final_status=final_status,
            last_gate_passed=last_gate_passed,
            rejection_reason=rejection_reason,
            total_duration_us=total_duration_us,
            snr=snr,
            ihsan_score=ihsan_score,
            snr_trace=snr_trace,
        )

        self._evaluations.append(chain_result)

        # Generate receipt
        metrics = Metrics(
            p99_us=total_duration_us,
            duration_ms=total_duration_us / 1000,
        )

        if final_status == GateStatus.PASSED:
            receipt = self.receipt_builder.accepted(
                query=query,
                policy=policy,
                payload=query.canonical_bytes(),
                snr=snr,
                ihsan_score=ihsan_score,
                gate_passed="commit",
                metrics=metrics,
                snr_trace=snr_trace,
            )
        else:
            receipt = self.receipt_builder.rejected(
                query=query,
                policy=policy,
                snr=snr,
                ihsan_score=ihsan_score,
                gate_failed=last_gate_passed,
                reason=rejection_reason or "Unknown failure",
                metrics=metrics,
                snr_trace=snr_trace,
            )

        return chain_result, receipt

    def evaluate_with_amber(
        self,
        query: CanonQuery,
        policy: CanonPolicy,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GateChainResult, Receipt]:
        """
        Evaluate with amber-restricted fallback.

        If safety gate fails but SNR is high, may return amber receipt.
        """
        context = context or {}
        result, receipt = self.evaluate(query, policy, context)

        # Check for amber eligibility
        if (
            not result.passed
            and result.last_gate_passed == "constraint"  # Failed at safety
            and result.snr >= 0.90  # High SNR
        ):
            # Create amber receipt
            metrics = Metrics(
                p99_us=result.total_duration_us,
                duration_ms=result.total_duration_us / 1000,
            )

            receipt = self.receipt_builder.amber_restricted(
                query=query,
                policy=policy,
                payload=query.canonical_bytes(),
                snr=result.snr,
                ihsan_score=result.ihsan_score,
                restriction_reason=result.rejection_reason or "Safety concerns",
                gate_passed="constraint",
                metrics=metrics,
                snr_trace=result.snr_trace,
            )

        return result, receipt

    def get_stats(self) -> Dict[str, Any]:
        """Get chain statistics."""
        if not self._evaluations:
            return {
                "total_evaluations": 0,
                "gates": [g.name for g in self.gates],
            }

        passed = sum(1 for e in self._evaluations if e.passed)
        failed = len(self._evaluations) - passed

        # Gate failure distribution
        failure_by_gate: Dict[str, int] = {}
        for e in self._evaluations:
            if not e.passed:
                gate = e.last_gate_passed
                failure_by_gate[gate] = failure_by_gate.get(gate, 0) + 1

        return {
            "total_evaluations": len(self._evaluations),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self._evaluations),
            "avg_duration_us": sum(e.total_duration_us for e in self._evaluations)
            / len(self._evaluations),
            "avg_snr": sum(e.snr for e in self._evaluations) / len(self._evaluations),
            "failure_by_gate": failure_by_gate,
            "gates": [g.name for g in self.gates],
        }

    def get_recent_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent evaluation results."""
        return [e.to_dict() for e in self._evaluations[-limit:]]
