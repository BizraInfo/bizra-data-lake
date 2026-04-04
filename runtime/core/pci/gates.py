"""
BIZRA PCI Protocol — Verification Gate Chain
=============================================
Tiered chain of responsibility with fail-fast semantics.

Status: FROZEN — Changes require version bump + test vector update
Semantics: First failure terminates chain (fail-closed)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .crypto import (
    check_nonce_replay,
    validate_nonce,
)
from .envelope import PCIEnvelope, validate_envelope_schema
from .reject_codes import (
    RejectCode,
    RejectionResponse,
    reject_fate_violation,
    reject_ihsan,
    reject_internal_error,
    reject_replay,
    reject_role_violation,
    reject_schema,
    reject_signature,
    reject_snr,
    reject_timestamp_future,
    reject_timestamp_stale,
)
from .types import (
    IHSAN_THRESHOLD,
    LATENCY_BUDGET_CHEAP,
    LATENCY_BUDGET_EXPENSIVE,
    LATENCY_BUDGET_MEDIUM,
    SNR_THRESHOLD_DEFAULT,
    TIMESTAMP_SKEW_SECONDS,
    AgentType,
    AuditTrail,
    Gate,
    GateTier,
    utc_now_iso,
)

# Import SNR enforcer (optional, falls back to direct threshold check)
try:
    from bizra_kernel.snr_enforcer import (
        get_snr_enforcer,
        OperationType,
        EnforcementContext,
    )

    SNR_ENFORCER_AVAILABLE = True
except ImportError:
    SNR_ENFORCER_AVAILABLE = False


# =============================================================================
# GATE RESULT
# =============================================================================


@dataclass
class GateResult:
    """Result of a single gate execution."""

    gate: Gate
    passed: bool
    latency_ms: float
    rejection: Optional[RejectionResponse] = None
    details: Optional[Dict[str, Any]] = None


# =============================================================================
# GATE CHAIN
# =============================================================================


class GateChain:
    """
    Verification gate chain with tiered execution.

    Tiers:
    - CHEAP (<10ms): SCHEMA, SIGNATURE, TIMESTAMP, REPLAY, ROLE
    - MEDIUM (<150ms): SNR, IHSAN, POLICY
    - EXPENSIVE (<2000ms): FATE, FORMAL

    Semantics:
    - Gates execute in strict order
    - First failure terminates chain (fail-fast)
    - Any timeout → REJECT_BUDGET_EXCEEDED
    - Any error → REJECT_INTERNAL_ERROR (fail-closed)
    """

    def __init__(
        self,
        current_policy_hash: str,
        current_state_hash: str,
        ihsan_threshold: float = IHSAN_THRESHOLD,
        snr_threshold: float = SNR_THRESHOLD_DEFAULT,
        fate_checker: Optional[
            Callable[[PCIEnvelope], Tuple[bool, str, Dict[str, Any]]]
        ] = None,
        formal_verifier: Optional[
            Callable[[PCIEnvelope], Tuple[bool, str, Dict[str, Any]]]
        ] = None,
        use_snr_enforcer: bool = True,
    ):
        """
        Initialize the gate chain.

        Args:
            current_policy_hash: BLAKE3 hash of current constitution
            current_state_hash: BLAKE3 hash of current state
            ihsan_threshold: Minimum Ihsān score (default 0.95)
            snr_threshold: Minimum SNR score for tier
            fate_checker: Optional FATE invariant checker
            formal_verifier: Optional formal verification function
            use_snr_enforcer: Whether to use SNREnforcer (with receipt emission)
        """
        self.current_policy_hash = current_policy_hash
        self.current_state_hash = current_state_hash
        self.ihsan_threshold = ihsan_threshold
        self.snr_threshold = snr_threshold
        self.fate_checker = fate_checker
        self.formal_verifier = formal_verifier
        self.use_snr_enforcer = use_snr_enforcer and SNR_ENFORCER_AVAILABLE

        # Initialize SNR enforcer if available
        if self.use_snr_enforcer:
            try:
                self.snr_enforcer = get_snr_enforcer()
            except Exception:
                # Fail gracefully, fall back to direct threshold check
                self.snr_enforcer = None
                self.use_snr_enforcer = False
        else:
            self.snr_enforcer = None

        self._gates_passed: List[Gate] = []
        self._total_latency_ms: float = 0.0

    def _time_ms(self) -> float:
        """Get current time in milliseconds."""
        return time.perf_counter() * 1000

    def _check_budget(self, tier: GateTier, elapsed_ms: float) -> bool:
        """Check if we're within the latency budget for the tier."""
        budgets = {
            GateTier.CHEAP: LATENCY_BUDGET_CHEAP,
            GateTier.MEDIUM: LATENCY_BUDGET_MEDIUM,
            GateTier.EXPENSIVE: LATENCY_BUDGET_EXPENSIVE,
        }
        return elapsed_ms <= budgets.get(tier, LATENCY_BUDGET_MEDIUM)

    def verify(
        self,
        envelope: PCIEnvelope,
        require_expensive: bool = False,
    ) -> Tuple[bool, Optional[RejectionResponse], List[GateResult]]:
        """
        Execute the full gate chain.

        Args:
            envelope: The envelope to verify
            require_expensive: If True, also run EXPENSIVE tier gates

        Returns:
            (passed, rejection_response, gate_results)
        """
        envelope_data = envelope.to_dict()
        digest = envelope.compute_digest()
        timestamp = utc_now_iso()

        results: List[GateResult] = []

        try:
            # CHEAP TIER (<10ms)
            cheap_gates = [
                (
                    Gate.SCHEMA,
                    lambda: self._gate_schema(envelope_data, digest, timestamp),
                ),
                (
                    Gate.SIGNATURE,
                    lambda: self._gate_signature(envelope, digest, timestamp),
                ),
                (
                    Gate.TIMESTAMP,
                    lambda: self._gate_timestamp(envelope, digest, timestamp),
                ),
                (Gate.REPLAY, lambda: self._gate_replay(envelope, digest, timestamp)),
                (Gate.ROLE, lambda: self._gate_role(envelope, digest, timestamp)),
            ]

            tier_start = self._time_ms()
            for gate, check_fn in cheap_gates:
                gate_start = self._time_ms()
                result = check_fn()
                result.latency_ms = self._time_ms() - gate_start
                results.append(result)

                if not result.passed:
                    return False, result.rejection, results

                self._gates_passed.append(gate)

            cheap_elapsed = self._time_ms() - tier_start
            if not self._check_budget(GateTier.CHEAP, cheap_elapsed):
                rejection = RejectionResponse.rejection(
                    code=RejectCode.REJECT_BUDGET_EXCEEDED,
                    message=f"CHEAP tier exceeded budget: {cheap_elapsed:.1f}ms > {LATENCY_BUDGET_CHEAP}ms",
                    envelope_digest=digest,
                    timestamp=timestamp,
                    audit_trail=AuditTrail(
                        gate=Gate.SCHEMA,
                        tier=GateTier.CHEAP,
                        latency_ms=cheap_elapsed,
                        details={"budget_ms": LATENCY_BUDGET_CHEAP},
                    ),
                )
                return False, rejection, results

            # MEDIUM TIER (<150ms)
            medium_gates = [
                (Gate.SNR, lambda: self._gate_snr(envelope, digest, timestamp)),
                (Gate.IHSAN, lambda: self._gate_ihsan(envelope, digest, timestamp)),
                (Gate.POLICY, lambda: self._gate_policy(envelope, digest, timestamp)),
            ]

            tier_start = self._time_ms()
            for gate, check_fn in medium_gates:
                gate_start = self._time_ms()
                result = check_fn()
                result.latency_ms = self._time_ms() - gate_start
                results.append(result)

                if not result.passed:
                    return False, result.rejection, results

                self._gates_passed.append(gate)

            medium_elapsed = self._time_ms() - tier_start
            if not self._check_budget(GateTier.MEDIUM, medium_elapsed):
                rejection = RejectionResponse.rejection(
                    code=RejectCode.REJECT_BUDGET_EXCEEDED,
                    message=f"MEDIUM tier exceeded budget: {medium_elapsed:.1f}ms > {LATENCY_BUDGET_MEDIUM}ms",
                    envelope_digest=digest,
                    timestamp=timestamp,
                    audit_trail=AuditTrail(
                        gate=Gate.SNR,
                        tier=GateTier.MEDIUM,
                        latency_ms=medium_elapsed,
                        details={"budget_ms": LATENCY_BUDGET_MEDIUM},
                    ),
                )
                return False, rejection, results

            # EXPENSIVE TIER (<2000ms) - only if required
            if require_expensive:
                expensive_gates = [
                    (Gate.FATE, lambda: self._gate_fate(envelope, digest, timestamp)),
                    (
                        Gate.FORMAL,
                        lambda: self._gate_formal(envelope, digest, timestamp),
                    ),
                ]

                tier_start = self._time_ms()
                for gate, check_fn in expensive_gates:
                    gate_start = self._time_ms()
                    result = check_fn()
                    result.latency_ms = self._time_ms() - gate_start
                    results.append(result)

                    if not result.passed:
                        return False, result.rejection, results

                    self._gates_passed.append(gate)

                expensive_elapsed = self._time_ms() - tier_start
                if not self._check_budget(GateTier.EXPENSIVE, expensive_elapsed):
                    rejection = RejectionResponse.rejection(
                        code=RejectCode.REJECT_BUDGET_EXCEEDED,
                        message=f"EXPENSIVE tier exceeded budget: {expensive_elapsed:.1f}ms > {LATENCY_BUDGET_EXPENSIVE}ms",
                        envelope_digest=digest,
                        timestamp=timestamp,
                        audit_trail=AuditTrail(
                            gate=Gate.FATE,
                            tier=GateTier.EXPENSIVE,
                            latency_ms=expensive_elapsed,
                            details={"budget_ms": LATENCY_BUDGET_EXPENSIVE},
                        ),
                    )
                    return False, rejection, results

            # All gates passed
            self._total_latency_ms = sum(r.latency_ms for r in results)
            return True, None, results

        except Exception as e:
            # Fail-closed on any error
            rejection = reject_internal_error(digest, timestamp, str(e))
            return False, rejection, results

    def get_gates_passed(self) -> List[Gate]:
        """Get the list of gates that passed."""
        return list(self._gates_passed)

    def get_total_latency_ms(self) -> float:
        """Get the total verification latency."""
        return self._total_latency_ms

    # =========================================================================
    # INDIVIDUAL GATE IMPLEMENTATIONS
    # =========================================================================

    def _gate_schema(
        self,
        envelope_data: Dict[str, Any],
        digest: str,
        timestamp: str,
    ) -> GateResult:
        """SCHEMA gate: Validate envelope structure."""
        errors = validate_envelope_schema(envelope_data)

        if errors:
            return GateResult(
                gate=Gate.SCHEMA,
                passed=False,
                latency_ms=0.0,
                rejection=reject_schema(digest, timestamp, "; ".join(errors)),
            )

        return GateResult(gate=Gate.SCHEMA, passed=True, latency_ms=0.0)

    def _gate_signature(
        self,
        envelope: PCIEnvelope,
        digest: str,
        timestamp: str,
    ) -> GateResult:
        """SIGNATURE gate: Verify Ed25519 signature."""
        if not envelope.verify_signature():
            return GateResult(
                gate=Gate.SIGNATURE,
                passed=False,
                latency_ms=0.0,
                rejection=reject_signature(digest, timestamp),
            )

        return GateResult(gate=Gate.SIGNATURE, passed=True, latency_ms=0.0)

    def _gate_timestamp(
        self,
        envelope: PCIEnvelope,
        digest: str,
        timestamp: str,
    ) -> GateResult:
        """TIMESTAMP gate: Check freshness (±120s skew)."""
        try:
            envelope_dt = datetime.fromisoformat(
                envelope.timestamp.replace("Z", "+00:00")
            )
            now_dt = datetime.now(timezone.utc)
            skew_seconds = (now_dt - envelope_dt).total_seconds()

            if skew_seconds > TIMESTAMP_SKEW_SECONDS:
                return GateResult(
                    gate=Gate.TIMESTAMP,
                    passed=False,
                    latency_ms=0.0,
                    rejection=reject_timestamp_stale(
                        digest, timestamp, envelope.timestamp, skew_seconds
                    ),
                )

            if skew_seconds < -TIMESTAMP_SKEW_SECONDS:
                return GateResult(
                    gate=Gate.TIMESTAMP,
                    passed=False,
                    latency_ms=0.0,
                    rejection=reject_timestamp_future(
                        digest, timestamp, envelope.timestamp, abs(skew_seconds)
                    ),
                )

            return GateResult(gate=Gate.TIMESTAMP, passed=True, latency_ms=0.0)

        except Exception as e:
            return GateResult(
                gate=Gate.TIMESTAMP,
                passed=False,
                latency_ms=0.0,
                rejection=reject_internal_error(
                    digest, timestamp, f"Timestamp parse error: {e}"
                ),
            )

    def _gate_replay(
        self,
        envelope: PCIEnvelope,
        digest: str,
        timestamp: str,
    ) -> GateResult:
        """REPLAY gate: Check nonce not reused."""
        if not validate_nonce(envelope.nonce):
            return GateResult(
                gate=Gate.REPLAY,
                passed=False,
                latency_ms=0.0,
                rejection=reject_schema(digest, timestamp, "Invalid nonce format"),
            )

        if not check_nonce_replay(envelope.nonce):
            return GateResult(
                gate=Gate.REPLAY,
                passed=False,
                latency_ms=0.0,
                rejection=reject_replay(digest, timestamp, envelope.nonce),
            )

        return GateResult(gate=Gate.REPLAY, passed=True, latency_ms=0.0)

    def _gate_role(
        self,
        envelope: PCIEnvelope,
        digest: str,
        timestamp: str,
    ) -> GateResult:
        """ROLE gate: Check agent has permission for action."""
        # PAT can propose, SAT can commit
        # Define forbidden actions per agent type
        pat_forbidden = ["commit", "issue_receipt", "modify_state"]
        sat_forbidden = ["propose"]  # SAT shouldn't propose, only verify

        action = envelope.payload.action.lower()
        agent_type = envelope.sender.agent_type

        if agent_type == AgentType.PAT:
            if any(forbidden in action for forbidden in pat_forbidden):
                return GateResult(
                    gate=Gate.ROLE,
                    passed=False,
                    latency_ms=0.0,
                    rejection=reject_role_violation(digest, timestamp, "PAT", action),
                )
        elif agent_type == AgentType.SAT:
            if any(forbidden in action for forbidden in sat_forbidden):
                return GateResult(
                    gate=Gate.ROLE,
                    passed=False,
                    latency_ms=0.0,
                    rejection=reject_role_violation(digest, timestamp, "SAT", action),
                )

        return GateResult(gate=Gate.ROLE, passed=True, latency_ms=0.0)

    def _gate_snr(
        self,
        envelope: PCIEnvelope,
        digest: str,
        timestamp: str,
    ) -> GateResult:
        """SNR gate: Check signal-to-noise ratio threshold."""
        snr = envelope.metadata.snr_score

        # Use SNR enforcer if available (provides receipt emission)
        if self.use_snr_enforcer and self.snr_enforcer:
            try:
                # Determine operation type from envelope
                operation_type = self._infer_operation_type(envelope)

                # Create enforcement context
                context = EnforcementContext(
                    operation_type=operation_type,
                    agent_id=envelope.sender.agent_id,
                    snr_score=snr,
                    task_id=envelope.envelope_id,
                    details={
                        "action": envelope.payload.action,
                        "envelope_digest": digest,
                        "agent_type": envelope.sender.agent_type.value,
                    },
                )

                # Enforce threshold
                result = self.snr_enforcer.enforce(context)

                if not result.passed:
                    # Rejection with receipt already emitted
                    return GateResult(
                        gate=Gate.SNR,
                        passed=False,
                        latency_ms=0.0,
                        rejection=reject_snr(digest, timestamp, snr, result.threshold),
                        details={
                            "receipt_id": result.receipt_id,
                            "enforcer_used": True,
                        },
                    )

                return GateResult(
                    gate=Gate.SNR,
                    passed=True,
                    latency_ms=0.0,
                    details={"enforcer_used": True},
                )

            except Exception:
                # Fall back to direct threshold check on error
                pass

        # Direct threshold check (fallback or when enforcer disabled)
        if snr < self.snr_threshold:
            return GateResult(
                gate=Gate.SNR,
                passed=False,
                latency_ms=0.0,
                rejection=reject_snr(digest, timestamp, snr, self.snr_threshold),
                details={"enforcer_used": False},
            )

        return GateResult(
            gate=Gate.SNR,
            passed=True,
            latency_ms=0.0,
            details={"enforcer_used": False},
        )

    def _infer_operation_type(self, envelope: PCIEnvelope) -> "OperationType":
        """Infer operation type from envelope action."""
        action = envelope.payload.action.lower()

        # Map actions to operation types
        if "reason" in action or "analyze" in action:
            return OperationType.REASONING
        elif "synthesize" in action or "generate" in action:
            return OperationType.SYNTHESIS
        elif "validate" in action or "verify" in action:
            return OperationType.VALIDATION
        elif "retrieve" in action or "query" in action:
            return OperationType.RETRIEVAL
        elif "probe" in action or "sape" in action:
            return OperationType.SAPE_PROBE
        elif envelope.sender.agent_type.value == "PAT":
            return OperationType.PAT_EXECUTION
        elif envelope.sender.agent_type.value == "SAT":
            return OperationType.SAT_VALIDATION
        else:
            return OperationType.DEFAULT

    def _gate_ihsan(
        self,
        envelope: PCIEnvelope,
        digest: str,
        timestamp: str,
    ) -> GateResult:
        """IHSAN gate: Check Ihsān score ≥ threshold."""
        ihsan = envelope.metadata.ihsan_score

        if ihsan < self.ihsan_threshold:
            return GateResult(
                gate=Gate.IHSAN,
                passed=False,
                latency_ms=0.0,
                rejection=reject_ihsan(digest, timestamp, ihsan, self.ihsan_threshold),
            )

        return GateResult(gate=Gate.IHSAN, passed=True, latency_ms=0.0)

    def _gate_policy(
        self,
        envelope: PCIEnvelope,
        digest: str,
        timestamp: str,
    ) -> GateResult:
        """POLICY gate: Check policy_hash matches current constitution."""
        if envelope.payload.policy_hash != self.current_policy_hash:
            return GateResult(
                gate=Gate.POLICY,
                passed=False,
                latency_ms=0.0,
                rejection=RejectionResponse.rejection(
                    code=RejectCode.REJECT_POLICY_MISMATCH,
                    message="policy_hash doesn't match current constitution",
                    envelope_digest=digest,
                    timestamp=timestamp,
                    audit_trail=AuditTrail(
                        gate=Gate.POLICY,
                        tier=GateTier.MEDIUM,
                        latency_ms=0.0,
                        details={
                            "expected": self.current_policy_hash[:16] + "...",
                            "received": envelope.payload.policy_hash[:16] + "...",
                        },
                    ),
                ),
            )

        return GateResult(gate=Gate.POLICY, passed=True, latency_ms=0.0)

    def _gate_fate(
        self,
        envelope: PCIEnvelope,
        digest: str,
        timestamp: str,
    ) -> GateResult:
        """FATE gate: Check FATE invariants (SMT/Z3)."""
        if self.fate_checker is None:
            # No FATE checker configured, pass
            return GateResult(gate=Gate.FATE, passed=True, latency_ms=0.0)

        try:
            passed, invariant, details = self.fate_checker(envelope)

            if not passed:
                return GateResult(
                    gate=Gate.FATE,
                    passed=False,
                    latency_ms=0.0,
                    rejection=reject_fate_violation(
                        digest, timestamp, invariant, details
                    ),
                )

            return GateResult(gate=Gate.FATE, passed=True, latency_ms=0.0)

        except Exception as e:
            return GateResult(
                gate=Gate.FATE,
                passed=False,
                latency_ms=0.0,
                rejection=reject_internal_error(digest, timestamp, f"FATE error: {e}"),
            )

    def _gate_formal(
        self,
        envelope: PCIEnvelope,
        digest: str,
        timestamp: str,
    ) -> GateResult:
        """FORMAL gate: Mathematical proof verification."""
        if self.formal_verifier is None:
            # No formal verifier configured, pass
            return GateResult(gate=Gate.FORMAL, passed=True, latency_ms=0.0)

        try:
            passed, invariant, details = self.formal_verifier(envelope)

            if not passed:
                return GateResult(
                    gate=Gate.FORMAL,
                    passed=False,
                    latency_ms=0.0,
                    rejection=RejectionResponse.rejection(
                        code=RejectCode.REJECT_INVARIANT_FAILED,
                        message=f"Formal invariant failed: {invariant}",
                        envelope_digest=digest,
                        timestamp=timestamp,
                        audit_trail=AuditTrail(
                            gate=Gate.FORMAL,
                            tier=GateTier.EXPENSIVE,
                            latency_ms=0.0,
                            details={"invariant": invariant, **details},
                        ),
                    ),
                )

            return GateResult(gate=Gate.FORMAL, passed=True, latency_ms=0.0)

        except Exception as e:
            return GateResult(
                gate=Gate.FORMAL,
                passed=False,
                latency_ms=0.0,
                rejection=reject_internal_error(
                    digest, timestamp, f"Formal verifier error: {e}"
                ),
            )


# =============================================================================
# HIGH-LEVEL VERIFICATION API
# =============================================================================


def verify_envelope(
    envelope: PCIEnvelope,
    policy_hash: str,
    state_hash: str,
    ihsan_threshold: float = IHSAN_THRESHOLD,
    snr_threshold: float = SNR_THRESHOLD_DEFAULT,
    require_expensive: bool = False,
) -> Tuple[bool, Optional[RejectionResponse], List[GateResult]]:
    """
    Verify an envelope through the gate chain.

    Args:
        envelope: The envelope to verify
        policy_hash: Current constitution hash
        state_hash: Current state hash
        ihsan_threshold: Minimum Ihsān score
        snr_threshold: Minimum SNR score
        require_expensive: Whether to run EXPENSIVE tier

    Returns:
        (passed, rejection_response, gate_results)
    """
    chain = GateChain(
        current_policy_hash=policy_hash,
        current_state_hash=state_hash,
        ihsan_threshold=ihsan_threshold,
        snr_threshold=snr_threshold,
    )

    return chain.verify(envelope, require_expensive=require_expensive)
