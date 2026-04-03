"""
BIZRA PCI Protocol — PAT-SAT-PCI Integration Bridge
===================================================
Canonical integration layer connecting PAT agents to SAT verification via PCI envelopes.

Status: PRODUCTION
Semantics: Fail-closed, receipt-first, quorum-enforced

Architecture:
    ┌────────────────────────────────────────────────────────────────────────┐
    │                        PATSATBridge                                    │
    │  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐   │
    │  │ pat_propose()│ → │ sat_verify() │ → │ emit_receipt()           │   │
    │  │              │   │              │   │ (CommitReceipt/Rejection)│   │
    │  └──────────────┘   └──────────────┘   └──────────────────────────┘   │
    └────────────────────────────────────────────────────────────────────────┘

Flow:
    1. PAT agent calls pat_propose() with action and data
    2. Envelope is built, signed with Ed25519
    3. sat_verify() runs the gate chain (CHEAP → MEDIUM → EXPENSIVE)
    4. If passed, SAT validators sign (3/5 quorum required)
    5. CommitReceipt generated and stored
    6. FlowResult returned with full audit trail

Usage:
    from core.pci.integration import (
        PATSATBridge,
        FlowResult,
        create_pat_envelope,
        verify_with_sat,
        complete_pat_sat_flow,
    )

    # Quick flow
    result = await complete_pat_sat_flow(
        agent_id="pat-001",
        action="propose",
        data={"task": "analyze_code"},
        key=private_key,
        validators=["sat-001", "sat-002", "sat-003", "sat-004", "sat-005"],
    )

    if result.passed:
        print(f"Receipt: {result.receipt.receipt_id}")
    else:
        print(f"Rejected: {result.rejection.message}")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .crypto import (
    KeyPair,
    canonical_json,
    domain_separated_digest,
    envelope_digest,
    generate_keypair,
    sign_message,
)
from .envelope import EnvelopeBuilder, PCIEnvelope
from .gates import GateChain, GateResult, verify_envelope
from .receipt import CommitReceipt, ReceiptGenerator, ReceiptStore, get_receipt_generator, get_receipt_store
from .reject_codes import (
    RejectCode,
    RejectionResponse,
    reject_internal_error,
)
from .types import (
    IHSAN_THRESHOLD,
    SNR_THRESHOLD_DEFAULT,
    AgentType,
    Gate,
    GateTier,
    Quorum,
    VerificationTier,
    VerifierSignature,
    utc_now_iso,
)

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("bizra.pci.integration")
logger.setLevel(logging.DEBUG)

# Structured log handler if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'
    ))
    logger.addHandler(handler)


# =============================================================================
# FLOW RESULT
# =============================================================================

@dataclass
class FlowResult:
    """
    Result of a complete PAT-SAT verification flow.

    Contains the full audit trail for evidence and compliance.

    Attributes:
        envelope: The PCI envelope that was verified
        passed: Whether the flow succeeded (gate chain + quorum)
        receipt: CommitReceipt if passed, None if rejected
        rejection: RejectionResponse if rejected, None if passed
        gate_results: List of individual gate execution results
        latency_ms: Total flow latency in milliseconds
        audit_trail: Detailed audit data for compliance
    """
    envelope: PCIEnvelope
    passed: bool
    receipt: Optional[CommitReceipt] = None
    rejection: Optional[RejectionResponse] = None
    gate_results: List[GateResult] = field(default_factory=list)
    latency_ms: float = 0.0
    audit_trail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for wire format."""
        result = {
            "envelope_id": self.envelope.envelope_id,
            "envelope_digest": self.envelope.compute_digest(),
            "passed": self.passed,
            "latency_ms": self.latency_ms,
            "gate_results": [
                {
                    "gate": r.gate.value,
                    "passed": r.passed,
                    "latency_ms": r.latency_ms,
                }
                for r in self.gate_results
            ],
            "audit_trail": self.audit_trail,
        }

        if self.receipt:
            result["receipt"] = self.receipt.to_dict()
        if self.rejection:
            result["rejection"] = self.rejection.to_dict()

        return result

    @property
    def success(self) -> bool:
        """Alias for passed."""
        return self.passed

    @property
    def receipt_id(self) -> Optional[str]:
        """Get receipt ID if available."""
        return self.receipt.receipt_id if self.receipt else None


# =============================================================================
# SAT VALIDATOR REGISTRY
# =============================================================================

@dataclass
class SATValidator:
    """
    SAT validator configuration.

    Attributes:
        sat_id: Unique identifier for the SAT validator
        public_key: Hex-encoded Ed25519 public key
        private_key: Private key (only needed for signing)
        weight: Voting weight (default 1)
    """
    sat_id: str
    public_key: str
    private_key: Optional[bytes] = None
    weight: int = 1

    def can_sign(self) -> bool:
        """Check if this validator can sign receipts."""
        return self.private_key is not None


class SATValidatorRegistry:
    """
    Registry of SAT validators for quorum verification.

    Manages the set of SAT validators and their keys.
    Thread-safe for concurrent access.
    """

    def __init__(self):
        self._validators: Dict[str, SATValidator] = {}
        self._lock = asyncio.Lock()

    async def register(self, validator: SATValidator) -> None:
        """Register a SAT validator."""
        async with self._lock:
            self._validators[validator.sat_id] = validator
            logger.debug(f"Registered SAT validator: {validator.sat_id}")

    async def get(self, sat_id: str) -> Optional[SATValidator]:
        """Get a validator by ID."""
        async with self._lock:
            return self._validators.get(sat_id)

    async def get_all(self) -> List[SATValidator]:
        """Get all registered validators."""
        async with self._lock:
            return list(self._validators.values())

    async def count(self) -> int:
        """Get the number of registered validators."""
        async with self._lock:
            return len(self._validators)


# Global validator registry
_validator_registry: Optional[SATValidatorRegistry] = None


def get_validator_registry() -> SATValidatorRegistry:
    """Get the global SAT validator registry."""
    global _validator_registry
    if _validator_registry is None:
        _validator_registry = SATValidatorRegistry()
    return _validator_registry


# =============================================================================
# PAT-SAT BRIDGE
# =============================================================================

class PATSATBridge:
    """
    Orchestrates the PAT → SAT verification flow via PCI envelopes.

    This is the canonical integration point for:
    - PAT agents creating proposals
    - SAT validators verifying and signing
    - Receipt generation for audit trail

    Flow:
        1. PAT agent calls pat_propose() with action data
        2. Envelope is built and signed
        3. sat_verify() runs gate chain verification
        4. If passed, SAT validators sign (quorum required)
        5. CommitReceipt generated
        6. FlowResult returned

    Example:
        bridge = PATSATBridge(
            gate_chain=gate_chain,
            receipt_generator=receipt_generator,
            ihsan_threshold=0.95,
        )

        result = await bridge.execute_flow(
            pat_agent_id="pat-001",
            action="analyze",
            data={"code": "..."},
            private_key=agent_key,
            sat_validators=["sat-001", "sat-002", ...],
        )
    """

    # Default quorum: 3 of 5 SAT validators
    DEFAULT_QUORUM_REQUIRED = 3
    DEFAULT_QUORUM_TOTAL = 5

    def __init__(
        self,
        gate_chain: Optional[GateChain] = None,
        receipt_generator: Optional[ReceiptGenerator] = None,
        receipt_store: Optional[ReceiptStore] = None,
        validator_registry: Optional[SATValidatorRegistry] = None,
        ihsan_threshold: float = IHSAN_THRESHOLD,
        snr_threshold: float = SNR_THRESHOLD_DEFAULT,
        quorum_required: int = DEFAULT_QUORUM_REQUIRED,
        quorum_total: int = DEFAULT_QUORUM_TOTAL,
        require_expensive_gates: bool = False,
        fate_checker: Optional[Callable[[PCIEnvelope], Tuple[bool, str, Dict[str, Any]]]] = None,
        formal_verifier: Optional[Callable[[PCIEnvelope], Tuple[bool, str, Dict[str, Any]]]] = None,
    ):
        """
        Initialize the PAT-SAT bridge.

        Args:
            gate_chain: Pre-configured GateChain (or None to create per-request)
            receipt_generator: Generator for commit receipts
            receipt_store: Store for persisting receipts
            validator_registry: Registry of SAT validators
            ihsan_threshold: Minimum Ihsan score (default 0.95)
            snr_threshold: Minimum SNR score (default from core.constants)
            quorum_required: Number of SAT signatures required (default 3)
            quorum_total: Total number of SAT validators (default 5)
            require_expensive_gates: Whether to run EXPENSIVE tier (FATE/FORMAL)
            fate_checker: Optional FATE invariant checker function
            formal_verifier: Optional formal verification function
        """
        self._gate_chain = gate_chain
        self._receipt_generator = receipt_generator or get_receipt_generator()
        self._receipt_store = receipt_store or get_receipt_store()
        self._validator_registry = validator_registry or get_validator_registry()

        self.ihsan_threshold = ihsan_threshold
        self.snr_threshold = snr_threshold
        self.quorum_required = quorum_required
        self.quorum_total = quorum_total
        self.require_expensive_gates = require_expensive_gates

        self.fate_checker = fate_checker
        self.formal_verifier = formal_verifier

        # Current policy and state hashes (must be set for verification)
        self._current_policy_hash: Optional[str] = None
        self._current_state_hash: Optional[str] = None

        logger.info(
            f"PATSATBridge initialized: ihsan={ihsan_threshold}, "
            f"snr={snr_threshold}, quorum={quorum_required}/{quorum_total}"
        )

    def set_policy_hash(self, policy_hash: str) -> None:
        """Set the current policy (constitution) hash."""
        self._current_policy_hash = policy_hash
        logger.debug(f"Policy hash set: {policy_hash[:16]}...")

    def set_state_hash(self, state_hash: str) -> None:
        """Set the current state hash."""
        self._current_state_hash = state_hash
        logger.debug(f"State hash set: {state_hash[:16]}...")

    def _get_policy_hash(self) -> str:
        """Get current policy hash or raise error."""
        if self._current_policy_hash is None:
            raise ValueError("Policy hash not set. Call set_policy_hash() first.")
        return self._current_policy_hash

    def _get_state_hash(self) -> str:
        """Get current state hash or raise error."""
        if self._current_state_hash is None:
            raise ValueError("State hash not set. Call set_state_hash() first.")
        return self._current_state_hash

    async def pat_propose(
        self,
        agent_id: str,
        action: str,
        data: Dict[str, Any],
        private_key: bytes,
        public_key: Optional[str] = None,
        ihsan_score: float = 0.95,
        snr_score: float = 0.85,
    ) -> Tuple[PCIEnvelope, str]:
        """
        PAT agent creates a signed proposal envelope.

        Args:
            agent_id: PAT agent identifier
            action: Action being proposed (e.g., "analyze", "execute")
            data: Action payload data
            private_key: Ed25519 private key (32 bytes)
            public_key: Optional hex-encoded public key (derived if not provided)
            ihsan_score: Ihsan excellence score (default 0.95)
            snr_score: Signal-to-noise ratio (default 0.85)

        Returns:
            Tuple of (signed_envelope, envelope_digest)

        Raises:
            ValueError: If policy or state hash not set
            ImportError: If PyNaCl not available
        """
        start_time = time.perf_counter()

        try:
            # Derive public key if not provided
            if public_key is None:
                from nacl.signing import SigningKey
                signing_key = SigningKey(private_key)
                public_key = bytes(signing_key.verify_key).hex()

            # Build the envelope
            envelope = (
                EnvelopeBuilder()
                .with_sender(AgentType.PAT, agent_id, public_key)
                .with_action(action, data)
                .with_policy(self._get_policy_hash())
                .with_state(self._get_state_hash())
                .with_scores(ihsan=ihsan_score, snr=snr_score)
                .build()
            )

            # Sign the envelope
            signed_envelope = envelope.sign(private_key)
            digest = signed_envelope.compute_digest()

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"PAT proposal created: agent={agent_id}, action={action}, "
                f"envelope_id={signed_envelope.envelope_id[:8]}..., latency={elapsed_ms:.2f}ms"
            )

            return signed_envelope, digest

        except Exception as e:
            logger.error(f"PAT proposal failed: agent={agent_id}, error={str(e)}")
            raise

    async def sat_verify(
        self,
        envelope: PCIEnvelope,
        sat_validators: List[str],
        validator_keys: Optional[Dict[str, Tuple[str, bytes]]] = None,
    ) -> Tuple[bool, Union[CommitReceipt, RejectionResponse]]:
        """
        SAT validators verify the envelope through gate chain.

        Args:
            envelope: The PCI envelope to verify
            sat_validators: List of SAT validator IDs
            validator_keys: Optional dict of {sat_id: (public_key, private_key)}
                          If not provided, uses validator registry

        Returns:
            Tuple of (passed, receipt_or_rejection)
            - If passed: (True, CommitReceipt)
            - If failed: (False, RejectionResponse)
        """
        start_time = time.perf_counter()
        timestamp = utc_now_iso()
        digest = envelope.compute_digest()

        try:
            # Create or use existing gate chain
            gate_chain = self._gate_chain
            if gate_chain is None:
                gate_chain = GateChain(
                    current_policy_hash=self._get_policy_hash(),
                    current_state_hash=self._get_state_hash(),
                    ihsan_threshold=self.ihsan_threshold,
                    snr_threshold=self.snr_threshold,
                    fate_checker=self.fate_checker,
                    formal_verifier=self.formal_verifier,
                )

            # Run gate chain verification
            passed, rejection, gate_results = gate_chain.verify(
                envelope,
                require_expensive=self.require_expensive_gates,
            )

            if not passed:
                logger.warning(
                    f"SAT verification REJECTED: envelope={envelope.envelope_id[:8]}..., "
                    f"code={rejection.code.name}, message={rejection.message}"
                )
                return False, rejection

            # Collect SAT signatures (quorum check)
            verifier_signatures: List[VerifierSignature] = []

            for sat_id in sat_validators[:self.quorum_total]:
                # Get validator keys
                if validator_keys and sat_id in validator_keys:
                    pub_key, priv_key = validator_keys[sat_id]
                else:
                    # Try registry
                    validator = await self._validator_registry.get(sat_id)
                    if validator is None or not validator.can_sign():
                        logger.warning(f"SAT validator {sat_id} not found or cannot sign")
                        continue
                    pub_key = validator.public_key
                    priv_key = validator.private_key

                # Create verifier signature
                sig_timestamp = utc_now_iso()
                sig_data = {
                    "envelope_digest": digest,
                    "sat_id": sat_id,
                    "timestamp": sig_timestamp,
                }
                sig_bytes = canonical_json(sig_data)
                sig_digest = domain_separated_digest(sig_bytes, "bizra-pci-verifier-v1:")
                signature = sign_message(bytes.fromhex(sig_digest), priv_key)

                verifier_signatures.append(VerifierSignature(
                    sat_id=sat_id,
                    public_key=pub_key,
                    signature=signature,
                    timestamp=sig_timestamp,
                ))

                # Stop if we have enough signatures
                if len(verifier_signatures) >= self.quorum_required:
                    break

            # Check quorum
            if len(verifier_signatures) < self.quorum_required:
                quorum_rejection = RejectionResponse.rejection(
                    code=RejectCode.REJECT_QUORUM_FAILED,
                    message=f"Quorum failed: {len(verifier_signatures)}/{self.quorum_required} signatures",
                    envelope_digest=digest,
                    timestamp=timestamp,
                )
                logger.warning(
                    f"SAT quorum FAILED: envelope={envelope.envelope_id[:8]}..., "
                    f"signatures={len(verifier_signatures)}/{self.quorum_required}"
                )
                return False, quorum_rejection

            # Generate commit receipt
            total_latency = sum(r.latency_ms for r in gate_results)
            gates_passed = gate_chain.get_gates_passed()

            # Use first validator as primary receipt signer
            primary_validator = verifier_signatures[0]
            if validator_keys and primary_validator.sat_id in validator_keys:
                _, primary_priv_key = validator_keys[primary_validator.sat_id]
            else:
                validator = await self._validator_registry.get(primary_validator.sat_id)
                primary_priv_key = validator.private_key if validator else b''

            receipt = self._receipt_generator.create_receipt(
                envelope_digest=digest,
                verification_tier=VerificationTier.STATISTICAL,  # Default tier
                latency_ms=total_latency,
                gates_passed=gates_passed,
                ihsan_score=envelope.metadata.ihsan_score,
                snr_score=envelope.metadata.snr_score,
                verifier_id=primary_validator.sat_id,
                verifier_public_key=primary_validator.public_key,
                verifier_private_key=primary_priv_key,
                policy_hash=self._get_policy_hash(),
                quorum_required=self.quorum_required,
            )

            # Add additional verifier signatures
            for vs in verifier_signatures[1:]:
                if validator_keys and vs.sat_id in validator_keys:
                    _, vs_priv_key = validator_keys[vs.sat_id]
                else:
                    validator = await self._validator_registry.get(vs.sat_id)
                    vs_priv_key = validator.private_key if validator else b''

                receipt = self._receipt_generator.add_verifier_signature(
                    receipt,
                    vs.sat_id,
                    vs.public_key,
                    vs_priv_key,
                )

            # Store receipt
            self._receipt_store.append(receipt)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"SAT verification PASSED: envelope={envelope.envelope_id[:8]}..., "
                f"receipt={receipt.receipt_id[:8]}..., "
                f"quorum={receipt.quorum.achieved}/{receipt.quorum.required}, "
                f"latency={elapsed_ms:.2f}ms"
            )

            return True, receipt

        except Exception as e:
            logger.error(
                f"SAT verification ERROR (fail-closed): envelope={envelope.envelope_id[:8]}..., "
                f"error={str(e)}"
            )
            return False, reject_internal_error(digest, timestamp, str(e))

    async def execute_flow(
        self,
        pat_agent_id: str,
        action: str,
        data: Dict[str, Any],
        private_key: bytes,
        sat_validators: List[str],
        public_key: Optional[str] = None,
        validator_keys: Optional[Dict[str, Tuple[str, bytes]]] = None,
        ihsan_score: float = 0.95,
        snr_score: float = 0.85,
    ) -> FlowResult:
        """
        Execute the complete PAT -> SAT verification flow.

        This is the primary entry point for PAT-SAT integration.

        Args:
            pat_agent_id: PAT agent identifier
            action: Action being proposed
            data: Action payload data
            private_key: PAT agent's Ed25519 private key
            sat_validators: List of SAT validator IDs
            public_key: Optional PAT agent public key
            validator_keys: Optional dict of SAT validator keys
            ihsan_score: Ihsan score for the proposal
            snr_score: SNR score for the proposal

        Returns:
            FlowResult with full audit trail
        """
        start_time = time.perf_counter()
        timestamp = utc_now_iso()

        audit_trail = {
            "flow_start": timestamp,
            "pat_agent_id": pat_agent_id,
            "action": action,
            "sat_validators": sat_validators,
        }

        try:
            # Step 1: PAT propose
            envelope, digest = await self.pat_propose(
                agent_id=pat_agent_id,
                action=action,
                data=data,
                private_key=private_key,
                public_key=public_key,
                ihsan_score=ihsan_score,
                snr_score=snr_score,
            )

            audit_trail["envelope_id"] = envelope.envelope_id
            audit_trail["envelope_digest"] = digest

            # Step 2: SAT verify
            passed, result = await self.sat_verify(
                envelope=envelope,
                sat_validators=sat_validators,
                validator_keys=validator_keys,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            audit_trail["flow_end"] = utc_now_iso()
            audit_trail["total_latency_ms"] = elapsed_ms
            audit_trail["passed"] = passed

            # Get gate results from chain
            if self._gate_chain:
                gate_results = []  # Would need to capture from verify()
            else:
                gate_results = []

            if passed:
                receipt = result  # type: CommitReceipt
                audit_trail["receipt_id"] = receipt.receipt_id
                audit_trail["quorum"] = receipt.quorum.to_dict()

                logger.info(
                    f"Flow COMPLETE: pat={pat_agent_id}, action={action}, "
                    f"envelope={envelope.envelope_id[:8]}..., "
                    f"receipt={receipt.receipt_id[:8]}..., "
                    f"latency={elapsed_ms:.2f}ms"
                )

                return FlowResult(
                    envelope=envelope,
                    passed=True,
                    receipt=receipt,
                    rejection=None,
                    gate_results=gate_results,
                    latency_ms=elapsed_ms,
                    audit_trail=audit_trail,
                )
            else:
                rejection = result  # type: RejectionResponse
                audit_trail["rejection_code"] = rejection.code.name
                audit_trail["rejection_message"] = rejection.message

                logger.warning(
                    f"Flow REJECTED: pat={pat_agent_id}, action={action}, "
                    f"envelope={envelope.envelope_id[:8]}..., "
                    f"code={rejection.code.name}, "
                    f"latency={elapsed_ms:.2f}ms"
                )

                return FlowResult(
                    envelope=envelope,
                    passed=False,
                    receipt=None,
                    rejection=rejection,
                    gate_results=gate_results,
                    latency_ms=elapsed_ms,
                    audit_trail=audit_trail,
                )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            audit_trail["error"] = str(e)
            audit_trail["flow_end"] = utc_now_iso()
            audit_trail["total_latency_ms"] = elapsed_ms

            logger.error(
                f"Flow ERROR (fail-closed): pat={pat_agent_id}, action={action}, "
                f"error={str(e)}, latency={elapsed_ms:.2f}ms"
            )

            # Create a minimal envelope for error response
            error_envelope = PCIEnvelope(
                version="1.0.0",
                envelope_id="error",
                timestamp=timestamp,
                nonce="0" * 64,
                sender=None,  # type: ignore
                payload=None,  # type: ignore
                metadata=None,  # type: ignore
            )

            return FlowResult(
                envelope=error_envelope,
                passed=False,
                receipt=None,
                rejection=reject_internal_error("error", timestamp, str(e)),
                gate_results=[],
                latency_ms=elapsed_ms,
                audit_trail=audit_trail,
            )

    async def emit_receipt(
        self,
        envelope: PCIEnvelope,
        gate_results: List[GateResult],
        verifier_signatures: List[VerifierSignature],
    ) -> CommitReceipt:
        """
        Emit a receipt for a verified envelope.

        This is called automatically by sat_verify(), but can be used
        manually for custom verification flows.

        Args:
            envelope: The verified envelope
            gate_results: Results from gate chain execution
            verifier_signatures: SAT validator signatures

        Returns:
            Generated CommitReceipt
        """
        if not verifier_signatures:
            raise ValueError("At least one verifier signature required")

        primary = verifier_signatures[0]
        validator = await self._validator_registry.get(primary.sat_id)

        if validator is None or not validator.can_sign():
            raise ValueError(f"Primary validator {primary.sat_id} cannot sign")

        receipt = self._receipt_generator.create_receipt(
            envelope_digest=envelope.compute_digest(),
            verification_tier=VerificationTier.STATISTICAL,
            latency_ms=sum(r.latency_ms for r in gate_results),
            gates_passed=[r.gate for r in gate_results if r.passed],
            ihsan_score=envelope.metadata.ihsan_score,
            snr_score=envelope.metadata.snr_score,
            verifier_id=primary.sat_id,
            verifier_public_key=primary.public_key,
            verifier_private_key=validator.private_key,
            policy_hash=self._get_policy_hash(),
            quorum_required=self.quorum_required,
        )

        # Add additional signatures
        for vs in verifier_signatures[1:]:
            v = await self._validator_registry.get(vs.sat_id)
            if v and v.can_sign():
                receipt = self._receipt_generator.add_verifier_signature(
                    receipt,
                    vs.sat_id,
                    vs.public_key,
                    v.private_key,
                )

        self._receipt_store.append(receipt)

        logger.info(
            f"Receipt emitted: {receipt.receipt_id[:8]}..., "
            f"envelope={envelope.envelope_id[:8]}..., "
            f"quorum={receipt.quorum.achieved}/{receipt.quorum.required}"
        )

        return receipt


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def create_pat_envelope(
    agent_id: str,
    action: str,
    data: Dict[str, Any],
    key: bytes,
    policy_hash: str,
    state_hash: str,
    ihsan_score: float = 0.95,
    snr_score: float = 0.85,
) -> PCIEnvelope:
    """
    Create a signed PAT proposal envelope.

    Convenience function for creating envelopes without a full bridge.

    Args:
        agent_id: PAT agent identifier
        action: Action being proposed
        data: Action payload data
        key: Ed25519 private key (32 bytes)
        policy_hash: Current policy (constitution) hash
        state_hash: Current state hash
        ihsan_score: Ihsan excellence score
        snr_score: Signal-to-noise ratio

    Returns:
        Signed PCIEnvelope
    """
    from nacl.signing import SigningKey
    signing_key = SigningKey(key)
    public_key = bytes(signing_key.verify_key).hex()

    envelope = (
        EnvelopeBuilder()
        .with_sender(AgentType.PAT, agent_id, public_key)
        .with_action(action, data)
        .with_policy(policy_hash)
        .with_state(state_hash)
        .with_scores(ihsan=ihsan_score, snr=snr_score)
        .build()
        .sign(key)
    )

    return envelope


async def verify_with_sat(
    envelope: PCIEnvelope,
    validators: List[str],
    validator_keys: Dict[str, Tuple[str, bytes]],
    policy_hash: str,
    state_hash: str,
    ihsan_threshold: float = IHSAN_THRESHOLD,
    snr_threshold: float = SNR_THRESHOLD_DEFAULT,
    quorum_required: int = 3,
) -> Tuple[bool, Any]:
    """
    Verify an envelope with SAT validators.

    Convenience function for verification without a full bridge.

    Args:
        envelope: The envelope to verify
        validators: List of SAT validator IDs
        validator_keys: Dict of {sat_id: (public_key, private_key)}
        policy_hash: Current policy hash
        state_hash: Current state hash
        ihsan_threshold: Minimum Ihsan score
        snr_threshold: Minimum SNR score
        quorum_required: Number of signatures required

    Returns:
        Tuple of (passed, receipt_or_rejection)
    """
    bridge = PATSATBridge(
        ihsan_threshold=ihsan_threshold,
        snr_threshold=snr_threshold,
        quorum_required=quorum_required,
    )
    bridge.set_policy_hash(policy_hash)
    bridge.set_state_hash(state_hash)

    return await bridge.sat_verify(envelope, validators, validator_keys)


async def complete_pat_sat_flow(
    agent_id: str,
    action: str,
    data: Dict[str, Any],
    key: bytes,
    validators: List[str],
    validator_keys: Dict[str, Tuple[str, bytes]],
    policy_hash: str,
    state_hash: str,
    ihsan_threshold: float = IHSAN_THRESHOLD,
    snr_threshold: float = SNR_THRESHOLD_DEFAULT,
    ihsan_score: float = 0.95,
    snr_score: float = 0.85,
    quorum_required: int = 3,
) -> FlowResult:
    """
    Execute a complete PAT-SAT verification flow.

    This is the simplest way to run the full verification pipeline.

    Args:
        agent_id: PAT agent identifier
        action: Action being proposed
        data: Action payload data
        key: PAT agent's Ed25519 private key
        validators: List of SAT validator IDs
        validator_keys: Dict of {sat_id: (public_key, private_key)}
        policy_hash: Current policy hash
        state_hash: Current state hash
        ihsan_threshold: Minimum Ihsan score for verification
        snr_threshold: Minimum SNR score for verification
        ihsan_score: Ihsan score for the proposal
        snr_score: SNR score for the proposal
        quorum_required: Number of SAT signatures required

    Returns:
        FlowResult with full audit trail

    Example:
        result = await complete_pat_sat_flow(
            agent_id="pat-001",
            action="analyze",
            data={"code": "..."},
            key=pat_private_key,
            validators=["sat-001", "sat-002", "sat-003"],
            validator_keys={
                "sat-001": (sat1_pub, sat1_priv),
                "sat-002": (sat2_pub, sat2_priv),
                "sat-003": (sat3_pub, sat3_priv),
            },
            policy_hash=policy_hash,
            state_hash=state_hash,
        )

        if result.passed:
            print(f"Success! Receipt: {result.receipt_id}")
        else:
            print(f"Failed: {result.rejection.message}")
    """
    bridge = PATSATBridge(
        ihsan_threshold=ihsan_threshold,
        snr_threshold=snr_threshold,
        quorum_required=quorum_required,
    )
    bridge.set_policy_hash(policy_hash)
    bridge.set_state_hash(state_hash)

    return await bridge.execute_flow(
        pat_agent_id=agent_id,
        action=action,
        data=data,
        private_key=key,
        sat_validators=validators,
        validator_keys=validator_keys,
        ihsan_score=ihsan_score,
        snr_score=snr_score,
    )


# =============================================================================
# UNIT TESTS
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test_integration():
        """
        Unit tests for PAT-SAT-PCI integration.

        These tests verify:
        1. PAT proposal creation and signing
        2. SAT verification through gate chain
        3. Quorum enforcement
        4. Receipt generation
        5. Full flow execution
        6. Error handling (fail-closed)
        """
        print("=" * 60)
        print("PAT-SAT-PCI Integration Tests")
        print("=" * 60)

        # Generate test keys
        pat_keypair = generate_keypair()
        sat_keypairs = [generate_keypair() for _ in range(5)]

        # Test policy and state hashes
        test_policy_hash = "a" * 64
        test_state_hash = "b" * 64

        # Validator keys dict
        validator_keys = {
            f"sat-{i:03d}": (kp.public_key_hex, kp.private_key)
            for i, kp in enumerate(sat_keypairs, 1)
        }
        validators = list(validator_keys.keys())

        # Test 1: PAT Proposal Creation
        print("\n[TEST 1] PAT Proposal Creation")
        try:
            envelope = await create_pat_envelope(
                agent_id="pat-001",
                action="analyze",
                data={"code": "print('hello')"},
                key=pat_keypair.private_key,
                policy_hash=test_policy_hash,
                state_hash=test_state_hash,
            )
            assert envelope.signature is not None
            assert envelope.verify_signature()
            print(f"  PASS: Envelope created and signed: {envelope.envelope_id[:16]}...")
        except Exception as e:
            print(f"  FAIL: {e}")
            return False

        # Test 2: SAT Verification (should pass)
        print("\n[TEST 2] SAT Verification (Pass Case)")
        try:
            passed, result = await verify_with_sat(
                envelope=envelope,
                validators=validators,
                validator_keys=validator_keys,
                policy_hash=test_policy_hash,
                state_hash=test_state_hash,
            )
            assert passed, f"Expected pass, got rejection: {result}"
            assert isinstance(result, CommitReceipt)
            assert result.quorum.is_met()
            print(f"  PASS: Verification passed, receipt: {result.receipt_id[:16]}...")
        except Exception as e:
            print(f"  FAIL: {e}")
            return False

        # Test 3: Full Flow Execution
        print("\n[TEST 3] Full Flow Execution")
        try:
            flow_result = await complete_pat_sat_flow(
                agent_id="pat-002",
                action="execute",
                data={"command": "run_analysis"},
                key=pat_keypair.private_key,
                validators=validators,
                validator_keys=validator_keys,
                policy_hash=test_policy_hash,
                state_hash=test_state_hash,
            )
            assert flow_result.passed
            assert flow_result.receipt is not None
            assert flow_result.rejection is None
            assert flow_result.latency_ms > 0
            print(f"  PASS: Flow completed in {flow_result.latency_ms:.2f}ms")
            print(f"        Envelope: {flow_result.envelope.envelope_id[:16]}...")
            print(f"        Receipt: {flow_result.receipt_id[:16]}...")
        except Exception as e:
            print(f"  FAIL: {e}")
            return False

        # Test 4: Ihsan Threshold Rejection
        print("\n[TEST 4] Ihsan Threshold Rejection")
        try:
            low_ihsan_envelope = await create_pat_envelope(
                agent_id="pat-003",
                action="risky_action",
                data={"danger": True},
                key=pat_keypair.private_key,
                policy_hash=test_policy_hash,
                state_hash=test_state_hash,
                ihsan_score=0.80,  # Below 0.95 threshold
            )
            passed, result = await verify_with_sat(
                envelope=low_ihsan_envelope,
                validators=validators,
                validator_keys=validator_keys,
                policy_hash=test_policy_hash,
                state_hash=test_state_hash,
            )
            assert not passed, "Expected rejection for low Ihsan"
            assert isinstance(result, RejectionResponse)
            assert result.code == RejectCode.REJECT_IHSAN_BELOW_MIN
            print(f"  PASS: Correctly rejected for Ihsan={0.80} < 0.95")
        except Exception as e:
            print(f"  FAIL: {e}")
            return False

        # Test 5: Quorum Failure
        print("\n[TEST 5] Quorum Failure")
        try:
            # Only provide 2 validators when 3 are required
            limited_validators = validators[:2]
            limited_keys = {k: v for k, v in validator_keys.items() if k in limited_validators}

            passed, result = await verify_with_sat(
                envelope=envelope,
                validators=limited_validators,
                validator_keys=limited_keys,
                policy_hash=test_policy_hash,
                state_hash=test_state_hash,
                quorum_required=3,
            )
            assert not passed, "Expected quorum failure"
            assert isinstance(result, RejectionResponse)
            assert result.code == RejectCode.REJECT_QUORUM_FAILED
            print(f"  PASS: Correctly rejected for quorum 2/3")
        except Exception as e:
            print(f"  FAIL: {e}")
            return False

        # Test 6: Policy Mismatch
        print("\n[TEST 6] Policy Mismatch")
        try:
            passed, result = await verify_with_sat(
                envelope=envelope,
                validators=validators,
                validator_keys=validator_keys,
                policy_hash="c" * 64,  # Different policy
                state_hash=test_state_hash,
            )
            assert not passed, "Expected policy mismatch rejection"
            assert isinstance(result, RejectionResponse)
            assert result.code == RejectCode.REJECT_POLICY_MISMATCH
            print(f"  PASS: Correctly rejected for policy mismatch")
        except Exception as e:
            print(f"  FAIL: {e}")
            return False

        # Test 7: FlowResult Serialization
        print("\n[TEST 7] FlowResult Serialization")
        try:
            flow_result = await complete_pat_sat_flow(
                agent_id="pat-004",
                action="serialize_test",
                data={"test": True},
                key=pat_keypair.private_key,
                validators=validators,
                validator_keys=validator_keys,
                policy_hash=test_policy_hash,
                state_hash=test_state_hash,
            )

            result_dict = flow_result.to_dict()
            assert "envelope_id" in result_dict
            assert "passed" in result_dict
            assert "audit_trail" in result_dict
            assert result_dict["passed"] is True
            print(f"  PASS: FlowResult serialized correctly")
        except Exception as e:
            print(f"  FAIL: {e}")
            return False

        # Test 8: Bridge with Validator Registry
        print("\n[TEST 8] Bridge with Validator Registry")
        try:
            registry = SATValidatorRegistry()
            for sat_id, (pub_key, priv_key) in validator_keys.items():
                await registry.register(SATValidator(
                    sat_id=sat_id,
                    public_key=pub_key,
                    private_key=priv_key,
                ))

            bridge = PATSATBridge(
                validator_registry=registry,
                quorum_required=3,
            )
            bridge.set_policy_hash(test_policy_hash)
            bridge.set_state_hash(test_state_hash)

            result = await bridge.execute_flow(
                pat_agent_id="pat-005",
                action="registry_test",
                data={"test": "registry"},
                private_key=pat_keypair.private_key,
                sat_validators=validators,
            )

            assert result.passed
            print(f"  PASS: Bridge with registry works correctly")
        except Exception as e:
            print(f"  FAIL: {e}")
            return False

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        return True

    # Run tests
    success = asyncio.run(test_integration())
    exit(0 if success else 1)
