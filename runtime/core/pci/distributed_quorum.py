"""
BIZRA PCI Protocol - Distributed Quorum System
===============================================
Collects SAT validator signatures for Byzantine fault-tolerant consensus.
Constitution requires 3/5 SAT validators for quorum.

Status: ACTIVE
BFT: f <= 1 for n=5 validators (tolerates 1 Byzantine failure)
Alignment: BIZRA_SOT.md Section 3.2 (SAT Consensus)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from .crypto import (
    canonical_json,
    domain_separated_digest,
    verify_signature,
    generate_keypair,
)
from .types import (
    VerifierSignature,
    utc_now_iso,
)

# Configure logging
logger = logging.getLogger("bizra.pci.quorum")


# =============================================================================
# CONSTANTS
# =============================================================================

# Quorum requirements from constitution
REQUIRED_VALIDATORS = 3  # 3/5 required for consensus
TOTAL_VALIDATORS = 5
DEFAULT_TIMEOUT_MS = 5000
SIGNATURE_DOMAIN = "bizra-pci-quorum-v1:"

# Required SAT roles that MUST sign for critical operations
REQUIRED_ROLES = frozenset({"RiskGuardian", "GovernanceEngine"})

# All SAT roles
SAT_ROLES = frozenset(
    {
        "PoiVerifier",
        "RiskGuardian",
        "GovernanceEngine",
        "ResourceAllocator",
        "EvidenceEngine",
    }
)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SATValidator:
    """
    SAT validator identity and keys.

    Each validator has a unique role in the governance structure:
    - PoiVerifier: Verifies Proof of Impact attestations
    - RiskGuardian: Assesses and mitigates operational risks
    - GovernanceEngine: Enforces policy and constitution compliance
    - ResourceAllocator: Manages URP and resource allocation
    - EvidenceEngine: Validates audit trails and receipts
    """

    sat_id: str
    public_key: bytes  # 32-byte Ed25519 public key
    role: str
    weight: float = 1.0
    available: bool = True
    endpoint: Optional[str] = None  # Optional network endpoint for remote validators

    def __post_init__(self):
        if self.role not in SAT_ROLES:
            raise ValueError(
                f"Invalid SAT role: {self.role}. Must be one of {SAT_ROLES}"
            )
        if len(self.public_key) != 32:
            raise ValueError(f"Public key must be 32 bytes, got {len(self.public_key)}")

    @property
    def public_key_hex(self) -> str:
        return self.public_key.hex()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sat_id": self.sat_id,
            "public_key": self.public_key_hex,
            "role": self.role,
            "weight": self.weight,
            "available": self.available,
            "endpoint": self.endpoint,
        }


@dataclass
class SignatureRequest:
    """
    Request for SAT validator signature.

    Contains all information needed for a validator to make a signing decision.
    """

    request_id: str
    envelope_digest: str  # BLAKE3 digest of the envelope being validated
    gate_results: List[Dict[str, Any]]  # Results from SAPE/IHSAN gates
    timestamp: str
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context

    @classmethod
    def create(
        cls,
        envelope_digest: str,
        gate_results: List[Dict[str, Any]],
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        context: Optional[Dict[str, Any]] = None,
    ) -> "SignatureRequest":
        """Factory method to create a new signature request."""
        return cls(
            request_id=f"sigr-{uuid4().hex[:12]}",
            envelope_digest=envelope_digest,
            gate_results=gate_results,
            timestamp=utc_now_iso(),
            timeout_ms=timeout_ms,
            context=context or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "envelope_digest": self.envelope_digest,
            "gate_results": self.gate_results,
            "timestamp": self.timestamp,
            "timeout_ms": self.timeout_ms,
            "context": self.context,
        }

    def to_signable_bytes(self) -> bytes:
        """Convert to canonical bytes for signing."""
        signable = {
            "request_id": self.request_id,
            "envelope_digest": self.envelope_digest,
            "gate_results": self.gate_results,
            "timestamp": self.timestamp,
        }
        return canonical_json(signable)


@dataclass
class SignatureResponse:
    """
    Response from SAT validator.

    Contains the validator's decision and cryptographic signature.
    """

    sat_id: str
    signature: bytes  # 64-byte Ed25519 signature
    timestamp: str
    approved: bool
    rejection_reason: Optional[str] = None
    latency_ms: float = 0.0

    @property
    def signature_hex(self) -> str:
        return self.signature.hex()

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "sat_id": self.sat_id,
            "signature": self.signature_hex,
            "timestamp": self.timestamp,
            "approved": self.approved,
            "latency_ms": self.latency_ms,
        }
        if self.rejection_reason:
            result["rejection_reason"] = self.rejection_reason
        return result


# =============================================================================
# QUORUM STATE
# =============================================================================


class QuorumState(str, Enum):
    """State of the quorum collection process."""

    COLLECTING = "collecting"  # Still collecting signatures
    ACHIEVED = "achieved"  # Quorum reached with required roles
    FAILED = "failed"  # Cannot achieve quorum (too many rejections)
    TIMEOUT = "timeout"  # Timed out before quorum
    INVALID = "invalid"  # Invalid signatures or missing required roles


@dataclass
class QuorumStatus:
    """
    Detailed status of quorum collection.

    Provides rich information for debugging and audit trails.
    """

    state: QuorumState
    required: int
    achieved: int
    approved_validators: List[str]
    rejected_validators: List[str]
    pending_validators: List[str]
    missing_required_roles: Set[str]
    elapsed_ms: float
    reason: str = ""

    def is_met(self) -> bool:
        return self.state == QuorumState.ACHIEVED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "required": self.required,
            "achieved": self.achieved,
            "approved_validators": self.approved_validators,
            "rejected_validators": self.rejected_validators,
            "pending_validators": self.pending_validators,
            "missing_required_roles": list(self.missing_required_roles),
            "elapsed_ms": self.elapsed_ms,
            "reason": self.reason,
        }


# =============================================================================
# DISTRIBUTED QUORUM ENGINE
# =============================================================================


class DistributedQuorum:
    """
    Manages distributed signature collection from SAT validators.

    Implements Byzantine fault tolerance (f <= 1 for n=5).

    Byzantine Fault Tolerance:
    - n = 5 total validators
    - f = 1 maximum Byzantine (malicious/faulty) validators
    - Quorum requirement: 3/5 (n - f - f = 3)

    Required Roles:
    - RiskGuardian: MUST sign for any operation
    - GovernanceEngine: MUST sign for any operation

    Flow:
    1. Broadcast signature request to all validators
    2. Collect responses with timeout
    3. Verify Ed25519 signatures
    4. Check quorum requirements (3/5 + required roles)
    5. Return aggregated VerifierSignatures for receipt
    """

    REQUIRED_VALIDATORS = REQUIRED_VALIDATORS
    TOTAL_VALIDATORS = TOTAL_VALIDATORS
    DEFAULT_TIMEOUT_MS = DEFAULT_TIMEOUT_MS
    REQUIRED_ROLES = REQUIRED_ROLES

    def __init__(
        self,
        validators: List[SATValidator],
        synapse_client: Optional[Any] = None,  # SynapseBus for Redis communication
        strict_mode: bool = True,  # Require all REQUIRED_ROLES
    ):
        """
        Initialize the distributed quorum system.

        Args:
            validators: List of SAT validators (should be 5 for production)
            synapse_client: Optional SynapseBus for distributed communication
            strict_mode: If True, require REQUIRED_ROLES to sign
        """
        self.validators = {v.sat_id: v for v in validators}
        self.synapse = synapse_client
        self.strict_mode = strict_mode

        # Tracking state
        self._pending_requests: Dict[str, SignatureRequest] = {}
        self._collected_signatures: Dict[str, List[SignatureResponse]] = {}
        self._request_start_times: Dict[str, float] = {}

        # Callback handlers for custom validation logic
        self._pre_sign_hooks: List[Callable[[SignatureRequest], bool]] = []
        self._post_collect_hooks: List[
            Callable[[str, List[SignatureResponse]], None]
        ] = []

        # Validate configuration
        if len(self.validators) != self.TOTAL_VALIDATORS:
            logger.warning(
                f"Expected {self.TOTAL_VALIDATORS} validators, got {len(self.validators)}. "
                "This may affect Byzantine fault tolerance."
            )

        # Check required roles are present
        available_roles = {v.role for v in validators if v.available}
        missing = self.REQUIRED_ROLES - available_roles
        if missing and strict_mode:
            logger.error(f"Missing required SAT roles: {missing}")

    def get_available_validators(self) -> List[SATValidator]:
        """Get list of currently available validators."""
        return [v for v in self.validators.values() if v.available]

    def get_validators_by_role(self, role: str) -> List[SATValidator]:
        """Get validators with a specific role."""
        return [v for v in self.validators.values() if v.role == role]

    def set_validator_availability(self, sat_id: str, available: bool) -> bool:
        """Update validator availability status."""
        if sat_id not in self.validators:
            return False
        self.validators[sat_id].available = available
        return True

    # =========================================================================
    # SIGNATURE REQUEST
    # =========================================================================

    async def request_signatures(
        self,
        envelope_digest: str,
        gate_results: List[Dict[str, Any]],
        timeout_ms: int = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Broadcast signature request to all SAT validators.

        Args:
            envelope_digest: BLAKE3 digest of the envelope being validated
            gate_results: Results from SAPE/IHSAN gates
            timeout_ms: Optional custom timeout
            context: Additional context for validators

        Returns:
            request_id for tracking
        """
        timeout = timeout_ms or self.DEFAULT_TIMEOUT_MS

        request = SignatureRequest.create(
            envelope_digest=envelope_digest,
            gate_results=gate_results,
            timeout_ms=timeout,
            context=context,
        )

        # Run pre-sign hooks
        for hook in self._pre_sign_hooks:
            if not hook(request):
                logger.warning(f"Pre-sign hook rejected request {request.request_id}")
                return request.request_id

        # Store request state
        self._pending_requests[request.request_id] = request
        self._collected_signatures[request.request_id] = []
        self._request_start_times[request.request_id] = time.monotonic()

        logger.info(
            f"Broadcasting signature request {request.request_id} "
            f"for envelope {envelope_digest[:16]}..."
        )

        # Broadcast via Synapse if available
        if self.synapse:
            await self._broadcast_via_synapse(request)

        return request.request_id

    async def _broadcast_via_synapse(self, request: SignatureRequest) -> None:
        """Broadcast signature request via Redis Synapse."""
        try:
            # Import here to avoid circular dependency
            from ..synapse import MessageType

            # Publish to SAT team channel
            self.synapse.publish_to_team(
                "sat",
                MessageType.CONSENSUS_REQUEST,
                {
                    "request_id": request.request_id,
                    "envelope_digest": request.envelope_digest,
                    "gate_results": request.gate_results,
                    "timestamp": request.timestamp,
                    "timeout_ms": request.timeout_ms,
                    "context": request.context,
                },
            )
        except Exception as e:
            logger.error(f"Failed to broadcast via Synapse: {e}")

    # =========================================================================
    # SIGNATURE COLLECTION
    # =========================================================================

    async def collect_signatures(
        self,
        request_id: str,
        timeout_ms: int = None,
    ) -> Tuple[QuorumState, List[SignatureResponse]]:
        """
        Collect signatures until quorum achieved or timeout.

        This method implements an async collection loop that:
        1. Waits for validator responses
        2. Validates signatures as they arrive
        3. Checks quorum status after each signature
        4. Returns early if quorum achieved or impossible

        Args:
            request_id: ID of the signature request
            timeout_ms: Optional custom timeout

        Returns:
            Tuple of (QuorumState, collected signatures)
        """
        if request_id not in self._pending_requests:
            logger.error(f"Unknown request ID: {request_id}")
            return QuorumState.INVALID, []

        request = self._pending_requests[request_id]
        timeout = timeout_ms or request.timeout_ms
        timeout_seconds = timeout / 1000.0

        start_time = time.monotonic()
        signatures = self._collected_signatures[request_id]

        # If using Synapse, wait for responses via pub/sub
        if self.synapse:
            return await self._collect_via_synapse(request_id, timeout_seconds)

        # Otherwise, simulate local validator signing
        return await self._collect_local(request_id, timeout_seconds)

    async def _collect_local(
        self,
        request_id: str,
        timeout_seconds: float,
    ) -> Tuple[QuorumState, List[SignatureResponse]]:
        """
        Collect signatures from local validators (simulation mode).

        In production, validators would be remote services.
        This mode is useful for testing and single-node deployments.
        """
        request = self._pending_requests[request_id]
        signatures: List[SignatureResponse] = []

        start_time = time.monotonic()

        for validator in self.get_available_validators():
            # Check timeout
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout_seconds:
                logger.warning(f"Timeout reached after {elapsed:.2f}s")
                break

            # Simulate validator signing
            try:
                response = await self._simulate_validator_sign(
                    validator,
                    request,
                )
                if response:
                    signatures.append(response)

                    # Check if quorum achievable
                    achieved, reason = self.check_quorum(signatures)
                    if achieved:
                        logger.info(f"Quorum achieved: {reason}")
                        self._collected_signatures[request_id] = signatures
                        return QuorumState.ACHIEVED, signatures

            except Exception as e:
                logger.error(f"Error from validator {validator.sat_id}: {e}")

        # Check final state
        self._collected_signatures[request_id] = signatures
        achieved, reason = self.check_quorum(signatures)

        if achieved:
            return QuorumState.ACHIEVED, signatures

        # Determine failure reason
        elapsed = time.monotonic() - start_time
        if elapsed >= timeout_seconds:
            return QuorumState.TIMEOUT, signatures

        return QuorumState.FAILED, signatures

    async def _collect_via_synapse(
        self,
        request_id: str,
        timeout_seconds: float,
    ) -> Tuple[QuorumState, List[SignatureResponse]]:
        """Collect signatures via Redis Synapse pub/sub."""
        signatures = self._collected_signatures[request_id]
        start_time = time.monotonic()

        # Poll for responses with timeout
        poll_interval = 0.1  # 100ms

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout_seconds:
                break

            # Check if quorum achieved
            achieved, reason = self.check_quorum(signatures)
            if achieved:
                return QuorumState.ACHIEVED, signatures

            # Check if quorum impossible
            remaining_validators = self.TOTAL_VALIDATORS - len(signatures)
            if (
                len([s for s in signatures if s.approved]) + remaining_validators
                < self.REQUIRED_VALIDATORS
            ):
                return QuorumState.FAILED, signatures

            await asyncio.sleep(poll_interval)

        # Final check
        achieved, reason = self.check_quorum(signatures)
        if achieved:
            return QuorumState.ACHIEVED, signatures

        return QuorumState.TIMEOUT, signatures

    async def _simulate_validator_sign(
        self,
        validator: SATValidator,
        request: SignatureRequest,
    ) -> Optional[SignatureResponse]:
        """
        Simulate a validator signing a request.

        In production, this would be an RPC call to the validator service.
        """
        # Simulate network latency
        await asyncio.sleep(0.01)  # 10ms

        start_time = time.monotonic()

        # Check gate results for approval decision
        approved = self._should_validator_approve(validator, request)
        rejection_reason = None if approved else "Gate requirements not met"

        # Sign the request
        signable = request.to_signable_bytes()
        digest = domain_separated_digest(signable, SIGNATURE_DOMAIN)

        # Note: In production, each validator has its own private key
        # For simulation, we create a deterministic signature
        signature_bytes = bytes(64)  # Placeholder for simulation

        latency_ms = (time.monotonic() - start_time) * 1000

        return SignatureResponse(
            sat_id=validator.sat_id,
            signature=signature_bytes,
            timestamp=utc_now_iso(),
            approved=approved,
            rejection_reason=rejection_reason,
            latency_ms=latency_ms,
        )

    def _should_validator_approve(
        self,
        validator: SATValidator,
        request: SignatureRequest,
    ) -> bool:
        """
        Determine if a validator should approve the request.

        Each validator role has specific criteria:
        - PoiVerifier: Check impact attestations
        - RiskGuardian: Check risk assessments
        - GovernanceEngine: Check policy compliance
        - ResourceAllocator: Check resource availability
        - EvidenceEngine: Check audit trail completeness
        """
        # Check gate results
        for result in request.gate_results:
            gate_name = result.get("gate", "").upper()
            passed = result.get("passed", False)

            # Role-specific gate requirements
            if validator.role == "RiskGuardian" and gate_name == "FATE":
                if not passed:
                    return False

            if validator.role == "GovernanceEngine" and gate_name == "POLICY":
                if not passed:
                    return False

            if validator.role == "PoiVerifier" and gate_name == "IHSAN":
                if not passed:
                    return False

        # Default: approve if all gates passed
        return all(r.get("passed", False) for r in request.gate_results)

    # =========================================================================
    # SIGNATURE SUBMISSION (External validators)
    # =========================================================================

    def submit_signature(
        self,
        request_id: str,
        response: SignatureResponse,
    ) -> bool:
        """
        Submit a signature response from an external validator.

        Used when validators submit signatures asynchronously.
        """
        if request_id not in self._pending_requests:
            logger.error(f"Unknown request ID: {request_id}")
            return False

        # Verify the validator exists
        if response.sat_id not in self.validators:
            logger.error(f"Unknown validator: {response.sat_id}")
            return False

        # Verify signature
        request = self._pending_requests[request_id]
        if not self.verify_signature(response, request.envelope_digest):
            logger.error(f"Invalid signature from {response.sat_id}")
            return False

        # Add to collected signatures
        self._collected_signatures[request_id].append(response)
        logger.info(f"Received signature from {response.sat_id}")

        return True

    # =========================================================================
    # QUORUM VERIFICATION
    # =========================================================================

    def check_quorum(
        self,
        signatures: List[SignatureResponse],
    ) -> Tuple[bool, str]:
        """
        Check if collected signatures meet quorum requirements.

        Requirements:
        1. At least 3/5 validators signed with approval
        2. RiskGuardian and GovernanceEngine MUST be included (if strict_mode)
        3. All signatures must be valid Ed25519

        Args:
            signatures: List of collected signature responses

        Returns:
            Tuple of (achieved, reason)
        """
        approved_sigs = [s for s in signatures if s.approved]
        approved_count = len(approved_sigs)

        # Check count requirement
        if approved_count < self.REQUIRED_VALIDATORS:
            return (
                False,
                f"Insufficient signatures: {approved_count}/{self.REQUIRED_VALIDATORS}",
            )

        # Check required roles
        if self.strict_mode:
            approved_roles = set()
            for sig in approved_sigs:
                if sig.sat_id in self.validators:
                    approved_roles.add(self.validators[sig.sat_id].role)

            missing_roles = self.REQUIRED_ROLES - approved_roles
            if missing_roles:
                return False, f"Missing required roles: {missing_roles}"

        return True, f"Quorum achieved: {approved_count}/{self.TOTAL_VALIDATORS}"

    def get_quorum_status(self, request_id: str) -> QuorumStatus:
        """Get detailed quorum status for a request."""
        if request_id not in self._pending_requests:
            return QuorumStatus(
                state=QuorumState.INVALID,
                required=self.REQUIRED_VALIDATORS,
                achieved=0,
                approved_validators=[],
                rejected_validators=[],
                pending_validators=list(self.validators.keys()),
                missing_required_roles=self.REQUIRED_ROLES.copy(),
                elapsed_ms=0,
                reason="Unknown request ID",
            )

        signatures = self._collected_signatures.get(request_id, [])
        start_time = self._request_start_times.get(request_id, time.monotonic())
        elapsed_ms = (time.monotonic() - start_time) * 1000

        approved = [s.sat_id for s in signatures if s.approved]
        rejected = [s.sat_id for s in signatures if not s.approved]
        responded = set(approved + rejected)
        pending = [v for v in self.validators.keys() if v not in responded]

        # Check required roles
        approved_roles = {
            self.validators[sat_id].role
            for sat_id in approved
            if sat_id in self.validators
        }
        missing_roles = self.REQUIRED_ROLES - approved_roles

        # Determine state
        achieved, reason = self.check_quorum(signatures)
        if achieved:
            state = QuorumState.ACHIEVED
        elif len(rejected) > self.TOTAL_VALIDATORS - self.REQUIRED_VALIDATORS:
            state = QuorumState.FAILED
            reason = f"Too many rejections: {len(rejected)}"
        elif elapsed_ms >= self._pending_requests[request_id].timeout_ms:
            state = QuorumState.TIMEOUT
            reason = f"Timeout after {elapsed_ms:.0f}ms"
        else:
            state = QuorumState.COLLECTING
            reason = f"Collecting: {len(approved)}/{self.REQUIRED_VALIDATORS}"

        return QuorumStatus(
            state=state,
            required=self.REQUIRED_VALIDATORS,
            achieved=len(approved),
            approved_validators=approved,
            rejected_validators=rejected,
            pending_validators=pending,
            missing_required_roles=missing_roles,
            elapsed_ms=elapsed_ms,
            reason=reason,
        )

    # =========================================================================
    # SIGNATURE VERIFICATION
    # =========================================================================

    def verify_signature(
        self,
        response: SignatureResponse,
        envelope_digest: str,
    ) -> bool:
        """
        Verify Ed25519 signature from SAT validator.

        Args:
            response: Signature response from validator
            envelope_digest: Expected envelope digest

        Returns:
            True if signature is valid, False otherwise
        """
        if response.sat_id not in self.validators:
            logger.error(f"Unknown validator: {response.sat_id}")
            return False

        validator = self.validators[response.sat_id]

        # Build message that was signed
        message_data = {
            "envelope_digest": envelope_digest,
            "sat_id": response.sat_id,
            "timestamp": response.timestamp,
            "approved": response.approved,
        }
        message_bytes = canonical_json(message_data)
        digest = domain_separated_digest(message_bytes, SIGNATURE_DOMAIN)

        # Verify Ed25519 signature
        try:
            return verify_signature(
                bytes.fromhex(digest),
                response.signature_hex,
                validator.public_key_hex,
            )
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

    # =========================================================================
    # COMPLETE QUORUM FLOW
    # =========================================================================

    async def achieve_quorum(
        self,
        envelope_digest: str,
        gate_results: List[Dict[str, Any]],
        timeout_ms: int = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[VerifierSignature]]:
        """
        Complete quorum achievement flow.

        This is the main entry point for quorum collection:
        1. Broadcast signature request to all validators
        2. Collect responses with timeout
        3. Verify quorum requirements
        4. Return VerifierSignatures for receipt

        Args:
            envelope_digest: BLAKE3 digest of the envelope
            gate_results: Results from SAPE/IHSAN gates
            timeout_ms: Optional custom timeout
            context: Additional context for validators

        Returns:
            Tuple of (achieved, verifier_signatures for receipt)
        """
        # Request signatures
        request_id = await self.request_signatures(
            envelope_digest=envelope_digest,
            gate_results=gate_results,
            timeout_ms=timeout_ms,
            context=context,
        )

        # Collect responses
        state, responses = await self.collect_signatures(
            request_id=request_id,
            timeout_ms=timeout_ms,
        )

        # Run post-collect hooks
        for hook in self._post_collect_hooks:
            hook(request_id, responses)

        # Check result
        if state != QuorumState.ACHIEVED:
            status = self.get_quorum_status(request_id)
            logger.warning(
                f"Quorum failed: {status.reason}. "
                f"Approved: {status.approved_validators}, "
                f"Rejected: {status.rejected_validators}"
            )
            return False, []

        # Convert to VerifierSignature format for receipt
        verifier_sigs = []
        for response in responses:
            if response.approved and response.sat_id in self.validators:
                validator = self.validators[response.sat_id]
                verifier_sigs.append(
                    VerifierSignature(
                        sat_id=response.sat_id,
                        public_key=validator.public_key_hex,
                        signature=response.signature_hex,
                        timestamp=response.timestamp,
                    )
                )

        logger.info(
            f"Quorum achieved with {len(verifier_sigs)} signatures: "
            f"{[s.sat_id for s in verifier_sigs]}"
        )

        return True, verifier_sigs

    # =========================================================================
    # HOOKS
    # =========================================================================

    def add_pre_sign_hook(self, hook: Callable[[SignatureRequest], bool]) -> None:
        """Add a pre-sign validation hook."""
        self._pre_sign_hooks.append(hook)

    def add_post_collect_hook(
        self,
        hook: Callable[[str, List[SignatureResponse]], None],
    ) -> None:
        """Add a post-collect notification hook."""
        self._post_collect_hooks.append(hook)

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def cleanup_request(self, request_id: str) -> None:
        """Clean up state for a completed request."""
        self._pending_requests.pop(request_id, None)
        self._collected_signatures.pop(request_id, None)
        self._request_start_times.pop(request_id, None)

    def cleanup_expired_requests(self, max_age_ms: int = 60000) -> int:
        """Clean up requests older than max_age_ms."""
        now = time.monotonic()
        expired = []

        for request_id, start_time in self._request_start_times.items():
            age_ms = (now - start_time) * 1000
            if age_ms > max_age_ms:
                expired.append(request_id)

        for request_id in expired:
            self.cleanup_request(request_id)

        return len(expired)


# =============================================================================
# DEFAULT VALIDATOR REGISTRY
# =============================================================================


def create_default_validators() -> List[SATValidator]:
    """
    Create default SAT validator registry.

    In production, keys would be loaded from secure storage (HSM/vault).
    """
    # Generate keypairs for each validator
    validators = []

    roles = [
        ("sat-poi-001", "PoiVerifier"),
        ("sat-risk-001", "RiskGuardian"),
        ("sat-gov-001", "GovernanceEngine"),
        ("sat-res-001", "ResourceAllocator"),
        ("sat-evi-001", "EvidenceEngine"),
    ]

    for sat_id, role in roles:
        try:
            keypair = generate_keypair()
            validators.append(
                SATValidator(
                    sat_id=sat_id,
                    public_key=keypair.public_key,
                    role=role,
                )
            )
        except ImportError:
            # PyNaCl not available, use placeholder keys
            validators.append(
                SATValidator(
                    sat_id=sat_id,
                    public_key=bytes(32),  # Placeholder
                    role=role,
                )
            )

    return validators


# Default validators (lazy initialization)
_default_validators: Optional[List[SATValidator]] = None


def get_default_validators() -> List[SATValidator]:
    """Get the default SAT validator registry."""
    global _default_validators
    if _default_validators is None:
        _default_validators = create_default_validators()
    return _default_validators


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_distributed_quorum(
    synapse_client: Optional[Any] = None,
    strict_mode: bool = True,
) -> DistributedQuorum:
    """
    Create a DistributedQuorum with default validators.

    Args:
        synapse_client: Optional SynapseBus for Redis communication
        strict_mode: If True, require REQUIRED_ROLES to sign

    Returns:
        Configured DistributedQuorum instance
    """
    return DistributedQuorum(
        validators=get_default_validators(),
        synapse_client=synapse_client,
        strict_mode=strict_mode,
    )


async def quick_quorum(
    envelope_digest: str,
    gate_results: List[Dict[str, Any]],
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> Tuple[bool, List[VerifierSignature]]:
    """
    Quick quorum achievement with default configuration.

    Convenience function for simple quorum collection.
    """
    quorum = create_distributed_quorum()
    return await quorum.achieve_quorum(
        envelope_digest=envelope_digest,
        gate_results=gate_results,
        timeout_ms=timeout_ms,
    )


# =============================================================================
# ALIASES FOR BACKWARD COMPATIBILITY
# =============================================================================

DEFAULT_SAT_VALIDATORS = get_default_validators
