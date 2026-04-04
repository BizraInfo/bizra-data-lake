"""
BIZRA Genesis Phase State Machine
=================================
Four-phase genesis lifecycle with fail-closed enforcement.

Phases:
    0. PRIMORDIAL     - Integrity checks, cryptographic validation
    1. AWAKENING      - Agent initialization, warm pool bootstrap
    2. CRYSTALLIZATION - Consensus formation, quorum establishment
    3. ACTIVATION     - Full operational mode

Status: PRODUCTION
Domain: bizra-genesis-v1:
Alignment: BIZRA_SOT.md Section 3.1 (Ihsan IM >= 0.95)

CRITICAL: Fail-closed enforcement - any ambiguous state MUST result in rejection.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

# Try to import BLAKE3 for cryptographic hashing
try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


# =============================================================================
# CONSTANTS
# =============================================================================

GENESIS_DOMAIN_PREFIX = "bizra-genesis-v1:"
IHSAN_THRESHOLD = 0.95
SNR_THRESHOLD = 0.98


# =============================================================================
# ENUMS
# =============================================================================


class PhaseState(IntEnum):
    """
    Genesis phase states.

    Phases progress monotonically: 0 -> 1 -> 2 -> 3
    Backward transitions are forbidden (fail-closed).
    """

    PRIMORDIAL = 0  # Phase 0: Integrity checks
    AWAKENING = 1  # Phase 1: Agent initialization
    CRYSTALLIZATION = 2  # Phase 2: Consensus formation
    ACTIVATION = 3  # Phase 3: Full operation

    def __str__(self) -> str:
        return self.name


class TransitionResult(str, Enum):
    """Result of a phase transition attempt."""

    SUCCESS = "SUCCESS"
    REJECTED_REQUIREMENTS_NOT_MET = "REJECTED_REQUIREMENTS_NOT_MET"
    REJECTED_INVALID_TARGET = "REJECTED_INVALID_TARGET"
    REJECTED_BACKWARD_TRANSITION = "REJECTED_BACKWARD_TRANSITION"
    REJECTED_AMBIGUOUS_STATE = "REJECTED_AMBIGUOUS_STATE"
    REJECTED_IHSAN_THRESHOLD = "REJECTED_IHSAN_THRESHOLD"
    REJECTED_INTEGRITY_FAILURE = "REJECTED_INTEGRITY_FAILURE"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class PhaseRequirement:
    """
    A requirement that must be satisfied before transitioning to a phase.

    Attributes:
        name: Human-readable requirement name
        description: Detailed description of the requirement
        validator: Async function that returns (passed, details)
        critical: If True, failure blocks transition; if False, warning only
    """

    name: str
    description: str
    validator: Optional[Callable[[], Tuple[bool, Dict[str, Any]]]] = None
    critical: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "critical": self.critical,
        }


@dataclass
class PhaseTransition:
    """
    Record of a phase transition.

    Attributes:
        from_phase: Source phase
        to_phase: Target phase
        timestamp: ISO8601 timestamp
        requirements_met: List of satisfied requirements
        requirements_failed: List of failed requirements
        result: Transition result
        integrity_hash: BLAKE3 hash of transition data
    """

    from_phase: PhaseState
    to_phase: PhaseState
    timestamp: str
    requirements_met: List[str] = field(default_factory=list)
    requirements_failed: List[str] = field(default_factory=list)
    result: TransitionResult = TransitionResult.SUCCESS
    integrity_hash: str = ""
    ihsan_score: float = 0.0
    latency_ms: float = 0.0

    def __post_init__(self):
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute BLAKE3 integrity hash of transition data."""
        data = {
            "from_phase": self.from_phase.value,
            "to_phase": self.to_phase.value,
            "timestamp": self.timestamp,
            "requirements_met": sorted(self.requirements_met),
            "requirements_failed": sorted(self.requirements_failed),
            "result": self.result.value,
        }
        json_bytes = json.dumps(data, sort_keys=True).encode("utf-8")
        prefixed = (GENESIS_DOMAIN_PREFIX + "transition:").encode("utf-8") + json_bytes

        if HAS_BLAKE3:
            return blake3.blake3(prefixed).hexdigest()
        else:
            return hashlib.sha256(prefixed).hexdigest()

    def is_success(self) -> bool:
        return self.result == TransitionResult.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_phase": self.from_phase.value,
            "to_phase": self.to_phase.value,
            "timestamp": self.timestamp,
            "requirements_met": self.requirements_met,
            "requirements_failed": self.requirements_failed,
            "result": self.result.value,
            "integrity_hash": self.integrity_hash,
            "ihsan_score": self.ihsan_score,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseTransition":
        return cls(
            from_phase=PhaseState(data["from_phase"]),
            to_phase=PhaseState(data["to_phase"]),
            timestamp=data["timestamp"],
            requirements_met=data.get("requirements_met", []),
            requirements_failed=data.get("requirements_failed", []),
            result=TransitionResult(data["result"]),
            integrity_hash=data.get("integrity_hash", ""),
            ihsan_score=data.get("ihsan_score", 0.0),
            latency_ms=data.get("latency_ms", 0.0),
        )


@dataclass
class PhaseReceipt:
    """
    Evidence receipt for phase operations.

    Attributes:
        receipt_id: Unique receipt identifier
        phase: Current phase state
        operation: Operation type (transition, validation, etc.)
        timestamp: ISO8601 timestamp
        data: Operation-specific data
        integrity_hash: BLAKE3 hash of receipt data
    """

    receipt_id: str
    phase: PhaseState
    operation: str
    timestamp: str
    data: Dict[str, Any] = field(default_factory=dict)
    integrity_hash: str = ""

    def __post_init__(self):
        if not self.receipt_id:
            self.receipt_id = str(uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute BLAKE3 integrity hash of receipt data."""
        hash_data = {
            "receipt_id": self.receipt_id,
            "phase": self.phase.value,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "data": self.data,
        }
        json_bytes = json.dumps(hash_data, sort_keys=True).encode("utf-8")
        prefixed = (GENESIS_DOMAIN_PREFIX + "receipt:").encode("utf-8") + json_bytes

        if HAS_BLAKE3:
            return blake3.blake3(prefixed).hexdigest()
        else:
            return hashlib.sha256(prefixed).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "phase": self.phase.value,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "data": self.data,
            "integrity_hash": self.integrity_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseReceipt":
        return cls(
            receipt_id=data["receipt_id"],
            phase=PhaseState(data["phase"]),
            operation=data["operation"],
            timestamp=data["timestamp"],
            data=data.get("data", {}),
            integrity_hash=data.get("integrity_hash", ""),
        )


# =============================================================================
# GENESIS STATE MACHINE
# =============================================================================


class GenesisStateMachine:
    """
    Genesis phase state machine with fail-closed enforcement.

    The state machine manages the 4-phase genesis lifecycle:
    - Phase 0 (PRIMORDIAL): Integrity checks, cryptographic validation
    - Phase 1 (AWAKENING): Agent initialization, warm pool bootstrap
    - Phase 2 (CRYSTALLIZATION): Consensus formation, quorum establishment
    - Phase 3 (ACTIVATION): Full operational mode

    CRITICAL: Fail-closed enforcement - ambiguous states result in rejection.

    Attributes:
        current_phase: Current phase state
        transition_history: List of all transitions
        requirements: Phase-specific requirements

    Usage:
        machine = GenesisStateMachine()

        # Check if transition is possible
        if machine.can_transition(PhaseState.AWAKENING):
            result = await machine.transition(PhaseState.AWAKENING)
            if result.is_success():
                print("Transition successful")
    """

    def __init__(self, initial_phase: PhaseState = PhaseState.PRIMORDIAL):
        """
        Initialize the genesis state machine.

        Args:
            initial_phase: Starting phase (default: PRIMORDIAL)
        """
        self._current_phase = initial_phase
        self._transition_history: List[PhaseTransition] = []
        self._receipts: List[PhaseReceipt] = []
        self._lock = threading.Lock()

        # Phase-specific requirements
        self._requirements: Dict[PhaseState, List[PhaseRequirement]] = {
            PhaseState.PRIMORDIAL: [],  # Entry point, no requirements
            PhaseState.AWAKENING: self._get_awakening_requirements(),
            PhaseState.CRYSTALLIZATION: self._get_crystallization_requirements(),
            PhaseState.ACTIVATION: self._get_activation_requirements(),
        }

        # Custom validators registered by external systems
        self._custom_validators: Dict[PhaseState, List[Callable]] = {
            phase: [] for phase in PhaseState
        }

    @property
    def current_phase(self) -> PhaseState:
        """Get the current phase state."""
        with self._lock:
            return self._current_phase

    @property
    def transition_history(self) -> List[PhaseTransition]:
        """Get the transition history."""
        with self._lock:
            return list(self._transition_history)

    @property
    def receipts(self) -> List[PhaseReceipt]:
        """Get all phase receipts."""
        with self._lock:
            return list(self._receipts)

    def _get_awakening_requirements(self) -> List[PhaseRequirement]:
        """Get requirements for transitioning to AWAKENING phase."""
        return [
            PhaseRequirement(
                name="integrity_check",
                description="All cryptographic integrity checks must pass",
                critical=True,
            ),
            PhaseRequirement(
                name="key_validation",
                description="Ed25519 key pair must be valid and accessible",
                critical=True,
            ),
            PhaseRequirement(
                name="constitution_loaded",
                description="Constitution (ihsan_v1.yaml) must be loaded and parsed",
                critical=True,
            ),
        ]

    def _get_crystallization_requirements(self) -> List[PhaseRequirement]:
        """Get requirements for transitioning to CRYSTALLIZATION phase."""
        return [
            PhaseRequirement(
                name="agents_initialized",
                description="All PAT and SAT agents must be initialized",
                critical=True,
            ),
            PhaseRequirement(
                name="warm_pools_ready",
                description="Warm agent pools must be populated",
                critical=False,  # Warning only
            ),
            PhaseRequirement(
                name="memory_systems_online",
                description="All memory tiers (L1-L5) must be accessible",
                critical=True,
            ),
        ]

    def _get_activation_requirements(self) -> List[PhaseRequirement]:
        """Get requirements for transitioning to ACTIVATION phase."""
        return [
            PhaseRequirement(
                name="quorum_established",
                description="SAT quorum (3/5) must be achievable",
                critical=True,
            ),
            PhaseRequirement(
                name="consensus_verified",
                description="Initial consensus round must pass",
                critical=True,
            ),
            PhaseRequirement(
                name="ihsan_threshold_met",
                description=f"System Ihsan score must be >= {IHSAN_THRESHOLD}",
                critical=True,
            ),
            PhaseRequirement(
                name="snr_threshold_met",
                description=f"System SNR must be >= {SNR_THRESHOLD}",
                critical=True,
            ),
        ]

    def register_validator(
        self,
        phase: PhaseState,
        validator: Callable[[], Tuple[bool, Dict[str, Any]]],
    ) -> None:
        """
        Register a custom validator for a phase transition.

        Args:
            phase: Target phase for the validator
            validator: Async function returning (passed, details)
        """
        with self._lock:
            self._custom_validators[phase].append(validator)

    def get_phase_requirements(
        self, phase: Optional[PhaseState] = None
    ) -> List[PhaseRequirement]:
        """
        Get requirements for a specific phase.

        Args:
            phase: Target phase (default: next phase)

        Returns:
            List of PhaseRequirement objects
        """
        if phase is None:
            # Get requirements for next phase
            next_phase_value = self._current_phase.value + 1
            if next_phase_value > PhaseState.ACTIVATION.value:
                return []
            phase = PhaseState(next_phase_value)

        return self._requirements.get(phase, [])

    def can_transition(self, target_phase: PhaseState) -> Tuple[bool, str]:
        """
        Check if transition to target phase is possible.

        This performs structural validation only (not requirement validation).

        Args:
            target_phase: The phase to transition to

        Returns:
            Tuple of (can_transition, reason)
        """
        with self._lock:
            current = self._current_phase

        # Check for backward transition (forbidden)
        if target_phase.value < current.value:
            return False, f"Backward transitions forbidden: {current} -> {target_phase}"

        # Check for same phase (no-op)
        if target_phase.value == current.value:
            return False, f"Already in phase {current}"

        # Check for skipping phases
        if target_phase.value > current.value + 1:
            return False, f"Cannot skip phases: {current} -> {target_phase}"

        return True, "Transition structurally valid"

    async def transition(
        self,
        target_phase: PhaseState,
        ihsan_score: float = 0.0,
        force: bool = False,
    ) -> PhaseTransition:
        """
        Attempt to transition to a target phase.

        This validates all requirements and performs fail-closed enforcement.

        Args:
            target_phase: The phase to transition to
            ihsan_score: Current system Ihsan score
            force: If True, skip non-critical requirements (DANGEROUS)

        Returns:
            PhaseTransition with result

        CRITICAL: Ambiguous states result in rejection (fail-closed).
        """
        import time

        start_time = time.monotonic()

        with self._lock:
            current_phase = self._current_phase

        timestamp = datetime.now(timezone.utc).isoformat()

        # Structural validation
        can_proceed, reason = self.can_transition(target_phase)
        if not can_proceed:
            transition = PhaseTransition(
                from_phase=current_phase,
                to_phase=target_phase,
                timestamp=timestamp,
                result=(
                    TransitionResult.REJECTED_BACKWARD_TRANSITION
                    if "Backward" in reason
                    else TransitionResult.REJECTED_INVALID_TARGET
                ),
            )
            self._emit_receipt(transition)
            return transition

        # Ihsan threshold check (fail-closed)
        if target_phase == PhaseState.ACTIVATION:
            if ihsan_score < IHSAN_THRESHOLD:
                transition = PhaseTransition(
                    from_phase=current_phase,
                    to_phase=target_phase,
                    timestamp=timestamp,
                    result=TransitionResult.REJECTED_IHSAN_THRESHOLD,
                    ihsan_score=ihsan_score,
                    requirements_failed=[
                        f"ihsan_score={ihsan_score} < {IHSAN_THRESHOLD}"
                    ],
                )
                self._emit_receipt(transition)
                return transition

        # Validate requirements
        requirements = self.get_phase_requirements(target_phase)
        requirements_met: List[str] = []
        requirements_failed: List[str] = []

        for req in requirements:
            passed = True
            details: Dict[str, Any] = {}

            # Run custom validators
            if req.validator:
                try:
                    if asyncio.iscoroutinefunction(req.validator):
                        passed, details = await req.validator()
                    else:
                        passed, details = req.validator()
                except Exception as e:
                    passed = False
                    details = {"error": str(e)}

            if passed:
                requirements_met.append(req.name)
            else:
                requirements_failed.append(req.name)
                if req.critical and not force:
                    # Fail-closed: critical requirement failed
                    transition = PhaseTransition(
                        from_phase=current_phase,
                        to_phase=target_phase,
                        timestamp=timestamp,
                        requirements_met=requirements_met,
                        requirements_failed=requirements_failed,
                        result=TransitionResult.REJECTED_REQUIREMENTS_NOT_MET,
                        ihsan_score=ihsan_score,
                    )
                    self._emit_receipt(transition)
                    return transition

        # Run custom validators
        custom_validators = self._custom_validators.get(target_phase, [])
        for validator in custom_validators:
            try:
                if asyncio.iscoroutinefunction(validator):
                    passed, details = await validator()
                else:
                    passed, details = validator()

                if not passed:
                    # Fail-closed: custom validator failed
                    transition = PhaseTransition(
                        from_phase=current_phase,
                        to_phase=target_phase,
                        timestamp=timestamp,
                        requirements_met=requirements_met,
                        requirements_failed=requirements_failed + ["custom_validator"],
                        result=TransitionResult.REJECTED_REQUIREMENTS_NOT_MET,
                        ihsan_score=ihsan_score,
                    )
                    self._emit_receipt(transition)
                    return transition
            except Exception as e:
                # Fail-closed: exception in validator
                transition = PhaseTransition(
                    from_phase=current_phase,
                    to_phase=target_phase,
                    timestamp=timestamp,
                    requirements_met=requirements_met,
                    requirements_failed=requirements_failed
                    + [f"validator_exception: {e}"],
                    result=TransitionResult.REJECTED_AMBIGUOUS_STATE,
                    ihsan_score=ihsan_score,
                )
                self._emit_receipt(transition)
                return transition

        # All checks passed - perform transition
        with self._lock:
            # Double-check state hasn't changed (race condition protection)
            if self._current_phase != current_phase:
                transition = PhaseTransition(
                    from_phase=current_phase,
                    to_phase=target_phase,
                    timestamp=timestamp,
                    result=TransitionResult.REJECTED_AMBIGUOUS_STATE,
                    ihsan_score=ihsan_score,
                )
                self._emit_receipt(transition)
                return transition

            # Transition successful
            self._current_phase = target_phase
            latency_ms = (time.monotonic() - start_time) * 1000

            transition = PhaseTransition(
                from_phase=current_phase,
                to_phase=target_phase,
                timestamp=timestamp,
                requirements_met=requirements_met,
                requirements_failed=requirements_failed,  # Non-critical failures
                result=TransitionResult.SUCCESS,
                ihsan_score=ihsan_score,
                latency_ms=latency_ms,
            )
            self._transition_history.append(transition)

        self._emit_receipt(transition)
        return transition

    def _emit_receipt(self, transition: PhaseTransition) -> PhaseReceipt:
        """Emit a receipt for a transition operation."""
        receipt = PhaseReceipt(
            receipt_id=str(uuid4()),
            phase=transition.to_phase,
            operation="phase_transition",
            timestamp=transition.timestamp,
            data=transition.to_dict(),
        )
        with self._lock:
            self._receipts.append(receipt)
        return receipt

    def emit_phase_receipt(
        self,
        operation: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> PhaseReceipt:
        """
        Emit a custom receipt for the current phase.

        Args:
            operation: Operation type
            data: Operation-specific data

        Returns:
            PhaseReceipt with integrity hash
        """
        receipt = PhaseReceipt(
            receipt_id=str(uuid4()),
            phase=self.current_phase,
            operation=operation,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data or {},
        )
        with self._lock:
            self._receipts.append(receipt)
        return receipt

    def get_state_digest(self) -> str:
        """
        Compute a digest of the current state machine state.

        Returns:
            BLAKE3 hex digest of the state
        """
        with self._lock:
            state_data = {
                "current_phase": self._current_phase.value,
                "transition_count": len(self._transition_history),
                "receipt_count": len(self._receipts),
                "last_transition_hash": (
                    self._transition_history[-1].integrity_hash
                    if self._transition_history
                    else None
                ),
            }

        json_bytes = json.dumps(state_data, sort_keys=True).encode("utf-8")
        prefixed = (GENESIS_DOMAIN_PREFIX + "state:").encode("utf-8") + json_bytes

        if HAS_BLAKE3:
            return blake3.blake3(prefixed).hexdigest()
        else:
            return hashlib.sha256(prefixed).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state machine to dictionary."""
        with self._lock:
            return {
                "current_phase": self._current_phase.value,
                "transition_history": [t.to_dict() for t in self._transition_history],
                "receipts": [r.to_dict() for r in self._receipts],
                "state_digest": self.get_state_digest(),
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenesisStateMachine":
        """Deserialize state machine from dictionary."""
        machine = cls(initial_phase=PhaseState(data["current_phase"]))
        machine._transition_history = [
            PhaseTransition.from_dict(t) for t in data.get("transition_history", [])
        ]
        machine._receipts = [
            PhaseReceipt.from_dict(r) for r in data.get("receipts", [])
        ]
        return machine

    def __repr__(self) -> str:
        return (
            f"GenesisStateMachine(phase={self.current_phase}, "
            f"transitions={len(self._transition_history)})"
        )
