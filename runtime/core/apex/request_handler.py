"""
BIZRA Apex Orchestrator - Request Handler
==========================================
Request ingestion layer handling HTTP, CLI, and A2A requests.
Creates PCI envelopes from raw requests and routes to appropriate processing mode.

Architecture:
    ┌────────────────────────────────────────────────────────────────────────┐
    │                         REQUEST HANDLER                                 │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   ┌───────────┐    ┌───────────┐    ┌───────────┐                      │
    │   │   HTTP    │    │    CLI    │    │    A2A    │                      │
    │   │ Requests  │    │ Requests  │    │ Requests  │                      │
    │   └─────┬─────┘    └─────┬─────┘    └─────┬─────┘                      │
    │         │                │                │                             │
    │         └────────────────┼────────────────┘                             │
    │                          │                                              │
    │                          ▼                                              │
    │                  ┌───────────────┐                                      │
    │                  │    VALIDATE   │                                      │
    │                  │    REQUEST    │                                      │
    │                  └───────┬───────┘                                      │
    │                          │                                              │
    │                          ▼                                              │
    │                  ┌───────────────┐                                      │
    │                  │   DETERMINE   │                                      │
    │                  │     MODE      │                                      │
    │                  └───────┬───────┘                                      │
    │                          │                                              │
    │        ┌────────┬────────┼────────┬────────┐                            │
    │        ▼        ▼        ▼        ▼        ▼                            │
    │   STANDARD  ELEVATED  SOVEREIGN  TRANSCENDENT                          │
    │                          │                                              │
    │                          ▼                                              │
    │                  ┌───────────────┐                                      │
    │                  │  BUILD PCI    │                                      │
    │                  │   ENVELOPE    │                                      │
    │                  └───────┬───────┘                                      │
    │                          │                                              │
    │                          ▼                                              │
    │                  ┌───────────────┐                                      │
    │                  │  SIGN + EMIT  │                                      │
    │                  │    RECEIPT    │                                      │
    │                  └───────────────┘                                      │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

Status: PRODUCTION
Alignment: BIZRA_SOT.md Section 3.1 (Ihsan IM >= 0.95)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# PCI Protocol imports
from core.pci.envelope import PCIEnvelope, EnvelopeBuilder
from core.pci.crypto import (
    generate_keypair,
    blake3_digest,
    KeyPair,
)
from core.pci.types import (
    AgentType,
    Urgency,
    IHSAN_THRESHOLD,
    SNR_THRESHOLD_DEFAULT,
    utc_now_iso,
    generate_receipt_id,
)
from core.pci.gates import verify_envelope
from core.pci.reject_codes import RejectionResponse

logger = logging.getLogger(__name__)


# =============================================================================
# ORCHESTRATION MODE ENUM
# =============================================================================


class OrchestrationMode(str, Enum):
    """
    Orchestration mode determining processing complexity and verification depth.

    Modes escalate in verification stringency and resource allocation:
    - STANDARD: Normal operation, basic verification
    - ELEVATED: Enhanced verification for sensitive operations
    - SOVEREIGN: Offline-capable, full cryptographic verification
    - TRANSCENDENT: Maximum verification, formal proofs required
    """

    STANDARD = "standard"
    ELEVATED = "elevated"
    SOVEREIGN = "sovereign"  # Offline mode with full local verification
    TRANSCENDENT = "transcendent"  # Maximum verification including formal proofs


# =============================================================================
# REQUEST SOURCE ENUM
# =============================================================================


class RequestSource(str, Enum):
    """Source of the orchestration request."""

    HTTP = "http"
    CLI = "cli"
    A2A = "a2a"  # Agent-to-Agent
    INTERNAL = "internal"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class OrchestrationRequest:
    """
    Incoming orchestration request from any source.

    Attributes:
        task: The task description/instruction to execute
        context: Additional context for task execution
        agent_id: ID of the requesting agent
        mode: Requested orchestration mode
        scores: Pre-computed quality scores (optional)
        metadata: Additional request metadata (optional)
        source: Request source (HTTP, CLI, A2A)
        parent_envelope_id: For chained requests, the parent envelope ID
        priority: Request priority (0-100, higher = more urgent)
    """

    task: str
    context: Dict[str, Any]
    agent_id: str
    mode: OrchestrationMode
    scores: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    source: RequestSource = RequestSource.INTERNAL
    parent_envelope_id: Optional[str] = None
    priority: int = 50

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task": self.task,
            "context": self.context,
            "agent_id": self.agent_id,
            "mode": self.mode.value,
            "scores": self.scores,
            "metadata": self.metadata,
            "source": self.source.value,
            "parent_envelope_id": self.parent_envelope_id,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestrationRequest":
        """Deserialize from dictionary."""
        return cls(
            task=data["task"],
            context=data.get("context", {}),
            agent_id=data["agent_id"],
            mode=OrchestrationMode(data.get("mode", "standard")),
            scores=data.get("scores"),
            metadata=data.get("metadata"),
            source=RequestSource(data.get("source", "internal")),
            parent_envelope_id=data.get("parent_envelope_id"),
            priority=data.get("priority", 50),
        )


@dataclass
class OrchestrationResult:
    """
    Result of orchestration processing.

    Attributes:
        success: Whether the orchestration succeeded
        result: The result data (if successful)
        receipt_id: Unique receipt ID for this operation
        envelope_digest: BLAKE3 digest of the PCI envelope
        quality_metrics: Quality metrics from processing
        rejection: Rejection details (if failed)
        evidence_chain: Chain of evidence receipt IDs
        processing_time_ms: Total processing time in milliseconds
        mode: The orchestration mode used
    """

    success: bool
    result: Optional[Any]
    receipt_id: str
    envelope_digest: str
    quality_metrics: Dict[str, float]
    rejection: Optional[Dict[str, Any]] = None
    evidence_chain: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    mode: OrchestrationMode = OrchestrationMode.STANDARD

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "result": self.result,
            "receipt_id": self.receipt_id,
            "envelope_digest": self.envelope_digest,
            "quality_metrics": self.quality_metrics,
            "rejection": self.rejection,
            "evidence_chain": self.evidence_chain,
            "processing_time_ms": self.processing_time_ms,
            "mode": self.mode.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestrationResult":
        """Deserialize from dictionary."""
        return cls(
            success=data["success"],
            result=data.get("result"),
            receipt_id=data["receipt_id"],
            envelope_digest=data["envelope_digest"],
            quality_metrics=data.get("quality_metrics", {}),
            rejection=data.get("rejection"),
            evidence_chain=data.get("evidence_chain", []),
            processing_time_ms=data.get("processing_time_ms", 0.0),
            mode=OrchestrationMode(data.get("mode", "standard")),
        )


@dataclass
class ValidationError:
    """Request validation error."""

    field: str
    message: str
    code: str


# =============================================================================
# REQUEST HANDLER
# =============================================================================


class RequestHandler:
    """
    Request ingestion layer for the BIZRA Apex Orchestrator.

    Handles:
    - HTTP requests (REST API)
    - CLI requests (command line)
    - A2A requests (Agent-to-Agent protocol)

    Creates cryptographically signed PCI envelopes from raw requests
    and routes them to the appropriate processing mode.

    Features:
    - Ed25519 envelope signing
    - BLAKE3 digest computation
    - Nonce generation for replay protection
    - Mode determination based on request complexity
    - Full type hints and validation

    Example:
        handler = RequestHandler(envelope_builder)
        envelope = await handler.handle(request)
    """

    # Mode thresholds for automatic determination
    MODE_THRESHOLDS = {
        "complexity_elevated": 0.6,  # Task complexity >= 0.6 -> ELEVATED
        "complexity_sovereign": 0.8,  # Task complexity >= 0.8 -> SOVEREIGN
        "complexity_transcendent": 0.95,  # Task complexity >= 0.95 -> TRANSCENDENT
        "sensitivity_elevated": 0.5,  # Data sensitivity >= 0.5 -> ELEVATED
        "sensitivity_sovereign": 0.7,  # Data sensitivity >= 0.7 -> SOVEREIGN
    }

    # Keywords that trigger mode escalation
    MODE_ESCALATION_KEYWORDS = {
        OrchestrationMode.ELEVATED: [
            "sensitive",
            "private",
            "confidential",
            "secure",
            "financial",
            "medical",
            "legal",
            "compliance",
        ],
        OrchestrationMode.SOVEREIGN: [
            "offline",
            "sovereign",
            "air-gapped",
            "isolated",
            "critical",
            "emergency",
            "failsafe",
        ],
        OrchestrationMode.TRANSCENDENT: [
            "formal",
            "proof",
            "verified",
            "certified",
            "audit",
            "regulatory",
            "immutable",
        ],
    }

    def __init__(
        self,
        envelope_builder: EnvelopeBuilder,
        keypair: Optional[KeyPair] = None,
        policy_hash: Optional[str] = None,
        state_hash: Optional[str] = None,
        ihsan_threshold: float = IHSAN_THRESHOLD,
        snr_threshold: float = SNR_THRESHOLD_DEFAULT,
        auto_sign: bool = True,
        validate_gates: bool = True,
    ):
        """
        Initialize the request handler.

        Args:
            envelope_builder: Builder for creating PCI envelopes
            keypair: Ed25519 keypair for signing (generates new if None)
            policy_hash: Current constitution hash
            state_hash: Current state hash
            ihsan_threshold: Minimum Ihsan score for validation
            snr_threshold: Minimum SNR score for validation
            auto_sign: Whether to automatically sign envelopes
            validate_gates: Whether to validate through gate chain
        """
        self.envelope_builder = envelope_builder
        self.keypair = keypair or generate_keypair()
        self.policy_hash = policy_hash or self._compute_default_policy_hash()
        self.state_hash = state_hash or self._compute_default_state_hash()
        self.ihsan_threshold = ihsan_threshold
        self.snr_threshold = snr_threshold
        self.auto_sign = auto_sign
        self.validate_gates = validate_gates

        # Evidence chain tracking
        self._evidence_chain: List[str] = []

        logger.info(
            "RequestHandler initialized",
            extra={
                "public_key": self.keypair.public_key_hex[:16] + "...",
                "policy_hash": self.policy_hash[:16] + "...",
                "ihsan_threshold": self.ihsan_threshold,
            },
        )

    def _compute_default_policy_hash(self) -> str:
        """Compute default policy hash from constitution."""
        # Default constitution data structure
        constitution = {
            "version": "1.0.0",
            "ihsan_threshold": self.ihsan_threshold,
            "snr_threshold": self.snr_threshold,
            "timestamp": utc_now_iso(),
        }
        return blake3_digest(str(constitution).encode())

    def _compute_default_state_hash(self) -> str:
        """Compute default state hash."""
        state = {
            "initialized": True,
            "timestamp": utc_now_iso(),
        }
        return blake3_digest(str(state).encode())

    async def handle(
        self,
        request: OrchestrationRequest,
    ) -> PCIEnvelope:
        """
        Handle an orchestration request and create a PCI envelope.

        This is the main entry point for request processing. It:
        1. Validates the request
        2. Determines the appropriate mode
        3. Creates a PCI envelope
        4. Signs the envelope (if auto_sign enabled)
        5. Validates through gate chain (if enabled)

        Args:
            request: The orchestration request to handle

        Returns:
            A cryptographically signed PCI envelope

        Raises:
            ValueError: If request validation fails
            RuntimeError: If envelope creation or signing fails
        """
        start_time = time.perf_counter()

        # Step 1: Validate request
        validation_errors = self.validate_request(request)
        if validation_errors:
            error_messages = [f"{e.field}: {e.message}" for e in validation_errors]
            raise ValueError(f"Request validation failed: {'; '.join(error_messages)}")

        # Step 2: Determine mode (may upgrade from requested mode)
        final_mode = self.determine_mode(request)
        if final_mode != request.mode:
            logger.info(
                f"Mode escalated from {request.mode.value} to {final_mode.value}",
                extra={"agent_id": request.agent_id, "task": request.task[:50]},
            )

        # Step 3: Extract scores or use defaults
        ihsan_score = (
            request.scores.get("ihsan", self.ihsan_threshold)
            if request.scores
            else self.ihsan_threshold
        )
        snr_score = (
            request.scores.get("snr", self.snr_threshold)
            if request.scores
            else self.snr_threshold
        )

        # Step 4: Build envelope
        envelope = self._build_envelope(request, final_mode, ihsan_score, snr_score)

        # Step 5: Sign envelope
        if self.auto_sign:
            envelope = envelope.sign(self.keypair.private_key)

        # Step 6: Validate through gate chain (if enabled)
        if self.validate_gates:
            passed, rejection, gate_results = verify_envelope(
                envelope,
                policy_hash=self.policy_hash,
                state_hash=self.state_hash,
                ihsan_threshold=self.ihsan_threshold,
                snr_threshold=self.snr_threshold,
                require_expensive=(
                    final_mode
                    in [OrchestrationMode.SOVEREIGN, OrchestrationMode.TRANSCENDENT]
                ),
            )

            if not passed and rejection:
                logger.warning(
                    f"Gate validation failed: {rejection.code.value}",
                    extra={
                        "envelope_id": envelope.envelope_id,
                        "rejection_code": rejection.code.value,
                    },
                )
                # Still return envelope, but log rejection for audit

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Request handled in {elapsed_ms:.2f}ms",
            extra={
                "envelope_id": envelope.envelope_id,
                "mode": final_mode.value,
            },
        )

        return envelope

    def _build_envelope(
        self,
        request: OrchestrationRequest,
        mode: OrchestrationMode,
        ihsan_score: float,
        snr_score: float,
    ) -> PCIEnvelope:
        """
        Build a PCI envelope from the request.

        Args:
            request: The orchestration request
            mode: The determined orchestration mode
            ihsan_score: The Ihsan score to use
            snr_score: The SNR score to use

        Returns:
            An unsigned PCI envelope
        """
        # Determine urgency based on priority and mode
        urgency = self._determine_urgency(request, mode)

        # Build action from mode and task type
        action = f"orchestrate_{mode.value}"

        # Build payload data
        data = {
            "task": request.task,
            "context": request.context,
            "mode": mode.value,
            "source": request.source.value,
            "priority": request.priority,
            "metadata": request.metadata or {},
        }

        if request.parent_envelope_id:
            data["parent_envelope_id"] = request.parent_envelope_id

        # Use the builder pattern
        builder = EnvelopeBuilder()
        envelope = (
            builder.with_sender(
                AgentType.PAT, request.agent_id, self.keypair.public_key_hex
            )
            .with_action(action, data)
            .with_policy(self.policy_hash)
            .with_state(self.state_hash)
            .with_scores(ihsan=ihsan_score, snr=snr_score)
            .with_urgency(urgency)
            .build()
        )

        return envelope

    def _determine_urgency(
        self,
        request: OrchestrationRequest,
        mode: OrchestrationMode,
    ) -> Urgency:
        """Determine the urgency level based on request and mode."""
        # High priority requests get real-time urgency
        if request.priority >= 90:
            return Urgency.REAL_TIME

        # Mode-based urgency
        if mode == OrchestrationMode.TRANSCENDENT:
            return Urgency.BATCH  # Formal verification takes time
        elif mode == OrchestrationMode.SOVEREIGN:
            return Urgency.NEAR_REAL_TIME
        elif mode == OrchestrationMode.ELEVATED:
            return Urgency.NEAR_REAL_TIME

        # Default based on priority
        if request.priority >= 70:
            return Urgency.NEAR_REAL_TIME
        elif request.priority >= 30:
            return Urgency.BATCH
        else:
            return Urgency.DEFERRED

    def determine_mode(self, request: OrchestrationRequest) -> OrchestrationMode:
        """
        Determine the appropriate orchestration mode for a request.

        Mode determination is based on:
        1. Explicit mode in request (baseline)
        2. Task complexity analysis
        3. Keyword detection for sensitivity
        4. Context flags

        The mode can only be escalated, never downgraded from the request.

        Args:
            request: The orchestration request

        Returns:
            The determined orchestration mode (may be higher than requested)
        """
        current_mode = request.mode

        # Check for keyword-based escalation
        task_lower = request.task.lower()
        context_str = str(request.context).lower()
        combined_text = f"{task_lower} {context_str}"

        # Check TRANSCENDENT keywords first (highest priority)
        for keyword in self.MODE_ESCALATION_KEYWORDS[OrchestrationMode.TRANSCENDENT]:
            if keyword in combined_text:
                if self._mode_value(OrchestrationMode.TRANSCENDENT) > self._mode_value(
                    current_mode
                ):
                    current_mode = OrchestrationMode.TRANSCENDENT
                break

        # Check SOVEREIGN keywords
        for keyword in self.MODE_ESCALATION_KEYWORDS[OrchestrationMode.SOVEREIGN]:
            if keyword in combined_text:
                if self._mode_value(OrchestrationMode.SOVEREIGN) > self._mode_value(
                    current_mode
                ):
                    current_mode = OrchestrationMode.SOVEREIGN
                break

        # Check ELEVATED keywords
        for keyword in self.MODE_ESCALATION_KEYWORDS[OrchestrationMode.ELEVATED]:
            if keyword in combined_text:
                if self._mode_value(OrchestrationMode.ELEVATED) > self._mode_value(
                    current_mode
                ):
                    current_mode = OrchestrationMode.ELEVATED
                break

        # Check complexity from scores
        if request.scores:
            complexity = request.scores.get("complexity", 0.0)

            if complexity >= self.MODE_THRESHOLDS["complexity_transcendent"]:
                if self._mode_value(OrchestrationMode.TRANSCENDENT) > self._mode_value(
                    current_mode
                ):
                    current_mode = OrchestrationMode.TRANSCENDENT
            elif complexity >= self.MODE_THRESHOLDS["complexity_sovereign"]:
                if self._mode_value(OrchestrationMode.SOVEREIGN) > self._mode_value(
                    current_mode
                ):
                    current_mode = OrchestrationMode.SOVEREIGN
            elif complexity >= self.MODE_THRESHOLDS["complexity_elevated"]:
                if self._mode_value(OrchestrationMode.ELEVATED) > self._mode_value(
                    current_mode
                ):
                    current_mode = OrchestrationMode.ELEVATED

        # Check context flags
        if request.context.get("requires_formal_verification"):
            if self._mode_value(OrchestrationMode.TRANSCENDENT) > self._mode_value(
                current_mode
            ):
                current_mode = OrchestrationMode.TRANSCENDENT

        if request.context.get("offline_mode") or request.context.get("air_gapped"):
            if self._mode_value(OrchestrationMode.SOVEREIGN) > self._mode_value(
                current_mode
            ):
                current_mode = OrchestrationMode.SOVEREIGN

        return current_mode

    def _mode_value(self, mode: OrchestrationMode) -> int:
        """Get numeric value for mode comparison (higher = more stringent)."""
        values = {
            OrchestrationMode.STANDARD: 1,
            OrchestrationMode.ELEVATED: 2,
            OrchestrationMode.SOVEREIGN: 3,
            OrchestrationMode.TRANSCENDENT: 4,
        }
        return values.get(mode, 0)

    def validate_request(self, request: OrchestrationRequest) -> List[ValidationError]:
        """
        Validate an orchestration request.

        Validation checks:
        - Task is non-empty
        - Agent ID is valid
        - Mode is valid
        - Scores are within valid ranges
        - Required context fields are present

        Args:
            request: The orchestration request to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors: List[ValidationError] = []

        # Task validation
        if not request.task or not request.task.strip():
            errors.append(
                ValidationError(
                    field="task",
                    message="Task cannot be empty",
                    code="EMPTY_TASK",
                )
            )
        elif len(request.task) > 100000:  # 100KB max
            errors.append(
                ValidationError(
                    field="task",
                    message="Task exceeds maximum length (100KB)",
                    code="TASK_TOO_LONG",
                )
            )

        # Agent ID validation
        if not request.agent_id or not request.agent_id.strip():
            errors.append(
                ValidationError(
                    field="agent_id",
                    message="Agent ID cannot be empty",
                    code="EMPTY_AGENT_ID",
                )
            )
        elif len(request.agent_id) > 256:
            errors.append(
                ValidationError(
                    field="agent_id",
                    message="Agent ID exceeds maximum length (256 chars)",
                    code="AGENT_ID_TOO_LONG",
                )
            )

        # Mode validation
        if request.mode not in OrchestrationMode:
            errors.append(
                ValidationError(
                    field="mode",
                    message=f"Invalid mode: {request.mode}",
                    code="INVALID_MODE",
                )
            )

        # Scores validation
        if request.scores:
            for score_name, score_value in request.scores.items():
                if not isinstance(score_value, (int, float)):
                    errors.append(
                        ValidationError(
                            field=f"scores.{score_name}",
                            message=f"Score must be numeric, got {type(score_value).__name__}",
                            code="INVALID_SCORE_TYPE",
                        )
                    )
                elif score_value < 0.0 or score_value > 1.0:
                    errors.append(
                        ValidationError(
                            field=f"scores.{score_name}",
                            message=f"Score must be between 0.0 and 1.0, got {score_value}",
                            code="SCORE_OUT_OF_RANGE",
                        )
                    )

        # Priority validation
        if request.priority < 0 or request.priority > 100:
            errors.append(
                ValidationError(
                    field="priority",
                    message=f"Priority must be between 0 and 100, got {request.priority}",
                    code="PRIORITY_OUT_OF_RANGE",
                )
            )

        # Context validation (basic structure check)
        if request.context is not None and not isinstance(request.context, dict):
            errors.append(
                ValidationError(
                    field="context",
                    message="Context must be a dictionary",
                    code="INVALID_CONTEXT_TYPE",
                )
            )

        return errors

    async def create_result(
        self,
        envelope: PCIEnvelope,
        success: bool,
        result: Optional[Any] = None,
        rejection: Optional[RejectionResponse] = None,
        processing_time_ms: float = 0.0,
    ) -> OrchestrationResult:
        """
        Create an orchestration result from a processed envelope.

        Args:
            envelope: The processed PCI envelope
            success: Whether processing succeeded
            result: The result data (if successful)
            rejection: The rejection response (if failed)
            processing_time_ms: Processing time in milliseconds

        Returns:
            An OrchestrationResult with evidence chain
        """
        receipt_id = generate_receipt_id()
        envelope_digest_value = envelope.compute_digest()

        # Extract mode from payload
        mode_str = envelope.payload.data.get("mode", "standard")
        mode = OrchestrationMode(mode_str)

        # Build quality metrics
        quality_metrics = {
            "ihsan_score": envelope.metadata.ihsan_score,
            "snr_score": envelope.metadata.snr_score,
        }

        # Add rejection to evidence chain if failed
        evidence_chain = list(self._evidence_chain)
        evidence_chain.append(receipt_id)

        rejection_dict = None
        if rejection:
            rejection_dict = rejection.to_dict()

        result_obj = OrchestrationResult(
            success=success,
            result=result,
            receipt_id=receipt_id,
            envelope_digest=envelope_digest_value,
            quality_metrics=quality_metrics,
            rejection=rejection_dict,
            evidence_chain=evidence_chain,
            processing_time_ms=processing_time_ms,
            mode=mode,
        )

        # Add to evidence chain for next operation
        self._evidence_chain.append(receipt_id)

        return result_obj

    def update_policy_hash(self, policy_hash: str) -> None:
        """Update the current policy hash."""
        self.policy_hash = policy_hash
        logger.info(f"Policy hash updated: {policy_hash[:16]}...")

    def update_state_hash(self, state_hash: str) -> None:
        """Update the current state hash."""
        self.state_hash = state_hash
        logger.info(f"State hash updated: {state_hash[:16]}...")

    def rotate_keypair(self) -> KeyPair:
        """
        Rotate the signing keypair.

        Returns:
            The new keypair
        """
        old_public = self.keypair.public_key_hex[:16]
        self.keypair = generate_keypair()
        logger.info(
            f"Keypair rotated: {old_public}... -> {self.keypair.public_key_hex[:16]}..."
        )
        return self.keypair

    def clear_evidence_chain(self) -> None:
        """Clear the evidence chain (for new sessions)."""
        self._evidence_chain.clear()
        logger.debug("Evidence chain cleared")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_request_handler(
    policy_hash: Optional[str] = None,
    state_hash: Optional[str] = None,
    keypair: Optional[KeyPair] = None,
    ihsan_threshold: float = IHSAN_THRESHOLD,
    snr_threshold: float = SNR_THRESHOLD_DEFAULT,
) -> RequestHandler:
    """
    Factory function to create a RequestHandler with sensible defaults.

    Args:
        policy_hash: Current constitution hash (computed if None)
        state_hash: Current state hash (computed if None)
        keypair: Ed25519 keypair (generated if None)
        ihsan_threshold: Minimum Ihsan score
        snr_threshold: Minimum SNR score

    Returns:
        A configured RequestHandler instance
    """
    builder = EnvelopeBuilder()

    return RequestHandler(
        envelope_builder=builder,
        keypair=keypair,
        policy_hash=policy_hash,
        state_hash=state_hash,
        ihsan_threshold=ihsan_threshold,
        snr_threshold=snr_threshold,
    )


# =============================================================================
# HTTP REQUEST ADAPTER
# =============================================================================


def from_http_request(
    body: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    agent_id: Optional[str] = None,
) -> OrchestrationRequest:
    """
    Create an OrchestrationRequest from an HTTP request.

    Args:
        body: The request body (JSON parsed)
        headers: HTTP headers (optional)
        agent_id: Override agent ID (uses header or body if None)

    Returns:
        An OrchestrationRequest configured for HTTP source
    """
    # Extract agent ID from headers or body
    resolved_agent_id = agent_id
    if not resolved_agent_id and headers:
        resolved_agent_id = headers.get("X-Agent-ID") or headers.get("x-agent-id")
    if not resolved_agent_id:
        resolved_agent_id = body.get("agent_id", "http-anonymous")

    # Extract mode
    mode_str = body.get("mode", "standard")
    try:
        mode = OrchestrationMode(mode_str)
    except ValueError:
        mode = OrchestrationMode.STANDARD

    return OrchestrationRequest(
        task=body.get("task", ""),
        context=body.get("context", {}),
        agent_id=resolved_agent_id,
        mode=mode,
        scores=body.get("scores"),
        metadata=body.get("metadata"),
        source=RequestSource.HTTP,
        parent_envelope_id=body.get("parent_envelope_id"),
        priority=body.get("priority", 50),
    )


# =============================================================================
# CLI REQUEST ADAPTER
# =============================================================================


def from_cli_request(
    task: str,
    args: Optional[Dict[str, Any]] = None,
    agent_id: str = "cli-user",
) -> OrchestrationRequest:
    """
    Create an OrchestrationRequest from a CLI invocation.

    Args:
        task: The task from command line
        args: Additional CLI arguments
        agent_id: CLI user identifier

    Returns:
        An OrchestrationRequest configured for CLI source
    """
    args = args or {}

    # Extract mode from args
    mode_str = args.get("mode", "standard")
    try:
        mode = OrchestrationMode(mode_str)
    except ValueError:
        mode = OrchestrationMode.STANDARD

    return OrchestrationRequest(
        task=task,
        context=args.get("context", {}),
        agent_id=agent_id,
        mode=mode,
        scores=args.get("scores"),
        metadata=args.get("metadata"),
        source=RequestSource.CLI,
        parent_envelope_id=args.get("parent_envelope_id"),
        priority=args.get("priority", 50),
    )


# =============================================================================
# A2A REQUEST ADAPTER
# =============================================================================


def from_a2a_request(
    message: Dict[str, Any],
    sender_agent_id: str,
) -> OrchestrationRequest:
    """
    Create an OrchestrationRequest from an Agent-to-Agent message.

    Args:
        message: The A2A message payload
        sender_agent_id: ID of the sending agent

    Returns:
        An OrchestrationRequest configured for A2A source
    """
    # A2A messages typically have higher priority
    priority = message.get("priority", 70)

    # Extract mode
    mode_str = message.get("mode", "elevated")  # Default to elevated for A2A
    try:
        mode = OrchestrationMode(mode_str)
    except ValueError:
        mode = OrchestrationMode.ELEVATED

    return OrchestrationRequest(
        task=message.get("task", ""),
        context=message.get("context", {}),
        agent_id=sender_agent_id,
        mode=mode,
        scores=message.get("scores"),
        metadata=message.get("metadata"),
        source=RequestSource.A2A,
        parent_envelope_id=message.get("parent_envelope_id"),
        priority=priority,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "OrchestrationMode",
    "RequestSource",
    # Data classes
    "OrchestrationRequest",
    "OrchestrationResult",
    "ValidationError",
    # Main class
    "RequestHandler",
    # Factory functions
    "create_request_handler",
    # Adapters
    "from_http_request",
    "from_cli_request",
    "from_a2a_request",
]
