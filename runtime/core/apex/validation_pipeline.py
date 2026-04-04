"""
BIZRA Apex Orchestrator — Validation Pipeline
==============================================
Coordinates all validation components with fail-fast semantics.

Status: PRODUCTION
Alignment: constitution/pat_enforcement_v1.yaml

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        VALIDATION PIPELINE                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   PCIEnvelope ──▶ [ValidationContext]                                   │
    │                          │                                               │
    │                          ▼                                               │
    │          ┌───────────────────────────────────────┐                      │
    │          │        PCI GATE CHAIN                 │                      │
    │          │   ┌─────────┬─────────┬─────────┐    │                      │
    │          │   │  CHEAP  │ MEDIUM  │EXPENSIVE│    │                      │
    │          │   │ <10ms   │ <150ms  │ <2000ms │    │                      │
    │          │   │ SCHEMA  │  SNR    │  FATE   │    │                      │
    │          │   │ SIGN    │ IHSAN   │ FORMAL  │    │                      │
    │          │   │ TIME    │ POLICY  │         │    │                      │
    │          │   │ REPLAY  │         │         │    │                      │
    │          │   │ ROLE    │         │         │    │                      │
    │          │   └────┬────┴────┬────┴────┬────┘    │                      │
    │          │        │         │         │         │                      │
    │          │        ▼ fail?   ▼ fail?   ▼ fail?   │                      │
    │          │      REJECT    REJECT    REJECT      │                      │
    │          └───────────────────────────────────────┘                      │
    │                          │                                               │
    │                      PASS ▼                                              │
    │          ┌───────────────────────────────────────┐                      │
    │          │     PAT 5-GATE ENFORCEMENT            │                      │
    │          │  (if context.require_pat_enforcement)│                      │
    │          │   ┌───────┬───────┬───────┬──────┬───┐│                      │
    │          │   │ G1    │ G2    │ G3    │ G4   │G5 ││                      │
    │          │   │PreRsn │MidSyn │PostSyn│Pract │Rsp││                      │
    │          │   └───┬───┴───┬───┴───┬───┴──┬───┴─┬─┘│                      │
    │          │       ▼       ▼       ▼      ▼     ▼   │                      │
    │          │    fail?   fail?   fail?  warn   fail?│                      │
    │          └───────────────────────────────────────┘                      │
    │                          │                                               │
    │                      PASS ▼                                              │
    │          ┌───────────────────────────────────────┐                      │
    │          │     SNR ENFORCER                      │                      │
    │          │   (constitutional threshold check)    │                      │
    │          │   SNR < threshold → REJECT            │                      │
    │          │   SNR ≥ threshold → PASS + receipt    │                      │
    │          └───────────────────────────────────────┘                      │
    │                          │                                               │
    │                          ▼                                               │
    │                  [ValidationResult]                                      │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

Components:
    - PCI Gate Chain: Tiered verification (cheap/medium/expensive)
    - PAT Enforcement: 5-gate maximum quality validation
    - SNR Enforcer: Constitutional threshold compliance

Fail-Fast Semantics:
    - First failure terminates pipeline
    - All rejections emit receipts
    - Comprehensive latency tracking
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# PCI Gate Chain imports
from core.pci.gates import GateChain, GateResult
from core.pci.envelope import PCIEnvelope
from core.pci.reject_codes import RejectCode, RejectionResponse
from core.pci.types import (
    IHSAN_THRESHOLD,
    SNR_THRESHOLD_DEFAULT,
    utc_now_iso,
)

# PAT Enforcement imports
from bizra_kernel.pat_enforcement_pipeline import (
    PATEnforcementPipeline,
    PATEnforcementResult,
    PATRequest,
    GateResult as PATGateResult,
    GateID,
)

# SNR Enforcer imports
from bizra_kernel.snr_enforcer import (
    EnforcementContext,
    EnforcementResult,
    OperationType,
    get_snr_enforcer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("apex.validation_pipeline")


# =============================================================================
# VALIDATION CONTEXT
# =============================================================================


@dataclass
class ValidationContext:
    """
    Context for validation pipeline execution.

    Controls which validation stages are executed and provides
    additional context for gate decisions.
    """

    # Execution control
    require_expensive_gates: bool = False
    require_pat_enforcement: bool = False
    skip_snr_enforcement: bool = False

    # PAT enforcement data (if required)
    pat_request: Optional[PATRequest] = None

    # Policy & state hashes
    policy_hash: str = ""
    state_hash: str = ""

    # Threshold overrides (None = use defaults)
    ihsan_threshold: Optional[float] = None
    snr_threshold: Optional[float] = None

    # FATE checker (optional, for expensive tier)
    fate_checker: Optional[
        Callable[[PCIEnvelope], Tuple[bool, str, Dict[str, Any]]]
    ] = None

    # Formal verifier (optional, for expensive tier)
    formal_verifier: Optional[
        Callable[[PCIEnvelope], Tuple[bool, str, Dict[str, Any]]]
    ] = None

    # Additional metadata
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "require_expensive_gates": self.require_expensive_gates,
            "require_pat_enforcement": self.require_pat_enforcement,
            "skip_snr_enforcement": self.skip_snr_enforcement,
            "policy_hash": self.policy_hash[:16] + "..." if self.policy_hash else "",
            "state_hash": self.state_hash[:16] + "..." if self.state_hash else "",
            "ihsan_threshold": self.ihsan_threshold,
            "snr_threshold": self.snr_threshold,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "has_fate_checker": self.fate_checker is not None,
            "has_formal_verifier": self.formal_verifier is not None,
            "has_pat_request": self.pat_request is not None,
        }


# =============================================================================
# VALIDATION RESULT
# =============================================================================


@dataclass
class ValidationResult:
    """
    Complete result from validation pipeline.

    Contains results from all validation stages with comprehensive
    latency tracking and receipt evidence.
    """

    passed: bool
    pci_gate_results: List[GateResult]
    pat_gate_results: Optional[List[PATGateResult]]
    snr_result: EnforcementResult
    rejection: Optional[RejectionResponse]
    total_latency_ms: float
    receipt: Optional[Dict[str, Any]]

    # Stage-specific latencies
    pci_latency_ms: float = 0.0
    pat_latency_ms: float = 0.0
    snr_latency_ms: float = 0.0

    # Metadata
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    envelope_digest: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "pci_gate_results": [
                {
                    "gate": r.gate.value,
                    "passed": r.passed,
                    "latency_ms": r.latency_ms,
                    "details": r.details,
                }
                for r in self.pci_gate_results
            ],
            "pat_gate_results": (
                [r.to_dict() for r in self.pat_gate_results]
                if self.pat_gate_results
                else None
            ),
            "snr_result": self.snr_result.to_dict() if self.snr_result else None,
            "rejection": self.rejection.to_dict() if self.rejection else None,
            "total_latency_ms": self.total_latency_ms,
            "pci_latency_ms": self.pci_latency_ms,
            "pat_latency_ms": self.pat_latency_ms,
            "snr_latency_ms": self.snr_latency_ms,
            "receipt": self.receipt,
            "timestamp": self.timestamp,
            "envelope_digest": self.envelope_digest,
        }

    @property
    def gates_passed(self) -> List[str]:
        """Get list of all gates that passed."""
        passed_gates = [r.gate.value for r in self.pci_gate_results if r.passed]
        if self.pat_gate_results:
            passed_gates.extend(
                [r.gate_id.value for r in self.pat_gate_results if r.passed]
            )
        return passed_gates

    @property
    def failed_gate(self) -> Optional[str]:
        """Get the first gate that failed, if any."""
        for r in self.pci_gate_results:
            if not r.passed:
                return r.gate.value
        if self.pat_gate_results:
            for r in self.pat_gate_results:
                if not r.passed:
                    return r.gate_id.value
        if self.snr_result and not self.snr_result.passed:
            return "SNR_ENFORCER"
        return None


# =============================================================================
# VALIDATION PIPELINE
# =============================================================================


class ValidationPipeline:
    """
    Coordinates all validation components with fail-fast semantics.

    Execution Flow:
        1. PCI Gate Chain (CHEAP → MEDIUM → EXPENSIVE)
        2. PAT 5-Gate Enforcement (optional)
        3. SNR Enforcer (constitutional threshold)

    Features:
        - Fail-fast: First failure terminates pipeline
        - Latency tracking per tier and stage
        - Receipt emission on rejection
        - Async/await for non-blocking execution
        - Full type hints
    """

    def __init__(
        self,
        constitution_path: str = "constitution/pat_enforcement_v1.yaml",
        receipt_dir: Optional[str] = None,
        emit_receipts: bool = True,
        ihsan_threshold: float = IHSAN_THRESHOLD,
        snr_threshold: float = SNR_THRESHOLD_DEFAULT,
    ):
        """
        Initialize the validation pipeline.

        Args:
            constitution_path: Path to PAT constitution YAML
            receipt_dir: Directory for receipt output (default: docs/evidence/receipts/validation)
            emit_receipts: Whether to emit receipts on rejection
            ihsan_threshold: Default Ihsan threshold (0.95)
            snr_threshold: Default SNR threshold (0.70)
        """
        self.constitution_path = constitution_path
        self.emit_receipts = emit_receipts
        self.ihsan_threshold = ihsan_threshold
        self.snr_threshold = snr_threshold

        # Receipt directory
        if receipt_dir is None:
            receipt_dir = "docs/evidence/receipts/validation"
        self.receipt_dir = Path(receipt_dir)
        self.receipt_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SNR enforcer
        self.snr_enforcer = get_snr_enforcer(
            constitution_path=constitution_path,
            force_reload=False,
        )

        # PAT enforcement pipeline (lazy initialization)
        self._pat_pipeline: Optional[PATEnforcementPipeline] = None

        # Statistics
        self._validations: int = 0
        self._rejections: int = 0
        self._receipts_emitted: int = 0

        logger.info(
            f"ValidationPipeline initialized: "
            f"ihsan={ihsan_threshold}, snr={snr_threshold}, "
            f"receipts={emit_receipts}"
        )

    @property
    def pat_pipeline(self) -> PATEnforcementPipeline:
        """Lazy initialization of PAT enforcement pipeline."""
        if self._pat_pipeline is None:
            self._pat_pipeline = PATEnforcementPipeline(
                snr_minimum=0.98,  # PAT threshold
                novelty_minimum=0.75,
                ihsan_minimum=self.ihsan_threshold,
            )
        return self._pat_pipeline

    async def validate(
        self,
        envelope: PCIEnvelope,
        context: ValidationContext,
    ) -> ValidationResult:
        """
        Execute the full validation pipeline.

        Args:
            envelope: The PCI envelope to validate
            context: Validation context with configuration

        Returns:
            ValidationResult with pass/fail decision and evidence

        Flow:
            1. PCI Gate Chain
            2. PAT Enforcement (if context.require_pat_enforcement)
            3. SNR Enforcement (if not context.skip_snr_enforcement)
        """
        start_time = time.perf_counter()
        self._validations += 1

        envelope_digest = envelope.compute_digest()
        timestamp = utc_now_iso()

        logger.info(
            f"Validation pipeline started: envelope={envelope_digest[:16]}..., "
            f"session={context.session_id}, task={context.task_id}"
        )

        pci_gate_results: List[GateResult] = []
        pat_gate_results: Optional[List[PATGateResult]] = None
        snr_result: Optional[EnforcementResult] = None
        rejection: Optional[RejectionResponse] = None
        receipt: Optional[Dict[str, Any]] = None

        pci_latency_ms: float = 0.0
        pat_latency_ms: float = 0.0
        snr_latency_ms: float = 0.0

        try:
            # =================================================================
            # STAGE 1: PCI GATE CHAIN
            # =================================================================
            logger.info("Stage 1: PCI Gate Chain")
            pci_start = time.perf_counter()

            # Create gate chain with context parameters
            gate_chain = GateChain(
                current_policy_hash=context.policy_hash,
                current_state_hash=context.state_hash,
                ihsan_threshold=context.ihsan_threshold or self.ihsan_threshold,
                snr_threshold=context.snr_threshold or self.snr_threshold,
                fate_checker=context.fate_checker,
                formal_verifier=context.formal_verifier,
                use_snr_enforcer=True,  # Use SNR enforcer for medium tier
            )

            # Execute gate chain
            pci_passed, pci_rejection, pci_gate_results = gate_chain.verify(
                envelope=envelope,
                require_expensive=context.require_expensive_gates,
            )

            pci_latency_ms = (time.perf_counter() - pci_start) * 1000

            if not pci_passed:
                # FAIL-FAST: PCI gate failed
                self._rejections += 1
                rejection = pci_rejection

                logger.warning(
                    f"PCI Gate Chain REJECTED: {pci_rejection.code.name if pci_rejection else 'Unknown'}"
                )

                # Emit rejection receipt
                if self.emit_receipts and rejection:
                    receipt = await self._emit_rejection_receipt(
                        envelope_digest=envelope_digest,
                        timestamp=timestamp,
                        rejection=rejection,
                        context=context,
                        stage="PCI_GATE_CHAIN",
                        gate_results=pci_gate_results,
                    )

                total_latency_ms = (time.perf_counter() - start_time) * 1000

                return ValidationResult(
                    passed=False,
                    pci_gate_results=pci_gate_results,
                    pat_gate_results=None,
                    snr_result=EnforcementResult(
                        passed=False,
                        snr_score=envelope.metadata.snr_score,
                        threshold=context.snr_threshold or self.snr_threshold,
                    ),
                    rejection=rejection,
                    total_latency_ms=total_latency_ms,
                    pci_latency_ms=pci_latency_ms,
                    receipt=receipt,
                    envelope_digest=envelope_digest,
                )

            logger.info(
                f"PCI Gate Chain PASSED: {len(pci_gate_results)} gates, "
                f"latency={pci_latency_ms:.2f}ms"
            )

            # =================================================================
            # STAGE 2: PAT ENFORCEMENT (Optional)
            # =================================================================
            if context.require_pat_enforcement and context.pat_request:
                logger.info("Stage 2: PAT 5-Gate Enforcement")
                pat_start = time.perf_counter()

                # Execute PAT enforcement
                pat_result = await self.pat_pipeline.enforce(context.pat_request)

                pat_latency_ms = (time.perf_counter() - pat_start) * 1000
                pat_gate_results = pat_result.gate_results

                if not pat_result.passed:
                    # FAIL-FAST: PAT gate failed (except Gate 4 which only warns)
                    self._rejections += 1

                    # Find the failed gate
                    failed_gate = None
                    for gate_result in pat_gate_results:
                        if (
                            not gate_result.passed
                            and gate_result.gate_id != GateID.GATE_4_PRACTITIONER
                        ):
                            failed_gate = gate_result
                            break

                    if failed_gate:
                        # Create rejection response
                        rejection = RejectionResponse.rejection(
                            code=RejectCode.REJECT_IHSAN_BELOW_MIN,  # PAT uses stricter thresholds
                            message=f"PAT enforcement failed at {failed_gate.gate_id.value}: {failed_gate.evidence}",
                            envelope_digest=envelope_digest,
                            timestamp=timestamp,
                        )

                        logger.warning(
                            f"PAT Enforcement REJECTED: {failed_gate.gate_id.value}"
                        )

                        # Emit rejection receipt
                        if self.emit_receipts:
                            receipt = await self._emit_rejection_receipt(
                                envelope_digest=envelope_digest,
                                timestamp=timestamp,
                                rejection=rejection,
                                context=context,
                                stage="PAT_ENFORCEMENT",
                                gate_results=pci_gate_results,
                                pat_result=pat_result,
                            )

                        total_latency_ms = (time.perf_counter() - start_time) * 1000

                        return ValidationResult(
                            passed=False,
                            pci_gate_results=pci_gate_results,
                            pat_gate_results=pat_gate_results,
                            snr_result=EnforcementResult(
                                passed=False,
                                snr_score=pat_result.final_snr,
                                threshold=0.98,
                            ),
                            rejection=rejection,
                            total_latency_ms=total_latency_ms,
                            pci_latency_ms=pci_latency_ms,
                            pat_latency_ms=pat_latency_ms,
                            receipt=receipt,
                            envelope_digest=envelope_digest,
                        )

                logger.info(
                    f"PAT Enforcement PASSED: {len(pat_gate_results)} gates, "
                    f"SNR={pat_result.final_snr:.4f}, "
                    f"latency={pat_latency_ms:.2f}ms"
                )

            # =================================================================
            # STAGE 3: SNR ENFORCEMENT
            # =================================================================
            if not context.skip_snr_enforcement:
                logger.info("Stage 3: SNR Enforcement")
                snr_start = time.perf_counter()

                # Determine operation type from envelope
                operation_type = self._infer_operation_type(envelope)

                # Create enforcement context
                snr_context = EnforcementContext(
                    operation_type=operation_type,
                    agent_id=envelope.sender.agent_id,
                    snr_score=envelope.metadata.snr_score,
                    task_id=context.task_id,
                    session_id=context.session_id,
                    details={
                        "action": envelope.payload.action,
                        "envelope_digest": envelope_digest,
                        "agent_type": envelope.sender.agent_type.value,
                    },
                )

                # Enforce SNR threshold
                snr_result = await self.snr_enforcer.enforce_async(snr_context)

                snr_latency_ms = (time.perf_counter() - snr_start) * 1000

                if not snr_result.passed:
                    # FAIL-FAST: SNR below threshold
                    self._rejections += 1

                    rejection = RejectionResponse.rejection(
                        code=RejectCode.REJECT_SNR_BELOW_MIN,
                        message=snr_result.message,
                        envelope_digest=envelope_digest,
                        timestamp=timestamp,
                    )

                    logger.warning(
                        f"SNR Enforcement REJECTED: "
                        f"SNR={snr_result.snr_score:.4f} < threshold {snr_result.threshold:.4f}"
                    )

                    # Emit rejection receipt (SNR enforcer already emits, but we add to validation receipt)
                    if self.emit_receipts:
                        receipt = await self._emit_rejection_receipt(
                            envelope_digest=envelope_digest,
                            timestamp=timestamp,
                            rejection=rejection,
                            context=context,
                            stage="SNR_ENFORCEMENT",
                            gate_results=pci_gate_results,
                            snr_result=snr_result,
                        )

                    total_latency_ms = (time.perf_counter() - start_time) * 1000

                    return ValidationResult(
                        passed=False,
                        pci_gate_results=pci_gate_results,
                        pat_gate_results=pat_gate_results,
                        snr_result=snr_result,
                        rejection=rejection,
                        total_latency_ms=total_latency_ms,
                        pci_latency_ms=pci_latency_ms,
                        pat_latency_ms=pat_latency_ms,
                        snr_latency_ms=snr_latency_ms,
                        receipt=receipt,
                        envelope_digest=envelope_digest,
                    )

                logger.info(
                    f"SNR Enforcement PASSED: "
                    f"SNR={snr_result.snr_score:.4f} >= threshold {snr_result.threshold:.4f}, "
                    f"latency={snr_latency_ms:.2f}ms"
                )
            else:
                # Create pass-through SNR result
                snr_result = EnforcementResult(
                    passed=True,
                    snr_score=envelope.metadata.snr_score,
                    threshold=context.snr_threshold or self.snr_threshold,
                    message="SNR enforcement skipped per context",
                )

            # =================================================================
            # ALL STAGES PASSED
            # =================================================================
            total_latency_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"Validation pipeline PASSED: "
                f"total_latency={total_latency_ms:.2f}ms "
                f"(PCI={pci_latency_ms:.2f}ms, PAT={pat_latency_ms:.2f}ms, SNR={snr_latency_ms:.2f}ms)"
            )

            return ValidationResult(
                passed=True,
                pci_gate_results=pci_gate_results,
                pat_gate_results=pat_gate_results,
                snr_result=snr_result,
                rejection=None,
                total_latency_ms=total_latency_ms,
                pci_latency_ms=pci_latency_ms,
                pat_latency_ms=pat_latency_ms,
                snr_latency_ms=snr_latency_ms,
                receipt=None,
                envelope_digest=envelope_digest,
            )

        except Exception as e:
            # FAIL-CLOSED: Any error terminates pipeline
            self._rejections += 1

            logger.error(f"Validation pipeline error (fail-closed): {e}", exc_info=True)

            rejection = RejectionResponse.rejection(
                code=RejectCode.REJECT_INTERNAL_ERROR,
                message=f"Validation pipeline error (fail-closed): {str(e)}",
                envelope_digest=envelope_digest,
                timestamp=timestamp,
            )

            # Emit error receipt
            if self.emit_receipts:
                receipt = await self._emit_rejection_receipt(
                    envelope_digest=envelope_digest,
                    timestamp=timestamp,
                    rejection=rejection,
                    context=context,
                    stage="INTERNAL_ERROR",
                    gate_results=pci_gate_results,
                    error=str(e),
                )

            total_latency_ms = (time.perf_counter() - start_time) * 1000

            return ValidationResult(
                passed=False,
                pci_gate_results=pci_gate_results,
                pat_gate_results=pat_gate_results,
                snr_result=snr_result
                or EnforcementResult(
                    passed=False,
                    snr_score=0.0,
                    threshold=self.snr_threshold,
                ),
                rejection=rejection,
                total_latency_ms=total_latency_ms,
                pci_latency_ms=pci_latency_ms,
                pat_latency_ms=pat_latency_ms,
                snr_latency_ms=snr_latency_ms,
                receipt=receipt,
                envelope_digest=envelope_digest,
            )

    def _infer_operation_type(self, envelope: PCIEnvelope) -> OperationType:
        """Infer operation type from envelope action."""
        action = envelope.payload.action.lower()

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

    async def _emit_rejection_receipt(
        self,
        envelope_digest: str,
        timestamp: str,
        rejection: RejectionResponse,
        context: ValidationContext,
        stage: str,
        gate_results: List[GateResult],
        pat_result: Optional[PATEnforcementResult] = None,
        snr_result: Optional[EnforcementResult] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Emit rejection receipt to file system.

        Args:
            envelope_digest: Envelope BLAKE3 digest
            timestamp: ISO8601 timestamp
            rejection: Rejection response
            context: Validation context
            stage: Which stage failed
            gate_results: PCI gate results
            pat_result: PAT enforcement result (if available)
            snr_result: SNR enforcement result (if available)
            error: Error message (for internal errors)

        Returns:
            Receipt dictionary
        """
        try:
            receipt_id = self._generate_receipt_id(envelope_digest, timestamp)

            receipt = {
                "receipt_type": "VALIDATION_REJECTION",
                "version": "1.0",
                "receipt_id": receipt_id,
                "timestamp": timestamp,
                "envelope_digest": envelope_digest,
                "failed_stage": stage,
                "rejection": rejection.to_dict(),
                "context": context.to_dict(),
                "pci_gate_results": [
                    {
                        "gate": r.gate.value,
                        "passed": r.passed,
                        "latency_ms": r.latency_ms,
                    }
                    for r in gate_results
                ],
                "pat_result": pat_result.to_dict() if pat_result else None,
                "snr_result": snr_result.to_dict() if snr_result else None,
                "error": error,
                "integrity_hash": self._compute_integrity_hash(
                    receipt_id, envelope_digest, timestamp, stage
                ),
            }

            # Write to JSONL file
            receipt_file = (
                self.receipt_dir
                / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
            )

            with open(receipt_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(receipt) + "\n")

            self._receipts_emitted += 1

            logger.info(f"Emitted validation rejection receipt: {receipt_id}")

            return receipt

        except Exception as e:
            logger.error(f"Failed to emit rejection receipt: {e}", exc_info=True)
            return {}

    def _generate_receipt_id(self, envelope_digest: str, timestamp: str) -> str:
        """Generate unique receipt ID."""
        data = f"validation:{envelope_digest}:{timestamp}"
        return f"val-reject-{hashlib.sha256(data.encode()).hexdigest()[:16]}"

    def _compute_integrity_hash(
        self,
        receipt_id: str,
        envelope_digest: str,
        timestamp: str,
        stage: str,
    ) -> str:
        """Compute SHA-256 integrity hash for receipt."""
        data = {
            "receipt_id": receipt_id,
            "envelope_digest": envelope_digest,
            "timestamp": timestamp,
            "failed_stage": stage,
        }
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "validations": self._validations,
            "rejections": self._rejections,
            "rejection_rate": self._rejections / max(1, self._validations),
            "receipts_emitted": self._receipts_emitted,
            "snr_enforcer_stats": self.snr_enforcer.get_statistics(),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global pipeline instance
_global_pipeline: Optional[ValidationPipeline] = None


def get_validation_pipeline(
    constitution_path: str = "constitution/pat_enforcement_v1.yaml",
    force_reload: bool = False,
) -> ValidationPipeline:
    """
    Get global validation pipeline instance (singleton).

    Args:
        constitution_path: Path to constitution
        force_reload: Force reload of pipeline

    Returns:
        ValidationPipeline instance
    """
    global _global_pipeline

    if _global_pipeline is None or force_reload:
        _global_pipeline = ValidationPipeline(constitution_path=constitution_path)

    return _global_pipeline


async def validate_envelope(
    envelope: PCIEnvelope,
    policy_hash: str,
    state_hash: str,
    require_expensive: bool = False,
    require_pat: bool = False,
    pat_request: Optional[PATRequest] = None,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> ValidationResult:
    """
    Convenience function for envelope validation.

    Args:
        envelope: PCI envelope to validate
        policy_hash: Current constitution hash
        state_hash: Current state hash
        require_expensive: Whether to run EXPENSIVE tier gates
        require_pat: Whether to run PAT enforcement
        pat_request: PAT request data (required if require_pat=True)
        session_id: Optional session ID
        task_id: Optional task ID

    Returns:
        ValidationResult
    """
    pipeline = get_validation_pipeline()

    context = ValidationContext(
        require_expensive_gates=require_expensive,
        require_pat_enforcement=require_pat,
        pat_request=pat_request,
        policy_hash=policy_hash,
        state_hash=state_hash,
        session_id=session_id,
        task_id=task_id,
    )

    return await pipeline.validate(envelope, context)


# =============================================================================
# MAIN & TESTING
# =============================================================================


async def main():
    """Example usage."""
    from core.pci.envelope import EnvelopeBuilder
    from core.pci.crypto import generate_keypair
    from core.pci.types import AgentType

    # Initialize pipeline
    pipeline = ValidationPipeline()

    # Generate test keypair
    keypair = generate_keypair()

    # Create test envelope
    envelope = (
        EnvelopeBuilder()
        .with_sender(AgentType.PAT, "pat-test-001", keypair.public_key_hex)
        .with_action("analyze", {"task": "test_validation"})
        .with_policy("test_policy_hash_" + "0" * 48)
        .with_state("test_state_hash_" + "0" * 48)
        .with_scores(ihsan=0.97, snr=0.85)
        .build()
        .sign(keypair.private_key)
    )

    # Create validation context
    context = ValidationContext(
        require_expensive_gates=False,
        require_pat_enforcement=False,
        policy_hash="test_policy_hash_" + "0" * 48,
        state_hash="test_state_hash_" + "0" * 48,
        session_id="test_session_001",
        task_id="test_task_001",
    )

    # Execute validation
    result = await pipeline.validate(envelope, context)

    # Print results
    print("\n" + "=" * 80)
    print("VALIDATION PIPELINE RESULT")
    print("=" * 80)
    print(f"Passed: {result.passed}")
    print(f"Total Latency: {result.total_latency_ms:.2f}ms")
    print(f"  PCI Latency: {result.pci_latency_ms:.2f}ms")
    print(f"  PAT Latency: {result.pat_latency_ms:.2f}ms")
    print(f"  SNR Latency: {result.snr_latency_ms:.2f}ms")
    print(f"Gates Passed: {result.gates_passed}")

    if result.rejection:
        print(f"\nRejection: {result.rejection.code.name}")
        print(f"Message: {result.rejection.message}")

    print(
        f"\nSNR Result: passed={result.snr_result.passed}, "
        f"score={result.snr_result.snr_score:.4f}, "
        f"threshold={result.snr_result.threshold:.4f}"
    )

    print("\nPipeline Statistics:")
    print(json.dumps(pipeline.get_statistics(), indent=2))
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
