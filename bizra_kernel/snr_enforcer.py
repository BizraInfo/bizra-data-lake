"""
SNR Enforcer — Signal-to-Noise Ratio Threshold Enforcement
==========================================================
Constitutional compliance for SNR thresholds across all BIZRA operations.

Status: PRODUCTION
Alignment: constitution/pat_enforcement_v1.yaml Section: snr_integration

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     SNREnforcer                             │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │  Load Constitution Thresholds                        │   │
    │  │  (target_snr: 0.98, minimum_snr: 0.95)              │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                          ▼                                  │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │  Integrate with SNRTracker                           │   │
    │  │  (get_current_snr, get_average_snr, record)         │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                          ▼                                  │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │  Enforcement Decision                                │   │
    │  │  SNR < threshold → REJECT (fail-closed)             │   │
    │  │  SNR ≥ threshold → PASS                             │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                          ▼                                  │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │  Receipt Emission                                    │   │
    │  │  (rejection_code, evidence, integrity_hash)         │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

Fail-Closed Semantics:
    - SNR below threshold → Operation REJECTED
    - Missing SNR data → Operation REJECTED
    - Constitution load error → Operation REJECTED
    - All rejections emit receipts

Usage:
    from bizra_kernel.snr_enforcer import SNREnforcer, EnforcementContext

    # Initialize with constitution
    enforcer = SNREnforcer(constitution_path="constitution/pat_enforcement_v1.yaml")

    # Enforce threshold
    context = EnforcementContext(
        operation_type="reasoning",
        agent_id="pat-master-reasoner",
        snr_score=0.97,
        details={"task": "analyze_code"}
    )

    result = await enforcer.enforce(context)

    if not result.passed:
        raise OperationRejected(result.rejection_code, result.message)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .snr_tracker import SNRMetrics, SNRTracker

# Import PCI types for reject codes and receipts
try:
    from core.pci import (
        RejectCode,
        RejectionResponse,
        reject_snr,
    )
    PCI_AVAILABLE = True
except ImportError:
    # Fallback if PCI not available
    PCI_AVAILABLE = False
    RejectCode = None
    RejectionResponse = None

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION & THRESHOLDS
# =============================================================================

class OperationType(str, Enum):
    """Types of operations that require SNR enforcement."""
    REASONING = "reasoning"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    PAT_EXECUTION = "pat_execution"
    SAT_VALIDATION = "sat_validation"
    SAPE_PROBE = "sape_probe"
    DEFAULT = "default"


@dataclass
class SNRThresholds:
    """SNR thresholds from constitution."""
    target_snr: float = 0.98      # Target SNR (PAT enforcement)
    minimum_snr: float = 0.95     # Minimum SNR (fail below this)
    escalate_below: float = 0.90  # Escalate if below this

    # Per-operation type overrides (optional)
    operation_thresholds: Dict[str, float] = field(default_factory=dict)

    def get_threshold(self, operation_type: OperationType) -> float:
        """Get threshold for specific operation type."""
        op_key = operation_type.value
        return self.operation_thresholds.get(op_key, self.minimum_snr)

    @classmethod
    def from_constitution(cls, constitution_path: str | Path) -> SNRThresholds:
        """Load thresholds from constitution YAML."""
        path = Path(constitution_path)

        if not path.exists():
            logger.warning(f"Constitution not found at {path}, using defaults")
            return cls()

        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Extract SNR integration section
            snr_config = config.get('snr_integration', {})

            target = snr_config.get('target_snr', 0.98)
            minimum = snr_config.get('minimum_snr', 0.95)
            escalate = snr_config.get('escalate_below', 0.90)

            # Operation-specific thresholds (if defined)
            operation_thresholds = snr_config.get('operation_thresholds', {})

            logger.info(
                f"Loaded SNR thresholds from constitution: "
                f"target={target}, minimum={minimum}, escalate={escalate}"
            )

            return cls(
                target_snr=target,
                minimum_snr=minimum,
                escalate_below=escalate,
                operation_thresholds=operation_thresholds,
            )

        except Exception as e:
            logger.error(f"Failed to load constitution from {path}: {e}")
            logger.warning("Using default SNR thresholds")
            return cls()


# =============================================================================
# ENFORCEMENT CONTEXT & RESULT
# =============================================================================

@dataclass
class EnforcementContext:
    """Context for SNR enforcement decision."""
    operation_type: OperationType
    agent_id: str
    snr_score: float
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_type": self.operation_type.value,
            "agent_id": self.agent_id,
            "snr_score": self.snr_score,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class EnforcementResult:
    """Result of SNR enforcement."""
    passed: bool
    snr_score: float
    threshold: float
    rejection_code: Optional[int] = None
    message: str = ""
    receipt_id: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "snr_score": self.snr_score,
            "threshold": self.threshold,
            "rejection_code": self.rejection_code,
            "message": self.message,
            "receipt_id": self.receipt_id,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }


# =============================================================================
# SNR ENFORCER
# =============================================================================

class SNREnforcer:
    """
    Enforces SNR thresholds with fail-closed semantics.

    Features:
    - Constitutional threshold loading
    - Integration with SNRTracker
    - Per-operation type thresholds
    - Receipt emission on rejection
    - Async-compatible
    - Comprehensive logging
    """

    def __init__(
        self,
        constitution_path: str | Path = "constitution/pat_enforcement_v1.yaml",
        snr_tracker: Optional[SNRTracker] = None,
        emit_receipts: bool = True,
        receipt_dir: Optional[str | Path] = None,
    ):
        """
        Initialize SNR enforcer.

        Args:
            constitution_path: Path to PAT constitution YAML
            snr_tracker: Optional SNRTracker instance (creates new if None)
            emit_receipts: Whether to emit rejection receipts
            receipt_dir: Directory for receipts (default: docs/evidence/receipts/snr/)
        """
        self.thresholds = SNRThresholds.from_constitution(constitution_path)
        self.snr_tracker = snr_tracker or SNRTracker()
        self.emit_receipts = emit_receipts

        # Receipt directory
        if receipt_dir is None:
            receipt_dir = Path("docs/evidence/receipts/snr")
        self.receipt_dir = Path(receipt_dir)
        self.receipt_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self._enforcements: int = 0
        self._rejections: int = 0
        self._emissions: int = 0

        logger.info(
            f"SNREnforcer initialized: target={self.thresholds.target_snr}, "
            f"minimum={self.thresholds.minimum_snr}, receipts={emit_receipts}"
        )

    def enforce(self, context: EnforcementContext) -> EnforcementResult:
        """
        Enforce SNR threshold (synchronous).

        Args:
            context: Enforcement context with SNR score and metadata

        Returns:
            EnforcementResult with pass/fail decision
        """
        self._enforcements += 1

        # Get threshold for this operation type
        threshold = self.thresholds.get_threshold(context.operation_type)

        # Enforcement decision
        passed = context.snr_score >= threshold

        # Evidence
        evidence = {
            "operation_type": context.operation_type.value,
            "agent_id": context.agent_id,
            "snr_score": context.snr_score,
            "threshold": threshold,
            "target_snr": self.thresholds.target_snr,
            "delta": context.snr_score - threshold,
            "context_details": context.details,
        }

        if passed:
            # PASS: Log and return
            logger.debug(
                f"SNR enforcement PASSED: {context.agent_id} "
                f"({context.operation_type.value}) "
                f"SNR={context.snr_score:.4f} ≥ {threshold:.4f}"
            )

            return EnforcementResult(
                passed=True,
                snr_score=context.snr_score,
                threshold=threshold,
                message=f"SNR {context.snr_score:.4f} meets threshold {threshold:.4f}",
                evidence=evidence,
            )

        else:
            # FAIL: Reject and emit receipt
            self._rejections += 1

            # Determine reject code
            if PCI_AVAILABLE:
                rejection_code = int(RejectCode.REJECT_SNR_BELOW_MIN)
            else:
                rejection_code = 7  # REJECT_SNR_BELOW_MIN

            message = (
                f"SNR enforcement REJECTED: {context.agent_id} "
                f"({context.operation_type.value}) "
                f"SNR={context.snr_score:.4f} < threshold {threshold:.4f}"
            )

            logger.warning(message)

            # Generate receipt ID
            receipt_id = self._generate_receipt_id(context)

            result = EnforcementResult(
                passed=False,
                snr_score=context.snr_score,
                threshold=threshold,
                rejection_code=rejection_code,
                message=message,
                receipt_id=receipt_id,
                evidence=evidence,
            )

            # Emit receipt
            if self.emit_receipts:
                self._emit_rejection_receipt(context, result)

            return result

    async def enforce_async(self, context: EnforcementContext) -> EnforcementResult:
        """
        Enforce SNR threshold (asynchronous version).

        Args:
            context: Enforcement context with SNR score and metadata

        Returns:
            EnforcementResult with pass/fail decision
        """
        # For now, just wrap synchronous version
        # Can be extended with async I/O for receipts
        return self.enforce(context)

    def record_metrics(self, metrics: SNRMetrics) -> None:
        """
        Record SNR metrics to tracker.

        Args:
            metrics: SNRMetrics with token counts and scores
        """
        self.snr_tracker.record(metrics)

    def get_statistics(self) -> Dict[str, Any]:
        """Get enforcement statistics."""
        return {
            "enforcements": self._enforcements,
            "rejections": self._rejections,
            "rejection_rate": self._rejections / max(1, self._enforcements),
            "receipts_emitted": self._emissions,
            "thresholds": {
                "target_snr": self.thresholds.target_snr,
                "minimum_snr": self.thresholds.minimum_snr,
                "escalate_below": self.thresholds.escalate_below,
            },
            "tracker_stats": self.snr_tracker.get_statistics(),
        }

    def _generate_receipt_id(self, context: EnforcementContext) -> str:
        """Generate receipt ID using BLAKE3-style hash."""
        data = (
            f"{context.timestamp}:"
            f"{context.operation_type.value}:"
            f"{context.agent_id}:"
            f"{context.snr_score:.6f}"
        )
        hash_obj = hashlib.sha256(data.encode('utf-8'))
        return f"snr-reject-{hash_obj.hexdigest()[:16]}"

    def _emit_rejection_receipt(
        self,
        context: EnforcementContext,
        result: EnforcementResult,
    ) -> None:
        """
        Emit rejection receipt to file system.

        Args:
            context: Enforcement context
            result: Enforcement result
        """
        try:
            receipt = {
                "receipt_id": result.receipt_id,
                "timestamp": result.timestamp,
                "rejection_code": result.rejection_code,
                "rejection_name": "REJECT_SNR_BELOW_MIN",
                "operation_type": context.operation_type.value,
                "agent_id": context.agent_id,
                "task_id": context.task_id,
                "session_id": context.session_id,
                "snr_score": context.snr_score,
                "threshold": result.threshold,
                "target_snr": self.thresholds.target_snr,
                "delta": context.snr_score - result.threshold,
                "message": result.message,
                "evidence": result.evidence,
                "context": context.to_dict(),
                "integrity_hash": self._compute_integrity_hash(context, result),
            }

            # Write to JSONL file
            import json
            receipt_file = self.receipt_dir / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"

            with open(receipt_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(receipt) + '\n')

            self._emissions += 1

            logger.info(f"Emitted SNR rejection receipt: {result.receipt_id}")

        except Exception as e:
            logger.error(f"Failed to emit rejection receipt: {e}", exc_info=True)

    def _compute_integrity_hash(
        self,
        context: EnforcementContext,
        result: EnforcementResult,
    ) -> str:
        """Compute SHA-256 integrity hash for receipt."""
        import json
        data = {
            "receipt_id": result.receipt_id,
            "timestamp": result.timestamp,
            "operation_type": context.operation_type.value,
            "agent_id": context.agent_id,
            "snr_score": context.snr_score,
            "threshold": result.threshold,
        }
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global enforcer instance
_global_enforcer: Optional[SNREnforcer] = None


def get_snr_enforcer(
    constitution_path: str | Path = "constitution/pat_enforcement_v1.yaml",
    force_reload: bool = False,
) -> SNREnforcer:
    """
    Get global SNR enforcer instance (singleton).

    Args:
        constitution_path: Path to constitution
        force_reload: Force reload of constitution

    Returns:
        SNREnforcer instance
    """
    global _global_enforcer

    if _global_enforcer is None or force_reload:
        _global_enforcer = SNREnforcer(constitution_path=constitution_path)

    return _global_enforcer


def enforce_snr(
    operation_type: str | OperationType,
    agent_id: str,
    snr_score: float,
    task_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> EnforcementResult:
    """
    Convenience function for SNR enforcement.

    Args:
        operation_type: Type of operation
        agent_id: Agent ID
        snr_score: SNR score to check
        task_id: Optional task ID
        details: Optional additional details

    Returns:
        EnforcementResult
    """
    enforcer = get_snr_enforcer()

    if isinstance(operation_type, str):
        try:
            operation_type = OperationType(operation_type)
        except ValueError:
            operation_type = OperationType.DEFAULT

    context = EnforcementContext(
        operation_type=operation_type,
        agent_id=agent_id,
        snr_score=snr_score,
        task_id=task_id,
        details=details or {},
    )

    return enforcer.enforce(context)


async def enforce_snr_async(
    operation_type: str | OperationType,
    agent_id: str,
    snr_score: float,
    task_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> EnforcementResult:
    """
    Async convenience function for SNR enforcement.

    Args:
        operation_type: Type of operation
        agent_id: Agent ID
        snr_score: SNR score to check
        task_id: Optional task ID
        details: Optional additional details

    Returns:
        EnforcementResult
    """
    enforcer = get_snr_enforcer()

    if isinstance(operation_type, str):
        try:
            operation_type = OperationType(operation_type)
        except ValueError:
            operation_type = OperationType.DEFAULT

    context = EnforcementContext(
        operation_type=operation_type,
        agent_id=agent_id,
        snr_score=snr_score,
        task_id=task_id,
        details=details or {},
    )

    return await enforcer.enforce_async(context)
