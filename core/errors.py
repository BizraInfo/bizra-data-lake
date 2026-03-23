"""Typed constitutional error taxonomy for runtime boundaries.

The goal of this module is narrow and operational:
- preserve fail-closed semantics
- turn boundary failures into receipts
- keep constitutional boundaries explicit
- let API/runtime edges map typed failures consistently
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import time
import traceback
from typing import Any


class Severity(str, Enum):
    """How the system should respond to a boundary failure."""

    HALT = "HALT"
    REJECT = "REJECT"
    DEGRADE = "DEGRADE"
    RETRY = "RETRY"
    LOG = "LOG"


class Boundary(str, Enum):
    """Constitutional or operational boundary crossed by an error."""

    IHSAN = "I-1_IHSAN_FLOOR"
    RIBA = "I-2_RIBA_ZERO"
    ADL = "I-3_ADL_LIMIT"
    ZANN = "I-4_ZANN_ZERO"
    FROZEN = "I-5_FROZEN_AGENTS"
    SOVEREIGNTY = "I-6_SOVEREIGNTY"
    SPINE = "I-7_SPINE_GUARD"
    AUTHORITY = "AUTHORITY"
    CHAIN = "CHAIN_INTEGRITY"
    MEMBRANE = "MEMBRANE"
    BRIDGE = "BRIDGE"
    INFERENCE = "INFERENCE"
    RESOURCE = "RESOURCE"


@dataclass
class ErrorReceipt:
    """Auditable representation of a boundary failure."""

    error_type: str
    severity: str
    boundary: str
    message: str
    timestamp: float = 0.0
    trace: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()


class BizraError(Exception):
    """Base typed error for constitutional and membrane failures."""

    severity: Severity = Severity.LOG
    boundary: Boundary = Boundary.MEMBRANE

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.context = context or {}
        self.original = original
        self.timestamp = time.time()

    def to_receipt(self) -> dict[str, Any]:
        """Return a machine-friendly receipt for this failure."""

        trace = ""
        if self.original is not None and self.original.__traceback__ is not None:
            trace = "".join(
                traceback.format_exception(
                    type(self.original),
                    self.original,
                    self.original.__traceback__,
                )
            )

        return asdict(
            ErrorReceipt(
                error_type=type(self).__name__,
                severity=self.severity.value,
                boundary=self.boundary.value,
                message=str(self),
                timestamp=self.timestamp,
                trace=trace,
                context=self.context,
            )
        )

    def __str__(self) -> str:
        return f"[{self.severity.value}:{self.boundary.value}] {super().__str__()}"


class MembraneError(BizraError):
    """Generic membrane failure when no sharper taxonomy is available."""

    severity = Severity.DEGRADE
    boundary = Boundary.MEMBRANE


class ConstitutionalViolation(BizraError):
    """Hard constitutional invariant breach."""

    severity = Severity.HALT

    def __init__(
        self,
        invariant: Boundary,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message, context=context, original=original)
        self.boundary = invariant
        self.invariant = invariant


class IhsanViolation(ConstitutionalViolation):
    """Ihsan floor violation."""

    def __init__(
        self,
        score: float,
        threshold: float = 0.95,
        *,
        context: dict[str, Any] | None = None,
        original: Exception | None = None,
    ) -> None:
        payload = {"score": score, "threshold": threshold}
        if context:
            payload.update(context)
        super().__init__(
            Boundary.IHSAN,
            f"Ihsan {score:.4f} < {threshold:.4f} floor",
            context=payload,
            original=original,
        )


class GateRejection(BizraError):
    """Constitutional or policy gate rejection."""

    severity = Severity.REJECT
    boundary = Boundary.MEMBRANE

    def __init__(
        self,
        gate: str,
        reason: str,
        *,
        score: float | None = None,
        context: dict[str, Any] | None = None,
        original: Exception | None = None,
    ) -> None:
        payload = {"gate": gate, "reason": reason, "score": score}
        if context:
            payload.update(context)
        detail = f"Gate '{gate}' rejected: {reason}"
        if score is not None:
            detail += f" (score={score:.4f})"
        super().__init__(detail, context=payload, original=original)
        self.gate = gate
        self.reason = reason
        self.score = score


class AuthorityError(BizraError):
    """Missing or invalid execution authority."""

    severity = Severity.REJECT
    boundary = Boundary.AUTHORITY


class MissingAuthority(AuthorityError):
    """No authority was provided."""

    def __init__(
        self,
        *,
        context: dict[str, Any] | None = None,
        original: Exception | None = None,
    ) -> None:
        super().__init__(
            "Execution authority missing", context=context, original=original
        )


class ReceiptChainError(BizraError):
    """Receipt chain integrity failure."""

    severity = Severity.HALT
    boundary = Boundary.CHAIN

    def __init__(
        self,
        index: int,
        expected_hash: str,
        actual_hash: str,
        *,
        context: dict[str, Any] | None = None,
        original: Exception | None = None,
    ) -> None:
        payload = {
            "index": index,
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
        }
        if context:
            payload.update(context)
        super().__init__(
            (
                f"Chain tamper at index {index}: expected {expected_hash[:8]}, "
                f"got {actual_hash[:8]}"
            ),
            context=payload,
            original=original,
        )


class BridgeError(BizraError):
    """Cross-component or cross-language bridge failure."""

    severity = Severity.DEGRADE
    boundary = Boundary.BRIDGE

    def __init__(
        self,
        bridge_name: str,
        detail: str,
        *,
        context: dict[str, Any] | None = None,
        original: Exception | None = None,
    ) -> None:
        payload = {"bridge": bridge_name, "detail": detail}
        if context:
            payload.update(context)
        super().__init__(
            f"Bridge '{bridge_name}' failed: {detail}",
            context=payload,
            original=original,
        )
        self.bridge_name = bridge_name


class InferenceError(BizraError):
    """Inference or reasoning backend failure."""

    severity = Severity.RETRY
    boundary = Boundary.INFERENCE

    def __init__(
        self,
        model: str,
        detail: str,
        *,
        context: dict[str, Any] | None = None,
        original: Exception | None = None,
    ) -> None:
        payload = {"model": model, "detail": detail}
        if context:
            payload.update(context)
        super().__init__(
            f"Inference failed on '{model}': {detail}",
            context=payload,
            original=original,
        )
        self.model = model


class ResourceError(BizraError):
    """Resource exhaustion or resource-health failure."""

    severity = Severity.DEGRADE
    boundary = Boundary.RESOURCE


def wrap_legacy_exception(
    exc: Exception,
    boundary: Boundary = Boundary.MEMBRANE,
    *,
    context: dict[str, Any] | None = None,
) -> BizraError:
    """Wrap an untyped legacy exception in the closest available taxonomy."""

    if isinstance(exc, BizraError):
        return exc
    if boundary == Boundary.BRIDGE:
        return BridgeError("legacy", str(exc), context=context, original=exc)
    if boundary == Boundary.INFERENCE:
        return InferenceError("legacy", str(exc), context=context, original=exc)
    if boundary == Boundary.AUTHORITY:
        return AuthorityError(str(exc), context=context, original=exc)
    if boundary == Boundary.RESOURCE:
        return ResourceError(str(exc), context=context, original=exc)
    return MembraneError(str(exc), context=context, original=exc)


def http_status_for_error(exc: BizraError) -> int:
    """Map typed errors to stable HTTP status codes."""

    if isinstance(exc, ConstitutionalViolation):
        return 403
    if isinstance(exc, GateRejection):
        return 422
    if isinstance(exc, AuthorityError):
        return 401
    if isinstance(exc, BridgeError):
        return 503
    if isinstance(exc, InferenceError):
        return 502
    if isinstance(exc, ResourceError):
        return 503
    return 500


__all__ = [
    "AuthorityError",
    "Boundary",
    "BizraError",
    "BridgeError",
    "ConstitutionalViolation",
    "ErrorReceipt",
    "GateRejection",
    "http_status_for_error",
    "IhsanViolation",
    "InferenceError",
    "MembraneError",
    "MissingAuthority",
    "ReceiptChainError",
    "ResourceError",
    "Severity",
    "wrap_legacy_exception",
]
