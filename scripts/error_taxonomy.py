"""
BIZRA Constitutional Error Taxonomy
════════════════════════════════════

Audit finding: api.py:991 and mission_nervous_system.py:374 use
broad `except Exception`. This collapses all failures into one
bucket, making degradation truthful but not observable.

This module defines typed error hierarchies that:
1. Preserve fail-closed semantics (no silent fallback)
2. Make errors into receipts (observable, auditable)
3. Map each error to its constitutional boundary
4. Enable typed catch blocks that know what failed and why

Usage:
    from core.errors import (
        BizraError, ConstitutionalViolation, GateRejection,
        AuthorityError, ReceiptChainError, MembraneError,
        BridgeError, InferenceError
    )

    try:
        result = execute_mission(task)
    except GateRejection as e:
        # Constitutional gate rejected — receipt the rejection
        log_rejection_receipt(e.invariant, e.score, e.threshold)
    except BridgeError as e:
        # Rust bridge failed — degrade to Python fallback
        log_bridge_failure(e.bridge_name, e.original)
    except BizraError as e:
        # Any BIZRA error — receipt it, don't swallow it
        log_error_receipt(e)

Drop-in replacement for `except Exception` blocks:
    # BEFORE (broad catch, epistemic loss):
    except Exception as e:
        logger.error(f"Mission failed: {e}")
        return {"error": str(e)}

    # AFTER (typed catch, error becomes receipt):
    except GateRejection as e:
        return e.to_receipt()
    except BridgeError as e:
        return e.to_receipt()
    except InferenceError as e:
        return e.to_receipt()
    except BizraError as e:
        return e.to_receipt()

Created: 2026-03-23 | BIZRA Error Taxonomy v1.0
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

# ═══════════════════════════════════════════════════════════════
# ERROR SEVERITY — Maps to constitutional response
# ═══════════════════════════════════════════════════════════════


class Severity(str, Enum):
    """How the system should respond to this error."""

    HALT = "HALT"  # Constitutional violation — full stop, M0 state
    REJECT = "REJECT"  # Gate rejection — receipted, mission aborted
    DEGRADE = "DEGRADE"  # Component failure — fallback path, receipted
    RETRY = "RETRY"  # Transient failure — retry with backoff
    LOG = "LOG"  # Informational — receipt but continue


class Boundary(str, Enum):
    """Which constitutional boundary was crossed."""

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


# ═══════════════════════════════════════════════════════════════
# ERROR RECEIPT — Every error becomes an auditable artifact
# ═══════════════════════════════════════════════════════════════


@dataclass
class ErrorReceipt:
    """Every error produces a receipt. Errors are evidence, not log lines."""

    error_type: str
    severity: str
    boundary: str
    message: str
    timestamp: float = 0.0
    trace: str = ""
    context: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


# ═══════════════════════════════════════════════════════════════
# BASE ERROR — All BIZRA errors inherit from this
# ═══════════════════════════════════════════════════════════════


class BizraError(Exception):
    """
    Base error for all BIZRA typed exceptions.

    Every BizraError:
    - Has a severity (HALT/REJECT/DEGRADE/RETRY/LOG)
    - Has a boundary (which constitutional boundary was crossed)
    - Can produce a receipt (error as evidence)
    - Preserves the original exception chain
    """

    severity: Severity = Severity.LOG
    boundary: Boundary = Boundary.MEMBRANE

    def __init__(
        self,
        message: str,
        *,
        context: dict | None = None,
        original: Exception | None = None,
    ):
        super().__init__(message)
        self.context = context or {}
        self.original = original
        self.timestamp = time.time()

    def to_receipt(self) -> dict:
        """Convert this error into an auditable receipt."""
        receipt = ErrorReceipt(
            error_type=type(self).__name__,
            severity=self.severity.value,
            boundary=self.boundary.value,
            message=str(self),
            timestamp=self.timestamp,
            trace=traceback.format_exc() if self.original else "",
            context=self.context,
        )
        return asdict(receipt)

    def __str__(self):
        base = super().__str__()
        return f"[{self.severity.value}:{self.boundary.value}] {base}"


# ═══════════════════════════════════════════════════════════════
# CONSTITUTIONAL VIOLATIONS — Severity: HALT
# ═══════════════════════════════════════════════════════════════


class ConstitutionalViolation(BizraError):
    """
    A constitutional invariant was violated.
    Triggers M0 (constitutional halt) in HHMM.
    No execution proceeds until violation is resolved.
    """

    severity = Severity.HALT

    def __init__(self, invariant: Boundary, message: str, **kwargs):
        self.invariant = invariant
        super().__init__(message, **kwargs)
        self.boundary = invariant


class IhsanViolation(ConstitutionalViolation):
    """I-1: Ihsan score below constitutional floor."""

    def __init__(self, score: float, threshold: float = 0.95, **kwargs):
        self.score = score
        self.threshold = threshold
        super().__init__(
            Boundary.IHSAN,
            f"Ihsan {score:.4f} < {threshold} floor",
            context={"score": score, "threshold": threshold},
            **kwargs,
        )


class RibaViolation(ConstitutionalViolation):
    """I-2: Interest detected in transaction."""

    def __init__(self, amount: float, **kwargs):
        super().__init__(
            Boundary.RIBA,
            f"Interest amount {amount} detected — RIBA_ZERO violated",
            context={"amount": amount},
            **kwargs,
        )


class AdlViolation(ConstitutionalViolation):
    """I-3: Gini coefficient exceeds constitutional limit."""

    def __init__(self, gini: float, limit: float = 0.35, **kwargs):
        self.gini = gini
        super().__init__(
            Boundary.ADL,
            f"Gini {gini:.4f} > {limit} limit — ADL violated",
            context={"gini": gini, "limit": limit},
            **kwargs,
        )


class ZannViolation(ConstitutionalViolation):
    """I-4: Claim without evidence artifact."""

    def __init__(self, claim: str, **kwargs):
        super().__init__(
            Boundary.ZANN,
            f"Claim '{claim}' has no evidence artifact — ZANN_ZERO violated",
            context={"claim": claim},
            **kwargs,
        )


class FrozenAgentViolation(ConstitutionalViolation):
    """I-5: Frozen agent (P5/S2) contributed to reasoning."""

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(
            Boundary.FROZEN,
            f"Agent {agent_id} is constitutionally frozen — cannot participate in reasoning",
            context={"agent_id": agent_id},
            **kwargs,
        )


class SovereigntyViolation(ConstitutionalViolation):
    """I-6: External cloud auth detected."""

    def __init__(self, service: str, **kwargs):
        super().__init__(
            Boundary.SOVEREIGNTY,
            f"Cloud auth to '{service}' — sovereignty boundary violated",
            context={"service": service},
            **kwargs,
        )


class SpineViolation(ConstitutionalViolation):
    """I-7: Enforceable Spine section violated."""

    def __init__(self, section: int, detail: str, **kwargs):
        super().__init__(
            Boundary.SPINE,
            f"Spine section {section} violated: {detail}",
            context={"section": section, "detail": detail},
            **kwargs,
        )


# ═══════════════════════════════════════════════════════════════
# GATE REJECTIONS — Severity: REJECT
# ═══════════════════════════════════════════════════════════════


class GateRejection(BizraError):
    """
    Constitutional gate rejected an execution.
    Mission aborted. Rejection receipted.
    """

    severity = Severity.REJECT
    boundary = Boundary.MEMBRANE

    def __init__(self, gate: str, reason: str, *, score: float | None = None, **kwargs):
        self.gate = gate
        self.reason = reason
        self.score = score
        msg = f"Gate '{gate}' rejected: {reason}"
        if score is not None:
            msg += f" (score: {score:.4f})"
        super().__init__(
            msg, context={"gate": gate, "reason": reason, "score": score}, **kwargs
        )


class FateRejection(GateRejection):
    """FATE gate specifically rejected the execution."""

    def __init__(self, reason_codes: list[str], **kwargs):
        self.reason_codes = reason_codes
        super().__init__("FATE", ", ".join(reason_codes), **kwargs)
        self.context["reason_codes"] = reason_codes


# ═══════════════════════════════════════════════════════════════
# AUTHORITY ERRORS — Severity: REJECT
# ═══════════════════════════════════════════════════════════════


class AuthorityError(BizraError):
    """Missing or invalid execution authority."""

    severity = Severity.REJECT
    boundary = Boundary.AUTHORITY

    def __init__(self, authority: Any, **kwargs):
        super().__init__(
            f"Invalid authority: {repr(authority)!s:.50}",
            context={"authority": str(authority)[:100]},
            **kwargs,
        )


class MissingAuthority(AuthorityError):
    """No authority provided at all."""

    def __init__(self, **kwargs):
        super().__init__(None, **kwargs)


class ExpiredAuthority(AuthorityError):
    """Authority token has expired."""

    def __init__(self, expired_at: float, **kwargs):
        super().__init__(f"expired at {expired_at}", **kwargs)
        self.context["expired_at"] = expired_at


# ═══════════════════════════════════════════════════════════════
# CHAIN INTEGRITY ERRORS — Severity: HALT
# ═══════════════════════════════════════════════════════════════


class ReceiptChainError(BizraError):
    """Receipt chain integrity violation. Tamper detected."""

    severity = Severity.HALT
    boundary = Boundary.CHAIN

    def __init__(self, index: int, expected_hash: str, actual_hash: str, **kwargs):
        super().__init__(
            f"Chain tamper at index {index}: expected {expected_hash[:8]}, got {actual_hash[:8]}",
            context={"index": index, "expected": expected_hash, "actual": actual_hash},
            **kwargs,
        )


# ═══════════════════════════════════════════════════════════════
# BRIDGE ERRORS — Severity: DEGRADE
# ═══════════════════════════════════════════════════════════════


class BridgeError(BizraError):
    """
    Cross-language bridge failure (Python ↔ Rust, AHK, LLM).
    Degrade to fallback, receipt the failure.
    """

    severity = Severity.DEGRADE
    boundary = Boundary.BRIDGE

    def __init__(
        self,
        bridge_name: str,
        detail: str,
        *,
        original: Exception | None = None,
        **kwargs,
    ):
        self.bridge_name = bridge_name
        super().__init__(
            f"Bridge '{bridge_name}' failed: {detail}",
            context={"bridge": bridge_name, "detail": detail},
            original=original,
            **kwargs,
        )


class RustBridgeError(BridgeError):
    """PyO3 FFI bridge to Rust crate failed."""

    def __init__(self, crate: str, detail: str, **kwargs):
        super().__init__(f"rust:{crate}", detail, **kwargs)
        self.context["crate"] = crate


class AHKBridgeError(BridgeError):
    """AutoHotKey desktop bridge failed."""

    def __init__(self, action: str, detail: str, **kwargs):
        super().__init__("ahk", f"{action}: {detail}", **kwargs)
        self.context["action"] = action


class OllamaBridgeError(BridgeError):
    """Ollama/LLM inference bridge failed."""

    def __init__(self, model: str, detail: str, **kwargs):
        super().__init__(f"ollama:{model}", detail, **kwargs)
        self.context["model"] = model


# ═══════════════════════════════════════════════════════════════
# INFERENCE ERRORS — Severity: RETRY or DEGRADE
# ═══════════════════════════════════════════════════════════════


class InferenceError(BizraError):
    """LLM inference failed."""

    severity = Severity.RETRY
    boundary = Boundary.INFERENCE

    def __init__(self, model: str, detail: str, **kwargs):
        super().__init__(
            f"Inference failed on '{model}': {detail}",
            context={"model": model, "detail": detail},
            **kwargs,
        )


class InferenceTimeout(InferenceError):
    """LLM inference timed out."""

    severity = Severity.RETRY


class InferenceFallback(InferenceError):
    """Primary model unavailable, using fallback."""

    severity = Severity.DEGRADE

    def __init__(self, primary: str, fallback: str, **kwargs):
        super().__init__(primary, f"falling back to {fallback}", **kwargs)
        self.context["fallback"] = fallback


# ═══════════════════════════════════════════════════════════════
# RESOURCE ERRORS — Severity: DEGRADE
# ═══════════════════════════════════════════════════════════════


class ResourceError(BizraError):
    """System resource exhaustion."""

    severity = Severity.DEGRADE
    boundary = Boundary.RESOURCE


class MemoryExhausted(ResourceError):
    def __init__(self, used_mb: float, limit_mb: float, **kwargs):
        super().__init__(
            f"Memory exhausted: {used_mb:.0f}MB / {limit_mb:.0f}MB",
            context={"used_mb": used_mb, "limit_mb": limit_mb},
            **kwargs,
        )


class StorageExhausted(ResourceError):
    def __init__(self, path: str, **kwargs):
        super().__init__(f"Storage full at {path}", context={"path": path}, **kwargs)


# ═══════════════════════════════════════════════════════════════
# MIGRATION HELPER — Replace broad catches incrementally
# ═══════════════════════════════════════════════════════════════


def wrap_legacy_exception(
    exc: Exception, boundary: Boundary = Boundary.MEMBRANE
) -> BizraError:
    """
    Wrap a legacy Exception into a typed BizraError.

    Use this during migration from broad `except Exception` blocks.
    It preserves the original exception while adding typed metadata.

    Usage:
        try:
            legacy_code()
        except BizraError:
            raise  # Already typed, pass through
        except Exception as e:
            raise wrap_legacy_exception(e, Boundary.BRIDGE) from e
    """
    return BridgeError(
        bridge_name="legacy",
        detail=str(exc),
        original=exc,
    )


# ═══════════════════════════════════════════════════════════════
# TYPED CATCH PATTERN — Drop-in replacement for except Exception
# ═══════════════════════════════════════════════════════════════

"""
Migration guide for api.py and mission_nervous_system.py:

STEP 1: Import this module
    from core.errors import *

STEP 2: Replace broad catches with typed catches

    # api.py:991 — current:
    except Exception as e:
        logger.error(f"API error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    # api.py:991 — replacement:
    except ConstitutionalViolation as e:
        logger.critical(f"CONSTITUTIONAL HALT: {e}")
        return JSONResponse(status_code=403, content=e.to_receipt())
    except GateRejection as e:
        logger.warning(f"Gate rejection: {e}")
        return JSONResponse(status_code=422, content=e.to_receipt())
    except AuthorityError as e:
        logger.warning(f"Authority error: {e}")
        return JSONResponse(status_code=401, content=e.to_receipt())
    except BridgeError as e:
        logger.error(f"Bridge degradation: {e}")
        return JSONResponse(status_code=503, content=e.to_receipt())
    except InferenceError as e:
        logger.error(f"Inference failure: {e}")
        return JSONResponse(status_code=502, content=e.to_receipt())
    except BizraError as e:
        logger.error(f"System error: {e}")
        return JSONResponse(status_code=500, content=e.to_receipt())
    except Exception as e:
        # LAST RESORT — wrap legacy exception, receipt it, don't swallow
        wrapped = wrap_legacy_exception(e)
        logger.error(f"Untyped error (needs migration): {wrapped}")
        return JSONResponse(status_code=500, content=wrapped.to_receipt())

    # mission_nervous_system.py:374 — current:
    except Exception as e:
        logger.error(f"Rust bridge failed: {e}")
        return self._python_fallback(task)

    # mission_nervous_system.py:374 — replacement:
    except RustBridgeError as e:
        logger.warning(f"Rust bridge degraded: {e}")
        self._receipt_degradation(e)
        return self._python_fallback(task)
    except BizraError as e:
        logger.error(f"Mission error: {e}")
        return e.to_receipt()
    except Exception as e:
        wrapped = wrap_legacy_exception(e, Boundary.BRIDGE)
        logger.error(f"Untyped bridge error: {wrapped}")
        self._receipt_degradation(wrapped)
        return self._python_fallback(task)

STEP 3: Add SEC-003b exception count check
    The pre-commit hook tracks bare exception count.
    Each migration reduces the count. Ratchet only goes down.

HTTP status mapping:
    ConstitutionalViolation → 403 (Forbidden — constitutional halt)
    GateRejection           → 422 (Unprocessable — gate rejected)
    AuthorityError          → 401 (Unauthorized — missing/invalid auth)
    BridgeError             → 503 (Service Unavailable — degraded)
    InferenceError          → 502 (Bad Gateway — LLM failure)
    BizraError              → 500 (Internal — generic system error)
    Exception (legacy)      → 500 (Internal — needs migration)
"""
