"""Budgeted Constitutional Review — Path C, Layer 2 Contract.

Every non-kernel constitutional evaluator operates under explicit
computational budgets. If the budget is exceeded:
  - Low-risk actions → REVIEW (escalate)
  - High-risk actions → REJECT (fail closed)

This module mirrors the Rust `ReviewBudget` and `ReviewAction` types
from `bizra-core/src/kernel_action_grammar.rs`.

Standing on Giants: Gödel (incompleteness), seL4 (verified microkernel),
BIZRA constitutional spine.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Coroutine


class ConstitutionalVerdict(Enum):
    """Four possible outcomes of any constitutional evaluation."""

    PERMIT = auto()
    """Action is constitutionally valid. Proceed."""

    REJECT = auto()
    """Action violates hard constitutional law. Block."""

    REVIEW = auto()
    """Action requires escalation — too complex for bounded decision."""

    SCORE_ONLY = auto()
    """Advisory evaluation only — produces score, does not gate."""

    @property
    def is_permissive(self) -> bool:
        return self in (ConstitutionalVerdict.PERMIT, ConstitutionalVerdict.SCORE_ONLY)

    @property
    def is_blocking(self) -> bool:
        return self is ConstitutionalVerdict.REJECT

    @property
    def requires_escalation(self) -> bool:
        return self is ConstitutionalVerdict.REVIEW


class ConstitutionalLayer(Enum):
    """The three layers of the stratified constitution."""

    HARD_LAW = auto()
    """Layer 1 — Decidable, blocking, replayable. This is physics."""

    BOUNDED_REVIEW = auto()
    """Layer 2 — Timeout-aware, fail-closed for high-risk classes."""

    JUDICIARY = auto()
    """Layer 3 — Advisory/deliberative. This is jurisprudence."""


class RiskClass(Enum):
    """Risk classification for budget-exceeded fallback behavior."""

    LOW = auto()
    """Budget exceeded → REVIEW (escalate, allow monitoring)."""

    HIGH = auto()
    """Budget exceeded → REJECT (fail closed)."""


@dataclass(frozen=True)
class ReviewBudget:
    """Computational budget for a bounded constitutional review.

    Every Layer 2 evaluator receives this contract. If the evaluation
    exceeds the budget, the fallback verdict applies automatically.
    """

    max_time_ms: float
    """Maximum wall-clock time in milliseconds."""

    max_memory_bytes: int
    """Maximum memory allocation in bytes."""

    max_depth: int
    """Maximum provenance chain depth to traverse."""

    allow_approximation: bool
    """Whether approximate algorithms are permitted."""

    risk_class: RiskClass
    """Determines fallback behavior when budget is exceeded."""

    @property
    def fallback_verdict(self) -> ConstitutionalVerdict:
        """Verdict to return if budget is exceeded."""
        if self.risk_class is RiskClass.HIGH:
            return ConstitutionalVerdict.REJECT
        return ConstitutionalVerdict.REVIEW


# ─────────────────────────────────────────────────────────
# Default budgets for Layer 2 review actions
# ─────────────────────────────────────────────────────────

REVIEW_BUDGETS: dict[str, ReviewBudget] = {
    "ihsan_projection": ReviewBudget(
        max_time_ms=100.0,
        max_memory_bytes=4 * 1024 * 1024,
        max_depth=8,
        allow_approximation=True,
        risk_class=RiskClass.LOW,
    ),
    "adl_delta_approx": ReviewBudget(
        max_time_ms=200.0,
        max_memory_bytes=8 * 1024 * 1024,
        max_depth=16,
        allow_approximation=True,
        risk_class=RiskClass.HIGH,
    ),
    "provenance_traversal": ReviewBudget(
        max_time_ms=500.0,
        max_memory_bytes=16 * 1024 * 1024,
        max_depth=64,
        allow_approximation=False,
        risk_class=RiskClass.LOW,
    ),
    "formal_verification": ReviewBudget(
        max_time_ms=5000.0,
        max_memory_bytes=64 * 1024 * 1024,
        max_depth=32,
        allow_approximation=False,
        risk_class=RiskClass.HIGH,
    ),
    "regime_check": ReviewBudget(
        max_time_ms=50.0,
        max_memory_bytes=2 * 1024 * 1024,
        max_depth=4,
        allow_approximation=True,
        risk_class=RiskClass.LOW,
    ),
}


@dataclass
class VerdictReceipt:
    """Extended verdict metadata for constitutional receipts.

    Records not just yes/no, but what was decided where and how.
    """

    verdict: ConstitutionalVerdict
    layer: ConstitutionalLayer
    action: str
    approximated: bool = False
    budget_exceeded: bool = False
    escalated_from: ConstitutionalLayer | None = None
    elapsed_ms: float = 0.0
    reason: str = ""
    score: float | None = None


async def budgeted_evaluate(
    action: str,
    evaluator: Callable[..., Coroutine[Any, Any, ConstitutionalVerdict]],
    budget: ReviewBudget | None = None,
    **kwargs: Any,
) -> VerdictReceipt:
    """Execute a constitutional evaluation under a bounded budget.

    If the evaluator exceeds the time budget, the fallback verdict
    is returned automatically. This is the core Layer 2 contract.

    Args:
        action: Name of the review action (must be in REVIEW_BUDGETS).
        evaluator: Async callable that performs the constitutional check.
        budget: Optional override budget. Defaults to REVIEW_BUDGETS[action].
        **kwargs: Passed to the evaluator.

    Returns:
        VerdictReceipt with full decision metadata.
    """
    if budget is None:
        budget = REVIEW_BUDGETS.get(action)
        if budget is None:
            return VerdictReceipt(
                verdict=ConstitutionalVerdict.REJECT,
                layer=ConstitutionalLayer.BOUNDED_REVIEW,
                action=action,
                reason=f"No budget defined for action: {action}",
            )

    start = time.monotonic()
    timeout_s = budget.max_time_ms / 1000.0

    try:
        verdict = await asyncio.wait_for(
            evaluator(**kwargs),
            timeout=timeout_s,
        )
        elapsed = (time.monotonic() - start) * 1000.0

        return VerdictReceipt(
            verdict=verdict,
            layer=ConstitutionalLayer.BOUNDED_REVIEW,
            action=action,
            approximated=budget.allow_approximation,
            budget_exceeded=False,
            elapsed_ms=elapsed,
        )

    except asyncio.TimeoutError:
        elapsed = (time.monotonic() - start) * 1000.0
        fallback = budget.fallback_verdict

        return VerdictReceipt(
            verdict=fallback,
            layer=ConstitutionalLayer.BOUNDED_REVIEW,
            action=action,
            approximated=budget.allow_approximation,
            budget_exceeded=True,
            elapsed_ms=elapsed,
            reason=f"Budget exceeded: {elapsed:.1f}ms > {budget.max_time_ms}ms → {fallback.name}",
        )

    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000.0

        return VerdictReceipt(
            verdict=ConstitutionalVerdict.REJECT,
            layer=ConstitutionalLayer.BOUNDED_REVIEW,
            action=action,
            budget_exceeded=False,
            elapsed_ms=elapsed,
            reason=f"Evaluator error: {type(exc).__name__}: {exc}",
        )
