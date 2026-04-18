"""
SAPE v2.0 Intent Gate — Al-Ghazali Pre-Gate

The Intent Gate is the constitutional pre-gate: no computation proceeds
without clear, honest intent. This is not a weight — it is a gate.

Standing on Giants: Al-Ghazali (1058–1111) — intent precedes action.

Created: 2026-04-10 | BIZRA SAPE v2.0
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from core.integration.constants import INTENT_FLOOR

from .types import (
    STAKES_TO_MODE,
    EvidenceLevel,
    ExecutionMode,
    IntentSlots,
    Module,
    ModuleResult,
)

logger = logging.getLogger(__name__)


def validate_intent(slots: IntentSlots) -> Tuple[bool, List[str]]:
    """
    Validate intent slots before any SAPE execution begins.

    Returns (passed, list_of_errors).
    An empty error list means the intent gate passed.
    """
    errors: List[str] = []

    if not slots.domain or not slots.domain.strip():
        errors.append("Intent Gate: domain is empty — what area is this about?")

    if not slots.objective or not slots.objective.strip():
        errors.append("Intent Gate: objective is empty — what is the goal?")

    if slots.stakes not in ("L", "M", "H"):
        errors.append(
            f"Intent Gate: invalid stakes '{slots.stakes}' — must be L, M, or H"
        )

    # Al-Ghazali correction: reject unclear intent before resource spend
    if slots.stakes == "H" and not slots.success_criteria:
        errors.append(
            "Intent Gate: high-stakes task requires explicit success criteria"
        )

    if slots.stakes == "H" and not slots.constraints:
        errors.append("Intent Gate: high-stakes task requires explicit constraints")

    return len(errors) == 0, errors


def resolve_mode(slots: IntentSlots) -> ExecutionMode:
    """
    Resolve execution mode from intent slots.

    Mode Rule:
      Low stakes  → Lite
      Medium stakes → Standard
      High stakes → Deep
    """
    return STAKES_TO_MODE.get(slots.stakes, ExecutionMode.STANDARD)


def run_intent_gate(
    slots: IntentSlots,
    *,
    intent_score: Optional[float] = None,
) -> ModuleResult:
    """
    Execute the Intent Gate module.

    This is Module 1 of the 7-module SAPE pipeline.
    If intent_score is provided and below INTENT_FLOOR, the gate fails.
    If slots fail structural validation, the gate fails.

    Returns a ModuleResult with metadata containing:
      - "passed": bool
      - "errors": list of validation errors
      - "resolved_mode": the ExecutionMode derived from stakes
      - "evidence_level": classification of the intent clarity
    """
    passed, errors = validate_intent(slots)

    # Apply intent score floor if provided
    effective_score = intent_score if intent_score is not None else 1.0
    if effective_score < INTENT_FLOOR:
        passed = False
        errors.append(
            f"Intent Gate: intent score {effective_score:.3f} "
            f"below floor {INTENT_FLOOR}"
        )

    resolved_mode = resolve_mode(slots)

    # Classify evidence level of the intent itself
    if slots.success_criteria and slots.constraints:
        evidence_level = EvidenceLevel.VERIFIED
    elif slots.success_criteria or slots.constraints:
        evidence_level = EvidenceLevel.GROUNDED_INFERENCE
    else:
        evidence_level = EvidenceLevel.CONJECTURE

    ihsan_score = effective_score if passed else 0.0

    output_lines = [
        f"Domain: {slots.domain}",
        f"Objective: {slots.objective}",
        f"Stakes: {slots.stakes} → Mode: {resolved_mode.value}",
        f"Constraints: {slots.constraints or '(none)'}",
        f"Success Criteria: {slots.success_criteria or '(none)'}",
        f"Forbidden: {', '.join(slots.forbidden_moves) or '(none)'}",
        f"Intent Score: {effective_score:.3f}",
        f"Gate: {'PASS' if passed else 'FAIL'}",
    ]
    if errors:
        output_lines.append(f"Errors: {'; '.join(errors)}")

    return ModuleResult(
        module=Module.INTENT_GATE,
        output="\n".join(output_lines),
        snr_score=effective_score if passed else 0.0,
        ihsan_score=ihsan_score,
        metadata={
            "passed": passed,
            "errors": errors,
            "resolved_mode": resolved_mode.value,
            "evidence_level": evidence_level.value,
            "intent_score": effective_score,
        },
    )
