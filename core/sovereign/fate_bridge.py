"""
FATE Bridge — Connects core proof_engine FATE gate to SovereignRuntime.

This is the integration adapter that allows SovereignRuntime's query pipeline
to validate responses through the promoted FATE gate (Evidence Auditor + SAT Validator)
without modifying the runtime's existing constitutional validation flow.

Integration point: called after LLM inference (STAGE 2), before SNR optimization (STAGE 3).
If FATE gate blocks, the result is marked as failed with the verdict reason.
If FATE gate passes or is unavailable, the pipeline continues normally.

Standing on Giants:
- Adapter Pattern (GoF, 1994): Bridge between incompatible interfaces
- Fail-open vs fail-closed: FATE bridge is fail-OPEN (graceful degradation)
  because the existing STAGE 4 constitutional validation is already fail-closed.
  This is additive verification, not replacement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger("sovereign.fate_bridge")


@dataclass
class FateBridgeInput:
    """Adapter type: maps SovereignRuntime query result to FATE gate input."""

    answer: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    confidence: str = "medium"


@dataclass
class FateBridgeResult:
    """Result of FATE bridge evaluation."""

    enabled: bool = False
    passed: bool = True
    verdict: str = ""
    reason: str = ""
    ihsan_score: float = 0.0
    evidence_valid: bool = True
    short_circuited: bool = False


def run_fate_bridge(
    answer: str,
    evidence_refs: Optional[List[str]] = None,
    confidence: str = "medium",
) -> FateBridgeResult:
    """Run the FATE gate bridge on a query result.

    This is the entry point called by SovereignRuntime. It:
    1. Imports the FATE gate (graceful degradation if unavailable)
    2. Constructs a PatOutput-compatible input
    3. Runs validate_with_evidence
    4. Returns a FateBridgeResult

    Args:
        answer: The LLM-generated response text.
        evidence_refs: Optional list of evidence refs (e.g., from RAG/retrieval).
                      If None or empty, FATE bridge is skipped (no evidence to audit).
        confidence: Confidence level string.

    Returns:
        FateBridgeResult with verdict and pass/fail status.
    """
    # No evidence refs → skip FATE (nothing to audit)
    if not evidence_refs:
        return FateBridgeResult(
            enabled=False,
            passed=True,
            reason="No evidence refs — FATE bridge skipped",
        )

    try:
        from core.proof_engine.fate_gate import validate_with_evidence
        from core.proof_engine.sat_validator import SimplePatOutput
    except ImportError as e:
        logger.debug("FATE gate unavailable: %s", e)
        return FateBridgeResult(
            enabled=False,
            passed=True,
            reason=f"FATE gate import failed: {e}",
        )

    pat_output = SimplePatOutput(
        answer=answer,
        evidence_refs=evidence_refs,
        confidence=confidence,
    )

    try:
        fate_result = validate_with_evidence(pat_output)
    except Exception as e:
        logger.warning("FATE gate execution error: %s", e)
        return FateBridgeResult(
            enabled=True,
            passed=True,
            reason=f"FATE gate error (fail-open): {e}",
        )

    return FateBridgeResult(
        enabled=True,
        passed=fate_result.passed,
        verdict=fate_result.verdict.verdict,
        reason=fate_result.verdict.reason,
        ihsan_score=fate_result.verdict.ihsan_score,
        evidence_valid=fate_result.evidence_audit.all_refs_valid,
        short_circuited=fate_result.short_circuited,
    )
