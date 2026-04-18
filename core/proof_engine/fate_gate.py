"""
FATE Gate — Governed crossing from PAT execution to SAT verdict.

Wires: PAT output → Evidence Auditor → SAT Validator → SatVerdict

This is the core orchestration boundary. Any PAT implementation that
produces a PatOutput-compatible object can be validated through this gate.

No MVDA dependency. No ledger dependency. Pure proof-engine orchestration.

Standing on Giants:
- Lamport (1982): Byzantine fault tolerance at boundaries
- Al-Ghazali (1095): Governance must precede action
- BIZRA FATE: Formal Assertion Through Execution
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.proof_engine.evidence_audit import (
    EvidenceAuditResult,
    audit_evidence,
)
from core.proof_engine.sat_validator import (
    PatOutput,
    SatVerdict,
    validate,
)


@dataclass
class FateResult:
    """Complete result of a FATE crossing: evidence audit + SAT verdict."""

    verdict: SatVerdict
    evidence_audit: EvidenceAuditResult
    short_circuited: bool = False
    telemetry_summary: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict.verdict == "PASS"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.to_dict(),
            "evidence_audit": {
                "all_refs_valid": self.evidence_audit.all_refs_valid,
                "valid_count": self.evidence_audit.valid_count,
                "invalid_count": self.evidence_audit.invalid_count,
                "invalid_refs": self.evidence_audit.invalid_refs,
            },
            "short_circuited": self.short_circuited,
            "passed": self.passed,
            "telemetry": self.telemetry_summary,
        }


def validate_with_evidence(
    pat_output: PatOutput, *, emit_telemetry: bool = True
) -> FateResult:
    """Run a full FATE crossing: evidence audit then SAT validation.

    This is the canonical entry point for governed PAT output validation.

    Flow:
        1. Audit all evidence refs for local existence
        2. If any ref is invalid or missing → BLOCKED_BY_EVIDENCE (short-circuit)
        3. Otherwise → SAT Validator evaluates quality/Ihsān
        4. Return FateResult with both audit and verdict

    Args:
        pat_output: Any object with .answer, .evidence_refs, .confidence
        emit_telemetry: Write telemetry events to FATE telemetry log (default True).

    Returns:
        FateResult containing the SatVerdict and EvidenceAuditResult.
    """
    # Initialize telemetry (fail-silent if unavailable)
    telem = None
    try:
        from core.proof_engine.fate_telemetry import FateTelemetry

        telem = FateTelemetry()
    except ImportError:
        pass

    # Step 1: Evidence Auditor
    audit = audit_evidence(pat_output.evidence_refs)

    if telem:
        telem.record(
            "evidence_audit",
            evidence_valid=audit.all_refs_valid,
            evidence_count=audit.total_count,
            invalid_refs=audit.invalid_refs,
        )

    # Step 2: Short-circuit if evidence fails
    if not audit.all_refs_valid or audit.total_count == 0:
        reason = (
            f"Evidence audit: {audit.invalid_count} invalid refs"
            if audit.invalid_count > 0
            else "No evidence refs provided"
        )
        blocked_verdict = SatVerdict(
            verdict="BLOCKED_BY_EVIDENCE",
            reason=reason,
            ihsan_score=0.0,
            evidence_sufficient=False,
            model="evidence-auditor-gate",
        )
        if telem:
            telem.record(
                "fate_result",
                verdict="BLOCKED_BY_EVIDENCE",
                short_circuited=True,
                evidence_valid=False,
            )
            if emit_telemetry:
                telem.emit()

        return FateResult(
            verdict=blocked_verdict,
            evidence_audit=audit,
            short_circuited=True,
            telemetry_summary=telem.summary() if telem else {},
        )

    # Step 3: SAT Validator
    sat_verdict = validate(pat_output)

    if telem:
        telem.record(
            "sat_verdict",
            verdict=sat_verdict.verdict,
            ihsan_score=sat_verdict.ihsan_score,
            evidence_valid=sat_verdict.evidence_sufficient,
            model=sat_verdict.model,
        )
        telem.record(
            "fate_result",
            verdict=sat_verdict.verdict,
            ihsan_score=sat_verdict.ihsan_score,
            short_circuited=False,
        )
        if emit_telemetry:
            telem.emit()

    return FateResult(
        verdict=sat_verdict,
        evidence_audit=audit,
        short_circuited=False,
        telemetry_summary=telem.summary() if telem else {},
    )
