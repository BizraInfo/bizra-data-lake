"""FATE Crossing — explicit PAT → Evidence Audit → SAT boundary with receipt emission."""

import hashlib
import time
from dataclasses import asdict

from mvda.evidence_auditor import EvidenceAuditResult, audit_evidence
from mvda.ledger import MvdaLedger
from mvda.pat_researcher import PatResult, run_pat_researcher
from mvda.receipt import MvdaReceipt
from mvda.sat_validator import SatVerdict, run_sat_validator


def execute_mvda(question: str, ledger: MvdaLedger) -> dict:
    """Execute full MVDA cycle: PAT → Evidence Audit → FATE → SAT → receipts → ledger."""

    # ── Step 1: PAT Researcher executes ──
    pat_result = run_pat_researcher(question)

    pat_receipt = MvdaReceipt(
        actor="pat_researcher",
        step="pat_execution",
        status="completed" if pat_result.answer else "failed",
        verdict="",
        reason=f"confidence={pat_result.confidence}, model={pat_result.model}",
        evidence_refs=pat_result.evidence_refs,
        evidence_sufficient=len(pat_result.evidence_refs) >= 1,
        content_hash=hashlib.blake2b(
            pat_result.answer.encode(), digest_size=16
        ).hexdigest() if pat_result.answer else "",
    )
    ledger.append(pat_receipt)

    # ── Step 2: Evidence Auditor verifies refs exist ──
    audit_result = audit_evidence(pat_result.evidence_refs)

    audit_receipt = MvdaReceipt(
        actor="evidence_auditor",
        step="evidence_audit",
        status="pass" if audit_result.all_refs_valid else "failed",
        verdict="BLOCKED_BY_EVIDENCE" if not audit_result.all_refs_valid else "",
        reason=(
            f"valid={audit_result.valid_count}/{audit_result.total_count}"
            + (f", invalid: {audit_result.invalid_refs}" if audit_result.invalid_refs else "")
        ),
        evidence_refs=pat_result.evidence_refs,
        evidence_sufficient=audit_result.all_refs_valid and audit_result.total_count > 0,
        metadata={
            "valid_count": audit_result.valid_count,
            "invalid_count": audit_result.invalid_count,
            "invalid_refs": audit_result.invalid_refs,
            "audit_notes": audit_result.audit_notes,
        },
    )
    ledger.append(audit_receipt)

    # If evidence audit fails, short-circuit to BLOCKED_BY_EVIDENCE
    if not audit_result.all_refs_valid or audit_result.total_count == 0:
        blocked_receipt = MvdaReceipt(
            actor="fate_crossing",
            step="evidence_gate_block",
            status="blocked",
            verdict="BLOCKED_BY_EVIDENCE",
            reason=f"Evidence audit failed: {audit_result.invalid_count} invalid refs of {audit_result.total_count}",
            evidence_refs=pat_result.evidence_refs,
            evidence_sufficient=False,
        )
        ledger.append(blocked_receipt)

        return {
            "question": question,
            "pat_answer": pat_result.answer,
            "pat_confidence": pat_result.confidence,
            "pat_evidence_refs": pat_result.evidence_refs,
            "pat_model": pat_result.model,
            "evidence_audit_valid": audit_result.all_refs_valid,
            "evidence_audit_invalid_refs": audit_result.invalid_refs,
            "sat_verdict": "BLOCKED_BY_EVIDENCE",
            "sat_reason": f"Evidence audit: {audit_result.invalid_count} invalid refs",
            "sat_ihsan_score": 0.0,
            "sat_evidence_sufficient": False,
            "sat_model": "evidence-auditor-gate",
            "ledger_path": str(ledger.path),
            "receipts_emitted": 3,
        }

    # ── Step 3: FATE boundary — evidence verified, hand off to SAT ──
    fate_receipt = MvdaReceipt(
        actor="fate_crossing",
        step="pat_to_sat_handoff",
        status="crossing",
        verdict="",
        reason=f"Evidence audit passed ({audit_result.valid_count}/{audit_result.total_count}). Handing to SAT.",
        evidence_refs=pat_result.evidence_refs,
        evidence_sufficient=True,
    )
    ledger.append(fate_receipt)

    # ── Step 4: SAT Validator evaluates quality/Ihsan ──
    sat_verdict = run_sat_validator(pat_result)

    sat_receipt = MvdaReceipt(
        actor="sat_validator",
        step="sat_verdict",
        status="pass" if sat_verdict.verdict == "PASS" else "blocked",
        verdict=sat_verdict.verdict,
        reason=sat_verdict.reason,
        ihsan_score=sat_verdict.ihsan_score,
        evidence_refs=pat_result.evidence_refs,
        evidence_sufficient=sat_verdict.evidence_sufficient,
        metadata={"sat_model": sat_verdict.model},
    )
    ledger.append(sat_receipt)

    return {
        "question": question,
        "pat_answer": pat_result.answer,
        "pat_confidence": pat_result.confidence,
        "pat_evidence_refs": pat_result.evidence_refs,
        "pat_model": pat_result.model,
        "evidence_audit_valid": audit_result.all_refs_valid,
        "evidence_audit_invalid_refs": audit_result.invalid_refs,
        "sat_verdict": sat_verdict.verdict,
        "sat_reason": sat_verdict.reason,
        "sat_ihsan_score": sat_verdict.ihsan_score,
        "sat_evidence_sufficient": sat_verdict.evidence_sufficient,
        "sat_model": sat_verdict.model,
        "ledger_path": str(ledger.path),
        "receipts_emitted": 4,
    }
