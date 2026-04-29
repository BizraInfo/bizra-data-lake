"""A2-A5 Proof Surface contract.

This module turns proof-convergence results into a small, deterministic
operator-facing surface. It is intentionally pure: no frontend, no receipt
store reads, no runtime boot, and no signed-export claim beyond an honest
``receipt_export_ready`` flag.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import blake3

from core.dema.csl import DISPLAY_TRUTH_LABELS, SCHEMA_VERSION, DecisionVerdict
from core.dema.proof_convergence import (
    ProofConvergenceResult,
    ProofSignal,
    converge_proofs,
)


def _canonical_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return blake3.blake3(encoded).hexdigest()


def _validate_truth_label(value: str) -> None:
    if value not in DISPLAY_TRUTH_LABELS:
        raise ValueError(
            f"truth_label must be one of {DISPLAY_TRUTH_LABELS}, got {value!r}"
        )


@dataclass(frozen=True)
class ClaimSource:
    """Claim + source pair accepted by the proof surface form."""

    claim: str
    source: str
    truth_label: str = "DERIVED"
    evidence_ref: str | None = None

    def __post_init__(self) -> None:
        claim = self.claim.strip()
        source = self.source.strip()
        if not claim:
            raise ValueError("claim must be non-empty")
        if not source:
            raise ValueError("source must be non-empty")
        _validate_truth_label(self.truth_label)
        object.__setattr__(self, "claim", claim)
        object.__setattr__(self, "source", source)
        if self.evidence_ref is not None:
            evidence_ref = self.evidence_ref.strip()
            object.__setattr__(self, "evidence_ref", evidence_ref or None)

    def to_dict(self) -> dict[str, object]:
        return {
            "claim": self.claim,
            "source": self.source,
            "truth_label": self.truth_label,
            "evidence_ref": self.evidence_ref,
        }


@dataclass(frozen=True)
class ProofSurface:
    """Canonical A2-A5 proof surface shape for Dema UI consumers."""

    schema_version: str
    surface_id: str
    claim: str
    source: str
    truth_label: str
    decision: str
    decision_reason: str
    evidence_auditor_verdict: str
    converged: bool
    receipt_id: str | None
    receipt_export_ready: bool
    evidence_refs: tuple[str, ...]
    reasons: tuple[str, ...]
    sources: tuple[str, ...]
    blocking_sources: tuple[str, ...]
    missing_sources: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "surface_id": self.surface_id,
            "claim": self.claim,
            "source": self.source,
            "truth_label": self.truth_label,
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "evidence_auditor_verdict": self.evidence_auditor_verdict,
            "converged": self.converged,
            "receipt_id": self.receipt_id,
            "receipt_export_ready": self.receipt_export_ready,
            "evidence_refs": list(self.evidence_refs),
            "reasons": list(self.reasons),
            "sources": list(self.sources),
            "blocking_sources": list(self.blocking_sources),
            "missing_sources": list(self.missing_sources),
        }


def build_proof_surface(
    claim_source: ClaimSource,
    signals: Iterable[ProofSignal],
    *,
    requested_decision: str | None = None,
    required_sources: Iterable[str] = (),
    receipt_id: str | None = None,
) -> ProofSurface:
    """Build a UI-safe proof surface from a claim/source and proof signals."""

    convergence = converge_proofs(
        signals,
        requested_decision=requested_decision,
        required_sources=required_sources,
    )
    return proof_surface_from_convergence(
        claim_source,
        convergence,
        receipt_id=receipt_id,
    )


def proof_surface_from_convergence(
    claim_source: ClaimSource,
    convergence: ProofConvergenceResult,
    *,
    receipt_id: str | None = None,
) -> ProofSurface:
    """Project an already-computed convergence result into a proof surface."""

    evidence_refs = sorted(
        {
            ref
            for ref in (
                *(convergence.evidence_refs),
                claim_source.evidence_ref,
            )
            if ref is not None
        }
    )
    decision_reason = (
        "; ".join(convergence.reasons) if convergence.reasons else "proof_surface_ready"
    )
    receipt_export_ready = bool(
        receipt_id
        and convergence.converged
        and convergence.verdict != DecisionVerdict.FORBID.value
    )
    evidence_auditor_verdict = convergence.verdict
    surface_seed = {
        "schema_version": SCHEMA_VERSION,
        "claim": claim_source.claim,
        "source": claim_source.source,
        "truth_label": claim_source.truth_label,
        "decision": convergence.verdict,
        "decision_reason": decision_reason,
        "evidence_auditor_verdict": evidence_auditor_verdict,
        "converged": convergence.converged,
        "receipt_id": receipt_id,
        "receipt_export_ready": receipt_export_ready,
        "evidence_refs": evidence_refs,
        "reasons": list(convergence.reasons),
        "sources": list(convergence.sources),
        "blocking_sources": list(convergence.blocking_sources),
        "missing_sources": list(convergence.missing_sources),
    }

    return ProofSurface(
        schema_version=SCHEMA_VERSION,
        surface_id=_canonical_digest(surface_seed),
        claim=claim_source.claim,
        source=claim_source.source,
        truth_label=claim_source.truth_label,
        decision=convergence.verdict,
        decision_reason=decision_reason,
        evidence_auditor_verdict=evidence_auditor_verdict,
        converged=convergence.converged,
        receipt_id=receipt_id,
        receipt_export_ready=receipt_export_ready,
        evidence_refs=tuple(evidence_refs),
        reasons=convergence.reasons,
        sources=convergence.sources,
        blocking_sources=convergence.blocking_sources,
        missing_sources=convergence.missing_sources,
    )
