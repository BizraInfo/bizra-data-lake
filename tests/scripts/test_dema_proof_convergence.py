"""A2.0 Proof Convergence Verifier contract tests."""

from __future__ import annotations

import pytest

from core.dema.proof_convergence import (
    ProofConvergenceResult,
    ProofConvergenceVerifier,
    ProofSignal,
    converge_proofs,
)


def test_no_signals_requires_approval():
    result = converge_proofs([])

    assert result.converged is False
    assert result.verdict == "require_approval"
    assert result.truth_label == "UNKNOWN"
    assert result.reasons == ("no_proof_signals",)


def test_passing_signals_converge_to_requested_decision():
    result = converge_proofs(
        [
            ProofSignal(source="receipt", passed=True, evidence_refs=("rcpt-1",)),
            ProofSignal(source="fate", passed=True, evidence_refs=("fate-1",)),
        ],
        requested_decision="notify",
        required_sources=("receipt", "fate"),
    )

    assert result == ProofConvergenceResult(
        converged=True,
        verdict="notify",
        truth_label="DERIVED",
        reasons=("proof_signals_converged",),
        sources=("fate", "receipt"),
        evidence_refs=("fate-1", "rcpt-1"),
    )


def test_failed_signal_fails_closed_to_forbid():
    result = converge_proofs(
        [
            ProofSignal(source="receipt", passed=True),
            ProofSignal(
                source="fate",
                passed=False,
                reason="ihsan floor violation",
                decision="forbid",
            ),
        ],
        requested_decision="auto_low_risk",
    )

    assert result.converged is True
    assert result.verdict == "forbid"
    assert result.blocking_sources == ("fate",)
    assert result.reasons == ("ihsan floor violation",)


def test_unknown_signal_requires_approval():
    result = converge_proofs(
        [
            ProofSignal(source="receipt", passed=True),
            ProofSignal(source="sat", passed=None, truth_label="UNKNOWN"),
        ],
        requested_decision="notify",
    )

    assert result.converged is False
    assert result.verdict == "require_approval"
    assert result.truth_label == "UNKNOWN"
    assert result.reasons == ("indeterminate_proof_state",)


def test_missing_required_source_requires_approval():
    result = converge_proofs(
        [ProofSignal(source="receipt", passed=True)],
        requested_decision="notify",
        required_sources=("receipt", "fate"),
    )

    assert result.converged is False
    assert result.verdict == "require_approval"
    assert result.missing_sources == ("fate",)
    assert result.reasons == ("missing_required_sources",)


def test_decision_disagreement_picks_safest_verdict_without_convergence():
    result = converge_proofs(
        [
            ProofSignal(source="receipt", passed=True, decision="auto_low_risk"),
            ProofSignal(source="policy", passed=True, decision="require_approval"),
        ]
    )

    assert result.converged is False
    assert result.verdict == "require_approval"
    assert result.reasons == ("decision_disagreement",)


def test_rejects_invalid_csl_values():
    with pytest.raises(ValueError, match="decision must be one of"):
        ProofSignal(source="policy", passed=True, decision="allow")

    with pytest.raises(ValueError, match="truth_label must be one of"):
        ProofSignal(source="policy", passed=True, truth_label="REAL")


def test_rejects_duplicate_sources():
    verifier = ProofConvergenceVerifier()

    with pytest.raises(ValueError, match="duplicate proof signal sources"):
        verifier.verify(
            [
                ProofSignal(source="receipt", passed=True),
                ProofSignal(source="receipt", passed=True),
            ]
        )


def test_source_identities_are_canonicalized_before_convergence():
    verifier = ProofConvergenceVerifier()

    with pytest.raises(ValueError, match="duplicate proof signal sources"):
        verifier.verify(
            [
                ProofSignal(source="receipt", passed=True),
                ProofSignal(source=" receipt ", passed=True),
            ]
        )

    result = verifier.verify(
        [ProofSignal(source=" receipt ", passed=True)],
        required_sources=("receipt", " public_key_replay "),
    )

    assert result.sources == ("receipt",)
    assert result.missing_sources == ("public_key_replay",)


def test_rejects_blank_required_sources():
    verifier = ProofConvergenceVerifier()

    with pytest.raises(ValueError, match="required_sources must be non-empty"):
        verifier.verify(
            [ProofSignal(source="receipt", passed=True)], required_sources=(" ",)
        )
