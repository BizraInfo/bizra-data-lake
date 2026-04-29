"""A2-A5 Proof Surface contract tests."""

from __future__ import annotations

import json

import pytest

from core.dema import ClaimSource, ProofSignal, build_proof_surface
from core.dema.proof_convergence import converge_proofs
from core.dema.proof_surface import proof_surface_from_convergence


def test_claim_source_blocks_missing_claim_or_source():
    with pytest.raises(ValueError, match="claim must be non-empty"):
        ClaimSource(claim=" ", source="receipt store")

    with pytest.raises(ValueError, match="source must be non-empty"):
        ClaimSource(claim="receipt landed", source=" ")


def test_claim_source_rejects_invalid_truth_label():
    with pytest.raises(ValueError, match="truth_label must be one of"):
        ClaimSource(claim="receipt landed", source="receipt store", truth_label="REAL")


def test_build_proof_surface_projects_evidence_auditor_verdict():
    surface = build_proof_surface(
        ClaimSource(
            claim="Mission receipt landed",
            source="sovereign_state/dema/receipts",
            evidence_ref="receipt:abc",
        ),
        [
            ProofSignal(source="receipt", passed=True, evidence_refs=("receipt:abc",)),
            ProofSignal(source="auditor", passed=True, evidence_refs=("audit:ok",)),
        ],
        requested_decision="notify",
        required_sources=("receipt", "auditor"),
        receipt_id="abc",
    )

    assert surface.converged is True
    assert surface.decision == "notify"
    assert surface.evidence_auditor_verdict == "notify"
    assert surface.receipt_export_ready is True
    assert surface.evidence_refs == ("audit:ok", "receipt:abc")
    assert surface.reasons == ("proof_signals_converged",)


def test_failed_signal_surfaces_forbid_and_blocks_export():
    surface = build_proof_surface(
        ClaimSource(claim="Claim rejected", source="evidence auditor"),
        [
            ProofSignal(source="receipt", passed=True),
            ProofSignal(
                source="auditor",
                passed=False,
                reason="receipt signature missing",
                decision="forbid",
            ),
        ],
        requested_decision="notify",
        receipt_id="abc",
    )

    assert surface.converged is True
    assert surface.decision == "forbid"
    assert surface.receipt_export_ready is False
    assert surface.blocking_sources == ("auditor",)
    assert surface.decision_reason == "receipt signature missing"


def test_missing_required_source_requires_approval_on_surface():
    surface = build_proof_surface(
        ClaimSource(claim="Needs public key replay", source="operator form"),
        [ProofSignal(source="receipt", passed=True)],
        requested_decision="notify",
        required_sources=("receipt", "public_key_replay"),
        receipt_id="abc",
    )

    assert surface.converged is False
    assert surface.decision == "require_approval"
    assert surface.receipt_export_ready is False
    assert surface.missing_sources == ("public_key_replay",)
    assert surface.decision_reason == "missing_required_sources"


def test_proof_surface_id_is_deterministic_for_same_inputs():
    claim_source = ClaimSource(
        claim="Sovereignty reveal is backed by a receipt",
        source="receipt panel",
    )
    left = build_proof_surface(
        claim_source,
        [
            ProofSignal(source="receipt", passed=True, evidence_refs=("r1",)),
            ProofSignal(source="auditor", passed=True, evidence_refs=("a1",)),
        ],
        requested_decision="notify",
        receipt_id="r1",
    )
    right = build_proof_surface(
        claim_source,
        [
            ProofSignal(source="auditor", passed=True, evidence_refs=("a1",)),
            ProofSignal(source="receipt", passed=True, evidence_refs=("r1",)),
        ],
        requested_decision="notify",
        receipt_id="r1",
    )

    assert left.surface_id == right.surface_id


def test_proof_surface_id_changes_for_different_missing_sources():
    claim_source = ClaimSource(
        claim="Sovereignty reveal is backed by a receipt",
        source="receipt panel",
    )
    receipt_missing = build_proof_surface(
        claim_source,
        [ProofSignal(source="auditor", passed=True, evidence_refs=("a1",))],
        requested_decision="notify",
        required_sources=("receipt", "auditor"),
    )
    replay_missing = build_proof_surface(
        claim_source,
        [ProofSignal(source="auditor", passed=True, evidence_refs=("a1",))],
        requested_decision="notify",
        required_sources=("public_key_replay", "auditor"),
    )

    assert receipt_missing.missing_sources == ("receipt",)
    assert replay_missing.missing_sources == ("public_key_replay",)
    assert receipt_missing.surface_id != replay_missing.surface_id


def test_proof_surface_id_changes_for_different_blocking_sources():
    claim_source = ClaimSource(
        claim="Export is blocked by a failed proof source",
        source="receipt panel",
    )
    receipt_blocked = build_proof_surface(
        claim_source,
        [
            ProofSignal(source="receipt", passed=False, evidence_refs=("r1",)),
            ProofSignal(source="auditor", passed=True, evidence_refs=("a1",)),
        ],
        requested_decision="notify",
    )
    auditor_blocked = build_proof_surface(
        claim_source,
        [
            ProofSignal(source="receipt", passed=True, evidence_refs=("r1",)),
            ProofSignal(source="auditor", passed=False, evidence_refs=("a1",)),
        ],
        requested_decision="notify",
    )

    assert receipt_blocked.blocking_sources == ("receipt",)
    assert auditor_blocked.blocking_sources == ("auditor",)
    assert receipt_blocked.surface_id != auditor_blocked.surface_id


def test_surface_to_dict_is_json_safe_and_matches_csl_shape():
    convergence = converge_proofs(
        [ProofSignal(source="receipt", passed=True)],
        requested_decision="notify",
    )
    surface = proof_surface_from_convergence(
        ClaimSource(claim="Receipt can be shown", source="receipt panel"),
        convergence,
        receipt_id="rcpt-1",
    )

    payload = surface.to_dict()
    assert payload["schema_version"] == "0.1.0"
    assert payload["decision"] == "notify"
    assert payload["receipt_export_ready"] is True
    json.dumps(payload, sort_keys=True)
