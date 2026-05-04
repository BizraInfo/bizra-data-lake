"""Singularity Pulse contract tests."""

from __future__ import annotations

import pytest

from core.dema.singularity_pulse import (
    PulseGateId,
    PulseVerdict,
    assess_singularity_pulse,
    bounded_diagnostic_scope,
    identity_bound_receipt_template,
    native_footprint,
)
from core.integration.constants import SNR_THRESHOLD_T1_HIGH, UNIFIED_IHSAN_THRESHOLD


def _assess_ready_infrastructure(
    *,
    runtime_evidence: bool = False,
    memory_next_action: bool = False,
):
    return assess_singularity_pulse(
        manifesto=True,
        node0_substrate=True,
        dema_role=True,
        fate_boundary=True,
        receipt_object=True,
        model_fluidity=True,
        rust_bus=True,
        runtime_evidence=runtime_evidence,
        memory_next_action=memory_next_action,
        evidence_refs={
            PulseGateId.MANIFESTO: "docs/dema-cli-manifesto-v1.md",
            PulseGateId.RUST_BUS: "scripts/node0_rust_bus_bootstrap.sh",
        },
    )


def test_ready_infrastructure_is_armed_not_materialized():
    assessment = _assess_ready_infrastructure()

    assert assessment.verdict is PulseVerdict.SINGULARITY_PULSE_ARMED
    assert assessment.materialized is False
    assert {gate.gate_id for gate in assessment.blockers} == {
        PulseGateId.RUNTIME_EVIDENCE,
        PulseGateId.MEMORY_NEXT_ACTION,
    }
    assert "AGI is alive" not in assessment.public_message
    assert assessment.minimum_snr == pytest.approx(SNR_THRESHOLD_T1_HIGH)
    assert assessment.minimum_ihsan == pytest.approx(UNIFIED_IHSAN_THRESHOLD)


def test_materialization_requires_runtime_receipt_and_memory_next_action():
    missing_memory = _assess_ready_infrastructure(
        runtime_evidence=True,
        memory_next_action=False,
    )
    materialized = _assess_ready_infrastructure(
        runtime_evidence=True,
        memory_next_action=True,
    )

    assert missing_memory.verdict is PulseVerdict.SINGULARITY_PULSE_ARMED
    assert materialized.verdict is PulseVerdict.MATERIALIZATION_THRESHOLD_REACHED
    assert materialized.materialized is True
    assert materialized.blockers == ()
    assert "first verified sovereign action loop" in materialized.public_message


def test_incomplete_infrastructure_blocks_pulse_language():
    assessment = assess_singularity_pulse(
        manifesto=True,
        node0_substrate=True,
        dema_role=True,
        fate_boundary=True,
        receipt_object=True,
        model_fluidity=True,
        rust_bus=False,
        runtime_evidence=False,
        memory_next_action=False,
    )

    assert assessment.verdict is PulseVerdict.INFRASTRUCTURE_INCOMPLETE
    assert PulseGateId.RUST_BUS in {gate.gate_id for gate in assessment.blockers}
    assert "do not claim materialization" in assessment.public_message


def test_native_footprint_exposes_artifact_and_code_signatures():
    footprint = native_footprint()

    assert "Every human is a node" in footprint.signature
    assert "Mission -> Consent -> Proof -> Receipt -> Impact" in (
        footprint.artifact_footer
    )
    assert "No action without consent and proof" in footprint.code_footer
    assert footprint.technical_loop == (
        "Niyyah",
        "PAT",
        "FATE",
        "Receipt",
        "URP",
        "PoI",
        "Forest",
    )


def test_bounded_diagnostic_scope_forbids_public_or_expansive_actions():
    scope = bounded_diagnostic_scope()

    assert scope.permits("run_one_diagnostic_mission") is True
    assert scope.permits("node1_activation") is False
    assert scope.permits("public_demo") is False
    assert scope.permits("external_provider_routing") is False
    assert scope.permits("economic_token_claim") is False


def test_identity_bound_receipt_template_is_inert_and_truthful():
    template = identity_bound_receipt_template(
        mission="Validate Node0 bounded diagnostic loop",
        next_admissible_action="verify receipt and daemon state",
    )

    assert template["schema"] == "bizra.receipt.identity_bound.v1"
    assert template["node"] == "Node0"
    assert template["gate"] == "FATE"
    assert template["proof"]["replayable"] is True
    assert template["proof"]["hash"] == "<required-after-runtime>"
    assert template["next_admissible_action"] == "verify receipt and daemon state"
    with pytest.raises(TypeError):
        template["mission"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        template["proof"]["hash"] = "mutated"  # type: ignore[index]


def test_empty_receipt_template_fields_fail_closed():
    with pytest.raises(ValueError, match="mission must be non-empty"):
        identity_bound_receipt_template(
            mission=" ",
            next_admissible_action="verify receipt",
        )

    with pytest.raises(ValueError, match="next_admissible_action must be non-empty"):
        identity_bound_receipt_template(
            mission="Validate Node0 bounded diagnostic loop",
            next_admissible_action=" ",
        )
