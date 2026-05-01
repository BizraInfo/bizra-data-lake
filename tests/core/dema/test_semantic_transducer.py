"""Semantic transducer trust-boundary tests."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from core.dema.semantic_transducer import (
    ConstitutionalPolicy,
    GateVerdict,
    IntentType,
    RawParsedClaim,
    ResourceScope,
    ResourceType,
    StepDescriptor,
    build_semantic_surface,
    describe_receipt_process,
    fate_gate,
    validate_raw_claim,
)
from core.integration.constants import CONSTITUTIONAL_GINI_THRESHOLD


def _scope(*resources: ResourceType) -> ResourceScope:
    return ResourceScope.of(*resources)


def _step(resource_type: ResourceType = ResourceType.FILESYSTEM_READ) -> StepDescriptor:
    return StepDescriptor(
        tool_id="local.tool",
        resource_type=resource_type,
        parameters={"path": "~/Downloads", "nested": {"mutable": ["before"]}},
    )


def _raw(
    *,
    intent_type: str = IntentType.ORGANIZE_FILES.value,
    evidence: dict[str, object] | None = None,
    scope: ResourceScope | None = None,
    steps: tuple[StepDescriptor, ...] | None = None,
) -> RawParsedClaim:
    return RawParsedClaim(
        intent_type=intent_type,
        evidence=evidence or {"operator_request": "organize Downloads"},
        proposed_steps=steps or (_step(),),
        requested_scope=scope or _scope(ResourceType.FILESYSTEM_READ),
        semantic_summary="untrusted model wording",
    )


def _policy(floor: float = 0.2) -> ConstitutionalPolicy:
    return ConstitutionalPolicy(version="semantic-transducer-v0.1", ihsan_floor=floor)


def test_raw_parsed_claim_cannot_set_parser_id_as_authority():
    raw = _raw(evidence={"parser_id": "model-supplied", "operator_request": "x"})

    claim = validate_raw_claim(raw, mission_id=uuid4(), parser_id="system.wrapper")

    assert claim.parser_id == "system.wrapper"
    assert claim.evidence["parser_id"] == "model-supplied"


def test_evidence_weight_ignores_model_supplied_score():
    raw = _raw(evidence={"evidence_weight": 0.99, "operator_request": "x"})

    claim = validate_raw_claim(raw, mission_id=uuid4(), parser_id="system.wrapper")

    assert claim.evidence_weight == pytest.approx(0.2)
    assert claim.evidence_weight != 0.99


def test_empty_frozen_collections_do_not_inflate_evidence_weight():
    raw = _raw(evidence={"empty_set": set(), "empty_frozenset": frozenset()})

    claim = validate_raw_claim(raw, mission_id=uuid4(), parser_id="system.wrapper")

    assert claim.evidence["empty_set"] == frozenset()
    assert claim.evidence_weight == pytest.approx(0.0)


def test_nested_claim_data_is_sealed_against_mutation():
    evidence = {"operator_request": "x", "nested": {"items": ["a"]}}
    params = {"path": "~/Downloads"}
    step = StepDescriptor(
        tool_id="local.tool",
        resource_type=ResourceType.FILESYSTEM_READ,
        parameters=params,
    )
    raw = _raw(evidence=evidence, steps=(step,))

    claim = validate_raw_claim(raw, mission_id=uuid4(), parser_id="system.wrapper")

    evidence["operator_request"] = "mutated"
    evidence["nested"]["items"].append("b")
    params["path"] = "/tmp"

    assert claim.evidence["operator_request"] == "x"
    assert claim.evidence["nested"]["items"] == ("a",)
    assert claim.proposed_steps[0].parameters["path"] == "~/Downloads"
    with pytest.raises(TypeError):
        claim.evidence["new"] = "blocked"  # type: ignore[index]


def test_claim_is_not_a_hash_key_contract():
    claim = validate_raw_claim(_raw(), mission_id=uuid4(), parser_id="system.wrapper")

    with pytest.raises(TypeError):
        hash(claim)


def test_unresolved_intent_always_escalates():
    raw = _raw(intent_type="intent.model_invented", evidence={"a": 1, "b": 2})
    claim = validate_raw_claim(raw, mission_id=uuid4(), parser_id="system.wrapper")

    decision = fate_gate(claim, _policy(floor=0.0))

    assert claim.intent_type is IntentType.UNRESOLVED
    assert decision.verdict is GateVerdict.ESCALATE
    assert decision.rule_id == "intent.unresolved"


def test_policy_rejects_non_escalating_unresolved_verdict():
    with pytest.raises(ValueError, match="intent.unresolved must always ESCALATE"):
        ConstitutionalPolicy(
            version="bad-policy",
            ihsan_floor=0.0,
            unresolved_verdict=GateVerdict.PERMIT,
        )


def test_policy_exposes_future_constitutional_stubs():
    policy = _policy()

    assert policy.zann_zero is True
    assert policy.riba_zero is True
    assert policy.gini_threshold == pytest.approx(CONSTITUTIONAL_GINI_THRESHOLD)


def test_gate_is_deterministic_for_same_claim_and_policy():
    timestamp = datetime(2026, 5, 1, tzinfo=timezone.utc)
    claim = validate_raw_claim(
        _raw(evidence={"a": 1, "b": 2, "c": 3}),
        mission_id=uuid4(),
        parser_id="system.wrapper",
        timestamp=timestamp,
    )
    policy = _policy(floor=0.4)

    left = fate_gate(claim, policy)
    right = fate_gate(claim, policy)

    assert left == right
    assert left.verdict is GateVerdict.PERMIT


def test_scope_containment_rejects_out_of_scope_step():
    raw = _raw(
        scope=_scope(ResourceType.FILESYSTEM_READ),
        steps=(_step(ResourceType.FILESYSTEM_WRITE),),
    )

    with pytest.raises(ValueError, match="step_scope_exceeds_claim_scope"):
        validate_raw_claim(raw, mission_id=uuid4(), parser_id="system.wrapper")


def test_malformed_raw_scope_gets_controlled_validation_error():
    raw = object.__new__(RawParsedClaim)
    object.__setattr__(raw, "intent_type", IntentType.ORGANIZE_FILES.value)
    object.__setattr__(raw, "evidence", {})
    object.__setattr__(raw, "proposed_steps", (_step(),))
    object.__setattr__(raw, "requested_scope", "filesystem.read")
    object.__setattr__(raw, "semantic_summary", "")

    with pytest.raises(ValueError, match="requested_scope must be a ResourceScope"):
        validate_raw_claim(raw, mission_id=uuid4(), parser_id="system.wrapper")


def test_sub_claim_scope_expansion_is_rejected():
    parent = validate_raw_claim(
        _raw(scope=_scope(ResourceType.FILESYSTEM_READ)),
        mission_id=uuid4(),
        parser_id="system.wrapper",
    )
    child_raw = _raw(
        scope=_scope(ResourceType.FILESYSTEM_READ, ResourceType.FILESYSTEM_WRITE),
        steps=(_step(ResourceType.FILESYSTEM_READ),),
    )

    with pytest.raises(ValueError, match="sub_claim_scope_expands_parent"):
        validate_raw_claim(
            child_raw,
            mission_id=uuid4(),
            parser_id="system.wrapper",
            parent=parent,
        )


def test_receipt_descriptor_does_not_claim_correctness():
    claim = validate_raw_claim(_raw(), mission_id=uuid4(), parser_id="system.wrapper")
    decision = fate_gate(claim, _policy())

    descriptor = describe_receipt_process(claim, decision)

    assert descriptor.correctness_claimed is False
    assert "does not prove correctness" in descriptor.process_integrity_statement
    with pytest.raises(ValueError, match="cannot claim correctness"):
        descriptor.__class__(
            mission_id=claim.mission_id,
            claim_intent=claim.intent_type,
            gate_verdict=decision.verdict,
            process_integrity_statement="This proves correctness.",
            correctness_claimed=True,
        )


def test_semantic_summary_is_untrusted_explanatory_surface():
    surface = build_semantic_surface(
        "I can explain the gate decision.",
        source="llm.summary",
    )

    assert surface.trust_label == "UNTRUSTED_EXPLANATORY"
    assert surface.source_of_truth is False
    with pytest.raises(ValueError, match="cannot be a source of truth"):
        surface.__class__(
            text="trusted text",
            source="llm.summary",
            source_of_truth=True,
        )
