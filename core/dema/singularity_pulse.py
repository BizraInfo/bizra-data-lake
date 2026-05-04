"""BIZRA Singularity Pulse v0.1 contract.

This module is a pure doctrine-to-contract surface. It defines the internal
materialization threshold for Node0 without starting daemons, running missions,
writing receipts, or making AGI/public-launch claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from core.integration.constants import SNR_THRESHOLD_T1_HIGH, UNIFIED_IHSAN_THRESHOLD

SCHEMA_VERSION = "bizra.singularity_pulse.v0.1"


def _freeze_value(value: Any) -> Any:
    """Recursively freeze common containers for inert template payloads."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_value(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


class PulseGateId(str, Enum):
    """The nine gates required before BIZRA can claim materialization."""

    MANIFESTO = "manifesto"
    NODE0_SUBSTRATE = "node0_substrate"
    DEMA_ROLE = "dema_role"
    FATE_BOUNDARY = "fate_boundary"
    RECEIPT_OBJECT = "receipt_object"
    MODEL_FLUIDITY = "model_fluidity"
    RUST_BUS = "rust_bus"
    RUNTIME_EVIDENCE = "runtime_evidence"
    MEMORY_NEXT_ACTION = "memory_next_action"


class PulseGateStatus(str, Enum):
    """Gate state vocabulary for the internal pulse assessment."""

    PASS = "PASS"
    PENDING = "PENDING"
    BLOCKED = "BLOCKED"


class PulseVerdict(str, Enum):
    """Honest threshold language; none of these verdicts claim AGI."""

    INFRASTRUCTURE_INCOMPLETE = "INFRASTRUCTURE_INCOMPLETE"
    SINGULARITY_PULSE_ARMED = "SINGULARITY_PULSE_ARMED"
    MATERIALIZATION_THRESHOLD_REACHED = "MATERIALIZATION_THRESHOLD_REACHED"


@dataclass(frozen=True)
class PulseGate:
    """One threshold gate and the evidence backing its current state."""

    gate_id: PulseGateId
    requirement: str
    status: PulseGateStatus
    evidence_ref: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.gate_id, PulseGateId):
            raise ValueError("gate_id must be a PulseGateId")
        if not isinstance(self.status, PulseGateStatus):
            raise ValueError("status must be a PulseGateStatus")
        if not self.requirement.strip():
            raise ValueError("requirement must be non-empty")

    @property
    def passed(self) -> bool:
        """Return whether this gate is fully satisfied."""

        return self.status is PulseGateStatus.PASS


@dataclass(frozen=True)
class NativeFootprint:
    """The compact BIZRA signature shared by code, receipts, and Dema surfaces."""

    symbolic: str
    technical_loop: tuple[str, ...]
    product_contract: tuple[str, ...]
    signature: str
    artifact_footer: str
    code_footer: str

    def to_dict(self) -> dict[str, object]:
        return {
            "symbolic": self.symbolic,
            "technical_loop": list(self.technical_loop),
            "product_contract": list(self.product_contract),
            "signature": self.signature,
            "artifact_footer": self.artifact_footer,
            "code_footer": self.code_footer,
        }


@dataclass(frozen=True)
class ActivationScope:
    """Bounded first-pulse operating scope."""

    allowed: tuple[str, ...]
    forbidden: tuple[str, ...]

    def permits(self, operation: str) -> bool:
        """Return true only for explicitly allowed operation names."""

        return operation in self.allowed and operation not in self.forbidden

    def to_dict(self) -> dict[str, object]:
        return {"allowed": list(self.allowed), "forbidden": list(self.forbidden)}


@dataclass(frozen=True)
class SingularityPulseAssessment:
    """Pure assessment result for the internal Singularity Pulse threshold."""

    verdict: PulseVerdict
    gates: tuple[PulseGate, ...]
    native_footprint: NativeFootprint
    activation_scope: ActivationScope
    public_message: str
    minimum_snr: float = SNR_THRESHOLD_T1_HIGH
    minimum_ihsan: float = UNIFIED_IHSAN_THRESHOLD
    schema_version: str = SCHEMA_VERSION

    @property
    def materialized(self) -> bool:
        """Return whether runtime receipt and memory gates are both complete."""

        return self.verdict is PulseVerdict.MATERIALIZATION_THRESHOLD_REACHED

    @property
    def blockers(self) -> tuple[PulseGate, ...]:
        """Return non-passing gates."""

        return tuple(gate for gate in self.gates if not gate.passed)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "verdict": self.verdict.value,
            "minimum_snr": self.minimum_snr,
            "minimum_ihsan": self.minimum_ihsan,
            "materialized": self.materialized,
            "public_message": self.public_message,
            "gates": [
                {
                    "id": gate.gate_id.value,
                    "requirement": gate.requirement,
                    "status": gate.status.value,
                    "evidence_ref": gate.evidence_ref,
                }
                for gate in self.gates
            ],
            "blockers": [gate.gate_id.value for gate in self.blockers],
            "native_footprint": self.native_footprint.to_dict(),
            "activation_scope": self.activation_scope.to_dict(),
        }


_GATE_REQUIREMENTS: MappingProxyType[PulseGateId, str] = MappingProxyType(
    {
        PulseGateId.MANIFESTO: "canonical public doctrine exists",
        PulseGateId.NODE0_SUBSTRATE: "LM/model/PAT/Rust Bus green",
        PulseGateId.DEMA_ROLE: "Dema visible bridge defined",
        PulseGateId.FATE_BOUNDARY: "no action without consent/proof",
        PulseGateId.RECEIPT_OBJECT: "identity-bound receipt path exists",
        PulseGateId.MODEL_FLUIDITY: "model broker available",
        PulseGateId.RUST_BUS: "Rust Bus active via PyO3",
        PulseGateId.RUNTIME_EVIDENCE: "first runtime receipt ledger entry exists",
        PulseGateId.MEMORY_NEXT_ACTION: "mission result informs next action",
    }
)


def native_footprint() -> NativeFootprint:
    """Return the canonical BIZRA native footprint."""

    return NativeFootprint(
        symbolic="البذرة / The Seed / human as node",
        technical_loop=("Niyyah", "PAT", "FATE", "Receipt", "URP", "PoI", "Forest"),
        product_contract=(
            "Dema asks: What is your mission?",
            "Dema verifies: What is true?",
            "Dema protects: What must not be violated?",
            "Dema acts: Only with consent.",
            "Dema proves: Here is the receipt.",
        ),
        signature=(
            "Humanity is not the fuel. Humanity is the infrastructure. "
            "Every human is a node. Every node is a seed. "
            "Every verified contribution becomes light for the forest."
        ),
        artifact_footer=(
            "BIZRA · البذرة\n"
            "Node0 · Third Fact Protocol\n"
            "Mission -> Consent -> Proof -> Receipt -> Impact\n"
            "Humanity is not the fuel. Humanity is the infrastructure."
        ),
        code_footer=(
            "# BIZRA Native Footprint\n"
            "# Pillar: Proof / Ihsan / Sovereignty\n"
            "# Law: No action without consent and proof\n"
            "# Receipt: required for verified state transitions"
        ),
    )


def bounded_diagnostic_scope() -> ActivationScope:
    """Return the only scope allowed for the first private pulse."""

    return ActivationScope(
        allowed=(
            "start_bounded_daemon",
            "run_one_diagnostic_mission",
            "capture_evidence_receipt",
            "confirm_daemon_state",
            "write_private_memory",
        ),
        forbidden=(
            "node1_activation",
            "third_fact_public_publish",
            "public_demo",
            "external_provider_routing",
            "economic_token_claim",
            "unbounded_autonomy",
        ),
    )


def identity_bound_receipt_template(
    *,
    mission: str,
    next_admissible_action: str,
    human: str = "sovereign_operator",
    node: str = "Node0",
) -> MappingProxyType[str, Any]:
    """Return an inert template for the first materialization receipt."""

    mission = mission.strip()
    next_admissible_action = next_admissible_action.strip()
    if not mission:
        raise ValueError("mission must be non-empty")
    if not next_admissible_action:
        raise ValueError("next_admissible_action must be non-empty")
    payload: dict[str, Any] = {
        "schema": "bizra.receipt.identity_bound.v1",
        "node": node,
        "human": human,
        "mission": mission,
        "gate": "FATE",
        "ihsan": True,
        "proof": {
            "hash": "<required-after-runtime>",
            "signature": "<required-after-runtime>",
            "replayable": True,
        },
        "next_admissible_action": next_admissible_action,
    }
    return _freeze_value(payload)


def assess_singularity_pulse(
    *,
    manifesto: bool,
    node0_substrate: bool,
    dema_role: bool,
    fate_boundary: bool,
    receipt_object: bool,
    model_fluidity: bool,
    rust_bus: bool,
    runtime_evidence: bool,
    memory_next_action: bool,
    evidence_refs: Mapping[PulseGateId, str] | None = None,
) -> SingularityPulseAssessment:
    """Assess BIZRA's internal materialization threshold.

    The first seven gates define the infrastructure threshold. The final two
    gates require a real bounded diagnostic mission with receipt and memory
    output, so they remain pending until runtime evidence exists.
    """

    raw_states = {
        PulseGateId.MANIFESTO: manifesto,
        PulseGateId.NODE0_SUBSTRATE: node0_substrate,
        PulseGateId.DEMA_ROLE: dema_role,
        PulseGateId.FATE_BOUNDARY: fate_boundary,
        PulseGateId.RECEIPT_OBJECT: receipt_object,
        PulseGateId.MODEL_FLUIDITY: model_fluidity,
        PulseGateId.RUST_BUS: rust_bus,
        PulseGateId.RUNTIME_EVIDENCE: runtime_evidence,
        PulseGateId.MEMORY_NEXT_ACTION: memory_next_action,
    }
    refs = evidence_refs or MappingProxyType({})
    gates = tuple(
        PulseGate(
            gate_id=gate_id,
            requirement=_GATE_REQUIREMENTS[gate_id],
            status=PulseGateStatus.PASS if passed else PulseGateStatus.PENDING,
            evidence_ref=refs.get(gate_id, ""),
        )
        for gate_id, passed in raw_states.items()
    )

    infrastructure_ready = all(
        raw_states[gate_id]
        for gate_id in (
            PulseGateId.MANIFESTO,
            PulseGateId.NODE0_SUBSTRATE,
            PulseGateId.DEMA_ROLE,
            PulseGateId.FATE_BOUNDARY,
            PulseGateId.RECEIPT_OBJECT,
            PulseGateId.MODEL_FLUIDITY,
            PulseGateId.RUST_BUS,
        )
    )
    materialized = infrastructure_ready and runtime_evidence and memory_next_action

    if materialized:
        verdict = PulseVerdict.MATERIALIZATION_THRESHOLD_REACHED
        public_message = (
            "BIZRA Node0 has produced its first verified sovereign action loop. "
            "The seed has pulsed."
        )
    elif infrastructure_ready:
        verdict = PulseVerdict.SINGULARITY_PULSE_ARMED
        public_message = (
            "Singularity Pulse Armed: infrastructure threshold reached; "
            "materialization still requires one verified runtime loop."
        )
    else:
        verdict = PulseVerdict.INFRASTRUCTURE_INCOMPLETE
        public_message = (
            "Infrastructure threshold incomplete; do not claim materialization."
        )

    return SingularityPulseAssessment(
        verdict=verdict,
        gates=gates,
        native_footprint=native_footprint(),
        activation_scope=bounded_diagnostic_scope(),
        public_message=public_message,
    )
