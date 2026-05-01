"""Semantic transducer contract for Node0 / DEMA.

This module keeps LLM or parser output outside the trusted system until it is
validated into a Claim. It does not execute tools, start daemons, or write
runtime state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol
from uuid import UUID

from core.integration.constants import CONSTITUTIONAL_GINI_THRESHOLD

SCHEMA_VERSION = "0.1.0"

_CONTROL_EVIDENCE_KEYS = frozenset(
    {
        "authority",
        "decision",
        "evidence_weight",
        "parser_id",
        "proof",
        "truth",
        "verdict",
    }
)


class GateVerdict(str, Enum):
    """Deterministic FATE verdicts for the v0.1 claim boundary."""

    PERMIT = "PERMIT"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"


class IntentType(str, Enum):
    """Known intent vocabulary for the v0.1 semantic boundary."""

    ORGANIZE_FILES = "intent.organize_files"
    SUMMARIZE_DOCUMENT = "intent.summarize_document"
    REPORT_STATUS = "intent.report_status"
    RELIEF_START_PRECHECK = "intent.relief_start_precheck"
    UNRESOLVED = "intent.unresolved"


class ResourceType(str, Enum):
    """Resource classes that a future executor must scope-check."""

    FILESYSTEM_READ = "filesystem.read"
    FILESYSTEM_WRITE = "filesystem.write"
    NETWORK_READ = "network.read"
    MODEL_INFERENCE = "model.inference"
    RECEIPT_WRITE = "receipt.write"


class IntentParser(Protocol):
    """Untrusted semantic parser interface.

    Implementations may use an LLM, CLI form, or deterministic parser. The
    parser returns RawParsedClaim only; parser identity is injected later by
    the system wrapper during validate_raw_claim(...).
    """

    async def parse(self, text: str) -> "RawParsedClaim":
        """Parse operator text into untrusted structured output."""
        ...


def _freeze_value(value: Any) -> Any:
    """Recursively freeze common container types."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_value(v) for k, v in value.items()})
    if isinstance(value, tuple):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, list):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(v) for v in value)
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({str(k): _freeze_value(v) for k, v in value.items()})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ResourceScope:
    """Allowed resource classes for a claim or step group."""

    allowed: frozenset[ResourceType]

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed", frozenset(self.allowed))
        invalid = [item for item in self.allowed if not isinstance(item, ResourceType)]
        if invalid:
            raise ValueError(f"invalid resource type(s): {invalid!r}")

    @classmethod
    def of(cls, *resources: ResourceType) -> "ResourceScope":
        """Construct a scope from resource enum members."""
        return cls(frozenset(resources))

    def permits(self, resource_type: ResourceType) -> bool:
        return resource_type in self.allowed

    def contains_step(self, step: "StepDescriptor") -> bool:
        return self.permits(step.resource_type)

    def contains_steps(self, steps: Iterable["StepDescriptor"]) -> bool:
        return all(self.contains_step(step) for step in steps)

    def is_subset_of(self, other: "ResourceScope") -> bool:
        return self.allowed.issubset(other.allowed)


@dataclass(frozen=True)
class StepDescriptor:
    """Proposed future executor step.

    Step descriptors are inert data. They are not executable commands.
    """

    tool_id: str
    resource_type: ResourceType
    parameters: Mapping[str, object] = field(default_factory=dict)
    produces_claim: bool = False

    def __post_init__(self) -> None:
        if not self.tool_id.strip():
            raise ValueError("tool_id must be non-empty")
        if not isinstance(self.resource_type, ResourceType):
            raise ValueError("resource_type must be a ResourceType")
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))


@dataclass(frozen=True)
class RawParsedClaim:
    """Untrusted parser output.

    This type may contain model-originated data. It is never a trusted Claim.
    """

    intent_type: str
    evidence: Mapping[str, object]
    proposed_steps: tuple[StepDescriptor, ...]
    requested_scope: ResourceScope
    semantic_summary: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.requested_scope, ResourceScope):
            raise ValueError("requested_scope must be a ResourceScope")
        invalid_steps = [
            step for step in self.proposed_steps if not isinstance(step, StepDescriptor)
        ]
        if invalid_steps:
            raise ValueError("proposed_steps must contain StepDescriptor values")
        object.__setattr__(self, "evidence", _freeze_mapping(self.evidence))
        object.__setattr__(self, "proposed_steps", tuple(self.proposed_steps))


@dataclass(frozen=True)
class Claim:
    """Validated trust-perimeter object.

    Claim records are immutable value records for gate input. They are not
    intended to be hash keys.
    """

    __hash__ = None

    mission_id: UUID
    intent_type: IntentType
    evidence: Mapping[str, object]
    proposed_steps: tuple[StepDescriptor, ...]
    scope: ResourceScope
    timestamp: datetime
    parser_id: str
    parent_mission_id: UUID | None = None
    semantic_summary: str = ""
    schema_version: str = SCHEMA_VERSION
    evidence_weight: float = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if not self.parser_id.strip():
            raise ValueError("parser_id must be injected by the system wrapper")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        object.__setattr__(self, "evidence", _freeze_mapping(self.evidence))
        object.__setattr__(self, "proposed_steps", tuple(self.proposed_steps))
        object.__setattr__(
            self,
            "evidence_weight",
            compute_evidence_weight(self.evidence),
        )


@dataclass(frozen=True)
class ConstitutionalPolicy:
    """Versioned policy knobs used by the v0.1 deterministic gate."""

    version: str
    ihsan_floor: float
    unresolved_verdict: GateVerdict = GateVerdict.ESCALATE
    zann_zero: bool = True
    riba_zero: bool = True
    gini_threshold: float = CONSTITUTIONAL_GINI_THRESHOLD

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("policy version must be non-empty")
        if not 0.0 <= self.ihsan_floor <= 1.0:
            raise ValueError("ihsan_floor must be between 0 and 1")
        if self.unresolved_verdict is not GateVerdict.ESCALATE:
            raise ValueError("intent.unresolved must always ESCALATE")
        if not 0.0 <= self.gini_threshold <= 1.0:
            raise ValueError("gini_threshold must be between 0 and 1")


@dataclass(frozen=True)
class GateDecision:
    """Pure deterministic FATE gate result."""

    verdict: GateVerdict
    rule_id: str
    evidence_weight: float
    gate_version: str
    reason_code: str
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class MissionReceiptDescriptor:
    """Receipt descriptor that avoids claiming correctness."""

    mission_id: UUID
    claim_intent: IntentType
    gate_verdict: GateVerdict
    process_integrity_statement: str
    correctness_claimed: bool = False
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        lowered = self.process_integrity_statement.lower()
        forbidden = ("proves correctness", "proves truth", "correct outcome")
        if self.correctness_claimed or any(token in lowered for token in forbidden):
            raise ValueError("receipt descriptors cannot claim correctness")


@dataclass(frozen=True)
class SemanticSurface:
    """Human-facing explanation surface, not a source of truth."""

    text: str
    source: str
    trust_label: str = "UNTRUSTED_EXPLANATORY"
    source_of_truth: bool = False
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.source_of_truth:
            raise ValueError("SemanticSurface cannot be a source of truth")
        if self.trust_label != "UNTRUSTED_EXPLANATORY":
            raise ValueError("SemanticSurface must be labeled untrusted/explanatory")


def compute_evidence_weight(evidence: Mapping[str, object]) -> float:
    """Compute deterministic v0.1 evidence weight.

    TODO(BIZRA-STC-v0.2): replace this placeholder with a policy-backed
    evidence scoring function that validates evidence refs and receipt/proof
    types. This version deliberately ignores model-supplied control keys.
    """
    material_keys = [
        key
        for key, value in evidence.items()
        if key not in _CONTROL_EVIDENCE_KEYS and _is_material_evidence_value(value)
    ]
    if not material_keys:
        return 0.0
    return min(1.0, len(material_keys) / 5.0)


def _parse_intent(value: str) -> IntentType:
    try:
        return IntentType(value)
    except ValueError:
        return IntentType.UNRESOLVED


def _is_material_evidence_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return len(value) > 0
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value) > 0
    return True


def validate_raw_claim(
    raw: RawParsedClaim,
    *,
    mission_id: UUID,
    parser_id: str,
    parent: Claim | None = None,
    timestamp: datetime | None = None,
) -> Claim:
    """Validate untrusted parser output into a trusted Claim."""
    if not parser_id.strip():
        raise ValueError("parser_id must be injected by the system wrapper")
    if not isinstance(raw.requested_scope, ResourceScope):
        raise ValueError("requested_scope must be a ResourceScope")
    invalid_steps = [
        step for step in raw.proposed_steps if not isinstance(step, StepDescriptor)
    ]
    if invalid_steps:
        raise ValueError("proposed_steps must contain StepDescriptor values")
    if not raw.requested_scope.contains_steps(raw.proposed_steps):
        raise ValueError("step_scope_exceeds_claim_scope")
    if parent is not None and not raw.requested_scope.is_subset_of(parent.scope):
        raise ValueError("sub_claim_scope_expands_parent")

    return Claim(
        mission_id=mission_id,
        intent_type=_parse_intent(raw.intent_type),
        evidence=raw.evidence,
        proposed_steps=raw.proposed_steps,
        scope=raw.requested_scope,
        timestamp=timestamp or _utc_now(),
        parser_id=parser_id,
        parent_mission_id=parent.mission_id if parent else None,
        semantic_summary=raw.semantic_summary,
    )


def fate_gate(claim: Claim, policy: ConstitutionalPolicy) -> GateDecision:
    """Return a pure deterministic gate decision for a validated Claim."""
    if claim.intent_type is IntentType.UNRESOLVED:
        return GateDecision(
            verdict=GateVerdict.ESCALATE,
            rule_id="intent.unresolved",
            evidence_weight=claim.evidence_weight,
            gate_version=policy.version,
            reason_code="INTENT_REQUIRES_HUMAN_REVIEW",
        )

    if claim.evidence_weight < policy.ihsan_floor:
        return GateDecision(
            verdict=GateVerdict.ESCALATE,
            rule_id="ihsan.floor",
            evidence_weight=claim.evidence_weight,
            gate_version=policy.version,
            reason_code="EVIDENCE_BELOW_IHSAN_FLOOR",
        )

    return GateDecision(
        verdict=GateVerdict.PERMIT,
        rule_id="default.permit",
        evidence_weight=claim.evidence_weight,
        gate_version=policy.version,
        reason_code="CLAIM_ADMISSIBLE",
    )


def describe_receipt_process(
    claim: Claim, decision: GateDecision
) -> MissionReceiptDescriptor:
    """Build a process-integrity-only receipt descriptor."""
    return MissionReceiptDescriptor(
        mission_id=claim.mission_id,
        claim_intent=claim.intent_type,
        gate_verdict=decision.verdict,
        process_integrity_statement=(
            "This descriptor records claim validation and gate evaluation "
            "process integrity only; it does not prove correctness, optimality, "
            "or external truth."
        ),
    )


def build_semantic_surface(text: str, *, source: str) -> SemanticSurface:
    """Build an untrusted explanatory surface for a human reader."""
    return SemanticSurface(text=text, source=source)


__all__ = [
    "SCHEMA_VERSION",
    "Claim",
    "ConstitutionalPolicy",
    "GateDecision",
    "GateVerdict",
    "IntentParser",
    "IntentType",
    "MissionReceiptDescriptor",
    "RawParsedClaim",
    "ResourceScope",
    "ResourceType",
    "SemanticSurface",
    "StepDescriptor",
    "build_semantic_surface",
    "compute_evidence_weight",
    "describe_receipt_process",
    "fate_gate",
    "validate_raw_claim",
]
