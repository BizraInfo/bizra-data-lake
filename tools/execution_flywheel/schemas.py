"""Dataclass schemas for the execution-flywheel kernel.

Stdlib-only. Frozen where safe to keep equality + hashing predictable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Trigger:
    keyword: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.keyword or not isinstance(self.keyword, str):
            raise ValueError("Trigger.keyword must be a non-empty string")


@dataclass
class Pattern:
    pattern_id: str
    name: str
    severity: str
    triggers: list[Trigger] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    guard_actions: list[str] = field(default_factory=list)
    source: list[str] = field(default_factory=list)
    default_decision: str = ""

    REQUIRED_FIELDS = ("pattern_id", "name", "severity")
    VALID_SEVERITIES = ("info", "minor", "major", "high", "critical")
    VALID_DEFAULT_DECISIONS = (
        "",
        "PROCEED",
        "REVALIDATE",
        "ABORT",
        "NEEDS_OPERATOR_CONFIRMATION",
    )

    @classmethod
    def from_dict(cls, data: dict) -> "Pattern":
        if not isinstance(data, dict):
            raise ValueError(f"Pattern source must be a dict, got {type(data).__name__}")
        missing = [f for f in cls.REQUIRED_FIELDS if not data.get(f)]
        if missing:
            raise ValueError(f"Pattern missing required fields: {missing}")
        severity = data["severity"]
        if severity not in cls.VALID_SEVERITIES:
            raise ValueError(
                f"Pattern.severity {severity!r} not in {cls.VALID_SEVERITIES}"
            )
        default_decision = data.get("default_decision") or ""
        if default_decision not in cls.VALID_DEFAULT_DECISIONS:
            raise ValueError(
                f"Pattern.default_decision {default_decision!r} not in {cls.VALID_DEFAULT_DECISIONS}"
            )
        raw_triggers = data.get("triggers") or []
        triggers: list[Trigger] = []
        for t in raw_triggers:
            if isinstance(t, dict):
                triggers.append(
                    Trigger(keyword=str(t["keyword"]), description=str(t.get("description", "")))
                )
            elif isinstance(t, str):
                triggers.append(Trigger(keyword=t))
            else:
                raise ValueError(f"Trigger must be dict or str, got {type(t).__name__}")
        return cls(
            pattern_id=str(data["pattern_id"]),
            name=str(data["name"]),
            severity=str(severity),
            triggers=triggers,
            risks=[str(x) for x in (data.get("risks") or [])],
            guard_actions=[str(x) for x in (data.get("guard_actions") or [])],
            source=[str(x) for x in (data.get("source") or [])],
            default_decision=default_decision,
        )

    def to_dict(self) -> dict:
        out: dict = {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "severity": self.severity,
            "triggers": [asdict(t) for t in self.triggers],
            "risks": list(self.risks),
            "guard_actions": list(self.guard_actions),
            "source": list(self.source),
        }
        if self.default_decision:
            out["default_decision"] = self.default_decision
        return out

    def matches(self, triggers_detected: list[str]) -> bool:
        detected = {t.lower() for t in triggers_detected if isinstance(t, str)}
        return any(t.keyword.lower() in detected for t in self.triggers)


@dataclass
class ActionContext:
    action_type: str
    target_files: list[str] = field(default_factory=list)
    triggers_detected: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardDecision:
    decision: str
    reason: str
    matched_patterns: list[str] = field(default_factory=list)

    VALID_DECISIONS = ("PROCEED", "REVALIDATE", "ABORT", "NEEDS_OPERATOR_CONFIRMATION")

    def __post_init__(self) -> None:
        if self.decision not in self.VALID_DECISIONS:
            raise ValueError(
                f"GuardDecision.decision {self.decision!r} must be in {self.VALID_DECISIONS}"
            )

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "matched_patterns": list(self.matched_patterns),
        }


@dataclass
class PrioritySignal:
    priority: str
    reason: str
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    VALID_PRIORITIES = (
        "SECURITY",
        "PUBLIC_CLAIMS",
        "CI_BASELINE",
        "SUPPLY_CHAIN",
        "RUNTIME_HARDENING",
        "NODE0_ACTIVATION",
        "STOP_AND_LAND",
    )

    def __post_init__(self) -> None:
        if self.priority not in self.VALID_PRIORITIES:
            raise ValueError(
                f"PrioritySignal.priority {self.priority!r} must be in {self.VALID_PRIORITIES}"
            )
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(
                f"PrioritySignal.confidence {self.confidence} must be in [0.0, 1.0]"
            )

    def to_dict(self) -> dict:
        return {
            "priority": self.priority,
            "reason": self.reason,
            "confidence": float(self.confidence),
            "evidence": list(self.evidence),
        }


@dataclass
class FlywheelResult:
    guard: GuardDecision
    priority: PrioritySignal
    explanations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "guard": self.guard.to_dict(),
            "priority": self.priority.to_dict(),
            "explanations": list(self.explanations),
        }
