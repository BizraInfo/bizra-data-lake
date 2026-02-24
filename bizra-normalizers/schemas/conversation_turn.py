"""Unified conversation schema used by BIZRA platform normalizers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FragmentKind(str, Enum):
    """Canonical fragment targets aligned to bizra-memory FragmentKind."""

    PREFERENCE = "Preference"
    GOAL = "Goal"
    EXPERTISE = "Expertise"
    FACT = "Fact"
    STYLE = "Style"
    PATTERN = "Pattern"
    EMOTION = "Emotion"
    RELATIONSHIP = "Relationship"
    TEMPORAL = "Temporal"
    DOMAIN = "Domain"


@dataclass(frozen=True)
class FragmentHint:
    """A parser-derived signal that can be mapped into memory fragments."""

    kind: FragmentKind
    signal: str
    confidence: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.signal.strip():
            raise ValueError("FragmentHint.signal must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("FragmentHint.confidence must be between 0.0 and 1.0")
        if not self.source.strip():
            raise ValueError("FragmentHint.source must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "signal": self.signal,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FragmentHint":
        return cls(
            kind=FragmentKind(str(data["kind"])),
            signal=str(data["signal"]),
            confidence=float(data.get("confidence", 0.0)),
            source=str(data.get("source", "unknown")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ConversationTurn:
    """Provider-normalized conversation turn."""

    provider: str
    conversation_id: str
    turn_id: str
    role: str
    content: str
    timestamp: int
    model: str = ""
    fragment_hints: list[FragmentHint] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("ConversationTurn.provider must be non-empty")
        if not self.conversation_id.strip():
            raise ValueError("ConversationTurn.conversation_id must be non-empty")
        if not self.turn_id.strip():
            raise ValueError("ConversationTurn.turn_id must be non-empty")
        if not self.content.strip():
            raise ValueError("ConversationTurn.content must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "model": self.model,
            "fragment_hints": [hint.to_dict() for hint in self.fragment_hints],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationTurn":
        return cls(
            provider=str(data["provider"]),
            conversation_id=str(data["conversation_id"]),
            turn_id=str(data["turn_id"]),
            role=str(data.get("role", "unknown")),
            content=str(data.get("content", "")),
            timestamp=int(data.get("timestamp", 0)),
            model=str(data.get("model", "")),
            fragment_hints=[
                FragmentHint.from_dict(item)
                for item in (data.get("fragment_hints") or [])
            ],
            metadata=dict(data.get("metadata") or {}),
        )
