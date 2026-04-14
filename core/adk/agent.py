"""Agent base class — the core of BIZRA-ADK.

Every agent:
- has an immutable charter (hashed at construction)
- declares a governance class
- can only produce output via self.draft() or self.refuse()
- is automatically FATE-gated and receipt-sealed
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from core.adk.mission import (
    GovernanceClass,
    Mission,
)
from core.adk.tools import get_tools


@dataclass(frozen=True)
class AgentIdentity:
    name: str
    charter_hash: str
    governance_class: GovernanceClass
    frozen: bool = False


@dataclass
class AgentResult:
    """The output of an agent run — either a receipted answer or a block."""
    success: bool
    content: str
    evidence_refs: list[str]
    ihsan_score: float
    verdict: str  # PASS | BLOCKED_BY_*
    reason: str
    receipt: Optional[object] = None  # core.proof_engine.receipt.Receipt
    loop_proof: Optional[object] = None  # core.proof_engine.loop_proof.LoopProof
    mission_id: str = ""


class _DraftOutput:
    """Intermediate: content + evidence before FATE evaluation."""
    __slots__ = ("content", "evidence_refs")

    def __init__(self, content: str, evidence_refs: list[str]):
        self.content = content
        self.evidence_refs = evidence_refs


class _RefuseOutput:
    """Intermediate: agent honestly refuses the mission."""
    __slots__ = ("reason",)

    def __init__(self, reason: str):
        self.reason = reason


def charter(text: str):
    """Decorator that binds an immutable charter to an Agent class."""
    def decorator(cls):
        cls._charter_text = text.strip()
        cls._charter_hash = hashlib.blake2b(
            text.strip().encode(), digest_size=32
        ).hexdigest()
        return cls
    return decorator


class Agent(ABC):
    """Base class for all BIZRA agents.

    Subclasses must:
    - be decorated with @charter(...)
    - set `name` and `governance_class`
    - implement `async def act(self, mission) -> self.draft(...) | self.refuse(...)`
    """
    name: str = "UnnamedAgent"
    governance_class: str = "PAT"
    model: str = "gemma4:26b-bizra-16k"

    _charter_text: str = ""
    _charter_hash: str = ""
    _active_mission: Optional[Mission] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Charter check is deferred to __init__ because @charter decorator
        # runs after __init_subclass__ in the class body evaluation order.

    @property
    def identity(self) -> AgentIdentity:
        return AgentIdentity(
            name=self.name,
            charter_hash=self._charter_hash,
            governance_class=GovernanceClass(self.governance_class),
        )

    @property
    def tools(self) -> list[str]:
        return get_tools(self)

    def draft(self, content: str, evidence: list | None = None) -> _DraftOutput:
        """Produce a draft output with evidence. FATE will evaluate it."""
        refs = []
        if evidence:
            for e in evidence:
                if isinstance(e, str):
                    refs.append(e)
                elif hasattr(e, "uri"):
                    refs.append(e.uri)
                elif hasattr(e, "ref"):
                    refs.append(e.ref)
                else:
                    refs.append(str(e))
        return _DraftOutput(content=content, evidence_refs=refs)

    def refuse(self, reason: str) -> _RefuseOutput:
        """Honestly refuse the mission with a reason."""
        return _RefuseOutput(reason=reason)

    @abstractmethod
    async def act(self, mission: Mission) -> _DraftOutput | _RefuseOutput:
        """Execute the mission. Must return self.draft() or self.refuse()."""
        ...

    async def run(self, mission: Mission) -> AgentResult:
        """Full 7-step lifecycle: NIYYAH -> BAYYINAH -> ... -> RETROSPECTIVE.

        This is the only public entry point. It calls self.act() internally
        and wraps the result in the full receipt + FATE + loop proof pipeline.
        """
        from core.adk.runner import execute_agent_lifecycle
        return await execute_agent_lifecycle(self, mission)
