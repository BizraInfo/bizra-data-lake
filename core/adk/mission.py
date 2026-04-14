"""Mission — the unit of work for a BIZRA agent."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class GovernanceClass(str, Enum):
    PAT = "PAT"
    SAT = "SAT"
    FROZEN = "FROZEN"
    SOVEREIGN = "SOVEREIGN"


@dataclass(frozen=True)
class Budget:
    max_tokens: int = 4096
    max_wall_seconds: int = 120
    max_tool_calls: int = 10
    max_evidence_fetches: int = 20

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.max_wall_seconds <= 0:
            raise ValueError(f"max_wall_seconds must be positive, got {self.max_wall_seconds}")
        if self.max_tool_calls <= 0:
            raise ValueError(f"max_tool_calls must be positive, got {self.max_tool_calls}")


DEFAULT_BUDGET = Budget()


@dataclass
class Mission:
    question: str
    governance_class: GovernanceClass = GovernanceClass.PAT
    requester: str = "human"
    budget: Budget = field(default_factory=lambda: DEFAULT_BUDGET)
    allow_external_unverified: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _tokens_used: int = field(default=0, repr=False, init=False)
    _tool_calls_used: int = field(default=0, repr=False, init=False)

    def consume_tokens(self, n: int) -> None:
        self._tokens_used += n
        if self._tokens_used > self.budget.max_tokens:
            raise BudgetExhausted("tokens", self._tokens_used, self.budget.max_tokens)

    def consume_tool_call(self) -> None:
        self._tool_calls_used += 1
        if self._tool_calls_used > self.budget.max_tool_calls:
            raise BudgetExhausted("tool_calls", self._tool_calls_used, self.budget.max_tool_calls)


class BudgetExhausted(Exception):
    def __init__(self, kind: str, used: int, limit: int):
        self.kind = kind
        self.used = used
        self.limit = limit
        super().__init__(f"Budget exhausted: {kind} ({used}/{limit})")
