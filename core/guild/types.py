"""
Guild Data Types — Community Organization Primitives
=====================================================

Standing on Giants:
- Ostrom (1990): Polycentric governance structures
- Shannon (1948): SNR as membership quality gate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


class GuildStatus(str, Enum):
    """Status of a guild."""

    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"


@dataclass
class GuildMember:
    """A node's membership in a guild."""

    node_id: str
    guild_id: str
    joined_at: str = ""
    role: str = "member"  # member | elder | founder
    ihsan_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.joined_at:
            self.joined_at = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )


@dataclass
class Guild:
    """A domain-aligned community of BIZRA nodes."""

    guild_id: str
    name: str
    description: str = ""
    status: GuildStatus = GuildStatus.ACTIVE
    members: List[GuildMember] = field(default_factory=list)
    created_at: str = ""
    online_count: int = 0

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )

    @property
    def member_count(self) -> int:
        return len(self.members)

    def has_member(self, node_id: str) -> bool:
        return any(m.node_id == node_id for m in self.members)


@dataclass
class GuildJoinResult:
    """Result of attempting to join a guild."""

    success: bool = False
    guild: Optional[Guild] = None
    member: Optional[GuildMember] = None
    message: str = ""
