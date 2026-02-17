"""
BIZRA Guild Types — Collaborative Community Data Structures
=============================================================

Guilds are thematic communities where nodes collaborate on shared
missions. Each guild tracks member Ihsan scores, forming emergent
fitness landscapes — guilds with higher collective Ihsan attract
stronger nodes, creating natural selection among communities.

Standing on Giants:
- Ostrom (1990): Common-pool resource governance
- Al-Ghazali (1058-1111): Ihsan as community excellence
- Nakamoto (2008): Decentralized membership (no gatekeeper)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class GuildStatus(Enum):
    """Guild lifecycle states."""

    PENDING = "pending"  # Awaiting minimum member threshold
    ACTIVE = "active"  # Fully operational
    SUSPENDED = "suspended"  # Temporarily halted (governance action)


class GuildRole(Enum):
    """Member roles within a guild."""

    MEMBER = "member"  # Standard participant
    ELDER = "elder"  # Experienced contributor
    STEWARD = "steward"  # Guild coordinator (elected)


@dataclass(frozen=True)
class GuildMember:
    """A node's membership in a guild."""

    node_id: str
    guild_id: str
    joined_at: str
    role: GuildRole = GuildRole.MEMBER
    ihsan_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "guild_id": self.guild_id,
            "joined_at": self.joined_at,
            "role": self.role.value,
            "ihsan_score": self.ihsan_score,
        }


@dataclass
class Guild:
    """A thematic community of BIZRA nodes."""

    guild_id: str
    name: str
    description: str
    members: List[GuildMember] = field(default_factory=list)
    status: GuildStatus = GuildStatus.ACTIVE
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    @property
    def member_count(self) -> int:
        return len(self.members)

    @property
    def online_count(self) -> int:
        """Simulated online count — in production, tracked via heartbeat."""
        return max(1, len(self.members))

    @property
    def mean_ihsan(self) -> float:
        if not self.members:
            return 0.0
        return sum(m.ihsan_score for m in self.members) / len(self.members)

    def has_member(self, node_id: str) -> bool:
        return any(m.node_id == node_id for m in self.members)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "guild_id": self.guild_id,
            "name": self.name,
            "description": self.description,
            "member_count": self.member_count,
            "online_count": self.online_count,
            "mean_ihsan": round(self.mean_ihsan, 4),
            "status": self.status.value,
            "created_at": self.created_at,
        }


@dataclass
class GuildJoinResult:
    """Result of a guild join operation."""

    success: bool
    guild: Optional[Guild] = None
    member: Optional[GuildMember] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "guild": self.guild.to_dict() if self.guild else None,
            "member": self.member.to_dict() if self.member else None,
            "message": self.message,
        }
