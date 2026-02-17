"""
BIZRA Guild Module — Collaborative Community Membership
=========================================================

Guilds are thematic communities where BIZRA nodes collaborate
on shared missions within impact domains. Each guild emerges
as an autopoietic unit — its collective Ihsan score determines
its fitness in the larger network ecology.

v1.0.0 — Genesis Guild System

Standing on Giants:
- Ostrom (1990): Polycentric commons governance
- Al-Ghazali (1058-1111): Ihsan as collective excellence
- Darwin (1859): Natural selection among communities
"""

from core.guild.registry import DEFAULT_GUILDS, GuildRegistry
from core.guild.types import (
    Guild,
    GuildJoinResult,
    GuildMember,
    GuildRole,
    GuildStatus,
)

__version__ = "1.0.0"

__all__ = [
    # Types
    "Guild",
    "GuildMember",
    "GuildJoinResult",
    "GuildStatus",
    "GuildRole",
    # Engine
    "GuildRegistry",
    # Constants
    "DEFAULT_GUILDS",
]
