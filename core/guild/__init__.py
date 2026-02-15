"""
BIZRA Guild System — Community Organization Layer
===================================================
Guilds are domain-aligned communities where BIZRA nodes collaborate
on impact missions. Each guild focuses on a sector: agriculture,
healthcare, education, energy, finance.

Standing on Giants:
- Ostrom (1990): Commons governance without centralized authority
- McGonigal (2011): Games and community organization for social good
- Al-Ghazali (1095): Community as ethical covenant

v1.0.0
"""

from __future__ import annotations

from .registry import GuildRegistry
from .types import Guild, GuildJoinResult, GuildMember, GuildStatus

__version__ = "1.0.0"

__all__ = [
    "Guild",
    "GuildJoinResult",
    "GuildMember",
    "GuildRegistry",
    "GuildStatus",
]
