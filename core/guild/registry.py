"""
Guild Registry — Community Registration and Membership
========================================================

Manages guild creation, membership, and discovery. Pre-seeds
default guilds aligned with BIZRA's impact domains.

Standing on Giants:
- Ostrom (1990): Commons governance without central authority
- McGonigal (2011): Community organization for social good
- Shannon (1948): SNR as quality gate for membership
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

from .types import Guild, GuildJoinResult, GuildMember, GuildStatus

logger = logging.getLogger(__name__)


# Pre-seeded guilds aligned with BIZRA impact domains
DEFAULT_GUILDS = [
    Guild(
        guild_id="agriculture",
        name="Agriculture & Food Sovereignty",
        description="Sustainable farming, water management, and food supply chain integrity.",
    ),
    Guild(
        guild_id="healthcare",
        name="Healthcare & Wellness",
        description="Health data sovereignty, community diagnostics, and wellness tracking.",
    ),
    Guild(
        guild_id="education",
        name="Education & Knowledge",
        description="Open curricula, knowledge graphs, and community learning systems.",
    ),
    Guild(
        guild_id="energy",
        name="Energy & Environment",
        description="Solar microgrids, carbon tracking, and peer-to-peer energy trading.",
    ),
    Guild(
        guild_id="finance",
        name="Finance & Economic Justice",
        description="Cooperative lending, zakat compliance, and equitable resource distribution.",
    ),
]


class GuildRegistry:
    """
    Guild management engine.

    Handles guild registration, membership, and discovery.
    Pre-seeds default guilds for BIZRA's five impact domains.

    Usage:
        registry = GuildRegistry()
        result = registry.join_guild("agriculture", "BIZRA-00000000")
        guilds = registry.list_guilds()
    """

    def __init__(self) -> None:
        self._guilds: Dict[str, Guild] = {}
        self._seed_default_guilds()

    def _seed_default_guilds(self) -> None:
        """Pre-seed default impact-aligned guilds."""
        for guild in DEFAULT_GUILDS:
            if guild.guild_id not in self._guilds:
                self._guilds[guild.guild_id] = Guild(
                    guild_id=guild.guild_id,
                    name=guild.name,
                    description=guild.description,
                    status=GuildStatus.ACTIVE,
                )

    def register_guild(
        self,
        guild_id: str,
        name: str,
        description: str = "",
    ) -> Guild:
        """Register a new guild."""
        guild = Guild(
            guild_id=guild_id,
            name=name,
            description=description,
            status=GuildStatus.ACTIVE,
        )
        self._guilds[guild_id] = guild
        logger.info("Guild registered: %s (%s)", guild_id, name)
        return guild

    def join_guild(
        self,
        guild_id: str,
        node_id: str,
        role: str = "member",
        ihsan_score: float = 0.0,
    ) -> GuildJoinResult:
        """
        Join a guild.

        Args:
            guild_id: ID of the guild to join
            node_id: Node ID of the joining node
            role: Membership role (member, elder, founder)
            ihsan_score: Current Ihsan score of the node

        Returns:
            GuildJoinResult with success/failure and details
        """
        guild = self._guilds.get(guild_id)
        if guild is None:
            return GuildJoinResult(
                success=False,
                message=f"Guild '{guild_id}' not found",
            )

        if guild.has_member(node_id):
            return GuildJoinResult(
                success=False,
                guild=guild,
                message=f"Node '{node_id}' is already a member of '{guild_id}'",
            )

        member = GuildMember(
            node_id=node_id,
            guild_id=guild_id,
            role=role,
            ihsan_score=ihsan_score,
        )
        guild.members.append(member)
        guild.online_count += 1

        logger.info("Node %s joined guild %s", node_id, guild_id)

        return GuildJoinResult(
            success=True,
            guild=guild,
            member=member,
            message=f"Joined '{guild.name}' as {role}",
        )

    def leave_guild(self, guild_id: str, node_id: str) -> bool:
        """Leave a guild. Returns True if successfully left."""
        guild = self._guilds.get(guild_id)
        if guild is None:
            return False

        original_count = len(guild.members)
        guild.members = [m for m in guild.members if m.node_id != node_id]

        if len(guild.members) < original_count:
            guild.online_count = max(0, guild.online_count - 1)
            logger.info("Node %s left guild %s", node_id, guild_id)
            return True

        return False

    def get_guild(self, guild_id: str) -> Optional[Guild]:
        """Get a guild by ID."""
        return self._guilds.get(guild_id)

    def list_guilds(self, status: Optional[GuildStatus] = None) -> List[Guild]:
        """List all guilds, optionally filtered by status."""
        guilds = list(self._guilds.values())
        if status is not None:
            guilds = [g for g in guilds if g.status == status]
        return guilds

    def get_online_count(self, guild_id: str) -> int:
        """Get the online member count for a guild."""
        guild = self._guilds.get(guild_id)
        return guild.online_count if guild else 0

    def get_member_guilds(self, node_id: str) -> List[Guild]:
        """Get all guilds a node is a member of."""
        return [
            g for g in self._guilds.values()
            if g.has_member(node_id)
        ]
