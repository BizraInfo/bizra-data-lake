"""
BIZRA Guild Registry — Community Membership Engine
=====================================================

Manages guild creation, membership, and discovery. Pre-seeds
default guilds aligned with BIZRA's impact domains.

The registry is in-memory for v1 with optional JSON persistence.
Future versions will use SQLite via core.memory.unified_store.

Standing on Giants:
- Ostrom (1990): Polycentric governance of commons
- Shannon (1948): SNR as guild health signal
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


from .types import Guild, GuildJoinResult, GuildMember, GuildRole, GuildStatus

logger = logging.getLogger(__name__)


# Pre-seeded guilds aligned with BIZRA impact domains
DEFAULT_GUILDS = [
    (
        "agriculture",
        "Agriculture & Food Security",
        "Sustainable farming, water management, food sovereignty",
    ),
    (
        "healthcare",
        "Healthcare & Wellbeing",
        "Community health, telemedicine, preventive care",
    ),
    (
        "education",
        "Education & Knowledge",
        "Open learning, skill development, mentorship",
    ),
    (
        "energy",
        "Energy & Environment",
        "Renewable energy, conservation, climate action",
    ),
    (
        "finance",
        "Finance & Economic Justice",
        "Microfinance, cooperative economics, fair trade",
    ),
]


class GuildRegistry:
    """
    Guild membership and discovery engine.

    Manages the lifecycle of guilds: creation, joining, leaving,
    and querying. Tracks member Ihsan scores to form emergent
    fitness landscapes among communities.

    Usage:
        registry = GuildRegistry()
        result = registry.join_guild("agriculture", "BIZRA-00000000")
        guilds = registry.list_guilds()
    """

    def __init__(self, persist_path: Optional[Path] = None) -> None:
        self._guilds: Dict[str, Guild] = {}
        self._persist_path = persist_path
        self._seed_default_guilds()

    def _seed_default_guilds(self) -> None:
        """Pre-seed default impact-domain guilds."""
        for guild_id, name, description in DEFAULT_GUILDS:
            if guild_id not in self._guilds:
                self._guilds[guild_id] = Guild(
                    guild_id=guild_id,
                    name=name,
                    description=description,
                    status=GuildStatus.ACTIVE,
                )

    def register_guild(
        self,
        guild_id: str,
        name: str,
        description: str = "",
    ) -> Guild:
        """Register a new guild."""
        if guild_id in self._guilds:
            return self._guilds[guild_id]

        guild = Guild(
            guild_id=guild_id,
            name=name,
            description=description,
            status=GuildStatus.ACTIVE,
        )
        self._guilds[guild_id] = guild
        logger.info("Guild registered: %s (%s)", guild_id, name)
        self._persist()
        return guild

    def join_guild(
        self,
        guild_id: str,
        node_id: str,
        ihsan_score: float = 0.0,
        role: GuildRole = GuildRole.MEMBER,
    ) -> GuildJoinResult:
        """
        Join a guild.

        Args:
            guild_id: ID of the guild to join
            node_id: Node ID of the joining member
            ihsan_score: Current Ihsan score of the node
            role: Role to assign (default: MEMBER)

        Returns:
            GuildJoinResult with success/failure and details
        """
        guild = self._guilds.get(guild_id)
        if guild is None:
            return GuildJoinResult(
                success=False,
                message=f"Guild '{guild_id}' not found",
            )

        # Check for duplicate membership
        if guild.has_member(node_id):
            existing = next(m for m in guild.members if m.node_id == node_id)
            return GuildJoinResult(
                success=True,
                guild=guild,
                member=existing,
                message=f"Already a member of guild '{guild_id}'",
            )

        # Create membership
        member = GuildMember(
            node_id=node_id,
            guild_id=guild_id,
            joined_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            role=role,
            ihsan_score=ihsan_score,
        )
        guild.members.append(member)

        logger.info(
            "Node %s joined guild %s (members: %d)",
            node_id,
            guild_id,
            guild.member_count,
        )
        self._persist()

        return GuildJoinResult(
            success=True,
            guild=guild,
            member=member,
            message=f"Joined guild '{guild.name}' successfully",
        )

    def leave_guild(self, guild_id: str, node_id: str) -> bool:
        """Remove a node from a guild. Returns True if removed."""
        guild = self._guilds.get(guild_id)
        if guild is None:
            return False

        original_count = len(guild.members)
        guild.members = [m for m in guild.members if m.node_id != node_id]
        removed = len(guild.members) < original_count

        if removed:
            logger.info("Node %s left guild %s", node_id, guild_id)
            self._persist()
        return removed

    def get_guild(self, guild_id: str) -> Optional[Guild]:
        """Get a guild by ID."""
        return self._guilds.get(guild_id)

    def list_guilds(self) -> List[Guild]:
        """List all registered guilds."""
        return list(self._guilds.values())

    def get_online_count(self, guild_id: str) -> int:
        """Get the online member count for a guild."""
        guild = self._guilds.get(guild_id)
        if guild is None:
            return 0
        return guild.online_count

    def _persist(self) -> None:
        """Optionally persist guild data to JSON."""
        if self._persist_path is None:
            return
        try:
            data = {gid: g.to_dict() for gid, g in self._guilds.items()}
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to persist guild data: %s", e)
