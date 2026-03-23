"""
BIZRA Guild System — Smoke Tests
==================================

8 tests covering guild registration, membership, and discovery.

Test naming: test_XX_descriptive_name
Coverage: GuildRegistry, Guild, GuildMember, GuildJoinResult
"""

import pytest

from core.guild import (
    GuildRegistry,
    GuildStatus,
)


class TestGuildSmoke:
    """Guild system smoke tests."""

    # ── test_01: guild registration ──────────────────────────────────────
    def test_01_register_guild(self) -> None:
        """A new guild can be registered and retrieved."""
        registry = GuildRegistry()
        guild = registry.register_guild(
            guild_id="test-guild",
            name="Test Guild",
            description="A test guild",
        )
        assert guild.guild_id == "test-guild"
        assert guild.name == "Test Guild"
        assert guild.status == GuildStatus.ACTIVE
        assert guild.member_count == 0

    # ── test_02: join guild ──────────────────────────────────────────────
    def test_02_join_guild(self) -> None:
        """A node can join an existing guild."""
        registry = GuildRegistry()
        result = registry.join_guild(
            guild_id="agriculture",
            node_id="BIZRA-00000001",
            ihsan_score=0.97,
        )
        assert result.success is True
        assert result.member is not None
        assert result.member.node_id == "BIZRA-00000001"
        assert result.member.guild_id == "agriculture"
        assert result.member.ihsan_score == 0.97
        assert result.guild is not None
        assert result.guild.member_count == 1

    # ── test_03: leave guild ─────────────────────────────────────────────
    def test_03_leave_guild(self) -> None:
        """A node can leave a guild."""
        registry = GuildRegistry()
        registry.join_guild("agriculture", "BIZRA-00000001")
        assert registry.leave_guild("agriculture", "BIZRA-00000001") is True
        guild = registry.get_guild("agriculture")
        assert guild is not None
        assert guild.member_count == 0

    # ── test_04: default guilds pre-seeded ───────────────────────────────
    def test_04_default_guilds_preseeded(self) -> None:
        """Default impact-domain guilds are pre-seeded on init."""
        registry = GuildRegistry()
        guilds = registry.list_guilds()
        guild_ids = {g.guild_id for g in guilds}
        assert len(guilds) >= 5
        assert "agriculture" in guild_ids
        assert "healthcare" in guild_ids
        assert "education" in guild_ids
        assert "energy" in guild_ids
        assert "finance" in guild_ids

    # ── test_05: online count tracking ───────────────────────────────────
    def test_05_online_count(self) -> None:
        """Online count reflects members (minimum 1 for active guilds)."""
        registry = GuildRegistry()
        # Pre-seeded guilds have 0 members but online_count >= 1
        count = registry.get_online_count("agriculture")
        assert count >= 1  # min 1 for active guilds

        # After joining, count increases
        registry.join_guild("agriculture", "BIZRA-00000001")
        registry.join_guild("agriculture", "BIZRA-00000002")
        count = registry.get_online_count("agriculture")
        assert count >= 2

    # ── test_06: guild not found ─────────────────────────────────────────
    def test_06_guild_not_found(self) -> None:
        """Joining a non-existent guild returns failure."""
        registry = GuildRegistry()
        result = registry.join_guild("nonexistent-guild", "BIZRA-00000001")
        assert result.success is False
        assert "not found" in result.message

    # ── test_07: duplicate join prevention ───────────────────────────────
    def test_07_duplicate_join(self) -> None:
        """Joining the same guild twice returns success without duplication."""
        registry = GuildRegistry()
        result1 = registry.join_guild("agriculture", "BIZRA-00000001")
        result2 = registry.join_guild("agriculture", "BIZRA-00000001")
        assert result1.success is True
        assert result2.success is True
        assert result2.message == "Already a member of guild 'agriculture'"
        guild = registry.get_guild("agriculture")
        assert guild is not None
        assert guild.member_count == 1  # Not duplicated

    # ── test_08: guild member ihsan tracking ──────────────────────────────
    def test_08_guild_mean_ihsan(self) -> None:
        """Guild tracks mean Ihsan score across members."""
        registry = GuildRegistry()
        registry.join_guild("agriculture", "BIZRA-00000001", ihsan_score=0.95)
        registry.join_guild("agriculture", "BIZRA-00000002", ihsan_score=0.99)
        guild = registry.get_guild("agriculture")
        assert guild is not None
        assert guild.mean_ihsan == pytest.approx(0.97, abs=0.01)
