"""
Tests: Guild Registry
=====================

Covers: GuildRegistry, Guild, GuildMember, GuildJoinResult, GuildStatus

Standing on Giants:
- Beck (2002, TDD): Tests specify behavior before implementation
- Feathers (2004): Dependency injection for seam isolation
"""

from __future__ import annotations

import pytest

from core.guild import Guild, GuildJoinResult, GuildMember, GuildRegistry, GuildStatus
from core.guild.types import GuildJoinResult as GuildJoinResultType


# ─────────────────────────────────────────────────────────────────────────────
# Module Import Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGuildImports:
    def test_import_registry(self):
        from core.guild import GuildRegistry
        assert GuildRegistry is not None

    def test_import_types(self):
        from core.guild import Guild, GuildJoinResult, GuildMember, GuildStatus
        assert Guild is not None
        assert GuildJoinResult is not None
        assert GuildMember is not None
        assert GuildStatus is not None

    def test_module_version(self):
        import core.guild as g
        assert g.__version__ == "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# GuildStatus Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGuildStatus:
    def test_status_values(self):
        assert GuildStatus.PENDING == "pending"
        assert GuildStatus.ACTIVE == "active"
        assert GuildStatus.SUSPENDED == "suspended"

    def test_status_is_str_enum(self):
        assert isinstance(GuildStatus.ACTIVE, str)


# ─────────────────────────────────────────────────────────────────────────────
# Guild Dataclass Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGuild:
    def test_basic_creation(self):
        g = Guild(guild_id="test", name="Test Guild")
        assert g.guild_id == "test"
        assert g.name == "Test Guild"
        assert g.status == GuildStatus.ACTIVE
        assert g.members == []
        assert g.member_count == 0

    def test_created_at_auto_set(self):
        g = Guild(guild_id="auto", name="Auto")
        assert g.created_at.endswith("Z")

    def test_has_member_false_when_empty(self):
        g = Guild(guild_id="x", name="X")
        assert not g.has_member("BIZRA-12345678")

    def test_has_member_true_after_append(self):
        g = Guild(guild_id="x", name="X")
        g.members.append(GuildMember(node_id="BIZRA-11111111", guild_id="x"))
        assert g.has_member("BIZRA-11111111")
        assert not g.has_member("BIZRA-99999999")

    def test_member_count_tracks_length(self):
        g = Guild(guild_id="x", name="X")
        g.members.append(GuildMember(node_id="A", guild_id="x"))
        g.members.append(GuildMember(node_id="B", guild_id="x"))
        assert g.member_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# GuildMember Dataclass Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGuildMember:
    def test_basic_creation(self):
        m = GuildMember(node_id="BIZRA-00000001", guild_id="agri")
        assert m.node_id == "BIZRA-00000001"
        assert m.guild_id == "agri"
        assert m.role == "member"

    def test_joined_at_auto_set(self):
        m = GuildMember(node_id="A", guild_id="B")
        assert m.joined_at.endswith("Z")

    def test_custom_role(self):
        m = GuildMember(node_id="A", guild_id="B", role="founder")
        assert m.role == "founder"


# ─────────────────────────────────────────────────────────────────────────────
# GuildRegistry Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGuildRegistry:
    def test_creation_seeds_defaults(self):
        r = GuildRegistry()
        guilds = r.list_guilds()
        assert len(guilds) >= 5

    def test_default_guild_ids(self):
        r = GuildRegistry()
        ids = {g.guild_id for g in r.list_guilds()}
        assert "agriculture" in ids
        assert "healthcare" in ids
        assert "education" in ids
        assert "energy" in ids
        assert "finance" in ids

    def test_all_default_guilds_are_active(self):
        r = GuildRegistry()
        for g in r.list_guilds():
            assert g.status == GuildStatus.ACTIVE

    def test_join_existing_guild(self):
        r = GuildRegistry()
        result = r.join_guild("agriculture", "BIZRA-00000001")
        assert result.success
        assert result.guild is not None
        assert result.member is not None
        assert result.member.node_id == "BIZRA-00000001"

    def test_join_nonexistent_guild_fails(self):
        r = GuildRegistry()
        result = r.join_guild("nonexistent-123", "BIZRA-00000001")
        assert not result.success
        assert "not found" in result.message

    def test_join_twice_fails(self):
        r = GuildRegistry()
        r.join_guild("agriculture", "BIZRA-DUPLICATE")
        result = r.join_guild("agriculture", "BIZRA-DUPLICATE")
        assert not result.success
        assert "already a member" in result.message

    def test_member_count_increments(self):
        r = GuildRegistry()
        initial = r.get_guild("agriculture").member_count
        r.join_guild("agriculture", "BIZRA-AAA")
        r.join_guild("agriculture", "BIZRA-BBB")
        assert r.get_guild("agriculture").member_count == initial + 2

    def test_leave_guild_succeeds(self):
        r = GuildRegistry()
        r.join_guild("energy", "BIZRA-LEAVER")
        left = r.leave_guild("energy", "BIZRA-LEAVER")
        assert left
        assert not r.get_guild("energy").has_member("BIZRA-LEAVER")

    def test_leave_nonmember_returns_false(self):
        r = GuildRegistry()
        left = r.leave_guild("energy", "BIZRA-NOTHERE")
        assert not left

    def test_get_member_guilds_empty_initially(self):
        r = GuildRegistry()
        guilds = r.get_member_guilds("BIZRA-NEWNODE")
        assert guilds == []

    def test_get_member_guilds_after_join(self):
        r = GuildRegistry()
        r.join_guild("healthcare", "BIZRA-MEMBER")
        r.join_guild("finance", "BIZRA-MEMBER")
        guilds = r.get_member_guilds("BIZRA-MEMBER")
        guild_ids = {g.guild_id for g in guilds}
        assert "healthcare" in guild_ids
        assert "finance" in guild_ids

    def test_register_custom_guild(self):
        r = GuildRegistry()
        g = r.register_guild("custom", "Custom Guild", "Test")
        assert g.guild_id == "custom"
        assert r.get_guild("custom") is not None

    def test_get_online_count_starts_at_zero(self):
        r = GuildRegistry()
        assert r.get_online_count("agriculture") == 0

    def test_get_online_count_increments_on_join(self):
        r = GuildRegistry()
        r.join_guild("agriculture", "BIZRA-ONLINE1")
        assert r.get_online_count("agriculture") == 1

    def test_list_guilds_filter_by_status(self):
        r = GuildRegistry()
        r.register_guild("suspended-test", "Suspended", "")
        r.get_guild("suspended-test").status = GuildStatus.SUSPENDED
        active = r.list_guilds(status=GuildStatus.ACTIVE)
        assert all(g.status == GuildStatus.ACTIVE for g in active)
        suspended = r.list_guilds(status=GuildStatus.SUSPENDED)
        assert any(g.guild_id == "suspended-test" for g in suspended)
