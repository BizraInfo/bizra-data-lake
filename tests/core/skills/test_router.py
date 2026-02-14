"""
Tests for Skill Router — Invocation Routing & FATE Gate Validation
===================================================================

Standing on the Shoulders of Giants:
- Martin Fowler (2003): Enterprise Integration Patterns (routing)
- Alistair Cockburn (2005): Hexagonal Architecture (adapters)

إحسان — Excellence in all things.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.skills.router import (
    SkillRouter,
    SkillInvocationResult,
)
from core.skills.registry import (
    SkillRegistry,
    SkillManifest,
    SkillContext,
    SkillStatus,
    RegisteredSkill,
)
from core.skills.mcp_bridge import MCPBridge


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_registry(tmp_path):
    """Registry with one loaded skill."""
    skill_dir = tmp_path / "router-test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: router-test\ndescription: Skill for router tests\n"
        "agent: sovereign-coder\ncontext: fork\nihsan_floor: 0.90\ntags: [test]\n"
        "---\n\n# Router Test Skill\n",
        encoding="utf-8",
    )
    reg = SkillRegistry(skills_dir=str(tmp_path))
    reg.load_all()
    return reg


@pytest.fixture
def router(mock_registry):
    bridge = MCPBridge()
    return SkillRouter(registry=mock_registry, mcp_bridge=bridge, ihsan_threshold=0.90)


# ═══════════════════════════════════════════════════════════════════════════════
# SkillInvocationResult
# ═══════════════════════════════════════════════════════════════════════════════


class TestSkillInvocationResult:

    def test_defaults(self):
        r = SkillInvocationResult(success=True, skill_name="x")
        assert r.success is True
        assert r.skill_name == "x"
        assert r.ihsan_passed is False
        assert r.duration_ms == 0.0
        assert r.tools_used == []

    def test_to_dict(self):
        r = SkillInvocationResult(success=False, skill_name="y", error="boom")
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"] == "boom"
        assert "skill_name" in d
        assert "tools_used" in d


# ═══════════════════════════════════════════════════════════════════════════════
# SkillRouter — Sync helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestRouterSync:

    def test_init_defaults(self, router):
        assert router.ihsan_threshold == 0.90
        assert isinstance(router._handlers, dict)

    def test_resolve_agent_found(self, router):
        assert router.resolve_agent("router-test") == "sovereign-coder"

    def test_resolve_agent_missing(self, router):
        assert router.resolve_agent("nonexistent") is None

    def test_validate_ihsan_pass(self, router):
        assert router.validate_ihsan(0.95) is True

    def test_validate_ihsan_fail(self, router):
        assert router.validate_ihsan(0.50) is False

    def test_validate_ihsan_boundary(self, router):
        assert router.validate_ihsan(0.90) is True

    def test_validate_tools(self, router):
        # MCP bridge with no registered tools defaults to True
        assert router.validate_tools("router-test") is True

    def test_register_handler(self, router):
        handler = AsyncMock(return_value="done")
        router.register_handler("sovereign-coder", handler)
        assert "sovereign-coder" in router._handlers

    def test_get_available_skills(self, router):
        skills = router.get_available_skills(ihsan_score=0.95)
        assert "router-test" in skills

    def test_get_available_skills_blocked(self, router):
        skills = router.get_available_skills(ihsan_score=0.50)
        assert "router-test" not in skills

    def test_get_skills_by_agent(self, router):
        names = router.get_skills_by_agent("sovereign-coder")
        assert "router-test" in names

    def test_stats_initial(self, router):
        s = router.get_stats()
        assert s["total_invocations"] == 0
        assert s["success_count"] == 0

    def test_history_initial(self, router):
        assert router.get_recent_history() == []


# ═══════════════════════════════════════════════════════════════════════════════
# SkillRouter — Async invoke
# ═══════════════════════════════════════════════════════════════════════════════


class TestRouterInvoke:

    @pytest.mark.asyncio
    async def test_invoke_unknown_skill(self, router):
        result = await router.invoke("no-such-skill", {})
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_invoke_low_ihsan(self, router):
        result = await router.invoke("router-test", {}, ihsan_score=0.50)
        assert result.success is False
        assert "too low" in result.error.lower()
        assert result.ihsan_passed is False

    @pytest.mark.asyncio
    async def test_invoke_no_handler(self, router):
        """Without a registered handler, router returns skill info for external execution."""
        result = await router.invoke("router-test", {"q": "hello"}, ihsan_score=0.95)
        assert result.success is True
        assert result.ihsan_passed is True
        assert isinstance(result.output, dict)
        assert result.output["agent"] == "sovereign-coder"

    @pytest.mark.asyncio
    async def test_invoke_with_handler(self, router):
        handler = AsyncMock(return_value={"answer": 42})
        router.register_handler("sovereign-coder", handler)
        result = await router.invoke("router-test", {"q": "test"}, ihsan_score=0.95)
        assert result.success is True
        assert result.output == {"answer": 42}
        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invoke_handler_error(self, router):
        handler = AsyncMock(side_effect=RuntimeError("kaboom"))
        router.register_handler("sovereign-coder", handler)
        result = await router.invoke("router-test", {}, ihsan_score=0.95)
        assert result.success is False
        assert "kaboom" in result.error

    @pytest.mark.asyncio
    async def test_stats_after_invocations(self, router):
        await router.invoke("router-test", {}, ihsan_score=0.95)
        await router.invoke("nonexistent", {})
        s = router.get_stats()
        assert s["total_invocations"] == 2
        assert s["success_count"] == 1
        assert s["blocked_count"] == 1

    @pytest.mark.asyncio
    async def test_history_recorded(self, router):
        await router.invoke("router-test", {}, ihsan_score=0.95)
        history = router.get_recent_history(5)
        assert len(history) == 1
        assert history[0]["skill_name"] == "router-test"
