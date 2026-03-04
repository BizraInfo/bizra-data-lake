"""
Tests for Skill Registry — Dynamic Skill Discovery and Management
==================================================================

Standing on the Shoulders of Giants:
- Eric Evans (2003): Domain-Driven Design
- Martin Fowler (2004): Plugin Architecture Patterns

إحسان — Excellence in all things.
"""


import pytest

pytest.importorskip("yaml")

from core.skills.registry import (
    RegisteredSkill,
    SkillContext,
    SkillManifest,
    SkillRegistry,
    SkillStatus,
    get_skill_registry,
)

# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_manifest():
    return SkillManifest(
        name="test-skill",
        description="A test skill for unit testing",
        agent="sovereign-researcher",
        context=SkillContext.FORK,
        version="1.0.0",
        ihsan_floor=0.95,
        tags=["test", "example"],
    )


@pytest.fixture
def mock_skill_dir(tmp_path):
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: Test skill\n"
        "agent: sovereign-researcher\ncontext: fork\ntags: [test, example]\n"
        "---\n\n# Test Skill\n\nContent here.\n",
        encoding="utf-8",
    )
    return tmp_path


# ═══════════════════════════════════════════════════════════════════════════════
# SkillManifest
# ═══════════════════════════════════════════════════════════════════════════════


class TestSkillManifest:

    def test_creation(self, sample_manifest):
        assert sample_manifest.name == "test-skill"
        assert sample_manifest.agent == "sovereign-researcher"
        assert sample_manifest.ihsan_floor == 0.95

    def test_defaults(self):
        m = SkillManifest(name="minimal", description="Minimal")
        assert m.context == SkillContext.FORK
        assert m.version == "1.0.0"
        assert m.tags == []
        assert m.author == "BIZRA"

    def test_to_dict(self, sample_manifest):
        d = sample_manifest.to_dict()
        assert d["name"] == "test-skill"
        assert d["context"] == "fork"
        assert "tags" in d

    def test_from_frontmatter(self):
        fm = {
            "name": "fm",
            "description": "FM skill",
            "agent": "coder",
            "context": "inline",
        }
        m = SkillManifest.from_frontmatter(fm, raw="# MD")
        assert m.name == "fm"
        assert m.context == SkillContext.INLINE
        assert m.raw_content == "# MD"

    def test_from_frontmatter_defaults(self):
        m = SkillManifest.from_frontmatter({"name": "bare", "description": "d"})
        assert m.agent == "general-purpose"
        assert m.context == SkillContext.FORK


# ═══════════════════════════════════════════════════════════════════════════════
# RegisteredSkill
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegisteredSkill:

    def test_creation(self, sample_manifest):
        skill = RegisteredSkill(manifest=sample_manifest, path="/skills/test/SKILL.md")
        assert skill.status == SkillStatus.AVAILABLE
        assert skill.invocation_count == 0

    def test_record_invocation(self, sample_manifest):
        skill = RegisteredSkill(manifest=sample_manifest, path="/p")
        skill.record_invocation(success=True, duration_ms=150.0)
        assert skill.invocation_count == 1
        assert skill.success_count == 1
        skill.record_invocation(success=False, duration_ms=50.0)
        assert skill.failure_count == 1

    def test_success_rate(self, sample_manifest):
        skill = RegisteredSkill(manifest=sample_manifest, path="/p")
        assert skill.success_rate == 1.0
        skill.record_invocation(True, 100.0)
        skill.record_invocation(True, 100.0)
        skill.record_invocation(False, 100.0)
        assert skill.success_rate == pytest.approx(2 / 3)

    def test_avg_duration(self, sample_manifest):
        skill = RegisteredSkill(manifest=sample_manifest, path="/p")
        skill.record_invocation(True, 100.0)
        skill.record_invocation(True, 200.0)
        assert skill.avg_duration_ms == pytest.approx(150.0)

    def test_to_dict(self, sample_manifest):
        d = RegisteredSkill(manifest=sample_manifest, path="/p").to_dict()
        assert d["manifest"]["name"] == "test-skill"
        assert d["status"] == "available"


# ═══════════════════════════════════════════════════════════════════════════════
# SkillRegistry
# ═══════════════════════════════════════════════════════════════════════════════


class TestSkillRegistry:

    def test_init(self):
        r = SkillRegistry(skills_dir="/nonexistent")
        assert isinstance(r._skills, dict)

    def test_load_from_dir(self, mock_skill_dir):
        r = SkillRegistry(skills_dir=str(mock_skill_dir))
        assert r.load_all() >= 1
        assert r.get("test-skill") is not None

    def test_get_nonexistent(self, mock_skill_dir):
        r = SkillRegistry(skills_dir=str(mock_skill_dir))
        r.load_all()
        assert r.get("fake") is None

    def test_get_all(self, mock_skill_dir):
        r = SkillRegistry(skills_dir=str(mock_skill_dir))
        r.load_all()
        assert len(r.get_all()) >= 1

    def test_find_by_tag(self, mock_skill_dir):
        r = SkillRegistry(skills_dir=str(mock_skill_dir))
        r.load_all()
        results = r.find_by_tag("test")
        assert len(results) >= 1

    def test_find_by_agent(self, mock_skill_dir):
        r = SkillRegistry(skills_dir=str(mock_skill_dir))
        r.load_all()
        results = r.find_by_agent("sovereign-researcher")
        assert len(results) >= 1

    def test_can_invoke_above(self, mock_skill_dir):
        r = SkillRegistry(skills_dir=str(mock_skill_dir))
        r.load_all()
        assert r.can_invoke("test-skill", 0.96) is True

    def test_can_invoke_below(self, mock_skill_dir):
        r = SkillRegistry(skills_dir=str(mock_skill_dir))
        r.load_all()
        assert r.can_invoke("test-skill", 0.80) is False

    def test_stats(self, mock_skill_dir):
        r = SkillRegistry(skills_dir=str(mock_skill_dir))
        r.load_all()
        s = r.get_stats()
        assert s["total_skills"] >= 1
        assert "by_agent" in s
        assert "performance_profile" in s
        assert "resource_fabric" in s
        assert "self_harness" in s

    def test_get_top_skills_has_scoring_fields(self, mock_skill_dir):
        r = SkillRegistry(skills_dir=str(mock_skill_dir))
        r.load_all()
        skill = r.get("test-skill")
        assert skill is not None
        skill.record_invocation(success=True, duration_ms=120.0)
        skill.record_invocation(success=True, duration_ms=180.0)

        ranked = r.get_top_skills(limit=5, ihsan_score=0.99)
        assert len(ranked) == 1
        row = ranked[0]
        assert row["name"] == "test-skill"
        assert 0 <= row["performance_score"] <= 1
        assert "score_components" in row
        assert set(row["score_components"].keys()) == {
            "success_rate",
            "latency",
            "usage_confidence",
            "ihsan_floor",
            "tag_boost",
        }

    def test_get_top_skills_sorts_by_score(self, tmp_path):
        alpha = tmp_path / "alpha-skill"
        beta = tmp_path / "beta-skill"
        alpha.mkdir()
        beta.mkdir()

        (alpha / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: Alpha\nagent: sovereign-coder\n"
            "context: fork\ntags: [security, performance]\n---\n\n# Alpha\n",
            encoding="utf-8",
        )
        (beta / "SKILL.md").write_text(
            "---\nname: beta\ndescription: Beta\nagent: sovereign-coder\n"
            "context: fork\ntags: [documentation]\n---\n\n# Beta\n",
            encoding="utf-8",
        )

        r = SkillRegistry(skills_dir=str(tmp_path))
        r.load_all()
        a = r.get("alpha")
        b = r.get("beta")
        assert a is not None and b is not None

        # Alpha: high reliability + low latency
        for _ in range(8):
            a.record_invocation(success=True, duration_ms=90.0)
        # Beta: lower success and slower latency
        for _ in range(4):
            b.record_invocation(success=True, duration_ms=900.0)
        for _ in range(4):
            b.record_invocation(success=False, duration_ms=900.0)

        ranked = r.get_top_skills(limit=2, ihsan_score=1.0)
        assert [x["name"] for x in ranked] == ["alpha", "beta"]


# ═══════════════════════════════════════════════════════════════════════════════
# Frontmatter Parsing
# ═══════════════════════════════════════════════════════════════════════════════


class TestFrontmatter:

    def test_valid(self, mock_skill_dir):
        r = SkillRegistry(skills_dir=str(mock_skill_dir))
        skill = r._load_skill(mock_skill_dir / "test-skill" / "SKILL.md")
        assert skill is not None
        assert skill.manifest.name == "test-skill"

    def test_missing(self, tmp_path):
        d = tmp_path / "no-fm"
        d.mkdir()
        (d / "SKILL.md").write_text("# No frontmatter\n", encoding="utf-8")
        r = SkillRegistry(skills_dir=str(tmp_path))
        assert r._load_skill(d / "SKILL.md") is None

    def test_invalid_yaml(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "SKILL.md").write_text(
            "---\nname: [invalid\n---\nContent\n", encoding="utf-8"
        )
        r = SkillRegistry(skills_dir=str(tmp_path))
        assert r._load_skill(d / "SKILL.md") is None


# ═══════════════════════════════════════════════════════════════════════════════
# Integration with real .claude/skills/
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistryIntegration:

    def test_real_skills_load(self):
        registry = get_skill_registry()
        assert registry.get_stats()["total_skills"] >= 10

    def test_all_valid_manifests(self):
        registry = get_skill_registry()
        for skill in registry.get_all():
            assert skill.manifest.name
            assert skill.manifest.description
            assert 0 <= skill.manifest.ihsan_floor <= 1
