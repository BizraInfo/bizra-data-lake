"""Tests for unified resource fabric scanning and scoring."""

from __future__ import annotations

from pathlib import Path

from core.skills.resource_fabric import ResourceFabric


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_resource_fabric_snapshot_from_profile(tmp_path: Path) -> None:
    # Minimal project topology
    _write(tmp_path / ".claude" / "agents" / "core" / "planner.md")
    _write(tmp_path / ".claude" / "commands" / "automation" / "smart.md")
    _write(tmp_path / ".claude" / "hooks" / "post-test.sh")
    _write(tmp_path / "core" / "skills" / "registry.py")
    _write(tmp_path / "core" / "memory" / "orchestrator.py")
    _write(tmp_path / "tools" / "mcp" / "gateway.py")
    _write(tmp_path / "bizra-omega" / "bizra-cli" / "config" / "skills.yaml")

    profile = tmp_path / "config" / "resource_fabric_profile.yaml"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(
        """
profile_name: test-fabric
profile_version: 1.0.0
cache_ttl_s: 10
category_weights:
  agents: 0.2
  commands: 0.2
  hooks: 0.2
  skills: 0.2
  memory: 0.1
  mcp: 0.1
expected_minimums:
  agents: 1
  commands: 1
  hooks: 1
  skills: 1
  memory: 1
  mcp: 1
sources:
  - name: agents
    category: agents
    path: .claude/agents
    patterns: ["*.md"]
    weight: 1.0
  - name: commands
    category: commands
    path: .claude/commands
    patterns: ["*.md"]
    weight: 1.0
  - name: hooks
    category: hooks
    path: .claude/hooks
    patterns: ["*.sh"]
    weight: 1.0
  - name: skills
    category: skills
    path: core/skills
    patterns: ["*.py"]
    weight: 1.0
  - name: memory
    category: memory
    path: core/memory
    patterns: ["*.py"]
    weight: 1.0
  - name: mcp
    category: mcp
    path: tools/mcp
    patterns: ["*.py"]
    weight: 1.0
""".strip(),
        encoding="utf-8",
    )

    fabric = ResourceFabric(project_root=tmp_path, profile_path=profile)
    snap = fabric.snapshot(limit=10, include_assets=True, force=True)

    assert snap["profile"]["profile_name"] == "test-fabric"
    assert snap["total_assets"] >= 6
    assert snap["active_sources"] == 6
    assert 0.0 <= snap["coverage_score"] <= 1.0
    assert 0.0 <= snap["fabric_score"] <= 1.0
    assert "agents" in snap["by_category"]
    assert len(snap["top_assets"]) >= 1


def test_resource_fabric_cache_compact_view(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "skills" / "router.py")
    profile = tmp_path / "config" / "resource_fabric_profile.yaml"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(
        """
profile_name: cache-test
sources:
  - name: skills
    category: skills
    path: core/skills
    patterns: ["*.py"]
    weight: 1.0
""".strip(),
        encoding="utf-8",
    )

    fabric = ResourceFabric(project_root=tmp_path, profile_path=profile)
    full = fabric.snapshot(limit=5, include_assets=True, force=True)
    compact = fabric.snapshot(limit=5, include_assets=False, force=False)

    assert full["total_assets"] == compact["total_assets"]
    assert "top_assets" in full
    assert "top_assets" not in compact
