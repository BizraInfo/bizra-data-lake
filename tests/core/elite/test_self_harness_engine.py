"""Tests for BIZRA agentic self harness engine."""

from __future__ import annotations

from pathlib import Path

from core.elite.self_harness_engine import SelfHarnessEngine


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_self_harness_detects_pytest_tail_masking(tmp_path: Path) -> None:
    _write(
        tmp_path / "scripts" / "ci.sh",
        "#!/usr/bin/env bash\npytest tests/ -q 2>&1 | tail -20\n",
    )

    profile = tmp_path / "config" / "self_harness_profile.yaml"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(
        """
profile_name: test-harness
include_paths: [scripts]
exclude_path_fragments: []
penalties: {critical: 0.05, high: 0.02, medium: 0.01, low: 0.005}
rules:
  - id: pytest_tail_masking
    category: ci
    severity: critical
    description: pytest tail
    file_globs: ["*.sh"]
    patterns: ['pytest[^\\n]*\\|\\s*tail']
    recommendation: fix pipeline
""".strip(),
        encoding="utf-8",
    )

    engine = SelfHarnessEngine(project_root=tmp_path, profile_path=profile)
    report = engine.run(include_findings=True, force=True)

    assert report["profile_name"] == "test-harness"
    assert report["total_findings"] == 1
    assert report["by_severity"]["critical"] == 1
    assert report["harness_score"] < 1.0
    assert report["findings"][0]["rule_id"] == "pytest_tail_masking"


def test_self_harness_applies_rule_exclude_patterns(tmp_path: Path) -> None:
    _write(
        tmp_path / "scripts" / "workspace_surgery.sh",
        "\n".join(
            [
                'echo "Fix: Replace pytest ... | tail -N with fail-closed logic"',
                "pytest tests/ -q 2>&1 | tail -20",
            ]
        ),
    )

    profile = tmp_path / "config" / "self_harness_profile.yaml"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(
        """
profile_name: test-harness
include_paths: [scripts]
exclude_path_fragments: []
penalties: {critical: 0.05, high: 0.02, medium: 0.01, low: 0.005}
rules:
  - id: pytest_tail_masking
    category: ci
    severity: critical
    description: pytest tail
    file_globs: ["*.sh"]
    patterns: ['pytest[^\\n]*\\|\\s*tail']
    exclude_patterns:
      - "echo.*pytest"
      - "Fix.*pytest"
      - "Replace.*pytest"
    recommendation: fix pipeline
""".strip(),
        encoding="utf-8",
    )

    engine = SelfHarnessEngine(project_root=tmp_path, profile_path=profile)
    report = engine.run(include_findings=True, force=True)

    assert report["total_findings"] == 1
    assert report["findings"][0]["line"] == 2
    assert "pytest tests/" in report["findings"][0]["snippet"]


def test_self_harness_cache_and_compact(tmp_path: Path) -> None:
    _write(tmp_path / "scripts" / "ok.sh", "echo ok\n")

    profile = tmp_path / "config" / "self_harness_profile.yaml"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(
        """
profile_name: cache-harness
cache_ttl_s: 60
include_paths: [scripts]
exclude_path_fragments: []
rules: []
""".strip(),
        encoding="utf-8",
    )

    engine = SelfHarnessEngine(project_root=tmp_path, profile_path=profile)
    full = engine.run(include_findings=True, force=True)
    compact = engine.run(include_findings=False, force=False)

    assert full["harness_score"] == compact["harness_score"]
    assert "findings" in full
    assert "findings" not in compact
