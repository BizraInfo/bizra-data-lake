"""
Tests for BIZRA Changelog Generator
=====================================

Validates conventional commit parsing, changelog generation,
markdown rendering, and evidence hashing.
"""

import json
from pathlib import Path

import pytest

from scripts.ci_changelog_gen import (
    ChangelogRelease,
    ParsedCommit,
    generate_changelog,
    parse_commit,
    render_markdown,
)

# ─────────────────────────────────────────────────────────────
# Tests: Commit Parsing
# ─────────────────────────────────────────────────────────────


class TestCommitParsing:
    """Test conventional commit message parsing."""

    def test_simple_feat(self) -> None:
        c = parse_commit("abc123", "Dev", "2025-01-01", "feat: add login page")
        assert c.type == "feat"
        assert c.description == "add login page"
        assert c.scope is None
        assert not c.is_breaking

    def test_scoped_fix(self) -> None:
        c = parse_commit(
            "def456", "Dev", "2025-01-01", "fix(auth): handle token expiry"
        )
        assert c.type == "fix"
        assert c.scope == "auth"
        assert c.description == "handle token expiry"

    def test_breaking_mark(self) -> None:
        c = parse_commit("ghi789", "Dev", "2025-01-01", "feat!: remove deprecated API")
        assert c.is_breaking

    def test_breaking_type(self) -> None:
        c = parse_commit("jkl012", "Dev", "2025-01-01", "breaking: schema migration v3")
        assert c.type == "breaking"
        assert c.is_breaking

    def test_breaking_footer(self) -> None:
        msg = "feat: new API\n\nBREAKING CHANGE: old endpoints removed"
        c = parse_commit("mno345", "Dev", "2025-01-01", msg)
        assert c.is_breaking

    def test_non_conventional(self) -> None:
        c = parse_commit("pqr678", "Dev", "2025-01-01", "random commit message")
        assert c.type == "chore"  # Default fallback
        assert c.description == "random commit message"

    def test_perf_commit(self) -> None:
        c = parse_commit(
            "stu901", "Dev", "2025-01-01", "perf(engine): reduce allocation by 40%"
        )
        assert c.type == "perf"
        assert c.scope == "engine"

    def test_security_commit(self) -> None:
        c = parse_commit(
            "vwx234", "Dev", "2025-01-01", "security(auth): patch CVE-2025-0001"
        )
        assert c.type == "security"

    def test_ci_commit(self) -> None:
        c = parse_commit("yza567", "Dev", "2025-01-01", "ci: add coverage ratchet")
        assert c.type == "ci"

    def test_body_extraction(self) -> None:
        msg = "feat: new feature\n\nThis is the body\nwith multiple lines"
        c = parse_commit("bcd890", "Dev", "2025-01-01", msg)
        assert "This is the body" in c.body
        assert "multiple lines" in c.body

    def test_sha_truncation(self) -> None:
        c = parse_commit("abc123def456789", "Dev", "2025-01-01", "feat: test")
        assert c.sha == "abc123de"  # First 8 chars

    def test_all_commit_types(self) -> None:
        types = [
            "feat",
            "fix",
            "perf",
            "refactor",
            "docs",
            "test",
            "ci",
            "chore",
            "security",
            "breaking",
        ]
        for t in types:
            c = parse_commit("abc123", "Dev", "2025-01-01", f"{t}: test message")
            assert c.type == t


# ─────────────────────────────────────────────────────────────
# Tests: Changelog Generation
# ─────────────────────────────────────────────────────────────


class TestChangelogGeneration:
    """Test structured changelog creation."""

    @pytest.fixture
    def sample_commits(self) -> list:
        return [
            parse_commit("a1b2", "Alice", "2025-01-01", "feat(ui): add dark mode"),
            parse_commit("c3d4", "Bob", "2025-01-02", "fix(api): null pointer in auth"),
            parse_commit(
                "e5f6", "Alice", "2025-01-03", "perf: reduce startup time 30%"
            ),
            parse_commit("g7h8", "Charlie", "2025-01-04", "docs: update README"),
            parse_commit(
                "i9j0", "Bob", "2025-01-05", "security: patch XSS vulnerability"
            ),
            parse_commit("k1l2", "Alice", "2025-01-06", "feat!: new config format"),
        ]

    def test_generates_sections(self, sample_commits: list) -> None:
        release = generate_changelog(sample_commits, "v2.1.0")
        section_types = [s.type_key for s in release.sections]
        assert "breaking" in section_types
        assert "feat" in section_types
        assert "fix" in section_types
        assert "perf" in section_types
        assert "docs" in section_types
        assert "security" in section_types

    def test_total_commits(self, sample_commits: list) -> None:
        release = generate_changelog(sample_commits, "v2.1.0")
        assert release.total_commits == 6

    def test_contributors(self, sample_commits: list) -> None:
        release = generate_changelog(sample_commits, "v2.1.0")
        assert "Alice" in release.contributors
        assert "Bob" in release.contributors
        assert "Charlie" in release.contributors

    def test_evidence_hash(self, sample_commits: list) -> None:
        release = generate_changelog(sample_commits, "v2.1.0")
        assert len(release.evidence_hash) == 16
        assert release.evidence_hash.isalnum()

    def test_evidence_hash_deterministic(self, sample_commits: list) -> None:
        r1 = generate_changelog(sample_commits, "v2.1.0")
        r2 = generate_changelog(sample_commits, "v2.1.0")
        assert r1.evidence_hash == r2.evidence_hash

    def test_breaking_separated(self, sample_commits: list) -> None:
        release = generate_changelog(sample_commits, "v2.1.0")
        breaking = [s for s in release.sections if s.type_key == "breaking"]
        assert len(breaking) == 1
        assert len(breaking[0].commits) == 1

    def test_empty_commits(self) -> None:
        release = generate_changelog([], "v2.1.0")
        assert release.total_commits == 0
        assert release.sections == []

    def test_version_label(self, sample_commits: list) -> None:
        release = generate_changelog(sample_commits, "v3.0.0-beta.1")
        assert release.version == "v3.0.0-beta.1"


# ─────────────────────────────────────────────────────────────
# Tests: Markdown Rendering
# ─────────────────────────────────────────────────────────────


class TestMarkdownRendering:
    """Test Markdown output format."""

    def test_version_heading(self) -> None:
        release = ChangelogRelease(version="v2.1.0", date="2025-01-15")
        md = render_markdown(release)
        assert "## [v2.1.0] - 2025-01-15" in md

    def test_commit_entries(self) -> None:
        commits = [
            parse_commit("abc123", "Dev", "2025-01-01", "feat(ui): add tooltips"),
        ]
        release = generate_changelog(commits, "v2.1.0")
        md = render_markdown(release)
        assert "**ui**: add tooltips (abc123" in md

    def test_section_ordering(self) -> None:
        commits = [
            parse_commit("a1", "Dev", "2025-01-01", "docs: update guide"),
            parse_commit("b2", "Dev", "2025-01-01", "feat: add feature"),
            parse_commit("c3", "Dev", "2025-01-01", "security: fix vuln"),
        ]
        release = generate_changelog(commits, "v2.1.0")
        md = render_markdown(release)
        # Security should appear before Features
        sec_pos = md.index("Security")
        feat_pos = md.index("Features")
        assert sec_pos < feat_pos

    def test_evidence_in_summary(self) -> None:
        commits = [parse_commit("xyz", "Dev", "2025-01-01", "feat: test")]
        release = generate_changelog(commits, "v2.1.0")
        md = render_markdown(release)
        assert f"Evidence: `{release.evidence_hash}`" in md

    def test_contributors_section(self) -> None:
        commits = [
            parse_commit("a1", "Alice", "2025-01-01", "feat: a"),
            parse_commit("b2", "Bob", "2025-01-01", "feat: b"),
        ]
        release = generate_changelog(commits, "v2.1.0")
        md = render_markdown(release)
        assert "Contributors" in md
        assert "Alice" in md
        assert "Bob" in md
