"""
Tests for BIZRA PR Quality Summary Generator
==============================================

Validates Markdown output, coverage status badges, trend classification.
"""

from scripts.ci_pr_quality_summary import generate_summary


class TestPRSummaryGeneration:
    """Test PR quality summary Markdown generation."""

    def test_generates_markdown(self) -> None:
        md = generate_summary(
            coverage=45.0,
            floor=38.0,
            ratcheted=False,
            new_floor="none",
            commit="abc123def456",
        )
        assert "## 🏛️ BIZRA Quality Dashboard" in md
        assert "45.0%" in md
        assert "38%" in md

    def test_regression_warning(self) -> None:
        md = generate_summary(
            coverage=35.0,
            floor=38.0,
            ratcheted=False,
            new_floor="none",
            commit="abc123",
        )
        assert "Regression" in md
        assert "🔴" in md

    def test_ratchet_eligible(self) -> None:
        md = generate_summary(
            coverage=45.0,
            floor=38.0,
            ratcheted=True,
            new_floor="40",
            commit="abc123",
        )
        assert "Eligible" in md
        assert "40%" in md
        assert "🔒" in md

    def test_stable_headroom(self) -> None:
        md = generate_summary(
            coverage=40.0,
            floor=38.0,
            ratcheted=False,
            new_floor="none",
            commit="abc123",
        )
        assert "Stable" in md

    def test_improving_high_coverage(self) -> None:
        md = generate_summary(
            coverage=50.0,
            floor=38.0,
            ratcheted=True,
            new_floor="43",
            commit="abc123",
        )
        assert "Improving" in md

    def test_at_threshold(self) -> None:
        md = generate_summary(
            coverage=38.0,
            floor=38.0,
            ratcheted=False,
            new_floor="none",
            commit="abc123",
        )
        assert "At Threshold" in md

    def test_green_badge_high_coverage(self) -> None:
        md = generate_summary(
            coverage=85.0,
            floor=38.0,
            ratcheted=True,
            new_floor="43",
            commit="abc123",
        )
        assert "🟢" in md

    def test_commit_truncation(self) -> None:
        md = generate_summary(
            coverage=42.0,
            floor=38.0,
            ratcheted=False,
            new_floor="none",
            commit="abc123def456789012345",
        )
        assert "abc123de" in md

    def test_constitutional_gates_present(self) -> None:
        md = generate_summary(
            coverage=42.0,
            floor=38.0,
            ratcheted=False,
            new_floor="none",
            commit="abc123",
        )
        assert "Ihsān" in md
        assert "SNR" in md
        assert "ADL Gini" in md

    def test_giants_footer(self) -> None:
        md = generate_summary(
            coverage=42.0,
            floor=38.0,
            ratcheted=False,
            new_floor="none",
            commit="abc123",
        )
        assert "Deming" in md
