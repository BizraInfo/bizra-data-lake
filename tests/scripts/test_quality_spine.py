"""
Tests for BIZRA Quality Spine
===============================

Single test file covering all five vertebrae:
  V1: Coverage Ratchet
  V2: Quality Trend
  V3: Release Gates
  V4: Changelog
  V5: PR Summary
"""

import json
import textwrap
from pathlib import Path

import pytest

from scripts.quality_spine import (
    GateResult,
    QualitySnapshot,
    RatchetResult,
    SpineVerdict,
    TrendStore,
    _linear_slope,
    aggregate_coverage,
    analyze_trend,
    evaluate_ratchet,
    parse_commit,
    parse_coverage_xml,
    parse_istanbul,
    parse_lcov,
    read_coverage_floor,
    render_changelog,
    render_pr_summary,
    write_coverage_floor,
)

# ═════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════


@pytest.fixture
def coverage_xml(tmp_path: Path) -> Path:
    xml = tmp_path / "coverage.xml"
    xml.write_text(
        textwrap.dedent("""\
        <?xml version="1.0" ?>
        <coverage line-rate="0.45" lines-valid="1000" lines-covered="450">
          <packages/>
        </coverage>
    """),
        encoding="utf-8",
    )
    return xml


@pytest.fixture
def pyproject(tmp_path: Path) -> Path:
    f = tmp_path / "pyproject.toml"
    f.write_text(
        textwrap.dedent("""\
        [project]
        name = "test"
        version = "2.0.0"
        [tool.coverage.report]
        fail_under = 38
        show_missing = true
    """),
        encoding="utf-8",
    )
    return f


@pytest.fixture
def store(tmp_path: Path) -> TrendStore:
    return TrendStore(tmp_path / "trend.jsonl")


# ═════════════════════════════════════════════════════════════
# V1: COVERAGE RATCHET
# ═════════════════════════════════════════════════════════════


class TestRatchetXML:
    def test_parse_valid(self, coverage_xml: Path) -> None:
        assert parse_coverage_xml(coverage_xml) == pytest.approx(45.0, abs=0.1)

    def test_parse_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_coverage_xml(tmp_path / "nope.xml")

    def test_parse_no_line_rate(self, tmp_path: Path) -> None:
        (tmp_path / "bad.xml").write_text("<coverage/>")
        with pytest.raises(ValueError, match="line-rate"):
            parse_coverage_xml(tmp_path / "bad.xml")

    def test_zero(self, tmp_path: Path) -> None:
        (tmp_path / "z.xml").write_text('<coverage line-rate="0.0"/>')
        assert parse_coverage_xml(tmp_path / "z.xml") == 0.0

    def test_full(self, tmp_path: Path) -> None:
        (tmp_path / "f.xml").write_text('<coverage line-rate="1.0"/>')
        assert parse_coverage_xml(tmp_path / "f.xml") == 100.0


class TestRatchetFloor:
    def test_read(self, pyproject: Path) -> None:
        assert read_coverage_floor(pyproject) == 38.0

    def test_read_missing(self, tmp_path: Path) -> None:
        (tmp_path / "e.toml").write_text("[project]\nname='t'\n")
        with pytest.raises(ValueError):
            read_coverage_floor(tmp_path / "e.toml")

    def test_write(self, pyproject: Path) -> None:
        write_coverage_floor(pyproject, 42)
        assert read_coverage_floor(pyproject) == 42.0

    def test_write_preserves(self, pyproject: Path) -> None:
        write_coverage_floor(pyproject, 50)
        assert "show_missing = true" in pyproject.read_text()


class TestRatchetLogic:
    def test_no_ratchet_below_step(self) -> None:
        r = evaluate_ratchet(38.5, 38.0, step=1)
        assert not r.ratcheted and not r.regression

    def test_triggers(self) -> None:
        r = evaluate_ratchet(40.0, 38.0, step=1)
        assert r.ratcheted and r.new_floor == 40

    def test_cap_max_bump(self) -> None:
        r = evaluate_ratchet(50.0, 38.0, step=1)
        assert r.new_floor == 43  # 38 + MAX_RATCHET_BUMP(5)

    def test_regression(self) -> None:
        r = evaluate_ratchet(35.0, 38.0)
        assert r.regression and r.headroom == pytest.approx(-3.0)

    def test_exact_floor(self) -> None:
        r = evaluate_ratchet(38.0, 38.0)
        assert not r.ratcheted and not r.regression

    def test_custom_step(self) -> None:
        assert not evaluate_ratchet(40.0, 38.0, step=3).ratcheted
        assert evaluate_ratchet(41.0, 38.0, step=3).ratcheted

    def test_hash_unique(self) -> None:
        assert (
            evaluate_ratchet(40.0, 38.0).evidence_hash
            != evaluate_ratchet(41.0, 38.0).evidence_hash
        )

    def test_hash_deterministic(self) -> None:
        r1 = RatchetResult("T", 40.0, 38.0, 40.0, True, False, 2.0, False)
        r2 = RatchetResult("T", 40.0, 38.0, 40.0, True, False, 2.0, False)
        assert r1.evidence_hash == r2.evidence_hash


class TestMultiLangCoverage:
    def test_python_only(self, coverage_xml: Path) -> None:
        r = aggregate_coverage(python_xml=coverage_xml)
        assert r["python"] == pytest.approx(45.0, abs=0.1)

    def test_lcov(self, tmp_path: Path) -> None:
        (tmp_path / "l.info").write_text("LF:3\nLH:2\n")
        assert parse_lcov(tmp_path / "l.info") == pytest.approx(66.67, abs=0.1)

    def test_istanbul(self, tmp_path: Path) -> None:
        (tmp_path / "c.json").write_text(
            json.dumps({"f": {"s": {"0": 1, "1": 1, "2": 0, "3": 1}}})
        )
        assert parse_istanbul(tmp_path / "c.json") == 75.0

    def test_aggregate_all(self, coverage_xml: Path, tmp_path: Path) -> None:
        (tmp_path / "l.info").write_text("LF:100\nLH:60\n")
        (tmp_path / "c.json").write_text(
            json.dumps({"f": {"s": {"0": 1, "1": 1, "2": 0}}})
        )
        r = aggregate_coverage(coverage_xml, tmp_path / "l.info", tmp_path / "c.json")
        assert "aggregate" in r and len(r) == 4

    def test_missing_skipped(self, tmp_path: Path) -> None:
        assert aggregate_coverage(python_xml=tmp_path / "nope.xml") == {}


# ═════════════════════════════════════════════════════════════
# V2: QUALITY TREND
# ═════════════════════════════════════════════════════════════


class TestSnapshot:
    def test_auto_timestamp(self) -> None:
        assert QualitySnapshot().timestamp

    def test_hash(self) -> None:
        s = QualitySnapshot(timestamp="T", snr_score=0.92)
        assert len(s.compute_hash()) == 32

    def test_finalize(self) -> None:
        s = QualitySnapshot(timestamp="T")
        s.finalize()
        assert s.snapshot_hash

    def test_hash_differs(self) -> None:
        a = QualitySnapshot(timestamp="T", snr_score=0.9)
        b = QualitySnapshot(timestamp="T", snr_score=0.91)
        assert a.compute_hash() != b.compute_hash()


class TestTrendStore:
    def test_empty(self, store: TrendStore) -> None:
        assert store.last() is None and store.count() == 0

    def test_append(self, store: TrendStore) -> None:
        store.append(QualitySnapshot(timestamp="T", snr_score=0.92))
        assert store.count() == 1
        assert store.last().snr_score == 0.92

    def test_chain(self, store: TrendStore) -> None:
        store.append(QualitySnapshot(timestamp="T1"))
        store.append(QualitySnapshot(timestamp="T2"))
        all_s = store.read_all()
        assert all_s[1].parent_hash == all_s[0].snapshot_hash

    def test_genesis_parent(self, store: TrendStore) -> None:
        store.append(QualitySnapshot(timestamp="T"))
        assert store.read_all()[0].parent_hash == "0" * 32

    def test_read_last_n(self, store: TrendStore) -> None:
        for i in range(10):
            store.append(QualitySnapshot(timestamp=f"T{i}"))
        assert len(store.read_last_n(3)) == 3


class TestLinearSlope:
    def test_flat(self) -> None:
        assert _linear_slope([1.0, 1.0, 1.0]) == 0.0

    def test_up(self) -> None:
        assert _linear_slope([1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_down(self) -> None:
        assert _linear_slope([3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_single(self) -> None:
        assert _linear_slope([5.0]) == 0.0


class TestTrendAnalysis:
    def _improving(self) -> list:
        return [
            QualitySnapshot(
                timestamp=f"T{i}",
                snr_score=0.85 + i * 0.01,
                coverage_pct=38.0 + i * 1.5,
                mypy_errors=1600 - i * 20,
                tests_total=200,
                tests_passed=195 + min(i, 5),
            )
            for i in range(10)
        ]

    def test_insufficient(self) -> None:
        assert analyze_trend([QualitySnapshot()]).direction == "insufficient_data"

    def test_improving(self) -> None:
        t = analyze_trend(self._improving())
        assert t.direction == "improving"

    def test_degrading(self) -> None:
        snaps = [
            QualitySnapshot(
                timestamp=f"T{i}",
                snr_score=0.95 - i * 0.02,
                coverage_pct=50.0 - i * 2.0,
                mypy_errors=1000 + i * 50,
                tests_total=200,
                tests_passed=190 - i * 5,
            )
            for i in range(10)
        ]
        assert analyze_trend(snaps).direction == "degrading"

    def test_stable(self) -> None:
        snaps = [
            QualitySnapshot(
                timestamp=f"T{i}",
                snr_score=0.92,
                coverage_pct=42.0,
                mypy_errors=1500,
                tests_total=200,
                tests_passed=195,
            )
            for i in range(10)
        ]
        assert analyze_trend(snaps).direction == "stable"

    def test_summary(self) -> None:
        assert "improving" in analyze_trend(self._improving()).summary


# ═════════════════════════════════════════════════════════════
# V3: GATES
# ═════════════════════════════════════════════════════════════


class TestGateResult:
    def test_dataclass(self) -> None:
        g = GateResult("test", "quality", True, 1.0, 0.1, "ok")
        assert g.passed and g.blocking

    def test_non_blocking(self) -> None:
        g = GateResult("test", "quality", False, 0.0, 0.1, "skip", blocking=False)
        assert not g.blocking


class TestSpineVerdict:
    def test_auto_timestamp(self) -> None:
        assert SpineVerdict().timestamp

    def test_defaults(self) -> None:
        v = SpineVerdict()
        assert not v.passed and v.overall_score == 0.0


# ═════════════════════════════════════════════════════════════
# V4: CHANGELOG
# ═════════════════════════════════════════════════════════════


class TestCommitParsing:
    def test_feat(self) -> None:
        c = parse_commit("abc123", "Dev", "feat: add login")
        assert c.type == "feat" and c.desc == "add login"

    def test_scoped(self) -> None:
        c = parse_commit("abc123", "Dev", "fix(auth): token expiry")
        assert c.scope == "auth"

    def test_breaking_bang(self) -> None:
        assert parse_commit("a", "D", "feat!: remove API").is_breaking

    def test_breaking_type(self) -> None:
        assert parse_commit("a", "D", "breaking: schema v3").is_breaking

    def test_breaking_footer(self) -> None:
        assert parse_commit(
            "a", "D", "feat: new\n\nBREAKING CHANGE: old removed"
        ).is_breaking

    def test_non_conventional(self) -> None:
        c = parse_commit("a", "D", "random message")
        assert c.type == "chore"

    def test_all_types(self) -> None:
        for t in [
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
        ]:
            assert parse_commit("a", "D", f"{t}: msg").type == t

    def test_sha_truncation(self) -> None:
        assert parse_commit("abc123def456", "D", "feat: x").sha == "abc123de"


class TestChangelog:
    def _commits(self) -> list:
        return [
            parse_commit("a1", "Alice", "feat(ui): dark mode"),
            parse_commit("b2", "Bob", "fix(api): null ptr"),
            parse_commit("c3", "Alice", "security: patch XSS"),
            parse_commit("d4", "Charlie", "feat!: new config"),
        ]

    def test_render(self) -> None:
        md = render_changelog(self._commits(), "v2.1.0")
        assert "[v2.1.0]" in md

    def test_sections(self) -> None:
        md = render_changelog(self._commits(), "v2.1.0")
        assert "Breaking" in md and "Features" in md and "Security" in md

    def test_section_order(self) -> None:
        md = render_changelog(self._commits(), "v2.1.0")
        assert md.index("Breaking") < md.index("Security") < md.index("Features")

    def test_contributors(self) -> None:
        md = render_changelog(self._commits(), "v2.1.0")
        assert "Alice" in md and "Bob" in md

    def test_empty(self) -> None:
        md = render_changelog([], "v2.1.0")
        assert "[v2.1.0]" in md


# ═════════════════════════════════════════════════════════════
# V5: PR SUMMARY
# ═════════════════════════════════════════════════════════════


class TestPRSummary:
    def test_basic(self) -> None:
        md = render_pr_summary(45.0, 38.0, False, None, "abc123")
        assert "Quality Dashboard" in md and "45.0%" in md

    def test_regression(self) -> None:
        assert "REGRESSION" in render_pr_summary(35.0, 38.0, False, None)

    def test_ratchet(self) -> None:
        md = render_pr_summary(45.0, 38.0, True, 40.0, "abc")
        assert "🔒" in md and "40%" in md

    def test_green_badge(self) -> None:
        assert "🟢" in render_pr_summary(85.0, 38.0, False, None)

    def test_constitutional(self) -> None:
        md = render_pr_summary(42.0, 38.0, False, None)
        assert "Ihsan" in md and "SNR" in md and "ADL" in md

    def test_trend_display(self) -> None:
        md = render_pr_summary(42.0, 38.0, False, None, trend_dir="improving")
        assert "improving" in md
