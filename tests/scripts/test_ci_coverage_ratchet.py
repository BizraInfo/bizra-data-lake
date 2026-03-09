"""
Tests for BIZRA Coverage Ratchet Engine
========================================

Validates the coverage ratchet mechanism: XML parsing, floor detection,
ratchet logic, and evidence chain integrity.
"""

import json
import textwrap
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def coverage_xml(tmp_path: Path) -> Path:
    """Create a sample Cobertura coverage.xml."""
    xml_content = textwrap.dedent("""\
        <?xml version="1.0" ?>
        <coverage version="7.3.2" timestamp="1709910000000"
                  lines-valid="1000" lines-covered="450"
                  line-rate="0.45" branches-valid="200"
                  branches-covered="100" branch-rate="0.50"
                  complexity="0">
          <packages>
            <package name="core" line-rate="0.45">
              <classes>
                <class name="engine.py" filename="core/engine.py" line-rate="0.50">
                  <lines>
                    <line number="1" hits="1"/>
                    <line number="2" hits="0"/>
                  </lines>
                </class>
              </classes>
            </package>
          </packages>
        </coverage>
    """)
    xml_path = tmp_path / "coverage.xml"
    xml_path.write_text(xml_content, encoding="utf-8")
    return xml_path


@pytest.fixture
def pyproject_toml(tmp_path: Path) -> Path:
    """Create a sample pyproject.toml with coverage config."""
    content = textwrap.dedent("""\
        [project]
        name = "bizra-data-lake"
        version = "2.0.0"

        [tool.coverage.run]
        source = ["core"]

        [tool.coverage.report]
        fail_under = 38
        show_missing = true
    """)
    toml_path = tmp_path / "pyproject.toml"
    toml_path.write_text(content, encoding="utf-8")
    return toml_path


@pytest.fixture
def evidence_path(tmp_path: Path) -> Path:
    return tmp_path / "evidence" / "ratchet.jsonl"


# ─────────────────────────────────────────────────────────────
# Tests: XML Parsing
# ─────────────────────────────────────────────────────────────

class TestCoverageXMLParsing:
    """Test Cobertura XML parsing."""

    def test_parse_valid_xml(self, coverage_xml: Path) -> None:
        from scripts.ci_coverage_ratchet import parse_coverage_xml
        result = parse_coverage_xml(coverage_xml)
        assert result == pytest.approx(45.0, abs=0.1)

    def test_parse_missing_file(self, tmp_path: Path) -> None:
        from scripts.ci_coverage_ratchet import parse_coverage_xml
        with pytest.raises(FileNotFoundError):
            parse_coverage_xml(tmp_path / "nonexistent.xml")

    def test_parse_missing_line_rate(self, tmp_path: Path) -> None:
        from scripts.ci_coverage_ratchet import parse_coverage_xml
        xml_path = tmp_path / "bad.xml"
        xml_path.write_text('<coverage></coverage>', encoding="utf-8")
        with pytest.raises(ValueError, match="line-rate"):
            parse_coverage_xml(xml_path)

    def test_parse_zero_coverage(self, tmp_path: Path) -> None:
        from scripts.ci_coverage_ratchet import parse_coverage_xml
        xml_path = tmp_path / "zero.xml"
        xml_path.write_text('<coverage line-rate="0.0"></coverage>', encoding="utf-8")
        assert parse_coverage_xml(xml_path) == 0.0

    def test_parse_full_coverage(self, tmp_path: Path) -> None:
        from scripts.ci_coverage_ratchet import parse_coverage_xml
        xml_path = tmp_path / "full.xml"
        xml_path.write_text('<coverage line-rate="1.0"></coverage>', encoding="utf-8")
        assert parse_coverage_xml(xml_path) == 100.0


# ─────────────────────────────────────────────────────────────
# Tests: pyproject.toml Floor
# ─────────────────────────────────────────────────────────────

class TestPyprojectParsing:
    """Test pyproject.toml floor reading and writing."""

    def test_read_floor(self, pyproject_toml: Path) -> None:
        from scripts.ci_coverage_ratchet import read_coverage_floor
        assert read_coverage_floor(pyproject_toml) == 38.0

    def test_read_floor_missing(self, tmp_path: Path) -> None:
        from scripts.ci_coverage_ratchet import read_coverage_floor
        toml = tmp_path / "empty.toml"
        toml.write_text("[project]\nname = 'test'\n", encoding="utf-8")
        with pytest.raises(ValueError, match="fail_under"):
            read_coverage_floor(toml)

    def test_write_floor(self, pyproject_toml: Path) -> None:
        from scripts.ci_coverage_ratchet import read_coverage_floor, write_coverage_floor
        write_coverage_floor(pyproject_toml, 42)
        assert read_coverage_floor(pyproject_toml) == 42.0

    def test_write_preserves_context(self, pyproject_toml: Path) -> None:
        from scripts.ci_coverage_ratchet import write_coverage_floor
        write_coverage_floor(pyproject_toml, 50)
        content = pyproject_toml.read_text(encoding="utf-8")
        assert "show_missing = true" in content  # Other settings preserved
        assert "fail_under = 50" in content


# ─────────────────────────────────────────────────────────────
# Tests: Ratchet Logic
# ─────────────────────────────────────────────────────────────

class TestRatchetLogic:
    """Test the core ratchet evaluation logic."""

    def test_no_ratchet_below_step(self) -> None:
        from scripts.ci_coverage_ratchet import evaluate_ratchet
        result = evaluate_ratchet(actual=38.5, floor=38.0, step=1)
        assert not result.ratcheted
        assert not result.regression
        assert result.new_floor is None

    def test_ratchet_triggers_at_step(self) -> None:
        from scripts.ci_coverage_ratchet import evaluate_ratchet
        result = evaluate_ratchet(actual=40.0, floor=38.0, step=1)
        assert result.ratcheted
        assert result.new_floor == 40

    def test_ratchet_caps_at_max_bump(self) -> None:
        from scripts.ci_coverage_ratchet import evaluate_ratchet
        # Actual jumped 10%, but max bump is 5
        result = evaluate_ratchet(actual=50.0, floor=38.0, step=1)
        assert result.ratcheted
        assert result.new_floor == 43  # 38 + 5 (MAX_RATCHET_BUMP)

    def test_regression_detected(self) -> None:
        from scripts.ci_coverage_ratchet import evaluate_ratchet
        result = evaluate_ratchet(actual=35.0, floor=38.0, step=1)
        assert result.regression
        assert not result.ratcheted
        assert result.headroom == pytest.approx(-3.0)

    def test_exact_floor_no_ratchet(self) -> None:
        from scripts.ci_coverage_ratchet import evaluate_ratchet
        result = evaluate_ratchet(actual=38.0, floor=38.0, step=1)
        assert not result.ratcheted
        assert not result.regression

    def test_custom_step_size(self) -> None:
        from scripts.ci_coverage_ratchet import evaluate_ratchet
        # With step=3, need 3% gain to trigger
        result = evaluate_ratchet(actual=40.0, floor=38.0, step=3)
        assert not result.ratcheted  # Only 2% gain < 3% step

        result = evaluate_ratchet(actual=41.0, floor=38.0, step=3)
        assert result.ratcheted
        assert result.new_floor == 41

    def test_evidence_hash_unique(self) -> None:
        from scripts.ci_coverage_ratchet import evaluate_ratchet
        r1 = evaluate_ratchet(actual=40.0, floor=38.0)
        r2 = evaluate_ratchet(actual=41.0, floor=38.0)
        assert r1.evidence_hash != r2.evidence_hash

    def test_evidence_hash_deterministic(self) -> None:
        from scripts.ci_coverage_ratchet import RatchetResult
        # Same inputs → same hash (excluding timestamp)
        r1 = RatchetResult(
            timestamp="2025-01-01T00:00:00Z",
            actual_coverage=40.0,
            current_floor=38.0,
            new_floor=40.0,
            ratcheted=True,
            regression=False,
            headroom=2.0,
            applied=False,
        )
        r2 = RatchetResult(
            timestamp="2025-01-01T00:00:00Z",
            actual_coverage=40.0,
            current_floor=38.0,
            new_floor=40.0,
            ratcheted=True,
            regression=False,
            headroom=2.0,
            applied=False,
        )
        assert r1.evidence_hash == r2.evidence_hash


# ─────────────────────────────────────────────────────────────
# Tests: Evidence Chain
# ─────────────────────────────────────────────────────────────

class TestEvidenceChain:
    """Test append-only evidence logging."""

    def test_append_creates_file(self, evidence_path: Path) -> None:
        from scripts.ci_coverage_ratchet import RatchetResult, append_evidence
        result = RatchetResult(
            timestamp="2025-01-01T00:00:00Z",
            actual_coverage=42.0,
            current_floor=38.0,
            new_floor=42.0,
            ratcheted=True,
            regression=False,
            headroom=4.0,
            applied=True,
        )
        append_evidence(result, evidence_path)
        assert evidence_path.exists()
        data = json.loads(evidence_path.read_text(encoding="utf-8").strip())
        assert data["actual_coverage"] == 42.0
        assert data["ratcheted"] is True

    def test_append_is_additive(self, evidence_path: Path) -> None:
        from scripts.ci_coverage_ratchet import RatchetResult, append_evidence
        for i in range(3):
            result = RatchetResult(
                timestamp=f"2025-01-0{i+1}T00:00:00Z",
                actual_coverage=38.0 + i,
                current_floor=38.0,
                new_floor=None,
                ratcheted=False,
                regression=False,
                headroom=float(i),
                applied=False,
            )
            append_evidence(result, evidence_path)

        lines = evidence_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3


# ─────────────────────────────────────────────────────────────
# Tests: Multi-Language Aggregation
# ─────────────────────────────────────────────────────────────

class TestMultiLanguageCoverage:
    """Test cross-language coverage aggregation."""

    def test_python_only(self, coverage_xml: Path) -> None:
        from scripts.ci_coverage_ratchet import aggregate_coverage
        result = aggregate_coverage(python_xml=coverage_xml)
        assert "python" in result
        assert result["python"] == pytest.approx(45.0, abs=0.1)
        assert "aggregate" in result

    def test_lcov_parsing(self, tmp_path: Path) -> None:
        from scripts.ci_coverage_ratchet import _parse_lcov_coverage
        lcov = tmp_path / "lcov.info"
        lcov.write_text("SF:src/lib.rs\nDA:1,1\nDA:2,0\nDA:3,1\nLF:3\nLH:2\nend_of_record\n")
        result = _parse_lcov_coverage(lcov)
        assert result == pytest.approx(66.67, abs=0.1)

    def test_istanbul_parsing(self, tmp_path: Path) -> None:
        from scripts.ci_coverage_ratchet import _parse_istanbul_coverage
        istanbul = tmp_path / "coverage-final.json"
        istanbul.write_text(json.dumps({
            "src/App.tsx": {
                "s": {"0": 1, "1": 1, "2": 0, "3": 1},
            }
        }), encoding="utf-8")
        result = _parse_istanbul_coverage(istanbul)
        assert result == 75.0

    def test_aggregate_all_languages(self, coverage_xml: Path, tmp_path: Path) -> None:
        from scripts.ci_coverage_ratchet import aggregate_coverage
        lcov = tmp_path / "lcov.info"
        lcov.write_text("LF:100\nLH:60\n")
        istanbul = tmp_path / "coverage.json"
        istanbul.write_text(json.dumps({
            "file.tsx": {"s": {"0": 1, "1": 1, "2": 0}},
        }))

        result = aggregate_coverage(
            python_xml=coverage_xml,
            rust_lcov=lcov,
            frontend_json=istanbul,
        )
        assert "python" in result
        assert "rust" in result
        assert "frontend" in result
        assert "aggregate" in result
        # Aggregate is mean of all three
        expected_avg = (45.0 + 60.0 + 66.67) / 3
        assert result["aggregate"] == pytest.approx(expected_avg, abs=0.5)

    def test_missing_sources_skipped(self, tmp_path: Path) -> None:
        from scripts.ci_coverage_ratchet import aggregate_coverage
        result = aggregate_coverage(
            python_xml=tmp_path / "nonexistent.xml",
            rust_lcov=tmp_path / "nonexistent.info",
        )
        assert result == {}
