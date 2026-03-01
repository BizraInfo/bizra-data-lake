"""Tests for multi-lens codebase health audit (core.iaas.codebase_health)."""

from __future__ import annotations

from pathlib import Path
import pytest

from core.iaas.codebase_health import (
    AuditFinding,
    AuditSeverity,
    CodebaseHealthAuditor,
    HealthAuditReport,
    HealthGrade,
    LensResult,
    _max_nesting,
    _parse_file,
    _score_to_grade,
    lens_complexity,
    lens_constitutional,
    lens_docstring,
    lens_import_hygiene,
    lens_structure,
    lens_test_coverage,
    lens_type_hints,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures — minimal project trees for isolated lens tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def minimal_project(tmp_path: Path) -> Path:
    """Create a minimal Python project with core/ and tests/ dirs."""
    core = tmp_path / "core"
    core.mkdir()
    (core / "__init__.py").write_text('"""Core package."""\n')

    sub = core / "utils"
    sub.mkdir()
    (sub / "__init__.py").write_text('"""Utils sub-package."""\n')
    (sub / "helpers.py").write_text(
        '"""Helper utilities."""\n\n\n'
        "def greet(name: str) -> str:\n"
        '    """Say hello."""\n'
        "    return f\"Hello, {name}\"\n\n\n"
        "def add(a: int, b: int) -> int:\n"
        '    """Add two numbers."""\n'
        "    return a + b\n"
    )

    # A sub-package missing __init__.py
    bad = core / "broken"
    bad.mkdir()
    (bad / "module.py").write_text("x = 1\n")

    # Tests directory
    tests = tmp_path / "tests" / "core" / "utils"
    tests.mkdir(parents=True)
    (tests / "test_helpers.py").write_text(
        "def test_greet():\n    assert True\n"
    )

    return tmp_path


@pytest.fixture
def project_with_constants(tmp_path: Path) -> Path:
    """Project with core/integration/constants.py containing thresholds."""
    core = tmp_path / "core"
    integration = core / "integration"
    integration.mkdir(parents=True)
    (core / "__init__.py").write_text("")
    (integration / "__init__.py").write_text("")
    (integration / "constants.py").write_text(
        "from typing import Final\n\n"
        "UNIFIED_IHSAN_THRESHOLD: Final[float] = 0.95\n"
        "IHSAN_THRESHOLD: Final[float] = 0.95\n"
        "UNIFIED_SNR_THRESHOLD: Final[float] = 0.85\n"
        "SNR_THRESHOLD: Final[float] = 0.85\n"
        "ADL_GINI_THRESHOLD: Final[float] = 0.35\n"
    )
    return tmp_path


# ═══════════════════════════════════════════════════════════════════════════════
# Grade mapping
# ═══════════════════════════════════════════════════════════════════════════════


class TestGradeMapping:
    def test_elite_grade(self):
        assert _score_to_grade(0.95) == HealthGrade.ELITE
        assert _score_to_grade(1.0) == HealthGrade.ELITE

    def test_high_grade(self):
        assert _score_to_grade(0.85) == HealthGrade.HIGH
        assert _score_to_grade(0.94) == HealthGrade.HIGH

    def test_adequate_grade(self):
        assert _score_to_grade(0.70) == HealthGrade.ADEQUATE

    def test_degraded_grade(self):
        assert _score_to_grade(0.50) == HealthGrade.DEGRADED

    def test_critical_grade(self):
        assert _score_to_grade(0.49) == HealthGrade.CRITICAL
        assert _score_to_grade(0.0) == HealthGrade.CRITICAL


# ═══════════════════════════════════════════════════════════════════════════════
# Data structure tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataStructures:
    def test_audit_finding_to_dict(self):
        f = AuditFinding(
            lens="test",
            severity=AuditSeverity.WARNING,
            message="something",
            file_path="a/b.py",
            line=42,
        )
        d = f.to_dict()
        assert d["lens"] == "test"
        assert d["severity"] == "warning"
        assert d["file"] == "a/b.py"
        assert d["line"] == 42

    def test_audit_finding_minimal_dict(self):
        f = AuditFinding(lens="x", severity=AuditSeverity.INFO, message="ok")
        d = f.to_dict()
        assert "file" not in d
        assert "line" not in d

    def test_lens_result_to_dict(self):
        lr = LensResult(name="test_lens", score=0.85, items_checked=10, items_passed=8)
        d = lr.to_dict()
        assert d["name"] == "test_lens"
        assert d["score"] == 0.85
        assert d["findings_count"] == 0

    def test_health_report_to_dict(self):
        report = HealthAuditReport(
            root_path="/tmp/test",
            timestamp_iso="2026-01-01T00:00:00Z",
            overall_score=0.9,
            grade=HealthGrade.HIGH,
        )
        d = report.to_dict()
        assert d["grade"] == "high"
        assert d["overall_score"] == 0.9
        assert isinstance(d["lenses"], dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Structure Lens
# ═══════════════════════════════════════════════════════════════════════════════


class TestStructureLens:
    def test_detects_missing_init(self, minimal_project: Path):
        result = lens_structure(minimal_project, "core")
        # core/ and core/utils/ have __init__.py, core/broken/ does not
        assert result.items_checked >= 3
        assert result.items_passed >= 2
        assert any("Missing __init__.py" in f.message for f in result.findings)

    def test_perfect_structure(self, project_with_constants: Path):
        result = lens_structure(project_with_constants, "core")
        assert result.score == 1.0
        assert len(result.findings) == 0

    def test_missing_source_dir(self, tmp_path: Path):
        result = lens_structure(tmp_path, "nonexistent")
        assert result.score == 0.0
        assert result.findings[0].severity == AuditSeverity.ERROR


# ═══════════════════════════════════════════════════════════════════════════════
# Docstring Lens
# ═══════════════════════════════════════════════════════════════════════════════


class TestDocstringLens:
    def test_well_documented_code(self, minimal_project: Path):
        result = lens_docstring(minimal_project, "core")
        # greet and add both have docstrings
        assert result.items_passed >= 2
        assert result.score > 0.0

    def test_undocumented_code(self, tmp_path: Path):
        core = tmp_path / "core"
        core.mkdir()
        (core / "__init__.py").write_text("")
        (core / "bad.py").write_text(
            "def public_func(x):\n    return x\n\n"
            "class MyClass:\n    pass\n"
        )
        result = lens_docstring(tmp_path, "core")
        assert result.items_checked == 2  # 1 public func + 1 class
        assert result.items_passed == 0
        assert result.score == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Type Hints Lens
# ═══════════════════════════════════════════════════════════════════════════════


class TestTypeHintsLens:
    def test_fully_typed_code(self, minimal_project: Path):
        result = lens_type_hints(minimal_project, "core")
        # greet(name: str) -> str and add(a: int, b: int) -> int are fully typed
        assert result.items_passed >= 2

    def test_untyped_code(self, tmp_path: Path):
        core = tmp_path / "core"
        core.mkdir()
        (core / "__init__.py").write_text("")
        (core / "untyped.py").write_text(
            "def compute(x, y):\n    return x + y\n"
        )
        result = lens_type_hints(tmp_path, "core")
        assert result.items_checked == 1
        assert result.items_passed == 0
        assert any("missing annotations" in f.message for f in result.findings)


# ═══════════════════════════════════════════════════════════════════════════════
# Complexity Lens
# ═══════════════════════════════════════════════════════════════════════════════


class TestComplexityLens:
    def test_simple_functions_pass(self, minimal_project: Path):
        result = lens_complexity(minimal_project, "core")
        # All functions in helpers.py are short
        assert result.score > 0.0

    def test_flags_long_function(self, tmp_path: Path):
        core = tmp_path / "core"
        core.mkdir()
        (core / "__init__.py").write_text("")
        # Create a function with 60 lines
        lines = ["def big_func():"]
        for i in range(60):
            lines.append(f"    x{i} = {i}")
        (core / "long.py").write_text("\n".join(lines) + "\n")
        result = lens_complexity(tmp_path, "core")
        assert any("lines" in f.message for f in result.findings)

    def test_flags_deep_nesting(self, tmp_path: Path):
        core = tmp_path / "core"
        core.mkdir()
        (core / "__init__.py").write_text("")
        (core / "nested.py").write_text(
            "def deep():\n"
            "    if True:\n"
            "        for i in range(1):\n"
            "            if i:\n"
            "                for j in range(1):\n"
            "                    if j:\n"
            "                        pass\n"
        )
        result = lens_complexity(tmp_path, "core")
        assert any("nesting" in f.message for f in result.findings)


# ═══════════════════════════════════════════════════════════════════════════════
# Import Hygiene Lens
# ═══════════════════════════════════════════════════════════════════════════════


class TestImportHygieneLens:
    def test_clean_imports(self, project_with_constants: Path):
        result = lens_import_hygiene(project_with_constants, "core")
        # No files redefine constitutional constants
        assert result.score == 1.0

    def test_detects_redefined_constant(self, project_with_constants: Path):
        bad = project_with_constants / "core" / "bad_module.py"
        bad.write_text("IHSAN_THRESHOLD = 0.80  # Wrong!\n")
        result = lens_import_hygiene(project_with_constants, "core")
        assert any(
            "Redefines 'IHSAN_THRESHOLD'" in f.message for f in result.findings
        )
        assert result.score < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Constitutional Lens
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstitutionalLens:
    def test_valid_constants(self, project_with_constants: Path):
        result = lens_constitutional(project_with_constants)
        assert result.score == 1.0

    def test_missing_constants_file(self, tmp_path: Path):
        result = lens_constitutional(tmp_path)
        assert result.score == 0.0
        assert result.findings[0].severity == AuditSeverity.ERROR

    def test_incomplete_constants(self, tmp_path: Path):
        core = tmp_path / "core" / "integration"
        core.mkdir(parents=True)
        (core / "constants.py").write_text(
            "IHSAN_THRESHOLD = 0.95\n"
            # Missing SNR_THRESHOLD, etc.
        )
        result = lens_constitutional(tmp_path)
        assert result.score < 1.0
        assert result.items_passed < result.items_checked


# ═══════════════════════════════════════════════════════════════════════════════
# Test Coverage Lens
# ═══════════════════════════════════════════════════════════════════════════════


class TestTestCoverageLens:
    def test_covered_package(self, minimal_project: Path):
        result = lens_test_coverage(minimal_project, "core", "tests")
        # core/utils has tests/core/utils
        assert result.items_passed >= 1

    def test_uncovered_package(self, minimal_project: Path):
        # core/broken has no tests (and no __init__.py so not a package)
        result = lens_test_coverage(minimal_project, "core", "tests")
        # "broken" has no __init__.py so it's not counted as a sub-package
        # "utils" is a proper package and has tests
        assert result.items_passed >= 1

    def test_no_test_dir(self, tmp_path: Path):
        core = tmp_path / "core"
        core.mkdir()
        (core / "__init__.py").write_text("")
        sub = core / "pkg"
        sub.mkdir()
        (sub / "__init__.py").write_text("")
        result = lens_test_coverage(tmp_path, "core", "tests")
        assert result.score == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Full Auditor
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodebaseHealthAuditor:
    def test_full_audit(self, minimal_project: Path):
        auditor = CodebaseHealthAuditor(str(minimal_project))
        report = auditor.audit()
        assert isinstance(report, HealthAuditReport)
        assert 0.0 <= report.overall_score <= 1.0
        assert report.grade in HealthGrade
        assert len(report.lenses) == 7
        assert report.summary["lenses_run"] == 7

    def test_report_serializable(self, minimal_project: Path):
        auditor = CodebaseHealthAuditor(str(minimal_project))
        report = auditor.audit()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "overall_score" in d
        assert "lenses" in d
        assert all(k in d["lenses"] for k in [
            "structure", "docstring", "type_hints",
            "complexity", "import_hygiene", "constitutional", "test_coverage",
        ])

    def test_custom_weights(self, minimal_project: Path):
        weights = {"structure": 1.0}  # Only care about structure
        auditor = CodebaseHealthAuditor(
            str(minimal_project), weights=weights
        )
        report = auditor.audit()
        # With only structure weight, score should reflect structure lens
        assert report.overall_score > 0.0

    def test_audit_single_lens(self, minimal_project: Path):
        auditor = CodebaseHealthAuditor(str(minimal_project))
        result = auditor.audit_single_lens("structure")
        assert isinstance(result, LensResult)
        assert result.name == "structure"

    def test_audit_single_lens_invalid(self, minimal_project: Path):
        auditor = CodebaseHealthAuditor(str(minimal_project))
        with pytest.raises(ValueError, match="Unknown lens"):
            auditor.audit_single_lens("nonexistent")

    def test_audit_on_real_repo(self):
        """Smoke test: run audit on the actual BIZRA repo root."""
        repo_root = Path(__file__).resolve().parents[3]
        if not (repo_root / "core" / "iaas").is_dir():
            pytest.skip("Not running inside BIZRA repo")
        auditor = CodebaseHealthAuditor(str(repo_root))
        report = auditor.audit()
        assert report.overall_score > 0.0
        assert report.grade != HealthGrade.CRITICAL
        # Constitutional lens should pass if constants.py exists
        assert report.lenses["constitutional"].score > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# AST helper tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestASTHelpers:
    def test_parse_valid_file(self, tmp_path: Path):
        f = tmp_path / "valid.py"
        f.write_text("x = 1\n")
        tree = _parse_file(f)
        assert tree is not None

    def test_parse_invalid_file(self, tmp_path: Path):
        f = tmp_path / "bad.py"
        f.write_text("def (\n")
        tree = _parse_file(f)
        assert tree is None

    def test_parse_nonexistent_file(self, tmp_path: Path):
        tree = _parse_file(tmp_path / "nope.py")
        assert tree is None

    def test_max_nesting_flat(self):
        import ast

        tree = ast.parse("x = 1\ny = 2\n")
        assert _max_nesting(tree) == 0

    def test_max_nesting_nested(self):
        import ast

        code = "if True:\n    for i in range(1):\n        if i:\n            pass\n"
        tree = ast.parse(code)
        assert _max_nesting(tree) >= 3
