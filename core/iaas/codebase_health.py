"""
Codebase Health Audit — Multi-Lens Professional Analysis
=========================================================
Comprehensive codebase health auditor examining code quality across
seven independent lenses. Each lens produces a normalized score (0.0–1.0)
and actionable findings. The aggregate score maps to BIZRA constitutional
thresholds (Ihsan ≥ 0.95, SNR ≥ 0.85).

Lenses:
  1. Structure   — Folder/module organization, __init__.py presence
  2. Docstring   — Function/class documentation coverage
  3. Type Hint   — Type annotation coverage on function signatures
  4. Complexity  — Function length and nesting depth
  5. Import      — Import hygiene and constants.py compliance
  6. Constitutional — BIZRA threshold compliance (Ihsan, SNR, ADL)
  7. Test Coverage — Test file presence per source module

Usage:
    from core.iaas.codebase_health import CodebaseHealthAuditor

    auditor = CodebaseHealthAuditor("/path/to/repo")
    report = auditor.audit()
    print(report.overall_score)   # 0.0 – 1.0
    print(report.to_dict())       # JSON-serializable

Standing on Giants: McCabe (1976) · Halstead (1977) · Martin (Clean Code, 2008)
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum function body length (lines) before penalty
_MAX_FUNCTION_LINES: int = 50

# Maximum nesting depth before penalty
_MAX_NESTING_DEPTH: int = 4

# Files/dirs to skip during scanning
_SKIP_DIRS: frozenset = frozenset(
    {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "egg-info",
        ".eggs",
        "dist",
        "build",
    }
)

# BIZRA-specific constants patterns that should be imported from constants.py
_CONSTITUTIONAL_PATTERNS: List[str] = [
    "IHSAN_THRESHOLD",
    "SNR_THRESHOLD",
    "ADL_GINI_THRESHOLD",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════


class AuditSeverity(str, Enum):
    """Severity level for audit findings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class HealthGrade(str, Enum):
    """Overall health grade aligned with BIZRA quality tiers."""

    ELITE = "elite"  # ≥ 0.95 (Ihsan-grade)
    HIGH = "high"  # ≥ 0.85 (SNR museum floor)
    ADEQUATE = "adequate"  # ≥ 0.70
    DEGRADED = "degraded"  # ≥ 0.50
    CRITICAL = "critical"  # < 0.50


@dataclass
class AuditFinding:
    """Single finding from a lens audit."""

    lens: str
    severity: AuditSeverity
    message: str
    file_path: str = ""
    line: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "lens": self.lens,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.file_path:
            d["file"] = self.file_path
        if self.line > 0:
            d["line"] = self.line
        return d


@dataclass
class LensResult:
    """Result from a single audit lens."""

    name: str
    score: float  # 0.0 – 1.0
    items_checked: int = 0
    items_passed: int = 0
    findings: List[AuditFinding] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "items_checked": self.items_checked,
            "items_passed": self.items_passed,
            "findings_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class HealthAuditReport:
    """Complete multi-lens health audit report."""

    root_path: str = ""
    timestamp_iso: str = ""
    overall_score: float = 0.0
    grade: HealthGrade = HealthGrade.CRITICAL
    lenses: Dict[str, LensResult] = field(default_factory=dict)
    summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_path": self.root_path,
            "timestamp": self.timestamp_iso,
            "overall_score": round(self.overall_score, 4),
            "grade": self.grade.value,
            "lenses": {k: v.to_dict() for k, v in self.lenses.items()},
            "summary": self.summary,
        }


def _score_to_grade(score: float) -> HealthGrade:
    """Map a 0.0–1.0 score to a HealthGrade."""
    if score >= 0.95:
        return HealthGrade.ELITE
    if score >= 0.85:
        return HealthGrade.HIGH
    if score >= 0.70:
        return HealthGrade.ADEQUATE
    if score >= 0.50:
        return HealthGrade.DEGRADED
    return HealthGrade.CRITICAL


# ═══════════════════════════════════════════════════════════════════════════════
# AST Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_file(path: Path) -> Optional[ast.Module]:
    """Safely parse a Python file, returning None on failure."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        return ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None


def _max_nesting(node: ast.AST, depth: int = 0) -> int:
    """Compute maximum nesting depth inside an AST node."""
    max_d = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(
            child,
            (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.ExceptHandler),
        ):
            max_d = max(max_d, _max_nesting(child, depth + 1))
        else:
            max_d = max(max_d, _max_nesting(child, depth))
    return max_d


def _function_line_count(node: ast.FunctionDef) -> int:
    """Return the body line span of a function definition."""
    if not node.body:
        return 0
    first = node.body[0].lineno
    last = node.body[-1].end_lineno or node.body[-1].lineno
    return last - first + 1


# ═══════════════════════════════════════════════════════════════════════════════
# Individual Lenses
# ═══════════════════════════════════════════════════════════════════════════════


def _collect_python_files(root: Path, subdir: str = "") -> List[Path]:
    """Collect .py files under root/subdir, skipping excluded directories."""
    target = root / subdir if subdir else root
    if not target.is_dir():
        return []
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(target):
        # Prune excluded directories in-place
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                files.append(Path(dirpath) / fn)
    return files


def lens_structure(root: Path, source_dir: str = "core") -> LensResult:
    """
    Structure Lens — Verify folder organization and __init__.py presence.

    Checks every directory under source_dir that contains .py files
    also contains an __init__.py (proper Python package).
    """
    lens_name = "structure"
    findings: List[AuditFinding] = []
    source = root / source_dir
    if not source.is_dir():
        return LensResult(
            name=lens_name,
            score=0.0,
            findings=[
                AuditFinding(
                    lens=lens_name,
                    severity=AuditSeverity.ERROR,
                    message=f"Source directory '{source_dir}/' not found",
                )
            ],
        )

    checked = 0
    passed = 0
    for dirpath, dirnames, filenames in os.walk(source):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        py_files = [f for f in filenames if f.endswith(".py")]
        if not py_files:
            continue
        checked += 1
        rel = os.path.relpath(dirpath, root)
        if "__init__.py" in filenames:
            passed += 1
        else:
            findings.append(
                AuditFinding(
                    lens=lens_name,
                    severity=AuditSeverity.WARNING,
                    message="Missing __init__.py in package directory",
                    file_path=rel,
                )
            )

    score = passed / max(checked, 1)
    return LensResult(
        name=lens_name,
        score=score,
        items_checked=checked,
        items_passed=passed,
        findings=findings,
    )


def lens_docstring(root: Path, source_dir: str = "core") -> LensResult:
    """
    Docstring Lens — Measure documentation coverage on functions and classes.

    Checks that public functions (not starting with _) and all classes
    have docstrings.
    """
    lens_name = "docstring"
    findings: List[AuditFinding] = []
    checked = 0
    passed = 0

    for path in _collect_python_files(root, source_dir):
        tree = _parse_file(path)
        if tree is None:
            continue
        rel = str(path.relative_to(root))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip private/dunder helpers
                if node.name.startswith("_"):
                    continue
                checked += 1
                if ast.get_docstring(node):
                    passed += 1
                else:
                    findings.append(
                        AuditFinding(
                            lens=lens_name,
                            severity=AuditSeverity.INFO,
                            message=f"Public function '{node.name}' lacks docstring",
                            file_path=rel,
                            line=node.lineno,
                        )
                    )
            elif isinstance(node, ast.ClassDef):
                checked += 1
                if ast.get_docstring(node):
                    passed += 1
                else:
                    findings.append(
                        AuditFinding(
                            lens=lens_name,
                            severity=AuditSeverity.INFO,
                            message=f"Class '{node.name}' lacks docstring",
                            file_path=rel,
                            line=node.lineno,
                        )
                    )

    score = passed / max(checked, 1)
    return LensResult(
        name=lens_name,
        score=score,
        items_checked=checked,
        items_passed=passed,
        findings=findings,
    )


def lens_type_hints(root: Path, source_dir: str = "core") -> LensResult:
    """
    Type Hint Lens — Measure type annotation coverage on function signatures.

    Checks that functions have a return type annotation and that all
    parameters (except self/cls) have type annotations.
    """
    lens_name = "type_hints"
    findings: List[AuditFinding] = []
    checked = 0
    passed = 0

    for path in _collect_python_files(root, source_dir):
        tree = _parse_file(path)
        if tree is None:
            continue
        rel = str(path.relative_to(root))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Skip private helpers for noise reduction
            if node.name.startswith("_"):
                continue
            checked += 1
            has_return = node.returns is not None
            params = node.args
            annotated_args = 0
            total_args = 0
            for arg in params.args + params.posonlyargs + params.kwonlyargs:
                if arg.arg in ("self", "cls"):
                    continue
                total_args += 1
                if arg.annotation is not None:
                    annotated_args += 1

            fully_typed = has_return and (annotated_args == total_args)
            if fully_typed:
                passed += 1
            else:
                missing: List[str] = []
                if not has_return:
                    missing.append("return type")
                if annotated_args < total_args:
                    missing.append(
                        f"{total_args - annotated_args} param(s)"
                    )
                findings.append(
                    AuditFinding(
                        lens=lens_name,
                        severity=AuditSeverity.INFO,
                        message=(
                            f"Function '{node.name}' missing annotations: "
                            + ", ".join(missing)
                        ),
                        file_path=rel,
                        line=node.lineno,
                    )
                )

    score = passed / max(checked, 1)
    return LensResult(
        name=lens_name,
        score=score,
        items_checked=checked,
        items_passed=passed,
        findings=findings,
    )


def lens_complexity(root: Path, source_dir: str = "core") -> LensResult:
    """
    Complexity Lens — Flag functions exceeding length or nesting thresholds.

    Checks function body length (> 50 lines) and nesting depth (> 4 levels).
    """
    lens_name = "complexity"
    findings: List[AuditFinding] = []
    checked = 0
    passed = 0

    for path in _collect_python_files(root, source_dir):
        tree = _parse_file(path)
        if tree is None:
            continue
        rel = str(path.relative_to(root))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            checked += 1
            lines = _function_line_count(node)
            depth = _max_nesting(node)
            issues: List[str] = []
            if lines > _MAX_FUNCTION_LINES:
                issues.append(f"{lines} lines (max {_MAX_FUNCTION_LINES})")
            if depth > _MAX_NESTING_DEPTH:
                issues.append(
                    f"nesting depth {depth} (max {_MAX_NESTING_DEPTH})"
                )
            if issues:
                findings.append(
                    AuditFinding(
                        lens=lens_name,
                        severity=AuditSeverity.WARNING,
                        message=(
                            f"Function '{node.name}': " + "; ".join(issues)
                        ),
                        file_path=rel,
                        line=node.lineno,
                    )
                )
            else:
                passed += 1

    score = passed / max(checked, 1)
    return LensResult(
        name=lens_name,
        score=score,
        items_checked=checked,
        items_passed=passed,
        findings=findings,
    )


def lens_import_hygiene(root: Path, source_dir: str = "core") -> LensResult:
    """
    Import Lens — Check import patterns and constitutional constant usage.

    Flags files that define their own IHSAN_THRESHOLD / SNR_THRESHOLD
    instead of importing from core.integration.constants.
    """
    lens_name = "import_hygiene"
    findings: List[AuditFinding] = []
    checked = 0
    passed = 0

    constants_path = root / source_dir / "integration" / "constants.py"
    constants_rel = str(constants_path.relative_to(root)) if constants_path.exists() else ""

    for path in _collect_python_files(root, source_dir):
        tree = _parse_file(path)
        if tree is None:
            continue
        rel = str(path.relative_to(root))
        # Skip the constants file itself
        if constants_rel and rel == constants_rel:
            continue
        checked += 1
        ok = True
        for pattern in _CONSTITUTIONAL_PATTERNS:
            # Check for local re-definition (assignment, not import)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == pattern:
                            # This file redefines a constitutional constant
                            findings.append(
                                AuditFinding(
                                    lens=lens_name,
                                    severity=AuditSeverity.ERROR,
                                    message=(
                                        f"Redefines '{pattern}' locally — "
                                        "must import from core.integration.constants"
                                    ),
                                    file_path=rel,
                                    line=node.lineno,
                                )
                            )
                            ok = False
        if ok:
            passed += 1

    score = passed / max(checked, 1)
    return LensResult(
        name=lens_name,
        score=score,
        items_checked=checked,
        items_passed=passed,
        findings=findings,
    )


def lens_constitutional(root: Path) -> LensResult:
    """
    Constitutional Lens — Verify BIZRA constitutional infrastructure.

    Checks:
      - core/integration/constants.py exists and defines canonical thresholds
      - Ihsan threshold is 0.95
      - SNR threshold is 0.85
      - ADL Gini threshold is defined
    """
    lens_name = "constitutional"
    findings: List[AuditFinding] = []
    checked = 0
    passed = 0

    constants_path = root / "core" / "integration" / "constants.py"
    checked += 1
    if not constants_path.exists():
        findings.append(
            AuditFinding(
                lens=lens_name,
                severity=AuditSeverity.ERROR,
                message="core/integration/constants.py not found",
            )
        )
        return LensResult(
            name=lens_name,
            score=0.0,
            items_checked=checked,
            items_passed=0,
            findings=findings,
        )
    passed += 1

    tree = _parse_file(constants_path)
    if tree is None:
        findings.append(
            AuditFinding(
                lens=lens_name,
                severity=AuditSeverity.ERROR,
                message="core/integration/constants.py failed to parse",
            )
        )
        return LensResult(
            name=lens_name,
            score=0.5,
            items_checked=checked,
            items_passed=passed,
            findings=findings,
        )

    # Check canonical thresholds are defined
    defined_names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined_names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined_names.add(node.target.id)

    expected = {
        "IHSAN_THRESHOLD": "Ihsan production threshold",
        "SNR_THRESHOLD": "SNR minimum threshold",
        "ADL_GINI_THRESHOLD": "ADL justice Gini threshold",
        "UNIFIED_IHSAN_THRESHOLD": "Unified Ihsan threshold",
        "UNIFIED_SNR_THRESHOLD": "Unified SNR threshold",
    }

    for name, desc in expected.items():
        checked += 1
        if name in defined_names:
            passed += 1
        else:
            findings.append(
                AuditFinding(
                    lens=lens_name,
                    severity=AuditSeverity.ERROR,
                    message=f"Missing constitutional constant: {name} ({desc})",
                    file_path="core/integration/constants.py",
                )
            )

    score = passed / max(checked, 1)
    return LensResult(
        name=lens_name,
        score=score,
        items_checked=checked,
        items_passed=passed,
        findings=findings,
    )


def lens_test_coverage(
    root: Path, source_dir: str = "core", test_dir: str = "tests"
) -> LensResult:
    """
    Test Coverage Lens — Check that source modules have corresponding tests.

    For each sub-package under source_dir, verifies a matching directory
    or test file exists under test_dir.
    """
    lens_name = "test_coverage"
    findings: List[AuditFinding] = []

    source = root / source_dir
    tests = root / test_dir / source_dir
    if not source.is_dir():
        return LensResult(name=lens_name, score=0.0, findings=[])

    # Collect direct sub-packages (directories with __init__.py)
    sub_packages: List[str] = []
    for entry in sorted(source.iterdir()):
        if entry.is_dir() and (entry / "__init__.py").exists():
            if entry.name not in _SKIP_DIRS:
                sub_packages.append(entry.name)

    checked = len(sub_packages)
    passed = 0
    for pkg in sub_packages:
        # Check for test directory or any test_*.py matching the package
        has_test_dir = (tests / pkg).is_dir()
        has_test_file = any(
            (root / test_dir).rglob(f"test_{pkg}*.py")
        ) if (root / test_dir).is_dir() else False

        if has_test_dir or has_test_file:
            passed += 1
        else:
            findings.append(
                AuditFinding(
                    lens=lens_name,
                    severity=AuditSeverity.WARNING,
                    message=f"No test directory/file found for '{source_dir}/{pkg}'",
                    file_path=f"{test_dir}/{source_dir}/{pkg}",
                )
            )

    score = passed / max(checked, 1)
    return LensResult(
        name=lens_name,
        score=score,
        items_checked=checked,
        items_passed=passed,
        findings=findings,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Auditor
# ═══════════════════════════════════════════════════════════════════════════════


class CodebaseHealthAuditor:
    """
    Multi-lens professional codebase health auditor.

    Examines a Python codebase from seven independent perspectives
    and produces a unified health score aligned with BIZRA constitutional
    thresholds.

    Usage:
        auditor = CodebaseHealthAuditor("/path/to/repo")
        report = auditor.audit()
        print(report.grade)  # "elite" | "high" | "adequate" | "degraded" | "critical"
    """

    # Default lens weights (sum to 1.0)
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "structure": 0.15,
        "docstring": 0.15,
        "type_hints": 0.10,
        "complexity": 0.20,
        "import_hygiene": 0.15,
        "constitutional": 0.15,
        "test_coverage": 0.10,
    }

    def __init__(
        self,
        root: str,
        source_dir: str = "core",
        test_dir: str = "tests",
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.source_dir = source_dir
        self.test_dir = test_dir
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)

    def audit(self) -> HealthAuditReport:
        """Run all lenses and produce a unified health report."""
        report = HealthAuditReport(
            root_path=str(self.root),
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
        )

        # Execute each lens
        report.lenses["structure"] = lens_structure(self.root, self.source_dir)
        report.lenses["docstring"] = lens_docstring(self.root, self.source_dir)
        report.lenses["type_hints"] = lens_type_hints(self.root, self.source_dir)
        report.lenses["complexity"] = lens_complexity(self.root, self.source_dir)
        report.lenses["import_hygiene"] = lens_import_hygiene(
            self.root, self.source_dir
        )
        report.lenses["constitutional"] = lens_constitutional(self.root)
        report.lenses["test_coverage"] = lens_test_coverage(
            self.root, self.source_dir, self.test_dir
        )

        # Compute weighted overall score
        total_weight = 0.0
        weighted_sum = 0.0
        for lens_name, result in report.lenses.items():
            w = self.weights.get(lens_name, 0.0)
            weighted_sum += result.score * w
            total_weight += w

        report.overall_score = weighted_sum / max(total_weight, 1e-9)
        report.grade = _score_to_grade(report.overall_score)

        # Summary counts
        total_findings = 0
        errors = 0
        warnings = 0
        infos = 0
        for result in report.lenses.values():
            for finding in result.findings:
                total_findings += 1
                if finding.severity == AuditSeverity.ERROR:
                    errors += 1
                elif finding.severity == AuditSeverity.WARNING:
                    warnings += 1
                else:
                    infos += 1

        report.summary = {
            "total_findings": total_findings,
            "errors": errors,
            "warnings": warnings,
            "infos": infos,
            "lenses_run": len(report.lenses),
        }

        return report

    def audit_single_lens(self, lens_name: str) -> LensResult:
        """Run a single lens by name."""
        lens_map = {
            "structure": lambda: lens_structure(self.root, self.source_dir),
            "docstring": lambda: lens_docstring(self.root, self.source_dir),
            "type_hints": lambda: lens_type_hints(self.root, self.source_dir),
            "complexity": lambda: lens_complexity(self.root, self.source_dir),
            "import_hygiene": lambda: lens_import_hygiene(
                self.root, self.source_dir
            ),
            "constitutional": lambda: lens_constitutional(self.root),
            "test_coverage": lambda: lens_test_coverage(
                self.root, self.source_dir, self.test_dir
            ),
        }
        if lens_name not in lens_map:
            raise ValueError(
                f"Unknown lens '{lens_name}'. "
                f"Available: {sorted(lens_map.keys())}"
            )
        return lens_map[lens_name]()
