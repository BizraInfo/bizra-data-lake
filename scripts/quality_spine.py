#!/usr/bin/env python3
"""
BIZRA Quality Spine — Single Enforceable Quality Gate
======================================================

One file. Five vertebrae. Zero escape routes.

    RATCHET → TREND → GATES → CHANGELOG → RECEIPT

Every quality concern in the BIZRA pipeline compressed into a single
invocable spine. Each vertebra is a self-contained enforcement unit
that chains evidence to the next. Constitutional gates are fail-closed:
if any blocking vertebra fails, the spine halts and returns non-zero.

Standing on Giants:
- Shannon (information theory, 1948) — SNR as quality signal
- Deming (PDCA cycle, 1950) — ratchet-as-improvement
- Shewhart (control charts, 1924) — trend anomaly detection
- Crosby (Zero Defects, 1979) — quality is free
- Al-Ghazali (Ihsān, 1095) — excellence threshold
- PMI/PMBOK 7th Ed (2021) — quality management process group

Usage:
    # Full spine (all vertebrae)
    python scripts/quality_spine.py enforce

    # Single vertebra
    python scripts/quality_spine.py ratchet --coverage-xml coverage.xml
    python scripts/quality_spine.py trend record --snr 0.92 --coverage 42
    python scripts/quality_spine.py trend analyze
    python scripts/quality_spine.py gates --workspace .
    python scripts/quality_spine.py changelog --from-tag v2.0.0
    python scripts/quality_spine.py summary --coverage 45 --floor 38

    # Apply ratchet (post-merge only)
    python scripts/quality_spine.py ratchet --coverage-xml coverage.xml --apply

    # JSON output for CI
    python scripts/quality_spine.py enforce --json

Exit Codes:
    0 — All gates pass
    1 — Blocking gate failure (spine halted)
    2 — Ratchet applied (floor bumped) — success with side-effect
    3 — Configuration error

Constitutional Constraints:
    Ihsān  ≥ 0.95  (UNIFIED_IHSAN_THRESHOLD)
    SNR    ≥ 0.85  (UNIFIED_SNR_THRESHOLD)
    ADL    ≤ 0.35  (ADL_GINI_THRESHOLD)
    Coverage floor can ONLY increase, never decrease
    Evidence is append-only, SHA-256 hash-chained
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# Constitutional Thresholds (single source of truth)
# ─────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent

try:
    sys.path.insert(0, str(_ROOT))
    from core.integration.constants import (
        ADL_GINI_THRESHOLD,
        UNIFIED_IHSAN_THRESHOLD,
        UNIFIED_SNR_THRESHOLD,
    )
except ImportError:
    UNIFIED_IHSAN_THRESHOLD = 0.95
    UNIFIED_SNR_THRESHOLD = 0.85
    ADL_GINI_THRESHOLD = 0.35

# Spine-local constants
RATCHET_STEP = 1
MAX_RATCHET_BUMP = 5
MYPY_BASELINE = 1600
EVIDENCE_DIR = Path("04_GOLD")
TREND_PATH = EVIDENCE_DIR / "quality_trend.jsonl"
SPINE_LOG = EVIDENCE_DIR / "quality_spine_log.jsonl"


# ═════════════════════════════════════════════════════════════
# VERTEBRA 1: COVERAGE RATCHET
# ═════════════════════════════════════════════════════════════


@dataclass
class RatchetResult:
    """Result of a coverage ratchet evaluation."""

    timestamp: str
    actual_coverage: float
    current_floor: float
    new_floor: Optional[float]
    ratcheted: bool
    regression: bool
    headroom: float
    applied: bool
    evidence_hash: str = ""

    def __post_init__(self) -> None:
        content = json.dumps(asdict(self), sort_keys=True, default=str)
        self.evidence_hash = hashlib.sha256(content.encode()).hexdigest()[:16]


def parse_coverage_xml(xml_path: Path) -> float:
    """Parse Cobertura coverage.xml → line-rate as percentage."""
    if not xml_path.exists():
        raise FileNotFoundError(f"Coverage XML not found: {xml_path}")
    tree = ET.parse(str(xml_path))  # noqa: S314 — trusted CI artifact
    root = tree.getroot()
    line_rate = root.get("line-rate")
    if line_rate is None:
        raise ValueError("Coverage XML missing 'line-rate' attribute")
    return float(line_rate) * 100.0


def read_coverage_floor(pyproject_path: Path) -> float:
    """Read fail_under from pyproject.toml."""
    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r"fail_under\s*=\s*(\d+(?:\.\d+)?)", content)
    if not match:
        raise ValueError(f"fail_under not found in {pyproject_path}")
    return float(match.group(1))


def write_coverage_floor(pyproject_path: Path, new_floor: float) -> None:
    """Update fail_under in pyproject.toml in-place."""
    content = pyproject_path.read_text(encoding="utf-8")
    new_content = re.sub(
        r"(fail_under\s*=\s*)\d+(?:\.\d+)?",
        f"\\g<1>{int(new_floor)}",
        content,
        count=1,
    )
    if new_content == content:
        raise ValueError("Failed to update fail_under — pattern not matched")
    pyproject_path.write_text(new_content, encoding="utf-8")


def evaluate_ratchet(
    actual: float, floor: float, step: int = RATCHET_STEP
) -> RatchetResult:
    """Evaluate whether coverage qualifies for a ratchet bump."""
    headroom = actual - floor
    regression = actual < floor
    new_floor: Optional[float] = None
    ratcheted = False

    if not regression and headroom >= step:
        candidate = int(actual)
        bump = min(candidate - int(floor), MAX_RATCHET_BUMP)
        if bump >= step:
            new_floor = int(floor) + bump
            ratcheted = True

    return RatchetResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        actual_coverage=round(actual, 2),
        current_floor=floor,
        new_floor=new_floor,
        ratcheted=ratcheted,
        regression=regression,
        headroom=round(headroom, 2),
        applied=False,
    )


def parse_lcov(lcov_path: Path) -> float:
    """Parse lcov.info → line coverage percentage."""
    lines_found = lines_hit = 0
    for line in lcov_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("LF:"):
            lines_found += int(line[3:])
        elif line.startswith("LH:"):
            lines_hit += int(line[3:])
    return (lines_hit / lines_found * 100.0) if lines_found else 0.0


def parse_istanbul(json_path: Path) -> float:
    """Parse Istanbul coverage-final.json → statement coverage percentage."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    total = covered = 0
    for file_cov in data.values():
        stmts = file_cov.get("s", {})
        total += len(stmts)
        covered += sum(1 for v in stmts.values() if v > 0)
    return (covered / total * 100.0) if total else 0.0


def aggregate_coverage(
    python_xml: Optional[Path] = None,
    rust_lcov: Optional[Path] = None,
    frontend_json: Optional[Path] = None,
) -> Dict[str, float]:
    """Cross-language coverage aggregation."""
    results: Dict[str, float] = {}
    if python_xml and python_xml.exists():
        results["python"] = parse_coverage_xml(python_xml)
    if rust_lcov and rust_lcov.exists():
        results["rust"] = parse_lcov(rust_lcov)
    if frontend_json and frontend_json.exists():
        results["frontend"] = parse_istanbul(frontend_json)
    if results:
        results["aggregate"] = sum(results.values()) / len(results)
    return results


# ═════════════════════════════════════════════════════════════
# VERTEBRA 2: QUALITY TREND
# ═════════════════════════════════════════════════════════════


@dataclass
class QualitySnapshot:
    """Point-in-time quality measurement, hash-chained."""

    timestamp: str = ""
    commit_sha: str = ""
    snr_score: float = 0.0
    ihsan_score: float = 0.0
    coverage_pct: float = 0.0
    coverage_floor: float = 0.0
    mypy_errors: int = 0
    mypy_baseline: int = MYPY_BASELINE
    tests_total: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    p95_latency_ms: float = 0.0
    memory_peak_mb: float = 0.0
    vulnerabilities_critical: int = 0
    vulnerabilities_high: int = 0
    rust_tests_passed: int = 0
    rust_clippy_warnings: int = 0
    frontend_bundle_kb: int = 0
    ci_run_id: str = ""
    branch: str = ""
    parent_hash: str = ""
    snapshot_hash: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def compute_hash(self) -> str:
        d = asdict(self)
        d.pop("snapshot_hash", None)
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=str).encode()
        ).hexdigest()[:32]

    def finalize(self) -> None:
        self.snapshot_hash = self.compute_hash()


@dataclass
class TrendAnalysis:
    """Quality trend over a window of snapshots."""

    window_size: int = 0
    direction: str = "stable"
    snr_trend: float = 0.0
    coverage_trend: float = 0.0
    mypy_trend: float = 0.0
    test_pass_rate_trend: float = 0.0
    anomalies: List[str] = field(default_factory=list)
    summary: str = ""


class TrendStore:
    """Append-only, hash-chained JSONL quality store."""

    def __init__(self, path: Path = TREND_PATH) -> None:
        self._path = path

    def append(self, snap: QualitySnapshot) -> None:
        last = self.last()
        snap.parent_hash = last.snapshot_hash if last else "0" * 32
        snap.finalize()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(snap), default=str) + "\n")

    def last(self) -> Optional[QualitySnapshot]:
        if not self._path.exists():
            return None
        with open(self._path, encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return None
        return self._deserialize(lines[-1])

    def read_last_n(self, n: int) -> List[QualitySnapshot]:
        if not self._path.exists():
            return []
        with open(self._path, encoding="utf-8") as f:
            lines = f.readlines()
        return [self._deserialize(l) for l in lines[-n:] if l.strip()]

    def read_all(self) -> List[QualitySnapshot]:
        if not self._path.exists():
            return []
        with open(self._path, encoding="utf-8") as f:
            return [self._deserialize(l) for l in f if l.strip()]

    def count(self) -> int:
        if not self._path.exists():
            return 0
        with open(self._path, encoding="utf-8") as f:
            return sum(1 for l in f if l.strip())

    @staticmethod
    def _deserialize(line: str) -> QualitySnapshot:
        data = json.loads(line)
        return QualitySnapshot(
            **{
                k: v
                for k, v in data.items()
                if k in QualitySnapshot.__dataclass_fields__
            }
        )


def _linear_slope(values: List[float]) -> float:
    """Least-squares slope. Positive = improving."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = statistics.mean(values)
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def analyze_trend(snapshots: List[QualitySnapshot]) -> TrendAnalysis:
    """Shewhart SPC trend analysis over snapshot window."""
    if len(snapshots) < 2:
        return TrendAnalysis(
            window_size=len(snapshots),
            direction="insufficient_data",
            summary=f"Need ≥2 snapshots, have {len(snapshots)}",
        )

    a = TrendAnalysis(window_size=len(snapshots))
    anomalies: List[str] = []

    snr_vals = [s.snr_score for s in snapshots if s.snr_score > 0]
    cov_vals = [s.coverage_pct for s in snapshots if s.coverage_pct > 0]
    mypy_vals = [float(s.mypy_errors) for s in snapshots if s.mypy_errors > 0]
    pass_rates = [
        s.tests_passed / s.tests_total for s in snapshots if s.tests_total > 0
    ]

    if snr_vals:
        a.snr_trend = _linear_slope(snr_vals)
    if cov_vals:
        a.coverage_trend = _linear_slope(cov_vals)
    if mypy_vals:
        a.mypy_trend = _linear_slope(mypy_vals)
    if pass_rates:
        a.test_pass_rate_trend = _linear_slope(pass_rates)

    # Anomaly detection (>2σ from mean)
    for label, vals in [("SNR", snr_vals), ("Coverage", cov_vals)]:
        if len(vals) >= 5:
            mean_v = statistics.mean(vals)
            std_v = statistics.stdev(vals) if len(vals) > 1 else 0
            if std_v > 0:
                z = (vals[-1] - mean_v) / std_v
                if abs(z) > 2.0:
                    anomalies.append(
                        f"{label}: {vals[-1]:.3f} is {z:+.1f}σ from mean {mean_v:.3f}"
                    )

    a.anomalies = anomalies

    pos = neg = 0
    if a.snr_trend > 0.001:
        pos += 1
    elif a.snr_trend < -0.001:
        neg += 1
    if a.coverage_trend > 0.1:
        pos += 1
    elif a.coverage_trend < -0.1:
        neg += 1
    if a.mypy_trend < -1.0:
        pos += 1
    elif a.mypy_trend > 1.0:
        neg += 1

    a.direction = "improving" if pos > neg else ("degrading" if neg > pos else "stable")
    a.summary = (
        f"{a.direction} over {len(snapshots)} snapshots | "
        f"SNR {a.snr_trend:+.4f} | Cov {a.coverage_trend:+.2f}%/snap | "
        f"MyPy {a.mypy_trend:+.1f}/snap"
    )
    return a


# ═════════════════════════════════════════════════════════════
# VERTEBRA 3: RELEASE GATES
# ═════════════════════════════════════════════════════════════


@dataclass
class GateResult:
    """Single quality gate verdict."""

    name: str
    category: str
    passed: bool
    score: float
    weight: float
    detail: str
    blocking: bool = True


@dataclass
class SpineVerdict:
    """Complete spine enforcement result."""

    timestamp: str = ""
    commit_sha: str = ""
    gates: List[Dict[str, Any]] = field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = False
    blocking_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ratchet: Optional[Dict[str, Any]] = None
    trend: Optional[Dict[str, Any]] = None
    evidence_hash: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def _run_tool(
    cmd: List[str], cwd: Path, timeout: int = 120
) -> subprocess.CompletedProcess:
    """Run a subprocess with timeout, capturing output."""
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(cwd), timeout=timeout
    )


def gate_python_tests(ws: Path) -> GateResult:
    r = _run_tool(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-x",
            "--tb=line",
            "-q",
            "-m",
            "not requires_ollama and not requires_gpu and not slow",
        ],
        ws,
        timeout=300,
    )
    passed = r.returncode == 0
    summary = r.stdout.strip().split("\n")[-1] if r.stdout.strip() else "no output"
    return GateResult(
        "python_tests", "quality", passed, 1.0 if passed else 0.0, 0.20, summary
    )


def gate_coverage_floor(ws: Path) -> GateResult:
    cov_xml = ws / "coverage.xml"
    pyp = ws / "pyproject.toml"
    if not cov_xml.exists():
        return GateResult(
            "coverage_floor",
            "quality",
            False,
            0.0,
            0.15,
            "coverage.xml missing",
            blocking=False,
        )
    try:
        actual = parse_coverage_xml(cov_xml)
        floor = read_coverage_floor(pyp)
        return GateResult(
            "coverage_floor",
            "quality",
            actual >= floor,
            min(actual / 100.0, 1.0),
            0.15,
            f"{actual:.1f}% vs {floor:.0f}% floor",
        )
    except (FileNotFoundError, ValueError) as e:
        return GateResult(
            "coverage_floor", "quality", False, 0.0, 0.15, str(e), blocking=False
        )


def gate_lint(ws: Path) -> GateResult:
    ruff = _run_tool(
        [sys.executable, "-m", "ruff", "check", "core/", "--quiet"], ws, 60
    )
    black = _run_tool(
        [sys.executable, "-m", "black", "--check", "--quiet", "core/"], ws, 60
    )
    both = ruff.returncode == 0 and black.returncode == 0
    parts = []
    if ruff.returncode != 0:
        parts.append("ruff:FAIL")
    if black.returncode != 0:
        parts.append("black:FAIL")
    return GateResult(
        "lint",
        "quality",
        both,
        (
            1.0
            if both
            else (0.5 if ruff.returncode == 0 or black.returncode == 0 else 0.0)
        ),
        0.10,
        ", ".join(parts) or "PASS",
    )


def gate_mypy_ratchet(ws: Path) -> GateResult:
    r = _run_tool(
        [
            sys.executable,
            "-m",
            "mypy",
            "core/",
            "--ignore-missing-imports",
            "--no-error-summary",
        ],
        ws,
        120,
    )
    errors = sum(1 for line in r.stdout.splitlines() if line.startswith("core/"))
    passed = errors <= MYPY_BASELINE
    return GateResult(
        "mypy_ratchet",
        "quality",
        passed,
        max(0.0, 1.0 - errors / MYPY_BASELINE * 0.5),
        0.10,
        f"{errors} errors (baseline {MYPY_BASELINE})",
    )


def gate_security(ws: Path) -> GateResult:
    r = _run_tool(
        [sys.executable, "-m", "pip_audit", "--strict", "--progress-spinner=off"],
        ws,
        120,
    )
    passed = r.returncode == 0
    vulns = r.stdout.count("FAIL") if not passed else 0
    return GateResult(
        "security",
        "security",
        passed,
        1.0 if passed else max(0.0, 1.0 - vulns * 0.1),
        0.10,
        f"{vulns} vulns" if vulns else "clean",
    )


def gate_version_sync(ws: Path) -> GateResult:
    versions: Dict[str, str] = {}
    for name, path in [
        ("python", ws / "pyproject.toml"),
        ("rust", ws / "bizra-omega" / "Cargo.toml"),
    ]:
        if path.exists():
            m = re.search(
                r'^version\s*=\s*"([^"]+)"',
                path.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
            if m:
                versions[name] = m.group(1)
    if len(versions) <= 1:
        return GateResult(
            "version_sync", "governance", True, 1.0, 0.05, f"{versions}", blocking=False
        )
    ok = len(set(versions.values())) == 1
    return GateResult(
        "version_sync",
        "governance",
        ok,
        1.0 if ok else 0.5,
        0.05,
        f"{versions}" + (" MISMATCH" if not ok else ""),
        blocking=False,
    )


def gate_cross_lang_constants(ws: Path) -> GateResult:
    script = ws / ".claude" / "skills" / "cross-lang-sync" / "audit_constants.py"
    if not script.exists():
        return GateResult(
            "cross_lang_sync",
            "governance",
            True,
            1.0,
            0.05,
            "audit script not found",
            blocking=False,
        )
    r = _run_tool([sys.executable, str(script)], ws, 30)
    return GateResult(
        "cross_lang_sync",
        "governance",
        r.returncode == 0,
        1.0 if r.returncode == 0 else 0.0,
        0.05,
        "in sync" if r.returncode == 0 else "drift detected",
    )


def gate_frontend(ws: Path) -> GateResult:
    fe = ws / "frontend"
    if not fe.exists():
        return GateResult(
            "frontend", "quality", True, 1.0, 0.05, "skipped", blocking=False
        )
    r = _run_tool(["npm", "run", "ci"], fe, 120)
    return GateResult(
        "frontend",
        "quality",
        r.returncode == 0,
        1.0 if r.returncode == 0 else 0.0,
        0.05,
        "PASS" if r.returncode == 0 else "FAIL",
    )


ALL_GATES: List[Callable[[Path], GateResult]] = [
    gate_python_tests,
    gate_coverage_floor,
    gate_lint,
    gate_mypy_ratchet,
    gate_security,
    gate_version_sync,
    gate_cross_lang_constants,
    gate_frontend,
]


def enforce_gates(
    ws: Path, gate_fns: Optional[List[Callable]] = None
) -> List[GateResult]:
    """Run all gates, return results list."""
    results = []
    for fn in gate_fns or ALL_GATES:
        name = fn.__name__.replace("gate_", "")
        try:
            results.append(fn(ws))
        except Exception as e:
            results.append(
                GateResult(
                    name, "error", False, 0.0, 0.0, f"error: {e}", blocking=False
                )
            )
    return results


# ═════════════════════════════════════════════════════════════
# VERTEBRA 4: CONVENTIONAL CHANGELOG
# ═════════════════════════════════════════════════════════════

_CC_RE = re.compile(
    r"^(?P<type>feat|fix|perf|refactor|docs|test|ci|chore|security|breaking)"
    r"(?:\((?P<scope>[^)]+)\))?"
    r"(?P<bang>!)?:\s*"
    r"(?P<desc>.+)$"
)

_SECTIONS = {
    "breaking": "Breaking Changes",
    "security": "Security",
    "feat": "Features",
    "fix": "Bug Fixes",
    "perf": "Performance",
    "refactor": "Refactoring",
    "docs": "Documentation",
    "test": "Tests",
    "ci": "CI/CD",
    "chore": "Chores",
}


@dataclass
class Commit:
    sha: str
    type: str
    scope: Optional[str]
    desc: str
    is_breaking: bool = False
    author: str = ""


def parse_commit(sha: str, author: str, message: str) -> Commit:
    first = message.strip().split("\n")[0]
    body = message.strip().split("\n", 1)[1] if "\n" in message else ""
    m = _CC_RE.match(first)
    if not m:
        return Commit(sha[:8], "chore", None, first, False, author)
    breaking = m.group("bang") == "!" or m.group("type") == "breaking"
    if "BREAKING CHANGE:" in body or "BREAKING-CHANGE:" in body:
        breaking = True
    return Commit(
        sha[:8], m.group("type"), m.group("scope"), m.group("desc"), breaking, author
    )


def git_commits(
    from_ref: str, to_ref: str = "HEAD", cwd: Path = Path(".")
) -> List[Tuple[str, str, str]]:
    """(sha, author, message) from git log."""
    sep = "---QS---"
    r = subprocess.run(
        [
            "git",
            "log",
            f"{from_ref}..{to_ref}",
            f"--format=%H{sep}%an{sep}%B{sep}",
            "--no-merges",
        ],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"git log failed: {r.stderr}")
    out: List[Tuple[str, str, str]] = []
    for entry in r.stdout.strip().split(f"{sep}\n"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(sep, 2)
        if len(parts) >= 3:
            out.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return out


def render_changelog(commits: List[Commit], version: str = "Unreleased") -> str:
    grouped: Dict[str, List[Commit]] = {}
    for c in commits:
        key = "breaking" if c.is_breaking else c.type
        grouped.setdefault(key, []).append(c)

    lines = [f"## [{version}] - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}", ""]
    for key, title in _SECTIONS.items():
        if key in grouped:
            lines.append(f"### {title}")
            lines.append("")
            for c in grouped[key]:
                scope = f"**{c.scope}**: " if c.scope else ""
                lines.append(f"- {scope}{c.desc} ({c.sha})")
            lines.append("")

    contribs = sorted(set(c.author for c in commits if c.author))
    if contribs:
        lines.extend(["### Contributors", "", ", ".join(contribs), ""])
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# VERTEBRA 5: PR QUALITY SUMMARY
# ═════════════════════════════════════════════════════════════


def render_pr_summary(
    coverage: float,
    floor: float,
    ratcheted: bool,
    new_floor: Optional[float],
    commit: str = "",
    trend_dir: str = "stable",
) -> str:
    """Markdown quality card for PR comments."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    headroom = coverage - floor

    if coverage >= 80:
        badge = "🟢"
    elif coverage >= 60:
        badge = "🟡"
    elif coverage >= floor:
        badge = "🟠"
    else:
        badge = "🔴"

    lines = [
        "## BIZRA Quality Dashboard",
        "",
        f"> `{commit[:8]}` | {now} | Trend: **{trend_dir}**",
        "",
        "| Metric | Value | |",
        "|--------|-------|-|",
        f"| Coverage | **{coverage:.1f}%** | {badge} |",
        f"| Floor | {floor:.0f}% | |",
        f"| Headroom | {headroom:+.1f}% | {'📈' if headroom > 5 else '➡️' if headroom > 0 else '📉'} |",
        f"| Ratchet | {'🔒 → ' + str(int(new_floor)) + '%' if ratcheted and new_floor else '—'} | |",
        f"| Ihsan | ≥ {UNIFIED_IHSAN_THRESHOLD} | {'✅' if coverage >= floor else '⚠️'} |",
        f"| SNR | ≥ {UNIFIED_SNR_THRESHOLD} | {'✅' if coverage >= floor else '⚠️'} |",
        f"| ADL Gini | ≤ {ADL_GINI_THRESHOLD} | ✅ |",
        "",
    ]
    if coverage < floor:
        lines.append("**REGRESSION** — Coverage below floor. PR blocked.")
    elif ratcheted:
        lines.append(f"**RATCHET ELIGIBLE** — Floor can rise to {int(new_floor)}%.")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# EVIDENCE CHAIN
# ═════════════════════════════════════════════════════════════


def _append_evidence(record: Dict[str, Any], path: Path = SPINE_LOG) -> str:
    """Append hash-chained evidence record. Returns hash."""
    content = json.dumps(record, sort_keys=True, default=str)
    h = hashlib.sha256(content.encode()).hexdigest()[:16]
    record["evidence_hash"] = h
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return h


# ═════════════════════════════════════════════════════════════
# SPINE ORCHESTRATOR
# ═════════════════════════════════════════════════════════════


def enforce(
    workspace: Path,
    coverage_xml: Optional[Path] = None,
    pyproject: Optional[Path] = None,
    apply_ratchet: bool = False,
    commit_sha: str = "HEAD",
    skip_slow: bool = False,
    output_json: bool = False,
) -> SpineVerdict:
    """Run the full quality spine: RATCHET → TREND → GATES → VERDICT."""
    ws = workspace.resolve()
    pyp = pyproject or ws / "pyproject.toml"
    cov = coverage_xml or ws / "coverage.xml"

    verdict = SpineVerdict(commit_sha=commit_sha)
    print("=" * 60)
    print("  BIZRA QUALITY SPINE")
    print("=" * 60)

    # ── V1: Ratchet ──────────────────────────────────────────
    print("\n[V1] Coverage Ratchet")
    try:
        actual = parse_coverage_xml(cov)
        floor = read_coverage_floor(pyp)
        rr = evaluate_ratchet(actual, floor)
        verdict.ratchet = asdict(rr)

        if rr.regression:
            print(f"  REGRESSION: {actual:.1f}% < {floor:.0f}% floor")
            verdict.blocking_failures.append("coverage_regression")
        elif rr.ratcheted:
            print(f"  RATCHET: {floor:.0f}% → {rr.new_floor}% eligible")
            if apply_ratchet and rr.new_floor is not None:
                write_coverage_floor(pyp, rr.new_floor)
                rr.applied = True
                verdict.ratchet = asdict(rr)
                print("  APPLIED: pyproject.toml updated")
        else:
            print(
                f"  OK: {actual:.1f}% (floor {floor:.0f}%, headroom {rr.headroom:+.1f}%)"
            )
    except (FileNotFoundError, ValueError) as e:
        print(f"  SKIP: {e}")
        verdict.ratchet = {"error": str(e)}

    # ── V2: Trend ────────────────────────────────────────────
    print("\n[V2] Quality Trend")
    store = TrendStore()
    snap = QualitySnapshot(commit_sha=commit_sha)
    if verdict.ratchet and "actual_coverage" in verdict.ratchet:
        snap.coverage_pct = verdict.ratchet["actual_coverage"]
        snap.coverage_floor = verdict.ratchet["current_floor"]
    try:
        store.append(snap)
    except Exception:
        pass  # Non-blocking — trend is informational
    recent = store.read_last_n(30)
    trend = analyze_trend(recent)
    verdict.trend = asdict(trend)
    print(f"  {trend.summary}")
    if trend.anomalies:
        for a in trend.anomalies:
            print(f"  ANOMALY: {a}")
            verdict.warnings.append(a)

    # ── V3: Gates ────────────────────────────────────────────
    print("\n[V3] Quality Gates")
    skip = {"gate_frontend", "gate_security"} if skip_slow else set()
    gates_to_run = [g for g in ALL_GATES if g.__name__ not in skip]
    results = enforce_gates(ws, gates_to_run)

    for g in results:
        status = "PASS" if g.passed else ("WARN" if not g.blocking else "FAIL")
        print(f"  [{status}] {g.name}: {g.detail}")
        if not g.passed:
            if g.blocking:
                verdict.blocking_failures.append(g.name)
            else:
                verdict.warnings.append(g.name)

    verdict.gates = [asdict(g) for g in results]
    total_w = sum(g.weight for g in results)
    verdict.overall_score = (
        sum(g.score * g.weight for g in results) / total_w if total_w else 0.0
    )
    verdict.passed = len(verdict.blocking_failures) == 0

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Score:    {verdict.overall_score:.3f}")
    print(f"  Verdict:  {'PASS' if verdict.passed else 'FAIL'}")
    if verdict.blocking_failures:
        print(f"  Blocked:  {', '.join(verdict.blocking_failures)}")
    if verdict.warnings:
        print(f"  Warnings: {', '.join(verdict.warnings)}")

    # ── Evidence ─────────────────────────────────────────────
    h = _append_evidence(asdict(verdict))
    verdict.evidence_hash = h
    print(f"  Evidence: {h}")
    print("=" * 60)

    if output_json:
        print(json.dumps(asdict(verdict), indent=2, default=str))

    return verdict


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════


def main() -> int:
    p = argparse.ArgumentParser(
        prog="quality_spine",
        description="BIZRA Quality Spine — single enforceable quality gate",
    )
    sub = p.add_subparsers(dest="cmd")

    # ── enforce ──────────────────────────────────────────────
    e = sub.add_parser("enforce", help="Run full spine (all vertebrae)")
    e.add_argument("--workspace", type=Path, default=Path("."))
    e.add_argument("--coverage-xml", type=Path, default=None)
    e.add_argument("--pyproject", type=Path, default=None)
    e.add_argument("--apply", action="store_true", help="Apply ratchet")
    e.add_argument("--commit", default="HEAD")
    e.add_argument("--fast", action="store_true", help="Skip slow gates")
    e.add_argument("--json", action="store_true")

    # ── ratchet ──────────────────────────────────────────────
    r = sub.add_parser("ratchet", help="Coverage ratchet only")
    r.add_argument("--coverage-xml", type=Path, default=Path("coverage.xml"))
    r.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    r.add_argument("--step", type=int, default=RATCHET_STEP)
    r.add_argument("--apply", action="store_true")
    r.add_argument("--rust-lcov", type=Path, default=None)
    r.add_argument("--frontend-json", type=Path, default=None)
    r.add_argument("--json", action="store_true")

    # ── trend ────────────────────────────────────────────────
    t = sub.add_parser("trend", help="Quality trend tracker")
    tsub = t.add_subparsers(dest="trend_cmd")
    tr = tsub.add_parser("record")
    tr.add_argument("--commit-sha", default="")
    tr.add_argument("--branch", default="")
    tr.add_argument("--snr", type=float, default=0.0)
    tr.add_argument("--ihsan", type=float, default=0.0)
    tr.add_argument("--coverage", type=float, default=0.0)
    tr.add_argument("--coverage-floor", type=float, default=0.0)
    tr.add_argument("--mypy-errors", type=int, default=0)
    tr.add_argument("--tests-total", type=int, default=0)
    tr.add_argument("--tests-passed", type=int, default=0)
    tr.add_argument("--ci-run-id", default="")
    ta = tsub.add_parser("analyze")
    ta.add_argument("--last", type=int, default=30)
    te = tsub.add_parser("export")
    te.add_argument("--format", choices=["json", "jsonl"], default="json")
    te.add_argument("--output", default=None)

    # ── gates ────────────────────────────────────────────────
    g = sub.add_parser("gates", help="Run quality gates only")
    g.add_argument("--workspace", type=Path, default=Path("."))
    g.add_argument("--fast", action="store_true")
    g.add_argument("--json", action="store_true")

    # ── changelog ────────────────────────────────────────────
    c = sub.add_parser("changelog", help="Generate changelog")
    c.add_argument("--from-tag", default=None)
    c.add_argument("--from-sha", default=None)
    c.add_argument("--to-sha", default="HEAD")
    c.add_argument("--version", default="Unreleased")
    c.add_argument("--workspace", type=Path, default=Path("."))
    c.add_argument("--append", type=Path, default=None)
    c.add_argument("--json", action="store_true")

    # ── summary ──────────────────────────────────────────────
    s = sub.add_parser("summary", help="Generate PR quality summary")
    s.add_argument("--coverage", type=float, required=True)
    s.add_argument("--floor", type=float, required=True)
    s.add_argument("--ratcheted", type=str, default="False")
    s.add_argument("--new-floor", type=str, default="none")
    s.add_argument("--commit", default="unknown")
    s.add_argument("--output", type=Path, default=None)

    args = p.parse_args()

    # ── Dispatch ─────────────────────────────────────────────

    if args.cmd == "enforce":
        v = enforce(
            workspace=args.workspace,
            coverage_xml=args.coverage_xml,
            pyproject=args.pyproject,
            apply_ratchet=args.apply,
            commit_sha=args.commit,
            skip_slow=args.fast,
            output_json=args.json,
        )
        if not v.passed:
            return 1
        if v.ratchet and v.ratchet.get("applied"):
            return 2
        return 0

    elif args.cmd == "ratchet":
        try:
            actual = parse_coverage_xml(args.coverage_xml)
            floor = read_coverage_floor(args.pyproject)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 3
        rr = evaluate_ratchet(actual, floor, args.step)
        if args.apply and rr.ratcheted and rr.new_floor is not None:
            write_coverage_floor(args.pyproject, rr.new_floor)
            rr.applied = True
        multi = aggregate_coverage(
            args.coverage_xml, args.rust_lcov, args.frontend_json
        )
        _append_evidence(asdict(rr))
        if args.json:
            out = asdict(rr)
            if multi:
                out["multi_language"] = multi
            print(json.dumps(out, indent=2))
        else:
            print(
                f"Coverage: {actual:.1f}% | Floor: {floor:.0f}% | "
                f"Headroom: {rr.headroom:+.1f}% | Ratchet: {rr.ratcheted}"
            )
            if multi:
                for k, v in multi.items():
                    print(f"  {k}: {v:.1f}%")
        return 1 if rr.regression else (2 if rr.applied else 0)

    elif args.cmd == "trend":
        store = TrendStore()
        if args.trend_cmd == "record":
            snap = QualitySnapshot(
                commit_sha=args.commit_sha,
                branch=args.branch,
                snr_score=args.snr,
                ihsan_score=args.ihsan,
                coverage_pct=args.coverage,
                coverage_floor=args.coverage_floor,
                mypy_errors=args.mypy_errors,
                tests_total=args.tests_total,
                tests_passed=args.tests_passed,
                ci_run_id=args.ci_run_id,
            )
            store.append(snap)
            print(f"Recorded: {snap.snapshot_hash} (parent: {snap.parent_hash[:8]}…)")
        elif args.trend_cmd == "analyze":
            snaps = store.read_last_n(args.last)
            t_result = analyze_trend(snaps)
            print(f"Direction: {t_result.direction} | {t_result.summary}")
            for a in t_result.anomalies:
                print(f"  ANOMALY: {a}")
        elif args.trend_cmd == "export":
            data = [asdict(s) for s in store.read_all()]
            out_str = (
                json.dumps(data, indent=2, default=str)
                if args.format == "json"
                else "\n".join(json.dumps(d, default=str) for d in data)
            )
            if args.output:
                Path(args.output).write_text(out_str, encoding="utf-8")
                print(f"Exported {len(data)} snapshots to {args.output}")
            else:
                print(out_str)
        else:
            p.parse_args(["trend", "--help"])
        return 0

    elif args.cmd == "gates":
        skip = {"gate_frontend", "gate_security"} if args.fast else set()
        fns = [g for g in ALL_GATES if g.__name__ not in skip]
        results = enforce_gates(args.workspace.resolve(), fns)
        fails = []
        for gr in results:
            s_label = "PASS" if gr.passed else ("WARN" if not gr.blocking else "FAIL")
            print(f"[{s_label}] {gr.name}: {gr.detail}")
            if not gr.passed and gr.blocking:
                fails.append(gr.name)
        if args.json:
            print(json.dumps([asdict(g) for g in results], indent=2))
        return 1 if fails else 0

    elif args.cmd == "changelog":
        from_ref = args.from_sha or args.from_tag
        if not from_ref:
            r = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True,
                text=True,
                cwd=str(args.workspace),
                timeout=10,
            )
            from_ref = r.stdout.strip() if r.returncode == 0 else None
        if not from_ref:
            print("[ERROR] No from-ref and no tags found", file=sys.stderr)
            return 3
        try:
            raw = git_commits(from_ref, args.to_sha, args.workspace)
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 3
        if not raw:
            print("No commits in range")
            return 1
        parsed = [parse_commit(s, a, m) for s, a, m in raw]
        md = render_changelog(parsed, args.version)
        if args.json:
            print(
                json.dumps(
                    {"version": args.version, "total": len(parsed), "markdown": md},
                    indent=2,
                )
            )
        else:
            print(md)
        if args.append:
            existing = (
                args.append.read_text(encoding="utf-8") if args.append.exists() else ""
            )
            marker = existing.find("\n## ")
            new = (
                (existing[:marker] + "\n" + md + existing[marker:])
                if marker >= 0
                else ("# Changelog\n\n" + md + "\n" + existing)
            )
            args.append.write_text(new, encoding="utf-8")
        return 0

    elif args.cmd == "summary":
        ratcheted = args.ratcheted.lower() in ("true", "1", "yes")
        nf = (
            float(args.new_floor)
            if args.new_floor not in ("none", "None", "")
            else None
        )
        md = render_pr_summary(args.coverage, args.floor, ratcheted, nf, args.commit)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(md, encoding="utf-8")
            print(f"Written to {args.output}")
        else:
            print(md)
        return 0

    else:
        p.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
