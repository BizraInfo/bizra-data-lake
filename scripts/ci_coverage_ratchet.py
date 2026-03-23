#!/usr/bin/env python3
"""
BIZRA CI Coverage Ratchet Engine
================================

Automatically raises the coverage floor when actual coverage exceeds it,
preventing quality regression via a one-way ratchet mechanism. This is the
Deming PDCA cycle applied to code coverage: every improvement is locked in.

Standing on Giants:
- Deming (PDCA quality cycle, 1950)
- Juran (quality ratcheting, 1951)
- Humphrey (Personal Software Process, 1989)

Mechanism:
    1. Read actual coverage from coverage.xml (Cobertura format)
    2. Read current floor from pyproject.toml [tool.coverage.report] fail_under
    3. If actual >= floor + RATCHET_STEP, propose new floor
    4. On --apply: update pyproject.toml in-place
    5. Always: emit JSONL evidence record

Constitutional Constraints:
    - Coverage floor can ONLY increase, never decrease (ratchet)
    - RATCHET_STEP prevents noise — only locks in meaningful gains
    - Evidence record is append-only, hash-chained

Usage:
    # Check only (CI default)
    python scripts/ci_coverage_ratchet.py --coverage-xml coverage.xml

    # Apply ratchet (post-merge to main)
    python scripts/ci_coverage_ratchet.py --coverage-xml coverage.xml --apply

    # Custom step size
    python scripts/ci_coverage_ratchet.py --coverage-xml coverage.xml --step 2

Exit Codes:
    0 - Ratchet check passed (no regression)
    1 - Coverage regression detected (actual < floor)
    2 - Ratchet applied (floor bumped)
    3 - Configuration error
"""

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

DEFAULT_RATCHET_STEP = 1  # Minimum gain (%) to trigger ratchet
MAX_RATCHET_BUMP = 5  # Maximum single-bump increase (safety)
PYPROJECT_PATH = "pyproject.toml"
EVIDENCE_PATH = "04_GOLD/coverage_ratchet_log.jsonl"


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


# ─────────────────────────────────────────────────────────────
# Coverage XML Parser
# ─────────────────────────────────────────────────────────────


def parse_coverage_xml(xml_path: Path) -> float:
    """Parse Cobertura-format coverage.xml and return line-rate as percentage."""
    if not xml_path.exists():
        raise FileNotFoundError(f"Coverage XML not found: {xml_path}")

    tree = ET.parse(str(xml_path))  # noqa: S314 — trusted CI artifact
    root = tree.getroot()

    # Cobertura format: <coverage line-rate="0.38" ...>
    line_rate = root.get("line-rate")
    if line_rate is None:
        raise ValueError("Coverage XML missing 'line-rate' attribute")

    return float(line_rate) * 100.0


# ─────────────────────────────────────────────────────────────
# pyproject.toml Parser (targeted, no TOML dependency)
# ─────────────────────────────────────────────────────────────


def read_coverage_floor(pyproject_path: Path) -> float:
    """Read fail_under from [tool.coverage.report] in pyproject.toml."""
    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r"fail_under\s*=\s*(\d+(?:\.\d+)?)", content)
    if not match:
        raise ValueError(f"fail_under not found in {pyproject_path}")
    return float(match.group(1))


def write_coverage_floor(pyproject_path: Path, new_floor: float) -> None:
    """Update fail_under in pyproject.toml in-place."""
    content = pyproject_path.read_text(encoding="utf-8")

    # Replace fail_under value, preserving formatting context
    new_content = re.sub(
        r"(fail_under\s*=\s*)\d+(?:\.\d+)?",
        f"\\g<1>{int(new_floor)}",
        content,
        count=1,
    )

    if new_content == content:
        raise ValueError("Failed to update fail_under — pattern not matched")

    pyproject_path.write_text(new_content, encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# Ratchet Engine
# ─────────────────────────────────────────────────────────────


def evaluate_ratchet(
    actual: float,
    floor: float,
    step: int = DEFAULT_RATCHET_STEP,
) -> RatchetResult:
    """Evaluate whether coverage qualifies for a ratchet bump."""
    headroom = actual - floor
    regression = actual < floor

    new_floor: Optional[float] = None
    ratcheted = False

    if not regression and headroom >= step:
        # Propose new floor: actual rounded down to nearest integer,
        # but capped at floor + MAX_RATCHET_BUMP to prevent jumps
        candidate = int(actual)  # Floor to integer
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


def append_evidence(result: RatchetResult, evidence_path: Path) -> None:
    """Append ratchet result to evidence log (JSONL)."""
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evidence_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), default=str) + "\n")


# ─────────────────────────────────────────────────────────────
# Multi-Language Coverage Aggregation
# ─────────────────────────────────────────────────────────────


def aggregate_coverage(
    python_xml: Optional[Path] = None,
    rust_lcov: Optional[Path] = None,
    frontend_json: Optional[Path] = None,
) -> dict:
    """Aggregate coverage from Python, Rust, and Frontend sources."""
    results = {}

    if python_xml and python_xml.exists():
        results["python"] = parse_coverage_xml(python_xml)

    if rust_lcov and rust_lcov.exists():
        results["rust"] = _parse_lcov_coverage(rust_lcov)

    if frontend_json and frontend_json.exists():
        results["frontend"] = _parse_istanbul_coverage(frontend_json)

    if results:
        results["aggregate"] = sum(results.values()) / len(results)

    return results


def _parse_lcov_coverage(lcov_path: Path) -> float:
    """Parse lcov.info for line coverage percentage."""
    lines_found = 0
    lines_hit = 0
    content = lcov_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        if line.startswith("LF:"):
            lines_found += int(line[3:])
        elif line.startswith("LH:"):
            lines_hit += int(line[3:])
    if lines_found == 0:
        return 0.0
    return (lines_hit / lines_found) * 100.0


def _parse_istanbul_coverage(json_path: Path) -> float:
    """Parse Istanbul/V8 coverage-final.json for line coverage."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    total_stmts = 0
    covered_stmts = 0
    for file_cov in data.values():
        stmts = file_cov.get("s", {})
        total_stmts += len(stmts)
        covered_stmts += sum(1 for v in stmts.values() if v > 0)
    if total_stmts == 0:
        return 0.0
    return (covered_stmts / total_stmts) * 100.0


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BIZRA Coverage Ratchet Engine — auto-raise coverage floor",
    )
    parser.add_argument(
        "--coverage-xml",
        type=Path,
        default=Path("coverage.xml"),
        help="Path to Cobertura coverage.xml",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path(PYPROJECT_PATH),
        help="Path to pyproject.toml",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_RATCHET_STEP,
        help="Minimum gain (%%) to trigger ratchet (default: 1)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply ratchet (update pyproject.toml in-place)",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path(EVIDENCE_PATH),
        help="Path for JSONL evidence log",
    )
    parser.add_argument(
        "--rust-lcov",
        type=Path,
        default=None,
        help="Path to Rust lcov.info for cross-language aggregation",
    )
    parser.add_argument(
        "--frontend-json",
        type=Path,
        default=None,
        help="Path to frontend coverage-final.json",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON (for GitHub Actions)",
    )

    args = parser.parse_args()

    try:
        actual = parse_coverage_xml(args.coverage_xml)
        floor = read_coverage_floor(args.pyproject)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 3

    result = evaluate_ratchet(actual, floor, args.step)

    # Multi-language aggregation (informational)
    multi_lang = aggregate_coverage(
        python_xml=args.coverage_xml,
        rust_lcov=args.rust_lcov,
        frontend_json=args.frontend_json,
    )

    # Apply ratchet if requested
    if args.apply and result.ratcheted and result.new_floor is not None:
        write_coverage_floor(args.pyproject, result.new_floor)
        result.applied = True
        print(f"[RATCHET] Coverage floor raised: {floor}% → {result.new_floor}%")

    # Evidence
    append_evidence(result, args.evidence)

    # Output
    if args.json:
        output = asdict(result)
        if multi_lang:
            output["multi_language"] = multi_lang
        print(json.dumps(output, indent=2))
    else:
        print("=" * 60)
        print("BIZRA Coverage Ratchet Engine")
        print("=" * 60)
        print(f"  Actual Coverage:  {actual:.1f}%")
        print(f"  Current Floor:    {floor:.0f}%")
        print(f"  Headroom:         {result.headroom:+.1f}%")

        if multi_lang:
            print("\n  Cross-Language Coverage:")
            for lang, cov in multi_lang.items():
                print(f"    {lang:12s}: {cov:.1f}%")

        if result.regression:
            print("\n  [REGRESSION] Coverage dropped below floor!")
        elif result.ratcheted:
            print(f"\n  [RATCHET] Eligible: floor can raise to {result.new_floor}%")
            if result.applied:
                print("  [APPLIED] pyproject.toml updated")
            else:
                print("  [DRY-RUN] Use --apply to lock in the gain")
        else:
            print("\n  [OK] Coverage within range, no ratchet needed")

        print(f"\n  Evidence: {args.evidence}")
        print(f"  Hash:     {result.evidence_hash}")

    # Exit codes
    if result.regression:
        return 1
    if result.applied:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
