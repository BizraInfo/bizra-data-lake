#!/usr/bin/env python3
"""
CI SEC-003 Gate — Exception Specificity Audit
═══════════════════════════════════════════════════════════════════════════════

Scans Python source files for broad `except Exception` and bare `except:`
patterns, enforces a decreasing baseline, and outputs JSONL evidence.

Usage:
    python scripts/ci_exception_audit.py                    # Default: scan core/
    python scripts/ci_exception_audit.py --scan-dirs core/ tests/
    python scripts/ci_exception_audit.py --baseline 30      # Fail if count > 30
    python scripts/ci_exception_audit.py --output report.json

Exit Codes:
    0 — Pass (count ≤ baseline)
    1 — Fail (count > baseline)
    2 — Script error

Blueprint Reference: Section 4.2 — SEC-003 exception audit CI gate
Standing on Giants: Deming (PDCA, 1950) · Shannon (SNR, 1948)

Constitutional: Bare except is forbidden (CLAUDE.md). This gate
enforces the ratchet toward zero broad exceptions.
"""

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Default baseline — ratchet this DOWN as exceptions are hardened
DEFAULT_BASELINE = 35

# Directories to scan
DEFAULT_SCAN_DIRS = ["core"]

# Files where broad except is intentional (with justification)
ALLOWLIST = {
    # Scanner code: expected to catch filesystem errors broadly
    "core/elite/self_harness_engine.py": "Filesystem boundary — scan errors expected",
    # Cognitive fusion: intentional Protocol degradation (P1 will convert these)
    "core/cognitive_fusion/fusion_engine.py": "Protocol-optional degradation (P1 tracked)",
}


@dataclass
class ExceptionFinding:
    """A single broad exception occurrence."""

    file: str
    line: int
    column: int
    pattern: str  # "except Exception" or "bare except"
    function: str  # Enclosing function name
    allowlisted: bool = False
    justification: str = ""


@dataclass
class AuditReport:
    """Aggregated audit results."""

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    scan_dirs: List[str] = field(default_factory=list)
    baseline: int = DEFAULT_BASELINE
    total_findings: int = 0
    enforced_findings: int = 0  # Findings not in allowlist
    allowlisted_findings: int = 0
    findings: List[dict] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class ExceptionAuditor(ast.NodeVisitor):
    """AST visitor that detects broad exception handlers."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.findings: List[ExceptionFinding] = []
        self._function_stack: List[str] = ["<module>"]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        pattern = None

        if node.type is None:
            # bare except:
            pattern = "bare except"
        elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
            # except Exception:
            pattern = "except Exception"
        elif isinstance(node.type, ast.Attribute):
            # except something.Exception — skip, this is specific
            pass

        if pattern:
            is_allowed = self.filepath in ALLOWLIST
            self.findings.append(
                ExceptionFinding(
                    file=self.filepath,
                    line=node.lineno,
                    column=node.col_offset,
                    pattern=pattern,
                    function=self._function_stack[-1],
                    allowlisted=is_allowed,
                    justification=ALLOWLIST.get(self.filepath, ""),
                )
            )

        self.generic_visit(node)


def scan_directory(scan_dir: str) -> List[ExceptionFinding]:
    """Scan a directory for broad exception patterns."""
    findings: List[ExceptionFinding] = []
    root = Path(scan_dir)

    if not root.exists():
        print(
            f"WARNING: Directory '{scan_dir}' does not exist, skipping", file=sys.stderr
        )
        return findings

    for py_file in sorted(root.rglob("*.py")):
        # Use forward slashes for cross-platform consistency
        try:
            rel_path = str(py_file.relative_to(Path.cwd())).replace("\\", "/")
        except ValueError:
            rel_path = str(py_file).replace("\\", "/")

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel_path)
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"WARNING: Cannot parse {rel_path}: {e}", file=sys.stderr)
            continue

        auditor = ExceptionAuditor(rel_path)
        auditor.visit(tree)
        findings.extend(auditor.findings)

    return findings


def run_audit(
    scan_dirs: List[str], baseline: int, output_path: str = ""
) -> AuditReport:
    """Execute the full audit and return the report."""
    all_findings: List[ExceptionFinding] = []

    for scan_dir in scan_dirs:
        all_findings.extend(scan_directory(scan_dir))

    enforced = [f for f in all_findings if not f.allowlisted]
    allowlisted = [f for f in all_findings if f.allowlisted]

    report = AuditReport(
        scan_dirs=scan_dirs,
        baseline=baseline,
        total_findings=len(all_findings),
        enforced_findings=len(enforced),
        allowlisted_findings=len(allowlisted),
        findings=[asdict(f) for f in all_findings],
        passed=len(enforced) <= baseline,
    )

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Report written to {output_path}")

    return report


def print_summary(report: AuditReport) -> None:
    """Print human-readable summary to stdout."""
    status = "✅ PASS" if report.passed else "❌ FAIL"
    print(f"\n{'═' * 60}")
    print(f"SEC-003 Exception Audit — {status}")
    print(f"{'═' * 60}")
    print(f"Scanned:     {', '.join(report.scan_dirs)}")
    print(f"Baseline:    {report.baseline}")
    print(f"Total:       {report.total_findings}")
    print(f"Enforced:    {report.enforced_findings} (must be ≤ {report.baseline})")
    print(f"Allowlisted: {report.allowlisted_findings}")
    print()

    if not report.passed:
        # Show top offenders
        enforced = [f for f in report.findings if not f["allowlisted"]]
        by_file: dict = {}
        for f in enforced:
            by_file.setdefault(f["file"], []).append(f)

        print("Top offenders:")
        for filepath, file_findings in sorted(
            by_file.items(), key=lambda x: -len(x[1])
        ):
            print(f"  {filepath}: {len(file_findings)} findings")
            for finding in file_findings[:3]:
                print(
                    f"    L{finding['line']}:{finding['column']} "
                    f"{finding['pattern']} in {finding['function']}()"
                )

    print(f"\n{'═' * 60}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SEC-003: Audit broad exception patterns in Python source."
    )
    parser.add_argument(
        "--scan-dirs",
        nargs="+",
        default=DEFAULT_SCAN_DIRS,
        help=f"Directories to scan (default: {DEFAULT_SCAN_DIRS})",
    )
    parser.add_argument(
        "--baseline",
        type=int,
        default=DEFAULT_BASELINE,
        help=f"Maximum allowed enforced findings (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Path to write JSON report (optional)",
    )

    args = parser.parse_args()

    report = run_audit(args.scan_dirs, args.baseline, args.output)
    print_summary(report)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
