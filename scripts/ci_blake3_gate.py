#!/usr/bin/env python3
"""
BLAKE3 Enforcement Gate — Prevent SHA-256 regression in PCI paths.

Scans core/pci/ and core/proof_engine/ for raw hashlib.sha256 usage
(excluding HMAC, which is standards-compliant RFC 2104).

SR-001 remediation: Python must use BLAKE3 for all content/chain hashing
to maintain cross-language interop with Rust (bizra-omega).

Exit code 0 = clean, 1 = violations found.

Standing on Giants: O'Connor et al. (BLAKE3, 2020)
"""

import re
import sys
from pathlib import Path

# Paths where BLAKE3 is mandatory for content hashing
ENFORCED_PATHS = [
    "core/pci",
    "core/proof_engine",
]

# Pattern: hashlib.sha256 NOT inside hmac.new(..., hashlib.sha256)
# We flag any line with hashlib.sha256 that isn't preceded by hmac on the same line
SHA256_PATTERN = re.compile(r"hashlib\.sha256")
HMAC_PATTERN = re.compile(r"hmac\.(new|HMAC|compare_digest).*hashlib\.sha256")
COMMENT_PATTERN = re.compile(r"^\s*#")


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Scan a file for non-HMAC SHA-256 usage. Returns (line_no, line) tuples."""
    violations = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return violations

    for i, line in enumerate(lines, 1):
        # Skip comments
        if COMMENT_PATTERN.match(line):
            continue
        # Skip lines that are HMAC usage (RFC 2104 — intentional)
        if HMAC_PATTERN.search(line):
            continue
        # Flag raw hashlib.sha256
        if SHA256_PATTERN.search(line):
            violations.append((i, line.strip()))

    return violations


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    total_violations = 0

    for rel_path in ENFORCED_PATHS:
        scan_dir = root / rel_path
        if not scan_dir.exists():
            continue

        for py_file in sorted(scan_dir.rglob("*.py")):
            violations = scan_file(py_file)
            if violations:
                rel = py_file.relative_to(root)
                for line_no, line_text in violations:
                    print(f"  BLAKE3-GATE: {rel}:{line_no}: {line_text}")
                    total_violations += 1

    if total_violations > 0:
        print(
            f"\n  {total_violations} SHA-256 violation(s) in PCI/proof paths."
        )
        print("  Use core.proof_engine.canonical.hex_digest() for content hashing.")
        print("  HMAC-SHA256 (hmac.new(..., hashlib.sha256)) is allowed (RFC 2104).")
        return 1

    print("  BLAKE3 gate: PASSED (0 violations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
