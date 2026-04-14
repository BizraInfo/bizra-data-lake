#!/usr/bin/env python3
"""
Cycle 1 Deterministic Re-verification Runner
=============================================
Single-pass, no REPL, no shell juggling.
Verifies the full Cycle 1 claim set from a clean process.

Exit 0 = all checks pass.  Exit 1 = at least one failure.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)

PYTHON = sys.executable
PASS_COUNT = 0
FAIL_COUNT = 0
REPORT: list[dict] = []


def section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}\n")


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    if ok:
        PASS_COUNT += 1
        REPORT.append({"check": name, "status": "PASS"})
        print(f"  ✓ {name}")
    else:
        FAIL_COUNT += 1
        REPORT.append({"check": name, "status": "FAIL", "detail": detail})
        print(f"  ✗ {name}: {detail}")


# ── 1. Syntax verification ─────────────────────────────────
section("1. Syntax verification — _connection_pool.py")

target = REPO / "core" / "inference" / "_connection_pool.py"
try:
    import py_compile
    py_compile.compile(str(target), doraise=True)
    check("py_compile _connection_pool.py", True)
except py_compile.PyCompileError as exc:
    check("py_compile _connection_pool.py", False, str(exc))

# ── 2. Smoke test (pytest) ─────────────────────────────────
section("2. Smoke test — deploy/node0/activation_smoke_test.py")

smoke_result = subprocess.run(
    [PYTHON, "-m", "pytest", "deploy/node0/activation_smoke_test.py",
     "-v", "--tb=short", "-q"],
    capture_output=True, text=True, timeout=120,
)
# Parse "N passed" from pytest output
smoke_lines = smoke_result.stdout + smoke_result.stderr
smoke_passed = 0
import re
m = re.search(r"(\d+)\s+passed", smoke_lines)
if m:
    smoke_passed = int(m.group(1))
check(f"Smoke test ({smoke_passed}/11 passed)", smoke_passed == 11,
      f"exit={smoke_result.returncode}")
if smoke_result.returncode != 0:
    print(smoke_lines[-500:] if len(smoke_lines) > 500 else smoke_lines)

# ── 3. Integration test ────────────────────────────────────
section("3. Integration test — deploy/node0/integration_test.py")

integ_result = subprocess.run(
    [PYTHON, "deploy/node0/integration_test.py"],
    capture_output=True, text=True, timeout=120,
)
integ_lines = integ_result.stdout + integ_result.stderr
# Parse "N/M passed"
integ_passed = 0
integ_total = 0
for line in integ_lines.splitlines():
    if "passed" in line and "/" in line:
        for token in line.split():
            if "/" in token:
                parts = token.split("/")
                try:
                    integ_passed = int(parts[0])
                    integ_total = int(parts[1])
                except (ValueError, IndexError):
                    pass
check(f"Integration test ({integ_passed}/{integ_total} passed)",
      integ_passed == integ_total and integ_total >= 10,
      f"exit={integ_result.returncode}")
if integ_result.returncode != 0:
    print(integ_lines[-500:] if len(integ_lines) > 500 else integ_lines)

total_tests = smoke_passed + integ_passed
check(f"Total tests: {total_tests}/21", total_tests >= 21)

# ── 4. Artifact set verification ───────────────────────────
section("4. Cycle 1 artifact set")

EXPECTED_ARTIFACTS = [
    "niyyah.md",
    "bayyinah_report.md",
    "hadd.md",
    "execution_trace.md",
    "reward_report.md",
    "manifest.md",
    "retrospective.md",
]
cycle_dir = REPO / "cycle-1"
for art in EXPECTED_ARTIFACTS:
    path = cycle_dir / art
    check(f"Artifact: {art}", path.exists() and path.stat().st_size > 50,
          "missing or empty")

# ── 5. Canonical hash ──────────────────────────────────────
section("5. BLAKE3 / BLAKE2B hash of canonical file set")

CANONICAL_FILES = sorted([
    "core/inference/_connection_pool.py",
    "core/sovereign/runtime_core.py",
    "core/pat/runtime.py",
    "core/sat/runtime.py",
    "core/sovereign/dema_router.py",
    "core/sovereign/fate_boundary.py",
    "deploy/node0/activation_smoke_test.py",
    "deploy/node0/integration_test.py",
])

blake2_hash = hashlib.blake2b(digest_size=32)
for f in CANONICAL_FILES:
    blob = (REPO / f).read_bytes()
    blake2_hash.update(blob)
blake2_hex = blake2_hash.hexdigest()
print(f"  BLAKE2B-256: {blake2_hex}")

blake3_hex = None
try:
    import blake3 as _b3
    blake3_hash = _b3.blake3()
    for f in CANONICAL_FILES:
        blake3_hash.update((REPO / f).read_bytes())
    blake3_hex = blake3_hash.hexdigest()
    print(f"  BLAKE3:      {blake3_hex}")
except ImportError:
    print("  blake3 module not installed — BLAKE2B-256 used as primary")

hash_hex = blake3_hex or blake2_hex
check("Hash computed", len(hash_hex) == 64)

# ── 6. Summary ─────────────────────────────────────────────
section("VERDICT")

print(f"  Checks: {PASS_COUNT} passed, {FAIL_COUNT} failed")
print(f"  Tests:  {total_tests}/21")
print(f"  Hash:   {hash_hex}")
print()

if FAIL_COUNT == 0:
    print("  ✅ ALL CHECKS PASSED — Node0 Activation evidence is STRONG")
    verdict = "CANONICAL"
else:
    print("  ⚠️  SOME CHECKS FAILED — evidence is INCOMPLETE")
    verdict = "CANDIDATE_CANONICAL"

print(f"  Recommended status: {verdict}")
print()

# Write machine-readable report
report_path = cycle_dir / "reverification_report.json"
report_data = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "python": sys.version.split()[0],
    "checks": REPORT,
    "passed": PASS_COUNT,
    "failed": FAIL_COUNT,
    "total_tests": total_tests,
    "hash_blake2b_256": blake2_hex,
    "hash_blake3": blake3_hex,
    "verdict": verdict,
}
report_path.write_text(json.dumps(report_data, indent=2) + "\n")
print(f"  Report written: {report_path.relative_to(REPO)}")

sys.exit(0 if FAIL_COUNT == 0 else 1)
