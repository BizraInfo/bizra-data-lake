#!/usr/bin/env python3
"""
BIZRA Delta Test Runner + Version Lock Tool
============================================
Lock once. Run delta. Ship fast.

Usage:
    python bizra_test.py                    # T0: Smoke (30 sec)
    python bizra_test.py --delta            # T1: Only affected tests
    python bizra_test.py --contract         # T2: Contract tests
    python bizra_test.py --full             # T3: All tests
    python bizra_test.py --lock             # T3 + lock receipt + tag
    python bizra_test.py --status           # Show lock status
    python bizra_test.py --genesis-gate     # T4: Full constitutional audit

Constitutional Anchor: Rule 7 (Discipline & Continuity)
Standing on Giants: Deming (PDCA), Shannon (SNR)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
LOCK_DIR = PROJECT_ROOT / "sovereign_state" / "test_locks"
LOCK_CURRENT = LOCK_DIR / "current.json"
CONSTANTS_FILE = PROJECT_ROOT / "core" / "integration" / "constants.py"

# Test tier file lists
SMOKE_FILES = [
    "tests/core/integration/test_constants_integrity.py",
    "tests/core/sovereign/test_api_exposure_policy.py",
]

CONTRACT_FILES = [
    "tests/core/sovereign/test_api_exposure_policy.py",
    "tests/integration/test_contract_integrity.py",
    "tests/core/sovereign/test_terminal.py",
    "tests/core/orchestration/test_learning_loop.py",
    "tests/core/test_learning_loop_bridges.py",
    "tests/core/test_living_ecosystem.py",
    "tests/core/constitutional/test_ticker.py",
]

PYTEST_BASE = "python -m pytest"
PYTEST_EXCLUDE = '-m "not slow and not requires_ollama and not requires_gpu and not requires_network and not e2e_http"'
PYTEST_IGNORE = "--ignore=tests/e2e_http"


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class TestLockReceipt:
    """Immutable proof that a version passed all tests."""

    version: str
    git_commit: str
    git_tag: str
    timestamp: str

    # Test results
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration_seconds: float

    # Quality metrics
    coverage_percent: float
    coverage_floor: float
    ihsan_score: float = 0.0
    snr_score: float = 0.0

    # Constitutional
    constants_hash: str = ""
    genesis_gate_passed: bool = False

    # Provenance
    node_id: str = "NODE0"
    prev_lock_hash: str = ""
    lock_hash: str = ""

    def compute_hash(self) -> str:
        """BLAKE2b hash of all fields except lock_hash itself."""
        data = {k: v for k, v in asdict(self).items() if k != "lock_hash"}
        raw = json.dumps(data, sort_keys=True).encode()
        return hashlib.blake2b(raw, digest_size=32).hexdigest()


# ============================================================================
# HELPERS
# ============================================================================


def run_cmd(cmd: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a shell command and return result."""
    return subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=PROJECT_ROOT
    )


def get_git_commit() -> str:
    """Get current git commit SHA."""
    r = run_cmd("git rev-parse HEAD")
    return r.stdout.strip()


def get_git_tag_at_head() -> Optional[str]:
    """Get tag at HEAD if any."""
    r = run_cmd("git tag --points-at HEAD")
    tags = r.stdout.strip().split("\n")
    for tag in tags:
        if tag.startswith("v"):
            return tag
    return None


def get_latest_lock() -> Optional[TestLockReceipt]:
    """Load the latest lock receipt."""
    if not LOCK_CURRENT.exists():
        return None
    data = json.loads(LOCK_CURRENT.read_text())
    return TestLockReceipt(**data)


def get_changed_files_since(tag: str) -> Set[str]:
    """Get files changed since a tag."""
    r = run_cmd(f"git diff --name-only {tag} HEAD")
    if r.returncode != 0:
        return set()
    return {f for f in r.stdout.strip().split("\n") if f}


def get_affected_tests(changed_files: Set[str]) -> Set[str]:
    """Find test files affected by changed source files."""
    affected = set()

    for f in changed_files:
        # If the changed file IS a test file, include it
        if f.startswith("tests/") and f.endswith(".py"):
            if Path(PROJECT_ROOT / f).exists():
                affected.add(f)
            continue

        # If it's a source file, find tests that import from it
        if f.startswith("core/") and f.endswith(".py"):
            module = f.replace("/", ".").replace(".py", "")
            r = run_cmd(f'grep -rl "{module}" tests/ --include="*.py" 2>/dev/null')
            for test_file in r.stdout.strip().split("\n"):
                if test_file and Path(test_file).exists():
                    affected.add(test_file)

    return affected


def hash_constants_file() -> str:
    """BLAKE2b hash of constitutional constants."""
    if not CONSTANTS_FILE.exists():
        return "MISSING"
    data = CONSTANTS_FILE.read_bytes()
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def parse_pytest_output(output: str) -> dict:
    """Parse pytest summary line to extract pass/fail/skip counts."""
    result = {"passed": 0, "failed": 0, "skipped": 0, "total": 0, "duration": 0.0}

    for line in output.split("\n"):
        line = line.strip()
        if "passed" in line and ("in " in line or "second" in line):
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "passed" or part == "passed,":
                    result["passed"] = int(parts[i - 1])
                elif part == "failed" or part == "failed,":
                    result["failed"] = int(parts[i - 1])
                elif part == "skipped" or part == "skipped,":
                    result["skipped"] = int(parts[i - 1])
                elif part.endswith("s") and "." in part:
                    try:
                        result["duration"] = float(part.rstrip("s").strip("("))
                    except ValueError:
                        pass

    result["total"] = result["passed"] + result["failed"] + result["skipped"]
    return result


def parse_coverage(output: str) -> float:
    """Parse coverage percentage from pytest-cov output."""
    for line in output.split("\n"):
        if "TOTAL" in line and "%" in line:
            parts = line.split()
            for part in parts:
                if part.endswith("%"):
                    try:
                        return float(part.rstrip("%"))
                    except ValueError:
                        pass
    return 0.0


# ============================================================================
# TIER COMMANDS
# ============================================================================


def run_smoke() -> int:
    """T0: Smoke tests — < 30 seconds."""
    existing = [f for f in SMOKE_FILES if Path(PROJECT_ROOT / f).exists()]
    if not existing:
        print("No smoke test files found. Run full suite instead.")
        return 1

    files = " ".join(existing)
    print(f"T0 SMOKE — {len(existing)} test files")
    start = time.time()
    r = run_cmd(f"{PYTEST_BASE} {files} -x -q --timeout=30", timeout=60)
    elapsed = time.time() - start

    print(r.stdout)
    if r.returncode == 0:
        print(f"Smoke passed in {elapsed:.1f}s")
    else:
        print(f"Smoke FAILED in {elapsed:.1f}s")
        if r.stderr:
            print(r.stderr[-500:])
    return r.returncode


def run_delta() -> int:
    """T1: Delta tests — only affected by changes since last lock."""
    lock = get_latest_lock()
    if lock is None:
        print("No locked version found. Run --full to create first lock.")
        print("Falling back to smoke tests.")
        return run_smoke()

    changed = get_changed_files_since(lock.git_tag)
    if not changed:
        print(f"No changes since {lock.git_tag}. {lock.total_tests} tests locked.")
        return 0

    affected = get_affected_tests(changed)
    if not affected:
        print(f"{len(changed)} files changed, but no tests affected. Lock holds.")
        return 0

    locked_count = lock.total_tests - len(affected)
    print(f"T1 DELTA — {len(affected)} affected / {locked_count} locked (skip)")
    print(f"   Changed: {len(changed)} files since {lock.git_tag}")

    files = " ".join(sorted(affected))
    start = time.time()
    r = run_cmd(f"{PYTEST_BASE} {files} -x -q --timeout=60", timeout=300)
    elapsed = time.time() - start

    print(r.stdout)
    if r.returncode == 0:
        print(f"Delta passed in {elapsed:.1f}s ({locked_count} tests skipped via lock)")
    else:
        print(f"Delta FAILED in {elapsed:.1f}s")
    return r.returncode


def run_contract() -> int:
    """T2: Contract tests — API shapes, type contracts, integration."""
    existing = [f for f in CONTRACT_FILES if Path(PROJECT_ROOT / f).exists()]
    if not existing:
        print("No contract test files found.")
        return 1

    files = " ".join(existing)
    print(f"T2 CONTRACT — {len(existing)} test files")
    start = time.time()
    r = run_cmd(f"{PYTEST_BASE} {files} -x -q --timeout=60", timeout=300)
    elapsed = time.time() - start

    print(r.stdout)
    if r.returncode == 0:
        print(f"Contract tests passed in {elapsed:.1f}s")
    else:
        print(f"Contract tests FAILED in {elapsed:.1f}s")
    return r.returncode


def run_full(with_coverage: bool = True) -> tuple:
    """T3: Full test suite — all tests."""
    cov_flag = "--cov=core --cov-report=term" if with_coverage else ""
    cmd = f'{PYTEST_BASE} tests/ -q --timeout=120 {PYTEST_EXCLUDE} {PYTEST_IGNORE} {cov_flag}'

    print("T3 FULL — Running ALL tests...")
    start = time.time()
    r = run_cmd(cmd, timeout=3600)
    elapsed = time.time() - start

    print(r.stdout[-3000:])  # Last 3000 chars
    results = parse_pytest_output(r.stdout)
    results["duration"] = elapsed
    coverage = parse_coverage(r.stdout) if with_coverage else 0.0

    if r.returncode == 0:
        print(f"\nFull suite: {results['passed']} passed in {elapsed:.0f}s (coverage: {coverage}%)")
    else:
        print(f"\nFull suite FAILED: {results['failed']} failures in {elapsed:.0f}s")

    return r.returncode, results, coverage


def run_lock() -> int:
    """T3 + Lock — Run full suite and create version lock receipt."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)

    # Run full suite
    returncode, results, coverage = run_full(with_coverage=True)

    if returncode != 0:
        print("\nCannot lock — tests failed. Fix failures first.")
        return 1

    if results["failed"] > 0:
        print(f"\nCannot lock — {results['failed']} failures.")
        return 1

    # Determine version
    prev_lock = get_latest_lock()
    if prev_lock:
        # Increment patch
        parts = prev_lock.version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_version = ".".join(parts)
        prev_hash = prev_lock.lock_hash
        coverage_floor = max(prev_lock.coverage_floor, coverage)
    else:
        new_version = "0.80.0"
        prev_hash = "GENESIS"
        coverage_floor = coverage

    # Check coverage ratchet
    if prev_lock and coverage < prev_lock.coverage_floor:
        print(f"\nCannot lock — coverage {coverage}% < floor {prev_lock.coverage_floor}%")
        print("   Coverage ratchet: coverage can only go UP.")
        return 1

    tag = f"v{new_version}"
    commit = get_git_commit()

    receipt = TestLockReceipt(
        version=new_version,
        git_commit=commit,
        git_tag=tag,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_tests=results["total"],
        passed=results["passed"],
        failed=results["failed"],
        skipped=results["skipped"],
        duration_seconds=results["duration"],
        coverage_percent=coverage,
        coverage_floor=coverage_floor,
        constants_hash=hash_constants_file(),
        prev_lock_hash=prev_hash,
    )
    receipt.lock_hash = receipt.compute_hash()

    # Save receipt
    receipt_file = LOCK_DIR / f"{tag}.json"
    receipt_file.write_text(json.dumps(asdict(receipt), indent=2))
    LOCK_CURRENT.write_text(json.dumps(asdict(receipt), indent=2))

    # Create git tag
    r = run_cmd(f'git tag -a {tag} -m "Lock: {results["passed"]} tests, {coverage}% coverage"')
    if r.returncode != 0:
        print(f"Git tag note: {r.stderr.strip()}")

    print(f"\n{'='*60}")
    print(f"VERSION {tag} LOCKED")
    print(f"{'='*60}")
    print(f"   Tests:    {results['passed']} / {results['total']} PASS")
    print(f"   Coverage: {coverage}% (floor: {coverage_floor}%)")
    print(f"   Constants: {receipt.constants_hash[:16]}...")
    print(f"   Lock hash: {receipt.lock_hash[:16]}...")
    print(f"   Prev lock: {prev_hash[:16]}...")
    print(f"   Receipt:  {receipt_file}")
    print(f"\nNext cycle: use 'python bizra_test.py --delta' (only changed modules)")

    return 0


def show_status() -> int:
    """Show current lock status and what needs testing."""
    lock = get_latest_lock()

    if lock is None:
        print("No locked version found.")
        print("   Run: python bizra_test.py --lock")
        return 0

    print(f"Current Lock: {lock.git_tag} (locked {lock.timestamp})")
    print(f"   Commit:   {lock.git_commit[:12]}")
    print(f"   Tests:    {lock.passed} / {lock.total_tests} PASS")
    print(f"   Coverage: {lock.coverage_percent}% (floor: {lock.coverage_floor}%)")
    print(f"   Duration: {lock.duration_seconds:.0f}s")

    # Check what changed since lock
    changed = get_changed_files_since(lock.git_tag)
    if not changed:
        print(f"\nNo changes since {lock.git_tag}. All {lock.total_tests} tests proven.")
        return 0

    affected = get_affected_tests(changed)
    locked = lock.total_tests - len(affected)

    print(f"\nSince lock:")
    print(f"   Changed files:   {len(changed)}")
    print(f"   Affected tests:  {len(affected)}")
    print(f"   Locked (skip):   {locked}")
    print(f"\nRun: python bizra_test.py --delta ({len(affected)} tests, fast)")
    print(f"   Skip: {locked} tests (proven by {lock.git_tag})")

    return 0


# ============================================================================
# MAIN
# ============================================================================


def main():
    args = sys.argv[1:]

    if not args or args[0] == "--smoke":
        return run_smoke()
    elif args[0] == "--delta":
        return run_delta()
    elif args[0] == "--contract":
        return run_contract()
    elif args[0] == "--full":
        rc, _, _ = run_full()
        return rc
    elif args[0] == "--lock":
        return run_lock()
    elif args[0] == "--status":
        return show_status()
    elif args[0] == "--genesis-gate":
        print("T4: Genesis Gate — running full suite + genesis gate checks")
        rc1, _, _ = run_full()
        rc2 = run_cmd("python genesis_gate.py --quick").returncode
        return max(rc1, rc2)
    else:
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
