# BIZRA Versioned Test Strategy
## Lock Once. Run Delta. Ship Fast.

> **Version:** 1.0 · LOCKED
> **Date:** March 9, 2026 · Dubai
> **Constitutional Anchor:** البذرة Rule 7 (الانضباط والاستمرارية — Discipline & Continuity)
> **Problem:** 8,495+ tests × every commit = 20+ min feedback loops = slow development
> **Solution:** Versioned baselines + tiered test pyramid + delta-only regression

---

## 1. The Problem

Today, BIZRA has 8,495+ tests. By Genesis, it'll have 10,000+. By Phase 2, 20,000+.

If every development cycle runs all tests:
- **20+ minutes** per commit on the best hardware
- **Developer flow broken** — waiting instead of building
- **CI costs escalate** — cloud CI minutes add up
- **False signal** — a failure in an unrelated module blocks your work

The constitutional heartbeat ticks every 60 seconds. The development heartbeat should be just as fast.

---

## 2. The Solution: Test Pyramid with Locked Baselines

### 2.1 Version Lock Concept

When all tests pass at a commit, that commit becomes a **locked version**. The test results are cached. Future development only runs tests **affected by changes since that locked version**.

```
v0.80.0 — 8,495 tests pass — LOCKED ✓
    ↓
Developer changes core/token/bloom.py
    ↓
Only run: tests/core/token/ + tests that import from core.token
    ↓
NOT: tests/core/zpk/, tests/core/autopoiesis/, tests/core/hrm/...
```

The unchanged modules don't need re-testing because their locked version already proved they work. The code hasn't changed. The tests haven't changed. Re-running them proves nothing new.

### 2.2 Test Tiers

| Tier | Name | When | Tests | Time Target | What It Catches |
|------|------|------|-------|-------------|----------------|
| **T0** | Smoke | Every save / `bizra doctor` | ~50 | < 30 sec | Import errors, syntax, constitutional constants |
| **T1** | Unit | Every commit / push | ~500 (changed modules only) | < 2 min | Logic errors in changed code |
| **T2** | Contract | Every PR / merge to main | ~1,200 (contracts + integration) | < 5 min | API shape drift, type mismatches, boundary violations |
| **T3** | Full | Version lock only (manual or nightly) | 8,495+ (all) | 20-30 min | Everything — proves the locked version |
| **T4** | Genesis Gate | Release candidate only | 68 checks + all tests | 30-60 min | Constitutional readiness for deployment |

### 2.3 The Rule

**You never run T3 during development.** T3 runs when you're ready to lock a version. That's it.

During active development, you run T0 (instant feedback), T1 (on commit), and T2 (on merge). If T0-T2 pass and the locked baseline is green, the system is proven.

---

## 3. Version Lock Specification

### 3.1 What Is a Locked Version?

A locked version is a git tag with a cryptographic proof that all tests passed at that exact commit.

```
Tag:     v0.80.0
Commit:  8e57ca2
Tests:   8,495 / 8,495 PASS
Coverage: 42.3%
Ihsān:   0.96
SNR:     0.941
Date:    2026-03-09 04:00 GST
Signed:  Ed25519 (Node0 key)
```

### 3.2 Lock Receipt

Every version lock generates a **TestLockReceipt** — a constitutional artifact on the evidence chain:

```python
@dataclass
class TestLockReceipt:
    """Immutable proof that a version passed all tests."""
    version: str                    # Semantic version (e.g., "0.80.0")
    git_commit: str                 # Full SHA-256 commit hash
    git_tag: str                    # Git tag (e.g., "v0.80.0")
    timestamp: datetime             # When the lock was created (UTC)
    
    # Test results
    total_tests: int                # Total test count
    passed: int                     # Must equal total_tests for lock
    failed: int                     # Must be 0 for lock
    skipped: int                    # Acceptable, recorded for audit
    duration_seconds: float         # How long the full run took
    
    # Quality metrics
    coverage_percent: float         # Coverage at lock time
    coverage_floor: float           # Ratchet floor (cannot decrease)
    ihsan_score: float              # Composite Ihsān at lock time
    snr_score: float                # SNR at lock time
    
    # Constitutional
    constitutional_constants_hash: str  # BLAKE2b of constants.py
    genesis_gate_passed: bool       # Did Genesis-100 gate pass?
    
    # Provenance
    node_id: str                    # Which node ran the tests
    signature: str                  # Ed25519 signature of the receipt
    prev_lock_hash: str             # Hash chain — links to previous lock
    
    def compute_hash(self) -> str:
        """BLAKE2b hash of all fields — the lock's identity."""
        ...
```

### 3.3 Lock Conditions

A version can be locked **only if ALL conditions are met**:

| Condition | Requirement | Rationale |
|-----------|------------|-----------|
| All tests pass | 0 failures | Broken code is never locked |
| Coverage ≥ floor | coverage_percent ≥ coverage_floor | Ratchet never goes down |
| No CRITICAL security findings | bandit + cargo-audit clean | Security is a lock gate |
| Constitutional constants unchanged OR reviewed | Hash matches previous OR explicit review | Prevents accidental threshold drift |
| Commit is on main branch | `git branch --contains` | Only main is lockable |
| Previous lock exists or this is v0.1.0 | Hash chain integrity | Every lock links to the previous one |

### 3.4 Semantic Versioning for BIZRA

```
v{MAJOR}.{MINOR}.{PATCH}

MAJOR = Constitutional epoch (0 = pre-Genesis, 1 = Genesis, 2 = Forest)
MINOR = Sprint/Phase number (77, 78, 79, 80...)
PATCH = Lock sequence within sprint (0, 1, 2...)

Examples:
  v0.77.0  — Phase 77 first lock
  v0.80.0  — Phase 80 first lock (current)
  v0.80.1  — Phase 80 second lock (after hotfix)
  v1.0.0   — GENESIS (Ramadan 2026)
  v1.1.0   — First post-Genesis sprint
```

---

## 4. Delta Test Engine

### 4.1 How Delta Detection Works

When you change a file, the delta engine determines which tests are affected:

```
1. Developer edits: core/token/bloom.py
2. Delta engine scans import graph:
   - core/token/bloom.py is imported by:
     - core/sovereign/api.py
     - core/living_ecosystem.py
     - tests/core/token/test_bloom.py
     - tests/core/token/test_ledger.py
     - tests/integration/test_wallet_flow.py
3. Only those test files run (not the other 8,490)
```

### 4.2 Implementation: pytest-incremental + Custom Graph

```python
# bizra_delta_test.py — Delta Test Runner

import subprocess
import json
from pathlib import Path
from typing import Set

# Location of the last locked version's test manifest
LOCK_MANIFEST = Path("sovereign_state/test_locks/current.json")

def get_changed_files(since_tag: str) -> Set[str]:
    """Get files changed since the last locked version."""
    result = subprocess.run(
        ["git", "diff", "--name-only", since_tag, "HEAD"],
        capture_output=True, text=True
    )
    return set(result.stdout.strip().split("\n"))

def get_affected_tests(changed_files: Set[str]) -> Set[str]:
    """Given changed source files, find all test files that depend on them."""
    affected = set()
    
    for changed in changed_files:
        # Direct test files for changed modules
        if changed.startswith("tests/"):
            affected.add(changed)
            continue
            
        # Find tests that import the changed module
        module_path = changed.replace("/", ".").replace(".py", "")
        result = subprocess.run(
            ["grep", "-rl", module_path, "tests/"],
            capture_output=True, text=True
        )
        for test_file in result.stdout.strip().split("\n"):
            if test_file:
                affected.add(test_file)
    
    return affected

def run_delta(tier: str = "T1") -> dict:
    """Run only affected tests since last locked version."""
    manifest = json.loads(LOCK_MANIFEST.read_text())
    last_tag = manifest["git_tag"]
    
    changed = get_changed_files(last_tag)
    
    if not changed:
        print(f"No changes since {last_tag}. All {manifest['total_tests']} tests still locked ✓")
        return {"status": "locked", "tests_run": 0}
    
    affected_tests = get_affected_tests(changed)
    
    if not affected_tests:
        print(f"Changes don't affect any test files. Locked baseline holds ✓")
        return {"status": "locked", "tests_run": 0}
    
    # Run only affected tests
    test_files = " ".join(sorted(affected_tests))
    print(f"Running {len(affected_tests)} affected test files (not {manifest['total_tests']})")
    
    result = subprocess.run(
        f"pytest {test_files} -x -q --timeout=60",
        shell=True, capture_output=True, text=True
    )
    
    return {
        "status": "pass" if result.returncode == 0 else "fail",
        "tests_run": len(affected_tests),
        "total_locked": manifest["total_tests"],
        "changed_files": len(changed),
    }
```

### 4.3 CLI Integration

```bash
# During development — fast feedback
bizra test                    # T0: Smoke (30 sec)
bizra test --delta            # T1: Only tests affected by your changes
bizra test --contract         # T2: Contract tests + affected integration

# When ready to lock a version
bizra test --full             # T3: All 8,495+ tests
bizra test --lock             # T3 + lock receipt + git tag + sign

# Release candidate
bizra test --genesis-gate     # T4: 68 checks + full tests + constitutional audit
```

### 4.4 What Each Tier Includes

**T0 — Smoke (30 seconds)**
```python
SMOKE_TESTS = [
    "tests/core/integration/test_constants_integrity.py",    # Constitutional thresholds
    "tests/core/test_imports.py",                             # All modules importable
    "tests/core/constitutional/test_ticker_smoke.py",         # Heartbeat functional
    "tests/core/token/test_bloom_smoke.py",                   # Token math correct
    "tests/core/sovereign/test_api_exposure_policy.py",       # API routes declared
    "tests/core/auth/test_middleware_smoke.py",                # Auth gates functional
]
# ~50 tests, < 30 seconds, catches: import errors, constant drift, 
# broken gates, syntax errors
```

**T1 — Unit (2 minutes)**
```
Delta-detected tests from changed modules only.
Uses import graph to find affected tests.
Runs with -x (stop on first failure).
Typical: 50-500 tests depending on change scope.
```

**T2 — Contract (5 minutes)**
```python
CONTRACT_TESTS = [
    "tests/core/sovereign/test_api_exposure_policy.py",      # Route contracts
    "tests/integration/test_contract_integrity.py",           # Type contracts (128)
    "tests/core/sovereign/test_terminal.py",                  # Terminal spine (47)
    "tests/core/orchestration/test_learning_loop.py",         # Learning loop (34)
    "tests/core/test_learning_loop_bridges.py",               # Bridge contracts (29)
    "tests/core/test_living_ecosystem.py",                    # Ecosystem wiring (34)
    "tests/core/constitutional/test_ticker.py",               # Constitutional (281)
]
# ~800-1200 tests, < 5 minutes, catches: API drift, type mismatches,
# integration breaks, constitutional violations
```

**T3 — Full (20-30 minutes)**
```bash
pytest tests/ -q --timeout=120 \
  -m "not slow and not requires_ollama and not requires_gpu and not requires_network" \
  --cov=core --cov-report=term
# All 8,495+ tests. Coverage measured. Lock candidate.
```

**T4 — Genesis Gate (30-60 minutes)**
```bash
python genesis_gate.py          # 68 constitutional checks
pytest tests/ -q --timeout=120  # Full test suite
bandit -r core/ -ll             # Security scan
cargo test --workspace          # Rust tests
cargo audit                     # Dependency audit
```

---

## 5. Lock Workflow

### 5.1 Daily Development Flow

```
Morning:
  1. Pull latest main
  2. Check: is the current HEAD a locked version? (bizra test --status)
  3. Start development

During coding:
  4. Save file → T0 smoke runs automatically (< 30 sec)
  5. Ready to commit → T1 delta tests (< 2 min)
  6. Push to main → T2 contract tests in CI (< 5 min)

End of sprint:
  7. Ready to lock → T3 full tests (20-30 min)
  8. All pass → bizra test --lock → creates v0.80.1 tag + receipt
  9. Receipt signed, pushed, locked forever
```

### 5.2 Lock Command

```bash
$ bizra test --lock

🔒 BIZRA Version Lock — Starting T3 Full Suite

Running 8,495 tests...
████████████████████████████████████████████ 100%

Results:
  ✅ 8,495 / 8,495 PASSED
  📊 Coverage: 42.3% (floor: 38.0% → ratcheted to 42.3%)
  🏛️  Ihsān: 0.96
  📡 SNR: 0.941
  🔐 Constants hash: a3f7c9... (unchanged ✓)

Lock version? [y/N]: y

Creating lock...
  ✅ Git tag: v0.80.0
  ✅ TestLockReceipt signed (Ed25519)
  ✅ Receipt chained to v0.79.0 (prev_hash: 4e2b...)
  ✅ Coverage floor ratcheted: 38.0% → 42.3%
  ✅ Lock manifest saved: sovereign_state/test_locks/v0.80.0.json

🔒 Version v0.80.0 LOCKED

Next development cycle: only T0-T2 needed until next lock.
```

### 5.3 Lock Status Command

```bash
$ bizra test --status

🔒 Current Lock: v0.80.0 (locked 2026-03-09 04:00 GST)
   Commit: 8e57ca2
   Tests: 8,495 / 8,495 PASS
   Coverage: 42.3%

📝 Since lock:
   Changed files: 3
   Affected tests: 47
   Locked (untouched): 8,448

💡 Run: bizra test --delta (47 tests, ~30 sec)
   Skip: 8,448 tests (proven by lock v0.80.0)
```

---

## 6. Coverage Ratchet Integration

### 6.1 Coverage Never Goes Down

Every locked version records its coverage percentage. The next lock must have coverage ≥ the previous lock's coverage. This is the **coverage ratchet** — it only moves up.

```
v0.77.0  → 38.0% (first measured floor)
v0.78.0  → 39.1% (ratcheted up)
v0.79.0  → 41.5% (ratcheted up)
v0.80.0  → 42.3% (ratcheted up)
v1.0.0   → 50.0% (Genesis target)
```

### 6.2 CI Integration

```yaml
# .github/workflows/ci.yml — Test tier integration

jobs:
  t0-smoke:
    name: "T0: Smoke"
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/core/integration/test_constants_integrity.py tests/core/test_imports.py -x -q
    # Runs on every push — 30 seconds

  t1-delta:
    name: "T1: Delta"
    runs-on: ubuntu-latest
    needs: t0-smoke
    steps:
      - run: python bizra_delta_test.py --tier T1
    # Runs on every push — only affected tests

  t2-contract:
    name: "T2: Contract"  
    runs-on: ubuntu-latest
    needs: t1-delta
    if: github.ref == 'refs/heads/main'
    steps:
      - run: python bizra_delta_test.py --tier T2
    # Runs on merge to main — contract + integration

  t3-full:
    name: "T3: Full Lock"
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
      - run: pytest tests/ --cov=core --cov-report=term -q
      - run: python bizra_lock.py --verify
    # Runs on version tags only — full suite + lock verification
```

---

## 7. File Structure

```
sovereign_state/
  test_locks/
    current.json                    # Points to latest lock
    v0.77.0.json                    # Lock receipt for v0.77.0
    v0.78.0.json                    # Lock receipt for v0.78.0
    v0.79.0.json                    # ...
    v0.80.0.json                    # Latest lock
    
  test_config/
    smoke_tests.json                # T0 test list
    contract_tests.json             # T2 test list
    import_graph.json               # Module dependency graph (auto-generated)
    
  coverage/
    v0.77.0.coverage                # Coverage data per version
    v0.78.0.coverage
    v0.80.0.coverage
```

---

## 8. Constitutional Constraints

| Rule | Requirement | Source |
|------|------------|--------|
| **Locks are immutable** | Once a version is locked, the receipt cannot be changed | Evidence chain integrity |
| **Lock chain is ordered** | Each lock links to the previous via prev_lock_hash | Hash chain (like blockchain) |
| **Coverage ratchet** | Coverage floor can only increase, never decrease | الإتقان (An-Naml 27:88) — mastery always improves |
| **Zero failures required** | A lock with even 1 failure is rejected | العدل — no exceptions for the system's own quality |
| **Constants hash check** | If constitutional constants change, the lock requires explicit review | Threshold drift prevention |
| **Lock receipts are signed** | Ed25519 signature by the node that ran the tests | Provenance — who verified this? |

---

## 9. Migration Path (Current → Locked)

### 9.1 Today (Phase 80 Complete)

```bash
# Step 1: Run full T3 suite
pytest tests/ -q --timeout=120 \
  -m "not slow and not requires_ollama and not requires_gpu" \
  --cov=core --cov-report=term

# Step 2: If all pass, create first lock
git tag -s v0.80.0 -m "Phase 80: 8,495 tests, closed-loop wired"
# Save test results as lock receipt

# Step 3: All future development uses delta testing
bizra test --delta   # Only changed modules
```

### 9.2 Implementation Priority

| Priority | Task | Hours | Deliverable |
|----------|------|-------|-------------|
| **P0** | Create first lock receipt (v0.80.0) | 2 | Baseline established |
| **P1** | `bizra test` CLI commands (smoke/delta/contract/full/lock) | 8 | Developer workflow |
| **P2** | Import graph generator | 4 | Delta detection accuracy |
| **P3** | CI tier integration (T0-T3 jobs) | 4 | Automated pipeline |
| **P4** | TestLockReceipt on evidence chain | 4 | Constitutional provenance |
| **P5** | Coverage ratchet automation | 2 | Floor auto-updates on lock |
| **TOTAL** | | **24 hours** | **Complete lock system** |

---

## 10. Why This Is Constitutional

البذرة Rule 7: **الانضباط والاستمرارية** — Discipline and Continuity.

Running 8,495 tests on every commit is not discipline — it's waste. Discipline is knowing which tests matter NOW and which are already proven. Continuity is the lock chain — an unbroken record of every verified state, signed and hashed, linking back to the first test that ever passed.

The lock receipt is a Proof-of-Quality. Just as SEED is earned through Proof-of-Impact, a version is earned through Proof-of-Tests. And just as BLOOM is soulbound, a lock receipt is immutable — once verified, it cannot be changed.

> الإتقان — "إن الله يحب إذا عمل أحدكم عملاً أن يتقنه"
> *"God loves that when one of you does a work, he perfects it."*
> — Hadith (Al-Bayhaqi)

You don't prove mastery by repeating the same test forever. You prove it by locking the result and building higher.

---

## 11. The Line

> **Lock once. Run delta. Ship fast.**

8,495 tests prove the system works. The lock receipt remembers that proof forever. Future development only tests what changed. The 20-minute wait becomes 30 seconds. The developer builds instead of waiting.

Every lock is a step. Every step is signed. Every signature links to the last. From v0.80.0 to v1.0.0-GENESIS — an unbroken chain of verified quality.

One lock, one proof, remembered forever.

**LOCKED: v1.0 · 2026-03-09 · Dubai · BIZRA Foundation**
