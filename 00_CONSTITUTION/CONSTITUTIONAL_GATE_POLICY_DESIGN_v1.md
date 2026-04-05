# Constitutional Gate Policy — Wire 2.5 Design Document

**Version:** 1.0  
**Date:** 2026-04-05  
**Status:** PROPOSED — requires review before implementation  
**Sprint:** Constitutional Gate Unification  
**Standing on Giants:** Al-Ghazali (Ihsan as obligation) · Shannon (measurable quality) · Lamport (distributed consistency) · Deming (quality at source)

---

## 1. Problem Statement

The same threshold check (`ihsan < 0.95`) triggers **5 different behaviors** across the codebase:

| Module | File | Behavior | Env-aware? |
|--------|------|----------|------------|
| IhsanGate (Rust) | `bizra-hooks/src/ihsan_gate.rs` | Policy-dependent (Observe/Flag/Throttle/Reject) | Yes (BIZRA_ENV) |
| SeedLedger (Rust) | `bizra-node/src/seed_ledger.rs` | Silent drop (`return None`) | No |
| WalkingSkeleton (Python) | `core/walking_skeleton.py` | Hard halt (`outcome: "halted"`) | No |
| SeedCalc (Python) | `core/proof_engine/seed_calc.py` | Zero reward with reason string | No |
| IhsanGate (Python) | `core/proof_engine/ihsan_gate.py` | APPROVED/REJECTED decision with reason codes | Partial (env var for floor) |

**Root cause:** `GatePolicy` lives in `bizra-hooks` (a nervous-system crate), not in `bizra-core` (the constitutional kernel). Every other module reinvents enforcement from scratch.

**Consequence:** Behavior drift — a developer adding a new Ihsan check has no canonical pattern to follow, and existing checks cannot be reconfigured without editing each module individually.

## 2. Proposed Solution

Move `GatePolicy` and a minimal verdict type into `bizra-core` as constitutional primitives. All enforcement surfaces import from `bizra-core` instead of defining their own logic.

### 2.1 Rust: `bizra-core/src/gate_policy.rs`

```rust
//! Constitutional Gate Policy — unified enforcement for Ihsan threshold violations.
//!
//! Standing on Giants: Al-Ghazali (Ihsan), Deming (quality at source)

/// What happens when a constitutional threshold is violated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GatePolicy {
    /// Log but allow (development/observation mode)
    #[default]
    Observe,
    /// Attach warning flag, still deliver
    Flag,
    /// Allow 1 in N events from violating component
    Throttle(u32),
    /// Hard reject — fail-closed
    Reject,
}

/// The result of applying a gate policy to a score.
#[derive(Debug, Clone)]
pub struct GateVerdict {
    /// The score that was evaluated
    pub score: f64,
    /// The threshold it was measured against
    pub threshold: f64,
    /// Whether the score met the threshold
    pub passed: bool,
    /// The policy that was applied
    pub policy: GatePolicy,
    /// The resulting action
    pub action: GateAction,
}

/// The action taken after policy evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateAction {
    /// Score passed threshold — proceed normally
    Allow,
    /// Score failed but policy is Observe — proceed with log
    AllowWithWarning,
    /// Score failed, policy is Flag — proceed with flag attached
    Flagged,
    /// Score failed, policy is Throttle — suppressed this time
    Throttled,
    /// Score failed, policy is Reject — hard stop
    Rejected,
}

/// Resolve the active gate policy from environment.
///
/// - `BIZRA_ENV=prod` or `BIZRA_ENV=production` → Reject
/// - Otherwise → Observe (development default)
///
/// Modules MAY override this with explicit configuration.
pub fn env_gate_policy() -> GatePolicy {
    match std::env::var("BIZRA_ENV").as_deref() {
        Ok("prod") | Ok("production") => GatePolicy::Reject,
        _ => GatePolicy::Observe,
    }
}

/// Apply a gate policy to a score/threshold pair.
///
/// This is the ONE function all enforcement surfaces should call.
/// Throttle state (counter) is the caller's responsibility.
pub fn apply_gate(score: f64, threshold: f64, policy: GatePolicy) -> GateVerdict {
    let passed = score >= threshold;
    let action = if passed {
        GateAction::Allow
    } else {
        match policy {
            GatePolicy::Observe => GateAction::AllowWithWarning,
            GatePolicy::Flag => GateAction::Flagged,
            GatePolicy::Throttle(_) => GateAction::Throttled,
            GatePolicy::Reject => GateAction::Rejected,
        }
    };
    GateVerdict { score, threshold, passed, policy, action }
}
```

**Size:** ~80 LOC. No new dependencies. No traits to implement — just types and a pure function.

### 2.2 Python: `core/governance/gate_policy.py`

```python
"""
Constitutional Gate Policy — Python mirror of bizra-core/src/gate_policy.rs.

Standing on Giants: Al-Ghazali (Ihsan), Deming (quality at source)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Union


class GatePolicy(Enum):
    """What happens when a constitutional threshold is violated."""
    OBSERVE = "observe"
    FLAG = "flag"
    THROTTLE = "throttle"
    REJECT = "reject"


class GateAction(Enum):
    """The action taken after policy evaluation."""
    ALLOW = "allow"
    ALLOW_WITH_WARNING = "allow_with_warning"
    FLAGGED = "flagged"
    THROTTLED = "throttled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class GateVerdict:
    """The result of applying a gate policy to a score."""
    score: float
    threshold: float
    passed: bool
    policy: GatePolicy
    action: GateAction


def env_gate_policy() -> GatePolicy:
    """Resolve gate policy from BIZRA_ENV environment variable."""
    env = os.environ.get("BIZRA_ENV", "").lower()
    if env in ("prod", "production"):
        return GatePolicy.REJECT
    return GatePolicy.OBSERVE


def apply_gate(
    score: float,
    threshold: float,
    policy: GatePolicy | None = None,
) -> GateVerdict:
    """
    Apply a gate policy to a score/threshold pair.

    If policy is None, resolves from BIZRA_ENV.
    """
    if policy is None:
        policy = env_gate_policy()

    passed = score >= threshold
    if passed:
        action = GateAction.ALLOW
    elif policy == GatePolicy.OBSERVE:
        action = GateAction.ALLOW_WITH_WARNING
    elif policy == GatePolicy.FLAG:
        action = GateAction.FLAGGED
    elif policy == GatePolicy.THROTTLE:
        action = GateAction.THROTTLED
    else:
        action = GateAction.REJECTED

    return GateVerdict(
        score=score,
        threshold=threshold,
        passed=passed,
        policy=policy,
        action=action,
    )
```

**Size:** ~70 LOC. No new dependencies.

## 3. Enforcement Surface Migration Plan

Each surface adopts `apply_gate()` in a minimal, behavior-preserving way. **No behavioral changes in v1** — only centralization of the decision point.

| # | Module | Current behavior | Migration | Behavior change? |
|---|--------|-----------------|-----------|-----------------|
| 1 | `bizra-hooks/ihsan_gate.rs` | Already has GatePolicy | Re-export from `bizra-core` instead of defining locally | No |
| 2 | `bizra-node/seed_ledger.rs` | `if ihsan < IHSAN_THRESHOLD → None` | Call `apply_gate()`, map `Rejected → None` | No (same effect) |
| 3 | `core/walking_skeleton.py` | `if ihsan < threshold → halt` | Call `apply_gate()`, map `Rejected → halt` | No (same effect) |
| 4 | `core/proof_engine/seed_calc.py` | `if ihsan < threshold → zero` | Call `apply_gate()`, map `Rejected → zero` | No (same effect) |
| 5 | `core/proof_engine/ihsan_gate.py` | `if score < threshold → REJECTED` | Call `apply_gate()`, map to existing decision | No (same effect) |

**Important:** In v1, surfaces 2-5 continue to hardcode `GatePolicy::Reject` (or resolve via `env_gate_policy()`). The migration makes the policy *visible* and *configurable* — it does NOT change what happens today.

## 4. Env-Policy Truth Table

| `BIZRA_ENV` | `env_gate_policy()` | IhsanGate behavior | SeedLedger behavior | WalkingSkeleton behavior |
|-------------|--------------------|--------------------|--------------------|-----------------------|
| unset | Observe | Log warning, allow | Return None (Reject*) | Halt (Reject*) |
| `"dev"` | Observe | Log warning, allow | Return None (Reject*) | Halt (Reject*) |
| `"prod"` | Reject | Hard reject event | Return None (Reject) | Halt (Reject) |
| `"production"` | Reject | Hard reject event | Return None (Reject) | Halt (Reject) |

*(\*) Surfaces 2-5 currently always reject regardless of env. Wire 2.5 v1 preserves this — the `apply_gate()` call is present but these surfaces pass `GatePolicy::Reject` explicitly. A future Wire (5+) may relax dev-mode surfaces to `Observe`.*

## 5. Cross-Language Test Matrix

### 5.1 Rust Tests (`bizra-core/tests/gate_policy_test.rs`)

| Test | Assertion |
|------|-----------|
| `test_passing_score_always_allows` | `apply_gate(0.96, 0.95, Reject).action == Allow` |
| `test_failing_observe_allows_with_warning` | `apply_gate(0.90, 0.95, Observe).action == AllowWithWarning` |
| `test_failing_flag_returns_flagged` | `apply_gate(0.90, 0.95, Flag).action == Flagged` |
| `test_failing_throttle_returns_throttled` | `apply_gate(0.90, 0.95, Throttle(3)).action == Throttled` |
| `test_failing_reject_returns_rejected` | `apply_gate(0.90, 0.95, Reject).action == Rejected` |
| `test_exact_threshold_passes` | `apply_gate(0.95, 0.95, Reject).passed == true` |
| `test_env_gate_default_is_observe` | `env::remove_var("BIZRA_ENV"); env_gate_policy() == Observe` |
| `test_env_gate_prod_is_reject` | `env::set_var("BIZRA_ENV", "prod"); env_gate_policy() == Reject` |
| `test_env_gate_production_long_form` | `env::set_var("BIZRA_ENV", "production"); env_gate_policy() == Reject` |
| `test_verdict_fields_populated` | All fields of `GateVerdict` are set correctly |

### 5.2 Python Tests (`tests/core/governance/test_gate_policy.py`)

Mirror of Rust tests 1:1, ensuring cross-language parity on all 10 cases.

### 5.3 Parity Test

A single test that computes `apply_gate(0.93, 0.95, Reject)` in both languages and asserts identical `(passed, action)` tuple. (Existing pattern from `golden_vector` cross-language parity test.)

## 6. Integration into bizra-core

### File placement

```
bizra-omega/bizra-core/src/
├── gate_policy.rs          ← NEW (this Wire)
├── lib.rs                  ← add `pub mod gate_policy;` + re-exports
└── ... (existing modules)
```

### Re-exports in `lib.rs`

```rust
pub mod gate_policy;
pub use gate_policy::{apply_gate, env_gate_policy, GateAction, GatePolicy, GateVerdict};
```

### bizra-hooks migration

`bizra-hooks/src/ihsan_gate.rs` changes from:

```rust
pub enum GatePolicy { Observe, Flag, Throttle(u32), Reject }
```

to:

```rust
pub use bizra_core::GatePolicy;
// (remove local enum definition)
```

The existing `GateConfig`, `IhsanGate`, and all tests remain in `bizra-hooks`. Only the enum and its semantic meaning moves to `bizra-core`.

## 7. Acceptance Criteria

- [ ] `bizra-core/src/gate_policy.rs` exists with GatePolicy, GateAction, GateVerdict, apply_gate, env_gate_policy
- [ ] `bizra-core/src/lib.rs` re-exports all 5 symbols
- [ ] `bizra-hooks/src/ihsan_gate.rs` imports GatePolicy from `bizra_core` (no local definition)
- [ ] `core/governance/gate_policy.py` exists with matching types and functions
- [ ] 10 Rust tests pass in `bizra-core`
- [ ] 10 Python tests pass in `tests/core/governance/`
- [ ] Cross-language parity test passes
- [ ] `cargo test --workspace` passes (no regressions)
- [ ] `pytest tests/` passes (no regressions)
- [ ] No behavioral changes in any existing module

## 8. Rollback Condition

If any existing test breaks due to the migration, revert the `bizra-hooks` import change and keep both definitions temporarily. The `bizra-core` definition becomes canonical; `bizra-hooks` keeps its local copy marked `#[deprecated]` until all consumers migrate.

## 9. What This Does NOT Cover

- **Wire 3-5** (autopoiesis wiring) — separate design doc exists (`AUTOPOIESIS_WIRING_DESIGN_v1.md`)
- **Behavioral changes** to existing surfaces (e.g., making SeedLedger env-aware) — future wire
- **GatePolicy persistence** or serialization — not needed yet
- **Throttle counter state management** — remains caller responsibility
- **PyO3 binding** of GatePolicy — not needed until Python calls Rust gate directly

## 10. Implementation Order

1. Create `bizra-core/src/gate_policy.rs` (types + functions)
2. Add `pub mod gate_policy;` and re-exports to `bizra-core/src/lib.rs`
3. Write 10 Rust tests
4. Migrate `bizra-hooks` to import from `bizra-core`
5. Run `cargo test --workspace` — must be green
6. Create `core/governance/gate_policy.py`
7. Write 10 Python tests
8. Run `pytest tests/` — must be green
9. Commit as single atomic commit

---

*Standing on Giants: Al-Ghazali (Ihsan as obligation, not option) · Shannon (quality is measurable) · Lamport (consistency across distributed surfaces) · Deming (build quality in, don't inspect it in)*
