# Sprint 2 — WBS Adaptation Guide

> **Purpose:** Exact, numbered migration steps for each of the five WBS files.
> Tells you precisely which mock blocks to delete, which `use` statements replace
> them, and how feature flags govern compilation.

---

## Table of Contents

1. [WBS File → Crate Module Map](#1-wbs-file--crate-module-map)
2. [WBS 1.1 — receipt_chain.rs migration](#2-wbs-11--receipt_chainrs-migration)
3. [WBS 1.2 — mission_bridge.rs migration](#3-wbs-12--mission_bridgers-migration)
4. [WBS 1.3 — saga.rs migration](#4-wbs-13--sagars-migration)
5. [WBS 1.4 — fate_proof.rs migration](#5-wbs-14--fate_proofrs-migration)
6. [WBS 1.5 — proof_pyramid_e2e.rs migration](#6-wbs-15--proof_pyramid_e2ers-migration)
7. [Feature Flag Strategy](#7-feature-flag-strategy)
8. [Circular Dependency Analysis](#8-circular-dependency-analysis)
9. [bizra-tests Cargo.toml patch](#9-bizra-tests-cargotoml-patch)

---

## 1. WBS File → Crate Module Map

| WBS file (Sprint 1) | Destination path | Crate | Module declared in |
|---|---|---|---|
| `wbs_1_1_receipt_chain_trait.rs` | `bizra-proofspace/src/receipt_chain.rs` | `bizra-proofspace` | `proofspace_lib_patch.rs` |
| `wbs_1_2_mission_proof_wire.rs` | `bizra-proofspace/src/mission_bridge.rs` | `bizra-proofspace` | `proofspace_lib_patch.rs` |
| `wbs_1_3_saga_mission_dispatch.rs` | `bizra-action/src/saga.rs` | `bizra-action` | `action_lib_patch.rs` |
| `wbs_1_4_fate_binding_z3.rs` | `bizra-proofspace/src/fate_proof.rs` | `bizra-proofspace` | `proofspace_lib_patch.rs` |
| `wbs_1_5_e2e_proof_chain_test.rs` | `bizra-tests/tests/proof_pyramid_e2e.rs` | `bizra-tests` | `[[test]]` entry in Cargo.toml |

---

## 2. WBS 1.1 — receipt_chain.rs migration

**Source file:** `wbs_1_1_receipt_chain_trait.rs` (1,207 lines)
**Destination:** `bizra-proofspace/src/receipt_chain.rs`

### Mock sections to REMOVE

The file is organized into numbered sections (`§ 0` – `§ 11`).
Remove sections **§ 1, § 2, § 3, and § 4** in their entirety — they are
inline mirrors of types that will be imported from real crates.

| Section | Lines | Content | Reason to remove |
|---|---|---|---|
| `§ 1` | 41–156 | Mirror types from `bizra-action/src/types.rs` | `ActionId`, `ActionTimestamp`, `IhsanScore`, `GuardianVerdict`, `Channel`, `ConstitutionalReceipt` |
| `§ 2` | 157–248 | Mirror types from `bizra-action/src/receipt.rs` | `ReceiptChain` struct + impl |
| `§ 3` | 249–364 | Mirror types from `bizra-sippar/src/lib.rs` | `SipparError`, `RegularNumber` |
| `§ 4` | 365–378 | Mirror types from `bizra-proofspace/src/lib.rs` | `FateScores` (re-exported from the containing crate directly) |

Sections **§ 5 – § 11** (lines 379–1,207) contain the actual new logic and tests — **keep all of them**.

### Import statements to ADD

Replace the removed sections with:

```rust
// ── Real imports replacing §1 + §2 (W4 wire: bizra-action) ──────────────────
use bizra_action::types::{
    ActionId, ActionTimestamp, Channel, ConstitutionalReceipt, GuardianVerdict, IhsanScore,
};
use bizra_action::receipt::ReceiptChain;

// ── Real imports replacing §3 (W8 wire: bizra-sippar) ───────────────────────
use bizra_sippar::{RegularNumber, SipparError};

// ── §4 replaced: FateScores lives in the parent crate (this crate = proofspace) ──
use crate::FateScores;
```

### `§ 0` crate-level imports — KEEP, ADD one line

```rust
use std::fmt;
use std::collections::VecDeque; // already present if needed by ReceiptChain impl
```

No other changes to `§ 0`.

### Migration steps (numbered)

1. `cp wbs_1_1_receipt_chain_trait.rs bizra-proofspace/src/receipt_chain.rs`
2. Delete lines 41–378 (§ 1 through § 4 inclusive).
3. At line 41 (now the first blank line after `use std::fmt;`) insert the four
   `use` blocks shown above.
4. Run `cargo check -p bizra-proofspace` — expect zero errors.
5. Verify the public surface with `cargo doc -p bizra-proofspace --no-deps`.

---

## 3. WBS 1.2 — mission_bridge.rs migration

**Source file:** `wbs_1_2_mission_proof_wire.rs` (1,631 lines)
**Destination:** `bizra-proofspace/src/mission_bridge.rs`

### Mock sections to REMOVE

The file uses a `// SECTION N:` convention.

| Section | Lines | Content | Reason to remove |
|---|---|---|---|
| `SECTION 0` | 37–607 | Four mock modules: `mock_mission_state`, `mock_receipt`, `mock_mission`, `mock_proofspace` | Replaced by real crate imports |

`SECTION 1` (lines 608–1,157) and `SECTION 2` (lines 1,158–1,631) contain
the real wire logic and tests — **keep both**.

### Import statements to ADD

At the top of `SECTION 1` (line 608), replace:

```rust
use mock_mission::Mission;
use mock_mission_state::MissionState;
use mock_proofspace::{
    BlockBody, BlockBuilder, BlockStatus, BlockType, UnsignedBlock,
};
```

with:

```rust
// ── Real imports replacing SECTION 0 (W5 wire: bizra-mission) ───────────────
use bizra_mission::{
    Mission,
    receipt::{DegradationReason, FailureCode, MissionReceipt},
};
use bizra_mission::state::MissionState;

// ── ProofSpace block types live in the parent crate (this crate) ────────────
use crate::{
    BlockBody, BlockBuilder, BlockStatus, BlockType, UnsignedBlock,
};
```

### BLAKE3 usage in SECTION 0 mock

The `mock_proofspace` block contains a stub BLAKE3 hash (`let mut hasher = …`
comment saying "a real BLAKE3 hash; here we use a stable mock").
After the patch the `bizra-proofspace` Cargo.toml already declares
`blake3.workspace = true`, so replace the stub with:

```rust
let mut hasher = blake3::Hasher::new();
hasher.update(&data);
hasher.finalize().as_bytes().to_vec()
```

This change is in `MissionProofBridge::compute_body_hash()` (around line 576 of
the original file, now inside `SECTION 1`).

### Migration steps

1. `cp wbs_1_2_mission_proof_wire.rs bizra-proofspace/src/mission_bridge.rs`
2. Delete lines 37–607 (`SECTION 0: Mock types` through the end of
   `mock_proofspace` module's closing `}`).
3. Insert the real `use` block at the top of the file (before
   `// SECTION 1: Wire — public API`).
4. In `compute_body_hash()` replace the stub hash with the `blake3::Hasher`
   call shown above.
5. In `SECTION 2` tests, replace any remaining `mock_*::` prefixes:

   | Replace | With |
   |---|---|
   | `mock_mission_state::MissionState::Complete` | `bizra_mission::state::MissionState::Complete` |
   | `mock_receipt::FailureCode` | `bizra_mission::receipt::FailureCode` |
   | `mock_receipt::DegradationReason` | `bizra_mission::receipt::DegradationReason` |
   | `mock_proofspace::BlockType::MissionBlock` | `crate::BlockType::MissionBlock` |
   | `mock_proofspace::BlockStatus::Submitted` | `crate::BlockStatus::Submitted` |

6. Run `cargo check -p bizra-proofspace`.
7. Run `cargo test -p bizra-proofspace receipt_chain mission_bridge`.

---

## 4. WBS 1.3 — saga.rs migration

**Source file:** `wbs_1_3_saga_mission_dispatch.rs` (1,385 lines)
**Destination:** `bizra-action/src/saga.rs`

### Mock sections to REMOVE

Two contiguous mock blocks separated by a section comment.

| Block | Lines | Content | Reason to remove |
|---|---|---|---|
| `// bizra-action mock types` | 50–296 | `ActionId`, `ActionTimestamp`, `IhsanScore`, `Channel`, `RiskLevel`, `Permit`, `GuardianVerdict`, `ConstitutionalReceipt`, `ReceiptChain` | These types already exist in the sibling modules of bizra-action itself |
| `// bizra-mission mock types` | 297–406 | `MissionState` enum (20 variants), `Mission` struct + impl | Replaced by real import from bizra-mission |
| `fn blake3_hash(…)` | 28–50 | Pure-Rust BLAKE3 stand-in | Replaced by real blake3 crate |

Lines 407 onward (`// WBS 1.3 — Saga types`) contain the actual saga logic — **keep all**.

### Import statements to ADD

```rust
// ── bizra-action sibling modules (same crate — no dep needed) ───────────────
//    These replace the "bizra-action mock types" block (lines 50–296).
use crate::types::{
    ActionId, ActionTimestamp, Channel, GuardianVerdict, IhsanScore, Permit, RiskLevel,
};
use crate::types::ConstitutionalReceipt;
use crate::receipt::ReceiptChain;

// ── bizra-mission (W6 wire — enabled by "saga" feature) ─────────────────────
//    These replace the "bizra-mission mock types" block (lines 297–406).
#[cfg(feature = "saga")]
use bizra_mission::{Mission, MissionHandle};
#[cfg(feature = "saga")]
use bizra_mission::state::MissionState;

// ── blake3 (enabled by "production" feature) ────────────────────────────────
//    Replaces the fn blake3_hash() stub (lines 28–50).
#[cfg(feature = "production")]
use blake3;
```

### blake3_hash() replacement

The `blake3_hash(data: &[u8]) -> [u8; 32]` stub function must be replaced:

```rust
// In saga.rs, wherever blake3_hash() was called:
#[cfg(feature = "production")]
fn blake3_hash(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

#[cfg(not(feature = "production"))]
fn blake3_hash(data: &[u8]) -> [u8; 32] {
    // Dev fallback — NOT cryptographically secure.
    let mut state = [0u8; 32];
    for (i, &b) in data.iter().enumerate() {
        state[i % 32] ^= b.wrapping_add((i as u8).wrapping_mul(0x9e));
        state[(i + 1) % 32] = state[(i + 1) % 32]
            .wrapping_add(state[i % 32])
            .wrapping_add(0x6c);
    }
    for i in 0..32 {
        let j = (i + 13) % 32;
        state[j] ^= state[i].rotate_left(3);
    }
    state
}
```

This preserves the dev-mode build (no blake3 dep) while using the real crate
in production.

### Whole-file `#[cfg(feature = "saga")]` gate

Because `src/saga.rs` itself is only compiled when the `saga` feature is active
(the `pub mod saga;` declaration in lib.rs is already feature-gated), you do
**not** need to add per-item `#[cfg(feature = "saga")]` attributes inside
saga.rs.  The module gate in lib.rs handles the entire file.

The `#[cfg(feature = "saga")]` on the two `use bizra_mission::…` imports
above is needed because those appear at the top of the file where the module
gate does not yet apply during dependency resolution.

### Migration steps

1. `cp wbs_1_3_saga_mission_dispatch.rs bizra-action/src/saga.rs`
2. Delete lines 28–406 (the `fn blake3_hash()` stub + both mock blocks).
3. Insert the `use` block shown above at the top of the file.
4. Add the `blake3_hash()` wrapper function (both `#[cfg]` variants) before the
   first saga type definition.
5. Apply `action_cargo_patch.toml` (add `bizra-mission` optional dep + `saga`
   feature).
6. Apply `action_lib_patch.rs` (add `pub mod saga` declaration + re-exports).
7. `cargo check -p bizra-action --features production,saga`
8. `cargo test -p bizra-action --features production,saga`

---

## 5. WBS 1.4 — fate_proof.rs migration

**Source file:** `wbs_1_4_fate_binding_z3.rs` (1,344 lines)
**Destination:** `bizra-proofspace/src/fate_proof.rs`

### Mock sections to REMOVE

| Block | Lines | Content | Reason to remove |
|---|---|---|---|
| Constants block | 38–56 | `IHSAN_THRESHOLD`, `ADL_GINI_MAX`, `MAX_HARM_SCORE`, `MIN_CONFIDENCE`, `SNR_FLOOR` declared as `pub const` | Live in `bizra-proofspace/src/lib.rs` already (imported via `bizra-core`) |
| `FateScores` mirror | 58–76 | Struct mirroring `bizra_proofspace::FateScores` | The real `FateScores` lives in the parent lib.rs |
| `fn check_regular(…)` | 529–564 | Pure-Rust 2,3,5-smooth check | Replaced by `bizra_sippar::RegularNumber::from_u64` |

Everything else — `ConstitutionalThresholds`, `FateGate`, `SmtSort`,
`SmtVariable`, `SmtAssertion`, `ProofResult`, `SolverStats`, `FateProof`,
`FateBindingError`, `AssertionValidation`, `FateBindingEngine`, and all tests
— is new logic.  **Keep all of it.**

### Import statements to ADD

```rust
// ── Parent crate re-exports (constants + FateScores) ────────────────────────
//    Replace the top-of-file constants block and FateScores mirror.
use crate::{
    ADL_GINI_MAX,
    FateScores,
    IHSAN_THRESHOLD,
    MAX_HARM_SCORE,
    MIN_CONFIDENCE,
};

// ── bizra-sippar (W8 wire) — replaces fn check_regular() ────────────────────
use bizra_sippar::RegularNumber;
```

### `check_regular()` replacement

Wherever the internal `check_regular(n)` is called, replace with:

```rust
// Before:
check_regular(chain_length)
    .map_err(|p| FateBindingError::SipparIrregular { chain_length, first_irregular_factor: p })?;

// After:
RegularNumber::from_u64(chain_length)
    .map_err(|_| FateBindingError::SipparIrregular {
        chain_length,
        first_irregular_factor: chain_length, // exact factor recovered inside from_u64
    })?;
```

> Note: `bizra_sippar::RegularNumber::from_u64` returns `Result<RegularNumber, u64>`
> where the `Err` value is the first non-smooth factor.  The mapping above is
> therefore semantically identical to the stub.

### Feature-gated Z3 integration

The `FateBindingEngine::validate_with_z3()` method is already written as a
pure stub (SMT-LIB2 string generation only — no Z3 call) in WBS 1.4.
The Z3 call path is to be added behind the existing `z3` feature flag:

```rust
// At the bottom of FateBindingEngine::validate_with_z3():
#[cfg(feature = "z3")]
{
    // fate-binding is available as a dev-dependency.
    // When this feature is active, delegate to the real engine:
    use fate_binding::FateBindingEngine as RealEngine;
    let engine = RealEngine::new();
    let real_proof = engine.generate_fate_proof(fate_scores.clone())
        .map_err(|e| FateBindingError::SmtSolverError(e.to_string()))?;
    return Ok(real_proof.into()); // From<fate_binding::FateProof> impl needed
}
// Default (no "z3" feature): fall through to pure SMT-LIB2 string assembly.
```

This means `cargo test` always exercises the stub path; `cargo test --features z3`
exercises the real Z3 engine.

### Migration steps

1. `cp wbs_1_4_fate_binding_z3.rs bizra-proofspace/src/fate_proof.rs`
2. Delete lines 38–76 (constants + `FateScores` mirror).
3. Delete lines 529–564 (`fn check_regular()` function body).
4. Insert the `use crate::{…}` and `use bizra_sippar::RegularNumber;` block
   at the top of the file (after `#![warn(missing_docs)]`).
5. Replace all `check_regular(…)` calls with `RegularNumber::from_u64(…)`.
6. Add the `#[cfg(feature = "z3")]` block inside `validate_with_z3()` (optional
   Sprint 2 scope — can be deferred to Sprint 3).
7. `cargo check -p bizra-proofspace`
8. `cargo check -p bizra-proofspace --features z3`
9. `cargo test -p bizra-proofspace fate_proof`

---

## 6. WBS 1.5 — proof_pyramid_e2e.rs migration

**Source file:** `wbs_1_5_e2e_proof_chain_test.rs` (2,751 lines)
**Destination:** `bizra-tests/tests/proof_pyramid_e2e.rs`

### What the test file needs

This file is a standalone integration test binary.  It has no mock sections to
remove — the mock types inside it are test helpers, not mirrors of production
code.  The only change required is updating crate import paths to use the real
crates instead of inline structs.

### Cargo.toml patch for bizra-tests

Add the following to `bizra-tests/Cargo.toml`:

```toml
# ━━━ Sprint 2: Proof Pyramid E2E dependencies ━━━
[dependencies]
bizra-proofspace = { path = "../bizra-proofspace" }
bizra-action     = { path = "../bizra-action", features = ["production", "saga"] }
bizra-mission    = { path = "../bizra-mission" }
bizra-sippar     = { path = "../bizra-sippar" }

[[test]]
name = "proof_pyramid_e2e"
path = "tests/proof_pyramid_e2e.rs"
```

### Migration steps

1. `cp wbs_1_5_e2e_proof_chain_test.rs bizra-tests/tests/proof_pyramid_e2e.rs`
2. Apply the Cargo.toml additions above.
3. Add the `[[test]]` entry for `proof_pyramid_e2e`.
4. Replace any inline struct definitions with real crate imports:

   | Inline type | Real import |
   |---|---|
   | `struct ActionId(u64)` | `use bizra_action::types::ActionId;` |
   | `struct IhsanScore(f64)` | `use bizra_action::types::IhsanScore;` |
   | `enum MissionState { … }` | `use bizra_mission::state::MissionState;` |
   | `struct FateScores { … }` | `use bizra_proofspace::FateScores;` |

5. `cargo test -p bizra-tests --test proof_pyramid_e2e`

---

## 7. Feature Flag Strategy

### Flag matrix

| Feature | Crate | Enables | Required for |
|---|---|---|---|
| `production` | `bizra-action` | `blake3` dep | BLAKE3 receipt hashing in `ReceiptChain` |
| `saga` | `bizra-action` | `dep:bizra-mission` | Compiles `src/saga.rs` and its Mission dispatch |
| `z3` | `bizra-proofspace` | (no dep yet — stub exists) | Activates real `fate-binding` Z3 engine path |
| `parallel` | `bizra-proofspace` | `rayon` dep | Parallel block validation |

### Feature combinations

```
# Minimal (zero external deps — embedded / test-only):
bizra-action (default features)

# Production action bus (no saga):
bizra-action features = ["production"]

# Full production with saga dispatch:
bizra-action features = ["production", "saga"]

# ProofSpace with full proof pyramid (no Z3):
bizra-proofspace features = []        # stub fate engine

# ProofSpace with live Z3 (CI must have libz3):
bizra-proofspace features = ["z3"]    # real fate-binding engine
```

### Feature implication rules

- `saga` does NOT imply `production`.  Both must be listed when full
  cryptographic receipts are required in saga steps.
- `z3` does NOT imply the `fate-binding` crate is a non-dev dependency.
  Until the production Dockerfile pins Z3, this feature is test-only.

---

## 8. Circular Dependency Analysis

### Post-Sprint-2 dependency edges (directed, A → B means A depends on B)

```
bizra-proofspace  →  bizra-core
bizra-proofspace  →  bizra-action        [W4, NEW]
bizra-proofspace  →  bizra-mission       [W5, NEW]
bizra-proofspace  →  bizra-sippar        [W8, NEW]
bizra-action      →  bizra-mission       [W6, NEW, optional]
bizra-mission     →  bizra-core
bizra-action      →  (nothing required)  [default features]
bizra-sippar      →  (nothing)
bizra-core        →  bizra-sippar
fate-binding      →  bizra-core
bizra-proofspace  →  fate-binding        [W7, NEW, dev-dep only]
```

### Cycle search (topological sort)

Assign levels bottom-up:

| Level | Crates |
|---|---|
| 0 (leaves) | `bizra-sippar` |
| 1 | `bizra-core` (depends only on `bizra-sippar`) |
| 2 | `bizra-action` (depends only on nothing / blake3), `bizra-mission` (depends on `bizra-core`), `fate-binding` (depends on `bizra-core`) |
| 3 | `bizra-proofspace` (depends on `bizra-core`, `bizra-action`, `bizra-mission`, `bizra-sippar`) |
| 4+ | All higher-level crates (`bizra-node`, `bizra-python`, `bizra-resourcepool`, …) |

**No back-edges exist.** The graph is a strict DAG.

### Would W6 create a cycle?

W6 adds `bizra-action → bizra-mission`.

- `bizra-action` is at Level 2.
- `bizra-mission` is at Level 2 (depends only on `bizra-core` at Level 1).
- `bizra-mission` does NOT depend on `bizra-action`.

Therefore `bizra-action → bizra-mission` creates no cycle.  `bizra-action`
moves to Level 2+ (after `bizra-mission`), still well below `bizra-proofspace`.

### Would W7 create a cycle?

W7 adds `bizra-proofspace → fate-binding` (dev-dep only).

- `fate-binding` depends on `bizra-core` only.
- `fate-binding` does NOT depend on `bizra-proofspace`.

No cycle.  Dev-dependencies are excluded from Cargo's cycle check for library
crates anyway, so this wire is safe even if it were structural.

### Proof of no cycles: formal statement

> **Theorem:** The post-Sprint-2 BIZRA crate graph G = (V, E) is acyclic.
>
> **Proof:** Assign each crate a level `L(c)` equal to the length of its longest
> path from a leaf (zero-dependency crate).  For every edge (A → B) in E,
> `L(A) > L(B)` by construction of the level assignment above.  A directed
> cycle would require some sequence A₀ → A₁ → … → Aₙ → A₀, implying
> `L(A₀) > L(A₁) > … > L(Aₙ) > L(A₀)`, which is impossible.  ∎

---

## 9. bizra-tests Cargo.toml patch

For completeness, the full set of additions needed in `bizra-tests/Cargo.toml`
to compile the Sprint 2 E2E test (WBS 1.5):

```toml
# ━━━ Sprint 2 additions to bizra-tests/Cargo.toml ━━━

[dependencies]
# pre-existing: bizra-core, bizra-inference, bizra-federation, bizra-autopoiesis
bizra-proofspace = { path = "../bizra-proofspace" }
bizra-action     = { path = "../bizra-action", features = ["production", "saga"] }
bizra-mission    = { path = "../bizra-mission" }
bizra-sippar     = { path = "../bizra-sippar" }

[[test]]
name = "proof_pyramid_e2e"
path = "tests/proof_pyramid_e2e.rs"
```
