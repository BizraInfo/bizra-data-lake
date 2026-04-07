# WIRE4_PATTERN_RECALL_ACCEPTANCE.md
**Sprint 1 — Wire 4: Tier-1b Pattern Recall**
**Status:** FROZEN — do not expand scope
**Author:** BIZRA Node0 Audit
**Date:** 2026-04-06

---

## 1. Insertion Point

Wire 4 inserts exactly one block into `bizra-agent/src/omni_kernel.rs`,
between the existing Tier-1 Reflex return and the existing Tier-2 Engram lookup.

```
// ─── Tier-1: Reflex Cache ────────────────────────────────────────────────
if let Some(rule) = self.reflex_cache.get_active(...) {
    return CycleReceipt { path: CyclePath::ReflexHit, ... };  // early return — UNCHANGED
}

// ─── Tier-1b: Pattern Recall ◄── WIRE 4 INSERTS HERE ───────────────────
// [new block, ~12 lines]

// ─── Tier-2: Engram Cache ────────────────────────────────────────────────
match self.engram_cache.lookup(...) {                          // UNCHANGED
    EngramResult::Hit { ... } => { return ...; }
    EngramResult::Miss => {}
}

// ─── Tier-3: Full PAT inference ──────────────────────────────────────────
// UNCHANGED
```

The `intent_bytes` from `OmniCycle` serve as the embedding source.
PatternMemory API is already available via `self.pattern_memory` (Wire 3).

---

## 2. Acceptance Criteria

### 2.1 Routing Precedence Order

The three-case precedence must hold deterministically:

| Case | Condition | Expected Path | Receipt Field |
|:-----|:----------|:--------------|:--------------|
| A | Reflex cache hit | `CyclePath::ReflexHit` | Unchanged from today |
| B | Reflex miss + Pattern hit (≥ threshold) | `CyclePath::PatternHit` | New variant |
| C | Reflex miss + Pattern miss + Engram hit | `CyclePath::EngramHit` | Unchanged from today |
| D | All miss → full inference | `CyclePath::FullInference` | Unchanged from today |

Case A must never reach the pattern recall block.
Case B must never reach the Engram cache.
Case C must reach Engram only after pattern miss.

### 2.2 Threshold

- Default similarity threshold: `0.7` (stricter than PatternMemory.recall default of 0.5)
- Configurable via `OmniConfig.pattern_recall_threshold: f32` (new field, default 0.7)
- A pattern hit only routes through Tier-1b if `similarity >= config.pattern_recall_threshold`

### 2.3 No Behavioral Side Effects

Wire 4 must not touch:

- [ ] SEED emission formula
- [ ] PoI yield calculation
- [ ] Gate maturation logic
- [ ] Ihsan gate threshold
- [ ] Engram cache write path
- [ ] PatternMemory.learn() call path
- [ ] Reflex cache promotion logic
- [ ] Cross-language policy (no Python changes)

### 2.4 Regression Tests (mandatory, same file, same test module)

Three deterministic tests prove the precedence order. All three must pass before merge.

**Test A — Reflex wins:**
```
// Load a reflex rule for hash H.
// Call run_cycle with intent that produces hash H.
// Assert: receipt.path == CyclePath::ReflexHit
// (pattern recall block must not be reached)
```

**Test B — Pattern wins on reflex miss:**
```
// No reflex rule loaded for hash H.
// Pre-learn a pattern with embedding derived from intent bytes.
// Call run_cycle with that intent.
// Assert: receipt.path == CyclePath::PatternHit
// Assert: receipt.response contains learned content
// Assert: engram cache was NOT consulted (mock or counter)
```

**Test C — Falls through to Engram on pattern miss:**
```
// No reflex. Pattern memory empty (or intent embedding dissimilar enough).
// Engram cache has a matching entry.
// Call run_cycle.
// Assert: receipt.path == CyclePath::EngramHit
```

### 2.5 CyclePath Variant

Add `PatternHit` variant to the `CyclePath` enum in `omni_kernel.rs`.
No other enum changes.

### 2.6 Ihsan Score on Pattern Hit

Use the same `ihsan_score` computed from `level_scores` (already computed before
the Tier-1 block). Do not introduce a pattern-specific Ihsan override.

### 2.7 PoI on Pattern Hit

Pattern hit mints PoI identically to Reflex and Engram hits — call
`self.metabolic_ledger.mint_poi_yield(true, self.config.network_size, cycle.now_ms)`.

---

## 3. Expected Observables

After Wire 4, a live cycle trace must show:

```
DEBUG omni_kernel: Omni-Kernel: Tier-1b pattern recall hit  similarity=0.82 content="..."
```

No other new log lines. Existing Tier-1 and Tier-2 debug lines are unchanged.

---

## 4. Wire 4 Boundaries — Explicit Non-Goals

These are **out of scope** and must not appear in the Wire 4 commit diff:

- Auto-learning from cycle results into PatternMemory (Wire 5+)
- Gate maturation policy changes (Wire 5)
- Pattern confidence update (Wire 5+)
- Any new Cargo.toml dependencies (all dependencies added in Wire 3)
- Python-side changes
- Frontend changes
- SEED ledger changes

If any of these appear in the diff, the commit fails the boundary check.

---

## 5. Rollback Condition

If any of the following degrade after Wire 4 merges:

- `cargo test -p bizra-agent` drops below 202+3 (existing 202 + 3 new Wire 4 tests)
- Proof Pyramid Gate fails
- Walking Skeleton fails
- Any pre-existing test newly fails

→ Revert Wire 4 commit, re-open Wire 4 scope analysis.
The pattern recall block is purely additive — rollback cost is one `git revert`.

---

## 6. Embedding Strategy

`PatternMemory.recall(&[f32], limit)` requires an `f32` embedding slice.
The `intent_bytes: Vec<u8>` from `OmniCycle` must be projected to `Vec<f32>`.

Wire 4 uses the simplest stable projection:

```rust
let embedding: Vec<f32> = cycle
    .intent_bytes
    .iter()
    .map(|&b| b as f32 / 255.0)
    .collect();
```

This is deterministic, zero-dependency, and produces a normalized [0,1] vector.
It is not semantically meaningful for production recall — that is Wire 6+
(learned embeddings from a real encoder). Wire 4's job is to prove the
routing path works, not to optimize recall quality.

---

## 7. Commit Shape

One commit. Two files max:

- `bizra-omega/bizra-agent/src/omni_kernel.rs` — enum variant, config field, routing block, 3 tests
- `bizra-omega/bizra-autopoiesis/src/pattern_memory.rs` — only if a minor API fix is required

Commit message:
```
feat(agent): Wire 4 — Tier-1b pattern recall between reflex and engram

- Add CyclePath::PatternHit variant
- Add OmniConfig.pattern_recall_threshold (default 0.7)
- Insert pattern recall block after Tier-1 reflex miss
- 3 regression tests: reflex wins, pattern wins, falls through to engram
- No routing, economic, or maturation changes (Wire 4 boundary respected)

Tests: 205 GREEN (202 existing + 3 new)
```

---

## 8. Wire Sequence Context

| Wire | Scope | Status |
|:-----|:------|:-------|
| 2.5 | GatePolicy unification | SHIPPED `2eaaff30` |
| 3 | PatternMemory field in OmniKernel | SHIPPED `6aeac583` |
| **4** | **Tier-1b recall routing** | **THIS WIRE** |
| 5 | GateMaturationPolicy | AFTER 4 |
| 6+ | Learned embeddings, auto-promote | FUTURE |
