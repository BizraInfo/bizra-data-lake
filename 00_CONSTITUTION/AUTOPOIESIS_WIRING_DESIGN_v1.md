# Autopoiesis Wiring Design — v1

> Design gate for Wire 3–5. No code until this document is reviewed.
>
> Standing on Giants: Maturana & Varela (autopoiesis, 1972) · Hebb (synaptic learning, 1949) · Al-Ghazali (Ihsān gate maturation, 1095) · Boyd (OODA loop, 1976)

## Status

| Wire | Description | Status | Risk |
|------|-------------|--------|------|
| Wire 1 | Emission feedback (System1/System2 → was_cache_hit) | **SHIPPED** (2191e1ae) | — |
| Wire 2 | Ihsān env-gate (BIZRA_ENV=prod → Reject) | **SHIPPED** (2191e1ae) | — |
| Wire 3 | PatternMemory field in OmniKernel | **DESIGN** | Medium — new crate dependency |
| Wire 4 | Pattern recall insertion in cognitive loop | **DESIGN** | High — changes routing behavior |
| Wire 5 | GateMaturationPolicy (Observe → Reject auto-promotion) | **DESIGN** | High — changes governance state |

---

## Wire 3: PatternMemory Integration

### What
Add `bizra-autopoiesis` as a dependency of `bizra-agent` and add a `PatternMemory` field to the `OmniKernel` struct.

### Exact Insertion Point

**Cargo.toml** (`bizra-omega/bizra-agent/Cargo.toml`):
```toml
[dependencies]
bizra-autopoiesis = { path = "../bizra-autopoiesis" }
```

**OmniKernel struct** (`bizra-omega/bizra-agent/src/runtime.rs`):
```rust
pub struct OmniKernel {
    // ... existing fields ...
    pattern_memory: bizra_autopoiesis::PatternMemory,  // NEW
}
```

**Constructor** (`OmniKernel::new()`):
```rust
pattern_memory: bizra_autopoiesis::PatternMemory::new(),
```

### Estimated Size
~12 LOC (Cargo.toml + struct field + constructor line + import)

### Acceptance Criteria
- [ ] `cargo build -p bizra-agent` compiles with the new dependency
- [ ] `cargo test -p bizra-agent` passes (no behavioral change yet)
- [ ] `PatternMemory` field is accessible but unused (placeholder wiring)
- [ ] No circular dependency introduced (bizra-autopoiesis must NOT depend on bizra-agent)

### Rollback
Remove the Cargo.toml dependency line and struct field. Zero behavioral change.

---

## Wire 4: Pattern Recall in Cognitive Loop

### What
Insert a Tier-1b pattern recall step between the reflex cache check (System1) and the engram cache check, inside the OmniKernel cognitive cycle.

### Exact Insertion Point

**File**: `bizra-omega/bizra-agent/src/runtime.rs`

**Location**: Between the reflex cache hit (currently ~line 281) and the engram cache fallback (~line 283). The current flow is:

```
reflex_cache.try_hit(input)  →  if miss  →  engram_cache.try_hit(input)  →  if miss  →  full inference
```

After Wire 4:

```
reflex_cache.try_hit(input)  →  if miss  →  pattern_memory.recall(input)  →  if miss  →  engram_cache.try_hit(input)  →  if miss  →  full inference
```

### Recall Priority Ownership

| Tier | Cache | Latency | What it stores | Decision mode |
|------|-------|---------|----------------|---------------|
| 1a | Reflex cache | < 1ms | Exact content-hash matches | System1 |
| **1b** | **Pattern memory** | **< 5ms** | **Fuzzy structural patterns** | **System1** |
| 2 | Engram cache | < 10ms | Semantic similarity matches | System1 |
| 3 | Full inference | 50–500ms | Novel reasoning | System2 |

**Key design decision**: Pattern memory recall produces a `System1` decision mode (same as reflex). It is a learned pattern, not novel reasoning. This means pattern memory hits will decay SEED emission (Wire 1 feedback loop).

### Estimated Size
~20 LOC (recall call + match arm + optional logging)

### Acceptance Criteria
- [ ] Pattern recall is attempted after reflex miss, before engram
- [ ] Pattern recall miss falls through to engram (zero-impact on existing path)
- [ ] Pattern recall hit sets `CognitiveMode::System1`
- [ ] Latency regression: < 2ms added to the miss path (pattern lookup overhead)
- [ ] `cargo test -p bizra-agent` passes with no regressions

### Rollback Condition
If latency on the miss path exceeds 5ms overhead, or if determinism regresses (same input produces different outputs), revert to Wire 3 state (PatternMemory present but unused).

---

## Wire 5: Gate Maturation Policy

### What
Implement automatic promotion of `GatePolicy::Observe` → `GatePolicy::Reject` when the Ihsān gate has been stable for a configurable number of consecutive evaluations.

### Exact Insertion Point

**File**: `bizra-omega/bizra-hooks/src/ihsan_gate.rs`

**Location**: Inside `IhsanGate::evaluate()`, after the stability counter increment (~line 232):

```rust
// Current:
self.consecutive_stable += 1;
if self.consecutive_stable > self.max_consecutive_stable {
    self.max_consecutive_stable = self.consecutive_stable;
}

// After Wire 5:
self.consecutive_stable += 1;
if self.consecutive_stable > self.max_consecutive_stable {
    self.max_consecutive_stable = self.consecutive_stable;
}
if let Some(threshold) = self.config.maturation_threshold {
    if self.consecutive_stable >= threshold
        && self.config.policy == GatePolicy::Observe
    {
        self.config.policy = GatePolicy::Reject;
        // Emit maturation event if events enabled
    }
}
```

### Configuration Change

**GateConfig** gets a new field:
```rust
pub struct GateConfig {
    // ... existing fields ...
    /// After this many consecutive stable evaluations, auto-promote
    /// Observe → Reject. None = never auto-promote.
    pub maturation_threshold: Option<u64>,
}
```

- `production()`: `maturation_threshold: None` (already in Reject)
- `development()`: `maturation_threshold: None` (never auto-promote in dev)
- `staged()`: `maturation_threshold: Some(1000)` (promote after 1000 stable evals)

### Estimated Size
~30 LOC (struct field + constructor lines + evaluate logic + 2 tests)

### Acceptance Criteria
- [ ] Observe → Reject promotion happens at exactly the configured threshold
- [ ] No promotion when `maturation_threshold` is `None`
- [ ] No promotion when already in Reject or Throttle mode
- [ ] `consecutive_stable` counter resets on violation (already implemented)
- [ ] Test: 999 stable → still Observe, 1000th → Reject
- [ ] Test: violation at 999 resets counter, needs 1000 more

### Rollback Condition
If auto-promotion causes false rejections in staging environments, set `maturation_threshold: None` to disable. The field is purely additive — removing it reverts to current behavior.

---

## End-to-End Test Plan

### Integration test: Pattern Memory → SEED Emission Feedback

**Test name**: `test_pattern_memory_hit_decays_seed_emission`

**Setup**:
1. Create OmniKernel with PatternMemory and SeedLedger
2. Train pattern memory with 10 known patterns
3. Submit a mission matching a known pattern

**Assertions**:
- Pattern memory returns a hit (not reflex, not engram)
- CognitiveMode is System1
- was_cache_hit is true in seed settlement
- SEED emission multiplier is < 1.0 (decay applied)

### Integration test: Gate Maturation Under Load

**Test name**: `test_gate_matures_from_observe_to_reject`

**Setup**:
1. Create IhsanGate with `maturation_threshold: Some(100)`
2. Send 100 events above the constitutional floor

**Assertions**:
- After 99 events: policy is still Observe
- After 100th event: policy auto-promoted to Reject
- 101st event below floor: Rejected (not just observed)

### Regression: Latency Benchmark

**Test name**: `bench_cognitive_loop_with_pattern_memory`

**Setup**:
1. Run 1000 cognitive cycles with pattern memory enabled
2. Run 1000 cognitive cycles with pattern memory disabled (control)

**Assertions**:
- Mean latency delta < 2ms on miss path
- P99 latency delta < 5ms on miss path

---

## Implementation Order

```
Wire 3 alone  →  test  →  commit  →  push
Wire 4 alone  →  test  →  commit  →  push
Wire 5 alone  →  test  →  commit  →  push
Integration tests  →  commit  →  push
```

Each wire is independently revertable. Never combine wires in the same commit.

## Open Questions

1. **Should pattern memory hits count as cache hits for emission decay?** Current answer: YES (they are learned patterns, System1). Revisit if this over-penalizes exploration nodes that have broad pattern libraries.

2. **Should gate maturation be reversible?** Current answer: NO (one-way ratchet). A node that has proven stability should not regress to Observe. If this causes issues, add a `dematuration_threshold` later.

3. **What is the right maturation threshold for production?** Current answer: TBD. 1000 is a placeholder. Need telemetry data from staging to calibrate.
