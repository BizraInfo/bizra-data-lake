# Phase 48 Spec — Part 2: Architecture Decision — Keep or Merge?

> Standing on Giants: Conway (organizational mirroring) · Fowler (microservices boundaries) · Lamport (composition vs monolith)

## The Question

We have two Rust workspaces:
- `bizra-omega/` — 14 crates, 40,570 lines, platform layer (PCI, federation, inference, API)
- `native/` — 4 crates, 8,768 lines, Node0 cognitive layer (hooks, memory, FATE, IPC)

Should they be one workspace or two?

---

## Option A: Keep Separate (Recommended)

### Rationale

1. **Different deployment targets**
   - `bizra-omega/` produces server binaries (API, resource pool, CLI)
   - `native/` produces libraries consumed by Python (via FFI) and future desktop agent (Tauri)

2. **Different dependency profiles**
   - `bizra-omega/` pulls tokio, reqwest, ed25519-dalek, rayon — heavy async + crypto
   - `native/` pulls napi, iceoryx2, z3 — FFI bridges + formal verification
   - Merging would force every crate to compile both dependency trees

3. **Different release cadences**
   - `native/` is at v2.2.0, `bizra-omega/` is at v1.0.0
   - They evolve independently

4. **Compilation time**
   - Combined workspace: every `cargo test` rebuilds everything
   - Separate: incremental builds stay fast (~5s native, ~33s omega)

5. **Conway's Law alignment**
   - `bizra-omega/` = the decentralized platform (nodes, federation, pools)
   - `native/` = the individual node's cognitive system (memory, hooks, FATE)

### What We Do Instead of Merging

Create a **shared protocol crate** that both workspaces depend on:

```
bizra-protocol/           # Shared types — no dependencies
├── Cargo.toml            # version = "0.1.0", no deps beyond serde
├── src/
│   ├── lib.rs
│   ├── ihsan.rs          # IhsanScore (single canonical type)
│   ├── snr.rs            # SNR thresholds + scoring contract
│   ├── thresholds.rs     # Constitutional thresholds (Rust mirror of constants.py)
│   └── atom.rs           # AtomKind, MemoryAtom (shared vocabulary)
```

Both workspaces reference it via path:
```toml
# In native/Cargo.toml
bizra-protocol = { path = "../bizra-protocol" }

# In bizra-omega/Cargo.toml
bizra-protocol = { path = "../bizra-protocol" }
```

This resolves the **IhsanScore type duplication** (f64 in omega vs f32 in native) with a single canonical definition.

---

## Option B: Merge Into One Workspace

### Would require
- Unified `Cargo.toml` with 18 members
- Resolving dependency conflicts (napi vs tokio-full, Z3 vs rayon)
- Single version for all crates
- Combined CI (slower, more breakage surface)

### Rejected because
- No crate in `native/` depends on any crate in `bizra-omega/` or vice versa
- The compilation cost is real (~2 min combined vs ~40s separate)
- Different teams/phases can work in parallel without blocking

---

## Decision: Option A — Keep Separate + Shared Protocol Crate

### Action Items

1. Create `bizra-protocol/` at repo root with canonical shared types
2. Migrate `IhsanScore` from both workspaces to use `bizra-protocol::IhsanScore`
3. Add `bizra-protocol` to both workspace `Cargo.toml` members
4. Ensure `constants.py` thresholds match `bizra-protocol::thresholds`
5. Add cross-workspace smoke test in CI

---

## TDD Anchors

```rust
// bizra-protocol/src/ihsan.rs
#[test]
fn ihsan_score_clamps_to_unit_interval() {
    assert_eq!(IhsanScore::new(1.5), IhsanScore::new(1.0));
    assert_eq!(IhsanScore::new(-0.1), IhsanScore::new(0.0));
}

#[test]
fn ihsan_threshold_matches_python_constants() {
    assert_eq!(UNIFIED_IHSAN_THRESHOLD.as_f64(), 0.95);
    assert_eq!(STRICT_IHSAN_THRESHOLD.as_f64(), 0.99);
}

#[test]
fn snr_threshold_matches_python_constants() {
    assert_eq!(UNIFIED_SNR_THRESHOLD, 0.85);
    assert_eq!(SNR_THRESHOLD_T1_HIGH, 0.95);
    assert_eq!(SNR_THRESHOLD_T0_ELITE, 0.98);
}
```
