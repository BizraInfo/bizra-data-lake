# Phase 48 Spec — Part 4: Bridge Strategy (Python ↔ Rust)

> Standing on Giants: PyO3 project (Rust-Python FFI) · Shannon (information channels) · Lamport (interface contracts)

## The Bridge Problem

Three codebases need to communicate:
1. Python `core/` — the living application (171K lines, 7K tests)
2. Rust `bizra-omega/` — the platform layer (40K lines, 501 tests)
3. Rust `native/` — the Node0 cognitive layer (8.7K lines, 109 tests)

Currently:
- `bizra-omega/bizra-python/` has PyO3 bindings (924 lines) — bridges omega ↔ Python
- `native/bizra-memory/src/bridge.rs` has FFI trait definitions — bridges native ↔ Python (traits only, no impl)
- No bridge exists between omega and native

---

## Bridge Map

```
                    ┌──────────────────┐
                    │   Python core/   │
                    │   (171K lines)   │
                    └────┬────────┬────┘
                         │        │
                    PyO3 │        │ FFI (planned)
                         │        │
              ┌──────────▼──┐  ┌──▼──────────────┐
              │ bizra-omega  │  │    native/       │
              │ (40K lines)  │  │  (8.7K lines)    │
              │              │  │                   │
              │ bizra-python │  │ bizra-memory      │
              │   (PyO3)     │  │   bridge.rs       │
              └──────────────┘  └───────────────────┘
                    ↕ (none)
```

---

## Strategy: Three Bridges

### Bridge 1: `bizra-python` (EXISTS — omega → Python)

**Location:** `bizra-omega/bizra-python/src/lib.rs`
**Technology:** PyO3 + maturin
**Status:** Operational — provides `import bizra` in Python
**Exposes:** core identity, PCI envelopes, inference, federation, autopoiesis

**No changes needed.** This bridge is mature and tested.

### Bridge 2: `native-python-bridge` (PLANNED — native → Python)

**Location:** `native/bizra-memory/src/bridge.rs` (traits defined)
**Technology:** PyO3 (matching omega's approach for consistency)
**Status:** Trait interfaces exist, no implementation

**What needs to happen:**
1. Add `pyo3` dependency to `native/bizra-memory/Cargo.toml` behind a feature flag
2. Implement `#[pyclass]` wrappers for `BizraMemory`, `TurnResult`, `MemoryHealth`
3. Build with `maturin develop --features python`
4. Python `core/living_memory/` calls native memory via `import bizra_memory`

**Pseudocode:**

```
// native/bizra-memory/src/python.rs (behind #[cfg(feature = "python")])

#[pyclass]
struct PyBizraMemory {
    inner: BizraMemory,
}

#[pymethods]
impl PyBizraMemory {
    #[new]
    fn new() -> Self { PyBizraMemory { inner: BizraMemory::new() } }

    fn process_turn(&mut self, content: &str, session: u64, turn: u32, ts: u64) -> PyResult<PyTurnResult> {
        Ok(self.inner.process_user_turn(content, session, turn, ts).into())
    }

    fn what_do_i_know(&mut self, now: u64) -> Vec<(String, f32)> {
        self.inner.what_do_i_know(now).iter()
            .map(|(s, c)| (s.to_string(), *c))
            .collect()
    }

    fn health(&self) -> PyResult<PyMemoryHealth> {
        Ok(self.inner.health().into())
    }
}
```

### Bridge 3: `bizra-protocol` (NEW — shared types between omega and native)

**Location:** repo root `bizra-protocol/`
**Technology:** Pure Rust crate, no external deps (only serde)
**Status:** Does not exist yet

**Purpose:** Canonical type definitions that both workspaces import, eliminating the `IhsanScore` duplication (f64 in omega vs f32 in native).

**This is NOT a runtime bridge** — it's a compile-time shared vocabulary. No IPC, no FFI, no network calls.

```
bizra-protocol/src/
├── lib.rs            # Re-exports
├── ihsan.rs          # IhsanScore: f64, clamped [0.0, 1.0]
├── snr.rs            # SNR thresholds matching constants.py
├── thresholds.rs     # All constitutional thresholds
├── atom.rs           # AtomKind enum (shared vocabulary)
└── event.rs          # Event topic vocabulary (shared naming)
```

---

## Priority Order

| Priority | Bridge | Effort | Impact |
|----------|--------|--------|--------|
| 1 | Build `bizra-agent` + `bizra-node` crates | 2-3 days | Unlocks Node0 binary |
| 2 | `bizra-protocol` shared types | 1 day | Eliminates type duplication |
| 3 | `native-python-bridge` (PyO3) | 1-2 days | Python can call native memory |
| 4 | Cross-workspace CI smoke test | 0.5 days | Catches integration breaks |

---

## What We Are NOT Doing

- **NOT merging workspaces** — compilation cost and deployment targets differ
- **NOT building IPC between omega and native at this phase** — the iceoryx-bridge exists for future use but Node0 doesn't need inter-process Rust↔Rust communication yet
- **NOT replacing Python core/ with Rust** — Python is the living application; Rust accelerates the hot paths (memory synthesis, FATE gates, IPC)

---

## TDD Anchors

```python
# tests/integration/test_native_bridge.py

def test_native_memory_roundtrip():
    """Python can call bizra_memory via PyO3 and get results."""
    import bizra_memory
    mem = bizra_memory.PyBizraMemory()
    result = mem.process_turn("I am Mumo", 1, 1, 1000)
    assert result.ingested is True
    facts = mem.what_do_i_know(2000)
    assert len(facts) > 0

def test_native_memory_health():
    """Health endpoint returns valid struct."""
    import bizra_memory
    mem = bizra_memory.PyBizraMemory()
    mem.process_turn("test", 1, 1, 1000)
    health = mem.health()
    assert health.active is True
    assert health.turns_processed == 1
```

```rust
// bizra-protocol/src/ihsan.rs
#[test]
fn ihsan_score_used_by_both_workspaces() {
    // This test exists in bizra-protocol to prove the type compiles
    let score = IhsanScore::new(0.95);
    assert!(score.meets_threshold(IhsanScore::new(0.95)));
    assert!(!score.meets_threshold(IhsanScore::new(0.96)));
}
```
