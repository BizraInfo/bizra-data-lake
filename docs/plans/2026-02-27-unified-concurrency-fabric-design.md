# Unified Concurrency Fabric (UCF) Design

**Date:** 2026-02-27
**Status:** Approved
**Scope:** P0 Trinity — 3 interdependent changes that unlock concurrency across the BIZRA nervous system

## Problem Statement

The BIZRA topology analysis identified three critical bottlenecks:

1. **Single-threaded EventBus** — 12 subscribers process sequentially on the caller's thread. Every emit scans all 512 handler slots.
2. **Mutex-locked OmniKernel** — Tier-1/Tier-2 cache lookups (read-only, O(1)) are serialized behind the same Mutex as Tier-3 inference (mutable, GPU-bound).
3. **Disconnected Rust/Python event systems** — Two isolated event buses with no bridge. Python orchestration cannot react to Rust-side events.

## Design

### 1. Namespace-Sharded EventBus

**File:** `bizra-omega/bizra-hooks/src/event_bus.rs`

Replace the flat `[Option<HandlerEntry>; 512]` array with 8 shards + 1 global shard. Topic prefix before the first `.` is hashed via FNV-1a to select the shard. Wildcards that span namespaces route to the global shard.

```
NUM_SHARDS = 8 (power of 2)
SHARD_CAPACITY = 64 (8 * 64 = 512 total)
```

Topic namespace → shard mapping (natural from existing subscriber wiring):
- `action.*` → shard 0
- `memory.*` → shard 1
- `telescript.*` → shard 2
- `session.*` → shard 3
- `system.*` → shard 4
- `ihsan.*` → shard 5
- `poi.*` → shard 6
- (overflow) → shard 7

**Expected improvement:** O(N) → O(N/K) per emit, where K = shard count. 3-5x throughput on broadcast events.

**Constraints preserved:**
- Zero external dependencies
- All types remain Copy + Clone
- No heap allocation
- no_std compatible
- Backward-compatible API (type alias preserves call sites)

### 2. RwLock-Split OmniKernel

**File:** `bizra-omega/bizra-agent/src/omni_kernel.rs`

Split the OmniKernel into a read layer (caches) and a write layer (ledger/TTRL):

```
OmniKernelReader {config, reflex_mode, policy_hash, reflex_cache, engram_cache}
OmniKernelWriter {ttrl_engine, metabolic_ledger}
```

`run_cycle()` attempts the read path first. Only on cache miss does it acquire the writer. For >80% cache hit rate (expected after warmup), concurrent cycles serve without contention.

**Expected improvement:** Near-linear concurrency for cache-hit workloads. Tier-3 misses still serialize (correct behavior — TTRL/metabolic state is mutable).

**API preservation:** `OmniKernel::run_cycle()` keeps its exact signature. The split is internal.

### 3. PyO3 Event Bridge

**File:** `bizra-omega/bizra-python/src/lib.rs` + `core/sovereign/event_bus.py`

Add `PyEventBridge` class exposed to Python:
- `emit(topic, payload, priority)` → forwards into Rust BizraSystem
- `wire_subscribers()` → wires all 12 subscribers
- `health()` → returns system health snapshot

V1 direction: Python → Rust only (synchronous PyO3 GIL calls). Rust → Python notification deferred to future iceoryx-based ring buffer.

Python side: `RustBridge` class in `core/sovereign/event_bus.py` wraps the PyO3 bindings.

### 4. Test Strategy

- **Sharded EventBus:** Shard isolation tests + throughput benchmark (50+ subscribers, 7 namespaces)
- **OmniKernel split:** Concurrent cache-hit test + Tier-3 miss lock acquisition test
- **PyO3 Bridge:** Integration test (create bridge, wire, emit, verify delivery)
- **Regression:** All 612+ existing tests must pass with zero modification

## Approach Selection

| Approach | Description | Chosen? |
|----------|-------------|---------|
| A: Namespace-Sharded | Sharded bus + RwLock kernel + PyO3 callbacks | Yes |
| B: Channel-Based Async | tokio broadcast channels + async handlers | No (breaks zero-dep/no_std) |
| C: Lock-Free Ring | SPMC ring buffer + atomic CAS | No (complexity disproportionate to 12 subscribers) |

## Standing on Giants

- **Shannon (1948):** Information-theoretic routing — events carry only to relevant shards (maximum SNR)
- **Lamport (1978):** Happened-before ordering preserved within each shard via monotonic EventId
- **Hoare (1978):** Monitor-based concurrency (RwLock) with minimal critical sections
- **Al-Ghazali (1095):** Ihsan gate remains the non-negotiable pre-emit filter, now per-shard
- **Boyd (1976):** OODA loop latency reduced by concurrent cache-hit cycles
