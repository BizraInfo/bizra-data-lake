# bizra-hooks v0.1.0

## Node0 Nervous System — The RSI Pillar I Implementation

**Zero dependencies. Pure Rust. Sovereign.**

```
┌─────────────────────────────────────────────────────────────────┐
│                     Node0Kernel                                  │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐  │
│  │ Registry │   │ EventBus │   │ HookChain│   │ EventStore  │  │
│  │          │──▶│          │──▶│          │   │             │  │
│  │Self-Model│   │Pub/Sub   │   │ Before/  │   │Append-Only  │  │
│  │RSI Pillar│   │28 event  │   │ After/   │   │100K events  │  │
│  │   I      │   │types     │   │ Error    │   │Queryable    │  │
│  └──────────┘   └──────────┘   └──────────┘   └─────────────┘  │
│       │              │              │                │           │
│       ▼              ▼              ▼                ▼           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              IhsanScore (Lyapunov Function)              │   │
│  │         SNR × Confidence ≥ 0.99 = إحسان threshold        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              FFI (C-ABI) — 12 exported symbols            │   │
│  │         Python ctypes ← Rust ← Zero dependencies         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Metrics

| Metric | Value |
|---|---|
| Lines of Rust | 2,966 |
| Lines of Python | 413 |
| Tests | 35 (34 unit/integration + 1 doctest) |
| External dependencies | 0 |
| Release binary (rlib) | 550 KB |
| Shared library (cdylib) | 373 KB |
| Event types | 28 |
| FFI exports | 12 C-ABI symbols |
| Build time (release) | 1.96s |
| Test execution time | 0.01s |

## RSI Framework Mapping

| RSI Pillar | Implementation | File |
|---|---|---|
| Pillar I: Self-Model | `Registry` + `ArchitectureGraph` | `registry.rs` |
| Pillar II: Prediction | `IhsanScore` + `EventStore.average_ihsan()` | `types.rs`, `store.rs` |
| Pillar III: Verification | `HookChain.before()` abort gates | `hook.rs` |
| Pillar IV: Safe Deploy | `HookAction::Abort` + `HookPriority::SYSTEM` | `hook.rs`, `types.rs` |
| Pillar V: Stable Iteration | `IhsanScore.passes()` (Lyapunov ≥ 0.99) | `types.rs` |

## Files

```
bizra-hooks/
├── Cargo.toml              # Zero deps. Features: ffi, metrics
├── src/
│   ├── lib.rs              # Public API + Node0Kernel + integration tests
│   ├── types.rs            # EventId, ComponentId, Event, IhsanScore, etc.
│   ├── registry.rs         # Component Registry + ArchitectureGraph
│   ├── bus.rs              # EventBus — typed pub/sub
│   ├── hook.rs             # HookChain — before/after/error interceptors
│   ├── store.rs            # EventStore — bounded append-only log
│   └── ffi.rs              # C-ABI exports (--features ffi)
├── python/
│   └── bizra_hooks.py      # Python ctypes wrapper
└── README.md
```

## Build

```bash
# Library (no FFI)
cargo build --release

# Library + shared library for Python
cargo build --release --features ffi

# Run tests
cargo test
```

## Python Integration

```python
from bizra_hooks import Kernel, ComponentKind, EventKind

kernel = Kernel.boot()

# Register Phase 46 engines
from bizra_hooks import register_phase46_engines
ids = register_phase46_engines(kernel)

# Publish with إحسان score
kernel.publish_scored(
    EventKind.RESONANCE_COMPLETE,
    ids["cognitive_resonance"],
    "pipeline output",
    snr=0.97, confidence=0.95, latency_us=1500
)

# Query self-model
arch = kernel.architecture()
print(f"Nodes: {len(arch['nodes'])}, Edges: {len(arch['edges'])}")
assert not kernel.has_cycles
```

## What This Enables

Everything that follows plugs into this kernel:

- **Memory Synthesis Pipeline** → subscribes to UserMessage, publishes MemorySynthesize
- **Desktop Agent** → subscribes to TaskStart, publishes DesktopAction
- **MCP Client** → subscribes to McpToolCall, publishes ApiCall
- **FATE Engine** → registers as before-hook with SYSTEM priority, aborts unsafe mutations
- **Onboarding UI** → subscribes to ComponentRegistered, renders architecture graph

## Standing on Giants

Gamma et al. (Observer, 1994) · Hewitt (Actors, 1973) · Hoare (Contracts, 1969) · Milner (Types, 1978) · Lamport (Ordering, 1978) · Kreps (Log, 2013) · Shannon (Information, 1948) · Lyapunov (Stability, 1892) · RSI Ultimate Framework (Z.ai, 2024)
