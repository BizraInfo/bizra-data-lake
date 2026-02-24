# Phase 49 Spec — Part 4: Remaining Rust Crates

> Standing on Giants: Brooks (mythical man-month — build only what's needed) · Boyd (OODA — don't build before you orient)

## Open Architectural Gaps from Phase 48 Spec

Three Rust crates were planned but not yet built. This spec evaluates whether they're still needed and what the minimal viable scope is.

### 1. `bizra-protocol` — Shared Types (P2)

**Problem:** `IhsanScore` type is defined in both:
- `bizra-hooks/src/types.rs` as `pub type IhsanScore = f32;`
- `bizra-core/src/constitution.rs` as a field in `IhsanThreshold`

**Assessment:** This is type aliasing, not structural duplication. Both are `f32`. The cost of creating a shared crate (compile time, dependency graph complexity) outweighs the benefit of eliminating a type alias.

**Decision:** DEFER. Keep the type alias in both crates. If a third crate needs `IhsanScore`, extract then. YAGNI applies.

```pseudocode
# IF we do create it later:
# bizra-protocol/src/lib.rs
pub type IhsanScore = f32;
pub type SNRScore = f64;
pub const IHSAN_THRESHOLD: f64 = 0.95;
pub const SNR_THRESHOLD: f64 = 0.85;

# Both bizra-hooks and bizra-core would dep on bizra-protocol
# Only worthwhile when 3+ crates share the same types
```

### 2. `bizra-agent` — Event-Driven Agent Runtime (P1)

**Problem:** There's no Rust-native agent runtime. Agents are currently Python-only (core/sovereign/, PAT/SAT system).

**Assessment:** The PyO3 bridge (Phase 48.1) already exposes `PyBizraMemory` and `PyThoughtGraph` to Python. Building a full Rust agent runtime duplicates the Python agent system without clear performance benefit — agents are I/O-bound (LLM calls, file operations), not compute-bound.

**Decision:** DEFER until there's a concrete use case requiring Rust agent performance. The PyO3 bridge satisfies the "Python calls Rust hot paths" need.

**When to revisit:** When agent dispatch latency becomes a measurable bottleneck (> 10ms per dispatch) or when edge deployment requires a single binary without Python.

```pseudocode
# IF we build it:
# bizra-omega/bizra-agent/src/lib.rs

struct BizraAgent {
    identity: AgentIdentityBlock,     # from bizra-core::pat
    memory: BizraMemory,              # from bizra-memory
    hooks: HookPipeline,              # from bizra-hooks
    capabilities: Vec<AgentCapability>,
}

impl BizraAgent {
    async fn dispatch(&mut self, event: AgentEvent) -> AgentResult {
        // Route event through hooks pipeline
        // Execute capability matching
        // Record to memory
    }
}

# Estimated: ~500 LOC, 25 tests
# Dependencies: bizra-core, bizra-hooks, bizra-memory
```

### 3. `bizra-node` — Node0 Binary (P1)

**Problem:** No single binary that bootstraps the full native stack.

**Assessment:** The Python `scripts/node0_activate.py` handles Node0 lifecycle. A Rust binary would provide faster startup and single-file deployment, but the current Python path works. The main value would be for edge deployment where Python isn't available.

**Decision:** DEFER until edge deployment is a concrete requirement. Python startup is ~200ms which is acceptable for a long-running daemon.

**When to revisit:** When deploying to environments without Python (embedded, Tauri desktop app, mobile).

```pseudocode
# IF we build it:
# bizra-omega/bizra-node/src/main.rs

#[tokio::main]
async fn main() {
    // 1. Load config (CLI args + env vars + config file)
    // 2. Initialize hooks pipeline
    // 3. Initialize memory system
    // 4. Initialize FATE binding (state persistence)
    // 5. Start agent runtime (if bizra-agent exists)
    // 6. Start MCP server (if configured)
    // 7. Enter event loop

    // Graceful shutdown on SIGTERM
}

# Estimated: ~300 LOC, 20 tests
# Dependencies: bizra-core, bizra-hooks, bizra-memory, fate-binding, bizra-agent
```

## Summary Decision Matrix

| Crate | Phase 48 Priority | Current Decision | Reason |
|-------|-------------------|------------------|--------|
| `bizra-protocol` | P2 | **DEFER** | Type duplication is trivial (f32 alias) |
| `bizra-agent` | P1 | **DEFER** | PyO3 bridge covers the hot path; agents are I/O-bound |
| `bizra-node` | P1 | **DEFER** | Python node0_activate.py works; no edge deployment need yet |

## What To Do Instead

The highest-leverage Rust work is:

1. **Ensure all 610 tests stay green** (CI enforcement)
2. **Delete `native/` duplicate** (spec 02 — reduce confusion)
3. **Expose PyHooksPipeline via PyO3** (small incremental — hooks are the only unwrapped dep)
4. **Run benchmarks** (verify SIMD, batch verification, parallel BLAKE3 actually deliver claimed speedups)

These are refinement actions, not new feature builds. Phase 49 is about consolidation.
