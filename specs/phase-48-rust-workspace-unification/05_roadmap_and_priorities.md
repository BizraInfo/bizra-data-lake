# Phase 48 Spec — Part 5: Roadmap and Priorities

> Standing on Giants: Boyd (OODA — observe before acting) · Deming (PDCA — quality gates) · Brooks (no silver bullet — incremental delivery)

## Current State (Post Phase 47.1)

### What's Working

| Component | Tests | Status |
|-----------|-------|--------|
| Python `core/` + rollout infrastructure | 7,278 | Active, CanaryRouter + Metrics wired |
| Phase 46 Cognitive Resonance (search, GoT, HMM) | 299 subset | Wired via canary routing |
| Rust `bizra-omega/` (14 crates) | 501 | All pass |
| Rust `native/` (4 crates) | 109 | All pass (fixed this session) |
| CI pipelines (Python + Native + Omega) | - | Green |

### What's Missing

| Gap | Impact | Priority |
|-----|--------|----------|
| `bizra-agent` crate | No agent runtime in Rust | P1 |
| `bizra-node` crate | No Node0 binary | P1 |
| `bizra-protocol` shared types | IhsanScore type duplication | P2 |
| Native → Python bridge (PyO3) | Can't call Rust memory from Python | P2 |
| Phase 47.1 Stage 0 rollback drills | Canary safety not validated | P1 |
| Cross-workspace CI smoke | No integration test across workspaces | P3 |

---

## Recommended Execution Order

### Sprint 1: Safety First (Phase 47.1 completion)

**Goal:** Validate that the canary infrastructure actually works before building more.

1. Run Stage 0 rollback drills (synthetic fault injection)
2. Verify kill switch precedence (already fixed this session)
3. Validate metrics flow to status endpoint
4. Begin canary ramp: Search 10% for 4h

**Estimated effort:** 1 day
**No new code** — just validation of what we built.

### Sprint 2: Agent Runtime (`bizra-agent`)

**Goal:** Build the event-driven agent runtime in `native/`.

1. Create `native/bizra-agent/` with module structure from spec 03
2. Implement `BizraAgent` facade
3. Implement event loop (hooks → dispatch → report)
4. Implement capability matching + tool registry
5. Write ~25 tests
6. Integrate with `bizra-hooks` and `bizra-memory`

**Estimated effort:** 2-3 days
**Dependencies:** bizra-hooks (done), bizra-memory (done)

### Sprint 3: Node Binary (`bizra-node`)

**Goal:** Build the Node0 binary that bootstraps the full native stack.

1. Create `native/bizra-node/` with module structure from spec 03
2. Implement bootstrap sequence (hooks → memory → agent → FATE)
3. Implement state persistence + FATE-signed continuity
4. Implement config (CLI args, env vars)
5. Write ~20 tests
6. Verify `the_four_word_test` through full stack

**Estimated effort:** 2-3 days
**Dependencies:** bizra-agent (Sprint 2)

### Sprint 4: Protocol + Bridge

**Goal:** Eliminate type duplication and enable Python → Rust calls.

1. Create `bizra-protocol/` shared types crate
2. Migrate IhsanScore, SNR thresholds to shared crate
3. Add PyO3 feature flag to `bizra-memory`
4. Implement `PyBizraMemory` wrapper
5. Write Python integration tests

**Estimated effort:** 2 days
**Dependencies:** None (can run parallel with Sprint 2-3)

---

## What NOT To Do Yet

| Temptation | Why Not |
|------------|---------|
| Build Tauri desktop UI | No stable agent runtime to wrap yet |
| Sacred geometry dashboard | Cosmetic — no production value yet |
| Merge Rust workspaces | Compilation cost, different targets |
| Replace Python `core/` with Rust | Python is working; Rust accelerates hot paths |
| Add new Python modules | 420 files is enough; optimize, don't expand |
| Production canary ramp beyond Stage 2 | Need Stage 0 drills to pass first |

---

## Success Metrics

After Phase 48 completion:

| Metric | Target |
|--------|--------|
| `native/` crate count | 6 (currently 4) |
| `native/` total tests | ~155+ (currently 109) |
| `native/` total lines | ~13,000+ (currently 8,768) |
| Node0 binary boots and processes conversation | Yes |
| State persistence survives restart | Yes |
| FATE gate rejects tampered state | Yes |
| Python can call `bizra_memory` via PyO3 | Yes |
| IhsanScore type duplication | Eliminated |
| Phase 47.1 Stage 0 drills | Passed |
| All CI pipelines | Green |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Z3 compile issues on CI | Medium | Blocks native-ci | z3-sys already works; pin version |
| PyO3 + maturin build complexity | Low | Blocks bridge | bizra-omega already does this successfully |
| iceoryx2 breaking changes | Low | Blocks IPC | Pin to 0.4.1, don't upgrade |
| Phase 47.1 rollback drills fail | Medium | Blocks canary ramp | Fix rollback engine before ramp |
| Scope creep (Tauri, dashboard, etc.) | High | Delays core delivery | Strict sprint boundaries |
