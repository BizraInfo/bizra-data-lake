# Phase 09 — Build Plan & Implementation Order

> **Purpose:** What to build, in what order, to ship the sovereign terminal.
> **Locked:** 2026-03-07

## 9.1 Current State Summary

| Metric | Value |
|--------|-------|
| Total existing LOC | 7,471 |
| Rust TUI | 3,210 LOC (6 tabs, 4 widgets, inference client) |
| Python REPL | 3,982 LOC (30+ commands, Rich output, MCP) |
| Spec files | 2,635 LOC (schemas, IA, mission loop, economic.ts, bloom.py, subscribers.py) |
| Terminal views specced | 7 |
| Terminal views built | 2 partial (Briefing stub, System stub) |
| EventBus subscribers | 12 defined, 0 wired to terminal |
| Event types | 40 defined in schema |
| Action types | 15 defined with tier/gate requirements |
| API endpoints available | 75 (sovereign API) |

## 9.2 Build Phases (Priority Order)

### Sprint 1: Foundation (P0) — ~600 LOC

**Goal:** All 7 views render with data (live or offline).

| Task | Surface | LOC | Spec |
|------|---------|-----|------|
| Rust: Add Briefing tab (View 1) | Rust | 120 | Phase 02 |
| Rust: Add Wallet widget (View 4) | Rust | 200 | Phase 05 |
| Rust: Add Sovereignty gauge | Rust | 80 | Phase 06 |
| Rust: Update tab nav (6 → 7 tabs) | Rust | 40 | Phase 01 |
| Python: Wire mission to /v1/plan | Python | 60 | Phase 03 |
| Python: Upgrade wallet() with factors | Python | 60 | Phase 05 |
| Both: Offline fallback for all views | Both | 40 | Phase 01 |

### Sprint 2: Proof Chain (P0) — ~400 LOC

**Goal:** Receipts view operational, evidence chain visible.

| Task | Surface | LOC | Spec |
|------|---------|-----|------|
| Rust: Receipt list widget | Rust | 150 | Phase 04 |
| Rust: Receipt detail (8-dim Ihsan) | Rust | 100 | Phase 04 |
| Both: Chain verification display | Both | 40 | Phase 04 |
| Rust: Mission result receipt render | Rust | 100 | Phase 03 |
| Both: Filter (all/qualified/rejected) | Both | 30 | Phase 04 |

### Sprint 3: EventBus Integration (P0) — ~300 LOC

**Goal:** Terminal receives live events from bus.

| Task | Surface | LOC | Spec |
|------|---------|-----|------|
| Python: TerminalSubscriber class | Python | 60 | Phase 08 |
| Rust: Event mpsc channel | Rust | 80 | Phase 08 |
| Both: Event-to-view routing | Both | 40 | Phase 08 |
| Both: Live mission stream | Both | 80 | Phase 03 |
| Both: Wallet live update on mint | Both | 40 | Phase 05 |

### Sprint 4: Memory & Skills (P1) — ~500 LOC

**Goal:** Full MMORPG progression visible.

| Task | Surface | LOC | Spec |
|------|---------|-----|------|
| Rust: Memory profile widget | Rust | 100 | Phase 06 |
| Rust: Skill tree grid | Rust | 120 | Phase 06 |
| Rust: Lifecycle progress bar | Rust | 60 | Phase 06 |
| Both: Compiled reflexes table | Both | 80 | Phase 06 |
| Both: Near-compilation display | Both | 40 | Phase 06 |
| Both: Recent missions list | Both | 60 | Phase 06 |
| Both: Stats (streak, preferred agent) | Both | 40 | Phase 06 |

### Sprint 5: System & Hardening (P1) — ~350 LOC

**Goal:** Full system health visible, constitutional alerts.

| Task | Surface | LOC | Spec |
|------|---------|-----|------|
| Both: Service health table | Both | 80 | Phase 07 |
| Both: Security posture panel | Both | 60 | Phase 07 |
| Both: Constitutional violation alert | Both | 40 | Phase 07 |
| Both: K8s/Docker status | Python | 40 | Phase 07 |
| Both: WebSocket status indicator | Both | 20 | Phase 07 |
| Both: Status bar event flash | Both | 30 | Phase 08 |
| Both: Export (JSON/CSV) | Python | 50 | Phase 04 |
| Both: Interrupt confirmation dialog | Both | 50 | Phase 03 |

## 9.3 Estimated Totals

| Sprint | LOC | Priority | Milestone |
|--------|-----|----------|-----------|
| 1: Foundation | ~600 | P0 | All 7 views render |
| 2: Proof Chain | ~400 | P0 | Receipts operational |
| 3: EventBus | ~300 | P0 | Live events flowing |
| 4: Memory & Skills | ~500 | P1 | MMORPG progression |
| 5: System & Hardening | ~350 | P1 | Full health + alerts |
| **TOTAL** | **~2,150** | | Complete terminal |

**Post-implementation total:** 7,471 (existing) + 2,150 (new) = **~9,621 LOC**

## 9.4 Testing Plan

| Test Category | Count | Framework |
|---------------|-------|-----------|
| Rust unit tests | ~30 | `cargo test` |
| Python unit tests | ~25 | `pytest` |
| Integration tests | ~10 | Both |
| Snapshot tests (UI) | ~7 | ratatui `TestBackend` |
| Property tests | ~5 | `hypothesis` / `proptest` |
| **Total** | **~77** | |

## 9.5 Files Modified (Existing) vs Created (New)

### Modified

| File | Change |
|------|--------|
| `bizra-cli/src/app.rs` | Add 7-tab navigation, new view states |
| `bizra-cli/src/main.rs` | Add event channel, new render branches |
| `bizra-cli/src/widgets/header.rs` | Update to 7 tabs |
| `bizra-cli/src/widgets/status_bar.rs` | Add wallet summary + event flash |
| `core/sovereign/__main__.py` | Wire receipts, skills, memory commands |
| `terminal/sovereign_terminal.py` | Upgrade wallet(), add receipts, skills |

### Created

| File | Purpose |
|------|---------|
| `bizra-cli/src/widgets/wallet.rs` | Wallet widget (balances, factors, gauge) |
| `bizra-cli/src/widgets/sovereignty.rs` | Sovereignty gauge + lifecycle bar |
| `bizra-cli/src/widgets/receipt_list.rs` | Receipt list (scrollable, filterable) |
| `bizra-cli/src/widgets/receipt_detail.rs` | Receipt detail (8-dim Ihsan) |
| `bizra-cli/src/widgets/briefing.rs` | Briefing panel (JARVIS greeting) |
| `bizra-cli/src/widgets/memory.rs` | Memory profile + recent missions |
| `bizra-cli/src/widgets/skill_tree.rs` | Skill tree grid + reflexes |
| `bizra-cli/src/widgets/system.rs` | System health table + security |
| `bizra-cli/src/api_client.rs` | HTTP client for sovereign API |
| `core/terminal/subscriber.py` | TerminalSubscriber for EventBus |

## 9.6 Dependencies

| Dependency | Already In | Needed For |
|------------|-----------|------------|
| `ratatui 0.30` | bizra-cli | All Rust widgets |
| `crossterm 0.28` | bizra-cli | Event handling |
| `reqwest` | bizra-cli | API calls |
| `tokio mpsc` | bizra-cli | Event channel |
| `rich` | sovereign_terminal.py | Python rendering |
| `argparse` | core/sovereign | CLI routing |

All dependencies are already in the workspace. **Zero new dependencies required.**

## 9.7 Definition of Done

The BIZRA Sovereign Terminal is DONE when:

1. All 7 views render correctly (live and offline)
2. EventBus subscribers flow events to terminal views
3. Mission execution produces visible receipt with SEED earned
4. Wallet shows all 3 token types + 5 earning factors + Gini
5. Skills show lifecycle stage + compiled reflexes + skill tree
6. System shows all service health + constitutional metrics
7. JSON mode works for all views (scripting support)
8. Cold start < 1 second
9. 77+ tests pass (Rust + Python)
10. No hardcoded secrets, URLs, or tokens

Standing on Giants: Thompson & Ritchie (Unix), General Magic (TeleScript), Boyd (OODA)
