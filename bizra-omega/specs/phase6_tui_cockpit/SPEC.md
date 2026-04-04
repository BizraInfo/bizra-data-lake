# Phase 6 — Living Terminal Cockpit

## Goal

Transform `bizra tui` from a chat-focused scaffold into a **7-zone sovereign mission control** cockpit, wired to the exact backends already proven in Phases 1-5.

**Success criterion:** When the operator opens the TUI, they immediately see node health, living parliament, trust state, latest receipt lineage, manifest status, proactive brief, and next recommended action — all from real data, no mocks.

## Architecture Constraint

**Face layer only.** The TUI reads truth; it does not originate authority.

Every panel calls the same functions already used by:
- `exec_brief()` — substrate, runtime, receipts, constitution, models, recommendations
- `exec_trust()` — 13-check constitutional surface
- `exec_manifest()` — daily receipt statistics + BLAKE3 manifest seal
- `exec_receipt()` — ledger entries, chain walk, verification
- `exec_agents()` — PAT-7 + SAT-5 topology
- `exec_node()` — substrate discovery + compliance

No alternate receipt path. No shadow state. No new crates.

## Existing Infrastructure (Inventory)

### Already Built (reuse directly)
- `ratatui 0.30` with `all-widgets` feature
- `crossterm 0.28` with `event-stream`
- `App` state struct (app.rs): PATRole, FATEGates, AgentState, SystemMetrics, ActiveView, InputMode
- Theme (theme.rs): full Dubai Night Sky palette, metric_style(), ihsan_style(), symbols
- Widgets: Header, StatusBar, AgentCard, FateGauge
- TUI runner: run_tui() → enable_raw_mode → Terminal → run_app loop (100ms poll)
- 6 views: Dashboard, Agents, Chat, Tasks, Treasury, Settings

### Already Built (data backends in genesis_spine.rs)
- `LedgerEntry` struct + `load_ledger()` + `ledger_path()`
- `ResourceManifest::discover()` → hardware, models, runtimes
- `AgentRuntime::new().health()` → RuntimeHealth
- `TopologyCanon::PAT_COUNT/SAT_COUNT/GATE_ORDER`
- `GenesisSeal::compute()` / `GenesisSeal::node0_default()`
- `bizra_core::IHSAN_THRESHOLD/SNR_THRESHOLD/STRICT_IHSAN_THRESHOLD/RUNTIME_IHSAN_THRESHOLD`
- `bizra_core::omega::ADL_GINI_THRESHOLD`
- `mission_bridge::extract_model_names()`
- `PatAgent::ALL` / `SatAgent::ALL` with callsign/role/index

---

## Zone Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│ [1] HEADER  ◈ BIZRA v1.0  MoMo  [Trust: SOVEREIGN]  [Models: 4]   │
├─────────────────────┬─────────────────────┬──────────────────────────┤
│ [2] PARLIAMENT      │ [3] GHOST FEED      │ [4] TRUST RAIL           │
│ PAT-7 + SAT-5       │ Brief + Events      │ 13-check surface         │
│ roles, status       │ Proactive recs      │ SOVEREIGN/DEGRADED       │
│                     │                     │                          │
├─────────────────────┼─────────────────────┤                          │
│ [5] SUBSTRATE       │ [6] RECEIPT RAIL    │                          │
│ CPU/RAM/GPU/Models  │ Latest receipts     │                          │
│ Platform info       │ Chain integrity     │                          │
│                     │ Manifest seal       │                          │
├─────────────────────┴─────────────────────┴──────────────────────────┤
│ [7] STATUS BAR  NORMAL │ ◆ Guardian │ q:Quit Tab:View  / :Cmd       │
└──────────────────────────────────────────────────────────────────────┘
```

### Zone Dimensions (Constraints)

| Zone | Direction | Constraint |
|------|-----------|------------|
| Header [1] | V | Length(2) |
| Status Bar [7] | V | Length(2) |
| Content Area | V | Min(10) — everything between header and status |
| Left column (Parliament [2] + Substrate [5]) | H | Percentage(30) |
| Center column (Ghost [3] + Receipt [6]) | H | Percentage(35) |
| Right column (Trust [4]) | H | Percentage(35) |
| Parliament [2] | V | Percentage(60) of left |
| Substrate [5] | V | Percentage(40) of left |
| Ghost Feed [3] | V | Percentage(50) of center |
| Receipt Rail [6] | V | Percentage(50) of center |

---

## Data Layer — Extract from genesis_spine.rs

### New: `DashboardData` struct (in genesis_spine.rs or new dashboard_data.rs)

```
DashboardData {
    // Substrate
    cpu_name: String,
    cpu_cores: u32,
    ram_total_gb: f64,
    ram_available_gb: f64,
    gpu: Option<(String, u64, u64)>,  // (name, used_mb, total_mb)
    model_count: usize,
    runtime_count: usize,
    platform: String,

    // Models
    text_models: Vec<String>,
    vision_models: Vec<String>,

    // Runtime
    runtime_state: RuntimeState,
    reflex_mode: String,
    agents_active: u32,
    agents_registered: u32,

    // Receipt chain
    total_receipts: usize,
    complete_count: usize,
    degraded_count: usize,
    failed_count: usize,
    chain_valid: bool,
    last_receipt: Option<ReceiptSummary>,

    // Manifest (today)
    today_count: usize,
    today_complete: usize,
    manifest_seal: Option<String>,  // hex

    // Trust surface
    trust_checks: Vec<TrustCheck>,
    trust_verdict: TrustVerdict,  // Sovereign | Degraded

    // Parliament
    pat_agents: Vec<AgentInfo>,
    sat_agents: Vec<AgentInfo>,

    // Recommendations
    recommendations: Vec<String>,
}

ReceiptSummary {
    id_short: String,   // first 16 hex chars
    objective: String,
    state: String,      // "Complete" / "Degraded" / "Failed"
    signed: bool,
}

TrustCheck {
    name: String,
    passed: bool,
    actual: String,
    expected: String,
}

AgentInfo {
    index: u8,
    callsign: String,
    role: String,
    icon: String,
    team: String,  // "PAT" or "SAT"
}
```

### `gather_dashboard_data() -> DashboardData`

Single function that collects all data for the dashboard in one pass:
1. `ResourceManifest::discover()` → substrate + models
2. `AgentRuntime::new().health()` → runtime state
3. `load_ledger()` → receipts + chain + manifest
4. Constitutional constants → trust checks
5. `PatAgent::ALL` / `SatAgent::ALL` → parliament
6. Recommendation engine (same logic as exec_brief)

This function is called:
- On TUI startup (initial render)
- Every 5 seconds via tick timer (periodic refresh)
- On manual refresh (key binding: `r`)

---

## Rendering — Per Zone

### Zone 1: Header (augment existing)

Modify existing `Header` widget to show trust verdict and model count:

```
◈ BIZRA v1.0 │ MoMo (محمد) │ ►[1]Dashboard [2]Agents ... │ ● SOVEREIGN │ 4 models
```

Add `trust_verdict: &str` and `model_count: usize` fields to Header.

### Zone 2: Parliament Panel

New widget: `ParliamentPanel`

```
╭─ Parliament (12 agents) ─────────╮
│ PAT-7 (Your Council)             │
│ P0 ♟ Atlas       Strategy        │
│ P1 🔍 Oracle      Research        │
│ P2 ⚙ Forge       Code            │
│ P3 📊 Judge       Quality         │
│ P4 ✓ Crown       Constitution    │
│ P5 ▶ Herald      Delivery        │
│ P6 🛡 Nexus       Orchestration   │
│                                  │
│ SAT-5 (System Immune)            │
│ S0 🔒 Sentinel    Security        │
│ S1 ⚖ OracleSat   Scoring         │
│ S2 📜 Ledger      Receipts        │
│ S3 ⚡ Conductor   Routing         │
│ S4 🔮 Ambassador  Federation      │
╰──────────────────────────────────╯
```

Compact list format: `{index} {icon} {callsign:<12} {role}`.
PAT header gold, SAT header purple. Uses `borders::ARABIC` (rounded).

### Zone 3: Ghost Feed

New widget: `GhostFeed`

Shows the proactive brief + event feed. Content from `DashboardData.recommendations` plus runtime state and last receipt.

```
╭─ Ghost ✦ ─────────────────────╮
│ Good afternoon, MoMo.          │
│                                │
│ Runtime: Ready                 │
│ Agents: 12/12 active           │
│ Reflex: fast_track (2 rules)   │
│                                │
│ → System healthy. Ready for    │
│   sovereign missions.          │
│ → Run: bizra mission "..."     │
╰────────────────────────────────╯
```

Greeting computed from `chrono::Local::now().hour()`.
Recommendations rendered as `→ {text}` lines.

### Zone 4: Trust Rail

New widget: `TrustRail`

Renders the 13-check trust surface with pass/fail indicators:

```
╭─ Trust Surface ───────────────────╮
│ VERDICT: ✓ SOVEREIGN              │
│                                   │
│ [Constitutional Law]              │
│ ✓ Ihsan       0.95 = 0.95        │
│ ✓ SNR         0.85 = 0.85        │
│ ✓ Gini        0.35 = 0.35        │
│ ✓ Strict      0.99 = 0.99        │
│ ✓ Runtime     1.00 = 1.00        │
│                                   │
│ [Topology]                        │
│ ✓ PAT-7   ✓ SAT-5   ✓ 3-gate    │
│                                   │
│ [Genesis]                         │
│ ✓ Seal computable                 │
│                                   │
│ [Ledger]                          │
│ ✓ 5 receipts (5 valid)            │
│ ✓ 5 signed                        │
│                                   │
│ [Substrate]                       │
│ ✓ 4 models   ✓ 128 GB RAM        │
╰───────────────────────────────────╯
```

Checks styled with `Theme::success()` / `Theme::error()`.
Verdict banner: gold SOVEREIGN or amber DEGRADED.

### Zone 5: Substrate Panel

New widget: `SubstratePanel`

```
╭─ Substrate ───────────────────╮
│ 14th Gen Core i9 • 32 cores   │
│ RAM: 128 GB (45% used)        │
│ GPU: NVIDIA (6/16 GB VRAM)    │
│ Models: 4 (3 text, 1 vision)  │
│ Platform: linux-x86_64        │
╰───────────────────────────────╯
```

RAM percentage displayed as inline gauge if width allows.

### Zone 6: Receipt Rail

New widget: `ReceiptRail`

```
╭─ Receipt Chain (5 total) ─────╮
│ Chain: ✓ All hashes valid     │
│ Today: 3 missions (2✓ 1⚠)    │
│                               │
│ #5 a7f68f1f… Complete "..."   │
│ #4 2fd0bc3f… Complete "..."   │
│ #3 d060def2… Degraded "..."   │
│                               │
│ Manifest: b3c9a1… (3 today)   │
╰───────────────────────────────╯
```

Shows last N receipts (as many as fit). Chain status at top.
Manifest seal at bottom (today's BLAKE3 of receipt IDs).

### Zone 7: Status Bar (augment existing)

Add manifest summary to the right side of the status bar:

```
NORMAL │ ◆ Guardian │ q:Quit Tab:View r:Refresh │ Manifest: 3/3✓ today
```

---

## Keyboard Bindings

| Key | Mode | Action |
|-----|------|--------|
| `q` | Normal | Quit |
| `Tab` | Normal | Next view |
| `Shift+Tab` | Normal | Previous view |
| `1`-`6` | Normal | Jump to view |
| `r` | Normal | Refresh dashboard data |
| `j`/`k` | Normal | Navigate agents |
| `i` | Normal | Enter insert mode |
| `Esc` | Insert | Back to normal |
| `/` | Normal | Command mode |
| `m` | Normal (Dashboard) | Run mission (prompt for objective) |
| `t` | Normal (Dashboard) | Show trust details (switch to trust view... or just expand) |

---

## Implementation Plan

### Step 6A: Extract data layer

1. Create `DashboardData` struct and `gather_dashboard_data()` in genesis_spine.rs
2. Add `DashboardData` field to `App` struct
3. Call `gather_dashboard_data()` on TUI init

### Step 6B: New widgets

4. Create `parliament_panel.rs` widget
5. Create `ghost_feed.rs` widget
6. Create `trust_rail.rs` widget
7. Create `substrate_panel.rs` widget
8. Create `receipt_rail.rs` widget
9. Register in `widgets/mod.rs`

### Step 6C: Wire dashboard view

10. Replace `render_dashboard()` with 7-zone layout
11. Augment Header with trust verdict + model count
12. Augment StatusBar with manifest summary
13. Add periodic refresh (5s tick in run_app loop)
14. Add `r` key binding for manual refresh

### Step 6D: Compile + test

15. `cargo clippy -p bizra-cli`
16. `cargo fmt -p bizra-cli`
17. `cargo build -p bizra-cli --release`
18. Live test: `bizra tui` — verify all 7 zones render with real data

---

## Non-Goals (explicitly deferred)

- No new crates or dependencies
- No marketplace, federation, or token UI
- No mission execution from within TUI (Phase 7+)
- No async data fetching (gather is synchronous, same as CLI commands)
- No websocket or streaming updates
- No persistent TUI state across sessions

---

## Quality Gates

- 0 clippy warnings
- rustfmt clean
- Release binary < 3.5 MB
- All 7 zones render with real substrate/receipt/trust data
- Same data as CLI commands (verified by visual comparison)
