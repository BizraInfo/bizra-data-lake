# Phase 7 — Spec-to-Shipped Delta Matrix

**Date:** 2026-04-04
**Base:** Phase 6 Living Terminal Cockpit (commit 2e8d51c5, 19/19 tests)
**Method:** Each terminal spec (phase_02 through phase_09) mapped against shipped code. Only deltas listed.

---

## Summary

| Spec | Coverage | Delta LOC | Sprint |
|------|----------|-----------|--------|
| phase_02 Briefing | **~70% shipped** | ~120 | 1 |
| phase_03 Mission | **~30% shipped** | ~350 | 2 |
| phase_04 Receipts | **~50% shipped** | ~250 | 2 |
| phase_05 Wallet | **~5% shipped** | ~280 | 3 |
| phase_06 Memory+Skills | **~10% shipped** | ~300 | 4 |
| phase_07 System | **~60% shipped** | ~180 | 3 |
| phase_08 EventBus | **~0% shipped** | ~300 | 1 (enabling) |
| phase_09 Build Plan | **superseded** | 0 | — |

**Total delta: ~1,780 LOC across 4 sprints**

---

## Detailed Delta Per Spec

### phase_02_view_briefing — 70% SHIPPED

**Already shipped:**
- [x] Time-aware greeting (GhostFeed: greeting varies by hour)
- [x] Recommendations display (GhostFeed: bullet list of proactive suggestions)
- [x] Runtime state display (GhostFeed: state + agent counts + reflex mode)
- [x] Receipt summary in dashboard (ReceiptRail: today count, chain status)

**Missing delta:**
- [ ] Last completed mission card (intent, Ihsan, SEED earned) — needs ReceiptSummary expansion
- [ ] Quality trend sparkline (last 10 Ihsan scores) — new widget (~60 LOC)
- [ ] Near-compiling reflexes display — needs reflex cache query (~40 LOC)
- [ ] Wallet compact view in briefing — needs wallet data in DashboardData (~20 LOC)
- [ ] Offline fallback label — minor: show "[OFFLINE]" when backends unreachable

**Priority:** LOW — most value already delivered via GhostFeed. Sparkline is nice-to-have.

---

### phase_03_view_mission — 30% SHIPPED

**Already shipped:**
- [x] Mission execution via `bizra mission "..."` CLI command
- [x] Constitutional gate chain (Schema, Ihsan, SNR)
- [x] Receipt generation with BLAKE3 + Ed25519
- [x] Receipt persistence to disk ledger
- [x] Model routing (S1/S2 tier selection)

**Missing delta:**
- [ ] Interactive mission input in TUI (not just CLI arg) — reuse Chat view input buffer (~80 LOC)
- [ ] Live agent progress stream (show each agent step as it executes) — needs EventBus (~100 LOC)
- [ ] Receipt card rendered after completion (Ihsan, SEED, hash, chain link) — new inline widget (~80 LOC)
- [ ] Gate confirmation dialogs (CONSTITUTIONAL_RISK, IRREVERSIBLE_ACTION) — interrupt handling (~60 LOC)
- [ ] Reflex hit indicator (S1 vs S2 route display) — minor UI addition (~30 LOC)

**Priority:** HIGH — this is where "Ghost becomes proactive." Event-driven mission tracking is the killer feature.

---

### phase_04_view_receipts — 50% SHIPPED

**Already shipped:**
- [x] Receipt rail in dashboard (last 10 receipts with state, ID, objective)
- [x] Chain integrity display (valid/broken badge)
- [x] Today summary (mission count + completions)
- [x] Manifest seal display

**Missing delta:**
- [ ] Scrollable full receipt list (beyond 10-item rail) — expand ReceiptRail or new view (~80 LOC)
- [ ] Per-receipt detail view (8-dimensional Ihsan breakdown) — new widget (~100 LOC)
- [ ] Filter buttons (all/qualified/rejected) — state + render logic (~40 LOC)
- [ ] Export JSON/CSV — file write from receipt data (~30 LOC)

**Priority:** MEDIUM — the rail covers 80% of daily use. Deep view is valuable for auditing.

---

### phase_05_view_wallet — 5% SHIPPED

**Already shipped:**
- [x] Treasury view exists as placeholder ("coming soon")
- [x] SEED token minting logic exists in bizra-core (CanonicalReceipt)

**Missing delta:**
- [ ] Balance display (SEED, BLOOM, BRANCH, Locked) — new widget (~60 LOC)
- [ ] Zakat + community pool metrics — needs token ledger query (~40 LOC)
- [ ] Supply cap gauge — progress bar widget (~30 LOC)
- [ ] Gini coefficient + headroom display — needs Gini calc from core (~40 LOC)
- [ ] 5 earning factors breakdown — factor bars (~50 LOC)
- [ ] Composite node value (geometric mean) — calc + display (~30 LOC)
- [ ] Live sync indicator — minor (~10 LOC)
- [ ] Tier/Stage display — from lifecycle calc (~20 LOC)

**Priority:** MEDIUM — important for SEED economics visibility but not blocking Ghost expansion.

---

### phase_06_view_memory_skills — 10% SHIPPED

**Already shipped:**
- [x] Agent topology (ParliamentPanel: PAT-7 + SAT-5 roster)
- [x] Reflex mode + rule count (GhostFeed)

**Missing delta:**
- [ ] Semantic user profile panel — needs Living Memory query (~60 LOC)
- [ ] Active projects list — needs project tracking (~40 LOC)
- [ ] Recent missions table (last 10 with stats) — extend receipt data (~40 LOC)
- [ ] Compiled reflexes table — needs reflex persistence query (~60 LOC)
- [ ] Skill tree grid (6 tiers) — new widget (~80 LOC)
- [ ] Lifecycle stage bar — new widget (~40 LOC)

**Priority:** LOW — MMORPG progression is compelling but depends on reflex compilation being operational.

---

### phase_07_view_system — 60% SHIPPED

**Already shipped:**
- [x] Node identity in header (node name, version)
- [x] Constitutional metrics (trust checks: Ihsan, SNR, Gini, Strict, Runtime)
- [x] Trust verdict (SOVEREIGN/DEGRADED)
- [x] Substrate health (CPU, RAM, GPU, models, platform)
- [x] Receipt chain integrity (valid/broken, count)
- [x] Model inventory (text/vision breakdown)

**Missing delta:**
- [ ] Service health table (9 services: ports, status, ping) — needs health probes (~80 LOC)
- [ ] Security posture panel (headers, auth, rate limit, SAST/DAST) — needs audit data (~60 LOC)
- [ ] K8s/Docker status — needs kubectl/docker probes (~40 LOC)

**Priority:** MEDIUM — substrate panel covers the core. Service health probes add operational depth.

---

### phase_08_event_bus_wiring — 0% SHIPPED

**Already shipped:**
- [x] 5-second polling refresh of DashboardData (gather_dashboard_data)
- [x] Manual refresh via `r` key

**Missing delta (ENABLING INFRASTRUCTURE):**
- [ ] EventBus subscriber integration — 12 subscribers routing to TUI views (~100 LOC)
- [ ] Event mpsc channel (tokio) — async event pipe to render loop (~80 LOC)
- [ ] Event-to-view routing table (40 event types → target view) (~40 LOC)
- [ ] Status bar event flash (transient high-priority notifications) (~30 LOC)
- [ ] Live mission stream (real-time agent progress during execution) (~50 LOC)

**Priority:** HIGHEST — this is the enabling layer. Without it, Ghost cannot be proactive in real-time. All other specs depend on events flowing.

---

### phase_09_build_plan — SUPERSEDED

The build plan spec assumed a fresh program. The shipped cockpit changes the baseline:
- 5 sprints → **4 sprints** (Sprint 1 foundation is mostly done)
- 2,150 LOC estimated → **~1,780 LOC delta** (30% already shipped)
- 77 tests → **19 shipped + ~58 new = ~77 total** (on track)

---

## Recommended Sprint Order

Based on the delta matrix, the highest-SNR implementation order is:

### Sprint 7.1: EventBus + Live Ghost (P0, ~400 LOC)
**Files:** `api_client.rs` (new), event channel in `main.rs`, status_bar flash
- Wire tokio mpsc channel for async events
- Implement 4 highest-priority subscribers (mission.*, receipt.*, economy.*, ihsan.gate.*)
- Status bar event flash for high-priority alerts
- Live mission progress stream in Ghost feed
- **Tests:** ~12 new

### Sprint 7.2: Mission View + Receipt Deep View (P0, ~430 LOC)
**Files:** Mission input in TUI, receipt detail widget, gate confirmations
- Interactive mission submission from dashboard (not just CLI)
- Receipt card after completion
- Scrollable receipt list with filter
- Receipt detail view (8-dim Ihsan)
- Gate interruption dialogs
- **Tests:** ~15 new

### Sprint 7.3: System + Wallet (P1, ~380 LOC)
**Files:** `wallet.rs` (new), `system.rs` (new), service health probes
- Wallet balance + factors + Gini + supply cap
- Service health table (9 services)
- Security posture panel
- **Tests:** ~12 new

### Sprint 7.4: Memory + Skills + Polish (P1, ~340 LOC)
**Files:** `memory.rs` (new), `skill_tree.rs` (new), `briefing.rs` enhancements
- Semantic profile + recent missions
- Compiled reflexes + skill tree grid
- Lifecycle stage bar
- Briefing sparkline + reflex candidates
- **Tests:** ~15 new

### Sprint 7.5: Hardening (P2, ~230 LOC)
- Export JSON/CSV
- Offline fallback labels
- K8s/Docker status
- Cold start optimization (<1s target)
- **Tests:** ~4 new

---

## Frozen Anchors (from Phase 6, carried forward)

- No new crates
- No shadow state — all data from existing backends
- Borrow-only reveal — widgets borrow from data, cannot mutate proof layer
- Same truth path — no new authority centers
- Constitutional thresholds imported from bizra-core, never redefined
