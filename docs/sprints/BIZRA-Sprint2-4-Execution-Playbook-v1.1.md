# BIZRA Sprint 2-4 Execution Playbook v1.1
## Terminal Genesis — Updated March 8, 2026

> **Updated:** 2026-03-08 14:00 GST
> **Changes from v1.0:** Phase 1 COMPLETE, Day 1 deliverables shipped, Phase 77 integration closed
> **Genesis Readiness:** 95% (was 92%)
> **Critical Blockers:** 0 (was 2)

---

## Current State (Updated)

| Metric | v1.0 (Mar 8 AM) | Now (Mar 8 PM) | Delta |
|--------|-----------------|----------------|-------|
| Terminal Views | 2/7 | **2/7** | Holding (Phase 2 starts) |
| Security CRITICALs | 2 | **0** | **-2 (ALL FIXED)** |
| Integration Bugs | 6 | **0** | **-6 (ALL FIXED)** |
| Frontend Type Drift | 5 mismatches | **0** | **-5 (ALL ALIGNED)** |
| Contract Tests | 0 | **128** | **+128 (PHASE77)** |
| CI Named Gates | 0 | **PHASE77-001** | **Added** |
| Deploy Smoke Tests | 0 endpoints | **15** | **+15** |
| CI Gates | 12 (9 hard, 3 soft) | 12 (9 hard, 3 soft) | Same |
| Dead Alias | evidence_receipt_id | **Removed** | Contract-clean |
| Genesis Readiness | 92% | **95%** | **+3%** |

---

## COMPLETED: Phase 1 — Security + Integration (Day 1) ✅

All Day 1 tasks from v1.0 are DONE:

| # | Task | Status | Evidence |
|---|------|--------|----------|
| 1 | ZPK atomic writes (3 locations) | **DONE** | kernel.py:739,754,788 |
| 2 | Rollback atomic write | **DONE** | rollback.py:221 |
| 3 | IhsanNotAchievedError magic number | **DONE** | Imports from constants |
| 4 | Mypy ratchet 1600 → 1200 | **DONE** | ci.yml updated |
| 5 | Auth mismatch: network/effect + milestones | **DONE** | sovereign-client.ts |
| 6 | Verb mismatch: verify endpoints | **DONE** | GET → POST |
| 7 | Phantom endpoint: sovereign.mission() | **DONE** | Redirected to /v1/plan |
| 8 | seed/potential + episodes auth guard | **DONE** | _authenticate_http_request added |
| 9 | Proxy timeout 10s → 30s | **DONE** | route.ts updated |
| 10 | Frontend type alignment (5 types) | **DONE** | All 5 aligned |
| 11 | Adapter hooks (3 transforms) | **DONE** | use-sovereign-api.ts |
| 12 | Response shape contract tests | **DONE** | 7 new tests |
| 13 | PHASE77-001 CI gate | **DONE** | quality-gates job |
| 14 | Deploy smoke tests (15 endpoints) | **DONE** | deploy.yml staging + production |
| 15 | evidence_receipt_id dead alias removed | **DONE** | terminal.py to_dict() |

**Phase 1 Deliverable: SHIPPED.** Zero CRITICALs. Zero integration bugs. Zero type drift. 128 contract tests. CI gate blocking.

---

## NEXT: Phase 2 — Terminal Views C.3-C.7 (Days 2-6)

### Day 2: C.3 Timeline — The Proof Surface

| Component | Source | Content |
|-----------|--------|---------|
| `terminal-timeline.tsx` | EventBus + `/v1/seed/episodes` | Event log grouped by mission_id |
| `use-seed-episodes.ts` | NEW hook | Fetches `/v1/seed/episodes` with auth |
| Timeline entry rendering | Build Contract §7 | Severity-colored (info/notice/warning/critical) |
| Receipt chain links | Build Contract §10.3 | prev_hash → event_hash visualization |

**Acceptance (Build Contract §10.3):**
- [ ] Renders latest 100 events
- [ ] Groups by mission_id
- [ ] Shows receipt chain links
- [ ] Shows tick entries with scored/minted counts
- [ ] Severity-colored (critical sticky until acknowledged)
- [ ] No synthetic entries
- [ ] Filterable by category
- [ ] 8/8 criteria met

**Prerequisites:** `useSeedEpisodes` hook (seed/episodes now has auth guard from Day 1 fix). Adapter hook may be needed if response shape evolves.

**Time:** 8 hours. **Gate:** tsc --noEmit, build OK, /terminal 200.

---

### Day 3: C.4 Memory — The Continuity Substrate

| Component | Source | Content |
|-----------|--------|---------|
| `terminal-memory.tsx` | Living Memory + `/v1/memory/stats` | Semantic + episodic + procedural |
| Morning briefing renderer | `BriefingContext` type | Last mission, near-compile, quality trend |
| Privacy notice | Hardcoded | "All data is local" |

**Acceptance (Build Contract §10.4):** 7 criteria. **Time:** 6 hours.

**Note:** `useMemoryStats` hook already exists with `toMemoryStatsView()` adapter from Phase 1.

---

### Day 4: C.5 Agents/Skills — The MMORPG Sheet

| Component | Source | Content |
|-----------|--------|---------|
| `terminal-skills.tsx` | Skill Tier calc + reflex cache | PAT/SAT state, tier progression |
| Reflex inventory | `reflex_cache` | Compiled patterns with stats |
| Human lifecycle stage | `node_value.py` | Seed → Catalyst |

**Acceptance (Build Contract §10.5):** 7 criteria. **Time:** 8 hours.

---

### Day 5: C.6 Network + C.7 Settings (Combined)

| Component | Source | Content |
|-----------|--------|---------|
| `terminal-network.tsx` | `/v1/network/effect`, `/v1/node/value` | Network projections + node value |
| `terminal-settings.tsx` | `/v1/health`, sovereign_state/ | Identity, model routing, auth state |

**Network Acceptance (Build Contract §10.6):** 5 criteria.
**Settings Acceptance (Build Contract §10.7):** 6 criteria.
**Time:** 8 hours combined.

**Note:** `useNetworkEffect` and `useNetworkMilestones` hooks now send auth headers (fixed Day 1).

---

### Day 6: Integration + Polish

| Task | Time |
|------|------|
| Wire all 7 views to shell navigation (keyboard shortcuts 1-7) | 2 hrs |
| EventBus real-time subscription in Timeline | 2 hrs |
| Offline fallback for all views (yellow indicator) | 2 hrs |
| Cross-view navigation (receipt → timeline entry) | 1 hr |
| Full build + typecheck + route test | 1 hr |

**Day 6 Deliverable:** All 7 views navigable. 49 acceptance checks verified.

---

## Phase 3: Production Hardening (Days 7-9)

### Day 7: CI Gate Hardening

| Gate | Change | Time |
|------|--------|------|
| Container Signing (Gate 10) | Remove `continue-on-error: true` | 30 min |
| SBOM Generation (Gate 11) | Remove `continue-on-error: true` | 30 min |
| Performance Gate (Gate 12) | Remove `continue-on-error: true` | 30 min |
| Production deploy condition | Require all 12 gates hard-green | 1 hr |

### Day 8: Security + Coverage

| Task | Time |
|------|------|
| Rate limiting on `/v1/verify/*` endpoints | 3 hrs |
| Activate `ihsan.breach` + `invariant.violation` topics | 2 hrs |
| Coverage ratchet: 38% → 45% | 3 hrs |

### Day 9: E2E Proof Demo

| Task | Time |
|------|------|
| Canonical E2E: mission → receipt → tick → reflex → cache hit → wallet | 4 hrs |
| Film Woow demo: 1800ms → 50ms reflex precipitation | 2 hrs |
| Capture all 7 view screenshots | 1 hr |
| Prepare Alpha-10 invitation package | 1 hr |

---

## Phase 4: Alpha-10 Launch (Days 10-13)

### Day 10: Invitations Sent

Select 10 individuals per the framework:
- 2 AI researchers/builders
- 2 blockchain/Web3 builders
- 2 developer tool creators
- 2 ethical tech community leaders
- 1 non-Western practitioner (Mother Test validation)
- 1 daily practitioner (not a celebrity)

Each receives: personal message + Identity Canon + Proof Canon + demo video + one line from البذرة.

### Days 11-13: Monitor, Iterate, Genesis

| Day | Focus |
|-----|-------|
| 11 (Thursday) | Monitor first 3 users. Hotfix pipeline ready. |
| 12 (Friday) | Collect feedback. Iterate on pain points. |
| 13 (Saturday Mar 21) | **GENESIS.** Gini check. Reflex check. Satisfaction check. |

---

## Evidence Package (Updated)

| Document | Version | Status | Purpose |
|----------|---------|--------|---------|
| Identity Canon | v1.0 | LOCKED | What BIZRA IS |
| **Proof Canon** | **v1.1** | **LOCKED** | What BIZRA HAS (updated evidence) |
| Build Contract | v1.0 | LOCKED | What engineering builds against |
| Evaluation Report | v1.0 | DELIVERED | Independent assessment |
| Elite Blueprint | v1.0 | DELIVERED | PMBOK + DORA + CMMI |
| **Execution Playbook** | **v1.1** | **LOCKED** | Day-by-day plan (this doc) |

---

## Quality Gates (Per Day)

```
Every view:    tsc --noEmit → next build → /terminal 200 → §10.N acceptance
Every day:     git push → CI green (24 jobs) → no regression → memory updated
Pre-Genesis:   128 PHASE77 tests + 4 E2E metabolism + 49 view acceptance + 0 CRITICAL
```

---

## Genesis Readiness Trajectory

| Date | Readiness | Key Event |
|------|-----------|-----------|
| Mar 7 (morning) | 74% | Session starts |
| Mar 7 (evening) | 92% | Sprint 1 complete, heartbeat wired |
| Mar 8 (morning) | 92% | Proof Canon v1.0, Build Contract locked |
| **Mar 8 (afternoon)** | **95%** | **Phase 77 complete, 0 CRITICALs** |
| Mar 9-14 | 97% | 5 terminal views built |
| Mar 15-17 | 99% | CI hardened, E2E proven, demo filmed |
| **Mar 21** | **100%** | **GENESIS** |

---

> **"One mission, one proof, remembered forever."**
>
> 13 days. 5 views. 10 seeds. One forest.
>
> **LOCKED: 2026-03-08 · Dubai · BIZRA Foundation**
