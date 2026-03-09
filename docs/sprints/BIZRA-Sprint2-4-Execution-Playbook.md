# BIZRA Sprint 2-4 Execution Playbook
## Terminal Genesis — 13 Days to Alpha-10

> **Locked:** 2026-03-08 12:30 GST
> **Scope:** Security fixes → Integration bugs → 5 terminal views → Forest proof
> **Rule:** Every task has a CI gate. Every view has acceptance criteria. Every day has a deliverable.

---

## Current State (March 8, 2026)

| Metric | Value | Target |
|--------|-------|--------|
| Terminal Views | 2/7 (Dashboard + Mission) | 7/7 |
| Security CRITICALs | 2 remaining | 0 |
| Integration Bugs | 6 found by audit | 0 |
| CI Gates | 12/12 (9 hard, 3 soft) | 12/12 hard |
| EventBus Topics | 12/38 active | 15+ |
| Coverage Floor | 38% | 45% |
| Genesis Readiness | 92% | 100% |

---

## Phase 1: Security + Integration Fixes (Day 1)
### Monday March 9 — Target: Zero CRITICALs, Zero Integration Bugs

#### Morning (4 hours)

| # | Task | File | Time | Gate |
|---|------|------|------|------|
| 1 | ZPK atomic writes (3 locations) | `core/zpk/kernel.py:739,754,788` | 30 min | `pytest tests/core/zpk/` |
| 2 | Rollback atomic write | `core/rollout/rollback.py:221` | 10 min | Smoke test |
| 3 | `IhsanNotAchievedError` magic number | `bizra_errors.py` | 5 min | `ruff check` |
| 4 | Mypy ratchet 1600 → 1200 | `ci.yml` | 5 min | CI green |
| 5 | Document Gini dual standard | `core/integration/constants.py` | 10 min | Comment |

**Acceptance:** `pytest tests/core/zpk/ tests/core/rollout/ -x` all green. Zero CRITICAL in security scan.

#### Afternoon (4 hours)

| # | Integration Bug | Fix | File | Time |
|---|----------------|-----|------|------|
| 1 | Auth mismatch: network/effect + milestones | Add `auth: true` to client calls | `sovereign-client.ts:445,448` | 2 min |
| 2 | Verb mismatch: verify endpoints | Change client from GET to POST | `sovereign-client.ts:520-527` | 5 min |
| 3 | Phantom endpoint: `sovereign.mission()` | Redirect to `/v1/plan` | `sovereign-client.ts:478` | 2 min |
| 4 | seed/potential + seed/episodes missing auth | Add `_authenticate_http_request()` to handlers | `api.py:2811,2834` | 10 min |
| 5 | Proxy timeout 10s → 30s for mutations | Increase `UPSTREAM_TIMEOUT_MS` for POST routes | `route.ts` | 5 min |
| 6 | `evidence_receipt_id` not in TS interface | Add to `MissionReceipt` interface | `sovereign-client.ts` | 2 min |

**Acceptance:** `tsc --noEmit` (0 errors), `pytest tests/integration/ -x` all green, `pytest tests/core/sovereign/test_api_exposure_policy.py` green.

#### Evening: Commit + Push

```bash
git add -A && git commit -m "fix: zero CRITICALs + 6 integration bugs (Day 1)"
git push origin main
# CI runs 24 jobs — verify all green
```

**Day 1 Deliverable:** Zero CRITICAL findings. Zero integration bugs. Security checklist 9/12 → 11/12 DONE.

---

## Phase 2: Terminal Views C.3-C.7 (Days 2-6)
### Build Contract §11 Terminal Build Matrix — Row Order

#### Day 2 (Tuesday): C.3 Timeline — The Proof Surface

| Component | Source | Content |
|-----------|--------|---------|
| `terminal-timeline.tsx` | EventBus + `/v1/seed/episodes` | Event log grouped by mission_id |
| `use-seed-episodes.ts` | NEW hook | Fetches `/v1/seed/episodes` with auth |
| Timeline entry rendering | Build Contract §7 | Severity-colored (info/notice/warning/critical) |
| Receipt chain links | Build Contract §10.3 | prev_hash → event_hash visualization |
| Tick entries | `tick.completed` events | scored/minted counts |

**Acceptance (Build Contract §10.3):**
- [ ] Renders latest 100 events from EventBus
- [ ] Groups by mission_id
- [ ] Shows receipt chain links (prev_hash → event_hash)
- [ ] Shows tick.completed entries with scored/minted counts
- [ ] Renders auth and invariant events with severity color
- [ ] Critical events sticky until acknowledged
- [ ] No synthetic entries — all from EventBus or ActionBus
- [ ] Filterable by category

**Time estimate:** 8 hours. **Gate:** `tsc --noEmit`, build OK, `/terminal` 200.

---

#### Day 3 (Wednesday): C.4 Memory — The Continuity Substrate

| Component | Source | Content |
|-----------|--------|---------|
| `terminal-memory.tsx` | Living Memory stores | Semantic + episodic + procedural |
| `use-memory-stats.ts` | NEW hook for `/v1/memory/stats` | Memory store statistics |
| Morning briefing renderer | `BriefingContext` type | Last mission, near-compile, quality trend |
| Active projects list | Semantic memory | Projects with last activity date |
| Privacy notice | Hardcoded | "All data is local. Nothing leaves your device." |

**Acceptance (Build Contract §10.4):**
- [ ] Shows semantic profile (domains, hours, preferences)
- [ ] Shows last 10 missions with outcomes
- [ ] Shows active projects with last activity
- [ ] Shows work streak
- [ ] Shows near-compilation patterns
- [ ] Morning briefing renders on startup
- [ ] Privacy note visible: "All data is local"

**Time estimate:** 6 hours. **Gate:** `tsc --noEmit`, build OK.

---

#### Day 4 (Thursday): C.5 Agents/Skills — The MMORPG Sheet

| Component | Source | Content |
|-----------|--------|---------|
| `terminal-skills.tsx` | Skill Tier calc + reflex cache | PAT/SAT state, tier progression |
| Skill tree visualization | `action_schema_v1.json` tiers | Novice → Adept → Expert → Master |
| Reflex inventory | `reflex_cache` | Compiled patterns with stats |
| Tier progress bar | Action count + Ihsan threshold | Visual progress toward next tier |
| Human lifecycle stage | `node_value.py` lifecycle calc | Seed → ... → Catalyst |

**Acceptance (Build Contract §10.5):**
- [ ] Shows PAT-7 agent list with status
- [ ] Shows SAT-5 agent list
- [ ] Shows current tier with progress bar
- [ ] Shows human lifecycle stage with progress
- [ ] Shows compiled reflexes with avg Ihsan, count, latency
- [ ] Shows near-compile candidates with N/3 threshold
- [ ] Shows locked/unlocked skills by tier

**Time estimate:** 8 hours. **Gate:** `tsc --noEmit`, build OK.

---

#### Day 5 (Friday): C.6 Network + C.7 Settings (Combined)

| Component | Source | Content |
|-----------|--------|---------|
| `terminal-network.tsx` | `/v1/network/effect`, `/v1/node/value` | Network projections + node value |
| `terminal-settings.tsx` | `/v1/health`, sovereign_state/ | Identity, privacy, model routing |
| Milestone projections | `/v1/network/milestones` | 1 → 8B node visualization |
| Node identity display | Ed25519 public key prefix | Sovereignty indicator |
| Model routing config | Runtime config | Which LLM, local/remote toggle |

**Network Acceptance (Build Contract §10.6):**
- [ ] Shows 5-factor node value composite
- [ ] Shows lifecycle stage + progress within stage
- [ ] Shows milestone projections (1 → 8B nodes)
- [ ] Shows diffusion eligibility for published reflexes
- [ ] Offline fallback: last cached projection

**Settings Acceptance (Build Contract §10.7):**
- [ ] Shows node ID + public key prefix
- [ ] Shows current model routing
- [ ] Shows dev/prod mode indicator
- [ ] Shows auth state (authenticated / anonymous-dev)
- [ ] Shows permission policy defaults
- [ ] Editable: model routing preferences

**Time estimate:** 8 hours combined. **Gate:** All 7 views render, `tsc --noEmit`, build OK.

---

#### Day 6 (Saturday): Integration + Polish

| Task | Time |
|------|------|
| Wire all 7 views to shell navigation (keyboard shortcuts 1-7) | 2 hrs |
| EventBus real-time subscription in Timeline | 2 hrs |
| Offline fallback for all views (yellow "estimates only" indicator) | 2 hrs |
| Cross-view navigation (mission receipt → timeline entry) | 1 hr |
| Full build + typecheck + route test | 1 hr |

**Day 6 Deliverable:** All 7 views navigable. Terminal v1 feature-complete. 49 acceptance checks verified.

---

## Phase 3: Production Hardening (Days 7-9)

#### Day 7 (Sunday): CI Gate Hardening

| Gate | Change | Time |
|------|--------|------|
| Container Signing (Gate 10) | Remove `continue-on-error: true` | 30 min |
| SBOM Generation (Gate 11) | Remove `continue-on-error: true` | 30 min |
| Performance Gate (Gate 12) | Remove `continue-on-error: true` | 30 min |
| Production deploy condition | Require all 12 gates green | 1 hr |

**Gate:** Full CI pipeline runs end-to-end, all 24 jobs green with hard gates.

#### Day 8 (Monday): Security Final Sweep

| Task | Time |
|------|------|
| Rate limiting on `/v1/verify/*` endpoints | 3 hrs |
| Activate `ihsan.breach` + `invariant.violation` event topics | 2 hrs |
| Coverage ratchet: 38% → 45% (add tests for 5 largest untested modules) | 3 hrs |

**Gate:** Security re-scan shows 0 CRITICAL, 0 HIGH. Coverage ≥ 45%.

#### Day 9 (Tuesday): E2E Proof Demo

| Task | Time |
|------|------|
| Canonical E2E test: submit → receipt → tick → reflex → cache hit → wallet → memory | 4 hrs |
| Film the Woow demo: 1800ms → 50ms reflex precipitation | 2 hrs |
| Capture terminal screenshots for all 7 views | 1 hr |
| Prepare Alpha-10 invitation package | 1 hr |

**Day 9 Deliverable:** Terminal v1 proven end-to-end. Demo filmed. Screenshots captured.

---

## Phase 4: Alpha-10 Launch (Days 10-13)

#### Day 10 (Wednesday): Alpha-10 Invitations Sent

| Task | Details |
|------|---------|
| Select 10 individuals | 2 AI, 2 blockchain, 2 dev tools, 2 ethics, 1 non-Western, 1 practitioner |
| Write 10 personal messages | Each explains why they were chosen + one line from البذرة |
| Send invitations | Personal email/DM, not bulk |
| Include: Identity Canon + Proof Canon + demo video | Alpha-10 package |

#### Days 11-13: Support, Monitor, Iterate

| Day | Focus |
|-----|-------|
| Thursday | Monitor first 3 users' sessions. Fix any breaking issues immediately. |
| Friday | Collect first feedback. Iterate on UX pain points. |
| Saturday March 21 | **Genesis.** Review metrics: Gini coefficient, reflex compilation rate, user satisfaction. |

---

## Quality Assurance Framework

### Per-View QA Gate (PMBOK Monitoring)

Every view must pass before merge:

```
1. tsc --noEmit (0 type errors)
2. next build (no build errors)
3. /terminal returns 200
4. Build Contract acceptance criteria (§10.N) all checked
5. Source-of-truth matrix (§4) verified — no invented state
6. Offline fallback tested (disconnect backend, verify graceful degradation)
```

### Per-Day QA Gate (DORA Alignment)

```
1. git push to main (at least 1 commit/day)
2. CI pipeline green (all 24 jobs)
3. No regression in existing tests
4. Memory file updated with progress
```

### Pre-Genesis QA Gate (Build Contract §14 Lock Condition)

```
[ ] All 7 views render without error
[ ] All 7 per-view acceptance criteria pass (49 checks)
[ ] Canonical E2E test passes
[ ] Security checklist items 1-9 complete
[ ] Mission loop stable (100 consecutive missions without crash)
[ ] Event spine alive (12+ topics firing)
[ ] Memory continuity visible (briefing shows last session)
[ ] Reflex benefit demonstrable (S1 path visibly faster)
[ ] Receipt normalization verified (all fields per §8)
[ ] Permission envelope enforced (escalation only on boundary crossing)
```

---

## DORA Target Metrics (Post-Genesis)

| Metric | Current | Sprint 2 Target | Elite Target |
|--------|---------|----------------|-------------|
| Deployment Frequency | Weekly | Daily (1 push/day) | On-demand |
| Lead Time for Changes | ~2 days | < 4 hours | < 1 hour |
| Change Failure Rate | ~5% | < 5% | < 5% |
| Time to Recovery | ~30 min | < 30 min | < 1 hour |

---

## Risk Mitigation During Sprint

| Risk | Mitigation | Owner |
|------|-----------|-------|
| View takes longer than estimate | Cut scope to acceptance criteria minimum, defer polish | Build Contract |
| Integration test fails on push | Fix before proceeding to next view | CI gate |
| Offline fallback breaks live mode | Test both modes per view | QA gate |
| Alpha-10 user hits undiscovered bug | Monitor sessions Day 11-13, hotfix pipeline ready | DevOps |
| Spec sprawl resumes | Blueprint rule: no new specs, all work references Build Contract only | Governance |

---

## Document Arsenal for Alpha-10

| Document | Purpose | Status |
|----------|---------|--------|
| Identity Canon v1.0 | Vision + doctrine + public narrative | LOCKED |
| Proof Canon v1.0 | Every claim mapped to evidence | LOCKED |
| Locked Build Contract v1.0 | Engineering execution contract | LOCKED |
| Evaluation Report (PDF) | Independent assessment with charts | DELIVERED |
| Elite Engineering Blueprint | PMBOK + DORA + CMMI + roadmap | DELIVERED |
| Sprint 2-4 Execution Playbook | This document — day-by-day plan | LOCKED |

---

## The Line

> **"One mission, one proof, remembered forever."**

13 days. 7 views. 1 terminal. 10 seeds.

The forest begins.

**LOCKED: 2026-03-08 · Dubai · BIZRA Foundation**
