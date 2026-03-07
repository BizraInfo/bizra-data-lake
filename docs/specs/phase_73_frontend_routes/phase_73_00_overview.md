# Phase 73: Frontend Route-Level Implementation

**Status:** SPECIFICATION
**Priority:** P0 — First user-facing product surface
**Author:** spec-pseudocode mode
**Date:** 2026-03-06
**Upstream:** `FRONTEND_MASTER_SPEC.md` (product contract, frozen surface order)
**Downstream:** Phase 43 (onboarding), Phase 45 (dashboard), Phase 72 (node value KPI)

## Scope

Cut the frozen frontend master spec into route-level implementation tickets.
Each route gets: path, data contracts, component tree, backend dependencies,
and TDD anchors.

**Frozen delivery order:**
1. Public trust site
2. Onboarding wizard
3. Daily dashboard shell
4. Contributor desktop client (GUI + TUI)
5. Operator/admin console

**This phase covers surfaces 1-3.** Surfaces 4-5 are deferred.

## What Already Exists

| Artifact | File | Status |
|---|---|---|
| Onboarding flow (5 steps) | `filedfs/onboarding/OnboardingFlow.jsx` | Functional prototype, simulated backend |
| Onboarding steps (5 files) | `filedfs/onboarding/steps/*.jsx` | Working, need token migration + live wiring |
| Dashboard prototype A | `filedfs/bizra-dashboard.jsx` | Design exploration, 33K LOC |
| Dashboard prototype B | `filedfs/node0-dashboard.jsx` | Node-focused, 43K LOC |
| Operator console | `docs/node0_operations_dashboard.html` | Static HTML prototype |
| MVP shell | `filedfs/node0-mvp.jsx` | Minimal viable product layout |
| Website plan | `docs/WEBSITE_PLAN.md` | Page structure, typography, colors frozen |
| Onboarding spec | `docs/specs/phase_43_onboarding_polish.md` | 6-step revised flow with pseudocode |
| Dashboard spec | `docs/specs/phase_45_daily_loop_dashboard.md` | Daily Loop, component tree, layout |

## Tech Stack (Frozen)

| Layer | Choice | Rationale |
|---|---|---|
| Framework | React 19 + Vite | Already in BIZRA-OS, proven |
| Routing | React Router v7 | File-based routes, lazy loading |
| Styling | Tailwind CSS + CSS custom properties | Design tokens from WEBSITE_PLAN.md |
| State | Zustand (client) + React Query (server) | Lightweight, no boilerplate |
| API client | fetch + typed wrappers | No heavy SDK, aligns with sovereign principle |
| Persistence | IndexedDB (via idb) | Offline-first, onboarding checkpoint resume |
| i18n | react-intl | Arabic support (Noto Sans Arabic) mandatory |
| Testing | Vitest + Testing Library | Fast, component-level |
| A11y | axe-core + manual audit | WCAG 2.1 AA minimum |

## Route Map

```
/                          → HomePage (Daily Loop dashboard)
/onboarding                → OnboardingFlow (6-step wizard)
/onboarding/contributor    → ContributorOnboardingFlow (node setup)
/learn                     → LearnPage (placeholder)
/earn                      → EarnPage (placeholder)
/community                 → CommunityPage (placeholder)
/wallet                    → WalletPage (placeholder)
/settings                  → SettingsPage (placeholder)

# Public site (separate build or subdomain)
/site                      → LandingPage
/site/how-it-works         → HowItWorksPage
/site/safety               → SafetyPage
/site/demo                 → DemoPage
/site/faq                  → FAQPage
/site/join                 → JoinPage
```

## Backend API Contracts Required

These endpoints must exist before frontend wiring. Existing endpoints are
marked; new endpoints reference Phase 72 or are net-new.

| Endpoint | Method | Status | Auth | Provides |
|---|---|---|---|---|
| `/v1/health` | GET | EXISTS | No | Readiness + seed_engine status |
| `/v1/health/live` | GET | EXISTS | No | Liveness probe (<5ms) |
| `/v1/health/deep` | GET | EXISTS | No | 11 subsystems, health_score |
| `/v1/seed/potential` | GET | EXISTS | No | Sovereignty score, tier, episodes, streak |
| `/v1/seed/episodes` | GET | EXISTS | No | Recent growth episodes with receipt hashes |
| `/v1/token/balance` | GET | EXISTS | Yes | Balances by token type (SEED/BLOOM/IMPT) |
| `/v1/token/supply` | GET | EXISTS | Yes | Total supply, yearly cap, ledger validity |
| `/v1/sel/episodes` | GET | EXISTS | Yes | Paginated experience episodes |
| `/v1/auth/login` | POST | EXISTS | No | JWT tokens (access + refresh) |
| `/v1/auth/register` | POST | EXISTS | No | User creation + tokens |
| `/v1/auth/me` | GET | EXISTS | Yes | Current user profile |
| `/v1/verify/genesis` | POST | EXISTS | No | Genesis identity chain verification |
| `/v1/suggestions` | GET | EXISTS | Yes | Proactive knowledge from living memory |
| `/v1/node/value` | GET | PHASE 72 | Yes | 5-factor KPI composite (geometric mean) |
| `/v1/node/lifecycle` | GET | PHASE 72 | Yes | Human stage, progress, next threshold |
| `/v1/network/effect` | GET | PHASE 72 | Yes | Skills, compute, latency projection |
| `/v1/onboarding/state` | GET/PUT | NEW | Yes | Checkpoint resume, step data |
| `/v1/onboarding/teach` | POST | NEW | Yes | TEACH verb (kind, content, confidence) |
| `/v1/onboarding/verify` | POST | NEW | Yes | Environment verification result |
| `/v1/missions/history` | GET | NEW | Yes | Recent mission results for feed |
| `/v1/wallet/summary` | GET | NEW | Yes | Combined SEED+BLOOM+IMPT snapshot |
| `/v1/agents/roster` | GET | NEW | Yes | PAT-7 + SAT-5 status grid |

**Note:** `api.py` contains 52+ endpoints (3,581 lines). The table above shows only
endpoints consumed by Surfaces A-C. Full reference: `core/sovereign/api.py`.

## Module Map

| Spec | File | Target |
|---|---|---|
| 73.01 | `phase_73_01_shared_contracts.md` | TypeScript types, API client, design tokens |
| 73.02 | `phase_73_02_website.md` | Public trust site (6 pages) |
| 73.03 | `phase_73_03_onboarding.md` | Consumer + contributor onboarding flows |
| 73.04 | `phase_73_04_dashboard.md` | Daily Loop dashboard shell |
| 73.05 | `phase_73_05_testing.md` | Test strategy, a11y gates, performance budget |

## Success Criteria

1. Public site loads in < 3s on 3G connection
2. Onboarding completion rate > 85% in usability tests
3. Time to first value < 15 minutes
4. Dashboard first meaningful paint < 1s (cached), < 3s (cold)
5. WCAG 2.1 AA — zero violations on axe-core audit
6. All routes lazy-loaded, total JS bundle < 200KB gzipped
7. Arabic layout renders correctly (RTL, Noto Sans Arabic)
8. All data displayed traces to a named API endpoint (no invented data)

## What This Phase Does NOT Do

- Does not build the contributor desktop client (Surface D)
- Does not build the operator console (Surface E)
- Does not implement backend endpoints (those are separate tickets)
- Does not design new visual language (uses existing design tokens)
- Does not implement trading, governance, or BIZRAverse surfaces
