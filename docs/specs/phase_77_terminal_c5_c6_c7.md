# Phase 77 — Terminal v1 Views C.5, C.6, C.7

**Date:** 2026-03-08
**Depends on:** Phase 76 (C.1–C.4 locked), Build Contract v1.0
**Scope:** Three remaining terminal views: Agents (C.5), Network (C.6), Settings (C.7)

---

## §1 — Source-of-Truth Matrix (Contract §4 Extension)

Each view binds to specific backend endpoints. No view invents state.

### C.5 Agents

| Data | Endpoint | Auth | Interval |
|------|----------|------|----------|
| Cognitive status | `GET /v1/cognitive/status` | PUBLIC | 30s |
| SAT stats | `GET /v1/sat/stats` | PUBLIC | 60s |
| Node value (5-factor KPI) | `GET /v1/node/value` | AUTH | 60s |
| Seed potential (reflexes) | `GET /v1/seed/potential` | AUTH | 30s |
| Constitutional status (reflexes count) | `GET /v1/constitutional/status` | AUTH | 15s |

### C.6 Network

| Data | Endpoint | Auth | Interval |
|------|----------|------|----------|
| Network effect | `GET /v1/network/effect?nodes=N` | AUTH | 120s |
| Network milestones | `GET /v1/network/milestones` | AUTH | 300s |
| Token supply | `GET /v1/token/supply` | PUBLIC | 120s |
| Constitutional status (Gini, asabiyyah) | `GET /v1/constitutional/status` | AUTH | 15s |
| PoI stats | `GET /v1/poi/stats` | PUBLIC | 120s |

### C.7 Settings

| Data | Endpoint | Auth | Interval |
|------|----------|------|----------|
| User profile | `GET /v1/auth/me` | AUTH | 60s |
| Lifecycle stage | `GET /v1/node/lifecycle` | AUTH | 60s |
| Token balance | `GET /v1/token/balance` | AUTH | 60s |
| Health deep | `GET /v1/health/deep` | PUBLIC | 30s |
| Circuit breaker | `sovereign.circuitState()` | LOCAL | — |

---

## §2 — C.5 Agents View

### §2.1 Purpose

Show the user's agentic team — what's active, what's idle, what's been compiled.
This is the **capability surface** of the terminal.

### §2.2 Layout (Three Panels)

```
┌──────────────────────────────────────────────────────────┐
│ AGENTS                                                    │
├──────────────────────────────────────────────────────────┤
│ ┌─────────────────────┐ ┌──────────────────────────────┐ │
│ │ Cognitive Overview   │ │ PAT-7 (Your Team)            │ │
│ │                      │ │                              │ │
│ │ Active agents: N     │ │ Atlas   — Planner   ● ACTIVE │ │
│ │ Memory usage: X MB   │ │ Oracle  — Research  ● IDLE   │ │
│ │ Status: healthy      │ │ Forge   — Code      ● ACTIVE │ │
│ │                      │ │ Judge   — Evaluate  ○ IDLE   │ │
│ │ ┌──────────────────┐ │ │ Crown   — Ethics    ○ IDLE   │ │
│ │ │ Reflex Inventory │ │ │ Herald  — Publish   ○ IDLE   │ │
│ │ │ Compiled: N      │ │ │ JARVIS  — Integrate ● ACTIVE │ │
│ │ │ Near-compile: M  │ │ │                              │ │
│ │ │ Total exec: K    │ │ ├──────────────────────────────┤ │
│ │ └──────────────────┘ │ │ SAT-5 (Forest Pool)          │ │
│ │                      │ │                              │ │
│ │ Node Quality         │ │ Sentinel  — Security  ● POOL │ │
│ │ ═══════════ 0.87     │ │ Oracle-S  — Balance   ● POOL │ │
│ │ composite score      │ │ Ledger    — Trust     ● POOL │ │
│ │                      │ │ Conductor — Capacity  ● POOL │ │
│ │                      │ │ Ambassador— Social    ● POOL │ │
│ └─────────────────────┘ └──────────────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│ Reflex Detail (if any compiled)                          │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Pattern: "code-review"  │ Ihsān: 0.97  │ Exec: 42   │ │
│ │ Pattern: "summarize"    │ Ihsān: 0.96  │ Exec: 31   │ │
│ │ Pattern: "deploy-check" │ Ihsān: 0.93  │ Near: 2/3  │ │
│ └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### §2.3 Data Binding

```pseudocode
FUNCTION AgentsView():
  cognitive ← useCognitiveStatus()     // active_agents, memory_usage_mb
  nodeValue ← useNodeValue()           // composite, quality, tier, human_stage
  seed      ← useSeedPotential()       // compiled, streak, weakest_dimension
  constit   ← useConstitutionalStatus() // reflexes count

  // PAT-7 agent roster — static definition, dynamic status from cognitive
  PAT_AGENTS = [
    { name: "Atlas",   role: "Planner",    icon: Compass },
    { name: "Oracle",  role: "Researcher", icon: Search },
    { name: "Forge",   role: "Coder",      icon: Hammer },
    { name: "Judge",   role: "Evaluator",  icon: Scale },
    { name: "Crown",   role: "Ethicist",   icon: Shield },
    { name: "Herald",  role: "Publisher",   icon: Megaphone },
    { name: "JARVIS",  role: "Integrator", icon: Cpu },
  ]

  SAT_AGENTS = [
    { name: "Sentinel",   role: "Security" },
    { name: "Oracle-S",   role: "Balance" },
    { name: "Ledger",     role: "Trust" },
    { name: "Conductor",  role: "Capacity" },
    { name: "Ambassador", role: "Social" },
  ]

  // Agent activity estimation from cognitive.active_agents
  // If backend provides individual agent status later, wire here
  activeCount = cognitive.data?.active_agents ?? 0
  FOR i IN 0..PAT_AGENTS.length:
    PAT_AGENTS[i].status = (i < activeCount) ? "active" : "idle"

  // SAT agents always show "pool" — they serve the forest
  FOR agent IN SAT_AGENTS:
    agent.status = "pool"

  RENDER CognitiveOverview(cognitive, nodeValue)
  RENDER AgentRoster(PAT_AGENTS, SAT_AGENTS)
  RENDER ReflexInventory(seed, constit)
```

### §2.4 Constraints

- **No synthetic agent activity**: Agent status derived from `cognitive.active_agents` count, not invented per-agent state
- **Reflex data from real endpoints**: `seed.compiled`, `constit.reflexes`, `seed.weakest_dimension`
- **PAT/SAT names are static UI labels** — the backend doesn't expose individual agent names yet. This is acceptable because the roster IS the specification (البذرة p.12)
- **Offline fallback**: Show roster with all agents "unknown" and "(offline)" label
- **No agent management actions**: C.5 is read-only. Agent configuration is C.7 territory

### §2.5 Acceptance Criteria

1. CognitiveOverview shows active_agents count, memory_usage_mb, status
2. PAT-7 roster renders all 7 agents with name, role, icon, status indicator
3. SAT-5 roster renders all 5 agents with "pool" status
4. Node quality composite score displayed with tier badge
5. Reflex inventory shows compiled count, near-compile count, total executions
6. If seed.compiled is true, show compiled reflex detail section
7. Offline state labeled explicitly — no silent degradation

---

## §3 — C.6 Network View

### §3.1 Purpose

Show the forest's health and the user's position within it.
This is the **collective surface** of the terminal.

### §3.2 Layout (Three Sections)

```
┌──────────────────────────────────────────────────────────┐
│ NETWORK                                                   │
├──────────────────────────────────────────────────────────┤
│ Forest Health                                             │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│ │ Nodes    │ │ Gini     │ │ Asabiyyah│ │ SEED     │     │
│ │ 1        │ │ 0.12     │ │ 0.85     │ │ Supply   │     │
│ │ (seed)   │ │ (healthy)│ │ (strong) │ │ 100,000  │     │
│ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
├──────────────────────────────────────────────────────────┤
│ Network Effect Projection                                 │
│                                                           │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ At 1 node:     Skills: 7   │ Compute: 0.1 TFLOPS    │ │
│ │ At 100 nodes:  Skills: 47  │ Compute: 10 TFLOPS     │ │
│ │ At 10K nodes:  Skills: 200 │ Compute: 1,000 TFLOPS  │ │
│ │ At 1M nodes:   Skills: 900 │ Compute: 100K TFLOPS   │ │
│ │                                                       │ │
│ │ Intelligence density: log(N)                          │ │
│ │ Cost per node: decreasing with N                      │ │
│ └──────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│ Milestones                                                │
│ ┌──────────────────────────────────────────────────────┐ │
│ │  ✓ 1 node      — Genesis ceremony complete           │ │
│ │  ○ 10 nodes    — First mesh gossip                   │ │
│ │  ○ 100 nodes   — Reflex marketplace active           │ │
│ │  ○ 1,000 nodes — Regional federation                 │ │
│ │  ○ 10K nodes   — Global consensus                    │ │
│ └──────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│ Token Economy                                             │
│ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ │
│ │ Total     │ │ Circulate │ │ Burned    │ │ Zakat Pool│ │
│ │ 100,000   │ │ 97,500    │ │ 0         │ │ 2,500     │ │
│ └───────────┘ └───────────┘ └───────────┘ └───────────┘ │
└──────────────────────────────────────────────────────────┘
```

### §3.3 Data Binding

```pseudocode
FUNCTION NetworkView():
  effect     ← useNetworkEffect(1000)  // current projection
  milestones ← useNetworkMilestones()  // growth targets
  supply     ← useTokenSupply()        // total/circulating/burned/zakat
  constit    ← useConstitutionalStatus() // Gini, asabiyyah

  // Forest health cards — color-coded
  giniColor = constit.network_gini <= 0.20 ? "green"
            : constit.network_gini <= 0.35 ? "yellow"
            : "red"

  asabiyyahColor = constit.network_asabiyyah >= 0.70 ? "green"
                 : constit.network_asabiyyah >= 0.50 ? "yellow"
                 : "red"

  // Multi-scale projection — call effect at 4 scales
  // Use single endpoint with different N values
  // Note: avoid 4 parallel calls — compute locally from effect formula
  // Backend returns for requested N; client extrapolates milestones
  scales = [1, 100, 10_000, 1_000_000]
  projections = scales.map(n => ({
    nodes: n,
    skills: effect.data ? Math.round(effect.data.skills_available * Math.log10(n) / Math.log10(1000)) : "—",
    compute: effect.data ? (effect.data.compute_tflops * n / 1000).toFixed(1) : "—",
  }))

  // Milestone progress — mark achieved based on current node count
  currentNodes = effect.data?.nodes ?? 1
  FOR milestone IN milestones.data:
    milestone.achieved = (currentNodes >= milestone.nodes)

  RENDER ForestHealth(constit, currentNodes, giniColor, asabiyyahColor)
  RENDER NetworkProjection(projections, effect)
  RENDER MilestoneTracker(milestones)
  RENDER TokenEconomy(supply)
```

### §3.4 Constraints

- **No synthetic node counts**: Node count from backend. If 1, show 1.
- **Projections are labeled as projections**: "At N nodes (projected)" — not "N nodes exist"
- **Gini and asabiyyah from constitutional status**: Same source as dashboard — no double-polling (share hook)
- **Token supply is public**: No auth required. Always available.
- **Milestone achieved state from backend**: Don't invent completion status
- **Offline fallback**: Show "Network data unavailable — operating in sovereign mode"

### §3.5 Acceptance Criteria

1. Forest health shows node count, Gini (color-coded ≤0.20/≤0.35/>0.35), asabiyyah (color-coded), SEED supply
2. Network projection table shows 4 scale points with skills, compute, intelligence density
3. Milestone tracker shows achieved/pending milestones with descriptions
4. Token economy shows total_supply, circulating, burned, zakat_pool
5. Gini breach triggers same critical event as dashboard (shared constitutional polling)
6. Offline state explicitly labeled

---

## §4 — C.7 Settings View

### §4.1 Purpose

Show sovereignty configuration and system health.
This is the **governance surface** of the terminal.

### §4.2 Layout (Four Sections)

```
┌──────────────────────────────────────────────────────────┐
│ SETTINGS                                                  │
├──────────────────────────────────────────────────────────┤
│ Identity                                                  │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Username: mumo           │ Tier: SEED               │ │
│ │ Stage: Seed              │ Score: 0.18              │ │
│ │ Created: 2026-01-15      │ Node ID: ed25519:a7f...  │ │
│ │                                                      │ │
│ │ Progress to Node ═════════░░░░░░░░░░░░░░░░░ 18%     │ │
│ │ Unlock: Complete first mission with Ihsān ≥ 0.85     │ │
│ └──────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│ Wallet                                                    │
│ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ │
│ │ SEED      │ │ Pending   │ │ Earned    │ │ Zakat     │ │
│ │ 975.00    │ │ 0.00      │ │ 1,000.00  │ │ 25.00     │ │
│ └───────────┘ └───────────┘ └───────────┘ └───────────┘ │
├──────────────────────────────────────────────────────────┤
│ System Health                                             │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ API Status: ● Connected    Circuit: CLOSED           │ │
│ │ Backend:    ● Healthy      Latency: ~45ms            │ │
│ │                                                      │ │
│ │ Subsystems:                                          │ │
│ │   evidence_ledger:  ● healthy  │ 12ms                │ │
│ │   snr_maximizer:    ● healthy  │ 8ms                 │ │
│ │   guardian_council:  ● healthy  │ 15ms                │ │
│ │   seed_engine:      ● active   │ 4 episodes          │ │
│ └──────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│ Session                                                   │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ [Sign Out]                                           │ │
│ │                                                      │ │
│ │ Terminal v1 — Build Contract locked                   │ │
│ │ Frontend: award-winner-design                        │ │
│ │ Backend: sovereign-api                               │ │
│ └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### §4.3 Data Binding

```pseudocode
FUNCTION SettingsView():
  profile   ← useAuthMe()            // username, tier, sovereignty_score, created_at
  lifecycle ← useLifecycleStage()     // current_stage, progress, next_stage, unlock_condition
  balance   ← useTokenBalance()       // balance, pending, total_earned, zakat_contributed
  healthD   ← useHealthDeep()         // subsystems with status + latency
  health    ← useSovereignHealth()    // seed_engine status

  circuitState = sovereign.circuitState()  // local circuit breaker
  isAuth       = sovereign.isAuthenticated()

  RENDER IdentityCard(profile, lifecycle)
  RENDER WalletOverview(balance)
  RENDER SystemHealth(healthD, health, circuitState)
  RENDER SessionInfo(isAuth)
```

### §4.4 New Hooks Required

```pseudocode
// Need to add to use-sovereign-api.ts:
FUNCTION useAuthMe():
  RETURN useSovereignPoll(sovereign.auth.me, 60_000)

FUNCTION useHealthDeep():
  RETURN useSovereignPoll(sovereign.healthDeep, 30_000)
```

### §4.5 Sign Out Action

```pseudocode
FUNCTION handleSignOut():
  sovereign.logout()        // clears localStorage tokens
  window.location.reload()  // clean state reset
```

### §4.6 Constraints

- **Identity from auth/me**: No synthetic user profiles
- **Lifecycle from node/lifecycle**: Shows real progression with unlock conditions
- **No settings mutation in v1**: Settings view is read-only. Configuration changes are post-v1
- **Sign out is the only action**: Clears JWT tokens, reloads page
- **Subsystem health from /health/deep**: Latency per subsystem, status per subsystem
- **Circuit breaker state is local**: From the client's CircuitBreaker instance, not backend
- **Offline fallback**: Show cached identity + "Backend unreachable" in system health

### §4.7 Acceptance Criteria

1. Identity card shows username, tier badge, sovereignty score, created date
2. Lifecycle progress bar with percentage and next-stage unlock condition
3. Wallet shows SEED balance, pending, total earned, zakat contributed
4. System health shows API connection status, circuit breaker state
5. Subsystem detail from /health/deep with per-subsystem status + latency
6. Seed engine status (active/inactive, episode count, streak)
7. Sign out button clears auth and reloads
8. Offline state shows cached identity with explicit "disconnected" label

---

## §5 — New Client Endpoints Required

Add to `lib/sovereign-client.ts`:

```pseudocode
// Already exists but no hook:
sovereign.auth.me()        → UserProfile
sovereign.healthDeep()     → HealthDeep

// May need for C.6 (check if backend exposes):
sovereign.poi.stats()      → PoIStats    // GET /v1/poi/stats [PUBLIC]
```

Add to `hooks/use-sovereign-api.ts`:

```pseudocode
useAuthMe()          → useSovereignPoll(sovereign.auth.me, 60_000)
useHealthDeep()      → useSovereignPoll(sovereign.healthDeep, 30_000)
useCognitiveStatus() → already exists
useNetworkEffect()   → already exists
useNetworkMilestones() → already exists
useTokenSupply()     → already exists
useTokenBalance()    → already exists
useLifecycleStage()  → already exists
```

### §5.1 PoI Stats Type (if backend exposes)

```pseudocode
INTERFACE PoIStats:
  total_contributions: number
  total_impact_score: number
  active_contributors: number
  recent_receipts: number
```

---

## §6 — Store Extensions

No new store state required. Existing `useTerminalStore` covers:
- `activeView` — already has "agents", "network", "settings" as valid values
- `criticalEvents` — shared across all views
- `isConnected` — shared circuit breaker state
- `briefing` — session context (used by Memory, available to all)

---

## §7 — Build Order

```
Step 1: Add useAuthMe() and useHealthDeep() hooks
Step 2: Build C.5 Agents (components/terminal/terminal-agents.tsx)
Step 3: Wire into shell, typecheck, build
Step 4: Build C.6 Network (components/terminal/terminal-network.tsx)
Step 5: Wire into shell, typecheck, build
Step 6: Build C.7 Settings (components/terminal/terminal-settings.tsx)
Step 7: Wire into shell, typecheck, build
Step 8: Remove ViewPlaceholder function (no longer needed)
Step 9: Final typecheck + build — all 7 views live
```

---

## §8 — Exit Criteria

Terminal v1 is COMPLETE when:

1. All 7 views render real data from backend endpoints
2. All 7 views have explicit offline fallback labeling
3. Zero ViewPlaceholder components remain
4. Typecheck: 0 errors
5. Build: clean production build
6. No synthetic state — every rendered value traces to an endpoint or local computation
7. Critical events visible across all views (Contract §7.3)
8. Two-touch law respected in Mission view (Contract §2)
9. Circuit breaker state visible in Settings (Contract §12)

---

## §9 — What This Spec Does NOT Cover

- Agent management / configuration UI (post-v1)
- Network peer discovery / gossip visualization (post-v1)
- Settings mutation / preference editing (post-v1)
- Chat / command interface within agents (post-v1)
- Real-time WebSocket streaming (post-v1, uses /v1/stream)
- Mobile responsive layout (post-v1)
- Accessibility audit (post-v1, but all views use semantic HTML)
