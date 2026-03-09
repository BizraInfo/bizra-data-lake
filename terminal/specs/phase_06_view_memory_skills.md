# Phase 06 — View 5: MEMORY + View 6: SKILLS

> **Purpose:** Memory = what the system knows about the user. Skills = MMORPG progression.
> **Status:** NOT BUILT — data models exist in backend, no terminal rendering.

## 6.1 MEMORY View Content

```
┌── MEMORY ─────────────────────────────────────────────────────┐
│                                                                │
│  Semantic Profile                                             │
│    Preferred domains:  security, infrastructure, tokenomics   │
│    Active hours:       09:00-14:00, 21:00-01:00 GST          │
│    Communication:      Direct, technical, Arabic-first        │
│                                                                │
│  Active Projects                                              │
│    BIZRA Frontend Wallet    last: 2h ago                      │
│    Sovereign Terminal       last: today                       │
│    Constitutional Kernel    last: 3 days ago                  │
│                                                                │
│  Last 10 Missions                                             │
│    #147  organize invoices      0.96  +2.38  QUALIFIED       │
│    #146  review auth PR         0.97  +1.85  QUALIFIED       │
│    #145  fix CORS headers       0.88  +0.00  BELOW FLOOR     │
│    ... (scrollable)                                           │
│                                                                │
│  Stats                                                        │
│    Streak: 7 days | Preferred Agent: Oracle (38% of missions) │
│    Strongest: Security | Weakest: Frontend (growth opp.)      │
│                                                                │
│  Privacy: All data is local. Nothing leaves your device.      │
└──────────────────────────────────────────────────────────────┘
```

## 6.2 SKILLS View Content

```
┌── SKILLS ─────────────────────────────────────────────────────┐
│                                                                │
│  Stage: BUILDER (0.45)  ████████████████████░░░░░░  45%       │
│  Tier:  SPROUT           Next: TREE at 0.50 (+0.05)          │
│                                                                │
│  ── Compiled Reflexes (System-1) ─────────────────────────    │
│  invoice-org    Ihsan: 0.943  x12 runs  avg: 48ms   [LIVE]   │
│  git-pr-review  Ihsan: 0.971  x8 runs   avg: 52ms   [LIVE]   │
│  test-runner    Ihsan: 0.922  x5 runs   avg: 31ms   [LIVE]   │
│                                                                │
│  ── Near Compilation (N/3) ───────────────────────────────    │
│  security-scan  2/3  Ihsan avg: 0.96  (1 more qualified)     │
│  deploy-k8s     1/3  Ihsan avg: 0.91  (2 more qualified)     │
│                                                                │
│  ── Skill Tree ───────────────────────────────────────────    │
│  T0 Novice:    [x] Clipboard  [x] Screen Cap  [x] Focus     │
│  T1 Apprentice:[x] Mouse      [x] Keyboard    [x] App Launch│
│  T2 Adept:     [x] File Read  [x] File Write  [x] Window Mgmt│
│  T3 Expert:    [ ] PowerShell  [ ] Multi-Step  [ ] Cross-App │
│  T4 Master:    [ ] Network    [ ] Governance   [ ] Federation│
│  T5 Grandmaster:[ ] Self-Mod  [ ] Validator    [ ] Fed Leader│
│                                                                │
│  ── Human Lifecycle ──────────────────────────────────────    │
│  Seed > Node > Apprentice > [BUILDER] > Verifier > Mentor > Cat│
│  ░░░░░░░████████████████████████████░░░░░░░░░░░░░░░░░░░░░    │
│                                     ^ You are here (0.45)     │
└──────────────────────────────────────────────────────────────┘
```

## 6.3 Data Model

```pseudocode
struct MemoryView:
    semantic_profile: {
        preferred_domains: str[]
        active_hours: str
        communication_pref: str
    }
    active_projects: [{name: str, last_activity: str}]
    recent_missions: [{index: int, summary: str, ihsan: float, seed: float, qualified: bool}]
    streak: int
    preferred_agent: {name: str, percentage: float}
    strongest_domain: str
    weakest_domain: str

struct SkillsView:
    // Progression
    stage: str              # Human lifecycle stage name
    sovereignty: float      # 0-1
    stage_progress: float   # Within current stage
    tier: str               # SEED/SPROUT/TREE/FOREST
    points_to_next: float

    // Reflexes
    compiled_reflexes: [{
        pattern: str
        avg_ihsan: float
        execution_count: int
        avg_latency_ms: float
        status: "LIVE" | "DECAYING"
    }]
    near_compiling: [{
        pattern: str
        current: int
        threshold: int      # 3
        avg_ihsan: float
    }]

    // Skill tree (from action_schema_v1.json tier_progression)
    skill_tiers: [{
        tier: str           # novice/adept/expert/master
        skills: [{name: str, unlocked: bool}]
    }]

    // Lifecycle
    lifecycle_stages: [{
        name: str
        low: float
        high: float
        current: bool
    }]
```

## 6.4 Data Sources

| Data | API Endpoint | Offline Source |
|------|-------------|---------------|
| Semantic profile | `/v1/memory/profile` | Living Memory local files |
| Recent missions | `/v1/seed/episodes?limit=10` | Local EventBus chain |
| Reflexes | Reflex cache (local) | Same |
| Skill tree | Skill Tier calculator | action_schema_v1.json |
| Lifecycle | `/v1/node/lifecycle` | `human_lifecycle.py` |
| Node value | `/v1/node/value` | `node_value.py` |

## 6.5 Existing Implementation

**Python (sovereign_terminal.py:360-412):** `agents()` — PAT-7 + SAT-5 display. `reflexes()` — reflex cache table.

**Python (__main__.py):** `run_impact()` — sovereignty progression, UERS dimensions, achievements. Closest to Skills view.

**Rust (app.rs):** Agents tab shows PAT cards. No skills or memory view.

**Frontend (tokens.ts:86-103):** `STAGES` array with all 7 lifecycle stages + boundaries.

**Frontend (types.ts:112-120):** `Skill` interface with id, name, tier, icon, unlocked.

**action_schema_v1.json (267-271):** `tier_progression` — novice (0 actions), adept (10), expert (100), master (1000).

## 6.6 What to Build

| Component | Surface | LOC Est | Priority |
|-----------|---------|---------|----------|
| Memory profile widget | Rust | 100 | P1 |
| Recent missions list | Both | 60 | P1 |
| Compiled reflexes table | Rust | 80 | P0 |
| Near-compilation candidates | Both | 40 | P0 |
| Skill tree grid | Rust | 120 | P1 |
| Human lifecycle bar | Rust | 60 | P0 |
| Sovereignty gauge | Rust | 80 | P0 |

## 6.7 TDD Anchors

```
TEST: skills_lifecycle_stage_correct
  GIVEN sovereignty = 0.45
  THEN stage = "Builder" and progress within stage = (0.45-0.35)/(0.55-0.35) = 0.50

TEST: skills_tier_progression
  GIVEN 15 actions completed, avg ihsan 0.91
  THEN tier = "adept" (10 actions, ihsan >= 0.85)

TEST: skills_reflex_near_compiling
  GIVEN pattern with 2 qualified executions
  THEN shows "2/3" and highlights as near-compiling

TEST: memory_streak_count
  GIVEN missions on days [Mar 1, 2, 3, 5, 6, 7]
  THEN current streak = 3 (Mar 5-7, gap breaks at Mar 4)

TEST: memory_preferred_agent
  GIVEN 20 missions: 8 routed to Oracle, 5 to Forge, 7 to Atlas
  THEN preferred_agent = {name: "Oracle", percentage: 0.40}
```
