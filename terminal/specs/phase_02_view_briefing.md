# Phase 02 — View 1: BRIEFING (Home)

> **Purpose:** Morning briefing from JARVIS. The first thing the user sees.
> **Status:** PARTIAL — sovereign_terminal.py has morning_briefing(), Rust TUI missing this view.

## 2.1 Content Spec

```
┌──────────────────────────────────────────────────────────────┐
│  JARVIS — Good morning, Mumo.                                │
│                                                               │
│  Last session: 4h 23m ago                                    │
│  Last mission: "organize invoices" — Ihsan 0.96 — +2.38 SEED│
│  Active project: BIZRA Frontend Wallet                       │
│                                                               │
│  Near-compiling reflexes:                                    │
│    "file organization" — 2/3 qualified (1 more to compile)   │
│                                                               │
│  Quality trend (last 10):                                    │
│    ■■■■■■■■□□  Ihsan avg: 0.943 (↑ 0.012)                  │
│                                                               │
│  Wallet: 42.5 SEED | 1.23 BLOOM | Tier: SPROUT              │
│                                                               │
│  Suggested: "Review PR #47 — security headers"               │
└──────────────────────────────────────────────────────────────┘
```

## 2.2 Data Requirements

```pseudocode
struct BriefingData:
    greeting: str             # Time-aware: "Good morning/afternoon/evening"
    node_name: str            # From sovereign_state/mission_signer.json
    time_since_last: Duration # Current time - last session timestamp
    last_mission: {
        intent: str
        ihsan: float
        seed_earned: float
        timestamp: str
    }
    active_project: str       # From Living Memory semantic profile
    near_compiling: [{
        pattern: str
        current_count: int
        threshold: int        # Always 3
        avg_ihsan: float
    }]
    quality_trend: {
        scores: float[10]     # Last 10 Ihsan scores
        avg: float
        delta: float          # Change from previous 10
    }
    wallet_compact: {
        seed: float
        bloom: float
        tier: str
    }
    suggested_mission: str    # From proactive recommendation engine

function fetch_briefing(node_state, api_client) -> BriefingData:
    identity = read_node_identity("sovereign_state")

    TRY:
        health = api_client.get("/v1/health")
        balance = api_client.get("/v1/token/balance")
        potential = api_client.get("/v1/seed/potential")
        episodes = api_client.get("/v1/seed/episodes?limit=10")
        mode = "live"
    CATCH:
        // Offline fallback
        health = None
        balance = derive_from(node_state)
        potential = derive_from(node_state)
        episodes = read_local_episodes()
        mode = "offline"

    RETURN BriefingData(
        greeting = time_greeting(),
        node_name = identity.node_id.split("-")[0].title(),
        time_since_last = now() - last_session_timestamp(),
        last_mission = episodes[0] if episodes else None,
        near_compiling = scan_reflex_candidates(),
        quality_trend = compute_trend(episodes),
        wallet_compact = {seed: balance.seed, bloom: balance.bloom, tier: potential.tier},
        suggested_mission = proactive_suggest(episodes, identity),
    )
```

## 2.3 Existing Implementation

**Python (sovereign_terminal.py:245-296):** `morning_briefing()` — Has greeting, uptime, Ihsan, SEED/BLOOM, reflex count, tips. Missing: last mission, quality trend, near-compiling, suggested mission.

**Rust (app.rs):** No briefing view exists. Dashboard tab shows agent cards + FATE gauges.

## 2.4 What to Build

| Component | Surface | LOC Est | Priority |
|-----------|---------|---------|----------|
| `BriefingData` struct | Rust | 30 | P0 |
| `fetch_briefing()` API call | Rust | 60 | P0 |
| `render_briefing()` widget | Rust | 120 | P0 |
| Upgrade `morning_briefing()` | Python | 40 | P1 |
| Quality trend sparkline | Both | 30 | P1 |
| Proactive suggestion engine | Python | 80 | P2 |

## 2.5 TDD Anchors

```
TEST: briefing_renders_offline
  GIVEN backend unreachable
  WHEN briefing view requested
  THEN shows greeting, offline wallet, "Backend offline" note

TEST: briefing_shows_quality_trend
  GIVEN 10 episodes with known Ihsan scores
  WHEN briefing renders
  THEN sparkline shows correct bars, avg matches

TEST: briefing_near_compiling_accuracy
  GIVEN pattern "file.move" with 2 qualified executions
  WHEN briefing renders
  THEN shows "2/3 qualified (1 more to compile)"

TEST: briefing_time_greeting
  GIVEN time = 08:00
  THEN greeting = "Good morning"
  GIVEN time = 14:00
  THEN greeting = "Good afternoon"
  GIVEN time = 20:00
  THEN greeting = "Good evening"
```
