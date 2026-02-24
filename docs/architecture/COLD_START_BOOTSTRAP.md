# Cold-Start Ecosystem Bootstrap Architecture

> **Version:** 1.0.0 | **Phase:** 57 | **Date:** 2026-02-22
> **Standing on Giants:** Nakamoto (2008) Genesis Block + General Magic (1990) TeleScript + WoW Character Templates

## Executive Summary

The Cold-Start Ecosystem Bootstrap solves the fundamental question: **what happens when a new user joins BIZRA with zero conversation history, zero custom models, and has never used AI?**

**BIZRA** (بذرة) is a **Proactive Dynamic DDAGI OS** — where every human is a node, every node is a seed, and every seed has infinite potential. It is not a chatbot. It is a sovereign agentic ecosystem with a dual token economy, desktop automation, 12 AI agents per user, and constitutional governance. The bootstrap architecture ensures the full flywheel spins from message #1.

**Design Principle:** Every seed starts with infinite potential. Every new node inherits the same template (7 PAT + 5 SAT agents), the same constitutional gates, and the same token economy. Conversation exports are accelerators, not requirements. The bootstrap unlocks that potential from the very first interaction.

## Architecture Overview

```
NEW USER ARRIVES (cold-start)
       |
       v
+---------------------------------------------+
|  MMORPG Node Template Clone (Task 6)        |
|  7 PAT + 5 SAT + Token wallet +             |
|  Constitutional gates + Event Bus            |
+---------+--------------+--------------------+
          |              |
    +-----+-----+  +----+------+  +-----------+
    | Onboarding |  | SEED     |  | Bootstrap |
    | Interview  |  | Grant    |  | Reflexes  |
    | (25 atoms) |  | (tokens) |  | (speed)   |
    +-----+------+  +----+-----+  +-----+-----+
          |              |              |
          +--------------+--------------+
                         |
                         v
+---------------------------------------------+
|  Tiered Genesis Gate (Task 2)               |
|  Seed -> Sprout -> Growing -> Rooted        |
+---------+--------------+--------------------+
          |              |
    +-----+-----+  +----+------+  +-----------+
    | Self-Comp |  | AHK+HDA  |  | KnowsMe  |
    | Loop      |  | Actions  |  | Gauge     |
    | (learns)  |  | (does)   |  | (shows)   |
    +-----+------+  +----+-----+  +-----+-----+
          |              |              |
          +--------------+--------------+
                         |
                         v
+---------------------------------------------+
|  Atlas Quality Transparency (Task 9)        |
|  User sees tier + growth path +             |
|  what unlocks next                          |
+---------------------------------------------+
                         |
                         v
+---------------------------------------------+
|  FLYWHEEL SPINS:                            |
|  Use BIZRA -> earn SEED -> agents learn     |
|  -> reflexes compile -> actions succeed     |
|  -> receipts prove -> IMPT grows ->         |
|  -> unlock capabilities -> more value       |
|  -> network effects (SAT) -> more users     |
|  -> Resource Pool grows -> costs drop       |
+---------------------------------------------+
```

## Component Map

| # | Component | Layer | Files | Tests |
|---|-----------|-------|-------|-------|
| 1 | Self-Compilation Loop | Rust + Python | `bridge.rs`, `engine.py`, `node.rs` | 12 Rust |
| 2 | Tiered Genesis Gate | Python | `genesis_gate.py`, `compile_stereoscopic_graph.py` | Existing gate suite |
| 3 | Onboarding Interview | React (JSX) | `TeachStep.jsx` | Frontend |
| 4 | KnowsMeGauge | React (JSX) | `App.jsx` | Frontend |
| 5 | Bootstrap Reflexes | Rust | `reflex_cache.rs` | 5 Rust |
| 6 | Node Template | Python | `onboarding.py` | Existing onboarding suite |
| 7 | Token Genesis Grant | Python | `ledger.py`, `onboarding.py` | Existing ledger suite |
| 8 | AHK Action Loop | AHK + Python + Rust | `ahk_bridge.ahk`, `desktop_bridge.py`, `action_types.rs` | 33 + 26 |
| 9 | Atlas Quality Tiers | React + Python | `App.jsx`, `DashboardStep.jsx`, `atlas_gap_report.py` | 16 Python |
| 10 | Custom Providers | Python | `normalizers/base.py`, `normalizers/__init__.py` | Existing normalizer suite |

## Tier Progression Model

```
Tier          min_cv  min_nodes  min_elite  Trigger
---------     ------  ---------  ---------  --------------------------
SEED          0.0     0          0          New user, zero data
SPROUT        0.0     1          0          10+ TEACH atoms
GROWING       0.5     3          1          100+ messages OR 1 import
ROOTED        1.0     5          1          3+ provider imports
FLOURISHING   --      --         3+         Multi-provider + elite
```

### Capability Unlock Matrix

| Tier | Capabilities Unlocked |
|------|----------------------|
| Seed | Chat, TEACH |
| Sprout | + Memory recall, + Bootstrap reflexes |
| Growing | + Reflex compilation, + Action Bus (ToolCall) |
| Rooted | + Desktop actions (AHK), + Token economy |
| Flourishing | + Full Action Bus, + Agent-as-Service |

### Atlas Priority Mapping

| Atlas Priority | User Tier | Description |
|---------------|-----------|-------------|
| P0 | Seed | Core infrastructure, always available |
| P1 | Growing | Reflex compilation + tool calls |
| P2 | Rooted | Desktop actions + token economy |
| P3+ | Flourishing | Full action bus + agent-as-service |

## Constitutional Thresholds

All thresholds are defined in `core/integration/constants.py` (single source of truth):

| Constraint | Value | Purpose |
|-----------|-------|---------|
| Ihsan (excellence) | >= 0.95 | Quality floor for all operations |
| SNR (signal quality) | >= 0.85 | Minimum signal-to-noise ratio |
| ADL Gini (justice) | <= 0.40 | Anti-concentration gate |

## Data Flow

```
                    TEACH atoms (onboarding)
                           |
                           v
+----------+    +-------------------+    +------------------+
| AHK      | -> | Memory Pipeline   | -> | Stereoscopic     |
| Desktop  |    | (Rust: bizra-     |    | Engine           |
| Actions  |    |  memory)          |    | (Python: engine) |
+----------+    +-------------------+    +------------------+
     ^                  |                        |
     |                  v                        v
+----------+    +-------------------+    +------------------+
| Action   | <- | Self-Compilation  | <- | Identity         |
| Bus      |    | Loop (every 50    |    | Signals          |
| (Permit) |    |  commands)        |    | (SNR scored)     |
+----------+    +-------------------+    +------------------+
     |                                           |
     v                                           v
+----------+    +-------------------+    +------------------+
| Receipt  | -> | Event Bus         | -> | Reflex           |
| Chain    |    | (PostDeliver      |    | Compiler         |
|          |    |  hooks)           |    | (Pattern->Rule)  |
+----------+    +-------------------+    +------------------+
```

## Verification Checklist

- [x] Cold-start smoke: New user -> onboarding -> first message -> response -> `knows_me_score > 0`
- [x] Self-compilation: 50 messages -> stereoscopic compiler produces >= 1 signal node
- [x] Gate progression: User starts at Seed tier, gate passes, progresses through tiers
- [x] Bootstrap reflex: Greeting routed to Diplomat without full orchestration
- [x] Token grant: New node receives initial SEED, can query balance
- [x] Action loop: AHK executes click -> screenshot verifies -> receipt seals with outcome_hash
- [x] Atlas tier: User at Growing tier sees correct capability set unlocked
- [x] All existing tests pass: 7,224 Python + 1,000+ Rust, zero regressions

## What Was NOT Changed

- SAP v0 protocol, receipt chain format, Ihsan gate thresholds
- HHMM temporal layers, existing 10 provider parsers
- Genesis seed format (TEACH protocol is the universal bootstrap)
- TeleScript 9 primitives, Event Bus architecture, Action Bus channels
- Token constants (ZAKAT_RATE, HARBERGER_TAX_RATE, supply caps)
