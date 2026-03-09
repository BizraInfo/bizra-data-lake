# Phase 01 — Terminal Architecture Overview

> **Locked:** 2026-03-07 | **Status:** Specification
> **Rule:** This is the single source of truth for the BIZRA Sovereign Terminal product.

## 1.1 Product Definition

The BIZRA Sovereign Terminal is THREE coordinated surfaces sharing one backend:

| Surface | Tech | Runtime | Status |
|---------|------|---------|--------|
| **Rust TUI** (primary) | ratatui 0.30 + crossterm 0.28 | Native binary | PARTIAL (3,210 LOC) |
| **Python REPL** (fallback) | Rich + argparse | `python -m core.sovereign` | PARTIAL (3,982 LOC) |
| **Web Terminal** (embedded) | React + xterm.js concept | Next.js / Vite | PARTIAL (279 LOC demo) |

**Total existing LOC:** 7,471 across 22 files.

## 1.2 Existing Inventory

### Rust TUI (bizra-cli) — 3,210 LOC

| File | LOC | What It Does | What's Missing |
|------|-----|-------------|----------------|
| `main.rs` | 823 | Event loop, render, keyboard | Wallet tab, receipts tab |
| `app.rs` | 534 | State machine, 6 tabs | Missing: Briefing, Receipts, Memory views |
| `theme.rs` | 283 | Dubai night palette | Complete |
| `config.rs` | 310 | Profile, identity, thresholds | Missing: API client config |
| `inference.rs` | 428 | LM Studio client | Complete |
| `commands/mod.rs` | 267 | status, info, agent list | Missing: wallet, receipts, skills |
| `widgets/header.rs` | 138 | ASCII logo, tabs | Needs 7-tab update |
| `widgets/status_bar.rs` | 77 | Mode indicator | Needs wallet summary |
| `widgets/agent_card.rs` | 138 | PAT agent display | Complete |
| `widgets/fate_gauge.rs` | 147 | FATE gate bars | Complete |

**Current tabs:** Dashboard, Agents, Chat, Tasks, Treasury, Settings
**Required tabs:** Briefing, Mission, Receipts, Wallet, Memory, Skills, System

### Python REPL (core/sovereign/) — 3,982 LOC

| File | LOC | What It Does | What's Missing |
|------|-----|-------------|----------------|
| `__main__.py` | 1,084 | Master CLI router, REPL | Missing: receipts, skills views |
| `agent_cli.py` | 730 | Agent kernel, MCP orchestration | Complete |
| `launch.py` | 443 | Production launcher | Complete |
| `runtime_cli.py` | 134 | Runtime-only CLI | Complete |
| constitutional `cli.py` | 501 | Logic layer (no I/O) | Complete |
| constitutional `__main__.py` | 286 | Sovereignty CLI | Complete |
| `cli_bridge.py` | 336 | Inference HTTP bridge | Complete |
| genesis `cli.py` | 224 | Node bootstrap | Complete |
| memory `__main__.py` | 64 | Memory migration | Complete |
| `sovereign_terminal.py` | 681 | Rich terminal UI | Missing: live API wiring |

### Terminal Spec Files (terminal/) — Already Written

| File | LOC | Purpose |
|------|-----|---------|
| `terminal_information_architecture.md` | 257 | 7-view IA spec |
| `node0_terminal_mission_loop.md` | 322 | Mission loop spec |
| `event_schema_v1.json` | 272 | 40 event types |
| `action_schema_v1.json` | 282 | 15 action types + gates |
| `economic.ts` | 237 | Unified EconomicReceipt type |
| `bloom.py` | 454 | BLOOM + CommunityPool + Gini |
| `subscribers.py` | 811 | 12 EventBus subscribers |
| `useWallet.ts` | — | Wallet hook (copy from frontend) |
| `wallet-hardening.test.ts` | — | Wallet tests (copy from frontend) |

## 1.3 Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    USER                                  │
│         (keyboard, one-line mission)                     │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
┌────────────┐ ┌──────────┐ ┌───────────┐
│ Rust TUI   │ │ Python   │ │ Web       │
│ (ratatui)  │ │ REPL     │ │ Terminal  │
│ Port: —    │ │ Port: —  │ │ Port:3000 │
└─────┬──────┘ └────┬─────┘ └─────┬─────┘
      │             │             │
      └─────────────┼─────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │ Sovereign API    │
         │ Port: 8010       │
         │ 75 endpoints     │
         └────────┬─────────┘
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ EventBus │ │ SeedEng  │ │ Living   │
│ 12 subs  │ │ NodeVal  │ │ Memory   │
│ BLAKE2b  │ │ Lifecycle│ │ 3 stores │
└──────────┘ └──────────┘ └──────────┘
```

## 1.4 Data Sources (per view)

| View | Primary API | Offline Fallback | Polling |
|------|------------|-----------------|---------|
| Briefing | `/v1/health` + `/v1/seed/potential` + Living Memory | Local state files | On startup |
| Mission | `/v1/plan` + EventBus stream | Offline mission (degraded) | Real-time |
| Receipts | `/v1/seed/episodes` + local EventBus chain | Local chain only | On demand |
| Wallet | `/v1/token/balance` + `/v1/token/supply` + `/v1/seed/potential` | `deriveOffline(nodeState)` | 30s poll |
| Memory | `/v1/memory/profile` + Living Memory stores | Local files only | On demand |
| Skills | Skill Tier calculator + Reflex cache | Local procedural memory | On demand |
| System | `/v1/health` + Docker/K3d status | Local process check | 10s poll |

## 1.5 Constitutional Visibility Contract

Every terminal surface MUST display:

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Ihsan | >= 0.95 | >= 0.85 | < 0.85 |
| SNR | >= 0.85 | >= 0.75 | < 0.75 |
| Gini | <= 0.35 | <= 0.40 | > 0.40 |
| Mint Status | "MINTING" (Ihsan >= 0.95) | "PAUSED" (below floor) | "VIOLATION" |

## 1.6 TDD Anchors

```
TEST: terminal_cold_start_under_1_second
  GIVEN clean node state
  WHEN terminal starts
  THEN banner + briefing renders in < 1000ms

TEST: terminal_offline_mode_no_crash
  GIVEN backend unreachable
  WHEN any view is requested
  THEN offline fallback renders (no panic, no hang)

TEST: terminal_all_7_views_navigable
  GIVEN terminal running
  WHEN user presses 1-7
  THEN corresponding view renders without error

TEST: terminal_json_mode_all_commands
  GIVEN --json flag
  WHEN any command runs
  THEN output is valid JSON (parseable by jq)

TEST: terminal_constitutional_metrics_visible
  GIVEN any view
  THEN Ihsan, SNR, Gini are displayed with color coding
```
