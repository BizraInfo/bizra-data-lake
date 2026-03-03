# Phase 57: First Heartbeat — End-to-End Sovereign Task Execution

**Status:** SPECIFICATION
**Priority:** P0 — This is the demo that makes BIZRA real
**Author:** spec-pseudocode mode
**Date:** 2026-03-02

## The Thesis

No system on the market does this:

```
Keystroke → Desktop Context Capture → Agent Decomposition →
Desktop Action + Browser Action (parallel) → Synthesis →
Constitutional Proof Trace → User Result with Evidence
```

OpenClaw got famous by being the fastest-growing GitHub repo because it showed
one compelling demo. BIZRA's demo is **sovereignty** — a single hotkey triggers
a task that crosses desktop and browser boundaries, executes with constitutional
governance, and returns a proof-traced result. No other system goes from
keystroke to browser to proof trace in one flow.

## What "First Heartbeat" Means

The first heartbeat is the first time NODE0 executes a real user task
end-to-end using the full stack:

1. **AHK Hotkey** — User presses `Win+Shift+B` to trigger a mission
2. **HDA Context Capture** — Desktop state captured (active window, clipboard)
3. **Channel Dispatch** — Task decomposed into DESKTOP + BROWSER channels
4. **Desktop Action** — File creation, text manipulation, or app interaction
5. **Browser Action** — Web search via MCP, page fetch, data extraction
6. **Synthesis** — Results merged by orchestrator with memory context
7. **Constitutional Gate** — Ihsan/SNR/PCI validation on the output
8. **Evidence Receipt** — Hash-chained proof of execution
9. **User Response** — Toast notification + file output with proof trace

## Component Inventory (Honest Assessment)

| Layer | Component | File | Status | Demo-Ready? |
|---|---|---|---|---|
| Input | AHK Client | `bin/bizra_bridge.ahk` | FUNCTIONAL | Yes |
| Input | AHK Server/HDA | `filedfs/ahk_bridge.ahk` | FUNCTIONAL | Yes |
| Bridge | Desktop Bridge | `core/bridges/desktop_bridge.py` | FUNCTIONAL | Yes |
| Router | Channel Dispatcher | `core/bridges/channel_dispatcher.py` | FUNCTIONAL | Yes |
| Browser | Browser MCP Client | `core/bridges/browser_mcp_client.py` | FUNCTIONAL (mock+direct) | Yes |
| Brain | Sovereign Orchestrator | `core/sovereign/orchestrator.py` | FUNCTIONAL | Needs wiring |
| Memory | Living Memory | `core/living_memory/core.py` | FUNCTIONAL | Yes |
| Comms | Event Bus | `core/sovereign/event_bus.py` | FUNCTIONAL | Yes |
| Comms | A2A Engine | `core/a2a/engine.py` | FUNCTIONAL (local) | Yes |
| Identity | PAT Agents | `core/pat/agent.py` | DATA MODEL | Partial |
| Proof | Evidence Ledger | `core/proof_engine/evidence_ledger.py` | FUNCTIONAL | Yes |
| Gateway | MCP Gateway | `tools/mcp/mcp_gateway.py` | STUB queries | No (not needed for v1) |

### Critical Gap: Port Collision on 9742

Three components all target port 9742:
- `desktop_bridge.py` (Python TCP server)
- `filedfs/ahk_bridge.ahk` (AHK TCP server)
- `bin/bizra_bridge.ahk` (AHK client connects to 9742)

**Resolution:** See `phase_57_02_port_architecture.md`

### Missing Components (Not Needed for v1)

- **Telescript** — Rust-only concept, no Python layer. Not needed for demo.
- **Action Bus** — Rust-only (`bizra-action`). Use Event Bus topics instead.
- **PAT Runtime Loop** — Data model exists, no execution loop. Orchestrator handles dispatch.

## Demo Task: "Research and Brief"

The signature demo task:

> "Research the latest AI agent frameworks and create a briefing document on my desktop."

This exercises:
1. **AHK** — Hotkey capture + toast notifications
2. **HDA** — Desktop context (what app is open, clipboard state)
3. **Browser** — Web search for "AI agent frameworks 2026"
4. **Desktop** — Create `~/Desktop/BIZRA_Brief_<timestamp>.md`
5. **Memory** — Store research as episodic memory for future retrieval
6. **Proof** — Hash-chained evidence receipt with Ed25519 signature
7. **Constitutional** — Ihsan gate on output quality

## File Index

| File | Purpose |
|---|---|
| `phase_57_00_overview.md` | This file — overview and inventory |
| `phase_57_01_architecture.md` | System architecture and data flow |
| `phase_57_02_port_architecture.md` | Port assignment and connection topology |
| `phase_57_03_mission_protocol.md` | Mission lifecycle pseudocode |
| `phase_57_04_wiring_pseudocode.md` | Component wiring and initialization |
| `phase_57_05_demo_script.md` | Step-by-step demo execution script |
| `phase_57_06_test_plan.md` | TDD anchors and validation criteria |
