# Phase 57.01: System Architecture

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER (Windows Desktop)                       │
│                                                                     │
│  [Win+Shift+B] ──→ AHK Client (bin/bizra_bridge.ahk)              │
│                      │                                              │
│                      │ TCP JSON-RPC (port 9742)                     │
│                      │ Auth: BIZRA_BRIDGE_TOKEN + timestamp + nonce │
│                      ▼                                              │
└──────────────────────┼──────────────────────────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────────────────┐
│                  WSL (Ubuntu / Python Runtime)                      │
│                      ▼                                              │
│  ┌─────────────────────────────────────┐                           │
│  │    DesktopBridge (port 9742)        │                           │
│  │    core/bridges/desktop_bridge.py   │                           │
│  │                                     │                           │
│  │  Methods:                           │                           │
│  │  - execute_mission(description)     │ ◄── NEW METHOD            │
│  │  - sovereign_query(prompt)          │                           │
│  │  - invoke_skill(name, inputs)       │                           │
│  └──────────┬──────────────────────────┘                           │
│             │                                                       │
│             │ MissionRequest                                        │
│             ▼                                                       │
│  ┌─────────────────────────────────────┐                           │
│  │    MissionOrchestrator              │ ◄── NEW MODULE             │
│  │    core/sovereign/mission.py        │                           │
│  │                                     │                           │
│  │  1. Capture desktop context         │                           │
│  │  2. Decompose via ChannelDispatcher │                           │
│  │  3. Execute channels (parallel)     │                           │
│  │  4. Synthesize results              │                           │
│  │  5. Gate through constitutional     │                           │
│  │  6. Emit evidence receipt           │                           │
│  │  7. Return to user                  │                           │
│  └──────┬───────────┬──────────────────┘                           │
│         │           │                                               │
│    ┌────┘           └────┐                                          │
│    ▼                     ▼                                          │
│  ┌──────────────┐  ┌──────────────────┐                            │
│  │ DESKTOP Chan │  │ BROWSER Channel  │                            │
│  │              │  │                  │                             │
│  │ AHK Server   │  │ BrowserMCPClient │                            │
│  │ (port 9743)  │  │ (direct mode)    │                            │
│  │              │  │                  │                             │
│  │ Commands:    │  │ Methods:         │                             │
│  │ - type_text  │  │ - search(query)  │                            │
│  │ - file_open  │  │ - fetch_page(url)│                            │
│  │ - click      │  │ - research(q)    │                            │
│  │ - screenshot │  │                  │                             │
│  └──────┬───────┘  └───────┬──────────┘                            │
│         │                  │                                        │
│         └────────┬─────────┘                                        │
│                  ▼                                                   │
│  ┌─────────────────────────────────────┐                           │
│  │    SynthesisEngine                  │ ◄── NEW MODULE             │
│  │                                     │                           │
│  │  - Merge DESKTOP + BROWSER results  │                           │
│  │  - Format as briefing document      │                           │
│  │  - Score via SNRApexEngine          │                           │
│  │  - Gate via Ihsan threshold         │                           │
│  └──────────┬──────────────────────────┘                           │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────────────────────┐                           │
│  │    Evidence Spine                   │                           │
│  │                                     │                           │
│  │  - EvidenceLedger.emit_receipt()    │                           │
│  │  - Ed25519 signature                │                           │
│  │  - BLAKE3 chain hash                │                           │
│  │  - LivingMemory.encode() episode    │                           │
│  └──────────┬──────────────────────────┘                           │
│             │                                                       │
│             │ MissionResult                                         │
│             ▼                                                       │
│  ┌─────────────────────────────────────┐                           │
│  │    DesktopBridge                    │                           │
│  │    → Returns JSON-RPC response      │                           │
│  └──────────┬──────────────────────────┘                           │
│             │                                                       │
└─────────────┼───────────────────────────────────────────────────────┘
              │
┌─────────────┼───────────────────────────────────────────────────────┐
│  USER       ▼                                                       │
│                                                                     │
│  AHK Client receives result                                         │
│  → Toast notification: "Mission complete. Briefing saved."          │
│  → File created: ~/Desktop/BIZRA_Brief_20260302_0830.md            │
│  → Proof receipt ID shown in tooltip                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Dependency Graph

```
bin/bizra_bridge.ahk (AHK client)
  └──→ core/bridges/desktop_bridge.py (TCP server)
         ├──→ core/sovereign/mission.py (NEW: MissionOrchestrator)
         │      ├──→ core/bridges/channel_dispatcher.py
         │      │      ├──→ filedfs/ahk_bridge.ahk (via TCP 9743)
         │      │      └──→ core/bridges/browser_mcp_client.py
         │      ├──→ core/sovereign/orchestrator.py (for LLM synthesis)
         │      ├──→ core/living_memory/core.py (episodic storage)
         │      ├──→ core/apex/snr_apex_engine.py (quality scoring)
         │      ├──→ core/proof_engine/evidence_ledger.py (receipt)
         │      └──→ core/sovereign/event_bus.py (notifications)
         └──→ core/inference/gateway.py (LM Studio backend)
```

## New Modules Required

### 1. `core/sovereign/mission.py` — MissionOrchestrator

The central coordinator for end-to-end task execution. Connects all existing
components into a single pipeline. This is the ONLY new Python module needed.

**Why a new module instead of extending orchestrator.py?**
- `orchestrator.py` handles agent-level task routing (RESEARCHER, ANALYST, etc.)
- `mission.py` handles user-level mission coordination (hotkey → result)
- Clean separation: missions contain tasks, tasks are routed by orchestrator

### 2. `execute_mission` method on DesktopBridge

One new JSON-RPC method added to the existing `desktop_bridge.py`. Routes
incoming mission requests to the MissionOrchestrator.

### 3. AHK hotkey addition to `bin/bizra_bridge.ahk`

One new hotkey: `Win+Shift+B` → InputBox for mission description →
calls `execute_mission` → displays result as toast.

## Existing Components Used As-Is

| Component | Used For | Changes Needed |
|---|---|---|
| `channel_dispatcher.py` | Route to DESKTOP/BROWSER | None |
| `browser_mcp_client.py` | Web search + page fetch | Switch to `direct` mode |
| `ahk_bridge.ahk` | Desktop actuator | Change port to 9743 |
| `event_bus.py` | Mission lifecycle events | None |
| `evidence_ledger.py` | Proof receipts | None |
| `living_memory/core.py` | Episodic storage | None |
| `snr_apex_engine.py` | Quality scoring | None |
| `pci/envelope.py` | Constitutional gate | None (optional for v1) |
