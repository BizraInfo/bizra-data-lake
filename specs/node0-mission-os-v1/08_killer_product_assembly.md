# Node0 Killer Product Assembly — From Existing Assets Only

**Status:** [ENFORCEMENT: WIRED]
**Rule:** Create NOTHING new. Assemble from the 200+ existing assets.
**Source:** Downloads/, BIZRA-Archive/, award-winner-design/, Claude sessions

## The Product (One Sentence)

Node0 boots with your machine. Your 7 agents are alive. They work for you.
You see the proof.

## Three Surfaces, All From Existing Assets

### Surface 1: Public Trust (bizra.ai)

**What the visitor sees:** A sovereign AI OS you can download and run.

**Assembled from:**
```
Landing:        BIZRA-Archive/BIZRA-Frontend-Final/bizra-clean.html (DEPLOYED)
Atlas:          BIZRA-Archive/BIZRA-Frontend-Final/BIZRA-DDAGI-OS-Atlas-v6_Peak.html
Constitution:   BIZRA-Archive/BIZRA-Frontend-Final/BIZRA_Universal_Constitution_v3.0.0_GENESIS.html
Proof Chain:    BIZRA-Archive/BIZRA-Frontend-Final/bizra-proof-chain-cortex.html
Brand:          BIZRA-Archive/last-update/BIZRA_Logo_Dark_4K.png
                BIZRA-Archive/last-update/BIZRA_Icon_Transparent_2K.png
                BIZRA-Archive/last-update/BIZRA_Favicon_512.png
                BIZRA-Archive/last-update/BIZRA_Brand_Guidelines_v2.pdf
Video:          BIZRA-Archive/where-we-stand/BIZRA__Sovereign_AI_Blueprint.mp4
Diagrams:       Downloads/bizra_architecture_diagram.png
                Downloads/bp_work_loop.png
                Downloads/bp_stack_integration.png
                Downloads/canonical_loop_proof_diagram.png
```

**Assembly:** Multi-page Vercel deployment. Landing → Atlas → Constitution → Proof. All pages exist. Just deploy.

---

### Surface 2: Activation (Genesis Flow)

**What Node0 (you) sees on first run.**

**Assembled from:**
```
Portal:         BIZRA-Archive/last-update/BIZRA_Genesis_Portal.html
Status:         BIZRA-Archive/BIZRA-Frontend-Final/BIZRA_Genesis_Status.html
Seed:           BIZRA-Archive/BIZRA-Frontend-Final/BIZRA-Constitutional-Seed.html
Activation:     .bizra-kernel/sovereign_activation.py (THIS SESSION)
Identity:       BIZRA-Archive/value-assets/identity_genesis.py
Pipeline:       BIZRA-Archive/value-assets/mission_pipeline.py
```

**Assembly:** The activation ceremony already runs. The Genesis Portal HTML already exists. Wire the activation output into the portal display.

---

### Surface 3: Node0 Mission OS (Daily Use)

**What you see every day. PAT-7 alive. Missions running. Receipts flowing.**

**Assembled from:**
```
Dashboard:      BIZRA-Archive/BIZRA-Frontend-Final/node0-pipeline-dashboard.jsx
JARVIS:         BIZRA-Archive/BIZRA-Frontend-Final/BIZRA_JARVIS.jsx
Architecture:   BIZRA-Archive/BIZRA-Frontend-Final/maestro-architecture.jsx
Knowledge:      BIZRA-Archive/last-update/BIZRA_Knowledge_Dashboard.jsx
KIS:            BIZRA-Archive/last-update/BIZRA_KIS_v2.jsx
PAT Launch:     BIZRA-Archive/value-assets/PAT_Genesis_Launch.jsx
Command:        BIZRA-Archive/value-assets/PCO_Command_Center.jsx
Asset Registry: BIZRA-Archive/value-assets/NODE0_Asset_Registry.jsx
Emulator:       BIZRA-Archive/value-assets/BIZRA_Node_Emulator.jsx
Node0 Arch:     BIZRA-Archive/last-update/BIZRA_Node0_Unified_Architecture.jsx

Backend:        BIZRA-Archive/value-assets/production_pipeline.py
Mission:        BIZRA-Archive/value-assets/mission_pipeline.py
Gate:           BIZRA-Archive/value-assets/ihsan_gate.py
SNR:            BIZRA-Archive/value-assets/snr.py
Reflex:         BIZRA-Archive/value-assets/reflex_cache.py
Receipts:       BIZRA-Archive/value-assets/evidence_receipt.py
HHMM:           BIZRA-Archive/value-assets/hhmm_router.py
LLM:            BIZRA-Archive/value-assets/ollama_provider.py
Genesis:        BIZRA-Archive/value-assets/genesis_engine.py
Tests:          BIZRA-Archive/value-assets/test_*.py (10 files)
```

**Assembly:** The JSX components ARE the UI. The Python files ARE the backend.
Wire them through the existing Kong gateway (port 8000) or kernel API (port 8010).

---

## PAT-7 Boot-On-Startup Spec

```
SYSTEM STARTUP:
  1. Windows boots → WSL starts → systemd triggers bizra-node0.service
  2. Service starts:
     a. Docker containers (already configured in docker-compose.yml)
     b. Kernel API on port 8010
     c. PAT-7 spawn from BIZRA-Archive/value-assets/genesis_engine.py
     d. Heartbeat daemon (.bizra-kernel/heartbeat_daemon.py --bg)
     e. Morning Brief compilation from overnight receipts
  3. Desktop notification: "Node0 alive. 3 items need attention."

DAILY LOOP:
  - Morning Brief: compiled from overnight heartbeat + receipt chain
  - Ghost Panel: proactive agent suggestions (from sovereign_cockpit donor)
  - Mission box: user types intent → canonical_loop processes → receipt emitted
  - Trust Rail: live Ihsan/SNR/Gini sidebar
  - SEED balance: accumulated from verified work

POWER OFF:
  - Graceful shutdown hook saves state
  - Receipt chain persisted
  - Flywheel state saved
  - Next boot resumes from last receipt
```

## Systemd Service File

```ini
# /etc/systemd/system/bizra-node0.service
[Unit]
Description=BIZRA Node0 Sovereign Intelligence
After=docker.service network.target
Requires=docker.service

[Service]
Type=forking
User=root
WorkingDirectory=/mnt/c/Users/BIZRA-OS
ExecStartPre=/usr/bin/docker compose -f /mnt/c/BIZRA-DATA-LAKE/docker-compose.yml up -d
ExecStart=/usr/bin/python3 .bizra-kernel/sovereign_activation.py
ExecStartPost=/usr/bin/python3 .bizra-kernel/heartbeat_daemon.py start --bg --hours 720 --interval 5
ExecStop=/usr/bin/python3 .bizra-kernel/heartbeat_daemon.py stop
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

## Assembly Order (Dependency-Sorted)

| Step | What | Source | Time |
|------|------|--------|------|
| 1 | Deploy multi-page website to Vercel | 6 HTML files from BIZRA-Archive | 30 min |
| 2 | Create systemd service for boot-on-start | Spec above | 15 min |
| 3 | Wire PAT_Genesis_Launch.jsx to kernel API | Existing JSX + API | 2 hours |
| 4 | Wire node0-pipeline-dashboard.jsx to live data | Existing JSX + heartbeat | 2 hours |
| 5 | Wire BIZRA_JARVIS.jsx as mission input | Existing JSX + canonical_loop | 3 hours |
| 6 | Add Morning Brief from receipt chain | ghost_ws.py + receipt_ledger.py | 2 hours |
| 7 | Add Trust Rail from sovereign cockpit donor | BIZRA_SovereignCockpit.jsx | 2 hours |
| 8 | Desktop notification bridge | Python notify-send or Windows toast | 1 hour |

**Total: ~12 hours of assembly. Zero new creation.**

## Acceptance Gate

The product is ready when:

- [ ] PAT-7 boots with the machine (systemd service)
- [ ] Morning Brief shows on login (notification or dashboard)
- [ ] User can type a mission and see a real receipt
- [ ] Trust Rail shows live Ihsan/SNR/Gini
- [ ] SEED balance visible and growing from verified work
- [ ] bizra.ai shows more than a landing page (atlas, constitution, proof)
- [ ] All from existing assets — nothing new created
