# Module 02 — Sovereign Runtime

> **Domain:** Node0 core, mission pipeline, OODA loop, sovereignty tiers
> **Source Specs:** Phase 57 (first heartbeat), Phase 58 (optimization), Phase 61-62 (genesis v5/v6), Phase 64 (asset registry), Phase 70 (bus wiring)
> **Key Path:** `core/sovereign/` (~109 files, largest module)

## 2.1 SovereignRuntime Core

**Status:** [x] BUILT
**Path:** `core/sovereign/__main__.py`, `core/sovereign/api.py`

Main runtime loop. Initializes all subsystems, wires EventBus, starts API server.
Entry: `python -m core.sovereign serve --host 0.0.0.0 --port 8000`

**API routes:** `/v1/health`, `/v1/verify/*`, `/v1/auth/*`, plus 8 auth-guarded POST routes
**Auth:** `_authenticate_http_request(request)` 3-tuple check on all mutation endpoints
**Intentionally open:** `/v1/verify/*` (external auditors), `/v1/auth/*` (bootstrap)

**Tests:** `tests/core/sovereign/` — 90+ test files

---

## 2.2 Mission Orchestrator (6-Phase Pipeline)

**Status:** [x] BUILT
**Path:** `core/sovereign/mission.py`

Pipeline: OBSERVE -> DECOMPOSE -> EXECUTE -> SYNTHESIZE -> GATE -> EVIDENCE

**Components used:**
- ChannelDispatcher, BrowserMCPClient, LivingMemoryCore
- SNRApexEngine, EvidenceLedger, EventBus
- HDAClient (async TCP JSON-RPC for AHK)

**Completion threshold:** `UNIFIED_IHSAN_THRESHOLD` (0.95) from constants.py
**Quality fallback:** 0.80/0.75 -> PARTIAL status (never above-threshold defaults)

**Tests:** 28 unit + 10 integration = 38 tests GREEN

---

## 2.3 EventBus

**Status:** [x] BUILT
**Path:** `core/sovereign/event_bus.py`

Publish-subscribe event system for inter-module communication.
Rust mirror: 8-shard FNV-1a EventBus in `bizra-hooks/src/event_bus.rs`

**Integration (Phase 70):** Bus infrastructure wired into SovereignRuntime

---

## 2.4 API Exposure Policy

**Status:** [x] BUILT
**Path:** `core/sovereign/api_exposure_policy.py`

Controls which API endpoints are exposed based on node tier and deployment context.
Prevents accidental exposure of internal endpoints.

**Tests:** `tests/core/sovereign/test_api_exposure_policy.py`

---

## 2.5 Sovereignty Tiers

**Status:** [x] BUILT
**Path:** `core/spearpoint/config.py` (TierPolicy)

Four tiers based on node maturity:
- SEED (0.00-0.25) — minimal capabilities
- SPROUT (0.25-0.50) — basic agent access
- TREE (0.50-0.75) — full local reasoning
- FOREST (0.75-1.00) — federation + governance rights

---

## 2.6 Node Identity & Credentials

**Status:** [x] BUILT
**Path:** `core/sovereign/` (identity management)

Persistent node signer at `sovereign_state/mission_signer.json`.
Loads existing or inherits from `identity/credentials.json`.
Ed25519 key pairs for signing all receipts and messages.

---

## 2.7 OODA Loop

**Status:** [x] BUILT
**Path:** Embedded in MissionOrchestrator and SovereignRuntime

Boyd's OODA loop (Observe-Orient-Decide-Act) implemented as the core
decision cycle. Infrastructure Guardian also uses OODA for probe cycles.

---

## 2.8 Receipt-Memory Feedback Loop

**Status:** [x] BUILT
**Path:** `core/sovereign/` (receipt feedback)

Approved receipts reinforce memory patterns. Rejected receipts trigger
re-evaluation. Creates learning loop from execution outcomes.

**Tests:** `test_runtime_core.py::TestReceiptMemoryFeedback`

---

## 2.9 Node0 Proactive Kernel

**Status:** [x] BUILT
**Path:** `core/sovereign/` (Node0ProactiveKernel class)
**Systemd:** `deploy/node0/bizra-node0.service`

Proactive mode: node monitors environment and initiates actions without
external prompts. PYTHONUNBUFFERED=1 required for crash traceback visibility.
ExecStartPre syntax check prevents crash-loop on SyntaxError.

---

## 2.10 Node0 Genesis Server (v6)

**Status:** [x] BUILT
**Path:** `core/sovereign/` (node0_server.py)
**Systemd:** `deploy/node0/bizra-node0-genesis.service` (port 7770)

Constitutional pipeline server. Separate from proactive kernel.
Docker: `deploy/Dockerfile.node0-genesis` (260MB, python:3.12-slim)

---

## 2.11 Floor Constraint

**Status:** [x] BUILT
**Path:** `core/sovereign/floor_constraint.py` (Phase 64)

Constitutional universality: GPU NEVER required, network NEVER required.
Minimum: 2GB RAM, 2 cores, 4GB disk. `daughter_test()` encodes this gate.

---

## 2.12 Asset Registry

**Status:** [x] BUILT
**Path:** `core/sovereign/asset_registry.py` (Phase 64)

Hardware body detection and asset registration. 52 tests.
`BIZRA_SOVEREIGN_ROOT` fallback: graceful degrade when B: drive unmounted.

---

## 2.13 Seed Potential Engine

**Status:** [x] BUILT
**Path:** `core/sovereign/` (Phase 71)

DDAGI Seed Potential Engine — scores node growth potential.

---

## 2.14 Node Value Calculator

**Status:** [x] BUILT
**Path:** `core/sovereign/node_value.py`

Computes intrinsic value of a node based on contributions, uptime,
and quality metrics.

**Tests:** `tests/core/sovereign/test_node_value.py`

---

## 2.15 Network Effect Model

**Status:** [x] BUILT
**Path:** `core/sovereign/network_effect.py`

Models Metcalfe's law and network value growth as nodes join.

---

## 2.16 Human Lifecycle Model

**Status:** [x] BUILT
**Path:** `core/sovereign/human_lifecycle.py`

Maps human life stages to node capabilities and needs.

---

## 2.17 Agent CLI

**Status:** [x] BUILT
**Path:** `core/sovereign/agent_cli.py`

Command-line interface for interacting with sovereign agent.

**Tests:** `tests/core/sovereign/test_agent_cli.py`

---

## 2.18 Channel Dispatcher

**Status:** [x] BUILT
**Path:** `core/sovereign/` (channel dispatch)

Routes messages to appropriate channels (local, network, HDA, browser).

---

## 2.19 Sovereignty Growth Tracker

**Status:** [~] PARTIAL
**Path:** Impact tracker exists but growth dashboard is frontend-dependent
**Gap:** Backend metrics exist, no persistent growth history timeline

### TDD Anchor
```
def test_sovereignty_growth_history():
    tracker = SovereigntyGrowthTracker(db=test_db)
    tracker.record_milestone("node_a", tier="SPROUT", score=0.30)
    history = tracker.get_growth_timeline("node_a")
    assert len(history) >= 1
    assert history[-1].tier == "SPROUT"
```

---

## 2.20 Multi-Node Coordination

**Status:** [~] PARTIAL
**Path:** EventBus supports multi-listener, but no cross-node protocol
**Gap:** No network-level node coordination (see Module 07 Federation)

---

## 2.21 Sovereignty Dashboard API

**Status:** [~] PARTIAL
**Path:** API routes exist for health/verify, but no dedicated dashboard data API
**Gap:** No aggregated sovereignty metrics endpoint for frontend consumption

### TDD Anchor
```
def test_sovereignty_dashboard_endpoint():
    response = client.get("/v1/sovereignty/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "tier" in data
    assert "ihsan_score" in data
    assert "growth_history" in data
    assert "active_agents" in data
```

---

## 2.22 Offline-First Resilience

**Status:** [ ] NOT BUILT
**Spec:** Node must operate fully without network connectivity
**Gap:** Many paths assume localhost services (Redis, Ollama). No offline fallback cache.

### Pseudocode
```
class OfflineResilienceLayer:
    """Ensure node operates without network"""

    def __init__(self):
        self.local_cache = SQLiteCache("sovereign_state/offline_cache.db")
        self.pending_sync = Queue()

    def get_with_fallback(self, key: str, fetcher: Callable):
        try:
            value = fetcher()
            self.local_cache.put(key, value)
            return value
        except ConnectionError:
            return self.local_cache.get(key)  # Graceful offline

    def queue_for_sync(self, action: Action):
        """Queue actions taken offline for later sync"""
        self.pending_sync.put(action)
        self.local_cache.put(f"pending:{action.id}", action.serialize())
```

---

## Completion

| Feature | Status | Coverage |
|---------|--------|----------|
| 2.1 Runtime Core | BUILT | 90+ tests |
| 2.2 Mission Orchestrator | BUILT | 38 tests |
| 2.3 EventBus | BUILT | Full |
| 2.4 API Exposure Policy | BUILT | Tests |
| 2.5 Sovereignty Tiers | BUILT | Full |
| 2.6 Node Identity | BUILT | Ed25519 |
| 2.7 OODA Loop | BUILT | Embedded |
| 2.8 Receipt Feedback | BUILT | Tests |
| 2.9 Proactive Kernel | BUILT | systemd |
| 2.10 Genesis Server | BUILT | Docker |
| 2.11 Floor Constraint | BUILT | 52 tests |
| 2.12 Asset Registry | BUILT | Phase 64 |
| 2.13 Seed Potential | BUILT | Phase 71 |
| 2.14 Node Value | BUILT | Tests |
| 2.15 Network Effect | BUILT | Model |
| 2.16 Human Lifecycle | BUILT | Model |
| 2.17 Agent CLI | BUILT | Tests |
| 2.18 Channel Dispatch | BUILT | Full |
| 2.19 Growth Tracker | PARTIAL | No timeline |
| 2.20 Multi-Node | PARTIAL | Local only |
| 2.21 Dashboard API | PARTIAL | No aggregate |
| 2.22 Offline-First | NOT BUILT | Zero |
| **TOTAL** | **18/22 + 3P + 1N** | **86%** |
