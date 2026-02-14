# Phase 06: Boot Sequence + Service Lifecycle

Last updated: 2026-02-14
Standing on: Hightower (SRE) · Thompson (Unix Process Model) · Nygard (2007, Release It!)

---

## Purpose

This phase specifies the **deterministic boot sequence** — the exact order in which the kernel initializes, verifies, and activates all 7 layers. The boot sequence is the critical path from power-on to "Node0 READY." Every step is observable, every failure is recoverable, and the total boot time target is **< 60 seconds**.

---

## Boot Sequence

```
┌───────────────────────────────────────────────────────────────┐
│ PHASE 0: HARDWARE VERIFICATION (Layer 1)           [0-5s]    │
│   ├── Load genesis state from sovereign_state/               │
│   ├── Verify birth certificate signature                     │
│   ├── Re-attest hardware (ROOT tier)                         │
│   └── GATE: If ROOT mismatch → KERNEL PANIC                 │
├───────────────────────────────────────────────────────────────┤
│ PHASE 1: LEDGER REPLAY (Layer 4)                   [5-15s]   │
│   ├── Replay tamper-evident log → reconstruct state          │
│   ├── Verify hash chain integrity                            │
│   ├── Replay token ledger → reconstruct balances             │
│   ├── Load experience ledger head                            │
│   └── GATE: If chain break detected → HALT + alert           │
├───────────────────────────────────────────────────────────────┤
│ PHASE 2: GATE CHAIN INITIALIZATION (Layer 3)       [15-20s]  │
│   ├── Load constitutional thresholds from constants.py       │
│   ├── Initialize Z3 solver with default constraints          │
│   ├── Initialize PCI gate chain (6 gates)                    │
│   ├── Warm nonce cache (empty on fresh boot)                 │
│   └── GATE: Z3 solver self-test (trivial SAT check)         │
├───────────────────────────────────────────────────────────────┤
│ PHASE 3: INFERENCE BACKEND ACTIVATION (Layer 2)    [20-35s]  │
│   ├── Probe LM Studio (192.168.56.1:1234)                   │
│   ├── Probe Ollama (localhost:11434)                         │
│   ├── Initialize connection pool (pre-warm 2 connections)    │
│   ├── Initialize circuit breakers (all CLOSED)               │
│   ├── Initialize rate limiter (10 req/s, burst 20)           │
│   └── GATE: At least 1 backend responsive                   │
├───────────────────────────────────────────────────────────────┤
│ PHASE 4: DUAL-AGENTIC ACTIVATION (Layers 2+5)     [35-45s]  │
│   ├── PAT Agent: Load user goals from experience ledger      │
│   │   ├── Initialize SkillRouter + registered skills         │
│   │   └── Ready for user interaction                         │
│   ├── SAT Agent: Initialize system health monitoring         │
│   │   ├── Start metrics collector (30s interval)             │
│   │   ├── Start SAT Controller (Gini monitoring)             │
│   │   └── Initialize autopoietic loop (DORMANT state)        │
│   └── GATE: Both agents report READY                         │
├───────────────────────────────────────────────────────────────┤
│ PHASE 5: COMPONENT ORCHESTRATION                   [45-55s]  │
│   ├── Docker health check (if containers configured)         │
│   ├── Desktop Bridge (TCP 127.0.0.1:9742) — if Windows      │
│   ├── API Server (HTTP :8000) — start listening              │
│   └── WebSocket stream — start accepting connections         │
├───────────────────────────────────────────────────────────────┤
│ PHASE 6: RECONCILIATION LOOP START (Layer 6)       [55-60s]  │
│   ├── Calculate initial gap: Current State ↔ Desired State   │
│   ├── Start RecursiveLoop (CONTINUOUS mode)                  │
│   ├── Start SpearPoint pipeline (async, post-query)          │
│   └── Node0 status: READY                                   │
└───────────────────────────────────────────────────────────────┘
```

---

## Pseudocode

### Main Boot Procedure

```pseudocode
PROCEDURE boot_node0() -> Result<RuntimeState, BootError>:
    boot_start = now()
    LOG_INFO("Node0 boot sequence initiated")

    # ─── PHASE 0: HARDWARE VERIFICATION ───
    LOG_INFO("[Phase 0] Hardware verification...")
    identity = verify_identity_on_boot()?   # From phase_01_identity.md
    LOG_INFO("[Phase 0] Identity verified: {}", identity.node_id)

    # ─── PHASE 1: LEDGER REPLAY ───
    LOG_INFO("[Phase 1] Replaying ledgers...")

    tamper_log = TamperEvidentLog.load("sovereign_state/tamper_evident.log")
    tamper_result = tamper_log.verify_chain()
    IF NOT tamper_result.valid:
        RETURN Error(BootError::ChainIntegrityFailure(tamper_result.error))
    LOG_INFO("[Phase 1] Tamper log verified: {} entries", tamper_result.entries_checked)

    experience_ledger = EpisodeLedger.load("sovereign_state/experience_ledger.db")
    token_ledger = TokenLedger.load(
        sqlite_path = ".swarm/memory.db",
        jsonl_path  = "04_GOLD/token_ledger.jsonl",
    )
    token_chain = token_ledger.verify_chain()
    IF NOT token_chain.valid:
        RETURN Error(BootError::TokenChainCorrupted(token_chain.error))
    LOG_INFO("[Phase 1] Token ledger verified: {} transactions", token_chain.entries_checked)

    # ─── PHASE 2: GATE CHAIN INITIALIZATION ───
    LOG_INFO("[Phase 2] Initializing gate chain...")

    thresholds = load_constitutional_thresholds()   # from core/integration/constants.py
    z3_gate = Z3FATEGate(thresholds)

    # Z3 self-test: trivial satisfiability check
    self_test = z3_gate.verify(Action(ihsan=1.0, snr=1.0, risk=0, reversible=true, cost=0.0))
    IF NOT self_test.satisfiable:
        RETURN Error(BootError::Z3SelfTestFailed)

    gate_chain = GateChain(thresholds, z3_gate)
    LOG_INFO("[Phase 2] Gate chain ready (6 gates, Z3 self-test passed)")

    # ─── PHASE 3: INFERENCE BACKEND ACTIVATION ───
    LOG_INFO("[Phase 3] Probing inference backends...")

    backends = []
    FOR EACH backend_config IN inference_config():
        health = probe_health(backend_config.url, timeout=5s)
        IF health.ok:
            backends.append(InferenceBackend(backend_config))
            LOG_INFO("[Phase 3] Backend online: {}", backend_config.name)
        ELSE:
            LOG_WARNING("[Phase 3] Backend offline: {}", backend_config.name)

    IF backends.is_empty():
        RETURN Error(BootError::NoInferenceBackend)

    gateway = InferenceGateway(
        backends       = backends,
        pool_size      = 2,     # Pre-warm connections
        rate_limit     = 10,    # req/s
        burst          = 20,
    )
    LOG_INFO("[Phase 3] Gateway ready: {} backends", len(backends))

    # ─── PHASE 4: DUAL-AGENTIC ACTIVATION ───
    LOG_INFO("[Phase 4] Activating PAT + SAT agents...")

    # PAT: Load user context
    user_goals = experience_ledger.retrieve("active goals", k=10)
    skill_router = SkillRouter()
    register_all_skills(skill_router)   # RDVE, smart_files, etc.
    pat_agent = PATAgent(
        identity       = identity,
        goals          = user_goals,
        skill_router   = skill_router,
        gateway        = gateway,
    )

    # SAT: System health
    metrics_collector = MetricsCollector(interval=30s)
    sat_controller = SATController(
        gini_threshold = thresholds.ADL_GINI_THRESHOLD,
        zakat_rate     = ZAKAT_RATE,
    )
    autopoietic_loop = AutopoieticLoopEngine(state=DORMANT, z3_gate=z3_gate)

    sat_agent = SATAgent(
        identity       = identity,
        metrics        = metrics_collector,
        controller     = sat_controller,
        autopoiesis    = autopoietic_loop,
    )
    LOG_INFO("[Phase 4] PAT ready ({} goals), SAT ready", len(user_goals))

    # ─── PHASE 5: COMPONENT ORCHESTRATION ───
    LOG_INFO("[Phase 5] Starting external components...")

    # API Server
    api_server = start_api_server(
        host = "127.0.0.1",
        port = 8000,
        engine = SovereignEngine(identity, gateway, gate_chain, ...),
    )

    # Desktop Bridge (Windows only)
    IF platform == "windows":
        desktop_bridge = start_desktop_bridge(
            host = "127.0.0.1",
            port = 9742,
            skill_router = skill_router,
        )

    LOG_INFO("[Phase 5] API server listening on 127.0.0.1:8000")

    # ─── PHASE 6: RECONCILIATION LOOP ───
    LOG_INFO("[Phase 6] Starting reconciliation loop...")

    recursive_loop = RecursiveLoop(
        researcher   = AutoResearcher(experience_ledger),
        evaluator    = AutoEvaluator(),
        mode         = CONTINUOUS,
    )
    spawn_async(recursive_loop.run())

    spearpoint = SpearPointPipeline(
        experience_ledger = experience_ledger,
        token_ledger      = token_ledger,
        tamper_log        = tamper_log,
    )

    boot_duration = now() - boot_start
    LOG_INFO("Node0 READY in {}ms", boot_duration.as_millis())

    # Record boot event in tamper log
    tamper_log.append("boot", {
        "duration_ms": boot_duration.as_millis(),
        "backends":    len(backends),
        "goals":       len(user_goals),
        "chain_length": token_chain.entries_checked,
    })

    RETURN Ok(RuntimeState(
        identity, gateway, gate_chain, pat_agent, sat_agent,
        experience_ledger, token_ledger, tamper_log,
        recursive_loop, spearpoint, api_server,
    ))
```

### Graceful Shutdown

```pseudocode
PROCEDURE shutdown_node0(state: RuntimeState):
    LOG_INFO("Node0 shutdown initiated")

    # Step 1: Signal reconciliation loop to stop
    state.recursive_loop.signal_shutdown()
    state.recursive_loop.await_completion(timeout=10s)

    # Step 2: Stop accepting new API requests
    state.api_server.stop_accepting()

    # Step 3: Drain in-flight requests (up to 30s)
    state.api_server.drain(timeout=30s)

    # Step 4: Flush metrics
    state.sat_agent.metrics.flush()

    # Step 5: Checkpoint experience ledger
    state.experience_ledger.checkpoint()

    # Step 6: Record shutdown in tamper log
    state.tamper_log.append("shutdown", {
        "uptime_seconds": state.uptime(),
        "reason": "graceful",
    })

    # Step 7: Close connections
    state.gateway.close_all()

    LOG_INFO("Node0 shutdown complete")
```

### Health Check

```pseudocode
PROCEDURE health_check(state: RuntimeState) -> HealthReport:
    checks = {}

    # Identity
    checks["identity"] = state.identity IS NOT None

    # Inference
    checks["inference"] = state.gateway.has_healthy_backend()

    # Ledger integrity
    checks["tamper_log"] = state.tamper_log.head_hash IS NOT None
    checks["token_ledger"] = state.token_ledger.chain_head IS NOT None

    # Constitutional compliance
    checks["ihsan_average"] = state.metrics.get_average("ihsan_score") >= 0.90

    # Reconciliation
    checks["reconciliation"] = state.recursive_loop.is_running()

    all_healthy = ALL(checks.values())

    RETURN HealthReport(
        status   = "healthy" IF all_healthy ELSE "degraded",
        checks   = checks,
        uptime   = state.uptime(),
        node_id  = state.identity.node_id,
    )
```

**Source:** `core/sovereign/launch.py`, `core/sovereign/__main__.py`, `bizra-omega/bizra-cli/src/main.rs`

---

## Service Integration

### Windows Service (Future)

```pseudocode
# Target: Windows Service via windows-service crate (Rust)
# Auto-start: SERVICE_AUTO_START
# Recovery: Restart on failure (5s → 10s → 30s exponential backoff)
# Session: Session 0 (isolated, pre-login)

SERVICE_CONFIG:
    name:            "BIZRANode0Kernel"
    display_name:    "BIZRA Node0 Sovereign Kernel"
    start_type:      AUTO_START
    error_control:   NORMAL
    dependencies:    []             # No external service dependencies
    recovery:
        first_failure:   RESTART (5s delay)
        second_failure:  RESTART (10s delay)
        subsequent:      RESTART (30s delay)
        reset_period:    86400s     # Reset failure count after 24h
```

### Current Activation (Python)

```bash
# Manual start
python -m core.sovereign launch

# Status check
python -m core.sovereign status

# Health doctor
python -m core.sovereign doctor

# Activation script (Windows)
# C:\BIZRA-Activate.bat starts all services:
#   PostgreSQL, Redis, Ollama, Rust API (3001), React Dashboard (5173)
```

**Source:** `core/sovereign/__main__.py`, `core/sovereign/launch.py`

---

## Error Handling

### Boot Failure Categories

```pseudocode
ENUM BootError:
    # Phase 0: Unrecoverable
    GenesisNotFound          # No sovereign_state/genesis.json
    BirthCertificateInvalid  # Signature verification failed
    HardwareRootMismatch     # CPU/GPU/Platform changed

    # Phase 1: Potentially recoverable
    ChainIntegrityFailure    # Tamper-evident log corrupted
    TokenChainCorrupted      # Token ledger hash chain broken

    # Phase 2: Recoverable
    Z3SelfTestFailed         # Z3 solver not functioning

    # Phase 3: Degraded operation possible
    NoInferenceBackend       # All LLM backends offline

    # Phase 4-6: Non-fatal
    SkillRegistrationFailed  # Individual skill failed to register
    DesktopBridgeFailed      # Bridge couldn't bind to port 9742
    ReconciliationFailed     # Loop failed to start
```

### Recovery Strategies

| Error | Severity | Recovery |
|-------|----------|----------|
| GenesisNotFound | CRITICAL | Run genesis ceremony |
| HardwareRootMismatch | CRITICAL | Re-genesis with new identity |
| ChainIntegrityFailure | HIGH | Investigate tampering, restore from backup |
| TokenChainCorrupted | HIGH | Rebuild SQLite from JSONL source of truth |
| NoInferenceBackend | MEDIUM | Start LM Studio / Ollama, retry in 30s |
| Z3SelfTestFailed | MEDIUM | Reinstall z3-solver package |
| SkillRegistrationFailed | LOW | Skip skill, log warning, continue boot |

---

## TDD Anchors

| Test | File | Validates |
|------|------|-----------|
| `test_boot_sequence_order` | `tests/core/sovereign/test_runtime_integration.py` | Phases execute in order |
| `test_boot_no_genesis` | `tests/core/sovereign/test_runtime_core.py` | Missing genesis halts boot |
| `test_boot_no_backend` | `tests/core/sovereign/test_runtime_core.py` | No inference backend halts boot |
| `test_graceful_shutdown` | `tests/core/sovereign/test_runtime_core.py` | All resources cleaned up |
| `test_health_check` | `tests/e2e_http/test_api_health.py` | /v1/health returns correct status |
| `test_boot_tamper_detection` | `tests/core/sovereign/test_tamper_evident_log.py` | Corrupted log stops boot |
| `test_runtime_core_pipeline` | `tests/core/sovereign/test_runtime_core_pipeline.py` | Full pipeline after boot |

---

## Observability

### Boot Telemetry

Every boot produces a tamper-evident log entry with:
- Total boot duration (ms)
- Per-phase duration breakdown
- Number of backends online
- Number of active goals loaded
- Chain lengths (tamper log, token ledger, experience ledger)
- Any warnings (MUTABLE tier changes, skills skipped)

### Runtime Telemetry

After boot, the following metrics are continuously collected:

| Metric | Frequency | Source |
|--------|-----------|--------|
| CPU/Memory/GPU usage | 30s | MetricsCollector |
| Inference latency (P50/P95/P99) | Per-request | InferenceGateway |
| SNR/Ihsan scores | Per-response | SovereignEngine |
| Gini coefficient | Per-epoch | SAT Controller |
| Circuit breaker state | On transition | InferenceGateway |
| Reconciliation cycle time | Per-cycle | RecursiveLoop |
| Chain length (all ledgers) | On append | Ledgers |

---

*Source of truth: `core/sovereign/__main__.py`, `core/sovereign/launch.py`, `core/sovereign/runtime_core.py`, `bizra-omega/bizra-cli/src/main.rs`*
