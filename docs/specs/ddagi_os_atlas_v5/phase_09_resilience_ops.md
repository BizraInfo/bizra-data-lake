# Phase 09 — Resilience & Operations: Self-Healing, Security, DevOps, API, BIZRA Box

> Source: Atlas v5.0 — Diagrams D21 (Security Threats), D24 (Self-Healing), D25 (DevOps), D26 (API Layer), D27 (BIZRA Box)
> Status: SPECIFICATION SEALED | SNR: 0.95

---

## 1. Functional Requirements

### FR-090: Security Threat Model

Four threat vectors, all fail-closed (`GATE_FAIL_MODE`). All incidents produce signed receipts.

**Threat 1 -- Prompt Injection Chain.** Adversary injects directives into upstream context to bypass FATE gates. Mitigations: (1) PSI-AST whitelist walk -- unknown constructs trigger `PSI = 0` (FR-041). (2) Sentinel entropy anomaly detection -- inputs diverging > 2 sigma quarantined before reaching any expert. (3) Human confirm required for all destructive/irreversible actions (AEGIS Gray Zone, FR-061).

**Threat 2 -- Malicious Skill Injection.** Adversary submits skill with hidden destructive behavior. Mitigations: (1) Ed25519 skill signing -- unsigned packages rejected. (2) Verify chain to constitutional trust root (`BLAKE3(constitution_hash + author_public_key)`); revoked keys checked against BlockGraph. (3) Sandbox probation -- 48h in FATE Layer T with shadow traffic; promotion requires SNR >= 0.85, zero Crown H0/H2 HALTs. (4) Runtime anomaly detection -- resource usage exceeding 3 sigma triggers immediate suspension.

**Threat 3 -- Federated Node Compromise.** Adversary controls a peer, poisons consensus or exfiltrates data. Mitigations: (1) BFT consensus `sat_frontier_quorum()` (2f+1). (2) Reputation-weighted acceptance (`bloom_balance * reputation_multiplier`). (3) Minimum SEED stake with slashing on misbehavior. (4) Auto-quarantine after 3 consecutive PCI gate failures within `FEDERATION_QUARANTINE_WINDOW` (3600s).

**Threat 4 -- Screen Capture Privacy.** HDA captures may contain PII. Mitigations: (1) Auto-redact PII via regex + NER -- matches replaced with `[REDACTED:<type>]`. (2) Zone fencing -- capture restricted to target window bounding rect. (3) Local-only flag -- bytes never transmitted externally, stored in ephemeral tmpfs with TTL = 300s. (4) Differential privacy (epsilon = 1.0) on aggregated UI metrics before cross-node sharing.

### FR-091: Self-Healing

**Detection.** Health endpoints polled every 5s (`{status, uptime_s, last_error, ihsan_score}`). Brain/Body heartbeat at 1s (3 misses = dead). Sliding-window (60s) anomaly detection on latency, error rate, memory. Threshold alerts on SNR < 0.85 or Ihsan < 0.85.

**Circuit Breaker (Nygard 2007).** Three states: CLOSED, OPEN, HALF_OPEN. CLOSED->OPEN: 3 failures in 60s. OPEN->HALF_OPEN: after `recovery_timeout` (30s). HALF_OPEN->CLOSED: 2 consecutive successes. HALF_OPEN->OPEN: any failure, timeout doubles (max 300s). Every subsystem maintains independent breaker state.

**Graceful Degradation.**

| Tier | Name       | Condition                     | Behavior                                  |
|------|------------|-------------------------------|-------------------------------------------|
| 1    | Full       | All subsystems healthy        | Full inference, HDA, federation, proactive |
| 2    | Fast Model | Primary inference OPEN        | FastExpert fallback (Phi-3 / Gemma 2B)    |
| 3    | Cache Only | All inference circuits OPEN   | ReflexCache (System-1) only               |
| 4    | Offline    | Cache miss or partition       | Dead letter queue, notify user             |

Monotonic-down during incident. Recovery walks up one tier at a time.

**Dead Letter Queue.** Transient: exponential backoff (1s..60s, max 5 retries, jitter [0, 0.5*backoff]). Permanent (auth fail, schema violation, constitutional DENY): escalate to human via Sentinel + Ghost Panel. All entries logged as typed receipts.

**SAT Healer Agent.** Five-phase cycle: (1) Diagnose -- collect health, breaker states, error receipts; rank failure hypotheses. (2) Prescribe -- match hypothesis to `RepairAction` from catalog (restart, clear cache, reduce concurrency, rotate credentials). (3) Execute -- dispatch via action bus, FATE-gated; destructive repairs require human confirmation. (4) Verify -- re-poll health, confirm breakers closing, measure SNR delta. (5) Learn -- record outcome to autopoietic loop (`core/autopoiesis/loop_engine.py`).

### FR-092: DevOps

**Kubernetes.** Namespace `bizra`. HPA: elite 3-10 pods, omega 3-15 pods, MCP gateway 2-6 pods; CPU 70%, memory 80%. PDB: `minAvailable=2` (elite, omega), `minAvailable=1` (MCP). Rolling updates: `maxSurge=1`, `maxUnavailable=0`.

**Containers.** SovereignRuntime (`bizra-elite`), OmegaService (`bizra-omega`), EmbeddingService (`bizra-embed`), InferenceBackend (LM Studio/Ollama), AgentDB (PostgreSQL 16 + pgvector). All non-root (uid=1000), seccomp RuntimeDefault, read-only rootfs, cosign-signed images.

**CI/CD.** `commit -> lint -> test (6781) -> build -> scan (Trivy+bandit+cargo-audit) -> canary (5%, 72h) -> promote (Crown verdict)`. All gates fail-closed. Coverage floor: 38%. Canary: zero H0 HALTs, SNR drop < 15%.

**Observability.** Structured JSON logs (no PII). OpenTelemetry traces across Python/Rust via `X-BIZRA-TRACE-ID`. PagerDuty (P1: primary inference OPEN, H0 HALT), Slack (P2: degraded tier). Grafana: System Health, Inference, Constitutional, Federation dashboards.

### FR-093: API Layer

**MCP Servers.**

| Server     | Tools                                 | Permission Scope         |
|------------|---------------------------------------|--------------------------|
| Core       | bash, edit, view, create, grep, glob  | READ + WRITE + EXECUTE   |
| FileSystem | smart_file_manager ops                | READ + WRITE             |
| Web        | web_fetch, web_search, browser        | READ + NETWORK           |
| Code       | git, lint, test, build, deploy        | READ + WRITE + EXECUTE   |
| Custom     | user-defined via skill registry       | Per-skill AEGIS zone     |

Auth: bearer token (`X-BIZRA-TOKEN`), rate-limited per client (`ACTION_BUS_MAX_PER_HOUR`).

**A2A Protocol.** (1) Discover -- capability cards via gossip (Ed25519-signed, mDNS + DHT). (2) Contract -- `TaskContract` with SEED budget, FATE-gated acceptance. (3) Invoke -- executes under receiver's FATE gate + Ihsan wall, WebSocket streaming. (4) Receipt -- bilateral signature, BlockGraph anchor, PoI minted.

**LM Studio / Ollama.** `/v1/chat/completions`, `/v1/embeddings` (OpenAI-compatible). SSE streaming. LM Studio at gateway:1234 -> Ollama at localhost:11434 -> cloud fallback. Circuit breaker per backend. Auth via `LM_API_TOKEN`.

**AHK Bridge.** Script gen -> PSI-AST validate -> JSON-RPC execute (TCP:9742, 8 HDA verbs) -> closed-loop verify -> signed receipt.

**External Tools.** Playwright (CDP, Gray Zone), IMAP/SMTP (TLS, Gray Zone), CalDAV (HTTPS, Rules), S3 (SigV4, Rules), Webhooks (HMAC, Rules). TLS 1.3 minimum. Credentials from env vars only.

**Auth.** Bearer token (HMAC-SHA256 over timestamp+nonce). DID challenge-response for federation. Scoped permissions (read, write, execute, network, destructive). Token bucket rate limiting (10/s, burst 20, HTTP 429).

### FR-094: BIZRA Box

**Hardware.** ARM64/x86-64, RTX 4060 (16 GB) or Apple Silicon, 32 GB RAM, 2 TB NVMe, 2.5 GbE + WiFi 6E.

**Pre-Installed.** Hardened Linux (Ubuntu 24.04 LTS, CIS L2), Docker + K3s, 5 Ollama models (~80 GB), SovereignRuntime v3.0.0, auto-generated Ed25519 identity at first boot.

**Zero-Config.** `Unbox -> Plug in -> mDNS discover -> https://bizra.local -> Name node -> Designate Guardians -> Ready (< 10 min)`. No cloud account, no telemetry.

**Auto-Maintenance.** Signed OTA with A/B rollback partition (user can defer). Nightly AES-256-GCM encrypted backup (Argon2id KDF). Continuous health monitoring (`deploy/node0/health-check.py`). Ghost Panel badges (amber=degraded, red=critical).

**Sovereignty.** Zero cloud (federation opt-in, Noise-encrypted). Local-first (partition = full function). User-owned (AGPL-3.0, open source). Physical kill switch (no WoL/remote boot).

---

## 2. Edge Cases

**EC-090: Cascading Circuit Breaker Storm.** All breakers trip simultaneously. (1) Sentinel detects >= 3 OPEN within 10s, triggers coordinated recovery. (2) Immediate drop to Tier 4. (3) Dead letter queue absorbs all work. (4) Guardian notified with `DiagnosisReport`.

**EC-091: SAT Healer Repair Loop.** Repair causes new failure, creating oscillation. (1) Same `repair_id` appearing 3 times in 300s suspends healer, escalates to human. (2) All repairs FATE-gated (Layer T). (3) Max repair depth: 3 chained repairs.

**EC-092: OTA Update Bricks BIZRA Box.** (1) A/B partition -- update writes to inactive partition. (2) Watchdog: health check fails within 120s triggers rollback. (3) Guardian notification. (4) USB recovery image as last resort.

**EC-093: MCP Tool Escapes Sandbox.** (1) Isolated containers with seccomp + AppArmor. (2) File access via declared bind mounts only. (3) Syscall anomaly monitoring. (4) Breach = kill + alert + quarantine.

**EC-094: Split-Brain Federation Partition.** (1) Strict majority quorum (`> n/2`). (2) Minority enters read-only mode. (3) On heal, minority replays majority chain, CRDT merge for non-conflicting state.

---

## 3. Pseudocode

### 3.1 circuit_breaker_tick(subsystem)

```
FUNCTION circuit_breaker_tick(subsystem: Subsystem, cb: CircuitBreaker, now: Timestamp) -> TickResult:
    MATCH cb.state:
        CASE CLOSED:
            recent_failures = cb.failures_in_window(now - cb.failure_window, now)
            IF recent_failures >= cb.failure_threshold:
                cb.state = OPEN; cb.opened_at = now
                cb.current_recovery_timeout = cb.base_recovery_timeout
                emit_alert(P2, f"{subsystem.name} circuit breaker OPEN")
                RETURN TickResult(TRIPPED)
            RETURN TickResult(HEALTHY)
        CASE OPEN:
            elapsed = now - cb.opened_at
            IF elapsed >= cb.current_recovery_timeout:
                cb.state = HALF_OPEN; cb.probe_successes = 0
                RETURN TickResult(PROBING)
            RETURN TickResult(OPEN, retry_in=cb.current_recovery_timeout - elapsed)
        CASE HALF_OPEN:
            probe = subsystem.health_check()
            IF probe.healthy:
                cb.probe_successes += 1
                IF cb.probe_successes >= cb.success_threshold:
                    cb.state = CLOSED; cb.reset_counters()
                    RETURN TickResult(RECOVERED)
                RETURN TickResult(PROBING, successes=cb.probe_successes)
            ELSE:
                cb.state = OPEN; cb.opened_at = now
                cb.current_recovery_timeout = MIN(cb.current_recovery_timeout * 2, 300)
                RETURN TickResult(RETRIPPED)
```

### 3.2 degrade_tier(current_tier, failure)

```
FUNCTION degrade_tier(current_tier: DegradationTier, failure: FailureEvent,
                      breaker_states: Dict[str, CircuitState]) -> DegradationTier:
    inference_open = ANY(breaker_states[b] == OPEN FOR b IN INFERENCE_BREAKERS)
    all_inference_open = ALL(breaker_states[b] == OPEN FOR b IN INFERENCE_BREAKERS)

    IF NOT inference_open:                          target = TIER_1_FULL
    ELIF NOT all_inference_open:                    target = TIER_2_FAST_MODEL
    ELIF reflex_cache.is_available():               target = TIER_3_CACHE_ONLY
    ELSE:                                           target = TIER_4_OFFLINE

    # Monotonic-down during incident; recover one tier at a time
    IF target.value > current_tier.value:           new_tier = target
    ELIF target.value < current_tier.value:         new_tier = DegradationTier(current_tier.value - 1)
    ELSE:                                           new_tier = current_tier

    IF new_tier != current_tier:
        emit_metric("degradation_tier", new_tier.value)
        IF new_tier == TIER_4_OFFLINE:
            emit_alert(P1, "Node entered OFFLINE tier")
    RETURN new_tier
```

### 3.3 heal(diagnosis)

```
FUNCTION heal(diagnosis: DiagnosisReport, healer: SATHealer, action_bus: ActionBus) -> HealResult:
    # Guard: repair loop detection
    IF len(healer.repairs_in_window(diagnosis.subsystem, 300)) >= 3:
        healer.suspend()
        escalate_to_human("repair_loop_detected", subsystem=diagnosis.subsystem)
        RETURN HealResult(ESCALATED, "repair_loop_limit_reached")

    hypotheses = sorted(diagnosis.hypotheses, key=lambda h: h.confidence, reverse=True)
    IF len(hypotheses) == 0: RETURN HealResult(NO_DIAGNOSIS)

    FOR hypothesis IN hypotheses:
        repair = healer.repair_catalog.match(hypothesis)
        IF repair IS None: CONTINUE

        # FATE-gate the repair
        IF fate_gate_check(repair.as_action(), healer.rsl, healer.crown).verdict == DENY: CONTINUE

        # Destructive repairs need human confirmation
        IF repair.is_destructive:
            IF NOT await_human_confirmation(repair, UNIFIED_AGENT_TIMEOUT_MS): CONTINUE

        receipt = action_bus.dispatch(repair.as_action())
        IF receipt.status != SUCCESS: CONTINUE

        sleep(repair.settle_time_ms OR 5000)
        post_health = diagnosis.subsystem.health_check()

        # Learn from outcome
        healer.autopoietic_loop.record(RepairOutcome(
            hypothesis, repair, post_health.healthy, post_health.snr - diagnosis.pre_snr))

        IF post_health.healthy:
            RETURN HealResult(HEALED, repair=repair.name)

    escalate_to_human("all_repairs_exhausted", diagnosis=diagnosis)
    RETURN HealResult(ESCALATED, "all_repair_hypotheses_failed")
```

### 3.4 mcp_route(tool_call)

```
FUNCTION mcp_route(tool_call: MCPToolCall, auth: AuthContext, aegis: AEGISEngine) -> MCPResult:
    IF NOT auth.verify_token(tool_call.bearer_token):
        RETURN MCPResult(DENIED, "auth_invalid", 401)
    IF NOT rate_limiter.try_acquire(auth.client_id):
        RETURN MCPResult(DENIED, "rate_limited", 429, retry_after=rate_limiter.next_available())

    server = mcp_registry.resolve(tool_call.tool_name)
    IF server IS None: RETURN MCPResult(DENIED, "unknown_tool", 404)

    FOR perm IN server.required_permissions(tool_call.tool_name):
        IF NOT auth.has_permission(perm):
            RETURN MCPResult(DENIED, f"missing_permission: {perm.name}", 403)

    zone = aegis.classify(tool_call.as_action())
    IF zone == BOUNDS: RETURN MCPResult(DENIED, "aegis_bounds_violation", 403)
    IF zone == GRAY:
        IF NOT await_human_confirmation(tool_call, UNIFIED_AGENT_TIMEOUT_MS):
            RETURN MCPResult(DENIED, "human_denied_gray_zone", 403)

    span = tracer.start_span("mcp_execute", tool=tool_call.tool_name)
    TRY:
        result = server.execute(tool_call.tool_name, tool_call.arguments)
    EXCEPT TimeoutError: RETURN MCPResult(ERROR, "tool_timeout", 504)
    EXCEPT Exception AS e: RETURN MCPResult(ERROR, f"tool_error: {type(e).__name__}", 500)
    FINALLY: span.end()

    evidence_ledger.append(create_receipt(tool_call, result, auth.client_id))
    RETURN MCPResult(SUCCESS, data=result, 200)
```

---

## 4. TDD Anchors

```
TEST circuit_breaker_trips_after_threshold:
    cb = CircuitBreaker(failure_threshold=3, failure_window=60, recovery_timeout=30)
    FOR i IN 1..3: cb.record_failure(now)
    ASSERT circuit_breaker_tick(mock_sub, cb, now).status == TRIPPED
    ASSERT cb.state == OPEN

TEST circuit_breaker_recovers_through_half_open:
    cb = make_open_breaker(opened_at=now - 31)
    circuit_breaker_tick(mock_healthy_sub, cb, now)
    ASSERT cb.state == HALF_OPEN
    circuit_breaker_tick(mock_healthy_sub, cb, now + 1)
    circuit_breaker_tick(mock_healthy_sub, cb, now + 2)
    ASSERT cb.state == CLOSED

TEST degradation_walks_tiers_monotonically:
    t2 = degrade_tier(TIER_1_FULL, fail, {"primary": OPEN, "fallback": CLOSED})
    ASSERT t2 == TIER_2_FAST_MODEL
    t3 = degrade_tier(t2, fail, {"primary": OPEN, "fallback": OPEN})
    ASSERT t3 == TIER_3_CACHE_ONLY

TEST healer_escalates_on_repair_loop:
    healer = make_healer(recent_repairs=3)
    result = heal(mock_diagnosis, healer, mock_bus)
    ASSERT result.status == ESCALATED AND "repair_loop" IN result.reason

TEST healer_fate_gates_repair_actions:
    mock_fate_gate(verdict=DENY)
    result = heal(mock_diagnosis, healer, mock_bus)
    ASSERT result.status == ESCALATED AND mock_bus.dispatch_count == 0

TEST mcp_route_rejects_missing_permission:
    auth = make_auth(permissions=[READ])
    result = mcp_route(make_tool_call("bash", "ls"), auth, aegis)
    ASSERT result.http_status == 403 AND "missing_permission" IN result.reason

TEST mcp_route_rate_limits_per_client:
    FOR i IN 1..ACTION_BUS_MAX_PER_HOUR: mcp_route(make_tool_call("glob"), auth, aegis)
    ASSERT mcp_route(make_tool_call("glob"), auth, aegis).http_status == 429

TEST bizra_box_first_boot_generates_identity:
    box = simulate_first_boot(BIZRA_BOX_SPEC)
    ASSERT box.keypair.algorithm == "Ed25519"
    ASSERT box.node_id.startswith("did:bizra:")
    ASSERT box.telemetry_enabled == False
```

---

## 5. Cross-References

### Python Modules

- `core/inference/_resilience.py` -- `CircuitBreaker`, `CircuitBreakerError`. Config: `CircuitBreakerConfig` (failure_threshold=5, recovery_timeout=30).
- `core/inference/_types.py` -- `CircuitState`, `CircuitBreakerConfig`, `RateLimiterConfig`.
- `core/inference/gateway.py` -- `InferenceGateway`. Circuit breaker + rate limiter per backend.
- `core/autopoiesis/loop_engine.py` -- `AutopoieticState`, 7-phase improvement cycle. SAT Healer feeds outcomes here.
- `core/autopoiesis/shadow_deploy.py` -- `DeploymentVerdict`. Shadow/canary testing for CI gate.
- `core/bridges/desktop_bridge.py` -- JSON-RPC (TCP:9742). Auth via `msg["headers"]`.
- `core/bridges/ghost_ws.py` -- WebSocket (port 9743). RPC proxy, auth injection, Ghost Panel alerts.
- `core/a2a/transport.py` -- `A2ATransport`. HTTP REST, WebSocket, UDP gossip.
- `core/skills/mcp_bridge.py` -- `MCPPermission`, `MCPToolCategory`, `SkillToolMapping`.
- `core/sovereign/mcp_disclosure.py` -- MCP capability disclosure for peer discovery.
- `core/integration/constants.py` -- `GATE_FAIL_MODE` ("closed"), `ACTION_BUS_MAX_PER_HOUR` (100), `UNIFIED_IHSAN_THRESHOLD` (0.95), `UNIFIED_SNR_THRESHOLD` (0.85), `ROLLBACK_SNR_DROP_THRESHOLD` (0.15).

### Rust Crates

- `bizra-omega/bizra-hooks/` -- EventBus (8 shards). Health/sentinel event namespaces.
- `bizra-omega/bizra-agent/src/omni_kernel.rs` -- OmniKernel. ReflexCache for Tier 3 degradation.
- `bizra-omega/bizra-agent/src/reflex_cache.rs` -- `ReflexCache`. O(1) System-1 lookup.
- `bizra-omega/bizra-autopoiesis/` -- Rust-side self-healing counterpart.
- `bizra-omega/bizra-api/` -- REST API, OpenTelemetry, MCP gateway backend.
- `bizra-omega/bizra-federation/` -- Gossip, BFT consensus, reputation, auto-quarantine.
- `bizra-omega/bizra-core/src/lib.rs` -- `IHSAN_THRESHOLD` (0.95), `SNR_THRESHOLD` (0.85).

### Deploy Artifacts

- `deploy/k8s/base/hpa.yaml` -- HPA + PDB definitions for all deployments.
- `deploy/k8s/base/deployment-elite.yaml` -- Non-root container, seccomp, Prometheus annotations.
- `deploy/k8s/canary/` -- Canary deployment and rollback scripts.
- `deploy/monitoring/` -- Prometheus alerting rules, Grafana dashboard.
- `deploy/node0/health-check.py` -- Continuous health monitoring (exit 0=pass, 1=critical, 2=degraded).

### Atlas v5 Phases

- Phase 00 -- FR-001/003: Sovereignty model, 12-step loop (self-healing keeps the loop alive)
- Phase 01 -- FR-010/013: Constitutional Self-Harness (RSL hash verification feeds healer)
- Phase 02 -- FR-023: G.R.A.S.P. reflex precipitation (Tier 3 fallback substrate)
- Phase 04 -- FR-040/043: HDA, PSI-AST, AHK recovery (threats 1 & 4 originate here)
- Phase 05 -- FR-050/051: BlockGraph + PoI (repair receipts, OTA provenance)
- Phase 06 -- FR-060/065: FATE Gate, AEGIS, Crown, governance (repairs are FATE-gated; canary reuses progressive gates)
- Phase 07 -- FR-070/076: Federation, BFT, reputation (threat 3 mitigations)
- Phase 08 -- FR-080/082: MoE, confidence cascade (degradation tiers map to MoE fallback chain)

### Standing on Giants

- Nygard (2007): Release It! -- Circuit breaker, graceful degradation
- Netflix (2012): Hystrix -- Production circuit breaker; Chaos Engineering
- Fowler (2010): Canary releases -- Progressive deployment gates
- Shannon (1948): Information entropy -- SNR as health quality metric
- Deming (1986): PDCA -- SAT Healer diagnose-prescribe-execute-verify-learn cycle
- Maturana & Varela (1972): Autopoiesis -- Self-creating, self-healing systems
- Lamport (1982): Byzantine Fault Tolerance -- Federation consensus
- Saltzer & Schroeder (1975): Fail-safe defaults -- GATE_FAIL_MODE = "closed"
- Anthropic (2023): Model Context Protocol -- MCP server architecture
- Google (2016): Site Reliability Engineering -- Observability, error budgets
- Al-Ghazali (1095): Ihsan -- Excellence as the minimum quality floor
