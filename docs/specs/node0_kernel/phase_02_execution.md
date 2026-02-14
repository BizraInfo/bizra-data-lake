# Phase 02: Deterministic Execution Layer

Last updated: 2026-02-14
Standing on: Lampson (1971, Protection) · Dijkstra (1968, Structured Programming) · Besta (2024, GoT)

---

## Purpose

Layer 2 governs **how the node executes**. Every computation runs inside a contract boundary with declared resource limits, timeout enforcement, and deterministic state transitions. No dynamic `eval()`, no untyped boundaries, no implicit side effects.

The execution layer answers:
1. **What are the boundaries?** — ContractBoundary defines resource/permission envelopes
2. **How does reasoning flow?** — SovereignEngine orchestrates a 6-stage pipeline
3. **Who controls resources?** — SAT Controller maintains homeostasis via Gini monitoring

---

## Data Structures

### ExecutionContext

```pseudocode
STRUCT ExecutionContext:
    context_id:     UUID
    node_identity:  NodeIdentity          # From Layer 1
    contract:       ContractBoundary      # Resource + permission envelope
    mode:           EngineMode            # Operating mode
    state:          Map<String, Any>      # Sandboxed state (no global access)
    started_at:     Timestamp
    timeout:        Duration              # Hard kill after this duration
    parent_context: Option<ContextId>     # For nested execution

    INVARIANT: state is isolated — no references to global mutable state
    INVARIANT: timeout <= 60 seconds (configurable, enforced by watchdog)
```

### ContractBoundary

```pseudocode
STRUCT ContractBoundary:
    max_memory_mb:     u32           # Memory ceiling
    max_cpu_time_s:    f64           # CPU time budget
    max_wall_time_s:   f64           # Wall clock timeout
    max_tokens:        u64           # LLM token budget per invocation
    allowed_tools:     Set<ToolId>   # Explicit tool whitelist
    allowed_models:    Set<ModelId>  # Model access whitelist
    network_access:    NetworkPolicy # NONE | LOCAL_ONLY | FEDERATED
    filesystem_access: FSPolicy      # NONE | READ_ONLY | DATA_LAKE_ONLY | FULL

ENUM NetworkPolicy:
    NONE          # No network access (pure computation)
    LOCAL_ONLY    # localhost only (LM Studio, Ollama)
    FEDERATED     # URP peer-to-peer (requires explicit consent)

ENUM FSPolicy:
    NONE           # No filesystem access
    READ_ONLY      # Read data lake, no writes
    DATA_LAKE_ONLY # Read/write within DATA_LAKE_ROOT only
    FULL           # Full access (requires human approval)

    INVARIANT: FULL requires explicit human consent via PAT
    INVARIANT: All paths validated against DATA_LAKE_ROOT (no traversal)
```

**Source:** `core/sovereign/runtime_core.py` (2,496 lines)

### EngineMode

```pseudocode
ENUM EngineMode:
    AUTONOMOUS      # Full self-directed operation (highest trust)
    SUPERVISED      # Human approves irreversible actions
    COLLABORATIVE   # Human and agents work together
    RESTRICTED      # Minimal capabilities (recovery mode)
    MAINTENANCE     # System maintenance only (no inference)

    # Mode transitions require:
    # 1. Current Ihsan score >= 0.95
    # 2. Human consent for AUTONOMOUS or MAINTENANCE
    # 3. SAT Controller health check passes
```

**Source:** `core/sovereign/engine.py:EngineMode` (683 lines)

### SovereignEngine

```pseudocode
STRUCT SovereignEngine:
    identity:         NodeIdentity
    mode:             EngineMode
    inference_gateway: InferenceGateway    # Tiered LLM access
    guardian_council:  GuardianCouncil     # Constitutional review
    got_engine:       GraphOfThoughts     # Multi-path reasoning
    snr_optimizer:    SNROptimizer        # Signal quality maximizer
    metrics:          MetricsCollector    # Observable execution

    # 6-Stage Processing Pipeline
    METHOD process(query: Query) -> Result<Response, RejectionReceipt>:
        RETURN pipeline(
            stage_0_classify,      # Compute tier selection
            stage_1_cache_check,   # Deduplicate known queries
            stage_2_reason,        # Graph-of-Thoughts exploration
            stage_3_infer,         # LLM inference via gateway
            stage_4_snr,           # SNR optimization
            stage_5_guardian,      # Constitutional validation
            stage_6_synthesize,    # Final response assembly
        ).execute(query)
```

**Source:** `core/sovereign/engine.py:SovereignEngine`

---

## Procedures

### 6-Stage Processing Pipeline

```pseudocode
PROCEDURE stage_0_classify(query: Query) -> ComputeTier:
    # Standing on: complexity theory — route by estimated cost
    complexity = estimate_complexity(query)

    IF complexity.tokens < 500 AND complexity.reasoning_depth < 2:
        RETURN ComputeTier.EDGE        # 1.5B model, <500ms
    ELIF complexity.tokens < 4000 AND complexity.reasoning_depth < 5:
        RETURN ComputeTier.LOCAL       # 7B model, <2000ms, GPU required
    ELSE:
        RETURN ComputeTier.POOL        # 70B+ model, federated compute

PROCEDURE stage_1_cache_check(query: Query) -> Option<CachedResponse>:
    # Check experience ledger for semantically similar past queries
    similar = experience_ledger.retrieve_similar(query, threshold=0.92)
    IF similar IS NOT None AND similar.snr_score >= SNR_THRESHOLD:
        RETURN Some(similar.response)
    RETURN None

PROCEDURE stage_2_reason(query: Query, tier: ComputeTier) -> ReasoningGraph:
    # Standing on: Besta (2024) — Graph-of-Thoughts
    # Multi-path exploration, not single-chain reasoning
    graph = GraphOfThoughts()
    graph.seed(query)

    FOR round IN 1..MAX_REASONING_ROUNDS:
        # Generate multiple candidate thoughts
        candidates = graph.expand(breadth=3)

        # Score each candidate
        FOR candidate IN candidates:
            score = snr_score(candidate)
            IF score >= SNR_THRESHOLD:
                graph.add_node(candidate, score=score)
            ELSE:
                graph.prune(candidate, reason="below_snr_threshold")

        # Check convergence
        IF graph.has_consensus(threshold=0.85):
            BREAK

    RETURN graph

PROCEDURE stage_3_infer(graph: ReasoningGraph, tier: ComputeTier) -> RawInference:
    # Route to appropriate inference backend
    # Standing on: circuit breaker pattern (Nygard 2007)
    backend = select_backend(tier)
    # Tiered fallback: LM Studio -> Ollama -> llama.cpp -> fail-closed

    context = graph.synthesize_context()
    raw = backend.complete(context, max_tokens=contract.max_tokens)

    IF raw IS Error:
        circuit_breaker.record_failure(backend)
        IF circuit_breaker.is_open(backend):
            backend = fallback_backend(tier)
            raw = backend.complete(context, max_tokens=contract.max_tokens)

    RETURN raw

PROCEDURE stage_4_snr(raw: RawInference) -> OptimizedResponse:
    # Standing on: Shannon (1948) — maximize signal, minimize noise
    snr = compute_snr(raw)
    IF snr < SNR_THRESHOLD:   # 0.85
        RETURN RejectionReceipt(reason="SNR below threshold", score=snr)

    optimized = snr_optimizer.amplify_signal(raw)
    RETURN optimized

PROCEDURE stage_5_guardian(response: OptimizedResponse) -> ValidatedResponse:
    # Standing on: constitutional governance
    # Guardian Council = multi-agent review panel
    ihsan = compute_ihsan(response)
    IF ihsan < IHSAN_THRESHOLD:   # 0.95
        RETURN RejectionReceipt(reason="Ihsan below threshold", score=ihsan)

    # FATE gate verification (formal)
    fate_result = fate_gate.verify(response)
    IF NOT fate_result.passed:
        RETURN RejectionReceipt(reason=fate_result.rejection_reason)

    RETURN response.with_validation(ihsan=ihsan, fate=fate_result)

PROCEDURE stage_6_synthesize(validated: ValidatedResponse) -> FinalResponse:
    # Wrap in PCI envelope with cryptographic proof
    envelope = PCIEnvelope(
        payload    = validated,
        ihsan      = validated.ihsan,
        snr        = validated.snr,
        signer     = node_identity.public_key,
        signature  = node_identity.sign(canonical_json(validated)),
        timestamp  = now(),
        chain_pos  = tamper_log.next_sequence(),
    )
    RETURN FinalResponse(envelope)
```

**Source:** `core/sovereign/engine.py:process()`, `core/sovereign/runtime_core.py`

### SAT Controller — Homeostasis

```pseudocode
STRUCT SATController:
    gini_threshold:     f64    # 0.40 (from constants.py)
    zakat_rate:         f64    # 0.025 (2.5%)
    check_interval:     Duration  # How often to check resource distribution

    METHOD check_homeostasis(ledger: TokenLedger) -> Option<RebalancingEvent>:
        # Standing on: Ostrom (1990) — common-pool resource governance
        # Standing on: Gini (1912) — inequality measurement

        # Step 1: Compute current Gini coefficient
        balances = ledger.get_all_balances()
        gini = compute_gini_coefficient(balances)

        # Step 2: Check against ADL threshold
        IF gini > self.gini_threshold:
            # Inequality exceeds justice ceiling — trigger rebalancing
            rebalancing = self.compute_rebalancing(balances, gini)
            self.execute_rebalancing(rebalancing, ledger)
            RETURN Some(RebalancingEvent(gini, rebalancing))

        RETURN None

    METHOD compute_gini_coefficient(balances: List<f64>) -> f64:
        # Standard Gini coefficient: 0 = perfect equality, 1 = total inequality
        n = len(balances)
        IF n == 0: RETURN 0.0
        sorted_b = sort(balances)
        numerator = SUM(i * sorted_b[i] for i in 0..n)
        denominator = n * SUM(sorted_b)
        IF denominator == 0: RETURN 0.0
        RETURN (2.0 * numerator) / denominator - (n + 1.0) / n
```

**Source:** `core/sovereign/sat_controller.py:SATController` (358 lines)

### Inference Gateway — Tiered Fallback

```pseudocode
STRUCT InferenceGateway:
    backends:        List<InferenceBackend>    # Ordered by preference
    circuit_breakers: Map<BackendId, CircuitBreaker>
    rate_limiter:    TokenBucket               # 10 req/s, burst 20
    connection_pool: ConnectionPool            # Pre-warmed, health-checked

    METHOD complete(request: InferenceRequest) -> Result<InferenceResponse, Error>:
        # Rate limit check
        IF NOT rate_limiter.acquire():
            RETURN Error("Rate limit exceeded")

        # Try backends in order with circuit breaker protection
        FOR backend IN backends:
            cb = circuit_breakers[backend.id]
            IF cb.state == OPEN:
                CONTINUE   # Skip broken backends

            conn = connection_pool.acquire(backend)
            TRY:
                response = backend.complete(request, conn)
                cb.record_success()
                RETURN Ok(response)
            CATCH Error AS e:
                cb.record_failure()
                connection_pool.release(conn, healthy=false)
                LOG_WARNING("Backend {} failed: {}", backend.id, e)

        # All backends failed — fail-closed
        RETURN Error("All inference backends unavailable")

STRUCT CircuitBreaker:
    state:              State          # CLOSED | OPEN | HALF_OPEN
    failure_count:      u32
    success_count:      u32
    failure_threshold:  u32 = 3        # Opens after 3 failures
    success_threshold:  u32 = 2        # Closes after 2 successes in HALF_OPEN
    timeout:            Duration = 30s # Time before OPEN -> HALF_OPEN
    last_failure_at:    Timestamp

    # Standing on: Nygard (2007) — Release It! circuit breaker pattern
```

**Source:** `core/inference/gateway.py`, `core/inference/_resilience.py`, `core/inference/_connection_pool.py`

---

## Resource Governance

### Compute Tiers

| Tier | Model Size | Latency Target | Hardware | Network |
|------|-----------|----------------|----------|---------|
| EDGE | 1.2-1.5B | <500ms | CPU only | NONE |
| LOCAL | 7-14B | <2000ms | GPU required | LOCAL_ONLY |
| POOL | 70B+ | <10000ms | Federated GPUs | FEDERATED |

**Source:** `core/inference/local_first_config.py`, `bizra_config.py`

### Fallback Chain

```
LM Studio (192.168.56.1:1234)
    ↓ failure
Ollama (localhost:11434)
    ↓ failure
llama.cpp (local binary)
    ↓ failure
FAIL-CLOSED (no cloud fallback without explicit human consent)
```

---

## TDD Anchors

| Test | File | Validates |
|------|------|-----------|
| `test_engine_process_pipeline` | `tests/core/sovereign/test_sovereign_runtime.py` | 6-stage pipeline produces valid output |
| `test_contract_boundary_enforcement` | `tests/core/sovereign/test_runtime_core.py` | Resource limits enforced (timeout, memory) |
| `test_circuit_breaker_state_machine` | `tests/core/inference/test_gateway.py` | CLOSED -> OPEN -> HALF_OPEN transitions |
| `test_rate_limiter` | `tests/core/inference/test_gateway.py` | Token bucket rejects over-limit requests |
| `test_connection_pool_health` | `tests/core/inference/test_gateway.py` | Unhealthy connections evicted |
| `test_gini_coefficient` | `tests/core/sovereign/test_sat_controller.py` | Gini computation matches known distributions |
| `test_rebalancing_trigger` | `tests/core/sovereign/test_sat_controller.py` | Gini > 0.40 triggers rebalancing event |
| `test_engine_modes` | `tests/core/sovereign/test_sovereign_runtime.py` | Mode transitions require Ihsan check |
| `test_snr_maximizer` | `tests/core/sovereign/test_snr_maximizer.py` | SNR optimization improves signal quality |
| `test_got_reasoning` | `tests/core/sovereign/test_got_bridge.py` | GoT generates multi-path reasoning graph |
| `test_batching` | `tests/core/inference/test_batching.py` | Batch size 8, 50ms timeout (12 tests) |

---

## Failure Modes

| Failure | Behavior | Recovery |
|---------|----------|----------|
| All backends down | Fail-closed, return error receipt | Wait for circuit breaker timeout (30s) |
| Timeout exceeded | Hard kill, return timeout receipt | Retry with reduced token budget |
| Memory exceeded | Process killed by OS | Reduce contract.max_memory_mb |
| Ihsan below threshold | Rejection receipt with score | Response discarded, not cached |
| SNR below threshold | Rejection receipt with score | Response discarded, not cached |
| Gini exceeds 0.40 | Automatic zakat-based rebalancing | SAT Controller corrects distribution |

---

*Source of truth: `core/sovereign/runtime_core.py`, `core/sovereign/engine.py`, `core/sovereign/sat_controller.py`, `core/inference/gateway.py`*
