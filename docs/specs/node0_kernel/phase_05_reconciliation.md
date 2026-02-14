# Phase 05: Reconciliation Loop + Audit Surface

Last updated: 2026-02-14
Standing on: Boyd (1995, OODA) · Deming (1950, PDCA) · Maturana & Varela (1972, Autopoiesis)

---

## Purpose

Layers 5-6 close the loop. The reconciliation engine continuously compares **current state** (C) against **desired state** (D) and generates hypotheses to close the gap. The audit surface exposes every internal operation to human inspection via REST API, WebSocket, CLI dashboard, and Prometheus metrics.

Together they answer:
1. **What should be improved?** — AutoResearcher generates hypotheses
2. **Is the improvement valid?** — AutoEvaluator verifies against CLEAR framework
3. **Should we deploy it?** — FATE gate + circuit breaker guard execution
4. **Can humans see everything?** — Full API + metrics + CLI dashboard

---

## Reconciliation Loop (OODA)

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   RECONCILIATION LOOP                         │
│                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │ OBSERVE  │───▶│  ORIENT  │───▶│  DECIDE  │             │
│   │ (Gaps)   │    │ (Hypo)   │    │ (Eval)   │             │
│   └──────────┘    └──────────┘    └──────────┘             │
│        ▲                               │                     │
│        │                               ▼                     │
│   ┌──────────┐                    ┌──────────┐             │
│   │  LEARN   │◀───────────────────│   ACT    │             │
│   │ (Ledger) │                    │ (Deploy) │             │
│   └──────────┘                    └──────────┘             │
│                                                              │
│   Circuit Breaker: 3 consecutive rejections → backoff       │
│   FATE Gate: Z3 verification BEFORE any deployment          │
│   Autopoietic State Machine: 7 states, formal transitions   │
└──────────────────────────────────────────────────────────────┘
```

### Data Structures

```pseudocode
STRUCT RecursiveLoop:
    auto_researcher:   AutoResearcher
    auto_evaluator:    AutoEvaluator
    circuit_breaker:   CircuitBreaker
    metrics:           LoopMetrics
    shutdown_signal:   AsyncEvent       # Graceful shutdown
    mode:              LoopMode         # CONTINUOUS | SINGLE_PASS | PATTERN_AWARE

STRUCT LoopMetrics:
    total_cycles:         u64
    hypotheses_generated: u64
    hypotheses_approved:  u64
    hypotheses_rejected:  u64
    consecutive_rejects:  u32    # Circuit breaker input
    avg_cycle_time_ms:    f64
    last_cycle_at:        Timestamp

ENUM LoopMode:
    CONTINUOUS     # Run indefinitely until shutdown signal
    SINGLE_PASS    # Run one cycle and stop
    PATTERN_AWARE  # Use Sci-Reasoning patterns to guide hypothesis generation
```

**Source:** `core/spearpoint/recursive_loop.py` (425 lines)

### AutoResearcher

```pseudocode
STRUCT AutoResearcher:
    hypothesis_generator: HypothesisGenerator
    knowledge_graph:      KnowledgeGraph
    experience_ledger:    EpisodeLedger

    METHOD research(gap: StateGap) -> ResearchResult:
        # Standing on: Boyd (1995) — Observe + Orient phases

        # Step 1: Retrieve relevant past episodes
        relevant = self.experience_ledger.retrieve(gap.description, k=5)

        # Step 2: Generate hypotheses
        hypotheses = self.hypothesis_generator.generate(
            gap         = gap,
            context     = relevant,
            max_count   = 3,
        )

        # Step 3: Rank by expected impact
        ranked = sort_by(hypotheses, key=lambda h: h.expected_poi)

        # Step 4: Return top hypothesis for evaluation
        # IMPORTANT: AutoResearcher NEVER evaluates.
        # It generates candidates. AutoEvaluator decides.
        RETURN ResearchResult(
            gap        = gap,
            hypotheses = ranked,
            outcome    = PENDING,   # Not yet evaluated
        )

ENUM ResearchOutcome:
    PENDING       # Awaiting evaluation
    APPROVED      # Evaluator approved
    REJECTED      # Evaluator rejected
    INCONCLUSIVE  # Evaluator could not determine
    GATED         # Blocked by FATE gate
```

**Source:** `core/spearpoint/auto_researcher.py` (652 lines)

### AutoEvaluator

```pseudocode
STRUCT AutoEvaluator:
    clear_framework:  CLEARFramework    # Citation, Logic, Evidence, Alternatives, Reproducibility
    guardrails:       Guardrails        # Safety boundaries
    ihsan_gate:       IhsanGate         # Excellence threshold

    METHOD evaluate(hypothesis: Hypothesis) -> EvaluationResult:
        # Standing on: scientific method — falsification principle

        # Step 1: CLEAR framework check
        clear_score = self.clear_framework.score(hypothesis)

        # Step 2: Guardrails check (safety)
        safe = self.guardrails.check(hypothesis)
        IF NOT safe:
            RETURN EvaluationResult(verdict=REJECTED, reason="Safety guardrail violation")

        # Step 3: Ihsan gate
        ihsan = self.ihsan_gate.score(hypothesis)
        IF ihsan < IHSAN_THRESHOLD:
            RETURN EvaluationResult(verdict=REJECTED, reason="Ihsan below threshold")

        # Step 4: Determine verdict
        IF clear_score.credibility >= 0.80:
            verdict = SUPPORTED
            tier = TierDecision.DEPLOY
        ELIF clear_score.credibility >= 0.50:
            verdict = INCONCLUSIVE
            tier = TierDecision.SHADOW_DEPLOY
        ELSE:
            verdict = REJECTED
            tier = TierDecision.DISCARD

        # Step 5: Generate signed receipt
        receipt = sign_evaluation_receipt(hypothesis, verdict, clear_score)

        RETURN EvaluationResult(
            verdict     = verdict,
            credibility = clear_score.credibility,
            tier        = tier,
            receipt     = receipt,
        )

    # IMPORTANT: AutoEvaluator is the EXCLUSIVE owner of:
    # - CLEAR framework
    # - Guardrails
    # - Ihsan gate
    # No other module may invoke these directly.

ENUM Verdict:
    SUPPORTED     # Evidence supports hypothesis
    REJECTED      # Evidence refutes hypothesis
    INCONCLUSIVE  # Insufficient evidence

ENUM TierDecision:
    DEPLOY         # Safe to deploy to production
    SHADOW_DEPLOY  # Deploy in shadow mode (observe only)
    DISCARD        # Discard hypothesis
```

**Source:** `core/spearpoint/auto_evaluator.py` (533 lines)

### Autopoietic Loop Engine

```pseudocode
STRUCT AutopoieticLoopEngine:
    state:            AutopoieticState
    z3_gate:          Z3FATEGate
    hypothesis_gen:   HypothesisGenerator
    risk_assessor:    RiskAssessor

    # 7-state machine (Maturana & Varela 1972)
    # Each transition requires explicit guard conditions

ENUM AutopoieticState:
    DORMANT         # Inactive, waiting for trigger
    OBSERVING       # Collecting system metrics and patterns
    HYPOTHESIZING   # Generating improvement candidates
    VALIDATING      # Z3 FATE gate verification
    IMPLEMENTING    # Shadow deployment of approved change
    INTEGRATING     # Merging shadow results into production
    REFLECTING      # Recording outcome to experience ledger

    # Transitions:
    # DORMANT -> OBSERVING:       trigger_received
    # OBSERVING -> HYPOTHESIZING: sufficient_observations (>= 10 data points)
    # HYPOTHESIZING -> VALIDATING: hypothesis_generated
    # VALIDATING -> IMPLEMENTING:  z3_gate.passed AND risk <= MEDIUM
    # VALIDATING -> DORMANT:       z3_gate.failed (reject)
    # IMPLEMENTING -> INTEGRATING: shadow_results.positive
    # IMPLEMENTING -> DORMANT:     shadow_results.negative (rollback)
    # INTEGRATING -> REFLECTING:   always (record outcome)
    # REFLECTING -> DORMANT:       always (cycle complete)

ENUM HypothesisCategory:
    PERFORMANCE   # Latency, throughput improvements
    QUALITY       # SNR, Ihsan score improvements
    EFFICIENCY    # Resource usage reduction
    ROBUSTNESS    # Error rate reduction
    CAPABILITY    # New functionality
    STRUCTURAL    # Architecture improvements

ENUM RiskLevel:
    NEGLIGIBLE    # No impact if wrong
    LOW           # Easily reversible
    MEDIUM        # Requires human review
    HIGH          # Requires human approval
    CRITICAL      # Requires SAT Council vote

    # INVARIANT: risk >= HIGH => human_approved == true (from Z3 constraint)
```

**Source:** `core/autopoiesis/loop_engine.py` (1,949 lines)

### SpearPoint Pipeline (Post-Query)

```pseudocode
STRUCT SpearPointPipeline:
    # 8-step fire-and-forget pipeline executed AFTER every successful query
    # Each step is independent — failure of one does not block others

    METHOD execute(query: Query, response: Response) -> PipelineResult:
        results = {}

        # Step 1: Generate Graph Artifact (GoT snapshot)
        results["graph_artifact"] = TRY graph_artifact_step(query, response)

        # Step 2: Generate Evidence Receipt (BLAKE3-signed)
        results["evidence_receipt"] = TRY evidence_receipt_step(response)

        # Step 3: Record Impact (PoI scoring)
        results["record_impact"] = TRY record_impact_step(response)

        # Step 4: PoI Contribution (token distribution trigger)
        results["poi_contribution"] = TRY poi_contribution_step(response)

        # Step 5: Living Memory Update (knowledge graph)
        results["living_memory"] = TRY living_memory_step(query, response)

        # Step 6: Experience Ledger Commit (episodic memory)
        results["experience_ledger"] = TRY experience_ledger_step(query, response)

        # Step 7: Judgment Observation (telemetry)
        results["judgment_observe"] = TRY judgment_observe_step(response)

        # Step 8: SAT Health Check
        results["sat_health"] = TRY sat_health_step()

        RETURN PipelineResult(steps=results)

    # Each step wrapped in TRY — errors logged but do not propagate
    # This ensures the user gets their response immediately
    # Pipeline runs asynchronously after response delivery
```

**Source:** `core/sovereign/spearpoint_pipeline.py` (639 lines)

---

## Audit Surface

### REST API Endpoints

```pseudocode
# All endpoints require Bearer token authentication (X-API-Key header)
# Rate limited: 100 req/min per key

# ─── CORE ENDPOINTS ───
POST   /v1/query              # Submit query → 6-stage pipeline
GET    /v1/status             # Runtime status (mode, uptime, health)
GET    /v1/health             # Health check (200 OK or 503)
GET    /v1/metrics            # Prometheus-format metrics export

# ─── EXPERIENCE LEDGER (SEL) ───
GET    /v1/sel/episodes       # List episodes (paginated: ?page=1&per_page=20)
GET    /v1/sel/episodes/{H}   # Get episode by content hash
POST   /v1/sel/retrieve       # RIR retrieval: {"query": "...", "k": 5}
GET    /v1/sel/verify         # Full chain integrity verification

# ─── JUDGMENT TELEMETRY (SJE) ───
GET    /v1/judgment/stats     # Verdict distribution (SUPPORTED/REJECTED/INCONCLUSIVE)
GET    /v1/judgment/stability # Stability check (entropy, oscillation detection)
POST   /v1/judgment/simulate  # Simulate epoch with mock data

# ─── STREAMING ───
WS     /v1/stream             # WebSocket: real-time reasoning trace
```

**Source:** `core/sovereign/api.py` (2,515 lines)

### Metrics Collector

```pseudocode
STRUCT MetricsCollector:
    series:   Map<String, MetricSeries>   # 1000-point deque per metric

    # System metrics (collected every 30s):
    #   cpu_percent, memory_percent, gpu_utilization, gpu_memory
    #
    # Inference metrics (per-request):
    #   inference_latency_ms, tokens_per_second
    #   snr_score, ihsan_score (per response)
    #
    # Federation metrics:
    #   peer_count, gossip_messages_per_minute
    #   consensus_round_time_ms
    #
    # Token metrics:
    #   seed_minted_total, zakat_distributed_total
    #   gini_coefficient

    METHOD export_prometheus() -> String:
        # Standard Prometheus text format
        # TYPE, HELP, metric_name{labels} value timestamp
        lines = []
        FOR EACH (name, series) IN self.series:
            lines.append("# TYPE {} gauge".format(name))
            lines.append("{} {}".format(name, series.latest()))
        RETURN join(lines, "\n")
```

**Source:** `core/sovereign/metrics.py` (583 lines)

### CLI Dashboard (Rust TUI)

```pseudocode
# bizra-omega/bizra-cli/ — ratatui-based terminal UI
# 2,941 lines total across 10 source files

STRUCT CliApp:
    state:            AppState
    inference_backend: InferenceClient
    config:           CliConfig
    theme:            Theme

    # Widgets:
    #   HeaderBar      — Node identity, uptime, version
    #   FateGauge      — Real-time Ihsan/SNR gauges (colored thresholds)
    #   AgentCard      — PAT/SAT agent status (active/idle/error)
    #   StatusBar      — Connection status, last query time

    # Key bindings:
    #   q       — Quit
    #   Tab     — Cycle between panels
    #   Enter   — Submit query
    #   /       — Command mode
    #   ?       — Help overlay
```

**Source:** `bizra-omega/bizra-cli/` (2,941 lines)

### Judgment Telemetry

```pseudocode
STRUCT JudgmentTelemetry:
    verdicts:       RingBuffer<JudgmentVerdict>   # Last 1000 verdicts
    epoch_stats:    Map<EpochId, EpochStats>

    METHOD observe(verdict: Verdict, scores: Scores) -> JudgmentEntry:
        entry = JudgmentEntry(
            verdict    = verdict,
            ihsan      = scores.ihsan,
            snr        = scores.snr,
            timestamp  = now(),
            entropy    = self.compute_entropy(),   # Shannon entropy of verdict distribution
        )
        self.verdicts.push(entry)
        RETURN entry

    METHOD check_stability() -> StabilityReport:
        # Detect oscillation: rapid SUPPORTED/REJECTED cycling
        recent = self.verdicts.last(50)
        transitions = count_transitions(recent)
        entropy = self.compute_entropy()

        stable = entropy < 1.5 AND transitions < 15

        RETURN StabilityReport(
            stable   = stable,
            entropy  = entropy,
            transitions = transitions,
            recommendation = IF stable THEN "CONTINUE" ELSE "INVESTIGATE",
        )
```

**Source:** `core/sovereign/judgment_telemetry.py` (from codebase), `bizra-omega/bizra-core/src/sovereign/judgment_telemetry.rs` (246 lines)

---

## TDD Anchors

| Test | File | Validates |
|------|------|-----------|
| `test_recursive_loop_circuit_breaker` | `tests/core/spearpoint/` | 3 rejects trigger backoff |
| `test_auto_researcher_no_evaluation` | `tests/core/spearpoint/` | Researcher generates, never evaluates |
| `test_auto_evaluator_clear_framework` | `tests/core/spearpoint/` | CLEAR scoring produces credibility |
| `test_autopoietic_state_transitions` | `tests/core/autopoiesis/test_loop_engine.py` | All 7 state transitions valid |
| `test_autopoietic_z3_gate` | `tests/core/autopoiesis/test_loop_engine.py` | FATE gate blocks risky changes |
| `test_spearpoint_pipeline_isolation` | `tests/core/sovereign/test_spearpoint_pipeline.py` | Step failure doesn't block others |
| `test_api_health_endpoint` | `tests/e2e_http/test_api_health.py` | /v1/health returns 200 |
| `test_api_metrics_prometheus` | `tests/core/sovereign/test_api_metrics.py` | Prometheus export format valid |
| `test_judgment_stability` | `tests/core/sovereign/test_judgment_telemetry.py` | Oscillation detected correctly |
| `test_sel_chain_verification` | `tests/core/sovereign/test_experience_ledger.py` | Chain integrity API works |

---

## Failure Modes

| Failure | Layer | Behavior | Recovery |
|---------|-------|----------|----------|
| Circuit breaker opens | Reconciliation | Loop pauses, backoff timer starts | Auto-retry after cooldown |
| Z3 solver timeout | Reconciliation | Hypothesis rejected (fail-closed) | Retry with simplified constraints |
| API rate limit exceeded | Audit | 429 Too Many Requests | Wait and retry |
| Metrics deque full | Audit | Oldest metrics evicted (ring buffer) | No action needed |
| Experience ledger chain break | Audit | verify() returns false | Investigate JSONL file |
| Autopoietic loop stuck | Reconciliation | Watchdog detects, forces DORMANT | Manual investigation |

---

*Source of truth: `core/spearpoint/`, `core/autopoiesis/loop_engine.py`, `core/sovereign/api.py`, `core/sovereign/metrics.py`, `bizra-omega/bizra-cli/`*
