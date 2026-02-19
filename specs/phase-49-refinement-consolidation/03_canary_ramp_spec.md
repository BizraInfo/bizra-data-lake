# Phase 49 Spec — Part 3: Canary Ramp Activation

> Standing on Giants: Fowler (canary releases, 2010) · Nygard (Release It!, 2007) · Shannon (observability, 1948)

## Problem

Phase 46 built three cognitive features (FAISS search, GoT bridge, HMM prediction). Phase 47.1 built the canary routing, metrics, and rollback infrastructure. Phase 48.1 wired metrics and validated the E2E pipeline with 23 integration tests. But **all three features remain at 0%** — they have never been activated in production.

## Prerequisites (All Met)

| Prerequisite | Status | Evidence |
|--------------|--------|----------|
| CanaryRouter deterministic hash routing | Done | 76 rollout tests |
| Phase46Metrics live collection | Done | mcp_gateway.py + sovereign_mcp_server.py |
| HMMCallerGate single-caller isolation | Done | 6 integration tests |
| RollbackEngine 2-breach auto-rollback | Done | 5 integration tests |
| E2E pipeline test (route → metrics → rollback) | Done | 23 tests, full lifecycle |
| Prometheus /metrics/prometheus live data | Done | Live Phase46Metrics export |
| Kill switch precedence (ENABLED=0 overrides PERCENT) | Done | Validated in integration tests |
| CI regression step ROLLOUT-001 | Done | Explicit CI step added |

## Ramp Strategy

Four stages, each with a hold period and success criteria before advancing.

```pseudocode
STAGE_0 = {
    name: "Smoke Validation",
    search_percent: 0,
    got_bridge_percent: 0,
    hmm_percent: 0,
    action: "Run E2E tests + synthetic fault injection",
    success_criteria: "All 23 pipeline tests + 76 rollout tests green",
    hold_period: "N/A — one-time gate",
    status: "PASSED (Phase 48.1)"
}

STAGE_1 = {
    name: "Search Only — 10%",
    search_percent: 10,
    got_bridge_percent: 0,
    hmm_percent: 0,
    action: "Activate FAISS vector search for 10% of MCP queries",
    success_criteria: [
        "search_hit_rate >= 0.5 (at least half of searches return results)",
        "search_latency_p95 < 200ms",
        "zero rollbacks triggered",
        "Prometheus metrics show non-zero counters"
    ],
    hold_period: "4 hours minimum",
    rollback_on: "search_error_rate > 0.1 OR latency_p95 > 500ms"
}

STAGE_2 = {
    name: "Search 50% + GoT 10%",
    search_percent: 50,
    got_bridge_percent: 10,
    hmm_percent: 0,
    action: "Ramp search to 50%, introduce GoT bridge at 10%",
    success_criteria: [
        "All Stage 1 criteria still met at 50%",
        "got_convergence_pass_rate >= 0.7",
        "got_fallback_rate < 0.3",
        "Zero rollbacks"
    ],
    hold_period: "8 hours minimum",
    rollback_on: "got_fallback_rate > 0.5 OR any Stage 1 breach"
}

STAGE_3 = {
    name: "Full Activation",
    search_percent: 100,
    got_bridge_percent: 100,
    hmm_percent: 10,
    action: "Full search + GoT, introduce HMM at 10%",
    success_criteria: [
        "All Stage 2 criteria met",
        "hmm_confidence_p50 >= 0.6",
        "observation_entropy > 0 (system is receiving diverse inputs)",
        "Zero rollbacks"
    ],
    hold_period: "24 hours minimum",
    rollback_on: "hmm_confidence_p50 < 0.4 OR any prior breach"
}
```

## Activation Commands

```pseudocode
FUNCTION activate_stage(stage_number: int):
    """Set env vars for the specified canary stage."""

    stages = {
        1: {"SEARCH": 10, "GOT_BRIDGE": 0, "HMM": 0},
        2: {"SEARCH": 50, "GOT_BRIDGE": 10, "HMM": 0},
        3: {"SEARCH": 100, "GOT_BRIDGE": 100, "HMM": 10},
    }

    config = stages[stage_number]
    FOR component, percent IN config:
        env_key = f"BIZRA_PHASE46_{component}_PERCENT"
        set_env(env_key, str(percent))
        log(f"Canary: {env_key}={percent}")

    # Verify routing is active
    canary = CanaryRouter()
    percents = canary.get_active_percents()
    log(f"Active percents: {percents}")


FUNCTION verify_stage(stage_number: int) -> bool:
    """Check success criteria for the current stage."""

    metrics = Phase46Metrics.global_instance().snapshot()

    IF stage_number >= 1:
        ASSERT metrics.search.hit_rate >= 0.5
        ASSERT metrics.search.latency_p95_ms < 200
        ASSERT metrics.counters.search_requests > 0

    IF stage_number >= 2:
        ASSERT metrics.got_bridge.convergence_pass_rate >= 0.7
        ASSERT metrics.got_bridge.fallback_rate < 0.3

    IF stage_number >= 3:
        ASSERT metrics.hmm.confidence_p50 >= 0.6
        ASSERT metrics.hmm.observation_entropy > 0

    RETURN True
```

## Monitoring During Ramp

```pseudocode
FUNCTION monitor_canary(interval_seconds: int = 60):
    """Continuous monitoring loop during canary ramp."""

    metrics = Phase46Metrics.global_instance()
    rollback = RollbackEngine(metrics=metrics)

    WHILE True:
        snap = metrics.snapshot()

        # Check each breach condition
        rollback.evaluate("search_error_rate",
            breached=(snap.search.hit_rate < 0.3))

        rollback.evaluate("got_fallback_rate",
            breached=(snap.got_bridge.fallback_rate > 0.5))

        rollback.evaluate("hmm_confidence",
            breached=(snap.hmm.confidence_p50 < 0.3))

        rollback.evaluate("latency_regression",
            breached=(snap.search.latency_p95_ms > 500))

        # Log status
        log(f"Canary monitor: search_hits={snap.counters.search_hits} "
            f"latency_p95={snap.search.latency_p95_ms}ms "
            f"hit_rate={snap.search.hit_rate}")

        sleep(interval_seconds)
```

## TDD Anchors

```pseudocode
TEST "stage_1_activates_search_only":
    set_env("BIZRA_PHASE46_SEARCH_PERCENT", "10")
    canary = CanaryRouter()
    # 10% of requests should route to search
    # GoT and HMM should remain at 0%
    ASSERT canary.get_active_percents() == {"search": 10, "got_bridge": 0, "hmm": 0}

TEST "stage_2_activates_search_and_got":
    set_env("BIZRA_PHASE46_SEARCH_PERCENT", "50")
    set_env("BIZRA_PHASE46_GOT_BRIDGE_PERCENT", "10")
    canary = CanaryRouter()
    ASSERT canary.get_active_percents() == {"search": 50, "got_bridge": 10, "hmm": 0}

TEST "rollback_on_breach_during_ramp":
    # Simulated by existing E2E test: test_full_lifecycle in test_phase46_full_pipeline.py
    # Already validated: 2 consecutive breaches → rollback → percent zeroed
    PASS  # Covered by existing test
```

## Risk

**Low.** Canary infrastructure is built, tested, and wired. The ramp is conservative (10% → 50% → 100%) with automatic rollback on 2 consecutive breaches. Kill switches provide immediate manual override.

## When To Activate

Stage 1 can be activated **now** — all prerequisites are met. The activation is a single env var change (`BIZRA_PHASE46_SEARCH_PERCENT=10`) with zero code changes required.
