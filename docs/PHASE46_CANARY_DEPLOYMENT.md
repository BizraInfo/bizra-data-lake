# Phase 46 Canary Deployment Guide

> Standing on Giants: Fowler (canary releases) · Shannon (SNR/entropy metrics) · Lamport (distributed reliability) · Deming (PDCA quality gates)

## Overview

Phase 46 introduces **Cognitive Resonance** — FAISS vector search (102K vectors, 384-dim), HMM cognitive state prediction, and Graph-of-Thoughts bridge reasoning — into the BIZRA sovereign stack. These features are deployed through a **deterministic canary routing system** that gates traffic at the application level, independent of infrastructure-level traffic splitting.

### Architecture

```
                    ┌─────────────────────────────────────┐
                    │          CanaryRouter                │
                    │  Gate 0: Kill switch (ENABLED)       │
                    │  Gate 1: Percent bounds (PERCENT)    │
                    │  Gate 2: Deterministic MD5 hash      │
                    └────────┬───────┬───────┬────────────┘
                             │       │       │
                    ┌────────▼──┐ ┌──▼────┐ ┌▼──────────┐
                    │  Search   │ │  GoT  │ │   HMM     │
                    │ (FAISS)   │ │Bridge │ │ Predict   │
                    └────────┬──┘ └──┬────┘ └┬──────────┘
                             │       │       │
                    ┌────────▼───────▼───────▼────────────┐
                    │        Phase46Metrics                │
                    │  Counters · Latency · SNR · Entropy  │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │        RollbackEngine                │
                    │  2 consecutive breaches → rollback   │
                    │  Zeroes PERCENT → CanaryRouter stops │
                    └─────────────────────────────────────┘
```

## Components

### CanaryRouter (`core/rollout/canary.py`)

Deterministic routing using stable MD5 hashing. Three gates checked in order:

| Gate | Check | Behavior |
|------|-------|----------|
| 0 — Kill switch | `BIZRA_PHASE46_{component}_ENABLED` | `"0"` → OFF (overrides all). `"1"` → ON (bypasses percent). Unset → defer to Gate 1. |
| 1 — Percent bounds | `BIZRA_PHASE46_{component}_PERCENT` | `0` → OFF. `100` → ON. Otherwise → Gate 2. |
| 2 — Hash bucket | `MD5(salt:component:request_key) % 100` | `bucket < percent` → route to Phase 46 path. |

**Critical**: `ENABLED=1` forces 100% routing, bypassing percent. To use percent routing, leave `ENABLED` **unset**.

### Phase46Metrics (`core/rollout/metrics.py`)

In-process metrics collector. Records counters, latency histograms, SNR values, HMM confidence scores, and Shannon entropy of observation symbols.

### RollbackEngine (`core/rollout/rollback.py`)

Monitors metric breaches. Two consecutive breaches on any metric trigger automatic rollback:

1. Zeroes `BIZRA_PHASE46_{component}_PERCENT` env var
2. Rollback order: HMM → GoT → Search → hard kill (all ENABLED=0)
3. Persists immutable JSON receipt to `rollback_receipts/`

### HMMCallerGate (`core/rollout/hmm_gate.py`)

Isolates HMM write access during staging. Modes: `single` (one caller), `multi` (all callers), `disabled` (no writes).

## Environment Variables

### Canary Routing

| Variable | Default | Description |
|----------|---------|-------------|
| `BIZRA_PHASE46_SEARCH_ENABLED` | *(unset)* | Kill switch for search. Leave unset for percent routing. |
| `BIZRA_PHASE46_GOT_BRIDGE_ENABLED` | `"0"` | Kill switch for GoT bridge. |
| `BIZRA_PHASE46_HMM_ENABLED` | `"0"` | Kill switch for HMM prediction. |
| `BIZRA_PHASE46_SEARCH_PERCENT` | `"0"` | Percent of requests routed to FAISS search (0-100). |
| `BIZRA_PHASE46_GOT_BRIDGE_PERCENT` | `"0"` | Percent routed to GoT bridge (0-100). |
| `BIZRA_PHASE46_HMM_PERCENT` | `"0"` | Percent routed to HMM (0-100). |
| `BIZRA_PHASE46_CANARY_SALT` | `"bizra-phase46-canary-v1"` | Stable hash salt for deterministic routing. |
| `BIZRA_PHASE46_HMM_CALLER_MODE` | `"single"` | HMM gate mode: `single`, `multi`, `disabled`. |
| `BIZRA_PHASE46_HMM_ALLOWED_CALLER` | `"mcp"` | Caller identity allowed to write HMM observations in `single` mode. |

### Runtime

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_HTTP_PORT` | `"8081"` | HTTP health/metrics port for sovereign MCP pod. |
| `IHSAN_THRESHOLD` | `"0.95"` | Constitutional quality threshold. |
| `SNR_THRESHOLD` | `"0.85"` | Signal-to-noise ratio minimum. |

## Production Call Sites

All Phase 46 traffic flows through CanaryRouter:

| # | File | Line | Component | Gate |
|---|------|------|-----------|------|
| 1 | `tools/mcp/sovereign_mcp_server.py` | 398 | search | `should_route("search", query)` |
| 2 | `tools/mcp/sovereign_mcp_server.py` | 444 | resonance | `should_route("search", query)` |
| 3 | `tools/mcp/sovereign_mcp_server.py` | 496 | predict | `should_route("hmm", action)` + HMMCallerGate |
| 4 | `core/living_memory/proactive.py` | 165 | HMM observe | `should_route("hmm", symbol)` |
| 5 | `core/sovereign/apex_engine.py` | 1383 | GoT bridge | `should_route("got_bridge", query)` |

## Staged Activation

### Stage 0 — Drills (Pre-deployment)

Run rollback drills to validate infrastructure:

```bash
pytest tests/core/rollout/test_stage0_drills.py -v
```

Validates: kill switch precedence, breach windows, receipt persistence, HMM gate blocking, metrics entropy, combined pass gate.

### Stage 1 — Search at 10%

```bash
# .env / ConfigMap values
# BIZRA_PHASE46_SEARCH_ENABLED=        # UNSET (critical — do not set to "1")
BIZRA_PHASE46_SEARCH_PERCENT=10
BIZRA_PHASE46_GOT_BRIDGE_ENABLED=0
BIZRA_PHASE46_HMM_ENABLED=0
BIZRA_PHASE46_GOT_BRIDGE_PERCENT=0
BIZRA_PHASE46_HMM_PERCENT=0
```

**Monitoring targets** (first 24 hours):
- `bizra_phase46_search_hit_rate` > 0 (search returning results)
- `bizra_phase46_search_latency_p95_ms` < 500 (acceptable latency)
- `bizra_phase46_search_errors_total` / `bizra_phase46_search_requests_total` < 0.20 (error rate)

**Ramp schedule**:
- Day 1-3: 10%
- Day 4-7: 25% (`SEARCH_PERCENT=25`)
- Day 8-14: 50%
- Day 15+: 100% (set `SEARCH_ENABLED=1` to lock)

### Stage 2 — GoT Bridge at 5%

After search is stable at 100%:

```bash
BIZRA_PHASE46_SEARCH_ENABLED=1
BIZRA_PHASE46_GOT_BRIDGE_PERCENT=5
# BIZRA_PHASE46_GOT_BRIDGE_ENABLED=   # UNSET for percent routing
BIZRA_PHASE46_HMM_ENABLED=0
```

**Monitoring targets**:
- `bizra_phase46_got_fallback_total` / `bizra_phase46_got_requests_total` < 0.10
- GoT bridge convergence rate via `mcp_health` tool

### Stage 3 — HMM at 5%

After GoT bridge is stable:

```bash
BIZRA_PHASE46_HMM_PERCENT=5
# BIZRA_PHASE46_HMM_ENABLED=          # UNSET for percent routing
BIZRA_PHASE46_HMM_CALLER_MODE=single  # Only MCP can write observations
```

**Monitoring targets**:
- `bizra_phase46_hmm_confidence_p50` > 0.3
- `bizra_phase46_hmm_entropy` increasing (symbols are diverse)

## Kubernetes Deployment

### ConfigMaps

Phase 46 env vars are injected via:

| ConfigMap | Deployment | File |
|-----------|------------|------|
| `bizra-mcp-config` | All MCP pods (gateway, sovereign, ecosystem) | `deploy/k8s/base/services-mcp.yaml` |
| `bizra-canary-config` | Elite canary pods only | `deploy/k8s/canary/canary-deployment.yaml` |

Stable Elite pods do **not** reference `bizra-canary-config`, so Phase 46 features remain disabled on stable.

### Sovereign MCP Pod

The sovereign MCP pod runs two servers concurrently:

1. **Stdio MCP transport** — handles MCP tool calls from Claude Code
2. **HTTP health/metrics server** — on `MCP_HTTP_PORT` (default 8081)

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | K8s liveness/readiness/startup probes |
| `GET /metrics` | Prometheus scrape (Phase 46 + MCP-level metrics) |

### Gateway Pod

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness probe |
| `GET /health/ready` | Readiness probe (checks Redis) |
| `GET /metrics` | Prometheus scrape |

### Apply Stage 1

```bash
# Update ConfigMap
kubectl apply -f deploy/k8s/base/services-mcp.yaml -n bizra

# Restart MCP pods to pick up new env vars
kubectl rollout restart deployment/bizra-mcp-gateway -n bizra
kubectl rollout restart deployment/bizra-mcp-sovereign -n bizra

# Verify pods are healthy
kubectl get pods -n bizra -l app=bizra-mcp
```

## Monitoring and Alerting

### Prometheus Metrics

Both sovereign and gateway pods emit `bizra_phase46_*` metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `bizra_phase46_search_requests_total` | counter | Total search requests |
| `bizra_phase46_search_hits_total` | counter | Searches returning results |
| `bizra_phase46_search_errors_total` | counter | Search failures |
| `bizra_phase46_resonance_requests_total` | counter | Resonance pipeline calls |
| `bizra_phase46_hmm_requests_total` | counter | HMM prediction requests |
| `bizra_phase46_got_requests_total` | counter | GoT bridge activations |
| `bizra_phase46_got_fallback_total` | counter | GoT bridge fallbacks |
| `bizra_phase46_search_latency_p95_ms` | gauge | Search p95 latency |
| `bizra_phase46_search_hit_rate` | gauge | Fraction of searches with results |
| `bizra_phase46_resonance_snr_p50` | gauge | Resonance SNR median |
| `bizra_phase46_hmm_confidence_p50` | gauge | HMM confidence median |
| `bizra_phase46_hmm_entropy` | gauge | Shannon entropy of observation symbols |

### Alert Rules

Defined in `deploy/monitoring/alerting-rules.yaml`. Key alerts:

| Alert | Condition | Severity |
|-------|-----------|----------|
| `Phase46SearchErrorRate` | error rate > 20% for 5m | warning |
| `Phase46GoTFallbackRate` | fallback rate > 10% for 5m | warning |
| `Phase46HMMConfidenceLow` | confidence p50 < 0.3 for 10m | warning |
| `Phase46SearchLatencyHigh` | p95 latency > 500ms for 5m | warning |
| `Phase46ResonanceSNRDrift` | SNR p50 < 0.80 for 10m | critical |

## Rollback Procedures

### Automatic Rollback

RollbackEngine triggers automatically on 2 consecutive metric breaches:

1. Zeroes component's `PERCENT` env var
2. Subsequent CanaryRouter calls read fresh env → return `False`
3. Phase 46 tool methods return "temporarily disabled" immediately
4. Persists JSON receipt to `rollback_receipts/`

### Manual Rollback

```bash
# Emergency: disable all Phase 46 features immediately
kubectl set env deployment/bizra-mcp-sovereign -n bizra \
  BIZRA_PHASE46_SEARCH_ENABLED=0 \
  BIZRA_PHASE46_GOT_BRIDGE_ENABLED=0 \
  BIZRA_PHASE46_HMM_ENABLED=0

# Or: zero percents only (keeps kill switches unchanged)
kubectl set env deployment/bizra-mcp-sovereign -n bizra \
  BIZRA_PHASE46_SEARCH_PERCENT=0 \
  BIZRA_PHASE46_GOT_BRIDGE_PERCENT=0 \
  BIZRA_PHASE46_HMM_PERCENT=0
```

### K8s Canary Rollback

```bash
# Full canary deployment rollback
./deploy/k8s/canary/rollback.sh --namespace bizra
```

## Test Coverage

| Suite | Tests | File |
|-------|-------|------|
| CanaryRouter | 17 | `tests/core/rollout/test_canary_router.py` |
| HMMCallerGate | 13 | `tests/core/rollout/test_hmm_caller_gate.py` |
| Phase46Metrics | 21 | `tests/core/rollout/test_phase46_metrics.py` |
| RollbackEngine | 15 | `tests/core/rollout/test_rollback_engine.py` |
| Stage 0 Drills | 10 | `tests/core/rollout/test_stage0_drills.py` |
| MCP Phase46Interface | 35 | `tests/core/mcp/test_sovereign_phase46_tools.py` |
| Integration Pipeline | 31 | `tests/integration/test_phase46_full_pipeline.py` |
| Apex GoT Bridge | 10 | `tests/core/sovereign/test_apex_got_bridge_integration.py` |
| **Total** | **152+** | |

Run full regression:

```bash
pytest tests/core/rollout/ tests/core/mcp/ \
  tests/integration/test_phase46_full_pipeline.py \
  tests/core/sovereign/test_apex_got_bridge_integration.py -v
```

## Commit History

| Commit | Description |
|--------|-------------|
| `366774c` | Phase 46: Cognitive Resonance — FAISS + GoT bridge + staged HMM |
| `477e9d9` | Phase 46.1: Wire cognitive resonance into Sovereign MCP server |
| `44bf0fa` | Phase 47.1: Safe activation infrastructure — canary, rollback, observability |
| `8c9c587` | Phase 49.1: Close test coverage gaps + CI hardening + CanaryRouter fix |
| `17cabd4` | Phase 49.2+49.4: Delete native/ duplicate + CI hardening |
| `f4071a7` | Phase 49.3: Wire RollbackEngine into production |
| `358d66f` | Phase 49.5: Canary Stage 1 activation — SEARCH_PERCENT=10 |
| `9f89d8d` | Phase 49.6: P0/P1 canary enforcement — rollback stops traffic |
| `91c3a7e` | Add GoT bridge counters to gateway Prometheus exposition |
