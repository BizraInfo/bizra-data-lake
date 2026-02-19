# Phase 47.1: Cognitive Resonance Safe Activation

## Standing on Giants
Shannon (information theory, 1948) · Johnson/FAISS (vector search, 2021) · Rabiner (HMM, 1989) · Besta (GoT, 2024) · Lamport (distributed reliability, 1978) · Al-Ghazali (Ihsan ethics, 1095) · Fowler (canary releases, 2010) · Nygard (Release It!, 2007)

## Problem Statement

Phase 46 (`366774c`) + Phase 46.1 (`477e9d9`) delivered 5 working modules and wired them into the sovereign MCP surface. However, all three feature flags default to OFF:

| Feature Flag | Default | Consumers |
|-------------|---------|-----------|
| `BIZRA_PHASE46_SEARCH_ENABLED` | `"0"` | `vector_search.py`, `resonance.py` |
| `BIZRA_PHASE46_GOT_BRIDGE_ENABLED` | `"0"` | `got_bridge.py`, `apex_engine.py` |
| `BIZRA_PHASE46_HMM_ENABLED` | `"0"` | `hmm_engine.py`, `proactive.py` |

The MCP tools (`sovereign_search`, `sovereign_resonance`, `sovereign_predict`) bypass these flags via direct instantiation in `Phase46Interface`, so they work regardless. But the runtime integrations (apex GoT bridge, proactive HMM) remain dormant.

**Phase 47.1 activates these capabilities in staging with production-grade safety controls.**

## Locked Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| Soak mode | Guarded Solo | Single operator, automated rollback covers gaps |
| HMM staging mode | Single-caller isolation | Prevent cross-caller state pollution |
| Rollback policy | Strict (2 consecutive breaches) | Conservative for first activation |
| New dependencies | Zero | All infrastructure exists from Phase 46 |

## Current State (Post Phase 46.1)

| Component | Status | Tests | MCP Exposed |
|-----------|--------|-------|-------------|
| VectorSearchEngine | Built, lazy FAISS load | 36 | `sovereign_search` |
| GoTBridge | Built, SNR convergence gate | 37 | via `sovereign_resonance` (pipeline) |
| HMMEngine | Built, 6-state Forward/Viterbi | 62 | `sovereign_predict` |
| CognitiveResonance | Built, search->reason->predict | 25 | `sovereign_resonance` |
| Apex GoT integration | Wired, flag OFF | 10 | N/A (runtime) |
| Proactive HMM | Wired, flag OFF | 40+ | N/A (runtime) |
| **Total Phase 46 tests** | **210 passing** | | |

## Target State

```
Phase 46 flags OFF (current)          Phase 47.1 (target)
────────────────────────              ─────────────────────
SEARCH_ENABLED=0                      SEARCH_ENABLED=1
GOT_BRIDGE_ENABLED=0                  GOT_BRIDGE_ENABLED=1
HMM_ENABLED=0                         HMM_ENABLED=1
No canary routing                     SEARCH_PERCENT=0..100
No observability                      CANARY_SALT=<random>
No rollback automation                HMM_CALLER_MODE=single
                                      Prometheus metrics + alerts
                                      Automatic rollback on breach
```

## Spec Modules

| File | Scope | ~Lines |
|------|-------|--------|
| `01_canary_routing.md` | Deterministic canary + kill switch precedence | ~200 |
| `02_hmm_caller_isolation.md` | Single-caller HMM gate + telemetry | ~150 |
| `03_observability.md` | Metrics, structured logs, alerts | ~250 |
| `04_rollback_automation.md` | Strict rollback policy + receipt persistence | ~200 |
| `05_release_isolation.md` | Branch strategy, manifest, semantic checks | ~150 |
| `06_validation_plan.md` | TDD anchors, Stage 0 drills, acceptance criteria | ~200 |

## Dependency Graph

```
05_release_isolation ─┐
                      ├──> 01_canary_routing ──> 03_observability
02_hmm_caller_isolation ─┘                            │
                                                      ▼
                                              04_rollback_automation
                                                      │
                                                      ▼
                                              06_validation_plan
```

## Files Modified (Projected)

| File | Change |
|------|--------|
| `core/integration/constants.py` | Phase 47.1 canary + rollback constants |
| `core/search/vector_search.py` | Canary routing wrapper |
| `core/reasoning/got_bridge.py` | Canary routing wrapper |
| `core/prediction/hmm_engine.py` | Caller isolation gate |
| `core/resonance.py` | Metrics emission hooks |
| `tools/mcp/sovereign_mcp_server.py` | Canary-aware Phase46Interface |
| `deploy/monitoring/alerting-rules.yaml` | Phase 46 alert rules |
| `deploy/monitoring/prometheus-config.yaml` | Phase 46 scrape targets |
| NEW `core/rollout/canary.py` | Canary routing + rollback engine |
| NEW `core/rollout/__init__.py` | Package init |
| NEW `tests/core/rollout/` | Canary + rollback tests |
| NEW `artifacts/known_failures_phase47_baseline.json` | Frozen failure baseline |
