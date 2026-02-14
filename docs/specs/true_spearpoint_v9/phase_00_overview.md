# True Spearpoint v9.0 — Specification Overview

## Status: SPEC COMPLETE
## Author: BIZRA Node0
## Date: 2026-02-13

---

## Executive Summary

True Spearpoint v9 is a **recursive Benchmark Dominance Loop** that composes
existing BIZRA evaluation infrastructure with new capabilities for parallel
consistency evaluation, multi-tier memory, intelligent routing, and
multi-battlefield campaign orchestration.

**Key principle:** Build on top of existing code, not alongside it.

---

## Architecture: Three Pillars → One Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRUE SPEARPOINT v9.0                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PILLAR 1: EVALUATION          PILLAR 2: ARCHITECTURE           │
│  ┌──────────────────┐          ┌──────────────────┐             │
│  │ HALOrchestrator   │          │ ZScorer           │             │
│  │   (wraps CLEAR)   │          │   (replaces kwds) │             │
│  │ SteeringAdapter   │          │ MIRASMemory       │             │
│  │ EnhancedABC       │          │ ZeroFailureTrainer│             │
│  └────────┬─────────┘          └────────┬─────────┘             │
│           │                              │                       │
│           └──────────┬───────────────────┘                       │
│                      │                                           │
│  PILLAR 3: SUBMISSION│                                           │
│  ┌──────────────────┐│                                           │
│  │ BattlefieldReg   ││                                           │
│  │ IntegrityValid   ││                                           │
│  │ CampaignOrch     ││                                           │
│  └────────┬─────────┘│                                           │
│           │           │                                           │
│           └─────┬─────┘                                           │
│                 ▼                                                 │
│     ┌─────────────────────┐                                      │
│     │ TrueSpearpointLoop  │ ← Phase 04: The Composer             │
│     │  (wraps DominanceLoop)│                                     │
│     │                     │                                      │
│     │ EVALUATE (HAL+CLEAR)│                                      │
│     │ ABLATE (AblationEng)│                                      │
│     │ ARCHITECT (ZScorer) │                                      │
│     │ SUBMIT (Campaign)   │                                      │
│     │ ANALYZE (Pareto)    │                                      │
│     │       ↻ REPEAT      │                                      │
│     └─────────┬───────────┘                                      │
│               │                                                   │
│     ┌─────────▼───────────┐                                      │
│     │ SpearpointSkillHndl │ ← Phase 05: Bridge Integration       │
│     │ GatewayAdapter      │                                      │
│     │ CLI Entry Point     │                                      │
│     └─────────────────────┘                                      │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## Spec Files

| File | Phase | Purpose | Est. Lines |
|------|-------|---------|-----------|
| `phase_00_overview.md` | 0 | This file | 150 |
| `phase_01_evaluation_harness.md` | 1 | HAL, Steering, ABC | 450 |
| `phase_02_architecture_engine.md` | 2 | ZScorer, MIRAS, ZeroFailure | 480 |
| `phase_03_leaderboard_submission.md` | 3 | Battlefields, Integrity, Campaign | 440 |
| `phase_04_dominance_loop_v9.md` | 4 | TrueSpearpointLoop composer | 450 |
| `phase_05_integration_wiring.md` | 5 | Gateway, Skill, CLI, Wiring | 400 |

---

## Deliverable Files (13 source + 10 test)

### Source Files

| # | File | Phase | Lines | Dependencies |
|---|------|-------|-------|-------------|
| 1 | `core/spearpoint/gateway_adapter.py` | 5 | ~60 | gateway (optional) |
| 2 | `core/benchmark/battlefield_registry.py` | 3 | ~120 | stdlib |
| 3 | `core/benchmark/z_scorer.py` | 2 | ~180 | moe_router |
| 4 | `core/benchmark/miras_memory.py` | 2 | ~200 | stdlib |
| 5 | `core/benchmark/zero_failure_trainer.py` | 2 | ~100 | stdlib |
| 6 | `core/benchmark/hal_orchestrator.py` | 1 | ~200 | clear_framework |
| 7 | `core/benchmark/steering_adapter.py` | 1 | ~100 | stdlib |
| 8 | `core/benchmark/abc_enhanced.py` | 1 | ~150 | clear_framework |
| 9 | `core/benchmark/integrity_validator.py` | 3 | ~200 | leaderboard |
| 10 | `core/benchmark/campaign_orchestrator.py` | 3 | ~200 | leaderboard, integrity |
| 11 | `core/spearpoint/true_spearpoint_loop.py` | 4 | ~350 | ALL of above |
| 12 | `core/spearpoint/spearpoint_skill.py` | 5 | ~180 | loop, registry |
| 13 | Update `core/bridges/desktop_bridge.py` | 5 | +6 | skill handler |

### Test Files

| # | File | Tests |
|---|------|-------|
| 1 | `tests/core/benchmark/test_hal_orchestrator.py` | ~150 |
| 2 | `tests/core/benchmark/test_steering_adapter.py` | ~80 |
| 3 | `tests/core/benchmark/test_abc_enhanced.py` | ~100 |
| 4 | `tests/core/benchmark/test_z_scorer.py` | ~120 |
| 5 | `tests/core/benchmark/test_miras_memory.py` | ~120 |
| 6 | `tests/core/benchmark/test_zero_failure_trainer.py` | ~60 |
| 7 | `tests/core/benchmark/test_battlefield_registry.py` | ~40 |
| 8 | `tests/core/benchmark/test_integrity_validator.py` | ~100 |
| 9 | `tests/core/benchmark/test_campaign_orchestrator.py` | ~100 |
| 10 | `tests/core/spearpoint/test_true_spearpoint_loop.py` | ~200 |

---

## Existing Code Reuse Map

| Existing Module | How Used in v9 |
|----------------|---------------|
| `CLEARFramework` | HALOrchestrator wraps it for N-run evaluation |
| `DominanceLoop` | TrueSpearpointLoop wraps it as inner engine |
| `MoERouter` | ZScorer replaces its ComplexityClassifier |
| `AblationEngine` | Used directly in ablate phase |
| `LeaderboardManager` | CampaignOrchestrator composes it |
| `AntiGamingValidator` | IntegrityValidator extends it |
| `AgenticBenchmarkChecklist` | EnhancedABC extends it |
| `GuardrailSuite` | Called by HAL and IntegrityValidator |
| `MetricsProvider` | Feeds real metrics through the loop |
| `AutoEvaluator` | Receives metrics from loop via MetricsProvider |
| `RecursiveLoop` | Inner heartbeat (orthogonal to v9 loop) |
| `PatternStrategySelector` | Used by RecursiveLoop (unchanged) |

---

## Quality Gates

- **SNR Target:** >= 0.99 (convergence condition)
- **Ihsan Threshold:** >= 0.95 (from `core/integration/constants.py`)
- **Test Coverage:** >= 80% on all new modules
- **Zero new dependencies:** stdlib + existing BIZRA imports only
- **Each source file:** < 400 lines
- **Each spec file:** < 500 lines

---

## Implementation Order (Dependency-safe)

```
Week 1: Foundation (no cross-deps)
  1. gateway_adapter.py
  2. battlefield_registry.py
  3. z_scorer.py
  4. miras_memory.py
  5. zero_failure_trainer.py

Week 2: Composition (depends on Week 1)
  6. hal_orchestrator.py
  7. steering_adapter.py
  8. abc_enhanced.py
  9. integrity_validator.py
  10. campaign_orchestrator.py

Week 3: Integration (depends on ALL)
  11. true_spearpoint_loop.py
  12. spearpoint_skill.py
  13. desktop_bridge.py update
  14. All test files
  15. CLI validation
```

---

## Verification

```bash
# Run all new tests
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE pytest tests/core/benchmark/test_hal_*.py \
    tests/core/benchmark/test_z_scorer.py \
    tests/core/benchmark/test_miras_*.py \
    tests/core/benchmark/test_zero_failure*.py \
    tests/core/benchmark/test_battlefield*.py \
    tests/core/benchmark/test_integrity*.py \
    tests/core/benchmark/test_campaign*.py \
    tests/core/benchmark/test_steering*.py \
    tests/core/benchmark/test_abc*.py \
    tests/core/spearpoint/test_true_spearpoint*.py \
    -v --tb=short

# Verify no regressions
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE pytest tests/core/benchmark/ \
    tests/core/spearpoint/ tests/core/skills/ \
    tests/core/bridges/ -q --tb=short -m "not slow"

# End-to-end: Run 3-iteration loop
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE python -m core.spearpoint.true_spearpoint \
    --max-iterations 3 --target-snr 0.90

# Skill test via Desktop Bridge
python3 -c "
import socket, json
s = socket.socket(); s.connect(('127.0.0.1', 9742))
msg = json.dumps({
    'jsonrpc': '2.0', 'method': 'invoke_skill',
    'params': {'skill': 'true_spearpoint', 'inputs': {
        'operation': 'run', 'max_iterations': 3
    }}, 'id': 1
}).encode() + b'\n'
s.sendall(msg); print(s.recv(8192).decode()); s.close()
"
```

---

*Standing on Giants: Shannon (SNR) | Boyd (OODA Loop) | Pareto (Efficiency Frontier) | Nygard (Circuit Breaker) | Anthropic (Constitutional AI)*
