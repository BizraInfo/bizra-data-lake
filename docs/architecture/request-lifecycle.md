# BIZRA Request Lifecycle & Critical Path

> **Version**: 1.0.0  
> **Status**: Canonical  
> **Last Updated**: 2025-12-21  
> **Scope**: This document covers the Rust backend request lifecycle. For Python orchestration components, see [Python Constellation Components](#python-constellation-components) below.

## Overview

This document describes the complete request lifecycle in the BIZRA Dual-Agentic System, mapping the critical path from request ingestion through response emission.

## Architecture Layers

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 0: INGESTION                                │
│                                                                             │
│   Request → HTTP Handler (src/http.rs) → BridgeCoordinator (src/bridge.rs) │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LAYER 1: SAT VALIDATION                               │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  SATOrchestrator (src/sat.rs)                                       │   │
│   │                                                                     │   │
│   │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐             │   │
│   │  │ Security      │ │ Ethics        │ │ Performance   │             │   │
│   │  │ Guardian      │ │ Validator     │ │ Monitor       │             │   │
│   │  └───────────────┘ └───────────────┘ └───────────────┘             │   │
│   │  ┌───────────────┐ ┌───────────────┐                               │   │
│   │  │ Consistency   │ │ Resource      │                               │   │
│   │  │ Checker       │ │ Optimizer     │                               │   │
│   │  └───────────────┘ └───────────────┘                               │   │
│   │                                                                     │   │
│   │  Consensus: veto-only (all 5 approve)                                  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
            │                                           │
            │ APPROVED                                  │ REJECTED
            ▼                                           ▼
┌───────────────────────────────────┐   ┌─────────────────────────────────────┐
│    LAYER 2: PAT EXECUTION         │   │      FATE ESCALATION                │
│                                   │   │                                     │
│  PATOrchestrator (src/pat.rs)     │   │  FATECoordinator (src/fate.rs)      │
│                                   │   │                                     │
│  ┌─────────┐ ┌─────────┐          │   │  Level: Low → Medium → High → Crit  │
│  │Strategic│ │Creative │          │   │                                     │
│  │ Agent   │ │ Agent   │          │   │  Actions:                           │
│  └─────────┘ └─────────┘          │   │  - Log rejection receipt            │
│  ┌─────────┐ ┌─────────┐          │   │  - Escalate to human review         │
│  │Analyst  │ │ Impl.   │          │   │  - Block execution                  │
│  │ Agent   │ │ Agent   │          │   │  - Emit structured rejection        │
│  └─────────┘ └─────────┘          │   │                                     │
│  ┌─────────┐ ┌─────────┐          │   │  Persistence: Redis (Synapse)       │
│  │Quality  │ │ User    │          │   └─────────────────────────────────────┘
│  │Guardian │ │Advocate │          │
│  └─────────┘ └─────────┘          │
│  ┌─────────┐                      │
│  │ Coord.  │                      │
│  │ Agent   │                      │
│  └─────────┘                      │
│                                   │
│  7 agents execute in parallel     │
└───────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LAYER 3: SAT EVALUATION                               │
│                                                                             │
│   SATOrchestrator.evaluate_results()                                        │
│   - Reviews PAT outputs for safety/quality                                  │
│   - Scores each contribution                                                │
│   - Can trigger secondary FATE escalation                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LAYER 4: IHSĀN GATE                                   │
│                                                                             │
│   IhsanConstitution (src/ihsan.rs)                                          │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  8 Dimensions (weights from constitution/ihsan_v1.yaml)             │   │
│   │                                                                     │   │
│   │  correctness: 0.22  │  safety: 0.22       │  user_benefit: 0.14    │   │
│   │  efficiency:  0.12  │  auditability: 0.12 │  anti_central: 0.08    │   │
│   │  robustness:  0.06  │  adl_fairness: 0.04                          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   Thresholds by environment:                                                │
│   - development: 0.80                                                       │
│   - ci:          0.90                                                       │
│   - production:  0.95                                                       │
│                                                                             │
│   If score < threshold AND enforcement enabled → FATE escalation            │
└─────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LAYER 5: RESPONSE EMISSION                            │
│                                                                             │
│   ReceiptEmitter (src/receipts.rs)                                          │
│   - Emit execution receipt (success path)                                   │
│   - Include SHA-256 integrity hash                                          │
│   - Store in Redis + local evidence directory                               │
│                                                                             │
│   DualAgenticResponse {                                                     │
│     pat_contributions,                                                      │
│     sat_contributions,                                                      │
│     synergy_score,                                                          │
│     ihsan_score,                                                            │
│     latency,                                                                │
│     meta                                                                    │
│   }                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Critical Path Sequence

```text
Request
   │
   ▼
┌──────────────────┐
│ bridge.execute() │ ← Entry point
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ sat.validate()   │ ← SAT consensus (veto-only)
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
 PASS      REJECT
    │         │
    │         ▼
    │    ┌──────────────────┐
    │    │ fate.escalate()  │
    │    └────────┬─────────┘
    │             │
    │             ▼
    │    ┌──────────────────┐
    │    │ receipts.emit_   │
    │    │ rejection()      │
    │    └────────┬─────────┘
    │             │
    │             ▼
    │         ERROR (blocked)
    │
    ▼
┌──────────────────┐
│ pat.execute_     │ ← 7 agents parallel
│ parallel()       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ sat.evaluate_    │ ← Post-execution review
│ results()        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ calculate_ihsan()│ ← Weighted composite score
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
 PASS      FAIL (score < threshold)
    │         │
    │         ▼
    │    ┌──────────────────┐
    │    │ fate.escalate_   │
    │    │ ihsan_failure()  │
    │    └────────┬─────────┘
    │             │
    │             ▼
    │         ERROR (quality gate)
    │
    ▼
┌──────────────────┐
│ receipts.emit_   │
│ execution()      │
└────────┬─────────┘
         │
         ▼
   DualAgenticResponse
```

## SAPE Integration Points

SAPE (Symbolic-Abstraction Probe Elevation) operates at multiple layers:

| Layer | SAPE Function | Module |
|-------|---------------|--------|
| 1 (SAT Validation) | 9-probe content analysis | `src/sape.rs` |
| 2 (PAT Execution) | Pattern caching for fast-path | `src/sape.rs` |
| 4 (Ihsān Gate) | Dimension scoring | `src/sape.rs::calculate_ihsan_score()` |

### SAPE Probe Dimensions → Ihsān Mapping

```text
SAPE Probes (9)              Ihsān Dimensions (8)
─────────────────            ──────────────────
ThreatScan ─────────────┐
                        ├──→ safety (0.22)
Safety ─────────────────┘

ComplianceCheck ────────────→ auditability (0.12)

BiasProbe ──────────────────→ adl_fairness (0.04)

UserBenefit ────────────────→ user_benefit (0.14)

Correctness ────────────────→ correctness (0.22)

Groundedness ───────────┐
Relevance ──────────────┼──→ efficiency (0.12) + robustness (0.06)
Fluency ────────────────┘    + anti_centralization (0.08)
```

## Bottlenecks & Optimization Opportunities

| Bottleneck | Location | Current State | Mitigation |
|------------|----------|---------------|------------|
| Single coordinator | `bridge.rs` | All requests serialize here | Add sharding by request hash |
| SAT consensus | `sat.rs` | 5 sequential validators | Already parallelized via tokio |
| FATE persistence | `fate.rs` | In-memory default | Redis persistence enabled (from_env) |
| SAPE pattern cache | `sape.rs` | MAX_PATTERNS=100 | Consider Redis-backed cache |

## Performance Bounds

From `model-family-genesis-v1-SEALED.yaml`:

| Metric | Target | Floor | Notes |
|--------|--------|-------|-------|
| SNR | 7.8 | 7.0 | Safe mode triggers at floor |
| P95 Latency | 1500ms | - | End-to-end request latency |
| SAT Consensus | veto-only | - | all 5 validators must approve |

## Python Constellation Components

The Python `constellation/` package provides higher-level orchestration capabilities that complement the Rust backend:

| Module | Purpose | Lifecycle Participation |
|--------|---------|------------------------|
| [constellation/__init__.py](../../constellation/__init__.py) | Package initialization | Import-time setup |
| [constellation/orchestrator.py](../../constellation/orchestrator.py) | Multi-agent orchestration | Layer 2 (PAT coordination) |
| [constellation/runtime.py](../../constellation/runtime.py) | Execution runtime | All layers |
| [constellation/audit/](../../constellation/audit/) | Audit logging and evidence | Layer 5 (Receipt emission) |
| [constellation/memory/](../../constellation/memory/) | Agent memory management | Layer 2-3 (PAT/SAT state) |
| [constellation/protocols/](../../constellation/protocols/) | Inter-agent communication | Layer 1-2 (SAT/PAT messaging) |
| [constellation/hooks/](../../constellation/hooks/) | Lifecycle event hooks | All layers |
| [constellation/triggers/](../../constellation/triggers/) | Event-driven activation | Layer 0 (Ingestion) |

The Rust backend (`src/`) handles the core request lifecycle with performance-critical paths, while Python (`constellation/`) provides flexible orchestration and extensibility.

## Cross-Reference

- [constitution/ihsan_v1.yaml](../../constitution/ihsan_v1.yaml) — Canonical dimension weights
- [src/bridge.rs](../../src/bridge.rs) — PAT-SAT coordination entry point
- [src/sape.rs](../../src/sape.rs) — SAPE probe engine
- [src/fate.rs](../../src/fate.rs) — Escalation handling
- [.github/copilot-instructions.md](../../.github/copilot-instructions.md) — Agent onboarding contract
