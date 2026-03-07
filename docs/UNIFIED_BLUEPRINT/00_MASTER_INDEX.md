# BIZRA Unified Blueprint v1.0.0 — Master Index

> **Locked:** 2026-03-07 | **Status:** Canonical Source of Truth
> **Supersedes:** All 330+ spec files across docs/specs/, specs/, bizra-normalizers/specs/
> **Rule:** No new spec files. All future work references this blueprint only.

## Purpose

This is the ONLY architecture document for BIZRA. It consolidates every phase,
every spec, every audit into 12 modules. Each feature is marked:

- [x] BUILT — Production code exists, tested
- [~] PARTIAL — Code exists, incomplete vs spec
- [ ] NOT BUILT — Spec only, zero implementation

## Document Map

| # | Module | File | Domain |
|---|--------|------|--------|
| 01 | Constitutional Layer | `01_CONSTITUTIONAL_LAYER.md` | FATE gates, thresholds, kernel invariants |
| 02 | Sovereign Runtime | `02_SOVEREIGN_RUNTIME.md` | Node0 core, mission pipeline, OODA loop |
| 03 | Cognition Engine | `03_COGNITION_ENGINE.md` | Reasoning, HRM, NorthStar, RLM, entropy routing |
| 04 | Agent Orchestration | `04_AGENT_ORCHESTRATION.md` | PAT-7, SAT-5, A2A, swarm, harness |
| 05 | Economic Engine | `05_ECONOMIC_ENGINE.md` | 3 tokens, PoI, treasury, flywheel, AaaS |
| 06 | Knowledge Layer | `06_KNOWLEDGE_LAYER.md` | Memory, hypergraph, embeddings, living memory |
| 07 | Federation & Network | `07_FEDERATION_NETWORK.md` | P2P gossip, DHT, distributed scaling |
| 08 | Desktop Automation | `08_DESKTOP_AUTOMATION.md` | HDA, AHK, Telescript, Ghost Overlay |
| 09 | Frontend Surfaces | `09_FRONTEND_SURFACES.md` | 5 UI surfaces, components, routes, design tokens |
| 10 | Infrastructure | `10_INFRASTRUCTURE.md` | K8s, CI/CD, resilience gate, monitoring |
| 11 | Security & Compliance | `11_SECURITY_COMPLIANCE.md` | SAP v0, hardening, certification, audits |
| 12 | Protocol Layer | `12_PROTOCOL_LAYER.md` | UMB, wire formats, transport, IPC |

## Codebase Stats (2026-03-07)

| Metric | Value |
|--------|-------|
| Python LOC | ~113K across 55 modules in core/ |
| Rust LOC | ~137K across 22 crates in bizra-omega/ |
| Tests passing | 8,819 Python + 610 Rust = 9,429 |
| Coverage floor | 38% (ratcheting toward 95%) |
| CI gates | 9/9 GREEN |
| K3d cluster | 1 server + 2 agents, Argo Rollouts live |
| Frontend | award-winner-design (Next.js, live at bizra.ai) |

## Completion Summary

| Domain | Built | Partial | Not Built | Coverage |
|--------|-------|---------|-----------|----------|
| Constitutional | 14/14 | 0 | 0 | 100% |
| Sovereign Runtime | 18/22 | 3 | 1 | 86% |
| Cognition | 11/14 | 2 | 1 | 82% |
| Agent Orchestration | 8/13 | 3 | 2 | 69% |
| Economic Engine | 6/11 | 2 | 3 | 64% |
| Knowledge Layer | 9/9 | 0 | 0 | 100% |
| Federation & Network | 6/10 | 2 | 2 | 70% |
| Desktop Automation | 4/7 | 2 | 1 | 71% |
| Frontend | 12/25 | 8 | 5 | 64% |
| Infrastructure | 18/21 | 1 | 2 | 86% |
| Security & Compliance | 12/18 | 2 | 4 | 72% |
| Protocol Layer | 3/8 | 2 | 3 | 50% |
| **TOTAL** | **121/172** | **27** | **24** | **78%** |

## Constants (Single Source of Truth)

All thresholds from `core/integration/constants.py` v3.0.0:

```
UNIFIED_IHSAN_THRESHOLD    = 0.95   # Production gate
STRICT_IHSAN_THRESHOLD     = 0.99   # Consensus gate
RUNTIME_IHSAN_THRESHOLD    = 1.0    # Apoptosis safeguard
UNIFIED_SNR_THRESHOLD      = 0.85   # Minimum quality floor
SNR_THRESHOLD_T1           = 0.95   # T1 expert tier
SNR_THRESHOLD_T0_ELITE     = 0.98   # T0 elite tier
ADL_GINI_THRESHOLD         = 0.35   # Justice invariant
ADL_HARBERGER_TAX_RATE     = 0.05   # Annual wealth tax
ZAKAT_RATE                 = 0.025  # Computational zakat
CONFIDENCE_HIGH            = 0.95   # Evaluator gate
FAISS_SIMILARITY_FLOOR     = 0.35   # Retrieval cutoff
GOT_MAX_HYPOTHESES         = 5      # Planner branching
```

## Standing on Giants

Shannon (information theory) . Lamport (distributed state) . Besta (Graph-of-Thoughts)
Vaswani (attention) . Nakamoto (trustless receipts) . Al-Ghazali (Ihsan ethics)
General Magic (Telescript) . Friston (active inference) . Kahneman (dual process)
Hewitt (actor model) . Deming (PDCA) . Boyd (OODA) . Ostrom (commons governance)

## Spec Provenance

This blueprint was compiled from:
- `docs/specs/` — 19 dirs + 53 files (phases 0-75)
- `specs/` — 15 dirs (alpha100, SAP, user-zero, v3, harness)
- `bizra-normalizers/specs/` — 1 file (PAT collection)
- `scripts/spec/` — 2 files (SAP validation)
- `artifacts/phase56/` — 2 files (engine report)
- `docs/*.md` — Enterprise Blueprint, QA Strategy, Risk Management, etc.
- `docs/adr/` — ADR-012 (canary strategy)
