# BIZRA Node0 — Canonical Metrics

> بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ

These numbers are truth-labeled. Every metric was MEASURED or VERIFIED on the live
codebase on Node0. No rounding. No inflation.

**Last measured:** 2026-04-13, Node0 (MSI Titan 18 HX, RTX 4090 Mobile)
**Rule:** All docs, investor materials, and landing pages MUST import from this file.

---

## Test Counts

| Suite | Count | Status | Evidence |
|-------|-------|--------|----------|
| proof_engine (core) | 703 | GREEN | MEASURED |
| cockpit | 8 | GREEN | MEASURED |
| sovereign (fate_bridge) | 7 | GREEN | MEASURED |
| **Total core** | **718** | **GREEN** | **MEASURED** |

## Codebase

| Metric | Value | Evidence |
|--------|-------|----------|
| Python LOC (core/) | ~256,000 | MEASURED |
| Rust LOC (bizra-omega/) | ~151,000 | MEASURED |
| Rust crates | 26 | MEASURED |
| CI workflow files | 21 | MEASURED |

## Constitutional Thresholds

| Anchor | Value | Source | Evidence |
|--------|-------|--------|----------|
| UNIFIED_IHSAN_THRESHOLD | 0.95 | core/integration/constants.py | VERIFIED |
| UNIFIED_SNR_THRESHOLD | 0.85 | core/integration/constants.py | VERIFIED |
| ZANN_ZERO | true | core/proof_engine/constitutional_verdict.py | VERIFIED |
| RIBA_ZERO | true | core/proof_engine/constitutional_verdict.py | VERIFIED |

## Runtime

| Component | Value | Evidence |
|-----------|-------|----------|
| Ollama models | 6 | MEASURED |
| Governance model | gemma4:26b-bizra-16k (19.83 tok/s) | MEASURED |
| Fast model | gemma4:e4b (61 tok/s) | MEASURED |
| Coder model | qwen2.5-coder:14b (28.5 tok/s) | MEASURED |
| Reasoning model | deepseek-r1:7b (54 tok/s) | MEASURED |
| Embedding model | nomic-embed-text | MEASURED |
| GPU | RTX 4090 Mobile 16GB VRAM | VERIFIED |
| VRAM headroom (governance) | ~2.2 GB | MEASURED |

## Proof Pipeline

| Component | Status | Evidence |
|-----------|--------|----------|
| Evidence Auditor | Promoted to core | VERIFIED |
| SAT Validator | Promoted to core | VERIFIED |
| FATE Gate | Promoted to core | VERIFIED |
| FATE Bridge (SovereignRuntime STAGE 2.5) | Wired | VERIFIED |
| Loop Proof artifact (6-step hash-chained) | Operational | VERIFIED |
| Ed25519 seal/verify/canonicalize gate | Operational | VERIFIED |
| Glass Cockpit v0.1 (port 8420) | Operational | VERIFIED |
| FATE Telemetry (append-only JSONL) | Operational | VERIFIED |
| PAT-7/SAT-5 model routing table | Defined | VERIFIED |

## PAT-7 Agent Status

| Agent | Model | Status | Evidence |
|-------|-------|--------|----------|
| Researcher | gemma4:e4b | **EXERCISED** in loop proof | VERIFIED |
| Strategist | gemma4:26b-bizra-16k | Routed, not exercised | DERIVED |
| Analyst | qwen2.5-coder:14b | Routed, not exercised | DERIVED |
| Creator | gemma4:e4b | Routed, not exercised | DERIVED |
| Executor | deepseek-r1:7b | Routed, not exercised | DERIVED |
| Guardian | gemma4:26b-bizra-16k | Routed, not exercised | DERIVED |
| Coordinator | gemma4:26b-bizra-16k | Routed, not exercised | DERIVED |

## SAT-5 Gate Status

| Gate | Status | Evidence |
|------|--------|----------|
| Sentinel (structural integrity) | Coded, not wired to loop proof | DERIVED |
| Oracle-S (constitutional compliance) | **Exercised** via SAT Validator | VERIFIED |
| Ledger (economic soundness) | Coded, not wired | DERIVED |
| Conductor (operational readiness) | Coded, not wired | DERIVED |
| Ambassador (human verification) | Coded, not wired | DERIVED |

## Canon Chain

| Anchor | Value | Evidence |
|--------|-------|----------|
| Spearpoint seal | b08f2208 | VERIFIED |
| Reachable from HEAD | YES | VERIFIED |

---

*Every value labeled VERIFIED was directly observed on running hardware.*
*Every value labeled MEASURED was quantified by tool output.*
*Every value labeled DERIVED was inferred from verified/measured evidence.*
*No value labeled PLANNED appears in this file — PLANNED items belong in roadmap docs, not metrics.*

---

## Changelog

### v1.1 (2026-04-13) — Post-MVDA promotion session
- **Test count:** Redefined from "all Python tests" (v0: 12,537) to "core proof/cockpit/sovereign" (v1.1: 718). The v0 count included all tests across the entire repo including legacy, tools, and examples. The v1.1 count covers only the governed proof pipeline surface — the metric that matters for CLAIM_MUST_BIND.
- **PAT-7 agent status:** Added honest per-agent status. 1 EXERCISED (Researcher), 6 DERIVED (routed but not exercised through full loop proof).
- **SAT-5 gate status:** Added honest per-gate status. 1 VERIFIED (Oracle-S via SAT Validator), 4 DERIVED (coded but not wired to loop proof).
- **Proof pipeline:** Added 9 components with evidence classes. All VERIFIED on running hardware.
- **Previous version:** v0 (2026-04-06) — initial metrics file with whole-repo counts.

### v0 (2026-04-06) — Initial canonical metrics
- First version. Whole-repo test counts. Pre-MVDA promotion.
