# User Zero Bootstrap: Agent-as-Marketing

## Vision

Mumo becomes User Zero. His 7000+ conversations across 10+ AI platforms become
compilation fuel. His sovereign agent becomes BIZRA's front door. The product
markets itself by being itself.

The most honest marketing is a product so compelling that using it _is_ the
pitch. User Zero's agent -- compiled from real cross-platform conversation
history, running on real GENESIS infrastructure -- demonstrates every BIZRA
capability by doing its actual job. No demos. No mockups. The live system is
the proof.

## Phase Dependency Graph

```
  Phase 01 (Multi-Platform Ingestion)
       |
       v
  Phase 02 (GENESIS Compilation Fuel)
       |
       v
  Phase 03 (Agent-as-Marketing Frontend)
       |
       v
  Phase 04 (SAP Agentic Ads Integration)
       |
       v
  Phase 05 (Viral Growth Loop)
```

Each phase is gated: output of phase N is a hard prerequisite for phase N+1.
No phase may be skipped. No phase may begin until the prior phase's TDD
anchors pass at Ihsan >= 0.95.

## Existing Infrastructure

| File | Purpose | Reference |
|------|---------|-----------|
| `ingest_conversations.py` | ChatGPT JSON parser, FAISS indexing | Parser pattern, deterministic_id() |
| `corpus_manager.py` | Parquet pipeline, BLAKE3/SHA-256 hashing | Content hashing, CorpusManager class |
| `bizra-omega/bizra-agent/src/reflex_compiler.rs` | System-2 to System-1 compilation, 4-gate pipeline | CompileSample, snr_score(), evaluate() |
| `bizra-omega/bizra-agent/src/hash_namespace.rs` | Domain-separated BLAKE3 hashes | TriggerHash, TRIGGER_DOMAIN constants |
| `bizra-omega/bizra-agent/src/runtime.rs` | Agent runtime, memory extraction, intent classification | receive() pipeline, 8-step execution |
| `bizra-omega/bizra-agent/src/reflex_cache.rs` | Compiled rule store, quarantine-not-evict | ReflexCache, QuarantineReason enum |
| `filedfs/node0-mvp.jsx` | Phase roadmap, growth math | Sigmoid model, phase definitions |
| `core/iaas/snr_v2_adapter.py` | SNR scoring bridge | SNRv2Adapter, calculate_snr_normalized() |

## Prerequisites

- **SAP v0**: 24/24 conformance tests passing (see `specs/sap-v0/04-conformance.md`)
- **Alpha-100 Sprint 2**: Complete (installer, onboarding, Filedfs wiring)
- **Rust test suite**: 971+ tests passing (`cargo test --workspace --release`)
- **CI pipeline**: Green on lint + test + security stages

## Standing on Giants

- **Shannon** (1948) -- Information theory informs SNR scoring across the pipeline
- **Al-Ghazali** (1058-1111) -- Ihsan (excellence as worship) becomes a hard gate, not a suggestion
- **General Magic** (1990) -- Agent-as-product vision: the device _is_ the demo
- **Lamport** (1978) -- Distributed consensus underpins cross-platform validation
- **Boyd** (1976) -- OODA loop maps to reflex compilation: observe, orient, decide, act

## Cross-References

- `specs/sap-v0/` -- SAP Agentic Ads specification (Phase 04 dependency)
- `schemas/sap/v0/` -- SAP wire format schemas
- `specs/alpha100-sprint3/` -- Alpha-100 sprint planning
- `docs/internal/SAP_AGENTIC_ADS_PILOT_KPIS.md` -- Pilot KPIs for Phase 04
- `docs/internal/USER_ZERO_SHADOW_PILOT_RUNBOOK.md` -- Shadow pilot operations

## File Index

| Spec File | Phase | Status |
|-----------|-------|--------|
| `phase_01_multi_platform_ingestion.md` | 01 | SPEC |
| `phase_02_genesis_compilation_fuel.md` | 02 | SPEC |
| `phase_03_agent_as_marketing.md` | 03 | PLANNED |
| `phase_04_sap_integration.md` | 04 | PLANNED |
| `phase_05_viral_growth_loop.md` | 05 | PLANNED |
