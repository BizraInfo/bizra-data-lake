# Phase 42: SNR Unification & GoT Activation

## Standing on Giants
Shannon (information theory, 1948) · Wiener (signal processing, 1949) · Besta (Graph-of-Thoughts, 2024) · Lamport (distributed reliability, 1978) · Al-Ghazali (Ihsan ethics, 1095)

## Problem Statement

The BIZRA audit identified two structural disconnections:

1. **Five SNR engines operate independently at runtime.** Each has a different return type, scale, and measurement domain. The production-grade engine (`snr_v2`) is rarely called with real inputs. An `SNRProtocol` + `SNRFacade` pattern exists in `core/snr_protocol.py` but has incomplete adoption and a normalization bug.

2. **The GoT reasoning pipeline is wired but not activated in the mission path.** `graph_reasoning.py` has full LLM integration via `_llm_generate()` with proper fallback to templates. `SovereignRuntime` wires `InferenceGateway` into `GraphOfThoughts` post-hoc. But `node0_activate.py` uses direct `httpx` calls to LM Studio, bypassing GoT entirely.

## Current State (Post Phase 41 Fixes 1-4)

| Component | Status | Gap |
|-----------|--------|-----|
| `snr_v2` (core/iaas/) | Production-grade, Renyi-wired | Not in SNRFacade; most call sites pass `iaas_score=0.8` constant |
| `snr_maximizer` (core/sovereign/) | Bounded scoring, 7 noise dims | SNRFacade uses `snr_linear` (unbounded) instead of `snr_normalized` |
| `snr_apex` (core/apex/) | Self-consistent | Closed loop — scores its own output |
| `arte_engine` (root) | Production-grade | Requires FAISS index; wired into SNRFacade as embedding engine |
| `sacred_wisdom` (tools/) | Domain-specific | Correctly isolated; not generalizable |
| `SNRFacade` (core/snr_protocol.py) | Exists | Bug on line 200; missing snr_v2 as primary; missing in mission path |
| GoT → LLM | TRUE SPEARPOINT bridge exists | Mission path bypasses GoT entirely |
| `node0_activate.py` | Direct httpx to LM Studio | No GoT, no SNRFacade, no ensemble scoring |

## Target State

```
Mission Request
    │
    ▼
┌───────────────────────────────────────────┐
│           node0_activate.py               │
│  ┌──────────────────────────────────────┐ │
│  │ GraphOfThoughts (with InferenceGW)   │ │
│  │  _llm_generate() → LM Studio/Ollama │ │
│  │  hypothesis → evidence → synthesis   │ │
│  └──────────────────┬───────────────────┘ │
│                     ▼                     │
│  ┌──────────────────────────────────────┐ │
│  │ SNRFacade.calculate()                │ │
│  │  ┌─ snr_v2 (embeddings) ───┐        │ │
│  │  │  Shannon + Renyi-2      │        │ │
│  │  │  Real CUDA embeddings   │        │ │
│  │  ├─ snr_maximizer (text) ──┤        │ │
│  │  │  7 noise dimensions     │        │ │
│  │  │  Bounded snr_normalized │        │ │
│  │  └─ ensemble (geo mean) ───┘        │ │
│  └──────────────────┬───────────────────┘ │
│                     ▼                     │
│  ┌──────────────────────────────────────┐ │
│  │ Constitutional Gate                  │ │
│  │  SNRResult.ihsan_achieved?           │ │
│  │  → APPROVED / AMBER / REJECTED      │ │
│  └──────────────────┬───────────────────┘ │
│                     ▼                     │
│  ┌──────────────────────────────────────┐ │
│  │ Evidence Ledger                      │ │
│  │  BLAKE3 hash chain + receipt         │ │
│  └──────────────────────────────────────┘ │
└───────────────────────────────────────────┘
```

## Phase Files

| File | Title | Scope |
|------|-------|-------|
| `01_snr_facade_fix.md` | Fix SNRFacade normalization bug | Bug fix + snr_v2 adapter |
| `02_snr_v2_protocol_adapter.md` | Wire snr_v2 into SNRFacade | Protocol conformance adapter |
| `03_mission_snr_facade.md` | Replace mission SNR with SNRFacade | node0_activate.py integration |
| `04_got_mission_wiring.md` | Activate GoT reasoning in missions | GoT + InferenceGateway in mission path |
| `05_validation_plan.md` | End-to-end validation | Test plan and acceptance criteria |

## Success Criteria

1. All 5 engines accessible through `SNRFacade.calculate()` — no direct engine calls in mission path
2. Mission receipts contain ensemble SNR (embedding + text) when embeddings available
3. GoT reasoning active for mission execution when InferenceGateway available
4. 15/15 smoke tests pass, 91/91 snr_maximizer tests pass, 42/42 iaas tests pass
5. Live mission produces GoT thought chain in receipt metadata

## Non-Goals

- Replacing arte_engine or sacred_wisdom (they have correct domain specialization)
- Removing snr_apex (it serves the autonomous reasoning loop)
- Changing the `SNRProtocol` interface (it is correct as designed)
- Adding new SNR dimensions (Phase 41 Fix 4 already completed this)
