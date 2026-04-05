# 00 — Tri-Partite Cognitive Architecture v0.91.0: Overview

**Status:** SPEC  
**Version:** v0.91.0  
**Predecessor:** v0.90.0 (MoE + SFE dual-mode)  
**Classification:** Internal Technical Specification

---

## 1. Architecture Summary

v0.91.0 adds a **third axis of sparsity** to the BIZRA cognitive architecture:

```
v0.90.0:  MoE (computation sparsity) + SFE (deterministic execution)
v0.91.0:  Engram (memory sparsity) + MoE (computation sparsity) + SFE (deterministic execution)
```

The Engram module sparsifies *memory* — static factual knowledge is retrieved in O(1)
via hash lookup rather than reconstructed through multi-layer transformer attention.
MoE continues to sparsify *computation* — only relevant experts activate per token.
SFE provides the deterministic execution substrate for constitutional gates.

### The Three Modes

```
┌────────────────────────────────────────────────────────────────────┐
│                     INCOMING TOKEN STREAM                          │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  CONTEXT GATE   │  alpha_t = sigmoid(h · e / √d)
                  │  (alpha_t)      │
                  └────┬───────┬────┘
                       │       │
          alpha_t > τ  │       │  alpha_t ≤ τ
          (static)     │       │  (novel)
                       ▼       ▼
              ┌──────────┐  ┌──────────┐
              │  ENGRAM   │  │   MoE    │
              │  O(1)     │  │  Routed  │
              │  Lookup   │  │  Experts │
              └─────┬─────┘  └────┬─────┘
                    │             │
                    └──────┬──────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │      SFE        │  Constitutional gates
                  │  Deterministic  │  Ihsan ≥ 0.95
                  │  Execution      │  Receipt chain
                  └─────────────────┘
```

## 2. Module Map

| Spec | Module | What | Crate |
|------|--------|------|-------|
| 01 | Engram Tiered Memory | L1 HBM → L2 DRAM → L3 NVMe hierarchy | `bizra-ttrl` (extend `engram.rs`) |
| 02 | Prefetch Pipeline | Deterministic PCIe masking for DRAM→HBM transfer | `bizra-ttrl` (new `prefetch.rs`) |
| 03 | Context Gate (alpha_t) | Sigmoid gating for Engram injection strength | `bizra-ttrl` (new `context_gate.rs`) |
| 04 | 75/25 Sparsity Law | Hard constraint on MoE vs Engram parameter allocation | `bizra-core` (new `sparsity_law.rs`) |
| 05 | Ihsan Composite | 8-dimension geometric mean with 0.95 floor per dimension | `bizra-core` (extend `lib.rs`) |
| 06 | Adaptive Routing | Route batches by alpha_t to GPU-light vs GPU-heavy paths | `bizra-agent` (extend `omni_kernel.rs`) |
| 07 | Benchmark Suite | 7 standard benchmarks + ablation methodology | `bizra-tests` (new `benchmark/`) |
| 08 | Cost Model | Per-inference cost tracking across tiers | `bizra-ttrl` (new `cost_model.rs`) |

## 3. Existing Code Anchor Points

These files already implement v0.90.0 infrastructure that v0.91.0 extends:

| File | What exists | What v0.91.0 adds |
|------|------------|-------------------|
| `bizra-ttrl/src/engram.rs` | HashMap<[u8;32], EngramEntry>, confidence gating, hit rate | Tiered storage (L1/L2/L3), prefetch integration |
| `bizra-agent/src/omni_kernel.rs` | 8-line loop, Tier-1/2/3 paths, two-phase R/W split | Context gate (alpha_t), adaptive routing |
| `bizra-ttrl/src/metabolic_ledger.rs` | PoI yield, emission decay, network bonus | Cost-per-inference tracking |
| `bizra-core/src/lib.rs` | IhsanScore (scalar), IHSAN_THRESHOLD | IhsanComposite (8 dimensions, geometric mean) |
| `bizra-hooks/src/subscribers.rs` | 12 event topics | ENGRAM_TIER_MIGRATION, PREFETCH_MISS topics |

## 4. Key Constants (from evaluation data)

```
ENGRAM_ALLOCATION_RATIO       = 0.25      # 75/25 law
ENGRAM_ALLOCATION_TOLERANCE   = 0.02      # ±2pp acceptable variance
IHSAN_COMPOSITE_FLOOR         = 0.95      # per-dimension minimum
IHSAN_COMPOSITE_DIMENSIONS    = 8         # knowledge, reasoning, code, instruction,
                                          # safety, multilingual, latency, cost
PREFETCH_L1_HIT_TARGET        = 0.81      # steady-state L1 cache hit rate
PREFETCH_PCIE_BW_TARGET       = 0.82      # steady-state PCIe utilization
EMISSION_DECAY_EMA_ALPHA      = 0.05      # cache hit rate smoothing (existing)
CONTEXT_GATE_THRESHOLD_DEFAULT = 0.50     # alpha_t boundary (entity vs reasoning)
ENGRAM_RAMP_UP_WEEKS          = 2         # throughput crossover period
```

## 5. Delta Summary: v0.90.0 → v0.91.0

| Metric | v0.90.0 | v0.91.0 | Delta | Driver |
|--------|---------|---------|-------|--------|
| NIAH Long Accuracy | 68.2% | 90.1% | +21.9pp | O(1) Engram lookup |
| Entity Retrieval Latency | 320ms | 48ms | -85.0% | Engram injection |
| Pattern Matching Latency | 260ms | 32ms | -87.7% | Hash retrieval |
| Inference Cost (140B) | $3.10 | $2.18 | -29.7% | DRAM offloading |
| Max Model Scale (HBM) | ~100B | 300B+ | 3x | Tiered storage |
| Ihsan Composite | 0.958 | 0.970 | +0.012 | All subscores ≥ 0.95 |
| Throughput (steady-state) | 850 tok/s | 1,150 tok/s | +35% | Per-token FLOP reduction |

## 6. Implementation Sequence

```
Phase 1 (Foundation):     04_sparsity_law → 05_ihsan_composite → 01_engram_tiered
Phase 2 (Infrastructure): 02_prefetch_pipeline → 03_context_gate
Phase 3 (Integration):    06_adaptive_routing → 08_cost_model
Phase 4 (Validation):     07_benchmark_suite
```

Each phase depends on the prior. Within a phase, modules can be developed in parallel.

## 7. Invariants (must hold across all specs)

1. **Ihsan floor**: No individual Ihsan dimension may drop below 0.95
2. **75/25 law**: Engram allocation = 25% ± 2pp of total sparse parameters
3. **Fail-closed**: All gates default-deny on error
4. **Receipt chain**: Every cycle produces a BLAKE3-chained receipt
5. **Offline-first**: Node operates without network; federation amplifies but doesn't create liveness
6. **No hardcoded thresholds**: All values from config or `constants.py` / `bizra-core/src/lib.rs`
