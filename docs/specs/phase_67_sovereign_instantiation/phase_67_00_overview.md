# Phase 67 — Sovereign Instantiation: Overview
# ═══════════════════════════════════════════════

## Goal

Integrate the complete BIZRA Sovereign Operating System — 15 native algorithms,
Declaration genesis block, Sovereignty CLI, AKIS knowledge pipeline, and 10 chaos
validators — into the production `core/` and `bizra-omega/` codebases. This phase
transforms standalone prototypes from `last update/` into production-grade,
constitutionally-enforced modules.

## Source Artifacts (from `last update/`)

| Artifact | Lines | Target Module |
|----------|------:|---------------|
| `BIZRA_Native_Algorithms_v2_ThreeMinds.py` | 990 | `core/constitutional/` (Python) + `bizra-omega/bizra-core/` (Rust) |
| `BIZRA_DECLARATION.md` | 80 | `core/constitutional/declaration.py` + `00_CONSTITUTION/` |
| `BIZRA_Chaos_Test_v2.py` | 546 | `tests/constitutional/` |
| `BIZRA_AKIS_v2.py` | 390 | `core/akis/` |
| `BIZRA_Knowledge_Extractor.py` | ~400 | `core/akis/` (merge with AKIS v2) |
| `BIZRA_KIS_v2.jsx` | ~600 | Frontend (deferred — React dashboard) |
| `BIZRA_Knowledge_Dashboard.jsx` | ~400 | Frontend (deferred — React dashboard) |
| `BIZRA_Node0_Unified_Architecture.jsx` | ~300 | Frontend (deferred — React dashboard) |

## 5-Layer Sovereignty Stack

```
┌──────────────────────────────────────────────────────────┐
│ L5: CREATIVE AI (System-2, PAT-7)                        │
│   Novel queries, chain discovery. ~5% of interactions.   │
│   Existing: core/pat/, core/inference/                   │
└──────────────────────────────────────────────────────────┘
                     ↓ only when L4 misses
┌──────────────────────────────────────────────────────────┐
│ L4: REFLEX CACHE (A10: Compiled Intelligence)            │
│   O(1) hash lookup. ~90% of interactions.                │
│   Existing: core/living_memory/, bizra-hooks             │
└──────────────────────────────────────────────────────────┘
                     ↓ always
┌──────────────────────────────────────────────────────────┐
│ L3: CONSTITUTIONAL KERNEL (15 Native Algorithms)         │
│   Ihsan, SEED minting, Gini, Zakat, Governance.         │
│   NEW: core/constitutional/                              │
└──────────────────────────────────────────────────────────┘
                     ↓ always
┌──────────────────────────────────────────────────────────┐
│ L2: SOVEREIGNTY KERNEL (Ed25519 + BLAKE3 + PCI)          │
│   Cryptographic identity, receipt verification.          │
│   Existing: core/pci/, core/proof_engine/                │
└──────────────────────────────────────────────────────────┘
                     ↓ always
┌──────────────────────────────────────────────────────────┐
│ L1: EVENT LOG (A14: Immutable History)                   │
│   Append-only ledger, Merkle chain integrity.            │
│   Existing: core/proof_engine/evidence_ledger.py         │
└──────────────────────────────────────────────────────────┘
```

## Spec Files

| # | File | Content |
|---|------|---------|
| 01 | `phase_67_01_fixed_point_arithmetic.md` | Fixed-point math kernel |
| 02 | `phase_67_02_native_algorithms.md` | 15 algorithms with Three Minds corrections |
| 03 | `phase_67_03_declaration_genesis.md` | Declaration as genesis block |
| 04 | `phase_67_04_sovereignty_cli.md` | CLI: init, work, attest, status |
| 05 | `phase_67_05_akis_pipeline.md` | AKIS v2 knowledge extraction |
| 06 | `phase_67_06_chaos_validators.md` | 10 chaos tests as constitutional probes |
| 07 | `phase_67_07_tdd_anchors.md` | Test-first contracts for all modules |

## Integration with Existing Codebase

### Reuse (NOT duplicate)
- `core/integration/constants.py` — All thresholds sourced here (IHSAN, SNR, GINI)
- `core/pci/` — PCI gates, receipt signing, BLAKE3
- `core/proof_engine/` — EvidenceLedger, genesis ceremony, canonical hashing
- `core/genesis/` — CLI framework, GenesisConfig, orchestrator
- `core/treasury/` — Token minting (extend, don't replace)
- `core/governance/` — Constitutional gates (wire A8 Shura through)

### New Modules
- `core/constitutional/` — The 15-algorithm kernel (Python)
- `core/constitutional/fixed_point.py` — Deterministic arithmetic
- `core/constitutional/algorithms.py` — A1-A15 implementations
- `core/constitutional/declaration.py` — Genesis Declaration handler
- `core/constitutional/types.py` — ActionReceipt, WalletState, etc.
- `core/akis/` — Knowledge extraction pipeline
- `core/akis/extractor.py` — Multi-source extraction
- `core/akis/relevance.py` — BIZRA relevance scoring

### Constants Alignment

All thresholds MUST be sourced from `core/integration/constants.py`:

```python
# Existing (already canonical)
UNIFIED_IHSAN_THRESHOLD = 0.95
ADL_GINI_THRESHOLD = 0.35

# New (add to constants.py)
FP_PRECISION = 1_000_000  # Fixed-point 6 decimal places
INTENT_FLOOR = 0.90       # Al-Ghazali intent pre-gate
GINI_HEALTHY = 0.35       # Khaldunian Curve healthy zone
GINI_WARNING = 0.50       # Throttle zone boundary
ZAKAT_RATE = 0.025        # 2.5% annual purification
DEMURRAGE_RATE = 0.001    # 0.1% per tick on idle balances
BLOOM_DECAY = 0.01        # 1% governance decay per tick
REFLEX_TTL = 86400        # 24h reflex cache lifetime
EQUITY_FACTOR_MIN = 1.0   # Floor for Ghazali Equity Factor
EQUITY_FACTOR_MAX = 5.0   # Cap for newcomer multiplier
ASABIYYAH_WEIGHTS = (0.4, 0.3, 0.3)  # attestations, votes, cooperation
```

## Success Criteria

1. `pytest tests/constitutional/ -q` — All tests GREEN
2. Fixed-point results identical on ARM and x86 (no float drift)
3. All 10 chaos tests pass under `tests/constitutional/test_chaos.py`
4. Declaration BLAKE2b hash matches canonical value on every platform
5. Sovereignty CLI `bizra init/work/attest/status` functional
6. AKIS extraction produces structured output for all source types
7. Zero duplicate thresholds — all sourced from `constants.py`
