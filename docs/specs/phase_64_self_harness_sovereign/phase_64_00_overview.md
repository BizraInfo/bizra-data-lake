# Phase 64 — Self-Harness Sovereign Engine

## Overview

Phase 64 transforms the node from "software running on hardware" to
"hardware that IS the node." The Self-Harness Sovereign Engine provides
five capabilities that close the economic loop:

1. **Asset Registry** — the node introspects its own body (CPU, GPU, RAM, disk, models)
2. **URP Contribution Protocol** — idle capacity flows to the network as إيثار
3. **Payback Tracker** — measures when device investment becomes net-positive
4. **Floor Constraint** — enforces minimum viable node universality
5. **Reverse Scaling Calibrator** — tunes SEED/BLOOM coupling constant

## Standing on Giants

- **Shannon (1948)** — capacity measurement: the node must know its own channel capacity
- **Ostrom (1990)** — commons governance: URP is a shared resource pool with constitutional rules
- **Baran (1964)** — distributed networks: surplus flows from strong nodes to weak ones
- **Boyd (1976)** — OODA: observe own state → orient in network → decide contribution → act
- **Al-Ghazali (1095)** — Ihsan: excellence in contribution is a constitutional requirement
- **Anthropic (2023)** — constitutional gates: every economic action is Ihsan-gated

## Architectural Position

```
Existing                              Phase 64 NEW
───────────────────                   ──────────────────────
core/elite/self_harness_engine.py     core/elite/asset_registry.py
  (code quality scanner)              core/elite/urp_contributor.py
core/genesis/urp.py                   core/elite/payback_tracker.py
  (pledge stub — unsigned)            core/elite/floor_constraint.py
core/sovereign/sat_controller.py      core/elite/scaling_calibrator.py
  (URPSnapshot — read-only)
core/token/minter.py                  All gated by:
  (SEED/BLOOM minting)                 - core/integration/constants.py
                                        - core/pci/gates.py
                                        - core/governance/constitutional_gate.py
```

## Phase Numbering

| File | Module | Lines (est) |
|------|--------|-------------|
| phase_64_01 | Asset Registry | ~450 |
| phase_64_02 | URP Contribution Protocol | ~450 |
| phase_64_03 | Payback Tracker | ~350 |
| phase_64_04 | Floor Constraint | ~350 |
| phase_64_05 | Reverse Scaling Calibrator | ~300 |
| phase_64_06 | TDD Anchors | ~400 |

## Key Principle: The Node IS the Computer

This is not a metaphor. It determines the API surface:

| Tenant Model (WRONG) | Sovereign Model (CORRECT) |
|----------------------|--------------------------|
| PAT "requests GPU time" | PAT "manages its own organ" |
| SAT "monitors infrastructure" | SAT "maintains its own body" |
| Gateway "discovers services" | Gateway "introspects capabilities" |
| Node "accesses resources" | Node "owns resources" |

Every API in Phase 64 uses sovereign framing. Resources are not
requested — they are managed. Capacity is not allocated — it is
contributed. The node doesn't have assets — the node IS its assets.

## Constitutional Gates

All Phase 64 operations are gated:

| Gate | Threshold | Source |
|------|-----------|--------|
| Ihsan | >= 0.95 | `UNIFIED_IHSAN_THRESHOLD` |
| SNR | >= 0.85 | `UNIFIED_SNR_THRESHOLD` |
| ADL Gini | <= 0.35 | `ADL_GINI_THRESHOLD` |
| Zakat | 2.5% on mint | `core/token/minter.py` |

No SEED is minted, no URP contribution is recorded, no payback
is credited without passing all constitutional gates.

## Dependencies

- `core/integration/constants.py` — thresholds (single source of truth)
- `core/token/minter.py` — SEED/BLOOM minting with Zakat deduction
- `core/genesis/urp.py` — URPPledge (will be extended, not replaced)
- `core/sovereign/sat_controller.py` — URPSnapshot (consumed, not modified)
- `core/proof_engine/evidence_ledger.py` — all economic actions are evidenced
- `core/pci/gates.py` — PCIGateKeeper for proof-carrying verification
