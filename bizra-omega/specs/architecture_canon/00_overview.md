# BIZRA Architecture Canon — Overview

> Frozen: 2026-04-04 (Dubai)
> Source: Multi-model convergence (PAT/SAT/URP visualization stabilized)

## One-Line Formula

```
Human -> DEMA -> PAT-7 (local action) -> FATE boundary -> SAT-5 (system validation) -> URP (shared world)
```

## Governing Law

**PAT serves the person. SAT serves the constitution. URP serves the world. FATE decides what may cross between them.**

## Spec Modules

| # | File | Domain | Scope |
|---|------|--------|-------|
| 01 | `01_node_sovereignty.md` | User Domain | Local node lifecycle, PAT-7 ownership, offline-first |
| 02 | `02_pat7_agent_council.md` | User Domain | PAT-7 roles, local mission loop, decomposition |
| 03 | `03_fate_trust_boundary.md` | Judiciary | Proof-carrying request wrapping, admissibility |
| 04 | `04_sat5_system_validators.md` | System Domain | SAT-5 roles, constitutional enforcement |
| 05 | `05_urp_fabric.md` | Shared Domain | Resource pool, federation, marketplace |
| 06 | `06_genesis_mint.md` | Bootstrap | Identity minting, agent minting, URP creation |
| 07 | `07_seed_economics.md` | Economics | SEED/BLOOM, PoI, zakat, Gini, Harberger |
| 08 | `08_federation_a2a.md` | Network | Node-to-URP, gossip, A2A, scaling model |

## Ownership Domains

```
USER DOMAIN                    JUDICIARY                      SYSTEM DOMAIN
-----------                    ---------                      -------------
PAT-7                          FATE trust boundary            SAT-5
local hardware                 admissibility gate             URP-governed
private keys + data            proof-carrying request         public / systemic
serves mission intent          verifies crossings             serves constitution
```

## Architecture Invariants

1. A node MUST be alive alone. The commons is optional for liveness.
2. PAT-7 agents are USER-OWNED. They serve the human, not the system.
3. SAT-5 agents are SYSTEM-OWNED. They serve the constitution, not the user.
4. Only proof-carrying requests cross the FATE boundary. No raw authority leaks.
5. Receipts are the proof-native output. Every visible effect has a receipt.
6. Federation amplifies already-live organisms. It does not create liveness.
7. Constitutional thresholds are compiled-in, not configurable at runtime.
