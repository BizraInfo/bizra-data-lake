# Phase 82 — URP Python→Rust Bridge Spec Index

> LOCKED: v1.0 | 2026-03-14
> Constitutional Anchor: Law 6 (Sovereign Economics) + Proof-of-Impact
> Standing on Giants: PyO3 team, Weyl & Posner (Harberger), Ostrom (Commons)

## Context

Python `core/genesis/urp.py` creates signed URP pledges. Rust `bizra-resourcepool` has the
full pool implementation (1,760 LOC: registration, contribution tracking, Zakat, Harberger,
Gini enforcement). **They don't talk to each other.** The PyO3 bridge (`bizra-python`) exposes
core types but zero resourcepool types.

This spec defines the bridge that lets Python submit pledges to Rust for validation,
register nodes, track contributions, and calculate rewards — all at Rust speed.

## Spec Modules

| # | File | Domain | Target LOC |
|---|------|--------|------------|
| 01 | `01_bridge_types.md` | PyO3 type wrappers for Rust structs | ~300 |
| 02 | `02_bridge_operations.md` | Submit pledge, register node, contribute, rewards | ~350 |
| 03 | `03_tdd_anchors.md` | Test plan: unit, integration, property-based | ~300 |

## Architecture

```
Python (core/genesis/urp.py)          Rust (bizra-resourcepool)
   pledge_resources()                    ResourcePool
   verify_pledge_signature()             ├── register_node()
        │                                ├── contribute_resources()
        ▼                                ├── process_zakat()
   PyO3 Bridge (bizra-python)           ├── calculate_gini()
   ├── PyURPPledge                      ├── check_adl()
   ├── PyResourcePool                   └── stats()
   ├── PyPoolNode
   ├── submit_pledge()
   ├── register_node()
   ├── contribute()
   └── get_rewards()
```

## Dependencies

- `bizra-omega/bizra-python/src/lib.rs` — existing PyO3 module (extend)
- `bizra-omega/bizra-resourcepool/src/lib.rs` — Rust pool (read-only)
- `core/genesis/urp.py` — Python pledge creator (read-only)
- `core/integration/constants.py` — thresholds (read-only)

## Design Laws

1. **Fail-closed**: If Rust bridge unavailable, Python returns `None` (never crashes)
2. **Same commit rule**: New PyO3 bindings + Python wrapper + tests in one commit
3. **No new dependencies**: Uses existing PyO3 + bizra-resourcepool crates
4. **Decimal precision**: All financial calculations use `rust_decimal::Decimal` (Rust side)
5. **Ed25519 signatures verified in Rust**: Never trust Python-side verification alone
