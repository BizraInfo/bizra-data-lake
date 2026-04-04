# RUNTIME_WORKSPACE_TRACKING_DECISION_v1

**Date:** 2026-04-05
**Status:** DECIDED — track in bizra-data-lake, gitignore .env only

---

## Q1: Is `runtime/` part of `bizra-data-lake` canonically, or a separate product?

**Answer: Part of bizra-data-lake, as a pre-omega artifact.**

`runtime/` is `meta_alpha_dual_agentic` v2.0.0 — an earlier full-stack production system
predating the canonical `bizra-omega/` workspace. It contains 105 Rust files, 144 Python
files, and 3 sub-crates (finance-v1, bizra-gateway, bizra_bridge).

It has **zero compile-time dependencies** on any `bizra-omega/` crate. The two workspaces
are architecturally independent.

Tracking it in this repo preserves the full evolutionary lineage of BIZRA. The data lake
is the historical record. `runtime/` is evidence, not active development target.

## Q2: What is the boundary?

```
bizra-omega/      ← CANONICAL. Active development. 25 crates, 1,657+ tests.
runtime/          ← HISTORICAL. Pre-omega prototype. Independent workspace.
```

- `bizra-omega/` is the authoritative implementation of the Architecture Canon
- `runtime/` is an earlier attempt with overlapping goals but independent code
- No code should flow from `runtime/` → `bizra-omega/` without explicit porting
- `runtime/` may be referenced for design archaeology but not imported

## Q3: What state/data artifacts are always excluded?

- `runtime/.env` — may contain local secrets (already in .gitignore)
- `runtime/.env.reference` — excluded as template with potential credential patterns
- Any `runtime/target/` build artifacts (already in .gitignore via global `target/`)
- Any runtime-generated state files (logs, databases, checkpoints)

## Q4: What evidence or dependency links must be documented?

- `runtime/crates/finance-v1/` shares SEED economics concepts with
  `bizra-omega/bizra-core/src/islamic_finance.rs` — same domain, independent code
- `runtime/crates/bizra-gateway/` covers similar ground to
  `bizra-omega/bizra-api/` — both implement HTTP gateways
- `runtime/crates/bizra_bridge/` may have design patterns relevant to
  `bizra-omega/bizra-python/` (PyO3 bridge) — review before discarding

Constitutional thresholds in `runtime/` are NOT authoritative.
Authoritative source: `core/integration/constants.py` + `bizra-omega/bizra-core/src/lib.rs`

---

## Decision

Track `runtime/` in bizra-data-lake. It is part of the evolutionary record.
Active development happens exclusively in `bizra-omega/`.
