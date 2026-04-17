# BIZRA Agent Instruction — Cycle-6 Execution Canon

بسم الله الرحمن الرحيم

**Filed:** 2026-04-17 (Friday), Dubai GST
**Status:** CANONICAL for Cycle-6 duration
**Authority:** Founder (Mumo / Muhammad Beshr) direction, 2026-04-17 14:07 GST session

---

## Scope

You are operating inside the BIZRA polyglot production system. Your job is to advance the current cycle without creating drift, shadow state, or cross-stack ambiguity.

## Read first, before any change

1. `cycle-6/niyyah.md`
2. `cycle-6/g2-authority-adr.md`
3. `runtime/TRACKING_DECISION.md`
4. `runtime/RUNTIME_STATUS.md`
5. `docs/BIZRA-Handover-v1.md`
6. `docs/BIZRA-Repo-Inventory-v1.md`
7. `Justfile`

## Current canonical truth

- `bizra-omega/` is the canonical active Rust authority.
- `runtime/` is historical / pre-omega and may be used as evidence, not as the source of new authority.
- Cycle-6 G2 is already **SEALED** (`cycle-6/g2-authority-adr.md`, commit `7c5315d6`).
- Cycle-6 G4 is **scaffolded and intentionally red by design** (commit `3e6e9ce1`, workflow `e2e-polyglot.yml`).
- Cycle-6.5 audit-tools mini-arc is **complete** (commit `4c8275a7`).
- Tool-produced evidence outranks grep / speculation for security claims.

## Hard rules

- No force-push.
- No history rewrite.
- No silent correction of wrong prior claims; keep superseded claims visible and annotate them honestly.
- No touching the 235-file drift set unless the founder explicitly opens that scope.
- No widening scope across multiple Cycle-6 gates in one pass.
- No new authority decisions when a prior sealed decision already exists in `TRACKING_DECISION.md` or a filed ADR.

## G1 rule (important)

Treat G1 as: **durable-read persistence only.**

Do NOT include signer audit in G1 unless the founder explicitly changes scope, or unless reading the real `sovereign_state/` format proves signer identity is structurally inseparable from the durable-read path.

Default founder answer: **`/@ no`**

Meaning:
- keep G1 narrow
- draft `cycle-6/g1-authority-adr.md` for durable-read persistence only
- defer signer audit to Cycle-6.5b or Cycle-7

## Execution order

1. Read `bizra-node0/core/sovereign/` and inspect `sovereign_state/`
2. Draft `cycle-6/g1-authority-adr.md`
3. Only then begin G1 code
4. Leave G3 and G4 untouched unless explicitly opened

## DevOps / CI discipline

- Respect fail-closed behavior
- Preserve the intentional-red `e2e-polyglot` workflow until the real G4 contract is implemented
- Use `just` recipes where available (`just audit-rust`, `just audit-python`, `just check`, `just dev`)
- Prefer tool-produced audit evidence (`cargo-audit`, `pip-audit`) over filesystem inference

## Constitutional filter

Every change must preserve:

- **IHSAN_FLOOR**
- **ZANN_ZERO**
- **RIBA_ZERO**
- **CLAIM_MUST_BIND**
- **NO_SHADOW_STATE**

## Completion format

When reporting, state:

- what changed
- which gate moved
- what remains open
- whether any claim is speculative or tool-verified

---

## Lineage note

This canon codifies founder direction given in-session on 2026-04-17 14:07 GST. It formalizes the narrow-real discipline that produced the Cycle-5 close-out and the Cycle-6 niyyah. It supersedes any implicit operating assumptions previously held by agents working this repo.

الحمد لله.
