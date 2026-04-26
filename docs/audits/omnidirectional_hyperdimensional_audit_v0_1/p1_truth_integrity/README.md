# P1 Truth-Integrity Closure Pack — BIZRA v0.1

**Date:** 2026-04-24 GST
**Scope:** Close the G-FW-003 guard by neutralising `PROHIBITED` claims (20) and
staging rewrites for `NEEDS_REWRITE` claims (94). `PROOF_REQUIRED` (367) is
backlogged, not rewritten, in this pass.
**Status:** Documentation-only. No source, website, runtime, or git state
changed by this pack.

## Why this pack exists

After the P0+1 hardening closed secret-pattern findings to zero, the Flywheel
Kernel v1 shifted priority from `P0_SECRET_TRIAGE` to `P1_TRUTH_INTEGRITY`.
The guard `G-FW-003 Public claims clean` is still `BLOCK` because the docs
tree carries 20 `PROHIBITED` and 94 `NEEDS_REWRITE` claims, and the live
website hero carries the highest-risk variants (C4/C5/C7/C9) identified in
`WEBSITE_PUBLIC_CLAIMS_AUDIT.md`.

This pack produces the artefacts an operator or implementation lane needs to
close G-FW-003 without rewriting the actual docs or website yet.

## Contents

| File | Role |
|------|------|
| [`PROHIBITED_CLAIMS_REGISTER.md`](./PROHIBITED_CLAIMS_REGISTER.md) | All 20 PROHIBITED claims with category, source, excerpt, action, safe replacement, launch impact. |
| [`PROHIBITED_CLAIMS_REGISTER.csv`](./PROHIBITED_CLAIMS_REGISTER.csv) | Same data, machine-readable. |
| [`NEEDS_REWRITE_REGISTER.md`](./NEEDS_REWRITE_REGISTER.md) | All 94 NEEDS_REWRITE claims grouped by the 8 operator-requested themes. |
| [`NEEDS_REWRITE_REGISTER.csv`](./NEEDS_REWRITE_REGISTER.csv) | Same data, machine-readable. |
| [`SAFE_REWRITE_PACK.md`](./SAFE_REWRITE_PACK.md) | Approved replacement copy — hero, sovereignty, receipts, Node0, Genesis 100, performance, security/privacy. |
| [`WEBSITE_PATCH_PLAN.md`](./WEBSITE_PATCH_PLAN.md) | Surface-by-surface patch plan with current risky language, safe replacement, priority, evidence required, owner. |
| [`RECEIPT_LINKING_BACKLOG.md`](./RECEIPT_LINKING_BACKLOG.md) | The 367 PROOF_REQUIRED claims grouped by evidence type + receipt/source-chain requirements. |
| [`GO_NO_GO_AFTER_P1.md`](./GO_NO_GO_AFTER_P1.md) | Decision criteria for whether the flywheel shifts to `P2_SUPPLY_CHAIN_TRUST` after this pack lands. |

## Authoritative inputs

- `docs/audits/omnidirectional_hyperdimensional_audit_v0_1/artifacts/claims_register.json` (500 entries)
- `docs/audits/omnidirectional_hyperdimensional_audit_v0_1/artifacts/website_claims.json` (operator pre-check)
- `docs/audits/omnidirectional_hyperdimensional_audit_v0_1/artifacts/website_snapshot.txt`
- `docs/audits/omnidirectional_hyperdimensional_audit_v0_1/WEBSITE_PUBLIC_CLAIMS_AUDIT.md` (C1–C9 + K1 classification)
- `docs/audits/omnidirectional_hyperdimensional_audit_v0_1/P0_PLUS_1_HARDENING_ADDENDUM_2026_04_24.md`
- `tools/audit/flywheel_kernel/patterns.json` (FW-P004 = truth-integrity debt)

## Scope boundaries

This pack must never (in this turn, without explicit operator authorisation):

- Edit website source (`frontend/`, `dema-console/`, any publish pipeline).
- Edit historical audit markdown reports except the existing P0+1 addendum,
  and that only with an explicit operator note.
- Rewrite or delete the ~367 PROOF_REQUIRED claims themselves.
- Touch runtime code, canon packs, Origin Kernel, MEMORY.md, or PR #49 files.
- Run `git add` / `commit` / `push` / `branch` / `tag`.
- Publish anything externally.

## How to use this pack

1. Operator reads [`GO_NO_GO_AFTER_P1.md`](./GO_NO_GO_AFTER_P1.md) for the
   shape of the close.
2. Implementation lane (whoever owns public claim hygiene) picks up
   [`WEBSITE_PATCH_PLAN.md`](./WEBSITE_PATCH_PLAN.md) and works row by row.
3. Each row cites back to an entry in
   [`PROHIBITED_CLAIMS_REGISTER.md`](./PROHIBITED_CLAIMS_REGISTER.md) or
   [`NEEDS_REWRITE_REGISTER.md`](./NEEDS_REWRITE_REGISTER.md) so reviewers can
   verify the pattern match.
4. The Flywheel Kernel is re-run after each batch of applied edits. The
   target end-state is `G-FW-003: PASS` on the default audit artefact path.
