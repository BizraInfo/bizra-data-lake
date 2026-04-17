# Cycle-6 — Gate G2 (Gateway Authority) Acceptance Note

بسم الله الرحمن الرحيم

**Cycle:** 6 (Persistence + Authority Unification)
**Gate:** G2 — Gateway authority decision
**Filed:** 2026-04-17 (Friday), Dubai GST
**Status:** FORMALIZATION — founder decision pre-dates cycle opening

---

## Summary (one sentence)

`bizra-omega/` is the authoritative BIZRA workspace; `runtime/` is historical lineage — a decision the founder committed on **2026-04-05** (twelve days before Cycle-6 niyyah) and which Cycle-6 hereby seals as canon.

## Evidence — no new deliberation

The G2 gate per `cycle-6/niyyah.md` §G2 asks:

> "reconcile `bizra-cognition-gateway` (Cycle-5 ship, bizra-omega workspace) with the pre-existing `bizra-gateway` in `runtime/`. Both implement HTTP surfaces for sovereign runtime state."

The reconciliation is **already on origin** in `runtime/TRACKING_DECISION.md` (committed 2026-04-05, well before Cycle-6). Verbatim:

> **bizra-omega/** ← CANONICAL. Active development. 25 crates, 1,657+ tests.
> **runtime/** ← HISTORICAL. Pre-omega prototype. Independent workspace.
>
> - `bizra-omega/` is the authoritative implementation of the Architecture Canon
> - `runtime/` is an earlier attempt with overlapping goals but independent code
> - No code should flow from `runtime/` → `bizra-omega/` without explicit porting
> - `runtime/` may be referenced for design archaeology but not imported
> - Constitutional thresholds in `runtime/` are NOT authoritative

**Scope implication for Cycle-6:** G2 is sealed by reference, not by new choice. This frees G2-allocated effort for G1 substance and G4 promotion.

## Decision sealed

| Question | Answer |
|---|---|
| Which gateway is authoritative for the BIZRA HTTP surface? | `bizra-omega/bizra-cognition-gateway` |
| What happens to `runtime/crates/bizra-gateway`? | Retained as historical lineage — no active development, no CI gates, no deployment |
| Can code flow `runtime/` → `bizra-omega/`? | Only via explicit porting with doctrinal review |
| Are `runtime/` constitutional thresholds authoritative? | NO — `core/integration/constants.py` + `bizra-omega/bizra-core/src/lib.rs` are SSOT |
| Does `runtime/` block omega releases? | NO (per `runtime/RUNTIME_STATUS.md` §"Does runtime/ block omega releases?") |

## Constitutional filter

| Invariant | How G2 upholds it |
|---|---|
| ZANN_ZERO | Retired gateway introduces no new economic surface |
| CLAIM_MUST_BIND | Single authoritative gateway = single evidence-binding root |
| RIBA_ZERO | No extractive pattern introduced by authority consolidation |
| **NO_SHADOW_STATE** | Primary motivator: two gateways = two truths. Naming one authoritative eliminates one shadow. |
| IHSAN_FLOOR | 0.95 enforcement stays at kernel layer `IhsanFloorGate` in `bizra-omega` |

## What G2 does NOT claim

- Does not delete `runtime/` from the repo — it stays as historical lineage per `TRACKING_DECISION.md` Q1
- Does not retrofit `runtime/` vulnerabilities onto omega — shared transitive deps (`rustls-webpki`, `bytes`) are audited independently per `RUNTIME_STATUS.md`
- Does not eliminate the `meta_alpha_dual_agentic` workspace — preserved for design archaeology
- Does not preclude future merging of specific patterns from `runtime/` → `bizra-omega/` (explicit porting path remains open)

## References

- Founder precedent: `runtime/TRACKING_DECISION.md` (committed 2026-04-05)
- Runtime vulnerability status: `runtime/RUNTIME_STATUS.md` (refresh filed separately this cycle)
- Cycle-6 niyyah: `cycle-6/niyyah.md` §G2
- Cycle-5 G3 acceptance (format precedent): `cycle-5/g3-acceptance-note.md`

## Signature

Filed: Mumo (Muhammad Beshr) — 2026-04-17 Dubai GST
Cycle chain position: 6 / G2
Canon status: **SEALED** — supersedes any contrary architectural statement in earlier cycles.

الحمد لله.
