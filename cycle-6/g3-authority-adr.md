# Cycle-6 — Gate G3 (Frontend Authority) ADR

بسم الله الرحمن الرحيم

**Cycle:** 6 (Persistence + Authority Unification)
**Gate:** G3 — Frontend authority decision
**Filed:** 2026-04-17 Dubai GST
**Status:** FORMALIZATION — decision settled by lived evidence in Cycle-5, sealed as canon here

---

## Summary (one sentence)

The external **`award-winner-design`** Next.js repo is the **authoritative BIZRA frontend**; the in-repo `frontend/` (Vite) is retained as historical lineage — the same evidence-pattern-led formalization that closed G2 (`bizra-omega/` canonical vs `runtime/` historical).

## Evidence — lived precedent, not new deliberation

Cycle-5 G3b acceptance note (`cycle-5/g3-acceptance-note.md`) records that the first real principal-activation receipt flowed through this exact bridge:

- `/api/missions POST` (external Next.js, `award-winner-design` commit `40a6832`)
  proxies to `bizra-cognition-gateway /mission` (omega commit `b031fec8`)
- Verified receipt: `62a35dcd4b141a24ebe789ca13e36ec5d7027a5c47c7752c0408e97da76d93e8` — sealed through the external Next.js loop end-to-end
- vitest in the external repo: **135/135 green** with `tsc --noEmit` clean
- The same acceptance note documents the TS type additions (`GatewayVerdict`, `GatewayGateVerdict`, `GatewayRejectedClaim`, `GatewayAdmissibility`) and the UI-stable shape preservation

**No in-repo `frontend/` endpoint participated in that lived flow.**

Per the Cycle-6 execution canon: *"Tool-produced evidence outranks grep / speculation."* Lived shipping precedent outranks architectural preference.

## Decision sealed

| Question | Answer |
|---|---|
| Which frontend is authoritative for the BIZRA operator face? | **External `award-winner-design` Next.js repo** |
| What happens to in-repo `frontend/` (Vite)? | Retained as historical lineage — no active development, no CI gates on the operator path, no production deployment target |
| What lives in the authoritative external repo? | The Dema operator surface — mission intent entry, gate viewer, receipt explorer, chain visualization, principal-activation flow |
| What lives in-repo `frontend/`? | Vite SPA artifact — design exploration, tokens.ts synced with `core/integration/constants.py`, may serve as a reference for future in-repo consolidation but not production |
| Rollback path when external repo unavailable? | See §Rollback below |

## Rollback path — frontend disaster recovery

External Next.js repo unavailability is the one operational risk this ADR must name explicitly. Three tiers:

1. **Transient outage** (seconds–minutes) — operator uses `dema` CLI against the gateway directly (`dema chain`, `dema mission submit`). The CLI is the fallback operator surface. Already shipped in Cycle-5.
2. **Repo-level outage** (hours–days) — the most recently-built production Next.js binary (from `pnpm build`) can be served from any static host against the gateway's REST surface. DNS redirect to a cached artifact. Operational playbook lives in `docs/ROLLBACK-RUNBOOK-Cycle-5.md` (to be extended with this DR scenario in a follow-on docs pass).
3. **Permanent loss** (external repo archived, wiped, or migrated) — in-repo `frontend/` (Vite) is the architectural fallback: it shares `tokens.ts` with the canonical constitutional constants, has independent build tooling (no external dependency), and can be promoted to primary via a new ADR. Trigger: founder decision.

This three-tier model preserves operational continuity without forcing the in-repo fallback into active maintenance until needed.

## Constitutional filter

| Invariant | How G3 upholds it |
|---|---|
| ZANN_ZERO | Retired frontend introduces no new economic surface |
| CLAIM_MUST_BIND | Single authoritative face = single source of operator claim |
| RIBA_ZERO | No extractive pattern in UX consolidation |
| **NO_SHADOW_STATE** | Primary motivator (shared with G2): *two frontends = two operator truths*. External declared canonical eliminates this shadow. |
| IHSAN_FLOOR | 0.95 enforcement stays at kernel layer; frontend is presentation of lawful state, never a bypass |

## What G3 does NOT claim

- Does not delete the in-repo `frontend/` directory — historical lineage preserved (same pattern as `runtime/` retention in G2)
- Does not preclude future consolidation (in-repo or external) — explicit new ADR required
- Does not decide the Dema UX itself — that is continuous product work on the external repo, not a cycle gate
- Does not address internal tool UIs (debugging dashboards, admin panels) — scope limited to the operator's principal face

## Implication for Cycle-6 G4 (E2E polyglot)

G3 closure unblocks G4's real implementation. The intentional-red `e2e-polyglot` workflow (`.github/workflows/e2e-polyglot.yml`) can now have its `scripts/e2e-polyglot/test.sh` scaffold replaced with a real end-to-end script that:

- Starts `bizra-cognition-gateway` (omega release binary)
- POSTs a mission through the external `award-winner-design` Next.js proxy
- Reads the sealed receipt via `dema chain --since today`
- Verifies chain integrity post-restart (G1's live criterion applied through external proxy)

That replacement is the next Cycle-6 arc after this ADR lands.

## Downstream enabler — `dema-overlay.jsx` placement

The 33-KB React component `dema-overlay.jsx` in `archive/downloads-files-7-2026-04-17/` (per that archive's INVENTORY §C2) now has a canonical destination: **the external `award-winner-design` repo**, not in-repo `frontend/src/components/`. Placement decision deferred to founder review of the component content against existing Dema component library.

## References

- Cycle-5 G3b acceptance note: `cycle-5/g3-acceptance-note.md`
- Gateway proxy precedent: `award-winner-design` commit `40a6832`
- Gateway HTTP surface: `cycle-6/g1-authority-adr.md` (gateway canonical via G2)
- G2 ADR (format precedent for evidence-led formalization): `cycle-6/g2-authority-adr.md`
- Cycle-6 execution canon (tool-evidence-outranks-speculation rule): `cycle-6/execution-canon.md`
- Downloads archive INVENTORY (C2 overlay placement queued): `archive/downloads-files-7-2026-04-17/INVENTORY.md`

## Signature

Filed: Mumo (Muhammad Beshr) — 2026-04-17 Dubai GST
Cycle chain position: 6 / G3
Canon status: **SEALED** — external `award-winner-design` is the authoritative BIZRA frontend. G4 real implementation may now proceed.

الحمد لله.
