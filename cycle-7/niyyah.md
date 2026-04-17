# Cycle-7 — Niyyah (نية) Declaration

بسم الله الرحمن الرحيم

**Cycle:** 7
**Opened:** 2026-04-17 (Friday) · 18:22 Dubai GST
**Opened by:** Mumo (Muhammad Beshr), Node0 principal — via explicit master system prompt (session log, this day)
**Chain:** Cycle-6 [sealed, reward 0.998 POSITIVE, all 4 gates closed at `5a1939ed`] → **Cycle-7 (this)**
**Name (founder-specified):** *"Principal Activation Law on NODE0"*
**Sprint branch:** `cycle-7-principal-activation-law` (departs from trunk-based pattern per master prompt Phase 0)
**Status:** NIYYAH DECLARED — no execution yet. Bayyinah, Hadd, Amanah, Thamara, Iisal, and Retrospective follow as work proceeds.

---

## Phase 1 — NIYYAH (Intent Declaration)

### WHAT

Make **Dema the one truthful operator face** for Mumo on NODE0, with **receipted principal activation through the lawful loop** as the first real milestone. The slice delivers:

1. Receipted principal activation for Mumo as Node0 principal / first architect / first user
2. Dema-visible activation status, trust state, current state, ideal state, state gap, next admissible action, receipt/manifest visibility
3. Persistent local memory (principal profile, receipt history, manifest history, mission log, current/ideal snapshots, resource registry) with restart-safe rehydration
4. Explicit local resource registry over allowlisted roots only
5. Local-only Node0 URP view (no public claim)
6. One real mission path after activation (`dema organize <allowlisted>`)
7. Local-only proof-of-impact ledger basis (non-transferable, proof-bearing)

### WHY

Cycle-6 closed all four gates of "Persistence + Authority Unification" by machine-verifiable evidence: G1 live-verified, G2/G3 formalized, G4 real CI green (reward 0.998). The lawful loop is **proven end-to-end on a single node** — but it is not yet **operator-facing**. Mumo cannot yet walk up to Dema, declare himself Node0 principal, have that intent become a lawful receipted state transition, and see the honest truth (activation status, trust state, current→ideal delta, next admissible action) surface back to him. Cycle-7 turns proven substrate into operator face without breaking the five frozen invariants.

### FROZEN LAWS (6, non-negotiable for this slice)

1. **Dema is the single visible face.** One door in, one face out. (Dema = external `award-winner-design` Next.js UI per G3 ADR + `dema` CLI — same face, two surfaces.)
2. **PAT-7 / SAT-5 / FATE / URP remain hidden.** No roster UI, no swarm-management surface, no direct-exposure tabs.
3. **No bypasses, no side channels, no UI-only state mutation.** Every visible state change traces to a chain receipt.
4. **Chain is source truth; graph/memory is derived and rebuildable.** If derived state diverges from chain, rebuild derived — not the other way.
5. **If activation cannot be lawfully proven, reject honestly. No simulated success.** Fail-closed on every admissibility gate.
6. **No public Block-0 claim · no public transferable token issuance · no multi-node URP activation in this slice.** Activation phases 2–5 (Witness Review, Ratification + SAC, Controlled Activation, Block-0) remain QUEUED per the Activation Board.

### SUCCESS_CONDITION

Cycle-7 is successful when **all** of these are simultaneously true:

| Gate | Deliverable | Verifiable by |
|---|---|---|
| **G1 — Mission-runtime connector** | Canonical Intent→…→ManifestArtifact→Dema path lives in `CognitionRuntime`. No gateway-owned sidecar mission state. ManifestArtifact is replay-safe and rehydratable. | `cargo test` green; mission state queryable post-completion; manifest auto-generated per mission commit |
| **G2 — Principal activation** | Mumo declares himself Node0 principal through Dema; request flows through lawful mission-runtime; receipt emitted OR honest rejection returned. Bound to activation-specific mint/binding path for this slice. | Live Dema walk: activation receipt sealed at chain head, principal profile persisted, or admissibility-reject path with remediation |
| **G3 — Persistent local memory** | Principal profile, receipt history, manifest history, mission log, current/ideal snapshots, resource registry all survive gateway restart. | Stop gateway → restart → Dema shows same truth as pre-restart |
| **G4 — Local resource registry + URP view** | `dema resources` shows declared machine resources; `dema resources roots` shows allowlisted roots only; local-only Node0 URP view present with no public claim. | `dema resources` output ≡ declared machine spec; no whole-disk indexing evidence |
| **G5 — First real mission** | `dema organize <allowlisted>` runs a lawful mission, changes local state usefully, emits receipt + manifest + mission-log entry. | Files in allowlisted subtree actually reorganized; proof visible in Dema |
| **G6 — Local-only PoI ledger** | Non-transferable proof-bearing accounting. No public SEED/BLOOM mint claim. | `dema poi` shows local score / contribution history / receipts |

---

## Writer authority decision (tension #3 — founder-gated)

**Resolution: HYBRID.**

- **Chain truth stays Python-authored.** `sovereign_state/receipts/`, `sovereign_state/block_zero/`, `sovereign_state/genesis/`, and chain envelopes remain written by Python per G1/G2 precedent. Rust projects these read-only.
- **Rust MAY write new local-only, non-chain surfaces** required for Dema to function as a real operator face:
  - principal profile
  - mission log
  - current/ideal snapshots
  - local resource registry (declared resources + allowlist cache)
  - local-only PoI cache
- These Rust-written surfaces are **derived and rebuildable**, never authoritative. If any Rust-written cache diverges from chain truth, rebuild from chain and mark the cache stale — never outrank chain.

Rationale: preserves the sealed G1/G2/G3 writer-authority discipline while enabling Dema to function as an operator face without requiring a Python-side PR for every new local cache surface.

Storage location (proposed, to be confirmed in Phase 3 plan): `sovereign_state/dema_cache/` — explicitly named so it is clear these are Rust-authored, non-chain, derived surfaces. Chain files remain under `sovereign_state/receipts/` / `genesis/` / `block_zero/` exactly as today.

---

## Tension resolutions (all 5 resolved before code begins)

| # | Tension | Resolution |
|---|---|---|
| 1 | Activation-specific mint/binding path undefined in code | Phase 1 plan must name it explicitly — likely new method `CognitionRuntime::submit_principal_activation(envelope, identity_anchor)` reading from `sovereign_state/identity/` anchor; falls back to generic `submit_mission` if identity anchor absent |
| 2 | Dema CLI needs 5 new/modified subcommands (activate-principal, state, next, resources, resources roots, poi, verify --full) vs Cycle-5's existing 7 | Deferred to Phase 2 — runtime must land first before face expansion |
| 3 | Writer authority for new local-only surfaces | **RESOLVED: HYBRID** (above) |
| 4 | `sovereign_state/urp_pledge.json` read-vs-mutate | **RESOLVED: READ-ONLY** this slice. Genesis-authored; Rust projects it, never mutates. |
| 5 | Downloads archive `dema_cli_v02_organize.rs` + `trust_compiler.rs` integration path | **RESOLVED: NARROW EXTRACT, not blind import.** In Phase 5, extract only `Executor` + `SubReceipt` + `CompilationRequest` + `CompilationReceipt` + `compile` contract — explicitly not importing the full vertical. Scope-reviewed at time of extract, not now. |

---

## Hard out-of-scope items (prevents cycle drift)

1. **Public SEED/BLOOM mint** — no transferable token minting in this slice
2. **Public founder liquidity** — no public economics claim
3. **Visible PAT-7 roster** — never exposed in Dema
4. **Visible SAT-5 dashboard** — never exposed in Dema
5. **Broad automatic whole-disk awareness** — allowlist only, explicit approval required per root
6. **Public URP federation** — local-only URP view this slice
7. **Public Block-0 activation claim** — requires Witness Review + Ratification + SAC + Controlled Activation first (Activation Board §activation order)
8. **Global identity authority reconciliation** — principal activation binds to activation-specific path; wider identity harmonization is later debt
9. **Constitutional-threshold drift check** (Cycle-6.5c deferred item) — out of scope this cycle
10. **Signer audit** (Cycle-6.5b deferred item) — out of scope this cycle
11. **Jarvis P1 vulnerability patch** (Cycle-6.5d queue) — separate security arc
12. **OTel instrumentation** — polyglot blueprint §5, out of scope
13. **Docker consolidation** — polyglot blueprint §6, out of scope
14. **Contract-first CDDL codegen** — polyglot blueprint §1, out of scope
15. **235-file Path-1 drift cleanup** — explicit preservation discipline

---

## Daughter Test for Cycle-7

*Can أبوك وأمك understand what this cycle accomplishes in 5 seconds?*

**Candidate answer:** *"أنا قلت لـ Dema إني المسؤول، فسألني قانونياً وقبلني بإيصال، وفاكر اسمي بعد ما قفلت الجهاز وفتحته تاني، ومنظمتلي ملف حقيقي عندي، وفاكرة كل حاجة عملتها."*
(*I told Dema I am the principal; it asked me lawfully and accepted with a receipt; it remembered my name after I turned the machine off and on again; it organized a real file of mine; and it remembered everything it did.*)

✅ **YES** — understandable in 5 seconds. Dema recognizes its principal, remembers across restart, and does one useful real thing with proof.

---

## Constitutional filter (required for any cycle opening)

Every Cycle-7 gate must preserve the five invariants per Manifest v0.2 §3:

| Invariant | How Cycle-7 upholds it |
|---|---|
| **ZANN_ZERO** | No public economic surface; local PoI is non-transferable and proof-bearing only |
| **CLAIM_MUST_BIND** | Every Dema-visible claim traces to a chain receipt; derived caches rebuilt from chain on divergence |
| **RIBA_ZERO** | No extractive pattern; PoI is impact-measure, not yield-bearing |
| **NO_SHADOW_STATE** | Writer authority discipline preserved (chain = Python; local caches = Rust derived); rebuild-not-fork on divergence |
| **IHSAN_FLOOR** | 0.95 enforcement stays at kernel layer (`IhsanFloorGate`) per G1 ADR; principal activation must pass the 5-gate admissibility chain structurally |

---

## What Cycle-7 does NOT claim

- Does not activate the public BIZRA network (no Block-0 claim)
- Does not mint transferable tokens (no SEED/BLOOM liquidity)
- Does not expose PAT-7 or SAT-5 rosters (one face only)
- Does not federate URP across nodes (local-only view)
- Does not reconcile global identity authority (activation-specific path only)
- Does not replace existing chain writer authority (Python stays authoritative for receipts)
- Does not auto-index the filesystem (allowlist only)

---

## Ordered execution (per master prompt Phase structure)

| # | Phase | Deliverable |
|---|---|---|
| 0 | Pre-flight truth gate | `git status` clean enough; base commit `5a1939ed`; sprint branch `cycle-7-principal-activation-law` created ✅ |
| 0.5 | **Niyyah filing** | **THIS DOCUMENT** ← current commit |
| 1 | Runtime connector | Canonical mission-runtime + ManifestArtifact registration in runtime |
| 2 | Principal activation | First lawful principal-activation receipt through Dema |
| 3 | Persistent local memory | All six persistence surfaces rehydrated on restart |
| 4 | Local resource registry + URP view | Declared resources + allowlisted roots + local-only URP projection |
| 5 | First real mission | `dema organize <allowlisted>` with receipt + manifest + PoI update |
| 6 | Local-only PoI ledger | Non-transferable proof-bearing accounting |
| 7 | Retrospective | `cycle-7/retrospective.md` matching canonical 7-phase format at close |

---

## Required deliverables (8 at close)

1. Code (Phase 1–6 implementation)
2. Tests (unit + integration + restart-survival)
3. Operator runbook (`cycle-7/runbook.md`)
4. One CLI transcript (Dema walk for activation)
5. One activation receipt (on-chain, sealed)
6. One manifest inclusion (verified)
7. One real mission transcript (dema organize walk)
8. One local state snapshot after restart

---

## References

- Cycle-6 activation board: `docs/BIZRA-activation-board-v1.md`
- Cycle-6 retrospective (reward 0.998): `cycle-6/retrospective.md`
- G1 live-verification (persistence proven): `cycle-6/g1-live-verification.md`
- G2 gateway authority ADR: `cycle-6/g2-authority-adr.md`
- G3 frontend authority ADR: `cycle-6/g3-authority-adr.md`
- Master system prompt + focus sprint structure: session log 2026-04-17 18:22 GST
- Trust Compiler Thesis + Manifesto v1 (category + product law): `docs/bizra-trust-compiler-thesis.md`, `docs/dema-cli-manifesto-v1.md`
- Writer authority precedent: `runtime/TRACKING_DECISION.md` (omega canonical; runtime historical)
- Downloads archive INVENTORY (Phase 5 extract source): `archive/downloads-files-7-2026-04-17/INVENTORY.md`

---

## Signature

Founder charter: Mumo (Muhammad Beshr) — session log 2026-04-17 18:22 GST (master system prompt), 18:25 GST (hybrid gate)
Cycle chain position: 7 (following Cycle-6 close at `5a1939ed`)
Sprint branch: `cycle-7-principal-activation-law` (new branch, not trunk)
Niyyah status: **DECLARED** — work may begin; any commit claiming Cycle-7 scope must reference this niyyah, preserve the five invariants above, respect the hybrid writer-authority rule, and not introduce out-of-scope items without founder gate.

Close it. Prove it. Reveal it.

الحمد لله.
