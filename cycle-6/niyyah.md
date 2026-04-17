# Cycle-6 — Niyyah (نية) Declaration

بسم الله الرحمن الرحيم

**Cycle:** 6
**Opened:** 2026-04-17 (Friday) · 13:28 Dubai GST
**Opened by:** Mumo (Muhammad Beshr), Node0 principal — via explicit written charter (session log, this day)
**Chain:** Cycle-5 [sealed, reward 0.971 POSITIVE] → **Cycle-6 (this)**
**Name (founder-specified):** *"Persistence + Authority Unification"*
**Status:** NIYYAH DECLARED — no execution yet. This document is the cycle-opening anchor. Bayyinah, Hadd, Amanah, Thamara, Iisal, and Retrospective follow as work proceeds.

---

## Phase 1 — NIYYAH (Intent Declaration)

### WHAT

Resolve the three authority-fragmentation findings surfaced in the Cycle-5 polyglot repo inventory audit, in the explicit order the founder named:

1. **Persistence arc** — bridge the gap between the Rust `bizra-cognition-gateway`'s ephemeral `InMemoryPayloadStore` and the Python stack's real persistent `sovereign_state/` (2,512 files: `block_zero/`, `genesis/`, `identity/`, `agent_db/`, `bridge_receipts/`, etc.). Until this closes, *"what did my node prove today"* remains a lifecycle-fragile question — receipts evaporate on gateway restart.
2. **Gateway authority decision** — reconcile `bizra-cognition-gateway` (Cycle-5 ship, bizra-omega workspace) with the pre-existing `bizra-gateway` in `runtime/` (part of `meta_alpha_dual_agentic v2.0.0` package). Both implement HTTP surfaces for sovereign runtime state. **Two parallel truth systems is the NO_SHADOW_STATE violation pattern at the architectural layer.** One must become authoritative; the other either retires or becomes a bounded adapter.
3. **Frontend authority decision** — reconcile external Next.js `award-winner-design` (Cycle-5 bridge target) with internal Vite `frontend/`. Same principle: one primary face, or explicitly separated roles.
4. **Promote E2E polyglot smoke** — take the ad-hoc `/tmp/g4-mumo-walk.sh` that proved the first activation receipt end-to-end and elevate it to a versioned, repo-committed asset under `scripts/e2e-polyglot/`. CI-integrated so every push verifies the full polyglot flow, not just per-language unit tests.
5. **DEFER** — contract-first codegen (CDDL/Proto) is NOT this cycle. It touches every language. Opening it before authority boundaries are clean would compound drift instead of closing it.

### WHY

The Cycle-5 session produced a new Rust-native chain bridge (gateway + dema CLI + doctrine) and an honest polyglot inventory. The inventory surfaced **three open authority questions** that are not implementation details — they are architectural decisions with real NO_SHADOW_STATE implications:

- Two gateways claiming to project sovereign state ≠ one source of truth
- Two frontends addressing the operator ≠ one face (violates Manifest §8)
- Ephemeral Rust-gateway receipts alongside real Python `sovereign_state/` ≠ one chain of truth

Leaving these unresolved compounds: every future arc (tool execution, LLM inference, FTAP, contract codegen) has to pick one side or bridge both. Forcing a decision NOW, before new scope lands on top, is the narrow-real discipline that Cycle-5 taught us works.

### SUCCESS_CONDITION

Cycle-6 is successful when all four gates pass:

| Gate | Deliverable | Verifiable by |
|---|---|---|
| **G1 — Persistence** | `bizra-cognition-gateway` startup reads existing chain from `sovereign_state/` (or a unified persistence layer). `dema chain --since today` returns a truthful answer after gateway restart. | Live curl: seal receipt X → restart gateway → `/chain/X` still returns the receipt. |
| **G2 — Gateway authority** | One of (`bizra-cognition-gateway`, `bizra-gateway`) is declared authoritative; the other either retires (removed) or becomes a bounded adapter (with documented scope). Decision committed to `docs/` as an architectural decision record (ADR). | ADR file on origin naming the decision, with rationale + migration plan. |
| **G3 — Frontend authority** | One of (`award-winner-design`, `frontend/`) is declared primary; the other either retires or has an explicitly separated role. ADR filed. | ADR file on origin. |
| **G4 — Polyglot E2E** | `scripts/e2e-polyglot/` contains the full-stack smoke test; one CI workflow runs it on every push; the test proves a real receipt sealed through the polyglot chain. | Green CI run of `e2e-polyglot` workflow on any push. |

---

## Ordered execution (the founder-specified sequence)

Per Mumo's explicit charter in the session log:

> *"Do not jump to OTel, Docker consolidation, or full contract-codegen first. The strongest Cycle-6 spearpoint is persistence and authority unification. In exact order..."*

| # | Sub-arc | Why this order |
|---|---|---|
| 1 | **Persistence arc** (G1) | Unblocks everything else. A durable chain is the prerequisite for any decision about which gateway to retire — because retiring a gateway before its receipts are persistent would lose history. |
| 2 | **Gateway authority decision** (G2) | Must be decided once persistence is unified — because the decision may include "merge both gateway surfaces into one crate that uses the unified persistence." |
| 3 | **Frontend authority decision** (G3) | Depends on G2 — the chosen gateway is what the chosen frontend binds to. |
| 4 | **Promote E2E polyglot smoke** (G4) | Only meaningful once G2 + G3 have produced stable boundaries to test against. Testing ephemeral gateways and competing frontends would be testing the wrong thing. |
| (deferred) | Contract-first CDDL/Proto codegen | NOT Cycle-6. Opens after G2+G3 are stable, because codegen must target one authoritative type surface per boundary, not two. |

---

## Hard out-of-scope items (prevents cycle drift)

1. **No OpenTelemetry wiring** — observability is Cycle-7+ per polyglot blueprint §5; adding it before persistence lands would produce traces of ephemeral work.
2. **No Docker consolidation** — multi-stage single-image consolidation is post-gateway-decision work.
3. **No LLM inference integration** — that's Cycle-7 arc per Manifest v1 §10.
4. **No federation / multi-node work** — per Manifest §12, federation is its own doctrinal track.
5. **No new FTAP development** — explicitly out-of-scope per FTAP seed and manifesto v1 non-goal #6.
6. **No rewrite of runtime/** — the second Rust workspace gets read and understood in this cycle, not rewritten.

---

## Daughter Test for Cycle-6

Can `أبوك وأمك` understand what this cycle accomplishes in 5 seconds?

**Candidate answer:** *"جعلنا السيستم يتذكر كل شئ صنعه، ومفيش اكتر من باب واحد للدخول، ومفيش اكتر من وجه واحد للبيت."*
(*We made the system remember everything it did; there is not more than one entrance door; there is not more than one face to the house.*)

✅ **YES** — understandable in 5 seconds.

---

## Constitutional filter (required for any cycle opening)

Every Cycle-6 gate must preserve the five invariants per Manifest v0.2 §3:

| Invariant | How Cycle-6 upholds it |
|---|---|
| **ZANN_ZERO** | Persistence layer must still require evidence binding for every receipt read/write |
| **CLAIM_MUST_BIND** | Unified chain must preserve the hash-binding property across Rust↔Python boundary |
| **RIBA_ZERO** | Gateway/frontend consolidation must not introduce extractive patterns |
| **NO_SHADOW_STATE** | This is the PRIMARY motivator — two parallel gateways + two parallel frontends are structural NO_SHADOW_STATE violations. Cycle-6 exists to close them. |
| **IHSAN_FLOOR** | 0.95 floor remains enforced at the kernel layer (`IhsanFloorGate`); unification must not bypass it |

---

## What Cycle-6 does NOT claim

- Does not claim to replace the founder-level architectural decisions with algorithmic ones — G2 and G3 end in ADRs filed with founder authority, not in code removal
- Does not claim to land contract-first codegen (explicit §deferred)
- Does not claim to ship new user-visible features — Cycle-6 is a consolidation cycle, value accrues in fewer surprises, not more surface area
- Does not claim to unify the 5 constitution variants (`bizra-constitution/`, `bizra_constitution/`, `bizra-constitution-v5/`, `bizra-node0-v6/`, `bizra-genesis-engine-v5/`) — those are historical lineage, not current parallel systems; their unification is not a blocker

---

## References

- Cycle-5 retrospective (the predecessor cycle): `cycle-5/retrospective.md`
- Repo inventory that surfaced the three findings: `docs/BIZRA-Repo-Inventory-v1.md`
- Manifesto v1 (product law): `docs/dema-cli-manifesto-v1.md`
- Trust Compiler Thesis (category thesis): `docs/bizra-trust-compiler-thesis.md`
- FTAP seed (explicitly deferred target): `docs/ftap-function-registry-rfc-seed.md`
- Handover v1 (onboarding canon): `docs/BIZRA-Handover-v1.md`
- Rollback runbook (if Cycle-6 needs to back out): `docs/ROLLBACK-RUNBOOK-Cycle-5.md`
- CI policy audit (verifies new E2E workflow integrates cleanly): `docs/CI-POLICY-AUDIT-v1.md`

---

## Signature

Founder charter: Mumo (Muhammad Beshr) — session log 2026-04-17 13:26 GST
Cycle chain position: 6 (following Cycle-5 commit `bb230fd9` body, most recent origin head at open: `6ae2c664`)
Niyyah status: **DECLARED** — work may begin; any commit claiming to be Cycle-6 scope must reference this niyyah and preserve the five invariants above.

Close it. Prove it. Reveal it.

الحمد لله.
