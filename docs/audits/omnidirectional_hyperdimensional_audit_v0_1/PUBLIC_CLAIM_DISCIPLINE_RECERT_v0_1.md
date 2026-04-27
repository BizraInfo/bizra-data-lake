# Public-Claim Discipline Recertification v0.1

**Date:** 2026-04-25 (GST) — Dubai
**Scope:** Audit-side document only. No website source edited. No PR/runtime/CI changes.
**Source of truth:** `docs/audits/omnidirectional_hyperdimensional_audit_v0_1/artifacts/{claims_register,findings,website_claims,audit_summary}.json` from latest no-network audit run.
**Status lock:** WAIT preserved. Phase 2 (PR #49 + #50) and Phase 3 (Claim Registry) remain blocked.

---

## 1. Recertified counts

From `audit_summary.json` (no-network run, `duration_seconds: 3.41`):

| Class | Count |
|---|---:|
| **PROHIBITED** | 20 |
| **NEEDS_REWRITE** | 94 |
| **PROOF_REQUIRED** | 367 |
| **BRAND_SAFE** | 19 |
| Total scanned | 500 |

PROHIBITED breakdown by category:

| Category | Count |
|---|---:|
| AGI claim | 14 |
| Unsubstantiated certification | 5 |
| Cryptographic-finality claim | 1 |

NEEDS_REWRITE breakdown:

| Category | Count |
|---|---:|
| Production-readiness implication | 75 |
| Exact SNR number | 12 |
| Explicit cost figure | 7 |

---

## 2. Highest-risk surface — bizra.ai SPA shell

Per `website_claims.json`, bizra.ai's non-JS shell currently contains this single block (operator pre-check capture):

> *"BIZRA | The Sovereign Future. local agents / no cloud dependency / no telemetry. Ed25519 receipt signatures. cost per action dropping from about $0.10 toward $0.008. SNR 0.974. 8,072 verified tests. 100% pass rate. Ihsan Gate >= 0.95. 73 of 100 nodes remaining."*

This single string carries **8 distinct claim-class hits**, including PROHIBITED, NEEDS_REWRITE, and PROOF_REQUIRED categories. It is both the smallest surface to fix and the highest-leverage move to make in the public-claim recert lane.

The previously-named C4/C5/C7/C9 (per `EXECUTIVE_SUMMARY.md`) all live inside this single string.

### 2.1 Per-substring claim-action table

| # | Substring | Class | Category | Risk | Recommended action |
|---|---|---|---|---|---|
| W1 | "The Sovereign Future" | NEEDS_REWRITE | Movement line | Aspirational, not falsifiable. Acceptable as poetry; risky if read as roadmap. | **Soften** to "A path toward sovereign personal compute." |
| W2 | "local agents" | BRAND_SAFE | Local-only descriptor | Architectural truth, low risk. | **Keep**. |
| W3 | "no cloud dependency" | PROOF_REQUIRED | Local-only / no-cloud claim | Plausible at architecture level; user must opt into cloud explicitly. Without published ops attestation it remains directional. | **Soften** to "designed to run without cloud dependencies for the local Node0 path." |
| W4 | "no telemetry" | PROOF_REQUIRED | Zero-telemetry claim | Strong absolute. Any observability shipping at any tier breaks it publicly. | **Soften** to "no telemetry on by default; opt-in observability is documented per surface." |
| W5 | "Ed25519 receipt signatures" | PROOF_REQUIRED | Cryptography claim | Architecturally true (`core/proof_engine/canonical_receipt_adapter.py`). External readers cannot verify without a receipt-link. | **Receipt-link** to a public sample receipt (BLAKE3 hash + Ed25519 signature) and a CLI verifier. |
| W6 | "cost per action dropping from about $0.10 toward $0.008" | NEEDS_REWRITE | Explicit cost figure | Specific numerics promise economics that aren't measurably reproducible. | **Remove** from public copy. Move to under-the-hood / cost-model page only when an instrumented receipt is published. |
| W7 | "SNR 0.974" | NEEDS_REWRITE | Exact SNR number | Exact decimal implies a measurement that has not been continuously instrumented. | **Remove** from public copy. Replace with directional language; only show numbers when a published benchmark receipt backs them. |
| W8 | "8,072 verified tests. 100% pass rate." | NEEDS_REWRITE / borderline PROHIBITED | Production-readiness implication | "100% pass rate" is a proven-fragile public claim — any single test flake on the next CI run makes the public copy false. | **Remove** "100% pass rate"; replace count with "Test suite: instrumented; latest pass-rate posted on the build dashboard". |
| W9 | "Ihsan Gate >= 0.95" | PROOF_REQUIRED | Ihsan-threshold claim | Constitutional threshold is real (`core/integration/constants.py`); public reader cannot verify without a receipt. | **Receipt-link** to constitutional thresholds doc + a sample Ihsan-gate receipt. |
| W10 | "73 of 100 nodes remaining" | NEEDS_REWRITE | Production-readiness implication / scarcity | Scarcity copy implies a live counter that public has no reason to trust until the counter is content-addressable. | **Remove** scarcity numerics until a verifiable counter ships; OR reframe as "Genesis 100 is the launch cohort cap" without live count. |

### 2.2 Drafted claim-safe replacement (single block)

Replace bizra.ai shell with text that respects directional/evidence-bound language and keeps Ihsān-aligned framing:

> **Suggested v0.1 shell (no live numbers, no certification claims, no AGI/finality language):**
>
> *"BIZRA — a path toward sovereign personal compute.*
> *Designed for local-first agents. No telemetry on by default. Cryptographically receipted actions for every meaningful step.*
> *Genesis 100 launch cohort.*
> *Receipts and benchmarks are published as we run them — see /receipts and /bench."*

This shell:
- Uses **direction**, not certainty ("a path toward", "designed for")
- Sets an **evidence boundary** ("published as we run them")
- Emits **no AGI / first-world / finality / cryptographic-immutability** language
- Stays **Ihsān-aligned** (no overclaim; no scarcity manipulation; truth boundary explicit)
- Reduces 8 claim-class hits to 0 PROHIBITED, 0 NEEDS_REWRITE, 1-2 PROOF_REQUIRED (which the linked `/receipts` page would satisfy)

**This document does NOT push the new shell to bizra.ai or edit the site source.** Operator must consciously decide to land the new shell via the website repo (separate to this audit lane).

---

## 3. PROHIBITED claims — full register (20)

All 20 must be rewritten or removed before any new public-facing reuse of the source documents. The source documents are internal canon/strategy material; CodeQL / claim scanners flag them when they leak into ad copy, decks, or public site text.

| # | Claim ID | Source | Category | Action |
|---|---|---|---|---|
| 1 | C00036 | `docs/PROJECT_HANDOVER.md` | Unsubstantiated certification (ISO 27001 / Lyapunov) | **Soften**: "C4 diagrams + risk register present; ISO 27001 alignment is in design, not certified." |
| 2 | C00145 | `docs/THREAT-MODEL-V3.md` | Unsubstantiated certification (GDPR / SOC 2 / ISO 27001 PARTIAL) | **Mark "PARTIAL" as "in-design / not-certified"**. Preserve PARTIAL semantics; do not imply certification. |
| 3 | C00146 | `docs/THREAT-MODEL-V3.md` | Unsubstantiated certification (continued) | Same as #2. |
| 4 | C00339 | `docs/ENTERPRISE_IMPLEMENTATION_BLUEPRINT.md` | Unsubstantiated certification (CMMI / SOC 2 Type II target) | **Move to roadmap**: explicitly label as "target", not "current state". |
| 5 | C00345 | `docs/KERNEL.md` | "decentralized developmental AGI operating system" | **Remove "AGI"**: replace with "decentralized developmental cognitive operating system" or "cognitive runtime". |
| 6 | C00353 | `docs/BIZRA_IDENTITY_CANON.md` | "Distributed Decentralized AGI Operating System" | **Remove AGI** as in #5. |
| 7 | C00354 | `docs/BIZRA_IDENTITY_CANON.md` | "AGI" labeled to specific PAT/HHMM mechanics | **Rename label**: the underlying construct is a typed agent ensemble, not AGI. Keep mechanics, drop the AGI label. |
| 8 | C00374 | `docs/BIZRA_STRATEGY_DECK_2026.md` | "Distributed Decentralized AGI Operating System" header | **Remove AGI** from deck header; investor-grade decks must not lead with AGI claims. |
| 9 | C00375 | `docs/BIZRA_STRATEGY_DECK_2026.md` | "world's first Distributed Decentralized AGI Operating System" | **Remove "world's first" + "AGI"**. Both are PROHIBITED-class. |
| 10 | C00376 | `docs/BIZRA_STRATEGY_DECK_2026.md` | "Only Shariah-compliant AGI OS" | **Remove "AGI OS" + "Only"**. Reframe as "Shariah-aligned design constraints applied to the cognitive runtime". |
| 11 | C00377 | `docs/BIZRA_STRATEGY_DECK_2026.md` | "Zero Compliant AGI Platforms" | **Remove**. Cannot be substantiated. |
| 12 | C00378 | `docs/BIZRA_STRATEGY_DECK_2026.md` | "First Mover in Verified AGI" | **Remove "Verified AGI"**. Use "first-mover in receipt-native cognitive runtimes" if a market-position claim is needed at all. |
| 13 | C00379 | `docs/BIZRA_STRATEGY_DECK_2026.md` | "Hardcoded. Immutable. Unbreakable." | **Soften**: "Constitutional invariants enforced by FATE gate + Ed25519 receipt chain. Any breach is detectable and replayable." Removes finality language; restores fail-closed semantics. |
| 14 | C00401 | `docs/QUALITY_ASSURANCE_STRATEGY.md` | "annual external review for SOC 2 scope" | **Move to roadmap**: explicitly conditional on scoping/funding decision. |
| 15 | C00418 | `docs/DDAGI_CONSTITUTION_v1.1.0-FINAL.md` | DDAGI OS title | **Rename document title** to remove AGI. |
| 16 | C00419 | `docs/DDAGI_CONSTITUTION_v1.1.0-FINAL.md` | "operating system for human sovereignty in the age of AGI" | **Reword**: "operating system for human sovereignty in an era of accelerating cognitive automation". |
| 17 | C00420 | `docs/DDAGI_CONSTITUTION_v1.1.0-FINAL.md` | "artificial general intelligence is not a distant dream" | **Remove or footnote**: positioning, not architectural claim. |
| 18 | C00421 | `docs/DDAGI_CONSTITUTION_v1.1.0-FINAL.md` | "AGI as extraction tool" | **Reword**: "powerful general-purpose AI as extraction tool". |
| 19 | C00422 | `docs/DDAGI_CONSTITUTION_v1.1.0-FINAL.md` | "AGI as existential threat" | Same as #18. |
| 20 | C00423 | `docs/DDAGI_CONSTITUTION_v1.1.0-FINAL.md` | "ethical artificial general intelligence" | **Reword**: "ethical large-scale cognitive infrastructure". |

---

## 4. Recommended commit boundary

If operator authorizes a `chore/public-claim-recert` lane:

1. **Always-safe**: this audit-side document (`PUBLIC_CLAIM_DISCIPLINE_RECERT_v0_1.md`) — already lands in `docs/audits/.../`.
2. **Document-rewrite scope** (operator-decided per document): apply the action column above to each PROHIBITED entry's source document. Net effect: ~14 AGI mentions removed, 5 certification claims softened, 1 finality claim softened.
3. **Website shell** (separate repo, separate decision): the v0.1 shell text from §2.2 is provided as a paste-ready replacement. **NOT pushed in this lane.**
4. **Receipts-and-benchmarks publication path**: a `/receipts` and `/bench` route should be scoped before any of the PROOF_REQUIRED claims (Ed25519, Ihsan-threshold, etc.) become eligible to land back into the shell. That work is its own future sprint, NOT in scope here.

---

## 5. What this lane does NOT do

- Does NOT edit any document under `docs/` other than this new audit-side file.
- Does NOT push or update bizra.ai or bizra.info.
- Does NOT modify the source documents flagged in §3 — that's a separate operator-controlled cleanup pass.
- Does NOT touch runtime, core, src, CI, or dependencies.
- Does NOT change Phase 2 / Phase 3 / WAIT lock state.
- Does NOT change PR #49 / #50 status.
- Does NOT remove pre-existing 38+ dirty WIP files.

---

## 6. Validation

- No runtime tests run.
- No-network audit re-run NOT performed in this lane (would not change inputs; this doc reads existing artifacts only).
- Audit recertification document is itself an artifact; future audits will pick it up under `evidence_class: DOC` in the next no-network run, raising evidence count by 1 entry.

---

## 7. Memory anchor

This lane preserves:

- `feedback_audit_label_inflation_guard` — every row above respects directional vs. operationally-current label discipline.
- `feedback_secret_triage_redacted_only` — no secret material discussed.
- `feedback_third_party_eval_does_not_override_canon` — recommendations are operator-decision-gated; nothing prescribed runs without typed GO.
- `feedback_land_the_plane` — the single highest-leverage move (replace the 8-claim-hit shell with a 0-PROHIBITED v0.1 string) is named explicitly; the rest is documentation hygiene that can wait.

## 8. Next operator command (suggested)

After review, the cleanest scope-disciplined options are:

```
A. WAIT — accept this audit-side document as a reference; no source-doc rewrites yet.
B. GO — chore/public-claim-recert v0.1 (source-doc rewrites only, audit + 8 source docs touched, no website push).
C. GO — apply v0.1 shell text to bizra.ai (separate website repo; out of THIS audit-lane scope).
```

Default: **WAIT**. This document holds the analysis; operator decides when to act on it.
