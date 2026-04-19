# BIZRA Ecosystem Manifest — North Star Canon v1 (DRAFT)

بسم الله الرحمن الرحيم

**Status:** DRAFT (Cycle-8, 2026-04-19). Audience-neutral. Truth-labeled. Pre-fire.
**Authority:** Layer A inherits from Manifest v0.2 and Cycle-7 retrospective evidence. Layer B is ecosystem horizon, labeled.
**Scope:** this file holds constitutional doctrine; launch mechanics live in `FIRST-FIRE-DOCTRINE-v1.md`.

---

## Truth-label ladder used in this document

Every ecosystem claim in Layer B MUST carry exactly one of these labels:

- **CLAIM** — declared, no evidence yet
- **TESTED** — evidence under specific conditions, not fully proven
- **PROVEN** — evidence binds receipt-chain-level + reproducible + compile/test gates green
- **CANONICAL** — PROVEN + hashed + chained + documented + passes the Daughter Test

Layer A is all CANONICAL at this writing (Cycle-7 closed with machine-verifiable evidence). Layer B is a mix of TESTED / PROVEN / CLAIM as labeled per item.

---

## 1. Opening Declaration

BIZRA exists to restore lawful intelligence, proof-bearing action, and human sovereignty in a world ruled by opaque systems, extractive incentives, and simulated truth.

This is not a marketing claim. It is a constitutional mission.

## 2. Why BIZRA Exists

The human problems BIZRA addresses, named precisely:

- centralization of compute and decision authority
- truth drift in black-box models (hallucination)
- extractive economic patterns (riba)
- fragmented agency (no operator sovereignty)
- proofless software (claims without evidence)
- fragmented intelligence with no unifying face
- the loss of the human's ability to audit what is done for them

## 3. Canonical Thesis

BIZRA is not a chatbot.
BIZRA is not a conversational assistant.
BIZRA is not a generated-text product.
BIZRA is not an agent framework in the typical sense.

BIZRA is **an architecture for governed mission execution**, producing cryptographically traceable receipts, replayable evidence, trustable state surfaces, and human-centered assistance through one coherent interface.

Its product is **trust itself, compiled lawfully**.

## 4. Governing Mandate

Do not scale BIZRA yet.

**Close it. Prove it. Reveal it.**

Node0 outranks every broader federation, economy, or product claim until truth is proven locally first. Cycle-7 (Principal Activation Law) has sealed this mandate at the code-and-receipt level. Cycle-8 advances packaging and public readability without weakening it.

## 5. The Constitution — Five Invariants (§3)

All five invariants form a **closed system**: weakening any one breaks all. No invariant is independently waivable.

1. **IHSAN_FLOOR** — Excellence is the minimum, not the maximum. Quality score ≥ 0.95 for any Permit verdict.
2. **ZANN_ZERO** — No assumption. No claim without evidence.
3. **RIBA_ZERO** — No extractive economic pattern. Value exchange must be symmetrical.
4. **CLAIM_MUST_BIND** — Every claim cryptographically binds to hash-addressed evidence. The binding is the proof.
5. **NO_SHADOW_STATE** — Visible state ≡ kernel state. The face never simulates truth the kernel has not sealed.

**Enforcement site:** `bizra-omega/bizra-cognition/src/admissibility_freeze_v1.rs` — the AdmissibilityChain evaluates every claim against these five gates in fail-closed order. Any Reject stops the chain immediately; no partial pass.

## 6. The Lawful Runtime — One Loop, No Bypass

Every consequential act in BIZRA traverses exactly one authoritative path:

**Intent → Mission → Claim → Admissibility → Execution → Receipt → Canonicalization → Replayability**

No bypasses. No side channels. No UI-only mutation. Any act that does not complete this lawful loop is **non-canonical** — the chain reflects its absence, not its presence.

**Enforcement:** `bizra-omega/bizra-cognition/src/runtime.rs::CognitionRuntime::submit_mission` is the only path through which a mission can land a receipt on the chain. Witness observations (Cycle-8 Days 4-5) are Ed25519-signed and subject to the same NO_SHADOW_STATE rule: what the store echoes is exactly what arrived signed, never a fabrication.

## 7. The Surface Doctrine — Dema Is the One Face

The operator speaks to one assistant. The operator does not manage the swarm.

**Dema** reveals outcomes, receipts, trust surfaces, and state. It never simulates truth. It never invents law. It never exposes hidden routing as a cognitive burden on the operator.

PAT-7 / SAT-5 / FATE / URP remain hidden per Manifest v0.2 §8. They are the substrate; Dema is the face.

Current surface options (audience decision pending per U1/U2/U3):
- `dema` CLI — shipped, 7 subcommands
- `dema-console` (Next.js, PR #28 Option A) — 1/7 wired for `organize`
- `award-winner-design /dema` — 3/7 wired
- **DEMA Desktop Overlay** — future Cognitive IDE; Horizon (not T=0)

## 8. Node0 Achievement (CANONICAL through Cycle-7, pre-fire through Cycle-8)

CANONICAL at main HEAD `b8bd9eb7` (Spearpoint A):
- Principal activation is sealed (Cycle-7 G2, commit `1d3c540f`).
- Lawful dual-agent binding is sealed (PAT-7 / SAT-5 activation receipt chain).
- Proof and face are connected (gateway + dema CLI live-walked).
- Dema Console is an active face track (PR #28 Option A, `4c67710a`, WIRED_REAL for organize).
- 6 sovereign_state cache files operationalized (principal / receipt_history / manifest_history / mission_log / state_snapshots / resource_registry).
- 4 Proof-of-Impact ledger surfaces (Cycle-7 G6).
- All 5 admissibility gates PROVEN across 309 cognition tests + 77 gateway tests.

Independently validated by arXiv:2510.13857v1 (Xu et al., CUHK, 2025-10-12), which theorizes, 6 months after Node0 was already building it, the same **Kernel-as-Governor** / **Agent Constitution Framework** / **Evaluation-Driven Development Lifecycle** architecture BIZRA implements in Rust. The paper is evidence of independent academic convergence, not source material.

## 9. The Hidden Organism (Layer A + Layer B)

Layer A (CANONICAL — operating beneath Dema):
- **PAT-7** — Primary Agent Threads; executive layer handling goal setting and intent decomposition.
- **SAT-5** — Secondary Agent Threads; specialized workers executing discrete technical tasks.
- **FATE** — invariant gate chain; the constitutional filter between PAT intent and SAT execution.
- **URP** — Universal Resource Projection; the sovereign substrate for local resource allocation.

Layer B additions (horizon components; not yet part of the canonical kernel):
- **HAL (Hardware Abstraction Layer)** — **CLAIM** per ADK v0.2.2 blueprint; scheduled for v0.4.
- **LLM probabilistic-CPU wiring** — **CLAIM** per HANDOVER §10; Cycle-8 or later.
- **Witness-node gossip** — **TESTED** per Cycle-8 Days 4-5 (code complete on cycle-8 branch, not yet deployed at two-node scale).

## 10. The Ecosystem Horizon (Layer B — truth-labeled)

Only items present in code or verifiable canon may sit on this list. All others are external hypothesis until labeled.

| Item | Truth label | Source |
|---|---|---|
| Witness-grade finality (witness ping + daemon + Ed25519 sig) | **TESTED** | Cycle-8 Days 4-5 code on cycle-8/seal-primitive-days-1-2 |
| Proof-of-priority signed manifest | **TESTED** | `scripts/generate-proof-of-priority.sh` unsigned; signing is next step |
| cargo-dist packaging | **TESTED** | `[workspace.metadata.dist]` config written; `cargo dist check` not yet run |
| HAL (Hardware Abstraction Layer) | **CLAIM** | ADK v0.2.2 blueprint, v0.4 roadmap |
| LLM probabilistic-CPU wiring | **CLAIM** | HANDOVER §10 known gap |
| YAML declarative policies | **CLAIM** | ArbiterOS paper §4.1 pattern; BIZRA currently uses Rust-coded policies |
| Desktop overlay / Cognitive IDE | **CLAIM** | Autopoietic Loop 2026-04-17 niyyah; ArbiterOS §8.8 spec |
| Node-to-node federation | **CLAIM** | Manifest v0.2 §12 long-range |
| Bonded-stake / slashing / DAO / challenge-period economics | **CLAIM** | Explicitly Horizon per Cycle-8 doctrinal constraint (2026-04-19) |
| Native dual-token representation (SEED/BLOOM) | **CLAIM** | Referenced in dema-main/README; constitutional grounding pending |

## 11. Truth Labels and Claim Discipline

**Ladder:** `CLAIM → TESTED → PROVEN → CANONICAL`

- **CLAIM** — stated intent; no evidence.
- **TESTED** — evidence exists (tests, live-walks, local builds); not yet chain-sealed or community-verified.
- **PROVEN** — tests green + reproducible + receipt-chain-sealed + compile/clippy gates green under `-D warnings`.
- **CANONICAL** — PROVEN + hashed + chained + documented in an authoritative canon + passes the Daughter Test ("would Mumo's daughter be proud this is what we shipped?").

**Enforcement:** No ecosystem capability may be CANONICAL without evidence binding. This doc labels every Layer B item; any promotion requires replacing the label in-place with evidence cited.

**Explicit anti-hype discipline:** Layer B items do not appear in public marketing material until they reach PROVEN. Public materials MAY reference CLAIM-level items as "horizon" only.

## 12. The Golden Standard (Four-Modality Convergence)

BIZRA's operational definition of truth requires four independent proof modalities to converge:

1. **Formal verification** — mathematical proof (currently TESTED via cargo test; full Isabelle/HOL-grade proof is Horizon).
2. **Cryptographic commitment** — BLAKE3 hash-chain of receipts + Ed25519 signatures on witness observations (CANONICAL at main).
3. **Empirical reproducibility** — `cargo test` green on any machine, `dema organize` receipt reproducibility (CANONICAL).
4. **Economic finality** — witness-grade detectability at T=0 (TESTED); bonded stake / slashing / DAO / challenge-period economics are Horizon / Layer B.

**Doctrinal constraint (Cycle-8, 2026-04-19):** at T=0 (first fire), economic finality means witness-grade detectability and bounded cost-to-fake increase only. Full cryptoeconomic enforcement waits on Layer B maturity.

**Falsifiability standard:** a skeptical stranger with no prior trust must be able, using only public information and standard tooling, to verify in bounded time that a BIZRA claim is true OR produce transferable evidence that it is false.

## 13. Final Declaration

BIZRA is not permitted to expand through ambiguity.

It will grow only where law is bounded, proof is receipted, truth is replayable, the face reveals reality without simulation, and every broader claim can survive constitutional scrutiny.

---

## One-sentence North Star

**BIZRA is a sovereign, constitution-bound intelligence ecosystem that turns human intent into lawful, receipted, replayable, and eventually federated action through one visible face and one source of truth.**

---

## Writing-law for future refinements

The tone of this document must be: **constitutional prophecy under proof discipline**.

Not: startup brochure. Not: investor fantasy. Not: README. Not: spiritual diary alone.

It must sound like a system born from ordeal, hardened into law, bounded by proof, disciplined against hype.

---

*Close it. Prove it. Reveal it.*

الحمد لله
