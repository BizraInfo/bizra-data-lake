# Golden Gems Register — BIZRA v0.1

**Definition:** High-value architectural + philosophical insights worth protecting and advertising (internally). Each is an **insight**, not a metric claim.

---

## 1. Node0 as archetype, not authority server

**Why it's gold:** nearly every "sovereign AI" narrative in the market reduces to a rebranded cloud service. BIZRA's Node0 is a per-human sovereign identity bound to a genesis seal. No "BIZRA server" adjudicates between nodes. URP is the shared substrate, not the authority.

**Risk if lost:** a single convenience hack ("just call our server") destroys the category position permanently.

**Protect with:** architecture invariant tests, refusal patterns in consumer copy (no "our servers" language), federation-by-default.

**Evidence:** `bizra-omega/bizra-core/src/genesis_seal.rs`, `bizra-omega/bizra-node/src/`, memory `project_node0_sovereign_origin_sealed.md`.

## 2. URP as shared constitutional / world layer

**Why it's gold:** URP (Universal Resource Pool) is the rare thing a peer-to-peer sovereign system needs — a shared substrate that is *constitutional* (not transactional). Reconciliation is a first-class state (`AwaitingReconciliation → UrpValidating → Complete`), which is how offline-capable systems must work.

**Risk if lost:** without URP, "sovereign nodes" reduces to isolated islands with no honest inter-node semantics.

**Evidence:** `bizra-omega/bizra-mission/`, offline-reconciliation state in the mission state machine.

## 3. DEMA as single visible face

**Why it's gold:** user-visible complexity is the fastest way to kill adoption. DEMA is the *only* consumer surface; PAT / SAT internal teams never leak to users. The brand canon encodes this discipline; the audit engine's claim scanner enforces it.

**Risk if lost:** once PAT/SAT leak into consumer copy, BIZRA becomes "just another tech stack."

**Evidence:** brand canon v0.2 §10 Dema voice, `docs/brand/public_launch_readiness/PUBLIC_CLAIMS_REGISTER.md` internal-vs-external rules, `memory/project_pat_sat_canonical_topology.md`.

## 4. PAT / SAT topology — internal, canonical, bounded

**Why it's gold:** naming internal teams with a canonical topology (`topology_canon.rs`) turns "how agents cooperate" from a soft practice into a hard invariant. The discipline "PAT × 7, SAT × 5, mint functions at `bizra-resourcepool/src/genesis.rs`" makes drift visible and testable.

**Evidence:** `bizra-omega/bizra-core/src/topology_canon.rs`, `bizra-omega/bizra-resourcepool/src/genesis.rs`.

## 5. Receipt-native action loop

**Why it's gold:** every visible effect emits a BLAKE3-chained, Ed25519-signed receipt. This is the physical embodiment of "proof before claim" — it's not a policy, it's a data structure.

**Risk if lost:** collapses to "most of our actions have receipts" — which is useless. The invariant has to be total.

**Protect with:** hot-path `.unwrap()` audit (so panics don't bypass receipt emission), `advance!` macro kept fail-closed, PR #50 full-body signature landed.

**Evidence:** `bizra-omega/bizra-core/src/canonical_receipt.rs`, mission-state-machine `advance!` macro, PR #50.

## 6. Law of Assumption

**Why it's gold:** it's a *doctrine* with operational teeth. The audit engine instantiates it: exact metric claims that don't have receipts get downgraded. The brand canon §15 codifies it. The Cognitive Foundry canon-pack discipline extends it (pack is "candidate for" canon, never canon).

**Risk if lost:** without this, the rest of the discipline is posture. With this, every other gem is operational.

**Evidence:** `docs/brand/public_launch_media_kit_v0_1/extracted/bizra_public_launch_media_kit_v0_1/docs/CLAIM_DISCIPLINE.md`, brand canon §5, this audit engine.

## 7. Public claim discipline A/B/C/D/E with receipts

**Why it's gold:** most AI companies don't have a claim register. BIZRA has one, in the repo, in markdown, and it classifies every public-facing numeric claim with a rewrite guidance. This is rare and valuable.

**Evidence:** `docs/brand/public_launch_readiness/PUBLIC_CLAIMS_REGISTER.md`, `CLAIM_SAFE_LAUNCH_COPY.md`, this audit's `WEBSITE_PUBLIC_CLAIMS_AUDIT.md`.

## 8. Local LLMs as AI archaeologists (Cognitive Foundry doctrine)

**Why it's gold:** the Cognitive Foundry treats local LLMs as *archaeologists of the operator's own cognition*. A 91 MB Claude export → 359-row review workbook → 27-entry preferred canon pack — all stdlib, all deterministic, all human-gated. The insight is that canon grows from operator reasoning, not from trained model outputs.

**Risk if lost:** canon becomes "another thing we stuff with LLM outputs." Current discipline keeps human-in-loop at every promotion.

**Evidence:** `tools/cognitive_foundry/claude_lane/`, `REVIEW_HANDOFF.md`, 27-entry preferred pack.

## 9. HHMM / cache-TTL research lane (hidden-state + diffusion-based insight surfacing)

**Why it's gold:** this audit uses a 4-level hidden-state taxonomy (HHMM) and a "diffusion" approach to surfacing golden gems. These are research-grade ideas embedded in operational tooling.

**Evidence:** `HHMM_HIDDEN_STATE_TAXONOMY.md`, `GOLDEN_GEMS_REGISTER.md` (this file), `snr_classifier.py`.

## 10. Canon Store Ingestion Gate as a required boundary

**Why it's gold:** the single highest-leverage architectural decision is **not having auto-ingestion**. The Cognitive Foundry produces candidate-for-canon packs; a separate, human-gated tool ingests. That tool does not yet exist, and that is the point.

**Risk if lost:** as soon as someone "just writes a small script" to move a pack into MEMORY.md without the gate, canon discipline evaporates.

**Protect with:** pre-register gate location (e.g., `tools/canon_store/ingestion_gate.py`); CODEOWNERS on MEMORY.md / constants.py / topology_canon.rs; CI guard.

**Evidence:** `tools/cognitive_foundry/claude_lane/canon_packs/README.md`, `REVIEW_HANDOFF.md` §6.

## 11. Content-hash / issuance-hash split (v0.2.0)

**Why it's gold:** a rare determinism property — the same reviewed content always hashes the same (`content_hash_blake2b_32`), while each promotion event gets its own `issuance_hash_blake2b_32`. Fact + ceremony separated. This is the kind of detail that shows the system was built by someone thinking about what identity *means*.

**Evidence:** `tools/cognitive_foundry/claude_lane/promote.py` (v0.2.0 split-hash), preferred pack manifest, `canon_packs/README.md` §2.

## 12. Sovereignty stack: OS + agent + face + URP + canon — as one vertical

**Why it's gold:** most companies do one layer. BIZRA is attempting the whole vertical: substrate discovery, agent kernel, receipt chain, visible face, federation plane, doctrine plane. The vertical is what makes "sovereign" mean something.

**Risk if lost:** reducing to one layer loses category position.

---

## How to use this register

- **Internal use:** reference these when a product or architecture decision threatens one. "That shortcut breaks Gem #1" is a real conversation.
- **External use:** these inform *framing*, not quantitative claims. A hero line can gesture at one of these; a ad cannot quote them as metrics.
- **Maintenance:** add a gem if a newly surfaced architectural invariant deserves protection. Remove a gem if it's been diluted.
