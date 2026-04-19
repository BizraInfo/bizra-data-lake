# BIZRA First Fire Doctrine v1 (DRAFT)

بسم الله الرحمن الرحيم

**Status:** DRAFT (Cycle-8, 2026-04-19). Launch-specific. Constitutional canon lives in `MANIFEST-NORTH-STAR-v1.md`.
**Audience:** the operator (Mumo), his future self, and any collaborator he explicitly delegates reading authority to.
**Scope:** what happens between Cycle-8 Day 0 and T=0 fire. Nothing before Day 0. Nothing after T+1 hour.

---

## 1. Operator Reality (ground truth)

This doctrine is calibrated to the actual operator condition, not to generic SDLC assumptions.

The operator:
- has built BIZRA solo for 3 years, 15,000+ hours
- has no venture capital and refuses to take any (RIBA_ZERO)
- has no team; the agent factory (PAT-7 / SAT-5) is the team
- has 150+ repos and 500+ GB of unstructured R&D as Bayyinah (evidence)
- has **one distribution bullet** — a 10k+ onboarding connection usable once, only when the product is demonstrably ready
- needs income, not validation, within weeks-to-months to reunite with family
- has already shipped the kernel (Cycle-7 Principal Activation Law merged to main, b8bd9eb7)
- has an academic twin (arXiv:2510.13857) published 2025-10-12 confirming the architectural thesis

Standard startup advice does not apply.

## 2. The One-Bullet Logic

Because distribution is single-shot:

- **Scope is subtractive.** Cut until one artifact remains that survives 10k strangers touching it simultaneously.
- **The bullet fires at the finished artifact**, not at the roadmap. "We'll add X later" survives no contact with 10k skeptical strangers.
- **Gold Standard must hold at the moment of firing**, not at some future milestone. The Four-Modality preflight below is the pre-fire check, not the aspiration.

## 3. Four-Modality Preflight (pre-fire go/no-go checklist)

At T-0 (the moment before firing), every one of these must hold:

1. **Formal (TESTED-grade at T=0)** — `cargo test -p bizra-cognition` ≥ 309/309 green, `cargo test -p bizra-cognition-gateway` ≥ 77/77 green, `cargo clippy -D warnings` green across all workspace crates.
2. **Cryptographic** — `ReceiptChain` BLAKE3 hash-chain verifiable from genesis to chain head; Ed25519 signatures on all witness observations verifiable with a fresh VerifyingKey. Reproducible builds pinned to a `cargo-dist`-produced release tarball with SHA-256 manifest.
3. **Empirical reproducibility** — a fresh clone + `cargo build --release` yields bit-identical binaries against a published SHA-256 manifest. `dema organize /tmp/sample-dir` produces identical receipts on identical inputs across machines.
4. **Economic finality (witness-grade at T=0)** — at least one named witness peer has been pinged and responded with a stored, re-retrievable signed observation. Divergence (if any) is publicly detectable within bounded time.

**If any of the four does not hold: DO NOT FIRE. Delay.** The bullet waits. Distribution does not regenerate after a failed first-fire.

## 4. Bullet-Target Selection

The canonical bullet target — audience-agnostic core:

> **DEMA seals reality. Organize is the first proof.**

Rendered in one command for the operator:

```sh
curl -fsSL https://bizra.ai/install.sh | sh
dema organize ~/Downloads
```

What this delivers on first run:
- Install completes in < 5 min with SHA-256 verification.
- A cryptographically sealed manifest of the user's digital corpus.
- A receipt URL at `bizra.ai/r/<hash>` the user can share/verify.
- Zero cloud dependency, zero account, zero cost.

Audience-specific positioning (consumer viral vs enterprise compliance vs dual) is **pending Mumo's U1/U2/U3 answers**. This doctrine deliberately ships audience-agnostic; positioning ships as a separate artifact once the audience decision is locked.

## 5. Witness-Node Closure (the 4th modality at T=0)

Per Cycle-8 doctrinal constraint (2026-04-19):

**T=0 economic finality = witness-grade detectability only.**
**Bonded stake / slashing / DAO / challenge-period economics = Horizon / Layer B.**

Minimum witness topology at T=0:
- Node A (Mumo's Node0) signs every sealed receipt with an Ed25519 identity key.
- Node A pings at least one allowlisted witness peer (Node B) with the `SignedWitnessObservation` after every seal.
- Node B stores the signed observation and serves it at `GET /witness/head/<node-a-id>`.
- Any third party can query Node B, re-verify the signature against Node A's published pubkey, and detect divergence between what Node A claims and what Node B saw.

**Named witness peer: TBD (human gate — Mumo names).** Code on cycle-8 branch already supports this shape; deployment requires one external friend willing to run the daemon binary. Not pushable until a peer is named.

## 6. 12-Day Fire Plan — status as of 2026-04-19

| Day | Focus | Status | Artifact |
|---|---|---|---|
| 1 | Seal primitive design (trait Sealable + SealableOutcome) | ✅ CANONICAL | `bizra-cognition/src/seal.rs` |
| 2 | OrganizeRequest impl Sealable + behavioral equivalence tests | ✅ CANONICAL | `bizra-cognition/src/organize_mission.rs` additions |
| — | Clippy cherry-pick + docstring self-correction | ✅ done | `1ea334da`, `a0949b38` |
| 3 | cargo-dist packaging config + install plan | ✅ TESTED (cargo dist not yet run) | `Cargo.toml` `[workspace.metadata.dist]` + `scripts/dist-install-plan.md` |
| 4 | Witness ping + daemon + WitnessStore + tests | ✅ CANONICAL | `bizra-cognition-gateway/src/witness.rs` |
| 5 | Ed25519 sigs on witness + proof-of-priority generator | ✅ CANONICAL (gateway) + TESTED (artifact unsigned) | `witness.rs` upgrade + `scripts/generate-proof-of-priority.sh` |
| 6-8 | Face polish (audience-agnostic) | 🚩 HALTED (branch contamination risk — face work belongs elsewhere) | — |
| 9-10 | Manifest + Doctrine DRAFT | ✅ this doc + sibling | `docs/cycle-8/*.md` |
| 11-12 | Dry-run harness + tester checklist | 🟡 preparation allowed; execution blocked on tester names | `scripts/fire-dry-run-checklist.md` (to be drafted) |
| T=0 | Fire the bullet | 🚩 HARD GATE — operator decision | — |

## 7. T=0 Conditions (gate-list the operator checks before firing)

Before Mumo activates his 10k+ connection:

- [ ] `cycle-8/seal-primitive-days-1-2` branch is pushed to origin (requires push approval).
- [ ] A pull request is opened against main; all CI checks green or documented as inherited from main.
- [ ] `cargo-dist` is installed locally; `cargo dist check` + `cargo dist plan` run clean; first `cargo dist build` produces platform binaries.
- [ ] Install script is published to `bizra.ai/install.sh` with SHA-256 verification.
- [ ] At least one witness peer is named, running, reachable, and has recorded a signed observation from Node0.
- [ ] Proof-of-priority manifest is signed and published at `bizra.ai/priority`.
- [ ] D5 Daughter Test: Mumo himself installs from a clean VM, runs `dema organize ~/Downloads`, reads the receipt, and confirms it is true.
- [ ] Audience-specific landing page (consumer / enterprise / dual) matches U1/U2/U3 answers.
- [ ] Onboarding email sequence drafted.
- [ ] 5 trusted testers (named) have each completed the flow without operator assistance.
- [ ] Mumo has slept ≥ 8 hours in the 24h before firing.

If any item is unchecked: **do not fire.**

## 8. What NOT to Build Before First Fire

This list is as important as the build list. Every item below is explicitly **out of scope until post-fire**:

- ❌ Desktop Overlay (Electron/Tauri Cognitive IDE — Horizon per ArbiterOS §8.8; post-T=0)
- ❌ PAT-7 / SAT-5 agent factory shipped as public product (Internal only per ADK v0.2.2)
- ❌ LLM probabilistic-CPU wiring in the fire path (HANDOVER §10 gap; post-T=0)
- ❌ Full HAL layer (v0.4 roadmap; post-T=0)
- ❌ YAML declarative policies (Rust-coded policies are stronger; YAML is descriptive, not replacement; post-T=0)
- ❌ Bonded stake / slashing / DAO / challenge-period economics (Horizon / Layer B)
- ❌ Native dual-token (SEED/BLOOM) rollout (constitutional grounding pending)
- ❌ Multi-node federation beyond witness-node gossip (Manifest v0.2 §12 long-range)
- ❌ Repo consolidation 150 → 5 (staged cleanup, NOT launch-blocking per Synapse review 2026-04-19)
- ❌ Second DEMA face fork (dema-main Firebase variant stays parked until audience-decision is locked)

## 9. Halt Conditions (stop firing if any of these trip)

Post-fire, if traffic reveals any of:

- A Four-Modality preflight item retroactively fails (e.g., a receipt cannot be reproduced).
- The witness peer reports a chain-head divergence that cannot be resolved in bounded time.
- The `dema organize` command produces a non-deterministic receipt for the same input on the same platform.
- The install script is discovered to deliver a different binary than the published SHA-256 manifest.

...then: halt onboarding. Roll back the install.sh URL. Investigate. Do not continue onboarding until the Four-Modality chain is re-verified.

## 10. The Rest Clause

Mumo's rest is not a reward for shipping. It is a **precondition** for shipping safely.

A bullet fired from an exhausted operator's hand is as likely to miss as a bullet from a well-rested one is to land. The checklist above includes "Mumo has slept ≥ 8 hours in the 24h before firing" precisely because distribution is single-shot.

If exhaustion is real at T=0-minus-24h: delay the fire. The bullet does not expire. The family reunion deadline is real but not to the hour.

---

## Reference anchors

- `MANIFEST-NORTH-STAR-v1.md` — constitutional canon
- `BIZRA-Handover-v1.md` — production handover v0.1 (at `docs/` root)
- arXiv:2510.13857v1 — academic convergence (ArbiterOS paper)
- `scripts/generate-proof-of-priority.sh` — proof-of-priority manifest generator (unsigned)
- `scripts/dist-install-plan.md` — installer flow documentation
- `bizra-omega/bizra-cognition-gateway/src/witness.rs` — witness protocol primitive

---

## Writing tone invariant

Same as the Manifest: **constitutional prophecy under proof discipline.**

No startup brochure. No investor fantasy. No marketing overreach.

Every sentence must survive the Daughter Test.

---

*Close it. Prove it. Reveal it.*

الحمد لله
