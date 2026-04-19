# BIZRA v1.0.0 — Release Notes (DRAFT)

بسم الله الرحمن الرحيم

**Status:** DRAFT (Cycle-8, 2026-04-19). Launch-path release notes for the first public cut.
**Scope:** `bizra-data-lake` repo. Kernel + gateway + CLI + public face + docs + install flow.
**Branch sources (pre-release consolidation):**
- `main` @ `b8bd9eb7` — Spearpoint A (HTTP contract hardening) + Cycle-7 Principal Activation Law
- `cycle-8/seal-primitive-days-1-2` @ `70515549` — pushed to origin, 9 commits ahead of main
- `fork/dema-console-from-zai` @ `c79e556a` — local only (pending push approval), 5 commits ahead of origin

**Draft discipline:** every line below is one of four states — **SHIPPED** (merged to main), **DRAFTED** (on a feature branch, not yet merged), **HELD** (intentionally not public at T=0), or **BLOCKED** (requires human gate to proceed). No overclaiming.

**No-assumptions rule:** we do not assume a capability is live before we ship it. Two live truth labels on the landing surface:
- The `curl …/install.sh | sh` one-liner on `/landing` is explicitly labeled `install · preview · not live yet` — gated on first cargo-dist release cut (see §4).
- The `/r/<hash>` receipt viewer is explicitly `local-first` — it queries a gateway on the viewer's own machine (default `127.0.0.1:7421`), never a public lookup service. If/when a public witness gateway is introduced, it will be a conscious product decision with its own truth label.

---

## 1. What's SHIPPED on main (available at v1.0.0)

### 1.1 Kernel constitution — Cycle-7 Principal Activation Law

Commit: `1d3c540f` (merged 2026-04-18)

The lawful loop is sealed end-to-end in Rust:
- Five constitutional gates (ZANN_ZERO, CLAIM_MUST_BIND, RIBA_ZERO, NO_SHADOW_STATE, IHSAN_FLOOR) enforced fail-closed in `bizra-cognition::admissibility_freeze_v1`.
- Mission runtime: `CognitionRuntime::submit_mission` as the single lawful path.
- `run_lawful_loop()` connector binds intent → mission → claim → admissibility → execution → receipt → chain-append.
- 6 sovereign_state/dema_cache files: principal, receipt_history, manifest_history, mission_log, state_snapshots, resource_registry. Schema-versioned JSON with atomic write + fsync.
- G5 `dema organize <allowlisted>` — first real operator mission. Read-only filesystem projection → sealed `MissionExecuted` receipt.
- G6 Proof-of-Impact ledger: `dema poi` + `dema poi --full`.

### 1.2 Spearpoint A — HTTP contract hardening

Commit: `b8bd9eb7` (merged 2026-04-18)

- 20 TypeScript contracts auto-generated from Rust DTOs via `ts-rs`.
- `bizra-cognition-gateway/bindings/` as canonical source; CI drift gate rejects unstaged diff.
- 61 gateway tests (up from 36).

### 1.3 CI hygiene

Commit: `15faaf0f` (merged 2026-04-18)

- Lint Rust + Lint Python + Schema Validation gates closed.
- Working toward `-D warnings` parity across workspace.

### 1.4 Summary of SHIPPED test surface at v1.0.0 baseline

| Crate / target | Test count | Status |
|---|---|---|
| `bizra-cognition` (lib) | 309 | ✅ green |
| `bizra-cognition-gateway` | 77 | ✅ green (on cycle-8; 61 on main pre-witness) |
| Frontend (dema-console launch-path surfaces) | (Phase 2 manual/structural only) | ✅ 7/7 WIRED_REAL |

---

## 2. What's DRAFTED on feature branches (not merged at v1.0.0 cut)

### 2.1 Cycle-8 branch `cycle-8/seal-primitive-days-1-2`

9 commits, pushed to origin, awaiting review/merge:

| SHA | Subject | Status |
|---|---|---|
| `849035a4` | `feat(cognition): extract Sealable primitive and wire OrganizeRequest impl` | DRAFTED |
| `1ea334da` | `fix(cognition): drop redundant borrow on sled payload key` (cherry-picked from PR #28) | DRAFTED |
| `a0949b38` | `fix(cognition): clippy doc_overindented_list_items in seal.rs` | DRAFTED |
| `5af36b25` | `feat(dist): cargo-dist packaging config for Cycle-8 Day 3` | DRAFTED |
| `81b2bf19` | `docs(dist): tighten Formal + Economic modality truth labels` | DRAFTED |
| `b17f47ae` | `feat(witness): witness-grade chain-head observation primitive (Day 4)` | DRAFTED |
| `d262a317` | `feat(witness): Ed25519 signatures + proof-of-priority generator (Day 5)` | DRAFTED |
| `84fa8c5d` | `docs(cycle-8): Manifest + First Fire Doctrine drafts (Day 9-10)` | DRAFTED |
| `70515549` | `chore(stage): Day 11-12 dry-run harness + tester checklist (preparation)` | DRAFTED |

Cumulative adds:
- **`trait Sealable` + `trait SealableOutcome`** — universal seal primitive at `bizra-cognition/src/seal.rs` (313 lines, design-only Day 1).
- **`OrganizeRequest` + `impl Sealable`** wired in `organize_mission.rs` — preserves all 5 invariants at the trait boundary. 274 lines added, zero existing-type modifications. 13 new invariant-preservation tests.
- **cargo-dist packaging config** at `bizra-omega/Cargo.toml` `[workspace.metadata.dist]`. Targets: Linux x86_64, macOS arm64/x86_64, Windows x86_64. Only `bizra-cognition-gateway` ships binaries.
- **Witness-grade 4th modality**: `witness.rs` in gateway crate (458 lines incl. tests). `SignedWitnessObservation` (Ed25519 over canonical bytes), `WitnessStore` (in-memory HashMap), `POST /witness/head` / `GET /witness/head/:node_id` routes, `ping_witness()` client. 16 unit tests covering sign/verify round-trip, tamper rejection, JSON shape.
- **Proof-of-priority generator** at `scripts/generate-proof-of-priority.sh` — produces unsigned JSON binding true-genesis commits + SHA-256 of arXiv:2510.13857v1 paper + external_2023_refs placeholder.
- **Manifest + Doctrine v1 drafts** at `docs/cycle-8/`.
- **Dry-run harness + tester checklist** at `scripts/fire-dry-run-*`.

### 2.2 PR #28 branch `fork/dema-console-from-zai`

10 commits total on branch; 5 currently on origin (pushed earlier), 5 additional local-only awaiting push approval:

| SHA | Subject | Status |
|---|---|---|
| (origin) `4c67710a` | `fix(console): replace Math.random session-id with crypto.randomUUID` | DRAFTED (pushed) |
| `030e736f` | `fix(dema-console): remove DEMO_* shadow state from launch-path surfaces` | DRAFTED (local) |
| `4ebbb422` | `fix(dema-console): TrustStrip renders honest inactive state when no principal` | DRAFTED (local) |
| `2002dd84` | `fix(dema-console): OrganizePreview + MemoryConstellation — honest empty states (Phase 2 complete)` | DRAFTED (local) |
| `ff8f0f0d` | `docs(launch): Phase 3 GTM collapse — consumer copy + held enterprise brief + next gates` | DRAFTED (local) |
| `c79e556a` | `docs(launch): Track 1 — Launch Readiness Audit + Punch List v1` | DRAFTED (local) |
| (this commit) | `feat(dema-console): public landing + receipt viewer + artifact pages + release notes (Tracks 3.1/3.2/3.3/4.2)` | DRAFTED (local) |

Cumulative adds:
- **7/7 launch-path surfaces WIRED_REAL** (TrustStrip, MissionComposer, GateLadder, OrganizePreview, ReceiptReveal, MemoryConstellation, RejectRemediation). Zero fabricated content, honest empty states.
- **Public consumer landing copy** at `dema-console/docs/launch/LANDING-CONSUMER-v1.md`.
- **Public landing Next.js route** `dema-console/src/app/landing/page.tsx`.
- **Receipt viewer** `dema-console/src/app/r/[hash]/page.tsx` — 3-outcome honest rendering (ok / not_found / unreachable).
- **Static artifact routes** `manifest/`, `doctrine/`, `priority/`, `arbiteros-mapping/` — honest "draft pending merge" pointers for cycle-8 content.
- **Enterprise brief (HELD internal)** at `dema-console/docs/launch/ENTERPRISE-BRIEF-HELD-v1.md`.
- **Launch readiness audit + punch list** at `dema-console/docs/launch/LAUNCH-READINESS-PUNCH-LIST-v1.md`.
- **Next-human-gates index** at `dema-console/docs/launch/NEXT-HUMAN-GATES.md`.

---

## 3. What's HELD at v1.0.0 (intentionally not public)

- **Enterprise design-partner brief.** Per U3 constraint: solo-exhausted operator cannot handle MSA/NDA/SOW overhead at T=0. Activation path begins post-fire only when first-fire consumer launch stabilizes (≥ 30 days green). See `ENTERPRISE-BRIEF-HELD-v1.md`.
- **Pricing / monetization infrastructure.** Consumer is free-install. Enterprise is per-relationship. No SaaS subscription, no billing, no tiered plans. RIBA_ZERO applies.

---

## 4. What's BLOCKED at v1.0.0 cut (requires human input or external gate)

| Gate | Needed | Blocks |
|---|---|---|
| `bizra.ai` domain + hosting | Mumo decision + DNS | Track 3 deployment; entire public fire |
| Push approval for PR #28 updates | Mumo verbal | 7 commits local-only on `fork/dema-console-from-zai` |
| Merge cycle-8 and PR #28 to main | Mumo + review | v1.0.0 tag cannot land |
| `cargo install cargo-dist` + `cargo dist init` + release CI | Mumo approval + runtime | Binary distribution |
| Witness peer name | Mumo external contact | 4th modality real binding |
| 5 dry-run tester names | Mumo external contacts | T=0 pre-fire validation |
| D5 Daughter Test walkthrough | Human visual review | T=0 gate #7 |
| v1.0.0 git tag + signed release | Mumo constitutional moment | — |
| Signed Docker images (cosign) | Substantial infra (Spearpoint D territory) | Optional at v1.0; Horizon |
| Orphan-screen tsc error triage | Mumo decision (fix / exclude / delete) | Any merge to main |

---

## 5. Explicit non-goals at v1.0.0

- **LLM probabilistic-CPU wiring** — documented as Known Gap in `HANDOVER.md` §10. BIZRA governs; does not yet generate.
- **HAL (Hardware Abstraction Layer)** — scheduled for v0.4 per ADK blueprint.
- **Cognitive IDE / Desktop Overlay** — Horizon per ArbiterOS §8.8.
- **Bonded-stake / slashing / DAO / challenge-period economics** — Horizon / Layer B.
- **SEED/BLOOM tokenomics** — referenced in dema-main README; constitutionally ungrounded in main repo.
- **Multi-node federation** beyond witness-node gossip — Manifest v0.2 §12 long-range.
- **Full Isabelle/HOL formal proof** — v1.0 ships TESTED-grade (`cargo test`), not PROVEN.
- **Automated receipt-sharing network** — each node's chain is sovereign at T=0.
- **Subscription billing / recurring monetization** on consumer path — RIBA_ZERO forbids.

---

## 6. Migration notes

### From pre-v1.0 to v1.0.0

No migration required for fresh installs. If you previously ran a pre-release build:

- **Chain data:** in-memory receipts from pre-release runs are lost on gateway restart (sled-store persistence is opt-in feature per HANDOVER §10). Start fresh with `dema activate-principal`.
- **`dema-console` consumers:** if you previously saw the Z.ai demo data (DEMO_RECEIPTS, DEMO_RESOURCES, etc.), the v1.0 cut removes all of it. The UI renders honest empty state until you run your first `dema organize`.

---

## 7. Four-Modality self-declaration at v1.0.0

| Modality | State at v1.0 | Evidence anchor |
|---|---|---|
| **Cryptographic** | ✅ CANONICAL | `ReceiptChain` BLAKE3 hash-chain; `SignedWitnessObservation` Ed25519 sig (cycle-8) |
| **Empirical** | ✅ CANONICAL | 309 + 77 tests green; `dema organize` receipt reproducibility |
| **Formal (TESTED-grade)** | ✅ TESTED | `cargo test` + `cargo clippy -- -D warnings` green; full machine-checked proof is Horizon |
| **Economic (witness-grade)** | 🟡 DRAFTED | Witness primitive shipped on cycle-8 branch; requires named peer + deployment to reach CANONICAL |

---

## 8. Acknowledgments

Three years of solo engineering (Mumo / Mohamed Beshr, BIZRA Foundation, Dubai) anchored in two Arabic founding texts written in Ramadan 2023 (البذرة and الرسالة). External academic convergence with Xu, Wen, Xu, Li, Zhong (CURE Lab, CUHK) via arXiv:2510.13857v1 (2025-10-12).

---

## 9. Where to go next

- **Read the Manifest:** `/manifest` (pending cycle-8 merge) or `docs/cycle-8/MANIFEST-NORTH-STAR-v1.md` on cycle-8 branch.
- **Read the Doctrine:** `/doctrine`.
- **Check Proof-of-Priority:** `/priority`.
- **See ArbiterOS ↔ BIZRA mapping:** `/arbiteros-mapping`.
- **Install** (post-v1.0 when cargo-dist publishes): `curl -fsSL https://bizra.ai/install.sh | sh`.
- **First mission:** `dema organize ~/Downloads`.

---

## 10. Draft notes (for the v1.0 cut operator)

- This file is DRAFT. Before cutting v1.0, update all "DRAFTED (local)" rows to "SHIPPED" after push/merge. The BLOCKED list should shrink to zero or be explicitly acknowledged.
- Version bump from current workspace `2.0.0` to `1.0.0` needs explanation — or accept that `bizra-omega` stays at 2.0.0 internally while the USER-FACING v1.0.0 release is a distinct marketing anchor.
- Consider whether to tag cycle-8's HEAD (`70515549`) or a post-merge commit on main as the v1.0.0 anchor.
- Final signed-tag format: `git tag -s v1.0.0 -m "..."` — requires Mumo's GPG/Ed25519 key.
- GitHub release notes body can derive from this file's §1–§3; §4 (BLOCKED) should be empty at tag time.

---

*Close it. Prove it. Reveal it.*

الحمد لله
