# BIZRA Launch Readiness Punch List v1 (Cycle-8 Track 1)

بسم الله الرحمن الرحيم

**Status:** DRAFT (Cycle-8, 2026-04-19 19:00 GST). Launch Readiness Audit per Mumo's Track 1 definition.
**Scope:** exactly what's missing to transition BIZRA from current state to publicly releasable, user-ready, verifiable product.
**Method:** survey repo surfaces (frontend, kernel API, Docker, CI, docs, install, billing, public face); gap each against a consumer-first T=0 fire profile.
**Positioning lock:** U1=consumer · U2=no · U3=only-with-help (enterprise held).

---

## 1. Current state matrix (what exists)

| Dimension | Artifact | Current state |
|---|---|---|
| **Kernel runtime** | `bizra-cognition` crate | 309 tests green, 5 invariants enforced, Cycle-7 CANONICAL on main @ `b8bd9eb7` |
| **HTTP gateway** | `bizra-cognition-gateway` | 14 endpoints (health/chain/mission/principal/resources/organize/poi); 77 tests green; local-only 127.0.0.1:7421 |
| **Gateway + witness (cycle-8)** | `witness.rs` + ping/daemon/Ed25519 | Shipped on cycle-8 branch; 77 gateway tests green |
| **CLI** | `dema` binary | 7 subcommands, live-walked successfully on Node0 |
| **Face (PR #28)** | `dema-console/` Next.js | 7/7 surfaces WIRED_REAL (Phase 2 complete); consumer landing copy drafted (Phase 3) |
| **Face (alternate)** | `award-winner-design/dema` | 3/7 WIRED_REAL (per prior memory); separate repo, not on launch path |
| **Packaging** | `[workspace.metadata.dist]` cargo-dist | Config on cycle-8 branch; `cargo dist init` not run; no release CI workflow |
| **Install script (existing)** | `scripts/install.sh` | Node1-reproducibility bundle (9.1 KB, pre-existing, not cargo-dist-compatible) |
| **Install plan** | `scripts/dist-install-plan.md` | Specifies future cargo-dist flow; cycle-8 only |
| **Proof-of-priority** | `scripts/generate-proof-of-priority.sh` | Produces unsigned JSON manifest; cycle-8 only |
| **CI workflows** | 15+ `.github/workflows/*.yml` | Substantial surface; main currently partial-red (ihsan_gate test, Python env) |
| **Docker** | 10+ Dockerfiles across `deploy/`, `services/`, `frontend/`, root | Legacy surface; no single clear "production image" identified |
| **CHANGELOG** | `CHANGELOG.md` | Exists — content not yet audited for completeness |
| **README** | `README.md` | Covers BIZRA framing, PAT-7/SAT-5 architecture; not install-first |
| **Docs tree** | `docs/` 100+ files | Rich canon; launch-specific docs now at `dema-console/docs/launch/` (consumer copy, enterprise brief held, next gates) |
| **Manifest canon** | `docs/cycle-8/MANIFEST-NORTH-STAR-v1.md` | On cycle-8 branch |
| **Doctrine** | `docs/cycle-8/FIRST-FIRE-DOCTRINE-v1.md` | On cycle-8 branch |

---

## 2. Gaps per dimension (prioritized)

### 2A. Kernel / gateway (P0 — launch-blocking if missing)

✅ **No gap at kernel level.** Cycle-7 is CANONICAL; cycle-8 adds witness primitive. Launch-path kernel work is complete.

### 2B. CLI (P0)

✅ **No gap.** `dema` CLI is shipped, live-walked, and matches the install/first-run flow documented in Doctrine §4.

### 2C. Face — PR #28 dema-console (P0)

🟡 **Open gaps:**
- **Landing page does not exist yet.** `dema-console/src/app/page.tsx` is the OPERATOR dashboard (the product). A PUBLIC LANDING PAGE (at `bizra.ai/`, NOT the app) does not yet exist as code. Copy exists at `LANDING-CONSUMER-v1.md`; Next.js page implementation is Track 3.
- **Public route for `bizra.ai/r/<hash>`** (shareable receipt viewer) does not exist. Referenced in LANDING-CONSUMER-v1.md; no route implementation.
- **Pre-existing orphan-screen tsc errors** (autopilot / onboarding / operations / api/operations/route.ts / lib/api/client.ts) block `tsc --noEmit` clean run. Not on launch path but blocks merge-to-main.
- **D5 Daughter Test not executed.** Requires `bun install && bun run dev -p 3005` + browser walkthrough OR a Playwright harness (~300 LOC, currently Horizon).

### 2D. Install / packaging (P0)

🔴 **Major gaps:**
- **No production install.sh.** The existing `scripts/install.sh` is for node1 reproducibility, not consumer cargo-dist. A cargo-dist-generated installer is the target but requires `cargo install cargo-dist && cargo dist init && cargo dist build`.
- **No release CI workflow.** `cargo dist init` generates `.github/workflows/release.yml` — not yet generated.
- **No published binary artifacts.** `cargo-dist` config specifies 4 targets (Linux x86_64 / macOS arm64 / macOS x86_64 / Windows x86_64); first release build not executed.
- **No SHA-256 manifest publication plan.** `dist-manifest.json` produced by cargo-dist release CI; publication URL (`bizra.ai/install.sh`, `bizra.ai/r/<hash>`) requires a hosting surface.
- **Naming collision:** existing `scripts/install.sh` (node1) will conflict with future `bizra.ai/install.sh` (cargo-dist). Doctrine §4 flags this as Day 4+ resolution; not resolved.

### 2E. Public docs / landing (P0 for T=0)

🔴 **Major gaps:**
- **`bizra.ai/` website does not exist** as a deployable artifact. Copy drafted; hosting + deployment surface NOT yet decided.
- **No onboarding flow for first-time visitor.** Landing copy assumes direct install; no "try in browser first" path (by design — local-first).
- **`bizra.ai/manifest` / `/doctrine` / `/priority` / `/arbiteros-mapping`** URLs referenced in landing copy but not deployed.
- **ArbiterOS mapping doc** drafted at `reference_arbiteros_paper.md` (operator memory) but NOT in the public repo as a shareable artifact.

### 2F. CI — release pipeline (P1)

🟡 **Gaps:**
- **No release workflow.** 15+ CI workflows exist for internal gates; none produce release artifacts for T=0 consumer distribution.
- **Signed Docker images: not built, not signed.** Multiple Dockerfiles exist but no production image pipeline with cosign/sigstore.
- **No GitHub Releases drafts.** No v1.0 tag, no release notes generator wired.
- **Main branch CI partially red** (Test Rust `ihsan_gate::tests::observe_mode_allows_violations` + Test Python env failures) — INHERITED from pre-Cycle-8 state; not caused by Cycle-8 work; documented in `project_main_instability.md` operator memory.

### 2G. Billing / monetization (P0 decision, not P0 code)

⚠️ **Intentional non-goal at T=0.**
- Consumer path is **free** (install + run locally; no server, no account).
- Enterprise path is held (`ENTERPRISE-BRIEF-HELD-v1.md`); activation post-T=0.
- **No billing infrastructure required for T=0 consumer fire.** RIBA_ZERO explicitly forbids extractive monetization on consumer path.
- If/when enterprise activates, billing = per-contract invoicing (manual, operator-handled, NOT recurring SaaS). No Stripe / LemonSqueezy / recurring billing infra needed.

### 2H. Proof-of-impact / investor evidence (P1)

🟡 **Gaps (Track 2 scope):**
- **Proof-forge skill not invoked.** `proof-forge` skill exists per skill registry; bundle not yet generated.
- **No signed proof-of-priority.** `generate-proof-of-priority.sh` produces unsigned JSON; Ed25519 signing requires witness identity key that can also sign static artifacts — infrastructure exists (witness.rs Day 5), not yet wired to priority manifest.
- **No benchmark report.** `cargo bench` exists per workspace Cargo.toml; no aggregated benchmark report artifact.
- **No investor-grade Proof Summary document.** Cycle-7 retrospective + NODE0 commit log + ArbiterOS cite + receipt chain from a real `dema organize` run would constitute this bundle; not yet assembled.

### 2I. Domain + hosting (P0 for public fire)

🔴 **Major gap:**
- **`bizra.ai` DNS status unknown.** Prior session memory shows `WebFetch(domain:bizra.ai)` permission in settings but domain registration / DNS not verified in this session.
- **No hosting provider identified.** Landing page needs to be served from somewhere. Options: Vercel (recommended for Next.js), Netlify, Cloudflare Pages, self-hosted on Node0.
- **No CDN / SSL strategy** documented.

Audit note: this is a hard launch gate. Without `bizra.ai` live, nothing else in the consumer flow works.

---

## 3. Punch list organized by Mumo's 4 tracks

### Track 1 — Launch Readiness Audit

| Item | Status | Owner |
|---|---|---|
| Write this doc | ✅ this commit | Claude |
| Walk the doc with Mumo | ⏳ next session | Mumo |
| Triage P0 items | ⏳ | Mumo |

### Track 2 — Investor Proof Package

| # | Item | Autonomy | Blocker |
|---|---|---|---|
| 2.1 | Invoke `proof-forge` skill to generate the full evidence bundle | 🟡 Claude-capable | Requires skill invocation authority + output location approval |
| 2.2 | Sign `generate-proof-of-priority.sh` output with Ed25519 via witness infra | 🟡 Claude-capable | Requires BIZRA signing key path (env or file) |
| 2.3 | Run `cargo bench` across workspace, aggregate into report | 🟡 Claude-capable | ~15 min runtime + decision on where to commit |
| 2.4 | Write `docs/cycle-8/PROOF-SUMMARY-v1.md` combining: Cycle-7 retrospective + priority manifest + benchmark report + ArbiterOS cite + real `dema organize` receipt | ✅ Claude-autonomous | None |
| 2.5 | Review bundle for investor consumption | 🔴 Human | Mumo only |

### Track 3 — Public Landing + Install Flow

| # | Item | Autonomy | Blocker |
|---|---|---|---|
| 3.1 | Implement `bizra.ai/` landing as a dedicated Next.js route in dema-console (`src/app/landing/page.tsx` or similar) from `LANDING-CONSUMER-v1.md` | ✅ Claude-autonomous | ~300 LOC React; needs scope approval |
| 3.2 | Implement `bizra.ai/r/<hash>` receipt viewer route | ✅ Claude-autonomous | ~150 LOC; needs gateway access from browser (CORS) |
| 3.3 | Implement `bizra.ai/manifest`, `/doctrine`, `/priority`, `/arbiteros-mapping` static routes that render the markdown docs | 🟡 Claude-autonomous | Requires markdown-to-HTML pipeline decision (simple: use Next.js MDX) |
| 3.4 | Decide and register `bizra.ai` domain + hosting provider | 🔴 Human | Mumo only — DNS + payment |
| 3.5 | `cargo install cargo-dist && cargo dist init && cargo dist build` on cycle-8 branch | 🔴 Human | Requires `cargo install` approval (ask gate) |
| 3.6 | Publish install.sh at `bizra.ai/install.sh` | 🔴 Human | Requires hosting + deployment |
| 3.7 | Resolve `scripts/install.sh` vs `bizra.ai/install.sh` naming collision (rename node1 installer, or merge flows) | 🟡 Claude-capable | Decision needed |

### Track 4 — Release Pipeline + v1 Cut

| # | Item | Autonomy | Blocker |
|---|---|---|---|
| 4.1 | Audit `CHANGELOG.md` for completeness; draft v1.0 section from git log | ✅ Claude-autonomous | None |
| 4.2 | Draft GitHub release notes for v1.0 | ✅ Claude-autonomous | None |
| 4.3 | Bump `bizra-omega/Cargo.toml` workspace.package.version to 1.0.0-rc1 as prep | ✅ Claude-autonomous | None (local-only commit) |
| 4.4 | Generate `.github/workflows/release.yml` (requires cargo-dist installed) | 🔴 Human | Requires cargo-dist install |
| 4.5 | Build signed Docker images (cosign integration) | 🔴 Human | Substantial new infra; Horizon per Spearpoint D |
| 4.6 | `git tag -s v1.0.0` — constitutional moment | 🔴 Human | Mumo only |
| 4.7 | `git push --tags` + `gh release create` | 🔴 Human | Mumo only |

---

## 4. Immediate-next autonomous-actionable items (no human gate)

Per the table above, the following are executable right now without Mumo input:
- **2.4** — Draft `docs/cycle-8/PROOF-SUMMARY-v1.md` (Track 2)
- **3.1** — Implement public landing page as Next.js route (Track 3)
- **3.2** — Implement receipt viewer route (Track 3)
- **3.3** — Implement static manifest/doctrine/priority routes (Track 3)
- **4.1, 4.2, 4.3** — Audit CHANGELOG + draft release notes + bump version to 1.0.0-rc1 (Track 4 prep)

All the rest have human gates (push approval, domain decision, signing key, hosting provider, v1.0 tag authority).

## 5. T=0 hard gates (launch cannot fire until all these green)

| # | Gate | Currently | To get green |
|---|---|---|---|
| G1 | `bizra.ai` domain registered + DNS resolves | ❓ unknown | Mumo action |
| G2 | `bizra.ai/install.sh` serves cargo-dist-generated installer | ❌ | Track 3 + hosting |
| G3 | At least 1 witness peer online + reachable + agreeing on chain head | ❌ TBD | Witness peer name + deploy |
| G4 | 5 / 5 dry-run testers complete flow | ❌ TBD | Tester names + harness run |
| G5 | D5 Daughter Test passes all 7 surfaces | ❌ untested | Run browser walkthrough |
| G6 | v1.0.0 tag on main | ❌ | All above green first |
| G7 | Signed proof-of-priority published | ❌ | Ed25519 signing step |
| G8 | Mumo slept ≥ 8 hours in 24h pre-fire | 🔴 unknown | Mumo only |
| G9 | 10k+ distribution connection confirmed ready | ❓ | Mumo only |

---

## 6. Recommended next-step sequence (autonomous + human-paired)

**Autonomous (Claude can execute without pause):**
1. Track 4.1–4.3 — CHANGELOG audit + v1.0 release notes draft + version bump prep (small commits)
2. Track 2.4 — Assemble `PROOF-SUMMARY-v1.md` from existing artifacts
3. Track 3.1 — Implement landing page Next.js route (requires scope approval since it's ~300 LOC)

**Human-gated (must wait):**
- Track 3.4 — `bizra.ai` domain + hosting decision (highest strategic gate)
- Track 3.5 — `cargo install cargo-dist` approval
- Track 4.6 — v1.0 tag authority
- Witness peer name
- 5 tester names
- Push approval for PR #28 + cycle-8 branches

**SNR ranking:** G1 (domain) is the single highest-leverage gate. Without `bizra.ai` resolving, Track 3 cannot complete and neither can T=0 fire. Resolving G1 unblocks more than any other single move.

---

## 7. What's explicitly NOT in scope at T=0

These are Horizon / Layer B / post-fire items that this audit intentionally does not include:
- Desktop overlay / Cognitive IDE (ArbiterOS §8.8)
- LLM probabilistic-CPU wiring (HANDOVER §10 known gap)
- HAL formalization (v0.4 roadmap)
- Bonded stake / slashing / DAO / challenge-period economics
- SEED/BLOOM tokenomics
- Multi-node federation (beyond witness gossip)
- Repo consolidation 150 → 5 (post-fire hygiene)
- Full Isabelle/HOL formal proof (full Formal modality — T=0 is TESTED-grade)
- Subscription billing / recurring monetization (RIBA_ZERO)

---

*Close it. Prove it. Reveal it.*

الحمد لله
