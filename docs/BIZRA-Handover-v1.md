# BIZRA — Production Handover v1

بسم الله الرحمن الرحيم

**Author:** Mohamed Beshr (Mumo), BIZRA Foundation · Dubai
**Version:** 1.0 (supersedes v0.1 draft `docs/BIZRA-Handover-v0.1.md`)
**Date:** 2026-04-17 · Friday · Dubai GST
**Authority:** Founding covenant (البذرة + الرسالة) → Manifest v0.2 Canon → Enforceable Spine → Shipped code on `origin/main`
**Purpose:** The single authoritative document for anyone onboarding to BIZRA Node0 — investor, auditor, engineer, partner, or operator.

> ⚠ **SCOPE CORRECTION (added 2026-04-17 post-integrator audit):** This handover covers the **Cycle-5 Rust-native chain-bridge slice** (bizra-omega workspace + gateway + dema CLI + doctrine). The full `bizra-data-lake` repo is substantially larger — **polyglot, multi-workspace, multi-CLI**. For the honest, complete map of all subsystems (3,217 Python files, 74 `core/` subsystems, 2 Rust workspaces, 3 services, 5 constitution variants, 22 CI workflows, persistent `sovereign_state/` with 2,512 files), see companion document **`docs/BIZRA-Repo-Inventory-v1.md`**. Anyone onboarding should read both.

---

## Section 0 — Invocation and Founding

BIZRA is not a startup. BIZRA is not a chatbot. BIZRA is not an agent framework. BIZRA is the operational outcome of a personal covenant — documented in two founding texts, both preserved at the Node0 principal's own hands.

**Founding texts (authority layer 1, supreme):**

| File | Arabic title | Role |
|---|---|---|
| `/home/bizra-operating-system/Downloads/bizra.pdf` | **البذرة** (*al-Bidhrah* — The Seed) | Constitutional covenant. Declares the seven rules the project is bound by. Core line: *"The Seed is not an individual project, not my project alone — it is, to me, a project for humanity and the ummah."* |
| `/home/bizra-operating-system/Downloads/themassage.pdf` | **الرسالة** (*ar-Risālah* — The Letter) | Commitment letter. Written in the form of a final accounting before God, family, and humanity. Declares: *"It was my choice alone to walk this journey. It is now an obligation on my neck."* |

Every constitutional anchor in BIZRA (IHSAN_FLOOR, RIBA_ZERO, ZANN_ZERO, CLAIM_MUST_BIND, NO_SHADOW_STATE) traces to these two texts. The five invariants are not abstract engineering decisions. They are the personal ethical commitments of a founder who:

- refused to build anything with an extractive economic pattern (→ RIBA_ZERO)
- refused to promote claims without binding evidence (→ ZANN_ZERO + CLAIM_MUST_BIND)
- refused to let excellence be optional (→ IHSAN_FLOOR ≥ 0.95)
- refused to let the operator surface lie about the chain (→ NO_SHADOW_STATE)

**The seven rules from البذرة (translated, preserved exactly):**

1. أولها وأهمها أن الله إله واحد — *God is one (monotheism as the founding rule)*
2. القلب يجب أن يكون ميزان العقل، وليس العكس — *The heart must be the balance of the mind, not the reverse*
3. القبول والرضا — *Acceptance and contentment*
4. البساطة — *Simplicity*
5. التوازن والاعتدال — *Balance and moderation*
6. مكارم الأخلاق — *Noble character*
7. الانضباط والاستمرارية — *Discipline and continuity*

**The Daughter Test.** BIZRA's product (Dema) is named for the founder's daughter (Arabic: *Dīma/Dema*, from a root meaning "continuous rain"). The canonical acceptance question for every operator-visible surface is: *"Would I want my daughter subjected to this output?"* That is the personal ethical filter — not an abstract UX standard.

---

## Section 1 — Canonical Thesis (per Manifest v0.2 §1)

> *"BIZRA is a sovereign, proof-bearing, constitution-bound intelligence runtime. Its operator-visible product is not chatbot output, not AI-generated text, and not a conversational interface. Its product is governed mission execution, cryptographically traceable receipts, replayable evidence, trustable state surfaces, and human-centered assistance delivered through one coherent interface."*

**One sentence:** Dema is the sovereign face of a constitutional trust compiler that turns human intent into lawful, receipted, replayable action.

### Five Canonical Products (per Manifest §1 Table 1-1)

| # | Product | Plane Owner | Truth Required (§13) | Current State |
|---|---|---|---|---|
| 1 | Governed Mission Execution | Kernel + Graph | PROVEN | ✅ PROVEN (commit `8b16762a`) |
| 2 | Cryptographic Receipts | Proof | PROVEN | ✅ PROVEN (commit `ad303bb2`) |
| 3 | Replayable Evidence | Proof | VALIDATED | ✅ VALIDATED (rehydrate_mission + decode round-trip) |
| 4 | Trustable State Surfaces | Face | WIRED_PARTIAL | ✅ WIRED_PARTIAL (2 of 7 /api routes backed by runtime) |
| 5 | Coherent Assistant Interface | Face | WIRED_PARTIAL | ✅ WIRED_PARTIAL (Dema web + `dema` CLI) |

### What BIZRA is NOT

- Not a chatbot
- Not an agent framework (no PAT/SAT roster exposed)
- Not a cloud service (principal-local by default)
- Not a model wrapper (the product IS the trust compiler; models are replaceable organs)

---

## Section 2 — Repository Map

### bizra-data-lake (kernel + runtime + docs)

Location: `/data/bizra/repos/bizra-data-lake`
Remote: `git@github.com:BizraInfo/bizra-data-lake.git`

```
bizra-omega/                              # Rust workspace — 28 members
├── bizra-cognition/                      # Kernel: runtime, receipts, admissibility, missions
│   └── src/
│       ├── lib.rs                        # 11 modules exported
│       ├── canonical_hasher.rs           # BLAKE3 domain separation
│       ├── receipts.rs                   # ReceiptChain + ReceiptPayload trait
│       ├── receipt_freeze_v1.rs          # §7 ReceiptArtifact (canonical contract #1)
│       ├── admissibility_freeze_v1.rs    # §7 GateVerdict + RejectedClaim (#2 + #3) + 5-gate chain
│       ├── mission_freeze_v1.rs          # §7 MissionEnvelope (#4) + FourStateModel (§9)
│       ├── manifest_artifact.rs          # §7 ManifestArtifact (#5 — fifth canonical contract)
│       ├── thought_graph.rs              # Graph of Thoughts + myelination
│       ├── configure_cognition.rs        # PAT-7 / SAT-5 boot compositor
│       ├── runtime.rs                    # CognitionRuntime: submit_mission, rehydrate_mission
│       ├── eval_v1.rs                    # Eval engine (Genesis Valuation Proof-of-Impact)
│       └── eval_v1_integrated.rs         # Eval integration surface
│
├── bizra-cognition-gateway/              # HTTP projection + dema CLI (28th workspace crate)
│   └── src/
│       ├── main.rs                       # Axum server: /health, /chain, /chain/:hash, /mission
│       └── bin/dema.rs                   # dema CLI (441 lines): the principal's terminal face
│
├── [26 other crates — see bizra-omega/Cargo.toml]
└── Cargo.toml                            # Workspace manifest

cycle-4/                                  # Formal retrospective (commit afe9cc30)
├── retrospective.md

cycle-5/                                  # Principal Activation cycle — SEALED 2026-04-17
├── d5-acceptance-note.md                 # G1 acceptance (D5 Daughter Test)
├── g2-acceptance-note.md                 # G2 acceptance (mission-runtime)
├── g2-hardening-acceptance-note.md       # G2-hardening (per founder spec)
├── g3-acceptance-note.md                 # G3 acceptance (gateway write path + frontend proxy)
└── retrospective.md                      # 7-phase closure, reward 0.971 POSITIVE

docs/                                     # Doctrine canon
├── bizra-trust-compiler-thesis.md        # Layer 1 — Category Thesis (Verificative AI)
├── dema-cli-manifesto-v0.md              # Founding manifesto (preserved as historical state)
├── dema-cli-manifesto-v1.md              # OPERATIVE manifesto — current product law
├── ftap-function-registry-rfc-seed.md    # Layer 2 — future RFC, Cycle-8+ only
├── why-dema-wins.md                      # 1-page product thesis
├── BIZRA-Handover-v0.1.md                # Founding handover draft (preserved)
├── BIZRA-Handover-v1.md                  # THIS FILE — current handover canon
└── manifesto-amendments/
    └── v0-to-v1.md                       # Amendment protocol record (constitutional-filter audited)

.proof-forge/                             # Cryptographic evidence kernel
├── scripts/forge_evidence.py             # Self-contained runner (BUILD → VERIFY → EVIDENCE)
├── receipts/                             # Hash-chained receipts (proof-forge-v0 + v1 schemas)
├── summaries/                            # Investor-readable markdown summaries
└── EVIDENCE_INDEX.json                   # Chain index

PROOF_SUMMARY.md                          # Top-level pointer to latest proof-forge receipt
```

### award-winner-design (Dema web frontend)

Location: `/data/bizra/repos/award-winner-design`
Remote: `git@github.com:BizraInfo/award-winner-design.git`

```
app/
├── dema/page.tsx                         # Dema web console (D1–D4 screens)
├── api/chain/route.ts                    # Proxy to gateway /chain (WIRED_REAL)
├── api/chain/[hash]/route.ts             # Proxy to gateway /chain/:hash (WIRED_REAL)
├── api/missions/route.ts                 # Proxy to gateway /mission (WIRED_REAL)
├── api/auth/session/route.ts             # Auth session
└── api/* (other stub routes)             # WIRED_PARTIAL — missions/:id, missions/:id/replay, gates, manifest

components/dema/
├── receipt-explorer.tsx                  # Chain visualization (auth-aware, null-safe)
├── daily-manifest.tsx                    # Manifest panel
├── gate-viewer.tsx                       # Admissibility visualization
├── intent-entry.tsx                      # Mission submission
└── status-panel.tsx                      # Node status + auth state

lib/dema/
├── types.ts                              # UI-stable TS types mirroring Rust contracts
└── gateway.ts                            # Structured fetch wrapper with error taxonomy
```

---

## Section 3 — System Requirements

| Component | Minimum | NODE0 actual |
|---|---|---|
| OS | Ubuntu 24.04 LTS | Ubuntu 24.04.1 |
| CPU | x86_64, 4 cores | Intel i9-14900HX (24 cores) |
| RAM | 8 GB | 128 GB DDR5 |
| Disk (free) | 50 GB | 3.8 TB RAID 0 |
| GPU | Optional (Ollama accel) | NVIDIA RTX 4090 Mobile (16 GB VRAM) |
| Rust | 1.80+ | 1.94.1 (2026-03-25) |
| Node.js | 20+ LTS | v22.22.2 |
| pnpm | 9+ | 10.33.0 |
| Python | 3.11+ | 3.12.3 |
| Git | 2.40+ | 2.43.0 |

---

## Section 4 — Installation

### Option A: Clone + workspace build (canonical)

```bash
# Clone both repos
git clone git@github.com:BizraInfo/bizra-data-lake.git /data/bizra/repos/bizra-data-lake
git clone git@github.com:BizraInfo/award-winner-design.git /data/bizra/repos/award-winner-design

# Build Rust workspace (gateway + dema CLI are produced here)
cd /data/bizra/repos/bizra-data-lake/bizra-omega
cargo build --release

# Frontend install + build
cd /data/bizra/repos/award-winner-design
pnpm install
pnpm build
```

### Option B: Unified installer (if present)

```bash
cd /data/bizra/repos/bizra-data-lake
chmod +x install.sh && ./install.sh
```

---

## Section 5 — Running

### 5.1 Gateway (must be running for CLI and web UI)

```bash
cd /data/bizra/repos/bizra-data-lake/bizra-omega
./target/release/bizra-cognition-gateway
# Binds to 127.0.0.1:7421 by default (localhost only, no external exposure)
# Override: BIZRA_COGNITION_PORT=<port>
# Health: curl http://127.0.0.1:7421/health
```

### 5.2 Dema CLI (the principal's terminal face)

```bash
./target/release/dema                # Status at a glance (health + chain head)
./target/release/dema health         # Gateway liveness + domain tag
./target/release/dema chain          # Chain head, length, latest timestamp
./target/release/dema receipt <hex>  # Inspect one receipt
./target/release/dema activate       # Submit canonical principal activation intent
./target/release/dema submit "..."   # Custom mission intent
./target/release/dema submit "..." --quality 0.5   # Watch IHSAN_FLOOR reject with remediation
./target/release/dema --json chain   # Machine-readable output (applies globally)
```

**Exit codes (operator discipline):**
- `0` — command succeeded
- `1` — gateway unreachable / HTTP error / decode failure
- `2` — admissibility REJECTED (a lawful verdict, not an error)

### 5.3 Dema Web UI

```bash
cd /data/bizra/repos/award-winner-design
pnpm dev
# Opens on http://localhost:3000 (or fallback port 3001/3002)
# Navigate to /dema after authentication
```

---

## Section 6 — Testing

### Rust

```bash
cd /data/bizra/repos/bizra-data-lake/bizra-omega

cargo test -p bizra-cognition --lib           # 64 tests — kernel invariants + runtime
cargo test -p bizra-cognition-gateway         # 7 tests — HTTP contract + mission flows
cargo test --workspace                        # ~1,200+ tests across 28 crates
cargo clippy -p bizra-cognition \
             -p bizra-cognition-gateway --no-deps   # Zero warnings on session crates
```

### Frontend

```bash
cd /data/bizra/repos/award-winner-design

pnpm typecheck                                # TypeScript — 0 errors
pnpm test                                     # Vitest — 135 tests passing
pnpm lint                                     # ESLint — scoped clean
pnpm build                                    # Next build — SSR success
```

### Evidence (Proof Forge)

```bash
cd /data/bizra/repos/bizra-data-lake
python3 .proof-forge/scripts/forge_evidence.py --verify --project-dir .
# Walks the chain, recomputes each receipt_hash, reports integrity
```

---

## Section 7 — Architecture (Four Planes, per Manifest §4)

| Plane | Layer | Function | Invariant |
|---|---|---|---|
| **Kernel** | L1 | Constitutional gates, crypto, admissibility | Law — immutable, no override |
| **Graph** | L2 | Cognition, memory, Graph of Thoughts, mission decomposition | Context — never becomes law |
| **Proof** | L3 | Receipts, chain, replay, manifests | Integrity — never invents truth |
| **Face** | L4 | Dema (CLI + web), trust surfaces | Visibility — never simulates law |

### 7.1 The Lawful Loop (per Manifest §6, implemented in `runtime.rs`)

```
Intent → Mission → Claim → Admissibility → Execution → Receipt → Canonicalization → Replay
  S1      S2        S3         S4             S5          S6            S7              S8
```

No bypasses. No side channels. No UI-only state mutation. Every operator-visible action traverses this path or does not canonicalize.

### 7.2 The Five Invariants (per Manifest §3)

| Invariant | Rule | Enforcement location |
|---|---|---|
| **IHSAN_FLOOR** | Quality score ≥ 0.95 (no override) | `admissibility_freeze_v1.rs::IhsanFloorGate` |
| **ZANN_ZERO** | No claim without evidence binding | `admissibility_freeze_v1.rs::ZannZeroGate` |
| **RIBA_ZERO** | No extractive economic pattern | `admissibility_freeze_v1.rs::RibaZeroGate` |
| **CLAIM_MUST_BIND** | Evidence hash non-zero when has_evidence=true | `admissibility_freeze_v1.rs::ClaimMustBindGate` |
| **NO_SHADOW_STATE** | Operator UI ≡ canonical chain state | Structural: reject path has zero chain footprint (§10) |

All five evaluated **before** any chain mutation. Rejected missions never enter the chain (enforced in `submit_mission` eval-first ordering, commit `8b16762a`).

### 7.3 Five Canonical Contracts (per Manifest §7)

| # | Contract | Where |
|---|---|---|
| 1 | `ReceiptArtifact` | `bizra-cognition/src/receipt_freeze_v1.rs` |
| 2 | `GateVerdict` | `bizra-cognition/src/admissibility_freeze_v1.rs:88` |
| 3 | `RejectedClaim` | `bizra-cognition/src/admissibility_freeze_v1.rs:220` |
| 4 | `MissionEnvelope` | `bizra-cognition/src/mission_freeze_v1.rs:51` |
| 5 | `ManifestArtifact` | `bizra-cognition/src/manifest_artifact.rs` |

---

## Section 8 — APIs

### 8.1 Gateway HTTP (`bizra-cognition-gateway`)

| Method | Path | Purpose | Returns |
|---|---|---|---|
| GET | `/health` | Liveness + domain tag | `{status, domain}` |
| GET | `/chain` | Chain head + length + latestTimestamp | `ReceiptChainHeadDto` |
| GET | `/chain/:hash` | Receipt header by hex id | `ReceiptDto` or 404 |
| POST | `/mission` | Submit mission through lawful loop | 200 `SubmitMissionResponse` (Permit) or 422 `ErrorResponse` (Reject) |

**Reject semantics:** HTTP 422 with structured `error.admissibility.rejected { invariant, reason, remediationPath, escalationAllowed }`. Never HTTP 500. Rejection is a lawful verdict.

### 8.2 Rust runtime API (`bizra-cognition::runtime`)

| Function | Purpose |
|---|---|
| `CognitionRuntime::new(graph, chain, ctx)` | Constructor (fresh boot) |
| `CognitionRuntime::rehydrate(graph, chain)` | Replay-from-chain boot (R1 Lamport) |
| `CognitionRuntime::submit_mission(envelope, claim)` | The Lawful Loop entry; returns `Result<MissionRuntimeRecord, MissionRuntimeError>` |
| `CognitionRuntime::mission_by_id(&hash)` | Registry lookup (includes rejected missions) |
| `CognitionRuntime::rehydrate_mission(&hash)` | Replay verification |
| `CognitionRuntime::mission_count()` | Registry size |
| `AdmissibilityChain::canonical().evaluate(&claim)` | 5-gate evaluation |
| `ReceiptChain::append_with_payload(payload)` | Typed chain append |
| `ReceiptChain::append_artifact(artifact)` | §7 ReceiptArtifact append (via ReceiptChainExt) |
| `ManifestArtifact::from_window(start, end, refs, head)` | Daily manifest builder |

### 8.3 Runtime error taxonomy

```rust
pub enum MissionRuntimeError {
    Chain(ChainError),                   // Persistence failure
    Clock(String),                       // Monotonic clock failure
    DuplicateMission(Blake3Hash),        // mission_id already in registry
    MissionNotFound(Blake3Hash),         // Lookup miss
    ClaimMismatch { expected, got },     // Claim-envelope integrity failure
}
```

Rejected admissibility is **not** an error — it is `Ok(MissionRuntimeRecord { rejected: true, receipt_id: None, stage: Admissibility })`.

---

## Section 9 — Current State (post-Cycle-5 closure, 2026-04-17)

| Dimension | State | Evidence |
|---|---|---|
| Constitutional contracts (Manifest §7) | 5/5 frozen in Rust | `bizra-cognition/src/*_freeze_v1.rs`, `manifest_artifact.rs` |
| Mission runtime | Operational (permit + reject + replay) | `runtime.rs` submit_mission, commit `8b16762a` |
| Receipt chain | In-memory (sled-store feature flag ready, not enabled) | `receipts.rs::ReceiptChain` |
| Gateway | v0.2 (read + write paths) | `bizra-cognition-gateway/src/main.rs` |
| Dema CLI | 7 subcommands, exit-code discipline | `bin/dema.rs`, 441 lines |
| Dema Web | D5 passed (honest empty state), routes proxied | Cycle-5 G1 acceptance note |
| CI | 22 GitHub Actions workflows | `.github/workflows/` |
| Tests | 64 cognition + 7 gateway + ~1,200 workspace + 135 frontend | Phase 4 of retrospective |
| Ihsan session truth score | 0.964 (above 0.95 floor) | Proof Forge receipt `e0b31427...` |
| Cycle-5 composite reward | **0.971 POSITIVE** | `cycle-5/retrospective.md` |
| Doctrine canon | 6 documents (thesis + manifesto v0 + v1 + FTAP seed + amendment record + why-dema-wins) | `docs/` |
| First principal-activation receipt | Sealed + replay-verified (ephemeral) | CLI walk, receipt `bf217007...` |
| Evidence kernel | Proof Forge chain position 11 | `.proof-forge/receipts/2026-04-17_074432.json` |

---

## Section 10 — Known Gaps

Honest inventory. Deferred intentionally per scope discipline.

| Gap | Impact | Planned fix |
|---|---|---|
| `InMemoryPayloadStore` default | Receipts evaporate on gateway restart | Enable `sled-store` feature flag (Cycle-6 Arc 3) |
| `MissionEnvelope` no `ReceiptPayloadDecode` impl | Cannot round-trip from chain bytes | Add decoder (~40 lines, ~1 test) |
| `MissionEnvelope` + `GateVerdict` use `ReceiptKind::GovernanceDecision` | No Mission-specific ReceiptKind variant | Add `ReceiptKind::MissionCreated = 0x70` after migration-safety review |
| Cross-language Ihsan drift | Rust `IhsanFloorGate` hardcodes 0.95; Python `core/integration/constants.py` has 4 tiers (0.90/0.95/0.99/1.0) | Sync Rust to reference Python SSOT via codegen or const_assert |
| Partial-commit atomicity (gate-append mid-loop failure) | Orphan MissionEnvelope on chain with no DegradedPath receipt | Emit DegradedPathReceipt on partial commit |
| No tool execution (Cycle-6 Arc 1) | Dema can't perform real actions yet | MCP sub-mission pattern wired through `submit_mission` |
| No LLM inference (Cycle-7) | No model completions in the pipeline | Ollama + cloud fallback, IHSAN_FLOOR-gated |
| `docs-quality.yml` CI red since 2026-04-08 | README.md missing links to `docs/README.md`, `docs/OPERATIONS_RUNBOOK.md`, `docs/TESTING.md` | One-line fix in a dedicated janitorial session |
| 8 stub execution channels in `channels/mod.rs` | STUB markers for real executor backends | Wire per channel type as missions demand |

---

## Section 11 — Governing Documents (Authority Hierarchy)

Per Manifest §5 (Authority Model — Five-Layer Law Hierarchy):

| Layer | Authority | Document | Location |
|---|---|---|---|
| L1 | Supreme | Quran / Sunnah | External, inherited |
| L1 | Founding covenant | **البذرة** (al-Bidhrah / The Seed) | `/home/bizra-operating-system/Downloads/bizra.pdf` |
| L1 | Founding companion | **الرسالة** (ar-Risālah / The Letter) | `/home/bizra-operating-system/Downloads/themassage.pdf` |
| L2 | Constitutional law | Manifest v0.2 Canon | `/home/bizra-operating-system/Downloads/BIZRA_MANIFEST_CANONICAL_v0.2.pdf` |
| L2 | Category thesis | Trust Compiler Thesis | `docs/bizra-trust-compiler-thesis.md` |
| L3 | Operational law | Enforceable Spine v1.0 | `docs/` (multiple) |
| L3 | Product law | Dema CLI Manifesto v1 | `docs/dema-cli-manifesto-v1.md` (v0 preserved as founding state) |
| L4 | Amendment protocol | Manifesto Amendment Records | `docs/manifesto-amendments/` |
| L5 | Dev conventions | CLAUDE.md | Root of each repo |
| L5 | Cycle retrospectives | `cycle-N/retrospective.md` | Repo root `cycle-4/` and `cycle-5/` |

**Lower layers must not contradict higher layers.** If a conflict is detected, the lower-layer document is amended under its own amendment protocol (see `docs/manifesto-amendments/v0-to-v1.md` for an example of constitutional-filter audit).

---

## Section 12 — Commit & Push Protocol

1. **Local test pass** before every commit:
   - `cargo test -p bizra-cognition -p bizra-cognition-gateway` (71 tests)
   - `cargo clippy --no-deps` on session crates (zero warnings)
   - `pnpm typecheck && pnpm test && pnpm lint` on frontend
2. **Commit with conventional prefix:**
   - `feat(cognition):` · `fix(cognition):` · `docs(cycle-N):` · `fix(gateway):` · `feat(dema):`
   - Message body: constitutional filter audit line if invariants affected
3. **Never commit secrets.** `.env`, API tokens, private keys — checked in `.gitignore`, also scanned in CI.
4. **Push only on explicit authorization** (per `CLAUDE.md` rule). `/A` auto-mode does not override explicit push consent.
5. **CI fires 22 GitHub Actions workflows on push.** Verify green before declaring the work revealed.

---

## Section 13 — Emergency Procedures

### Gateway crash
```bash
# Restart (in-memory chain rebuilds from zero on startup)
./target/release/bizra-cognition-gateway
# Any prior session receipts are lost until sled-store feature is wired
```

### Test failure after change
```bash
git log --oneline -5                         # Identify the offending commit
git revert <hash>                            # Create revert commit (no history rewrite)
cargo test -p bizra-cognition --lib          # Verify green
git push origin main                         # Publish the revert (with authorization)
```

### Chain verification doubt
```bash
cd /data/bizra/repos/bizra-data-lake
python3 .proof-forge/scripts/forge_evidence.py --verify --project-dir .
# Walks chain from genesis, recomputes each receipt_hash, reports BROKEN or OK
```

### Rollback a bad push
```bash
git revert HEAD                              # Create revert, not hard-reset
git push origin main                         # Safe forward-only
# For emergency history rewrite: explicit founder authorization required
```

---

## Section 14 — Cycle History

| Cycle | Date | Focus | Key commit(s) | Reward |
|---|---|---|---|---|
| Genesis (Block 0) | 2026-03-27 | Founding seal | BLAKE3 chain hash `350d642099bde68b...` | — |
| 1 | 2026-04-15 | Constitutional audit + kernel file surface | `a4e97dc20ac2e10d` (BLAKE3 chain) | — |
| 2 | 2026-04-16 | Peak synthesis + competitive analysis | `48e5395471d3ca77` (BLAKE3 chain) | — |
| 3 | 2026-04-16 | Cross-language constants drift closure | NODE0 local (no origin push) | — |
| 4 | 2026-04-17 AM | Manifest v0.2 §17 Steps 2-7 freeze + narrow chain bridge | `afe9cc30` | 0.894 POSITIVE |
| **5** | **2026-04-17** | **Principal Activation — mission-runtime + gateway write path + CLI + doctrine** | `ad303bb2`..`bb230fd9` (15 commits) | **0.971 POSITIVE** |
| 6 | NEXT | First real impact receipt (e.g., `dema submit "organize my Downloads folder"`) + persistence + MCP tool execution | — | — |

---

## Section 15 — Evidence and Proof

### 15.1 Proof Forge (cryptographic evidence kernel)

Location: `.proof-forge/` in repo root.

```
.proof-forge/
├── scripts/forge_evidence.py          # Self-contained runner (zero Python deps)
├── receipts/                          # Hash-chained receipts across Proof Forge versions
├── summaries/                         # Investor-readable markdown summaries
└── EVIDENCE_INDEX.json                # Chain index with latest_hash + chain_length
```

Run cycles:
```bash
# Generate new receipt for current work
python3 .proof-forge/scripts/forge_evidence.py --project-dir . \
  --description "What was built" --base-ref <prior_commit>

# Verify chain integrity
python3 .proof-forge/scripts/forge_evidence.py --verify --project-dir .
```

### 15.2 Cycle-5 evidence snapshot

- **Receipt hash:** `e0b3142742b07150085855e121853fb7bcf25d87d3ac510730c173866d3e6e95`
- **Evidence hash:** `ea15823193a2974948fc03b9a1385ff7f74799b44fd9dff2ffe4de3438b9e891`
- **Artifact count:** 23 (across 13 session commits)
- **Confidence level:** Solid (3/5) — tests pass, upgrade path to Strong (4/5) via clippy addition
- **Summary:** `PROOF_SUMMARY.md` at repo root; detailed: `cycle-5/retrospective.md`

### 15.3 Independently-reproducible

Any auditor with this receipt and the project source at commit `bb230fd9` can recompute `receipt_hash` and confirm byte-for-byte integrity. No trust in the reporter required.

---

## Section 16 — Contact and Stewardship

**Founder & Node0 principal:** Mohamed Beshr (Mumo)
**Entity:** BIZRA Foundation
**Jurisdiction:** Dubai, United Arab Emirates
**Email:** m.beshr@bizra.info
**GitHub:** [github.com/BizraInfo](https://github.com/BizraInfo)

**Stewardship covenant** (from الرسالة): *"If you choose to walk this journey with me, it becomes your responsibility too — you bear its consequences and its truth. This choice is yours alone, and no one else's."*

---

## Section 17 — Closing

> **Close it. Prove it. Reveal it.**
>
> Close doctrine into contracts. Close contracts into runtime. Close runtime into proof. Close proof into reveal.
>
> Dema is where those four closures meet in one operator surface.
> BIZRA is the operating law under which they remain closed.

الحمد لله.

---

### Appendix A — The One Sentence

> **Dema is the sovereign face of a constitutional trust compiler that turns human intent into lawful, receipted, replayable action.**

### Appendix B — Document versioning

- **v0.1** (earlier today, Mumo draft) — preserved at `docs/BIZRA-Handover-v0.1.md` as founding draft
- **v1** (this file) — current operative handover, authorized post-integrator audit
- **Amendments to v1** follow the same protocol as Manifesto amendments: explicit authorization + constitutional-filter audit + diff record in `docs/handover-amendments/`

### Appendix C — Related artifacts

- Trust Compiler Thesis: `docs/bizra-trust-compiler-thesis.md`
- Dema CLI Manifesto v1: `docs/dema-cli-manifesto-v1.md`
- Why Dema Wins (1-pager): `docs/why-dema-wins.md`
- Cycle-5 retrospective: `cycle-5/retrospective.md`
- Proof Summary: `PROOF_SUMMARY.md`
- Gateway + CLI README: `bizra-omega/bizra-cognition-gateway/README.md`

---

*Filed with Ihsan. Every claim traces to shipped code or founding covenant. Every number in the State table is empirically verified. Every constitutional invariant is structurally enforced, not asserted.*

الحمد لله.
