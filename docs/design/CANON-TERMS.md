# BIZRA · CANON-TERMS · v1

> Single source of truth for names, thresholds, and taxonomies used across
> BIZRA / DEMA artifacts: code (Rust + Python), specs (Brain Activation,
> DEMA Charter), pitch materials (North Star, Whitepaper, Manifesto),
> brand materials, and memory anchors.

**Motivating incident**: on 2026-04-19, a drift sweep across eight source
surfaces found ≥4 concurrent name-sets for the Ihsān gate, 2 phase
taxonomies, 3 taglines, and 2 variants of the 5th invariant. No single
artifact was authoritative; every downstream surface had to guess. This
document terminates that guessing.

---

## § 00 — Authority hierarchy

When two sources conflict, the higher-authority source wins. Lower
sources update to match.

1. **Code** — `core/integration/constants.py` (Python SSOT) and
   `bizra-omega/bizra-core/src/lib.rs` (Rust parity). Code is the
   final authority. Any name/value disagreeing with code is drift.
2. **Sealed canon docs** — docs explicitly marked *Sealed* or merged
   to `main` under `docs/design/`. *(None yet, at v1 of this file.)*
3. **Draft canon docs** — issued but unsealed. Current set:
   - DEMA Charter 001 (2026-04-17, Draft · Unsealed)
   - OMNI-SYNTHESIS Whitepaper 002 · v5.0.0-Ω (2026-04-17, Draft · Unsealed)
   - ADK North Star V2 (2026-04-19, INTERNAL)
4. **Pitch / brand materials** — Node0 Manifesto deck, Brand
   Identity v2.0. Consumes canon; never sets it.
5. **Memory anchors** — convenience layer for LLM agents. Updated to
   match whatever higher-authority source settles on.

**Isnad rule**: every term below carries its source. If a surface
references a canonical term without source, that surface is drift.

---

## § 01 — Constitutional invariants

The system enforces **five** active invariants. The count is agreed
across memory, Z.ai sandbox `/api/evaluate-gates`, Brain Activation
Spec v0.1 §8.1, and Whitepaper §07 — **names differ; this is the fix**.

| Canonical name | Meaning | Sources agreeing | Retired variants |
|---|---|---|---|
| `ZANN_ZERO` | Zero interest-based financial instruments | memory `user_mumo`; sandbox gate; Whitepaper §07 | — |
| `RIBA_ZERO` | No usury in any form | memory; sandbox gate; Whitepaper §07 | — |
| `CLAIM_MUST_BIND` | Every claim must have verifiable proof | memory; sandbox gate; Whitepaper §01 "Third Fact" | — |
| `NO_SHADOW_STATE` | Face renders truth; does not invent it. No duplicate state outside kernel. | Charter R-001; sandbox gate; Whitepaper SNR §08 | — |
| `IHSAN_FLOOR` | Minimum excellence threshold binding the kernel gate | **`00_CONSTITUTION/BIZRA_KERNEL_SPEC.md` INV-003** (canonical: "Ethical quality score of any user-facing output must be ≥ 0.95 — GATE HOLD"); `00_CONSTITUTION/BIZRA_KERNEL_PRD.md` (R-005); `specs/urp_genesis/01_urp_genesis_spec.md` (runtime check); sandbox `/api/evaluate-gates`; Brain Activation Spec v0.1 §8.1 | `IHSAN_THRESHOLD` (name), `STRICT_IHSAN_THRESHOLD` (name), `Iscore ≥ 0.95` (expression); `deploy/github-profile/README.md:44` has 0.99 (deploy-facing aspirational, not Spec gate) — see §02 tier mapping |

The legacy memory anchor lists **"Daughter Test"** as the 5th invariant
alongside the first four. This doc resolves that tension by promoting
`NO_SHADOW_STATE` (code-bearing, appears in Charter R-001 and sandbox
gates) to the 5th invariant slot, and reclassifying the Daughter Test
as a **design heuristic** below the invariant tier:

> *Daughter Test* — *"Would you want your daughter subjected to this output?"* — applied during human design review, not as a runtime gate. Retain. Do not code-enforce.

---

## § 02 — Ihsān tier values

`IHSAN_FLOOR` is the *name*; the *value* is context-dependent and
resolved from `core/integration/constants.py` (authoritative):

| Tier | Constant (Python SSOT) | Value | Role |
|---|---|---|---|
| 0.90 | *(CI floor, inferred, not a named constant yet)* | 0.90 | CI pass bar |
| 0.95 | `IHSAN_THRESHOLD` / `UNIFIED_IHSAN_THRESHOLD` | 0.95 | Production default; Rust gate default |
| 0.99 | `STRICT_IHSAN_THRESHOLD` | 0.99 | Claim-bearing strict mode |
| 1.0  | *(implicit, not a constant)* | 1.0 | Runtime perfection (asymptote) |

**Reconciliation of Whitepaper `Iscore ≥ 0.95`**: maps to tier 0.95
(Production). The Whitepaper value is correct; the name "Iscore" is a
synonym for the Ihsān Vector composite score referenced against
`IHSAN_THRESHOLD`.

**Rule for specs and pitch decks**: whenever a threshold is cited,
**name the tier** explicitly (e.g., "0.95 Production" or "0.99 Strict")
to prevent cross-language drift with the Python SSOT. This rule was
previously held only in memory (`feedback_canon_over_skill_tables`);
it is promoted to canon here.

---

## § 03 — Phase discipline (truth labels)

Two taxonomies circulate. **North Star V2 wins**. The Manifesto deck
set is retired:

| Canonical (North Star V2 · slide 6) | Retired (Manifesto slide 7) | When to use |
|---|---|---|
| **VERIFIED** | ~~PROVEN~~ | Shipped, receipt-backed, physically audited. Evidence = receipt or commit SHA. |
| **MEASURED** | ~~VALIDATED~~ | Quantified; methodology visible. Evidence = benchmark name + reproduction path. |
| **DERIVED** | *(no equivalent — new)* | Architected; exists as spec or diagram; no runtime yet. Evidence = `docs/design/...` path. |
| **PLANNED** | **PLANNED** | Roadmap; scoped but not built. Evidence = backlog line / issue. |
| — | ~~DEFERRED~~ | **Retired.** Content previously labeled DEFERRED reclassifies as PLANNED. |

Any pitch deck / brand surface using the retired Manifesto set should
migrate to the canonical set on next edit.

---

## § 04 — Agent topology

Two canonical counts exist, serving different roles:

### 27-agent design catalogue
Whitepaper §06 · *Islamic Masterminds Constellation* · 3 tiers.

**Tier A — Meta-orchestration**
- Adaptive Orchestrator — decomposes intents, selects reasoning modes, targets SNR per task

**Tier B — Domain specialists** *(named so far)*
- Al-Khwarizmi — 96 % SNR, linear algorithmic chains
- Ibn al-Haytham — 97 % SNR, hypothesis-test loops
- Ibn Khaldun — temporal/social graphs

**Tier C — Evidence verification** *(named so far)*
- Imam Bukhari — Isnad chain-of-transmission audit; rejects unverifiable lineage
- Polymath Integrator — dialectical context quarantine

**5 of 27 named.** The remaining 22 slots are reserved. Do not invent names.

### 12-agent runtime mint (per human node)
North Star V2 · slide 4 · **PAT-7 + SAT-5 = 12**.

- **PAT-7** — seven roles instantiated on the user's node. **Individual role names are not declared.** Do not list partial rosters (the Node0 Manifesto slide 5 listed "Architect/Executor/Archivist" — that list has no source and must be removed or sourced).
- **SAT-5** — five roles instantiated inside the Universal Resource Pool (URP). Same rule: shape before names.

### Relationship between 27 and 12

**Inferred, not yet sealed**: the 12-agent runtime mint is an
instantiation of selected roles drawn from the 27-agent catalogue.

Any surface that cites both numbers **must** explain the relationship
explicitly in one sentence, or drop one of the two. Until a canonical
doc seals this, the relationship is marked *DERIVED* (§03).

---

## § 05 — Receipt kinds

**Authoritative source**: `bizra-omega/bizra-cognition/src/receipts.rs::ReceiptKind`
(Rust is the final authority for on-chain receipts).

| Byte | Kind | Purpose |
|---|---|---|
| 0x00 | `Genesis` | Chain origin |
| 0x10 | `CognitionBoot` | Cognition substrate activation |
| 0x20 | `Myelination` | Reinforce pathway |
| 0x21 | `Demyelination` | Weaken pathway |
| 0x30 | `ReasoningSession` | One cognition round (prompt-hash, response-hash, provenance) |
| 0x40 | `GovernanceDecision` | Kernel verdict over a claim |
| 0x50 | `NodeLifecycle` | Node state transition |
| 0x60 | `Manifest` | Cycle-7 G1 manifest artifact |
| 0x61 | `PrincipalActivation` | Cycle-7 G2 principal binding |
| 0x70 | `MissionExecuted` | Cycle-7 G5 operator mission |
| 0xF0 | `DegradedPath` | Fail-closed record |

**"CognitionRound"** from Brain Activation Spec v0.1 §11 resolves to a
`ReasoningSession` payload (kind 0x30). See PR #30 for the payload
struct (`bizra-omega/bizra-cognition/src/cognition_round.rs::ReasoningSessionPayload`)
which also carries the `ProvenanceDescriptor` mirror shared with
`bizra-installer::install_receipt::ProvenanceDescriptor`.

---

## § 06 — Provider identity (brain HAL)

**Authoritative source**: `bizra-omega/bizra-installer/src/install_receipt.rs::ProviderIdentity`
and schema-parity mirror in
`bizra-omega/bizra-cognition/src/cognition_round.rs::ProviderIdentity`
(PR #30).

```rust
pub enum ProviderIdentity {
    CoreNone,
    LocalModel  { weights_path: String },
    LocalServer { endpoint: String, vendor: String },
    RemoteApi   { vendor: String },
}
```

Shape mirrors Brain Activation Spec v0.1 §3.1 HAL enum. Any new provider
class requires updating **both crates in the same commit** and the JSON
shape parity tests. Parity is enforced by
`provenance_json_shape_is_stable` in each crate.

**Existing `bizra-inference` crate** (`InferenceBackend` trait with
`LMStudioBackend`, `OllamaBackend`, `LlamaCpp`) is the runtime layer
that **implements** provider access. `ProviderIdentity` is the *identity
carried on the receipt*, not the runtime connection. Keep them distinct.

---

## § 07 — Tag-line set

Use these canonical phrases in preference to improvisation:

| Phrase | Source | Role |
|---|---|---|
| *"BIZRA reveals truth, never simulates it."* | North Star V2 · L-001 | Constitutional law |
| *"One face. One law. One sovereign jurisdiction."* | North Star V2 · closing | Closing anchor |
| *"Dema is the one assistant."* | North Star V2 · slide 3 | Product thesis |
| *"Twelve agents per human, seven yours, five ours."* | North Star V2 · slide 4 | Hidden organism |
| *"Honest rejection always outranks fake success."* | North Star V2 · L-001 body | Engineering discipline |
| *"Code is authoritative, not rhetoric."* | North Star V2 · slide 7 footer | Authority rule |
| *"DDAGI · Sovereign Survivor."* | Whitepaper · designation | System positioning |
| *"Third Fact."* | Whitepaper §01 | Truth doctrine |

**Retired / do not use**: *"Verifiable Constitutional Intelligence"*
(Node0 Manifesto only; not in North Star, Charter, or Whitepaper).
Either promote with an explicit canon update on this file, or stop using.

### Skill-family suffix (naming convention)

| Suffix | Use | Sources |
|---|---|---|
| `(AI Cowork)` | Operator-facing skill name in SKILL.md files | `core/skills/smart_file_manager.py`; `.claude/skills/sovereign/file-management.md`, `browser-control.md`; `bizra-omega/docs/skills/file-management/SKILL.md`, `browser-management/SKILL.md`; `DEFAULT_SKILL_REGISTRY.md` |

Form: `<Skill Name> (AI Cowork)` — e.g., *"Smart File Management (AI Cowork)"*,
*"Autonomous Browser Control (AI Cowork)"*. Not a tagline; a skill-class
marker distinguishing operator-facing composable skills from internal
infrastructure. Any new skill that wants this suffix must be registered
in `bizra-omega/docs/skills/DEFAULT_SKILL_REGISTRY.md`.

---

## § 08 — Dual-token model

Source: North Star V2 · slide 7; Whitepaper §07.

| Token | Class | Transferable | Earned how |
|---|---|---|---|
| `SEED` | Utility / consumption | **Yes** (price-discovered on open market) | Pays compute, storage, routed agent work |
| `BLOOM` | Governance | **No — soulbound** | Verified impact; cannot be transferred or purchased |

**Status**: PLANNED (§03). Not deployed. Not trading. Activation
condition per founder stance: *"when economic finality is
mathematically proven."*

The Manifesto deck phrasing *"We refuse to tokenize…"* is corrected in
this canon to *"Tokenomics deferred to PLANNED tier; activates
post-finality."* Not opposition — sequencing.

---

## § 09 — Phase tier evidence expectations

A deck/doc row citing a capability at phase tier X must carry the
expected evidence:

| Tier | Evidence required |
|---|---|
| VERIFIED | Receipt, commit SHA, audit URL, or reproducible command |
| MEASURED | Benchmark name + reproduction path or data source |
| DERIVED | `docs/design/...` path or architecture diagram |
| PLANNED | Backlog / issue / target date (if any) |

Any row without this evidence is drift back into pre-canon marketing.

---

## § 10 — Change protocol

A term in this file may change only when:

1. **A higher-authority source disagrees** (code, sealed canon)
   → update this file. Commit message references the disagreeing source.
2. **A lower-authority source disagrees** (pitch, brand, memory)
   → update the lower source. This file is unchanged.
3. **A canonical doc seals** (Draft · Unsealed → merged into `docs/design/`)
   → this file absorbs its terms. Seal commit SHA recorded here.

Any commit touching this file should include a brief *Why* block
naming the drift source and the fix applied. Drive-by edits without a
provenance tag will be reverted.

---

## § 11 — Seal log

| Date | Doc / term | Seal commit | Note |
|---|---|---|---|
| 2026-04-20 | CANON-TERMS v1 initial | *this commit* | First reconciliation of 4 drift axes |
| 2026-04-20 | Sweep evidence amendment | *this commit* | §01 IHSAN_FLOOR grounded in 00_CONSTITUTION/BIZRA_KERNEL_SPEC.md INV-003; §07 `(AI Cowork)` skill-suffix promoted; §12 added |

*(All three referenced canon docs — DEMA Charter 001, OMNI-SYNTHESIS
Whitepaper 002, North Star V2 — remain **Draft · Unsealed** as of this
file's creation. When any of them seals, record the seal here and
update the "Sources agreeing" columns.)*

---

## § 12 — Active drift sources detected in tree (2026-04-20 sweep)

Output of a repo-wide grep sweep against the retired phrase set. These
are current-on-main content that should be reconciled on next edit:

### Phase-label drift (uses retired `PROVEN`/`VALIDATED`/`DEFERRED`)

- `specs/node0-mission-os-v1/04_phase3_elevation.md:139,141` — assigns
  `label = "PROVEN"` / `label = "VALIDATED"` in a code snippet. Retire
  per §03 when this spec is next touched.
- Retrospectives (`cycle-3/hadd.md`, `cycle-4/retrospective.md`,
  `archive/downloads-files-7-2026-04-17/cycle-3-retrospective.md`,
  `…/cycle-5-phase-7-retrospective.md`) **already document the
  "PROVEN overclaim" pattern** and its correction to "TESTED
  (candidate)". These retrospectives are the *evidence* that
  CANON-TERMS §03 discipline was learned from experience. Leave them;
  they are history, not drift.

### Capability-vs-copy overreach (L-001 risk)

`core/skills/smart_file_manager.py` and its doc mirrors
(`docs/SMART_FILE_MANAGEMENT.md`, `bizra-omega/docs/skills/file-management/SKILL.md`,
`.claude/skills/sovereign/file-management.md`) describe
`auto-classify`, `batch-rename`, `merge` operations. Per §03 phase
taxonomy:

- **Listing + digest + sealed receipt**: `VERIFIED` (Cycle-7 G5 `dema organize` primitive, read-only)
- **Content-aware classification**: `DERIVED` (requires Starter Brain per Brain Activation Spec v0.1)
- **Filesystem mutation (rename, merge)**: `PLANNED` (no admissibility gate for mutations yet)

External-facing copy that claims any of these as current reality must
carry its tier tag per §03 evidence expectations. The operator-facing
skill SKILL.md files should add a **"Phase tier"** block at the top
before external use.

### Root-level vendor leakage

- `SKILL.md` (root) — duplicates `.claude/skills/sovereign/browser-control.md`
- `SKILL (1).md` (root) — duplicates `.claude/skills/sovereign/file-management.md`

These appear to be Z.ai sandbox artifacts shadowing the canonical
copies. Recommend `git rm` on next hygiene arc after confirming no
caller references them directly.

### IHSAN_FLOOR value — not drift, documented tier variance

`deploy/github-profile/README.md:44` says *"IHSAN_FLOOR → Excellence
is the minimum. 0.99 threshold."* This is **not** drift against the
Spec's `≥ 0.95` — the deploy-facing README is aspirational
(Strict tier per §02), while the Spec gate is Production (0.95 tier).
If external audiences see both, they need the tier tag per §02 rule.
Recommend updating the deploy README to read *"IHSAN_FLOOR → Excellence is the minimum. Production tier gate: ≥ 0.95. Claim-bearing strict: ≥ 0.99."*

---

**Version**: v1
**Issued**: 2026-04-20
**Status**: Draft · Unsealed (lives in `docs/design/`; not a code gate yet)
**Maintainers**: Core maintainers
**Expected update cadence**: whenever a referenced canonical doc seals or code constants shift
