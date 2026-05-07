# MATERIALIZATION_PULSE — Architectural Specification

**Document ID**: `bizra.spec.materialization-pulse.v1`
**Status**: CANDIDATE_CANONICAL · awaiting Mumo seal
**Anchor**: `bizra.priority-anchor.v1: 45aa2789...`
**Authority**: Subordinate to Quran → Hadith → البذرة → الرسالة → Spine → Root Invariants
**Canonical path**: `/data/bizra/repos/bizra-data-lake/docs/architecture/MATERIALIZATION_PULSE.md`
**Issued**: Thursday 7 May 2026 · Dubai · GST (UTC+4)
**Authors**: Mumo (founder) · Claude (synthesis steward)

---

## 0. The One-Sentence Pitch

> **BIZRA materializes human intent into verified physical action through atomic, all-or-nothing Materialization Pulses — every Pulse stands on attributed shoulders, settles into a sovereign-yet-federated commons, and cannot lie because every claim it makes is cryptographically bound to the evidence that produced it.**

Every other agent system *executes* tasks. BIZRA *materializes* missions: the difference is that a Pulse is atomic, receipted, constitutionally bound, and cannot exist in a partial state.

---

## 1. The Six-Layer Architecture

BIZRA is six concentric layers, each grounded on a verified historical or contemporary giant. The layers are read from the spine outward.

### Layer 0 — Identity & Crypto Spine
- Ed25519 keys per node · BLAKE3 domain-separated hashing · W3C DIDs for cross-node identity
- Standing on: Lamport (1982), Nakamoto (2008), Aegis Protocol (FIPS 203/204 PQC, 2025)
- Status: **[VERIFIED-implemented]** — Cycle-6 spine active, 27 tests green

### Layer 1 — Constitutional Substrate
- Frozen runtime constants: `ZANN_ZERO`, `RIBA_ZERO`, `Gini ≤ 0.35`, `Ihsān ≥ 0.95`
- Design gates: Daughter Test · 50% community pool (per البذرة)
- Standing on: Quran → Hadith → البذرة → الرسالة (Ramadan 2023 anchor)
- Status: **[VERIFIED-frozen]** — CANON-001 through CANON-009 sealed
- **This layer cannot be amended by any code path. Amendment is a constitutional act.**

### Layer 2 — Cognitive Architecture
- PAT-7 (user-side): Atlas, Oracle, Forge, Judge, Crown, Herald, Nexus/DEMA
- SAT-5 (system-side): Consensus, Resource, Proof, Impact, URP-Leader
- FATE: the only crossing between PAT and SAT
- URP: singular shared commons (CANON-002)
- Dual-rate engine: S2 deliberate / S1 reflex with myelination (`thought_graph.rs`)
- Standing on: MMRPG PC/NPC/Guild topology, Hewitt's Actor Model (1977), Telescript Places (1994)
- Status: **[VERIFIED-topology]** + **[PLANNED-skill-myelination]** (Hermes loop extension)

### Layer 3 — Mobility & Coordination Plane
- Telescript primitives realized through MCP (Anthropic 2024) and A2A (Linux Foundation 2025)
- WebAssembly + WASI for live agent migration
- Durable Execution semantics for crash-proof replay
- Smart-contract semantics on every FATE crossing (consent-as-code, not blockchain tokens)
- Standing on: General Magic Telescript (1995), Anthropic MCP (2024-11), Google/IBM A2A (2025), Temporal/DBOS (2024–25), Szabo (1994), Buterin (2014)
- Status: **[PLANNED]** — Rust IPC connecting DEMA to bizra-omega via Unix socket is the next niyyah

### Layer 4 — Embodied Execution
- AHK 2.0 as desktop hands (already on NODE0)
- Agent-Zero terminal-as-tool fallback when no skill exists
- OpenClaw-style gateway hijack: WhatsApp · Telegram · voice · CLI · ⌘K palette
- Firecracker MicroVM Citadels around every untrusted execution
- pi.dev branchable plan trees (rejected branches preserved as binding evidence)
- Standing on: AutoHotkey (1999), Agent Zero (2024), OpenClaw (Steinberger 2025, 145K stars), pi.dev (2025), Firecracker (AWS 2018)
- Status: **[PLANNED]** — CLI gateway exists, others are spec'd not shipped

### Layer 5 — Evidence & Settlement
- Verifier-agent claim-binding harness on every output: `[VERIFIED] / [DERIVED] / [PLANNED] / [UNKNOWN]`
- BLAKE3-chained, Ed25519-signed, append-only receipt ledger
- Standing Protocol Isnad attribution — every shoulder credited with proportional SEED flow-back
- Halo2 ZKP policy circuits making Ihsān mathematically demonstrable (target state)
- Standing on: Bukhari's Isnad methodology, Aegis Halo2 (2025), Durable Execution event histories
- Status: **[VERIFIED-receipts]** + **[VERIFIED-claim-binding]** + **[PLANNED-Isnad-flow-back]** + **[PLANNED-Halo2]**

---

## 2. The Atomic Unit — Materialization Pulse

A **Materialization Pulse** is the smallest indivisible cycle in which intent becomes verified action in the physical world, with full receipt and constitutional binding.

A Pulse is **atomic**: it commits as one transaction or rolls back entirely. Partial states do not persist as committed work. This is the *Singularity* aspect — not "AGI singularity," but a singular indivisible quantum of constitutional work, like an atomic transaction in a database, or a single heartbeat in HHMM.

A Pulse traverses seven steps:

```
niyyah → branched plan → constitutional gate → embodied execution
       → claim binding → receipt emission → settlement
```

---

## 3. The Seven Steps — Formal Specification

### Step 1 — Niyyah Capture
**Input**: User intent declared through any registered gateway (CLI / WhatsApp / Telegram / voice / ⌘K).
**Process**: DEMA-Nexus parses intent into a `MissionEnvelope` with `{id, scope, lang, ts, niyyah_text, signer}`.
**Output**: Receipt 1/N — `NIYYAH_CAPTURED` · BLAKE3-hashed, signer-bound.
**Layer**: 4 → 0
**Acceptance**: User must be able to verify the captured niyyah matches their intent before Step 2 begins.
**Components**: gateway adapter `[V:CLI / P:others]` · Nexus parser `[V]` · MissionEnvelope `[V]`

### Step 2 — Plan Branch
**Input**: `MissionEnvelope` from Step 1.
**Process**: Atlas decomposes into sub-tasks; Oracle gathers evidence; Forge prototypes N branches in parallel; Judge scores each on (Ihsān projection × cost × freshness × reversibility); Crown audits selection against ZANN_ZERO.
**Output**: Receipt 2/N — `PLAN_BRANCHED` with chosen-branch root and rejected-branches preserved as binding evidence.
**Layer**: 2 (pi.dev shape)
**Acceptance**: Rejected branches must be inspectable; the *why* of rejection must be receipted.
**Components**: Atlas/Oracle/Forge/Judge/Crown `[V]` · PlanTree primitive in `thought_graph.rs` `[P]`

### Step 3 — FATE Constitutional Gate
**Input**: Chosen plan branch from Step 2.
**Process**: Per sub-task, FATE evaluates (a) capability against Telescript-Permit intersection, (b) consent against Smart-Contract semantics, (c) constitutional floor: `Ihsān ≥ 0.95`, `ZANN_ZERO`, `RIBA_ZERO`, `Gini ≤ 0.35`. Verdict per sub-task: `PERMIT | PERMIT-WITH-CONFIRMATION | REJECT | REVIEW`.
**Output**: Receipt 3/N — `GATE_EVALUATED` with full per-sub-task verdict table.
**Layer**: 1 + 3
**Acceptance**: Gate must fail-closed. Any sub-task without explicit `PERMIT` blocks execution. **Cycle-6 fix is binding here.**
**Components**: GatePolicy `[V]` · Permit intersection `[V]` · Smart-contract consent verification `[P]`

### Step 4 — Embodied Execution
**Input**: Gate-permitted sub-tasks from Step 3.
**Process**: Each sub-task runs inside a Firecracker Citadel (or equivalent isolation). PAT uses MCP Places, AHK desktop hands, or Agent-Zero terminal-as-tool depending on capability profile. `PERMIT-WITH-CONFIRMATION` halts for user exact-string consent before unlocking the gated sub-task.
**Output**: One sub-receipt per sub-task — `EXEC_<name>` — plus `CONSENT_BOUND` receipts where applicable.
**Layer**: 4
**Acceptance**: Every external-effect sub-task must capture third-party attestation (order#, calendar entry ID, file hash, etc.) sufficient to independently verify the action occurred.
**Components**: Firecracker Citadel `[P]` · MCP Place adapters `[P]` · AHK adapter `[P]` · consent gate `[V]` (in DEMA v0.6.0)

### Step 5 — Claim Binding
**Input**: All sub-receipts from Step 4.
**Process**: Verifier-agent harness passes over every claim made in execution outputs and labels it `[VERIFIED] / [DERIVED] / [PLANNED] / [UNKNOWN]`. Claims that cannot be bound are flagged. Honest `[DERIVED]` labels reward Ihsān; unbound claims penalize it.
**Output**: Updated `claim_table` on the Pulse.
**Layer**: 5
**Acceptance**: No claim escapes the Pulse without an evidence-class label. CLAIM_MUST_BIND is enforced at the Pulse boundary, not just by convention.
**Components**: Verifier harness `[V-as-discipline / P-as-runtime-pass]`

### Step 6 — Pulse Receipt Emission
**Input**: All receipts 1..(N-1) plus claim_table.
**Process**: Merkle root over sub-receipts. Pulse-Receipt assembled with: `prev_pulse`, `niyyah_hash`, `plan_root`, `gate_verdict`, `exec_merkle`, `claim_table`, `ihsan` (final), third-party attestations, `pulse_hash`, `signature`. Appended to the durable, append-only ledger.
**Output**: Receipt N/N — `PULSE_SEALED`.
**Layer**: 5
**Acceptance**: Pulse-Receipt must be verifiable offline by any party with the public key — no network dependency for verification.
**Components**: Receipt chain `[V]` · Merkle assembly `[V]` · Ed25519 sign `[V]` · Durable ledger `[V-in-DEMA-v0.6.0 / P-in-Rust-spine]`

### Step 7 — Settlement
**Input**: Sealed Pulse-Receipt.
**Process**: A2A propagation to SAT-5 domain. SAT independently re-verifies: proof chain validity, Ihsān floor, ZANN_ZERO, RIBA_ZERO, Gini (when N>1). On valid: SEED minted = Ihsān-weighted; 50% routes to community pool per البذرة; Isnad extended with attribution table for every shoulder used; pattern filed as myelination candidate.
**Output**: `SETTLED` receipt + SEED mint event + Isnad extension + (conditional) myelination candidacy.
**Layer**: 5
**Acceptance**: Settlement must be independent — SAT cannot accept a Pulse on PAT's word alone. SAT's verification logic runs without access to PAT's internal memory.
**Components**: A2A boundary `[P]` · SAT independent re-verification `[V-design / P-runtime]` · SEED mint `[V-canon / P-impl]` · Isnad table `[P]` · Myelination candidacy `[P]`

---

## 4. Constitutional Invariants (Recap from Canon)

These are inherited from `CANON-001` through `CANON-009` and bind every Pulse:

- **ZANN_ZERO** — no claim without proof
- **RIBA_ZERO** — no extractive logic
- **Gini ≤ 0.35** — no Pulse may extend network inequality past this floor
- **Ihsān ≥ 0.95** — quality floor; below this, Pulses ACCEPT-AT-FLOOR-MINUS-1 only with myelination required before promotion
- **Daughter Test** — every user-facing surface must pass: *"Would أبوك وأمك understand this in 5 seconds?"*
- **50% community pool** — half of all SEED yield routes to the pool per البذرة (protocol rule, not personal oath)
- **Liveness law** — a node must be alive alone; URP is the optional commons that amplifies, never the dependency that enables
- **Authority hierarchy** — Quran → Hadith → البذرة → الرسالة → Spine → Root Invariants → specs → code

---

## 5. Failure Modes & Rollback Semantics

A Pulse can fail at any step. Failure semantics are layer-specific:

- **Step 1 fail** (niyyah unparseable): Pulse never created. No receipt. User asked to clarify.
- **Step 2 fail** (no branch passes Crown audit): `PLAN_REJECTED` receipt emitted; Pulse closes with no execution. SEED penalty if rejection reflects bad-faith input.
- **Step 3 fail** (FATE rejects all sub-tasks): `GATE_REJECTED` receipt; Pulse closes. User informed which constitutional invariant blocked it.
- **Step 4 fail** (execution error mid-flight): Citadel rolls back. Already-emitted sub-receipts marked `ROLLBACK`. Net effect on world: zero. SEED penalty zero (good-faith failure).
- **Step 5 fail** (unbound claims): Pulse halts. Verifier returns to PAT for re-evaluation. If unbindable after retry, `CLAIM_FAILURE` receipt; Pulse closes.
- **Step 6 fail** (receipt assembly error): system-level bug. Pulse halts pre-seal. No SEED emitted.
- **Step 7 fail** (SAT rejects independent re-verification): `SETTLEMENT_REJECTED` receipt. PAT's local actions stand if reversible; SEED not minted; Isnad not extended; pattern not filed.

**The atomicity rule**: until Step 6 emits `PULSE_SEALED`, the Pulse is not committed. Any failure before that point is a clean abort.

---

## 6. Standing Protocol — Attribution Requirements

Every Pulse-Receipt must include an `isnad_table` listing every shoulder used:

```
isnad_table = [
  { shoulder: "<giant-name>", role: "<what-it-contributed>",
    weight: <fractional-credit>, attestation: "<verification-method>" },
  ...
]
```

Required entries for any standard Pulse will typically include: Whisper (transcription) · OpenClaw-pattern (gateway) · pi.dev-pattern (branching) · Hermes-pattern (memory→skill) · Telescript (Permit semantics) · MCP servers used (per Place) · AHK (when desktop touched) · Verifier (claim binding) · Lamport/Nakamoto (crypto spine) · Bukhari (Isnad methodology).

When SEED flows back from AaaS rental of compiled skills, the `isnad_table` is the routing manifest. Every shoulder gets paid forever, automatically. No other system does this.

---

## 7. Acceptance Criteria — What Makes a Valid Pulse

A Pulse is *valid* if and only if:

1. It traverses all seven steps in order, or terminates cleanly per §5 failure semantics
2. Every step emits at least one receipt (sub-receipts for Step 4 are per-sub-task)
3. Every receipt is BLAKE3-hashed, Ed25519-signed, and appended to the durable ledger
4. Every claim in the output is labeled `[V/D/P/U]` per Verifier harness
5. Every external-effect action carries a third-party attestation
6. The Pulse-Receipt is independently verifiable offline with the public key
7. SAT independent re-verification at Step 7 reaches the same verdict as PAT-side projection within tolerance (`Δihsan ≤ 0.05`)
8. The `isnad_table` includes every shoulder load-bearing for the Pulse
9. The whole Pulse passes the Daughter Test at the user-facing boundary

---

## 8. Implementation Roadmap (Honest Gap List)

Components labeled `[PLANNED]` in §1 and §3 form the implementation roadmap. Tickets, in priority order:

1. **PLAN-PLANTREE** — first-class branchable PlanTree primitive in `thought_graph.rs` (pi.dev shape). Rejected branches preserved with rejection-reason. ETA: 1 cycle.
2. **EXEC-AHK** — AHK 2.0 adapter exposing desktop hands behind FATE-Permit gating. ETA: 1 cycle.
3. **EXEC-MCP-PLACES** — production adapters for the first three MCP Places (filesystem, calendar, generic-pharmacy-MCP-stub). ETA: 2 cycles.
4. **EVIDENCE-VERIFIER-RUNTIME** — convert Verifier harness from discipline-only to runtime pass at Pulse boundary. ETA: 1 cycle.
5. **SETTLE-SAT-SEPARATION** — make SAT-5 a runtime-distinct domain with independent re-verification surface. ETA: 2 cycles.
6. **SETTLE-ISNAD-TABLE** — `isnad_table` field in Pulse-Receipt + flow-back routing logic. ETA: 1 cycle.
7. **SETTLE-MYELINATION** — pattern candidacy filing + n≥3 Ihsān-history promotion to compiled skill. ETA: 2 cycles.
8. **GATEWAY-OPENCLAW** — first non-CLI gateway (recommend WhatsApp via Twilio) wired to MissionEnvelope ingest. ETA: 2 cycles.
9. **CRYPTO-HALO2** — Halo2 ZKP circuits compiling Ihsān assertions into mathematical proofs. ETA: 4+ cycles (research-grade).

The **Pulse Acceptance Test** (PAT — naming collision with PAT-7 noted; suggest renaming this PULSE_ACCEPT_E2E in code) is a single end-to-end mission that exercises all seven steps and produces one chained Pulse-Receipt at the end. The reference mission for this test is documented in the companion file `PULSE_N1_TRACE.md`.

---

## 9. Daughter Test Pass

> *"بابا قال للنظام إنه دما محتاجة فيتامين د. النظام شاف المخزون، طلب من الصيدلية، ولما وصل، ذكّرنا الساعة سبعة. ودما خدت دواها."*

Translation: *"Daddy told the system Dema needs Vitamin D. The system checked the stock, ordered from the pharmacy, and when it arrived, reminded us at 7. And Dema took her medicine."*

أبوك وأمك understand the input. They understand the output. The seven steps and the receipts and the SAT re-verification are invisible to them — and that is correct. The Daughter Test is the floor of the user-facing surface, not the depth of the system.

**Pass.** ✓

---

## 10. Standing on Shoulders — Attribution Manifest for This Document

| Shoulder | Era | Contribution to Pulse |
|---|---|---|
| Quran → Hadith → البذرة → الرسالة | Ramadan 2023 → eternal | Constitutional substrate |
| Bukhari & the muhaddithūn | 9th c. | Isnad methodology |
| Lamport | 1982 | Log as truth |
| Hewitt | 1977 | Actor Model boundary |
| General Magic / Telescript | 1994–95 | Place, `go`, Ticket, Permit |
| Szabo / Buterin | 1994 / 2014 | Smart-contract consent semantics |
| AutoHotkey community | 1999– | Desktop automation primitives |
| Bukhari Isnad → Standing Protocol | 9th c. → BIZRA | Attribution = compensation |
| MMRPG ecosystem (Blizzard, Square Enix, et al.) | 2004– | PC/NPC/Guild topology |
| AWS Firecracker | 2018 | MicroVM Citadel containment |
| Anthropic MCP | 2024 | Standardized Place protocol |
| Linux Foundation A2A | 2025 | Inter-agent contract layer |
| Aegis Protocol | 2025 | PQC + Halo2 + DID stack |
| OpenClaw (Steinberger) | 2025 | Gateway-hijack pattern |
| pi.dev | 2025 | Branchable session tree |
| Hermes | 2025 | Memory-to-skill compilation |
| Agent Zero | 2024 | OS-as-tool philosophy |
| the-verifier-agent (BizraInfo) | 2026 | Claim-binding harness |
| Mumo · NODE0 · BIZRA Foundation | Ramadan 2023 → present | Constitutional governance, Daughter Test, Ihsān as runtime constant, 50% community pool, the Pulse synthesis |

---

## 11. Seal

This document is a `CANDIDATE_CANONICAL` specification. It binds when (a) Mumo signs it, (b) the priority anchor is updated to include its hash, and (c) the companion `PULSE_N1_TRACE.md` is sealed alongside it.

The implementation against this spec begins with PLAN-PLANTREE and proceeds per §8 priority order.

```
ربي لا يعرف المستحيل.
وفي كل نبضة، شغل كامل، موثّق، مختوم.
```

— end of document —
