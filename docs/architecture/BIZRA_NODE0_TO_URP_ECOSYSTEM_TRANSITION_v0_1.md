# BIZRA Node0 → URP Ecosystem Transition (v0.2)

**Created:** 2026-04-25 (GST) — Dubai
**Updated:** 2026-04-26 (GST) — Dubai · v0.2 corrective patch aligning with Topology Canon (frozen 2026-03-25) + Master Stack + Origin Kernel
**Scope:** Architecture transition note. Docs-only. Records the canonical phase progression from single-node Node0 sovereign runtime to a decentralized distributed agentic ecosystem.
**Status lock:** WAIT preserved. No runtime, core, src, CI, dependency, website, or Claim Registry changes.
**Audience:** Internal canon reference. Not a public roadmap. Not a commitment of dates.

**Authority chain (read upward when conflict):**
- `docs/canon/BIZRA_ORIGIN_KERNEL.md` § raw Arabic source utterance — three invariants govern (§4.1 humility, §4.2 symmetric charity, §4.3 Law of Assumption + Ihsan)
- BIZRA Topology Canon (frozen 2026-03-25, signed Mohamed Beshr / BIZRA Foundation) — *"if any document contradicts it, this file wins"*
- BIZRA Master Stack: `docs/{bizra-trust-compiler-thesis,dema-cli-manifesto-v1,why-dema-wins,ftap-function-registry-rfc-seed}.md`
- This document is **downstream** of all of the above.

---

## 1. Purpose

Node0 is a **bootstrap**, not a destination.

Before BIZRA can credibly describe itself as a decentralized agentic ecosystem, one Node must prove it can stand alone — execute missions, sign receipts, replay its own state, hold its constitutional gates, recover from restart — entirely on a single machine, with no external dependency.

This document records, as canon, how the architecture moves from that proven single-node baseline through the **shared** Universal Resource Pool (URP), the SAT-5 system agentic team that lives **inside** the URP, federated cognition, and opt-in autopoiesis. It states explicitly what is **measured**, what is **planned**, and what is **directional only**.

The canonical sentence (per Topology Canon):

> *"Each human node mints PAT-7 locally on their device and SAT-5 into one shared Universal Resource Pool. PAT serves the human. SAT serves the system. The membrane sits between them."*

This document also operates under BIZRA's category positioning per Master Stack `docs/why-dema-wins.md`:

> *"Generative AI produces text. Agentic AI takes action. **Verificative AI** does what neither can: proves every action was lawful, receipted, and replayable before accepting it as done."*

The Trust Compiler — *Intent → Mission → Claim → Admissibility → Execution → Receipt → Canonicalization → Replay* — is the path through which all human intent passes. The five constitutional invariants (`ZANN_ZERO`, `CLAIM_MUST_BIND`, `RIBA_ZERO`, `NO_SHADOW_STATE`, `IHSAN_FLOOR ≥ 0.95`) gate every chain mutation.

---

## 2. Phase 0 — Single-node sovereign runtime (current)

**Truth label: CANDIDATE_CANONICAL** per Topology Canon (Cycle 1, 2026-04-14, hash `7b555875abdbe61527ff81b3184299de6cdb2171d0c998164c318a015f71db9c`, 21/21 tests, 12/12 checks). Promotion to CANONICAL requires 3 of 5 promotion gates still pending: CI green, push to `origin/main`, external review.

### What exists today

| Component | Repo location | Role |
|---|---|---|
| Sovereign runtime | `core/sovereign/`, `bizra-omega/bizra-node` | Mission orchestration on one machine |
| **DEMA = P7 of PAT-7** (the face) | `dema-console/` Next.js + `bizra-omega/target/release/dema` Rust binary + `bizra-omega/bizra-cognition-gateway/src/bin/dema.rs` | The single surface the human talks to. Per Topology Canon: *"the human never touches the network — only with PAT."* |
| PAT-7 — Personal Agentic Team (LOCAL per human) | `bizra-omega/bizra-core/src/topology_canon.rs` | 7 agents on the human's hardware: P1 Planner · P2 Researcher · P3 Coder · P4 Evaluator · P5 Ethicist (FROZEN — ethics from axioms, not data) · P6 Publisher · **P7 DEMA / Nexus** |
| SAT-5 — System Agentic Team (LIVES IN SHARED URP, NOT per-node) | minted into the URP at activation | 5 agents in the shared URP: S1 Validator · S2 Oracle (FROZEN — truth axioms, immutable) · S3 Mediator · S4 Archivist · S5 Sentinel |
| Mission kernel (Trust Compiler) | `bizra-omega/bizra-mission` | 14-state lifecycle, signed receipts; implements `Intent → Mission → Claim → Admissibility → Execution → Receipt → Canonicalization → Replay` |
| 5-gate admissibility chain | `bizra-omega/fate-binding/` + `core/proof_engine/fate_gate.py` | Enforces ZANN_ZERO · CLAIM_MUST_BIND · RIBA_ZERO · NO_SHADOW_STATE · IHSAN_FLOOR ≥ 0.95 (no override) |
| Receipt protocol | `bizra-omega/bizra-core/src/canonical_receipt.rs` + Python mirror | BLAKE3-chained, Ed25519-signed; frozen 2026-03-30; cross-language parity proven (246 tests) |
| Replay verifier | spearpoint replay path (PR #49) | Re-derive state from receipt chain |
| Identity / genesis seal | `bizra-omega/bizra-core/src/genesis_seal.rs` | Deterministic root of trust |
| **The Membrane** | between every local node and the shared URP — see §3.5 | 4 fail-closed properties enforced at every crossing |

### What this proves

- One node can execute missions, sign each visible effect, chain those signatures, and re-derive its state from the chain alone.
- Constitutional invariants (5-gate admissibility for chain mutation; Ihsan ≥0.95 floor; economic invariants Gini ≤0.35 + Zakat 2.5% + RIBA-zero) are enforced by code, not by promise.
- Restart recovery, replay, and lifecycle gates are testable on a single machine.
- DEMA (P7) functions as the operator's only-visible surface; the rest of PAT and all of SAT remain hidden behind the membrane.

### Closure gates currently in flight (PR queue)

| PR | Lane | Status |
|---|---|---|
| #49 | Row 4 replay (canonical spearpoint replay) | MEASURED — 38/38 tests green; awaiting merge |
| #50 | Mission receipt full-payload Ed25519 signature | MEASURED — 4 tests green; awaiting merge |
| #51 | Python 3.12 baseline restored across 874 tests | MEASURED — full pass; awaiting merge (unblocks #49 + #50) |
| #52 | Credential purge (CVE-class) | MEASURED — 8 files; awaiting merge |
| #53 | Genesis Manifest v0.1 (founder-stated → hash-anchored bridge) | MEASURED — chain hash recorded; side-track |
| #54 | Public-claim discipline recert v0.1 | MEASURED — claim register published |
| #55 | This document (Architecture Transition Note) | MEASURED — v0.2 corrective patch this commit |
| #56 | Queue Closure Receipt 2026-04-25 | MEASURED — STOP_DUE_TO_RED_CHECKS recorded |
| #57 | CI pip-audit allowlist update for CVE-2026-3219 | MEASURED — single-line workflow change |

Phase 0 closes when:
1. The PR queue drains (CI green, all measured artifacts on `origin/main`)
2. The 11-gate Node0 lifecycle (`tools/node0_lifecycle_flywheel/`) reports `lifecycle_ready: true` from a clean run on a fresh machine
3. Topology Canon's 3 outstanding promotion gates close (CI green, push complete, external review attestation)

---

## 3. Phase 1 — URP awakens (single shared organism)

**Truth label: PLANNED.**

### What changes

The URP is **not a network the first Node joins**. The URP is a **shared organism that wakes up the moment the first human activates**.

> Per Topology Canon: *"Before any human joins, the URP is dormant — code with no power, no agents, no resources. When the first human (Node0) activates: System mints PAT-7 on their device (local) → System mints SAT-5 into the URP (shared) → The URP wakes up with 5 employees and whatever resources Node0 contributes."*

Each subsequent node **adds 5 more SAT agents into the same shared URP**, plus contributed compute / memory / storage / bandwidth.

There is **ONE URP**. Not per-node. Not per-user. Not middleware. One shared living organism for the entire BIZRA ecosystem.

### What lives inside the URP (per Topology Canon)

- The 5×N SAT agents contributed by all activated nodes
- Constitutional Spine (the 5-gate admissibility law + Ihsan/Gini/Zakat invariants)
- House of Wisdom (long-form synthesized knowledge across all attestations)
- Proof Engine (cross-node receipt verification + chain integrity)
- SEED Treasury (token / value-state primitives)
- Compute Pool (aggregated compute from contributing nodes)
- Storage Pool (aggregated storage)
- Bandwidth Pool (aggregated bandwidth)
- Shared Reflex Registry (compiled patterns the network has learned)
- Receipt Log (the canonical chain across all admitted missions)

### Required components (current readiness)

| Component | Repo readiness | Required state |
|---|---|---|
| URP transport (heartbeat → real bootnode) | `bizra-omega/bizra-resourcepool/` (heartbeat prototype) | Production substrate with discovery, peer attestation, receipt-channel ABI |
| Identity attestation across nodes | `bizra-omega/bizra-core/src/genesis_seal.rs` + `fate-binding/` | Cross-node Ed25519 + Dilithium post-quantum verification, both ends agree on the same chain integrity |
| Bootnode list / DHT | NOT BUILT | Stable bootnode addressing |
| First second-node join ceremony | NOT BUILT | Constitutional onboarding sequence: new node mints PAT-7 locally + 5 SAT agents into shared URP + opens membrane |
| URP wake-up sequence | partial (Cycle 1 Node0 activation proven) | One Node has woken the URP locally; cross-node URP membership not yet demonstrated |

### Closure criterion (corrected per Topology Canon)

A second human activates a Node — their device mints PAT-7 locally, 5 more SAT agents materialize into the **same** shared URP that Node0 woke. Both nodes interact with each other **only through their respective membranes and SAT-in-URP**, never peer-to-peer. A signed receipt produced on either node is verified by the SAT-5 layer in the URP, not by direct node-to-node trust.

That single verifiable URP-mediated handshake is the Phase 1 milestone. Per Topology Canon: *"PAT → Membrane → SAT. No peer-to-peer."*

---

## 3.5 The Membrane (per Topology Canon §The membrane)

The constitutional membrane sits between every local node and the shared URP. Every receipt that crosses produces a BLAKE3-chained, Ed25519-signed entry.

**Four properties:**

1. **Fail-closed** — Incomplete verification = reject
2. **Axiomatic filtering** — All constitutional invariants must hold (5-gate admissibility + economic invariants)
3. **Cryptographic provenance** — Every crossing produces a signed receipt
4. **Receipt completeness** — No gaps in the provenance log

**What never crosses the membrane:** Human identity, raw private data, unverified claims, untagged information.

The membrane is a *constitutional* construct, not just a network primitive. It is the boundary that makes the human's sovereignty meaningful: their data and identity stay local; only signed, verified attestations cross.

---

## 4. Phase 2 — SAT-5 fully operational + multi-node URP

**Truth label: PLANNED.**

### What changes (corrected per Topology Canon)

PAT-7 (Personal Agentic Teams) runs **inside each Node**. SAT-5 (System Agentic Teams) lives **inside the shared URP** — not "between" Nodes. Phase 2 is when SAT-5 in the URP grows from 5 (Node0 only) to 5×N (N nodes contributing).

The previous draft of this document framed SAT-5 as running "between Nodes." That contradicts Topology Canon and is corrected here. **SAT-5 does not run peer-to-peer. SAT-5 lives in the URP.**

### What this means concretely

| Capability | Repo readiness | Required state |
|---|---|---|
| URP federation gossip | `bizra-omega/bizra-federation` (scaffolded) | Gossip protocol operational across the SAT-in-URP layer; signed-message exchange respects membrane discipline |
| BFT consensus inside the URP | `bizra-omega/bizra-federation` (scaffolded) | SAT-5 layer in URP achieves BFT agreement on chain integrity |
| Membrane-mediated receipt exchange | Receipt protocol exists; membrane scaffolded | Transport + replication policy + dedup, all crossings signed and receipted |
| Constitutional gate enforcement | `bizra-omega/bizra-protocol` (scaffolded) | The 5-gate admissibility chain runs at every membrane crossing |
| SAT-5 wired into runtime gateway | Currently flagged as drift in canonical topology memory | Gateway routes SAT-related calls to the URP-resident SAT layer, not to a per-node stub |

### Closure criterion

A network of ≥5 Nodes maintains chain integrity under partition, rejoin, and one-node-malicious scenarios — verified through deterministic test scenarios. SAT-5 in the shared URP arbitrates any disputes via S2 Oracle (FROZEN axioms) and S3 Mediator. No node trusts another node directly; all trust is mediated through the URP's SAT layer.

---

## 5. Phase 3 — Federated cognition (opt-in autopoiesis)

**Truth label: PLANNED.**

### What changes

Each Node continues to do its own work locally via PAT-7. **Raw private data never leaves a Node** (the membrane forbids it). What leaves the Node, after passing the membrane's 5-gate admissibility, is signed optimization signals — gradient deltas, mission-pattern observations, anonymized receipts of completed missions — that other Nodes can opt into via their own membranes.

### Mechanism (corrected to use the membrane)

```
Local mission execution (PAT-7, on private data)
   └─→ Local RL update via bizra-ttrl (parameter delta only, not data)
        └─→ Receipt of "I learned X" passes Membrane (5-gate admissibility)
             └─→ Lands in URP's Shared Reflex Registry, attested by SAT-5
                  └─→ Other Nodes' PAT layers query the registry via their own Membranes
                       └─→ Their local FATE gate validates pooled update before applying
                            └─→ Each Node's runtime improves; membrane preserves sovereignty
```

### What is opt-in

- Whether to publish local learnings at all (per-Node consent)
- Whether to consume any specific pooled signal (per-Node consent, FATE-gated)
- Whether to pool with the entire network or only a chosen subset (per-Node policy)
- Whether to retain pooled improvements across reboots (per-Node persistence policy)

### What is **not** in Phase 3

- ❌ Centralized model aggregation
- ❌ Raw data leaving any Node (membrane forbids)
- ❌ Forced participation in pooled cognition
- ❌ Network-wide model "ownership" by any party
- ❌ Peer-to-peer between Nodes (always via URP/Membrane)

### Required components

| Capability | Repo readiness |
|---|---|
| `bizra-ttrl` on-device RL with SSO spectral norm | Scaffolded |
| `bizra-memory` synthesis pipeline (cross-node pooling layer) | Scaffolded for local; URP pooling layer NOT BUILT |
| `bizra-autopoiesis` self-healing | Scaffolded |
| Federated signal protocol via Membrane + URP Shared Reflex Registry | NOT BUILT |

---

## 6. Phase 4 — Decentralized self-growing agentic ecosystem

**Truth label: DIRECTIONAL.**

### What this is

The architectural intent: a network of sovereign Nodes that, by virtue of being honestly federated under a shared constitution and a shared URP, becomes more capable as more Nodes join — without any single Node ceding sovereignty over its data, identity, or mission gates.

Per Topology Canon's scaling table:

| Nodes | Local PAT (total) | SAT in shared URP (total) | Effect |
|---|---|---|---|
| 1 | 7 | 5 | System alive, flywheel starts |
| 1,000 | 7,000 | 5,000 | Serious governance capacity |
| 1,000,000 | 7M | 5M | Self-securing, self-evolving |
| 8,000,000,000 | 56B | 40B | Planetary intelligence |

### What this is **not**

- ❌ Not AGI. The network is a federation of typed agent ensembles operating under a shared constitution, not a single emergent mind.
- ❌ Not "world-first" anything. This document does not claim primacy.
- ❌ Not finality. There is no "the network is done" state. Constitutional invariants are continuously verifiable, not historically locked.
- ❌ Not a token economy primer. SEED Treasury primitives exist as Rust crates inside the URP; their network-economy semantics are explicitly out of this document's scope.

### Closure criterion

There is no closure for Phase 4. It is the operating mode that follows from Phases 0–3 succeeding. Its progress is measured continuously by:

- Number of Nodes that maintain chain integrity for ≥30 days
- Cross-Node receipt verification pass-rate via Membrane
- Constitutional gate enforcement rate (5-gate admissibility at every crossing)
- Per-Node opt-in rate for pooled cognition signals

These metrics will become measurable when Phase 2 + 3 complete. They are unmeasurable today.

---

## 7. Truth labels — phase summary

| Phase | Label | Rationale |
|---|---|---|
| Phase 0 | **CANDIDATE_CANONICAL** per Topology Canon (Cycle 1, hash `7b555875…db9c`, 21/21 tests) | 2 of 5 promotion gates passed; 3 pending (CI green, push, external review) |
| Phase 1 | **PLANNED** | URP heartbeat exists; production substrate doesn't; cross-Node URP-mediated handshake not demonstrated |
| Phase 2 | **PLANNED** | Federation crate scaffolded; SAT-in-URP at multi-node scale not operational |
| Phase 3 | **PLANNED** | TTRL + memory crates scaffolded; URP Shared Reflex Registry not built |
| Phase 4 | **DIRECTIONAL** | Architectural intent only; depends on Phases 0–3 |

These labels follow:
- BIZRA Genesis Manifest v0.1 truth-label discipline (PR #53, `evidence/node0_genesis_manifest/`)
- Public-claim discipline registered in PR #54 (`PUBLIC_CLAIM_DISCIPLINE_RECERT_v0_1.md`)
- Topology Canon's CANDIDATE → CANONICAL promotion ladder

---

## 8. Explicit non-claims

This document **does not** claim:

- That any production URP transport exists today.
- That cross-Node Gini computation exists today.
- That any second Node has joined the URP today.
- That SAT-5 is wired into the runtime gateway at multi-node scale today.
- That a public SEED-token economy is activated today.
- That the network operates "trustlessly" today (it operates fail-closed on a single Node, all gates passed locally).
- That the URP runs as a server (it is a shared organism — see Topology Canon's forbidden mistakes).
- That any of the planned phases has a committed delivery date.
- That this document supersedes the Topology Canon — *it does not; the Topology Canon wins on every conflict.*

This document **does not** authorize:

- Any runtime, core, src, CI, or dependency change.
- Any website source edit.
- Any Claim Registry implementation.
- Any merge of any open PR.
- Any change to Phase 2 or Phase 3 WAIT lock.
- Any modification to the Origin Kernel (raw §1 forbidden by §6.1).
- Any commit of the Origin Kernel itself to runtime canon (forbidden by Origin Kernel §6.3 until Canon Store Ingestion Gate exists).

---

## 9. The canonical sentence — Topology Canon authoritative

> *"Each human node mints PAT-7 locally on their device and SAT-5 into one shared Universal Resource Pool. PAT serves the human. SAT serves the system. The membrane sits between them."*

Earlier drafts of this document used a different one-liner ("Node0 proves the seed can live alone…"). That is preserved as a poetic *companion* phrase, not the canonical sentence. **Topology Canon's wording is the authoritative one.**

---

## 10. Origin Kernel provenance

This document is downstream of `docs/canon/BIZRA_ORIGIN_KERNEL.md`. Specifically:

- **§4.1 (Knowledge → humility):** every truth label downgraded from MEASURED to CANDIDATE_CANONICAL or PLANNED reflects the discipline that more learning = more awareness of what remains unknown.
- **§4.2 (Symmetric epistemic charity):** Phase 4's "what this is **not**" list applies symmetrically — neither over-claiming for BIZRA nor dismissing competing approaches.
- **§4.3 (Law of Assumption + Ihsan):** every PLANNED label is a declared uncertainty; every "NOT BUILT" entry is the refusal of bare speculation; every "Required state" column is the Ihsan-form when assumption is unavoidable.

The Kernel itself is **not** runtime canon (per its §5) and is **not** committed to `origin/main` yet (its §6.3 awaits the Canon Store Ingestion Gate). This document cites it for provenance only.

---

## Appendix A — Repo evidence map (verified 2026-04-26)

| Phase | Crate / module | File path | Current state |
|---|---|---|---|
| Phase 0 | `bizra-node` | `bizra-omega/bizra-node/` | Operational on single machine |
| Phase 0 | `bizra-mission` (Trust Compiler) | `bizra-omega/bizra-mission/` | 14-state lifecycle, signed receipts |
| Phase 0 | `bizra-core` (constitution) | `bizra-omega/bizra-core/src/` | 5 frozen root objects |
| Phase 0 | `bizra-cognition` (5 frozen contracts) | `bizra-omega/bizra-cognition/src/` | `admissibility_freeze_v1`, `mission_freeze_v1`, `eval_v1`, `eval_v1_integrated`, `receipt_freeze_v1` |
| Phase 0 | `bizra-cognition-gateway` (HTTP projection + dema binary) | `bizra-omega/bizra-cognition-gateway/` | Axum gateway + `src/bin/dema.rs` real binary |
| Phase 0 | PAT-7 / SAT-5 topology | `bizra-omega/bizra-core/src/topology_canon.rs` | Names canonical; SAT-in-URP wiring partial |
| Phase 0 | FATE gate (5-gate admissibility) | `bizra-omega/fate-binding/` | Z3 + Dilithium post-quantum |
| Phase 0 | Active Node0 receipt chain | `sovereign_state/bridge_receipts/` | 2473 receipts; founder MoMo (محمد) |
| Phase 0 | DEMA = P7 surface | `dema-console/` (Next.js) + Rust `dema` binary | Real Rust binary at `bizra-omega/target/release/dema` |
| Phase 1 | URP heartbeat | `bizra-omega/bizra-resourcepool/` | Prototype only |
| Phase 1 | Identity attestation | `bizra-omega/bizra-core/src/genesis_seal.rs` | Single-Node operational |
| Phase 1/2 | Membrane | `bizra-omega/bizra-protocol/` | Scaffolded |
| Phase 2 | Federation gossip | `bizra-omega/bizra-federation/` | Scaffolded |
| Phase 3 | TTRL on-device RL | `bizra-omega/bizra-ttrl/` | Scaffolded |
| Phase 3 | Memory synthesis | `bizra-omega/bizra-memory/` | Local layer present; URP pooling not built |
| Phase 3 | Autopoiesis | `bizra-omega/bizra-autopoiesis/` | Scaffolded |

---

## Appendix B — Memory anchors

This document preserves and reinforces:

- `reference_bizra_topology_canon_frozen_2026_03_25` — **AUTHORITY** for all topology claims
- `reference_bizra_master_stack_canon_2026_04_26` — Verificative AI category, Trust Compiler vocabulary, 5-gate admissibility chain
- `reference_origin_kernel_invariant_trace` — §4.1 / §4.2 / §4.3 trace through recent work
- `reference_bizra_full_host_topology_2026_04_26` — physical map of where all BIZRA material lives on host
- `feedback_audit_label_inflation_guard` — every phase row carries an honest Truth label
- `feedback_third_party_eval_does_not_override_canon` — phase claims backed by repo evidence
- `feedback_land_the_plane` — explicit non-claims (§8) prevent scope expansion
- `project_pat_sat_canonical_topology` — PAT-7 / SAT-5 naming canon
- `project_node0_closure_scoreboard_2026_04_21` — current closure gate status
- `project_node0_activation_complete_2026_04_25` — local Node0 active, 11/11 lifecycle gates green

---

## Appendix C — v0.1 → v0.2 changelog

This v0.2 corrects the following drifts from v0.1:

1. **Authority chain header added** — Origin Kernel + Topology Canon + Master Stack referenced explicitly.
2. **DEMA = P7 identification** — Was treated as a separate "Dema face" component. Corrected: DEMA is P7 of PAT-7. The dema-console is the surface FOR P7.
3. **PAT-7 names enumerated** — P1–P7 with FROZEN annotations on P5 (Ethicist).
4. **SAT-5 names enumerated** — S1–S5 with FROZEN annotation on S2 (Oracle).
5. **SAT-5 location corrected** — Was framed as running "between Nodes." Corrected: SAT-5 lives **inside the shared URP**.
6. **URP framing corrected** — Was framed as a network the first Node "joins" or "becomes a member of." Corrected: URP is a **single shared organism that wakes from dormant** when the first human activates.
7. **The Membrane added** — New §3.5 dedicated to the constitutional membrane (4 properties + what never crosses).
8. **Closure criterion for Phase 1 corrected** — Was "node-to-node receipt verification." Corrected: receipts cross via Membrane; verification by SAT-5 in URP.
9. **Phase 3 mechanism corrected** — Now routes signal flow through Membrane + URP Shared Reflex Registry, not peer-to-peer.
10. **Truth label upgrade for Phase 0** — Was "MEASURED for proven rows; PARTIAL for whole." Corrected: **CANDIDATE_CANONICAL** per Topology Canon (Cycle 1, hash `7b555875…db9c`), with 3 of 5 promotion gates pending.
11. **Master Stack vocabulary added** — Verificative AI category, Trust Compiler architectural metaphor, 5-gate admissibility chain (ZANN_ZERO / CLAIM_MUST_BIND / RIBA_ZERO / NO_SHADOW_STATE / IHSAN_FLOOR ≥ 0.95).
12. **Origin Kernel provenance added** — New §10 explicitly trace the document's normative force back to the Kernel's three invariants.
13. **Forbidden mistakes from Topology Canon honored** — "ONE URP not per-node", "SAT in URP not on local node", "no peer-to-peer", "URP not a server".
14. **Appendix A evidence map expanded** — Added `bizra-cognition` 5 frozen contracts, `bizra-cognition-gateway` HTTP projection + dema binary, active 2473-receipt chain, DEMA=P7 verified path.
15. **Appendix B memory anchors updated** — References Topology Canon, Master Stack, Origin Kernel trace.
16. **§9 canonical sentence corrected** — Topology Canon's wording promoted to authoritative; v0.1's "Node0 proves the seed can live alone…" preserved as poetic companion only.
17. **§8 non-claims expanded** — Added: this document does not supersede Topology Canon; does not modify Origin Kernel §1; does not commit Origin Kernel to runtime canon (§6.3).

This document is **not** a roadmap, **not** a commitment of dates, and **not** an external-facing piece. It is internal canon, downstream of the Topology Canon, downstream of the Master Stack, downstream of the Origin Kernel.

هذه هي البذرة. **The Seed.**
