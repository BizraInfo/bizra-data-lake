# BIZRA Node0 → URP Ecosystem Transition (v0.1)

**Date:** 2026-04-25 (GST) — Dubai
**Scope:** Architecture transition note. Docs-only. Records the canonical phase progression from single-node Node0 sovereign runtime to a decentralized distributed agentic ecosystem.
**Status lock:** WAIT preserved. No runtime, core, src, CI, dependency, website, or Claim Registry changes.
**Audience:** Internal canon reference. Not a public roadmap. Not a commitment of dates.

---

## 1. Purpose

Node0 is a **bootstrap**, not a destination.

Before BIZRA can credibly describe itself as a decentralized agentic ecosystem, one Node must prove it can stand alone — execute missions, sign receipts, replay its own state, hold its constitutional gates, recover from restart — entirely on a single machine, with no external dependency.

This document records, as canon, how the architecture moves from that proven single-node baseline through the Universal Resource Pool (URP), the SAT-5 network layer, federated cognition, and opt-in autopoiesis. It states explicitly what is **measured**, what is **planned**, and what is **directional only**.

The canonical sentence:

> *"Node0 proves the seed can live alone; URP lets seeds connect; SAT lets the forest coordinate; autopoiesis lets the forest improve without breaking node sovereignty."*

---

## 2. Phase 0 — Single-node sovereign runtime (current)

**Truth label: MEASURED for proven rows; PARTIAL for the 11-gate Node0 lifecycle as a whole.**

### What exists today

| Component | Repo location | Role |
|---|---|---|
| Sovereign runtime | `core/sovereign/`, `bizra-omega/bizra-node` | Mission orchestration on one machine |
| Dema face | `dema-console/`, frontend | Operator-visible surface |
| PAT-7 (Personal Agentic Teams) | `bizra-omega/bizra-core/src/topology_canon.rs` | 7 agents running inside the local node |
| Mission kernel | `bizra-omega/bizra-mission` | 14-state lifecycle, signed receipts |
| Receipt protocol | `bizra-omega/bizra-core/src/canonical_receipt.rs` + Python mirror | BLAKE3-chained, Ed25519-signed |
| Replay verifier | spearpoint replay path (PR #49) | Re-derive state from receipt chain |
| FATE gate | `bizra-omega/fate-binding/` | Z3 + post-quantum constitutional check |
| Identity / genesis seal | `bizra-omega/bizra-core/src/genesis_seal.rs` | Deterministic root of trust |

### What this proves

- One node can execute missions, sign each visible effect, chain those signatures, and re-derive its state from the chain alone.
- Constitutional invariants (Ihsan ≥0.95, RIBA-zero, ZANN-zero, Gini ≤0.35 single-participant) are enforced by code, not by promise.
- Restart recovery, replay, and lifecycle gates are testable on a single machine.

### Closure gates currently in flight

| PR | Lane | Status |
|---|---|---|
| #49 | Row 4 replay (canonical spearpoint replay) | MEASURED — 38/38 tests green; awaiting merge |
| #50 | Mission receipt full-payload Ed25519 signature | MEASURED — 4 tests green; awaiting merge |
| #51 | Python 3.12 baseline restored across 874 tests | MEASURED — full pass; awaiting merge (unblocks #49 + #50) |
| #52 | Credential purge (CVE-class) | MEASURED — 8 files; awaiting merge |
| #53 | Genesis Manifest v0.1 (founder-stated → hash-anchored bridge) | MEASURED — chain hash recorded; side-track |
| #54 | Public-claim discipline recert v0.1 | MEASURED — claim register published |

Phase 0 closes when these PRs merge and the 11-gate Node0 lifecycle (`tools/node0_lifecycle_flywheel/`) reports `lifecycle_ready: true` from a clean run on a fresh machine.

---

## 3. Phase 1 — URP Sentinel production bootnode

**Truth label: PLANNED.**

### What changes

The single Node0 stops being alone. It becomes the first member of the **Universal Resource Pool (URP)** — a network of constitutionally-compliant nodes that can discover each other, attest each other's identities, and exchange signed receipts.

### Required components

| Component | Repo readiness | Required state |
|---|---|---|
| URP Sentinel transport | `bizra-omega/bizra-resourcepool/` (heartbeat prototype) | Production bootnode with discovery, peer attestation, receipt-channel ABI |
| Identity attestation | `bizra-omega/bizra-core/src/genesis_seal.rs` + `fate-binding/` | Cross-node Ed25519 + Dilithium post-quantum verification |
| Bootnode list / DHT | NOT BUILT | Stable bootnode addressing |
| URP heartbeat → real bootnode upgrade | NOT BUILT | The current heartbeat prototype is **not** the production bootnode |
| First second-node join ceremony | NOT BUILT | Constitutional onboarding sequence |

### Closure criterion

Two physically separate Nodes, each with its own genesis seal, exchange a signed receipt that each verifies independently using only the other's published public key — and both Nodes agree on chain integrity without any shared state authority.

That single verifiable handshake is the Phase 1 milestone.

---

## 4. Phase 2 — SAT-5 network layer

**Truth label: PLANNED.**

### What changes

PAT-7 (Personal Agentic Teams) runs inside each Node. SAT-5 (System Agentic Teams) runs **between** Nodes. Phase 2 lights up the SAT-5 layer.

### What this means concretely

| Capability | Repo readiness | Required state |
|---|---|---|
| Federation gossip | `bizra-omega/bizra-federation` (scaffolded) | Gossip protocol operational; signed-message exchange across N nodes |
| BFT consensus | `bizra-omega/bizra-federation` (scaffolded) | Byzantine fault tolerance for receipt-chain agreement |
| Cross-node receipt exchange | Receipt protocol exists | Transport + replication policy + dedup |
| Network trust boundary | `bizra-omega/bizra-protocol` (scaffolded) | Constitutional gate enforcement at every cross-node call |
| SAT-5 wired into gateway | Currently flagged as drift in canonical topology memory | Gateway routes SAT-5 calls to the federation layer |

### Closure criterion

A network of ≥5 Nodes maintains chain integrity under partition, rejoin, and one-node-malicious scenarios — verified through deterministic test scenarios and signed by an independent observer.

---

## 5. Phase 3 — Federated cognition (opt-in autopoiesis)

**Truth label: PLANNED.**

### What changes

Each Node continues to do its own work locally. **Raw private data never leaves a Node.** What leaves the Node is signed optimization signals — gradient deltas, mission-pattern observations, anonymized receipts of completed missions — that other Nodes can opt into to improve their own runtime.

### Mechanism

```
Local mission execution (PAT-7, on private data)
   └─→ Local RL update via bizra-ttrl (parameter delta only)
        └─→ Receipt of "I learned X" signed and published to URP
             └─→ Other Nodes evaluate signal, decide whether to opt in
                  └─→ FATE gate validates pooled update before applying locally
                       └─→ Each Node's runtime improves; sovereignty preserved
```

### What is opt-in

- Whether to publish local learnings at all (per-Node consent)
- Whether to consume any specific pooled signal (per-Node consent, FATE-gated)
- Whether to pool with the entire network or only a chosen subset (per-Node policy)
- Whether to retain pooled improvements across reboots (per-Node persistence policy)

### What is **not** in Phase 3

- ❌ Centralized model aggregation
- ❌ Raw data leaving any Node
- ❌ Forced participation in pooled cognition
- ❌ Network-wide model "ownership" by any party

### Required components

| Capability | Repo readiness |
|---|---|
| `bizra-ttrl` on-device RL with SSO spectral norm | Scaffolded |
| `bizra-memory` synthesis pipeline (cross-node pooling layer) | Scaffolded for local; pooling layer NOT BUILT |
| `bizra-autopoiesis` self-healing | Scaffolded |
| Federated signal protocol + FATE-gated opt-in | NOT BUILT |

---

## 6. Phase 4 — Decentralized self-growing agentic ecosystem

**Truth label: DIRECTIONAL.**

### What this is

The architectural intent: a network of sovereign Nodes that, by virtue of being honestly federated under a shared constitution, becomes more capable as more Nodes join — without any single Node ceding sovereignty over its data, identity, or mission gates.

### What this is **not**

- ❌ Not AGI. The network is a federation of typed agent ensembles, not a single emergent mind.
- ❌ Not "world-first" anything. This document does not claim primacy.
- ❌ Not finality. There is no "the network is done" state. Constitutional invariants are continuously verifiable, not historically locked.
- ❌ Not a token economy primer. CAP / Zakat primitives exist as Rust crates; their network-economy semantics are explicitly out of this document's scope.

### Closure criterion

There is no closure for Phase 4. It is the operating mode that follows from Phases 0–3 succeeding. Its progress is measured continuously by:

- Number of Nodes that maintain chain integrity for ≥30 days
- Cross-Node receipt verification pass-rate
- Constitutional gate enforcement rate (Ihsan / RIBA-zero / ZANN-zero / Gini)
- Per-Node opt-in rate for pooled cognition signals

These metrics will become measurable when Phase 2 + 3 complete. They are unmeasurable today.

---

## 7. Truth labels — phase summary

| Phase | Label | Rationale |
|---|---|---|
| Phase 0 | **MEASURED for individual rows; PARTIAL for the 11-gate Node0 lifecycle as a whole** | Replay, signing, identity, lifecycle harness exist with passing tests; whole-lifecycle gate not yet `ready` on a clean machine |
| Phase 1 | **PLANNED** | URP heartbeat exists; production bootnode does not; cross-Node handshake not demonstrated |
| Phase 2 | **PLANNED** | Federation crate scaffolded; gossip + BFT consensus not operational |
| Phase 3 | **PLANNED** | TTRL + memory crates scaffolded; pooling layer not built |
| Phase 4 | **DIRECTIONAL** | Architectural intent only; depends on Phases 0–3 |

These labels follow the BIZRA Genesis Manifest v0.1 truth-label discipline (`evidence/node0_genesis_manifest/`) and the public-claim discipline registered in PR #54 (`PUBLIC_CLAIM_DISCIPLINE_RECERT_v0_1.md`).

---

## 8. Explicit non-claims

This document **does not** claim:

- That any production federation transport exists today.
- That cross-Node Gini computation exists today.
- That any URP bootnode is operational today.
- That SAT-5 is wired into the runtime gateway today.
- That a public CAP-token / Zakat-distribution economy is activated today.
- That any second Node has joined the URP today.
- That the network operates "trustlessly" today (it operates fail-closed on a single Node).
- That any of the planned phases has a committed delivery date.

This document **does not** authorize:

- Any runtime, core, src, CI, or dependency change.
- Any website source edit.
- Any Claim Registry implementation.
- Any merge of any open PR.
- Any change to Phase 2 or Phase 3 WAIT lock.
- Any shift in the position of PR #49 / #50 / #51 / #52 / #53 / #54 in the queue.

---

## 9. The canonical sentence

> *"Node0 proves the seed can live alone; URP lets seeds connect; SAT lets the forest coordinate; autopoiesis lets the forest improve without breaking node sovereignty."*

This is the one-line synthesis of Sections 2–6. It compresses the architectural intent without overstating it.

---

## Appendix A — Repo evidence map

| Phase | Crate / module | File path | Current state |
|---|---|---|---|
| Phase 0 | `bizra-node` | `bizra-omega/bizra-node/` | Operational on single machine |
| Phase 0 | `bizra-mission` | `bizra-omega/bizra-mission/` | 14-state lifecycle, signed receipts |
| Phase 0 | `bizra-core` (constitution) | `bizra-omega/bizra-core/src/` | 5 frozen root objects |
| Phase 0 | PAT-7 / SAT-5 topology | `bizra-omega/bizra-core/src/topology_canon.rs` | Names canonical; SAT wiring partial |
| Phase 0 | FATE gate | `bizra-omega/fate-binding/` | Z3 + Dilithium post-quantum |
| Phase 1 | URP heartbeat | `bizra-omega/bizra-resourcepool/` | Prototype only |
| Phase 1 | Identity attestation | `bizra-omega/bizra-core/src/genesis_seal.rs` | Single-Node operational |
| Phase 2 | Federation gossip | `bizra-omega/bizra-federation/` | Scaffolded |
| Phase 2 | Trust boundary | `bizra-omega/bizra-protocol/` | Scaffolded |
| Phase 3 | TTRL on-device RL | `bizra-omega/bizra-ttrl/` | Scaffolded |
| Phase 3 | Memory synthesis | `bizra-omega/bizra-memory/` | Local layer present; pooling not built |
| Phase 3 | Autopoiesis | `bizra-omega/bizra-autopoiesis/` | Scaffolded |

---

## Appendix B — Memory anchor

This document preserves and reinforces:

- `feedback_audit_label_inflation_guard` — every phase row carries an honest Truth label, with directional vs. operational separation.
- `feedback_third_party_eval_does_not_override_canon` — phase claims are backed by repo evidence in Appendix A, not by external endorsements.
- `feedback_land_the_plane` — explicit non-claims (§8) prevent scope expansion past Phase 0 closure.
- `project_pat_sat_canonical_topology` — PAT-7 / SAT-5 distinction recorded canonically.
- `project_urp_sentinel_v0_design_contract` — URP Sentinel design contract referenced as Phase 1 prerequisite.
- `project_node0_closure_scoreboard_2026_04_21` — current closure gate status mirrored in §2.

---

## Appendix C — What this document is for

- An onboarding artifact for any contributor or reviewer who needs to understand where Node0 sits on a longer arc without overstating where the arc has actually reached.
- A canonical reference that future audit recertification runs can cite without rebuilding the phase model from scratch.
- A claim-discipline anchor: any future communication that references "BIZRA's distributed ecosystem" can point here for the truth-labeled state at the time of that communication.

This document is **not** a roadmap, **not** a commitment of dates, and **not** an external-facing piece. It is internal canon.
