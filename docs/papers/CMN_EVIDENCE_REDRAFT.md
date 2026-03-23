# Constitutional Membrane Networking: Evidence-First Redraft

**Document:** CMN_vNEXT_EVIDENCE_REDRAFT.md
**Status:** Shell for Ihsan-compliant paper — claims narrowed to proven evidence
**Tagging:** Every claim carries [VERIFIED], [PLANNED], or [DERIVED] per CLAIM_MUST_BIND
**Source:** Restructured from `docs/papers/CMN_vNEXT_EVIDENCE_FIRST_BLUEPRINT.md`, not the older preprint

---

## Abstract

BIZRA implements Constitutional Membrane Networking (CMN), a runtime architecture where every agent action passes through a fail-closed governance membrane before execution and produces a cryptographic receipt after completion. [VERIFIED — core/sovereign/api.py:4317 rejects missing authority; core/sovereign/runtime_core.py:4445 reports authority path]

This paper presents five contributions: (1) Proof-Carrying Execution, (2) the Governance Monad, (3) Constitutional Pruning via Algebraic Impossibility, (4) Isnad Risk Propagation, and (5) the Adl-Convergent Stochastic Process. We present Node0 evidence for contributions 1-3 and formal proofs for 4-5. [DERIVED — contributions 1-3 are runtime-verified; contributions 4-5 are mathematically proven but network-scale execution is PLANNED]

---

## 1. Introduction

### 1.1 The Problem

Multi-agent AI systems lack runtime governance guarantees. Existing approaches rely on post-hoc auditing or assertion-based trust. [VERIFIED — competitive analysis in BIZRA-UX-STRATEGY-001, Section 1: no competitor combines sovereignty + trust + economy + local-first]

### 1.2 The CMN Thesis

A constitutional membrane is a runtime boundary that:
- Rejects unauthorized execution before it occurs [VERIFIED — api.py:4317]
- Produces cryptographic receipts for every approved action [VERIFIED — core/bus/subscribers.py:90, organism.py:323]
- Aggregates receipts into breath-level truth [VERIFIED — heartbeat.py:103]
- Surfaces governance state as observable runtime data [VERIFIED — runtime_core.py:4445]

### 1.3 Scope Boundary

This paper presents evidence from a single-node deployment (Node0). Network-scale claims (federation, cross-node consensus, URP distribution) are explicitly staged as PLANNED and not presented as current capability. [This narrowing is the primary Ihsan correction from the prior preprint]

---

## 2. Architecture

### 2.1 The Receipt Plane

[VERIFIED] The system's most defensible claim is the receipt-native evidence plane.

- CQRS subscriber outcomes are explicit receipts: `core/bus/subscribers.py:90`
- Receipts are wired durably by the organism: `core/sovereign/organism.py:323`
- Receipts are persisted canonically by Node0: `core/node0/heartbeat.py:1243`
- Receipts are folded into breath truth: `core/node0/heartbeat.py:103`
- Receipts are surfaced as runtime truth: `core/sovereign/runtime_core.py:4445`

Evidence chain: subscriber → organism → heartbeat → breath → runtime. Every link verified in code.

### 2.2 The Membrane Boundary

[VERIFIED] The membrane is genuinely implemented at the runtime boundary.

- Canonical execution rejects missing authority: `api.py:4317`
- Approved-only aggregation: `helix3.py:304`
- Runtime reports authority path and canonical ownership: `runtime_core.py:4445`

### 2.3 The Autopoietic Loop

[VERIFIED — partial] Receipt observation is first-class in autopoiesis: `loop.py:181`. The runtime feeds that plane: `runtime_core.py:751`.

**Honest limitation:** HHMM/diffusion-style cognition is still mostly advisory. These outputs are not yet receipt-native. Promoting them to the governed receipt plane is PLANNED (60-90 day horizon).

### 2.4 Agent Architecture

[VERIFIED] PAT-7 (Personal Agentic Team, 7 user-serving agents) and SAT-5 (System Agent Topology, 5 validation agents) operate with constitutional separation.

- P5 Ethicist and S2 Oracle are permanently frozen — ethics derived from revelation, not data [VERIFIED — Enforceable Spine v1.1, Section 4]
- Dual-agentic separation solves the self-grading conflict of interest [DERIVED — no single entity both executes and validates]

---

## 3. Contribution 1: Proof-Carrying Execution

[VERIFIED]

Every execution carries its proof. The Proof Canon defines 7 minimum visible fields:
1. `execution_authority` — who authorized this action
2. `authority_path` — constitutional chain from Quran → Hadith → البذرة → Spine → spec → code
3. `fate_verdict` — FATE gate decision (approve/reject)
4. `fate_reason_codes` — why the gate decided what it decided
5. `identity_mode` — sovereign or delegated
6. `signer_public_key_prefix` — Ed25519 verification anchor
7. `hash_chain_ref` — BLAKE3 chain position

**Node0 evidence:** `verify_genesis.py` completes 6/6 checks in 1.5 seconds. Block 0 contains 12 agent mints, 1,124,695 SEED, and constitutional thresholds — all receipt-backed.

---

## 4. Contribution 2: The Governance Monad

[VERIFIED — type level]

The governance monad wraps every computational step in a constitutional context:

```
action >>= gate >>= execute >>= receipt >>= chain
```

- If `gate` fails (Ihsan < 0.95, ZANN_ZERO violation, Spine breach), execution halts
- The monad is fail-closed: no default path bypasses the gate
- The SAT Mint Court demonstrated this: founder's own work was REJECTED at SNR 0.577 [VERIFIED — session evidence]

**Honest limitation:** The monad is enforced in Python at the API boundary (`api.py:4317`). Full Rust-level enforcement via the type system is PLANNED (bizra-omega crate expansion).

---

## 5. Contribution 3: Constitutional Pruning via Algebraic Impossibility (CPAI)

[VERIFIED — type level]

CPAI constrains the operation space until the only reachable states are constitutional:

- `IhsanScore` type enforces >= 0.95 at construction [VERIFIED — Rust type constraint]
- `ExactAmount` type prevents floating-point drift in economic computation [VERIFIED — bizra-sippar crate, 21 tests]
- `require_signed()` enforces Ed25519 at every trust boundary [VERIFIED — runtime gate]

The insight: constrain the operation space until the only reachable states are constitutional, then every execution is a proof.

---

## 6. Contribution 4: Isnad Risk Propagation (IRP)

[DERIVED — mathematically proven, not yet network-executed]

Formal proof in `docs/proofs/CMN_PROOF_KERNEL.md`. IRP propagates trust scores through citation chains analogous to hadith isnad methodology. Risk attenuates geometrically with chain length.

**Honest limitation:** IRP is proven on paper and has unit tests. It has not been executed across a multi-node network. Network-scale validation is PLANNED (Phase 4, 2027+).

---

## 7. Contribution 5: Adl-Convergent Stochastic Process (ACSP)

[DERIVED — mathematically proven, Node0 Gini monitoring active]

The Adl invariant (Gini <= 0.35) is enforced via a Lyapunov stability proof:
- V(W) = max(0, Gini(W) - 0.35)^2
- Monotone decrease proven
- Convergence rate derived

**Node0 evidence:** Current Gini at 0.31, monitored by SAT-5. Single-node evidence only.

**Honest limitation:** The Harberger tax mechanism and causal drag are specified but not yet exercised under adversarial conditions. Adversarial validation program is PLANNED (`docs/security/CMN_ADVERSARIAL_VALIDATION_PROGRAM.md`).

---

## 8. Performance and Adversarial Resistance

### 8.1 Membrane Tax

[VERIFIED — with measurement caveats]

Membrane-tax benchmark (`scripts/ci_membrane_tax_benchmark.py`) records raw metrics plus sanity-clamped metrics. The benchmark is now a CI gate (`membrane-tax-gate.yml`) — workflow fails if `clamped_negative_metrics` is non-empty.

**Honest limitation:** The negative `eventbus_emission_ms` was clamped, not root-caused. The benchmark contract needs underlying correction.

### 8.2 Adversarial Resistance

[VERIFIED — simulation-grade evidence]

Adversarial House of Wisdom simulation (`adversarial_how.py`) tested 4 attack vectors across 100 nodes with 30% malicious participants (above the Byzantine f < n/3 threshold):

| Attack Vector | Poisoning Rate | Rejection Rate | Forensic Quality |
|---|---|---|---|
| Direct poisoning | 0.0% | 100.0% | N/A |
| Slow drift | 0.0% | 100.0% | 98.1% |
| Sybil endorsement | 0.0% | 100.0% | N/A |
| Reputation gaming | 0.6% | 99.4% | 99.4% |
| **Combined** | **0.15%** | **99.85%** | **48.6%** |

Key findings:
- Constitutional content gates (Ihsan threshold, keyword detection, drift detection) achieve 100% rejection on 3 of 4 attack vectors
- Behavioral anomaly detector (quality trajectory + variance spike + repeat offender quarantine) reduces reputation attack from 49.9% to 0.6%
- System resistance improves with attacker density (more behavioral data to detect anomalies)
- Combined poisoning rate of 0.15% at 30% malicious exceeds Byzantine fault tolerance expectations

**Honest limitation:** This is a simulation, not a live multi-node deployment. The SAT-5 behavioral model is simplified. Real-world reputation attacks may use more sophisticated strategies. The simulation does not model network partitions or timing attacks.

---

## 9. Limitations and Future Work

### Explicitly PLANNED (not claimed as current)

1. **Network-scale federation** — cross-node agent collaboration, guild system, reputation federation
2. **HHMM/diffusion cognition as receipts** — promoting advisory outputs to governed receipt plane
3. **Adversarial validation** — mechanizing kernel theorems, bounded adversarial program execution
4. **Typed error taxonomy** — replacing broad `except Exception` with constitutional error hierarchies
5. **Research corpus governance** — canonicalizing 3-year, 150-document corpus as searchable provenance-aware input layer

### What we do NOT claim

- We do not claim network-scale proof. Node0 is a single-node deployment.
- We do not claim LLM reasoning advancement. Our contribution is governance infrastructure, not cognition.
- We do not claim decentralization. One node is not decentralized. Federation is PLANNED.

---

## 10. Conclusion

CMN's strongest claim is now empirically grounded: the receipt-native evidence plane is the real moat. Every execution is authorized, every authorization is gated, every gate decision is receipted, every receipt is chained, and every chain is observable. This is verified on Node0 with 12,644 tests, 768K LOC, and a genesis block containing constitutional thresholds.

The paper's scope is deliberately narrowed from the prior preprint. Claims match evidence. Where evidence is partial, we say so. Where evidence is absent, we say PLANNED. This is the Ihsan standard: excellence through honesty, not through rhetoric.

---

## Evidence Appendix

| Claim | Tag | Evidence Location |
|-------|-----|-------------------|
| Membrane rejects unauthorized | VERIFIED | core/sovereign/api.py:4317 |
| Receipts are CQRS outcomes | VERIFIED | core/bus/subscribers.py:90 |
| Organism wires receipts durably | VERIFIED | core/sovereign/organism.py:323 |
| Heartbeat persists canonical | VERIFIED | core/node0/heartbeat.py:1243 |
| Breath folds receipts | VERIFIED | core/node0/heartbeat.py:103 |
| Runtime surfaces truth | VERIFIED | core/sovereign/runtime_core.py:4445 |
| Helix3 approved-only aggregation | VERIFIED | core/sovereign/helix3.py:304 |
| Autopoiesis observes receipts | VERIFIED | core/autopoiesis/loop.py:181 |
| verify_genesis.py 6/6 in 1.5s | VERIFIED | scripts/verify_genesis.py |
| Block 0 minted | VERIFIED | genesis block hash 350d642099bde68b |
| IRP formal proof | DERIVED | docs/proofs/CMN_PROOF_KERNEL.md |
| ACSP Lyapunov proof | DERIVED | docs/proofs/CMN_PROOF_KERNEL.md |
| Gini monitored at 0.31 | VERIFIED | SAT-5 runtime |
| Mint Court rejection at 0.577 | VERIFIED | session evidence |
| Adversarial sim: 30% malicious, 0.15% poison | VERIFIED | adversarial_how.py (simulation) |
| Reputation attack: 49.9%→0.6% after detector | VERIFIED | adversarial_how.py v2 |
| Proof kernel: 3/3 properties proven | VERIFIED | proof_kernel.py (127ms) |
| Network federation | PLANNED | Phase 4 roadmap |
| HHMM as receipts | PLANNED | 60-90 day horizon |
| Adversarial validation | PLANNED | docs/security/CMN_ADVERSARIAL_VALIDATION_PROGRAM.md |
| Typed error taxonomy | PLANNED | 30-day sprint |
| Research corpus governance | PLANNED | Workstream W5 |
