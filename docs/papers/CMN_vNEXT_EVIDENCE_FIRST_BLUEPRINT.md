# CMN vNext: Evidence-First Redraft Blueprint

Date: 2026-03-22  
Status: Drafting companion for the CMN preprint  
Primary paper: [CMN_Preprint_Beshr_2026_CORRECTED.md](CMN_Preprint_Beshr_2026_CORRECTED.md)

## Purpose

This blueprint defines how the next CMN paper should be redrafted so that every major claim is explicitly labeled as one of:

- canonical truth proven in the repo today
- specified-by-design but not yet empirically complete
- future work with named proof obligations

The goal is to increase proof density without expanding the claim surface.

## Core Editorial Rule

The paper should win by being more honest than competing systems papers, not by claiming more.

The central thesis should therefore be reduced to one statement:

**Topology-level governance is stronger than model-level alignment because topology can be made fail-closed and receipt-verifiable.**

## Canonical Truth Plane

The current repo already supports a strong paper-ready evidence plane:

`intent -> governed execution -> receipt -> verification -> canonical memory -> reflex compilation -> faster governed execution`

That flow is supported by:

- runtime-owned organism authority
- Node0 receipt and breath authority
- CQRS subscriber ack and dead-letter receipts
- Node0 canonical delivery persistence
- sovereign async bus mirror of delivery receipts
- canonical spearpoint replay proof

These are the surfaces the paper can speak about in the present tense.

## Claim Discipline Table

| Claim class | Paper wording | Evidence requirement |
| --- | --- | --- |
| Canonical truth | “We implement / we verify / we demonstrate” | code + tests + receipts + artifact |
| Specified-by-design | “We specify / we define / we architect” | design doc + code boundary + explicit limitation |
| Future work | “We will test / mechanize / validate” | named proof obligation + planned experiment |

## What Is Safe To Claim Now

### 1. Membrane governance exists as code

Safe claim:

CMN’s membrane is already partially realized as a fail-closed, receipt-native enforcement plane with canonical mission authority, explicit delivery receipts, and chained provenance.

Evidence classes:

- sovereign API boundary posture
- runtime-owned organism authority
- Node0 receipt and breath chain
- CQRS delivery receipts + canonical persistence
- spearpoint replay artifact

### 2. Governed self-optimization is demonstrated at minimal scale

Safe claim:

The system demonstrates one replay-verifiable minimal self-improvement loop in which a deliberative path yields a receipt, a bounded state delta is applied, and a later run takes a faster governed path while preserving quality thresholds.

Evidence classes:

- `CANONICAL_SPEARPOINT_V1`
- run1/run2 receipt link
- reward calculation
- persisted bounded state delta

### 3. Governance sits above cognition

Safe claim:

BIZRA’s strongest novelty is not more reasoning depth. It is that cognition remains subordinate to governance, thresholds, and receipt continuity.

Evidence classes:

- fail-closed runtime boundaries
- approved-only aggregation
- receipt-driven autopoiesis observation
- canonical delivery truth

## What Must Move Out Of “Present Tense”

The following claims should be demoted from current-state wording unless supported by new evidence:

- network-wide anonymity or anti-tracing guarantees
- full adversarial multi-node BFT resilience
- poisoning resistance at realistic scale
- production-grade federated privacy theorems
- state-of-the-art cognition quality claims beyond the spearpoint artifact

## New Paper Structure

### Pillar 1. Topology Claim

CMN is a membrane-mediated topology distinct from client-server, peer-to-peer, and federation.

Minimum support:

- local sovereignty plane
- membrane governance plane
- network participation plane
- explicit crossing invariant

### Pillar 2. Proof Kernel Claim

The membrane satisfies a minimal kernel of formally tractable properties:

- fail-closed routing
- constitutional acceptance
- authenticated crossing
- provenance recording
- tamper-evident receipt chaining

Minimum support:

- proof kernel doc
- theorem statements
- mapping to concrete code boundaries

### Pillar 3. Canonical Evidence Claim

Governed self-optimization can be empirically demonstrated via a bounded artifact rather than broad benchmark rhetoric.

Minimum support:

- spearpoint artifact
- chain verification
- bounded delta
- replay-visible effect

## Required New Sections In The Paper

### Proof Kernel and Mechanisation Plan

Add a section that states exactly which properties will be mechanised in Coq or Isabelle, what the minimal kernel is, and how those theorems map onto concrete runtime boundaries.

### Adversarial Validation Program

Add a section describing bounded adversarial experiments:

- poisoning against retrieval and memory canonization
- Byzantine participation in node contribution
- provenance traceback and receipt forensics
- fail-closed rejection preservation as evidence

### Membrane Tax

Add a systems-style section that quantifies the governance overhead:

- canonical bytes and signing
- receipt building
- event membrane fanout
- Node0 breath and delivery persistence

The key rule is to separate governance tax from task work.

## Evidence Pack Requirements For Submission

Before a vNext paper is submitted, produce a companion evidence pack containing:

- the spearpoint artifact bundle
- membrane tax benchmark JSON
- delivery scorecard
- proof-kernel theorem list
- adversarial validation protocol
- explicit list of unsatisfied proof obligations

## Proof Roadmap

### M0. Present paper

- topology thesis
- proof kernel definitions
- spearpoint empirical evidence
- honest limitation framing

### M1. Mechanised membrane kernel

- formalize fail-closed route theorem
- formalize accept-implies-invariants theorem
- formalize receipt-chain tamper-evidence theorem

### M2. Bounded adversarial evaluation

- simulated malicious SAT participation
- poisoned retrieval contribution tests
- provenance traceback artifacts

### M3. Membrane tax publication

- latency and RSS breakdown
- strict and default gate outputs
- per-stage governance cost table

## Editorial Guidance

- Reduce broad literature claims unless directly tied to the membrane thesis.
- Keep advanced cognition components in the paper only where they are connected to governed receipts.
- Prefer “proof obligation” over “planned feature.”
- Treat Ihsān as a truth-label discipline: present what is proven, not what is wished.

## Closing Position

The paper should not try to prove all of CMN at once.

It should prove one sharper claim:

**A constitutionally governed membrane can be implemented as a fail-closed, receipt-native execution topology, and that topology can support bounded self-improvement without surrendering evidence integrity.**
