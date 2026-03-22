# CMN Proof Kernel

Date: 2026-03-22  
Status: Mechanisation-ready specification  
Scope: Minimal theorem set for Constitutional Membrane Networking

## Objective

Define the smallest stable proof kernel that can be mechanised independently of the full BIZRA implementation while still proving the central membrane thesis.

## Kernel Scope

The proof kernel models:

- membrane requests
- membrane decisions
- constitutional verification outcomes
- authenticated receipts
- receipt-chain linkage

It does not attempt to mechanise the whole runtime, full federation, or full economics layer.

## Core Types

### Request

Represents an attempted membrane crossing.

Fields:

- requester identity
- requested action
- verification context
- prior receipt hash

### Decision

One of:

- `Accept`
- `Reject`
- `DegradedReject`

### VerificationResult

One of:

- `Verified`
- `Failed`
- `Unknown`

### Receipt

Contains:

- canonical payload bytes
- signer identity
- signature
- previous receipt hash
- current receipt hash
- constitutional verdict

## Kernel Definitions

### Definition 1. Fail-Closed Routing

If verification is not complete and positive, the membrane must reject.

### Definition 2. Constitutional Acceptance

Any accepted request must satisfy all configured invariants.

### Definition 3. Authenticated Crossing

Any accepted crossing must be bound to a signer and verifiable signature.

### Definition 4. Provenance Recording

Any crossing attempt that produces a receipt must link to the prior receipt hash.

### Definition 5. Tamper-Evident Chain

If any receipt in a chain is modified, chain verification fails at or after that receipt.

## Theorem Set

### T1. FailClosedRejectsUnknown

If `verify(request) != Verified`, then `route(request) = Reject`.

### T2. AcceptImpliesConstitutional

If `route(request) = Accept`, then `all_invariants_hold(request) = true`.

### T3. ReceiptVerificationSoundness

If a receipt verifies successfully, then its signature and canonical payload bytes correspond to the claimed signer and payload.

### T4. ChainTamperEvidence

If any receipt payload or link in a verified chain is altered, chain verification fails for that receipt or a subsequent receipt.

### T5. LinkMonotonicity

If receipt `r_n` verifies and links to `r_(n-1)`, then the chain prefix through `r_(n-1)` is a necessary precondition for `r_n` verification.

## Code Boundary Mapping

| Kernel concept | Production boundary |
| --- | --- |
| fail-closed route | `core/sovereign/api.py`, runtime authority checks |
| constitutional acceptance | `core/sovereign/helix3.py`, FATE and threshold gating |
| authenticated receipt | proof-engine signing and canonicalisation modules |
| provenance recording | Node0 receipts, canonical delivery receipts, spearpoint chain files |
| tamper-evident chain | artifact receipt hashes and chain verification |

## Mechanisation Strategy

### Phase A. Small-model proof

Mechanise a minimal state machine with:

- abstract request type
- abstract verifier
- route function
- receipt constructor
- chain verifier

This phase should target theorem statements T1-T5 only.

### Phase B. Code correspondence

Prove or test correspondence between:

- kernel route rules and runtime boundary behavior
- kernel receipt structure and implementation receipt format
- kernel chain model and artifact-chain validator

### Phase C. Adversarial extension

Extend the model with:

- malicious receipt mutation
- malformed signatures
- missing authority context
- replay and reordered receipt attempts

## Proof Assistant Options

### Coq / Rocq

Best fit for:

- inductive route semantics
- theorem-first presentation
- extraction-friendly proof kernels

### Isabelle

Best fit for:

- structured theorem development
- archive-backed cryptography formalisms
- higher-level protocol reasoning

The project should pick one primary assistant for the kernel and use the other only if it materially improves reviewability or library leverage.

## Deliverables

- proof-kernel theorem file set
- README mapping theorem names to production boundaries
- generated theorem summary for the paper appendix
- artifact demonstrating receipt-chain tamper rejection

## Definition Of Done

The kernel is “paper-ready” when:

- all five theorems are stated precisely
- at least T1 and T4 are mechanised
- the mapping to code boundaries is documented
- the paper cites the kernel as a minimal verified core, not as total-system proof
