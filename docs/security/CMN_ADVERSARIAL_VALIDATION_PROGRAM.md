# CMN Adversarial Validation Program

Date: 2026-03-22  
Status: Validation design  
Scope: Bounded adversarial evaluation of the CMN membrane thesis

## Objective

Test whether the constitutional membrane preserves truth, provenance, and fail-closed behavior under adversarial participation without claiming more than the evidence supports.

## Primary Threat Classes

### 1. Retrieval poisoning

Question:

Can a small number of malicious contributions cause poisoned artifacts to become retrieved or canonized?

### 2. Byzantine participation

Question:

Can malicious participants distort collective outcomes, routing, or receipt acceptance without being detected or rejected?

### 3. Traceability and provenance abuse

Question:

Can a malicious participant inject artifacts in a way that avoids provenance traceback or creates ambiguity in responsibility?

### 4. Silent rejection loss

Question:

When the membrane rejects malicious or malformed inputs, are those failures preserved as evidence rather than disappearing into logs?

## Evaluation Principles

- bounded experiments over grandiose scale claims
- reproducible input sets
- explicit malicious node percentages
- JSONL receipt outputs as first-class artifacts
- constitutional thresholds held constant across runs

## Validation Matrix

| Scenario | Nodes | Malicious share | Goal | Success signal |
| --- | --- | --- | --- | --- |
| Retrieval poisoning | 100 | 10% | poison top-k retrieval | poison rejected or traceable |
| Retrieval poisoning stress | 100 | 20% | increase attack pressure | bounded degradation with evidence |
| Byzantine contribution | 50 | 10% | distort collective contribution | rejected or isolated with receipts |
| Byzantine contribution stress | 50 | 20% | push agreement edges | explicit failure evidence, no silent acceptance |
| Provenance attack | 25 | 10% | obscure source identity | traceback still possible from receipts |
| Dead-letter preservation | 1 local + simulated adversaries | N/A | force malformed crossings | dead letters and canonical evidence persist |

## Metrics

### Retrieval metrics

- poisoning success rate
- top-k poisoned artifact rate
- generated-output corruption rate

### Governance metrics

- constitutional rejection rate
- dead-letter rate
- accepted malicious artifact count

### Provenance metrics

- traceback completion rate
- mean hops to malicious source identification
- receipt-chain integrity pass/fail

### System metrics

- latency under adversarial load
- RSS growth under adversarial load
- event membrane failure rate

## Required Artifacts

Each experiment run should emit:

- input manifest
- malicious node manifest
- query set
- raw result JSONL
- dead-letter JSONL
- provenance traceback report
- summary Markdown and machine-readable scorecard

## Phase Plan

### Phase 1. Single-node adversarial membrane

Focus:

- malformed receipt injection
- invalid signature attempts
- delivery dead-letter preservation

Goal:

Prove the local membrane never silently accepts malformed crossings.

### Phase 2. Simulated multi-node poisoning

Focus:

- poisoned knowledge contributions
- retrieval corruption attempts
- provenance traceback

Goal:

Quantify how often poisoning changes retrieval and whether provenance remains usable.

### Phase 3. Byzantine participation program

Focus:

- malicious contribution behavior
- agreement distortion attempts
- rejection and isolation semantics

Goal:

Show bounded failure with receipts instead of narrative resilience claims.

## Publication Rule

The paper should treat this program as:

- present-tense design if not yet executed
- present-tense empirical evidence only after artifact publication

## Definition Of Done

The adversarial validation program becomes paper-grade when:

- at least one poisoning experiment is executed with published artifacts
- at least one Byzantine simulation produces traceback-ready evidence
- failed membrane crossings are shown to persist as dead-letter artifacts
- all claims are labeled as proven, staged, or future work
