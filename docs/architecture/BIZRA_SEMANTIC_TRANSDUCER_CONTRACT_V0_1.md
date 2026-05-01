# BIZRA Semantic Transducer Contract v0.1

**Date:** 2026-05-01 GST
**Status:** PLANNED -> first executable contract slice
**Scope:** Node0 / Mumu-DEMA trust boundary before daemon start.
**Truth label:** DERIVED until the contract is wired into mission execution.

---

## 1. Purpose

BIZRA must let Mumu speak naturally without letting model text become trusted
state, permission, proof, or action. The Semantic Transducer Contract defines
that boundary.

The LLM is a semantic transducer:

- It may parse messy human language into structured candidate intent.
- It may propose explanatory trajectories.
- It may summarize deterministic receipts for a human.
- It is not the trusted authority for truth, permission, key custody, or proof.

The trusted path is:

```text
human language
-> untrusted parser / LLM output
-> RawParsedClaim
-> deterministic validation
-> Claim
-> deterministic FATE gate
-> scoped execution in a later slice
-> deterministic receipt
-> explanatory SemanticSurface
```

`Claim` is the trust perimeter. Nothing before `Claim` is trusted.

---

## 2. Core Doctrine

### LLM As Peripheral

The LLM may perform reasoning-like language computation, but BIZRA does not
trust it as the authority of truth, permission, proof, or state mutation.

The LLM must never:

- execute tools directly
- decide admissibility
- hold or receive signing keys
- mutate memory or mission state directly
- create a trusted receipt
- mark a claim as proven
- override FATE, Ihsan, consent, or scope gates

### Claim As Trust Perimeter

`RawParsedClaim` is untrusted parser output. It may be produced by an LLM, a
rule parser, a CLI form, or a future multimodal adapter.

`Claim` is the first trusted representation. It exists only after deterministic
validation:

- parser identity is injected by the system wrapper
- intent is normalized to a known enum or `intent.unresolved`
- evidence weight is computed by BIZRA code
- scope is sealed
- step resource use is checked against requested scope
- sub-claim scope is proven not to expand beyond the parent claim

### FATE As Deterministic Gate

The FATE gate in this contract is pure and deterministic. Given the same
`Claim` and `ConstitutionalPolicy`, it returns the same `GateDecision`.

For v0.1:

- `intent.unresolved` always escalates.
- evidence below the policy floor escalates.
- admissible known intents with enough evidence are permitted.

This v0.1 gate does not replace the older sovereign FATE engines. It is a
small Node0/DEMA claim-boundary contract that later slices can adapt into the
broader runtime.

### Receipts Prove Process Integrity

Receipt descriptors prove process integrity only. They can state that a claim
was validated, gated, and recorded. They must not claim that the outcome was
correct, optimal, morally perfect, or externally true.

### SemanticSurface Is Explanatory

Natural-language summaries are display surfaces, not truth sources. They may
explain a receipt or gate decision to Mumu, but they cannot redefine the claim,
gate decision, policy, receipt, or evidence.

---

## 3. Model-Invariant Safety Vs Model-Dependent Utility

Safety properties must hold regardless of model quality:

- raw parser output remains untrusted
- parser identity is system-injected
- evidence weight is deterministic
- unresolved intent escalates
- scope cannot expand in sub-claims
- receipts do not claim correctness
- semantic summaries are labeled untrusted/explanatory

Utility properties may vary by model:

- how well natural language is parsed
- how useful step suggestions are
- how readable explanations are
- how quickly ambiguity is reduced

BIZRA can still operate through structured CLI claims without any LLM.

---

## 4. v0.1 Interfaces

The first code slice defines:

- `RawParsedClaim`
- `Claim`
- `GateDecision`
- `ResourceScope`
- `StepDescriptor`
- `ConstitutionalPolicy`
- `MissionReceiptDescriptor`
- `SemanticSurface`
- `IntentParser`
- `validate_raw_claim(...)`
- `fate_gate(...)`

These interfaces do not start the DEMA daemon, do not execute tools, do not
bulk-ingest memory, and do not publish public claims.

---

## 5. Known v0.1 Limits

`compute_evidence_weight(...)` is intentionally a placeholder. It is
deterministic and ignores model-supplied scores, but the real evidence policy
must later define which evidence types count, how they are weighted, and which
receipt/proof refs are required for action-bearing claims.

`ConstitutionalPolicy` includes stubs for:

- `zann_zero`
- `riba_zero`
- `gini_threshold`

Those fields are visible now so future policy work does not silently invent a
separate schema, but they are not enforced by the v0.1 gate.

`Claim` records are immutable value records for validation and gate input. They
are not intended to be used as set members or dictionary keys.

---

## 6. Implementation Bounds

This contract slice must not:

- start Mumu-DEMA daemon
- start Node1
- publish Third Fact
- bulk-ingest memory
- enable destructive actions
- wire a full mission executor
- let model output become a trusted `Claim` directly

The next slice after this contract should surface FATE/status proof-of-health
metadata before any actual Relief Mode daemon launch.

---

## 7. Single Current Instruction

Canonical implementation instruction for v0.1:

```text
Implement the Semantic Transducer Contract as architecture doc + minimal
types/tests only. Keep model output untrusted, validate into Claim, gate
deterministically, label semantic summaries as explanatory, and stop before
daemon start or mission execution.
```

