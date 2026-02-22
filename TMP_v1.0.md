# BIZRA Technical Master Plan (TMP) v1.0

**بسم الله الرحمن الرحيم**

**Document Classification:** Internal — Unified Source of Truth  
**Version:** 1.0.0  
**Date:** 2026-02-21  
**Status:** ACTIVE (authoritative for priority and conflict resolution)

> "Every seed has infinite potential." — البذرة

This file is the priority authority for current BIZRA execution. When conflicts occur between plans/specs/docs, this file resolves them. If unresolved, the decision framework in Part 10 governs.

## Part 1: Foundation

### 1.1 Core Principles (Non-Negotiable)
1. Sovereignty (local-first ownership).
2. Ihsan (measured quality, not slogans).
3. Constitutional binding (policy hash and fail-closed behavior).
4. Honesty (evidence-tagged claims + uncertainty where evidence is incomplete).

### 1.2 Ihsan as Quantitative Gate
- Target: `>=0.95` for production-facing claims.
- Mathematical form:

```text
V(state) = ihsan_score(state)
For each mutation m:
V(apply(m, state)) >= V(state) - epsilon
where epsilon = 0.01
```

## Part 2: Architecture

### 2.1 Layer Model
```text
L5 Sovereignty (SAP)
L4 Agent orchestration (Node0/GENESIS/Guardian/Action)
L3 Execution rails (LLM bridge/MCP/Desktop rail)
L2 Interop transports (MCP/A2A)
L1 Model reasoning (providers)
L0 Compute/OS
```

### 2.2 Strategic Position
BIZRA differentiation is concentrated in `L3-L5`, with `L5` as the protocol sovereignty layer.

## Part 3: Evidence-Locked Current State (2026-02-21)

The following values are measured from repository artifacts and test runs in this workspace.

### 3.1 Conformance and Pilot Evidence
1. SAP v0 conformance: `22/22` passing.
2. Shadow pilot tests: `4/4` passing.

### 3.2 Corpus Truth Evidence
From `artifacts/corpus/v1/corpus_manifest.v1.json`:
1. Providers covered: `5` (`chatgpt_openai`, `claude`, `gemini_google`, `deepseek`, `perplexity`).
2. Raw conversations: `2587`.
3. Unique conversations: `1478`.
4. Raw messages: `83212`.
5. Unique messages: `31938`.
6. Duplication factor: `2.605423`.
7. Manifest hash: `78ea7f1da1fe892d38b4a7a4b115485f5c3bd6f38cd369d6d50d71da63032809`.

### 3.3 Mathematical Derived Metrics
```text
core8_coverage_ratio = providers_covered / 8 = 5/8 = 0.6250
unique_conversation_ratio = 1478/2587 = 0.571318
unique_message_ratio = 31938/83212 = 0.383815
duplicate_message_rate = 1 - unique_message_ratio = 0.616185
```

### 3.4 Truth Classification
- Implemented (artifact-verified): SAP docs/schemas/fixtures/validator, corpus manifest pipeline, shadow pilot scripts/tests.
- Partial: Core-8 corpus coverage (currently 5/8 detected).
- Hypothesis/Post-v0: cross-node transport/federation and token settlement.

## Part 4: Protocol Stack (SAP + Profile)

### 4.1 SAP v0 Scope
SAP v0 in this cycle is a specification/conformance package with no runtime protocol verb changes.

### 4.2 Canonical SAP Objects (Normative)
1. `SovereignAgentCard` (includes mandatory `compilation` block).
2. `PermitEnvelope`.
3. `MeetOpen` (strict ceilings: `50/300/65536`).
4. `MeetMessage`.
5. `Offer` (mandatory `provenance_hashes`).
6. `Disclosure` (mandatory `source_refs` + `uncertainty`).
7. `ConsentReceipt` (mandatory for any data sharing).
8. `OutcomeReceipt` (hash-chain fields).
9. `RedlineViolation`.

### 4.3 Wire Constraint
SAP remains additive metadata over existing local-first commands (`PLAN_ACTION`, `RUN_ACTION`, `ACTION_STATUS`, `ACTION_HISTORY`, `EXPLAIN`), no new verbs in v0.

## Part 5: Corpus Truth First (Execution Order)

1. Contract-lock SAP artifacts.
2. Build deterministic multi-provider corpus truth.
3. Attest counts and derived metrics.
4. Run internal-only shadow marketing loop under strict disclosure.
5. Gate release on test and safety thresholds.

## Part 6: Release Gates (Mathematical)

Define:
```text
conformance_rate = passed_conformance / total_conformance
shadow_rate = passed_shadow / total_shadow
provider_coverage = providers_covered / 8

release_gate = min(conformance_rate, shadow_rate, provider_coverage)
```

Current measured values:
```text
conformance_rate = 22/22 = 1.0000
shadow_rate = 4/4 = 1.0000
provider_coverage = 5/8 = 0.6250
release_gate = 0.6250
```

Interpretation:
- SAP and shadow safety are green.
- Core-8 corpus requirement is not complete yet; this is the blocking constraint.

## Part 7: 2026 Operational Priorities

### P0 (Immediate)
1. Complete Core-8 provider normalization coverage to move `provider_coverage` from `0.625` to `1.0`.
2. Preserve deterministic dedup reproducibility and manifest hash traceability.
3. Keep all claim docs aligned to measured artifacts.

### P1 (Next)
1. Expand internal shadow session volume while keeping fail-closed evidence behavior.
2. Upgrade KPI reporting to include distribution metrics (median/p95 latency, redline rate).

### P2 (Post-v0)
1. Runtime MEET/GO transport.
2. Federation and external audit phases.

## Part 8: Honest Boundaries

Not claimed as implemented in this cycle:
1. Public network federation.
2. Production cross-node migration.
3. External security audit completion.
4. Token settlement rollout.

## Part 9: Decision Framework

Priority hierarchy (highest to lowest):
1. Constitutional principles.
2. Safety/fail-closed behavior.
3. Honesty/evidence discipline.
4. User Zero quality.
5. Technical quality.
6. Shipping speed.

When conflicts remain unresolved after evidence review, founder authority decides and this file is updated with rationale.

## Part 10: Canonical Evidence Links

1. `scripts/spec/validate_sap_v0.py`
2. `tests/conformance/sap_v0/`
3. `artifacts/corpus/v1/corpus_manifest.v1.json`
4. `artifacts/corpus/v1/dedup_report.v1.json`
5. `scripts/pilot/run_user_zero_shadow.py`
6. `tests/pilot/test_shadow_marketing_flow.py`
7. `docs/internal/SAP_V0_EVIDENCE_MATRIX.md`
8. `STATUS.md`

---

This TMP is a living operational authority. Claims in this document must remain evidence-backed and uncertainty-tagged where proof is incomplete.
