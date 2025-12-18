# ADR-003: FATE Recursive Correction Loop

**Status:** Accepted  
**Date:** 2025-12-19  
**Context Finding:** F-SEC-001 (Probabilistic Ethics)

## Context

The FATE Gate validates intents but previously lacked a **Recursive Correction Loop** 
to guide users back to compliance — it just blocked without explanation.

This creates poor UX and prevents learning from rejections.

## Decision

Implement **Structured Correction Feedback** with:

1. **Rejection Codes**: Standardized taxonomy (RJ-IH-001, RJ-SV-001, etc.)
2. **Correction Guidance**: Explanation + Fix Suggestion + Examples
3. **Bounded Retry**: Max 2 attempts per request hash
4. **Evidence Logging**: All rejections recorded with context

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   FATE RECURSIVE CORRECTION                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Request ──▶ [FATE Evaluate] ──▶ Decision                  │
│                     │                │                       │
│                     │                ├── APPROVE ──▶ Execute │
│                     │                │                       │
│                     │                └── REJECT ──▶ Feedback │
│                     │                       │                │
│                     │              ┌────────┴────────┐       │
│                     │              │  CORRECTION     │       │
│                     │              │  FEEDBACK       │       │
│                     │              │  • Reason Code  │       │
│                     │              │  • Explanation  │       │
│                     │              │  • Fix Suggest  │       │
│                     │              │  • Examples     │       │
│                     │              └────────┬────────┘       │
│                     │                       │                │
│                     │              [Retry Permitted?]        │
│                     │                 │         │            │
│                     │            YES  │         │ NO         │
│                     └──────────◀─ Resubmit    Block          │
│                      (max 2)                                 │
└─────────────────────────────────────────────────────────────┘
```

## Rejection Taxonomy

| Code | Category | Description | Retryable |
|------|----------|-------------|-----------|
| RJ-IH-001 | Ihsān | Score below threshold | ✅ |
| RJ-IH-002 | Ihsān | Intent unclear | ✅ |
| RJ-SV-001 | Sovereignty | External API dependency | ✅ |
| RJ-EG-001 | Ethics | Harmful content | ✅ |
| RJ-RS-001 | Resource | Capacity exceeded | ✅ |
| RJ-EV-001 | Evidence | Missing attestation | ✅ |
| RJ-KB-001 | Security | Kernel bypass | ❌ |

## API Endpoints

### POST /v1/fate/evaluate
Evaluate an action and receive approval or structured rejection.

**Request:**
```json
{
  "intent": "Explain how SAPE works",
  "context": "learning_session",
  "evidence_hash": null
}
```

**Response (Approved):**
```json
{
  "approved": true,
  "composite_score": 0.97,
  "verdict": "APPROVED"
}
```

**Response (Rejected):**
```json
{
  "approved": false,
  "composite_score": 0.72,
  "verdict": "REJECTED",
  "feedback": {
    "code": "RJ-IH-001",
    "explanation": "Ihsān score (0.72) is below threshold (0.95).",
    "fix_suggestion": "Reframe constructively...",
    "examples": ["Instead of X, try Y"],
    "retryable": true,
    "retry_count": 0,
    "max_retries": 2
  }
}
```

### POST /v1/sape/feedback
Submit a corrected request after receiving feedback.

**Request:**
```json
{
  "original_hash": "a1b2c3d4e5f6",
  "corrected_intent": "Identify vulnerabilities for defensive purposes",
  "context": "security_audit"
}
```

## Implementation

**File**: `core/fate.py` (enhanced with `FateEngineWithCorrection`)

```python
from core.fate import get_fate_engine

engine = get_fate_engine()

# Evaluate with feedback
seal, feedback = engine.audit_request_with_feedback(
    intent="My request",
    context=""
)

if seal.verdict == "REJECTED" and feedback.retryable:
    # Show feedback to user, get corrected input
    corrected = get_user_correction(feedback)
    seal, feedback = engine.submit_correction(
        original_hash=feedback.request_hash,
        corrected_intent=corrected
    )
```

## Invariants

- **I2**: `Action_Approved == True` ONLY IF `composite_score >= threshold`
- Max 2 retries per request hash
- All rejections logged with structured feedback
- Non-retryable codes (RJ-KB-001) immediately block

## Consequences

### Positive
- Users understand WHY they were rejected
- Clear path to compliance
- Learning opportunity from rejections
- Prevents repeated identical rejections

### Negative
- Slight complexity increase
- Retry tracking requires memory

### Risks Mitigated
- **Ethics Drift**: Structured feedback reinforces alignment
- **UX Frustration**: Clear guidance reduces confusion

## Evidence

- Implementation: `core/fate.py` (`FateEngineWithCorrection`)
- Schema: `schemas/rejection_reason_v1.schema.json`
- Spec: `docs/architecture/P1_recursive_correction_loop.md`
- Logs: `docs/evidence/fate/rejections.jsonl`

## Related

- ADR-002: URP Implementation
- F-SEC-001: Probabilistic Ethics finding
