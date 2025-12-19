# P1 Recursive Correction Loop Specification

**Document ID:** `BIZRA-P1-RCL-SPEC-v1.0.0`  
**Status:** DRAFT → REVIEW  
**Parent:** `P1_APEX_IMPLEMENTATION_FRAMEWORK.md`  
**Created:** 2025-12-19

---

## 1) Purpose

Define a **bounded, auditable feedback mechanism** that enables the SAPE (Symbolic-Abstraction-Probe-Elevation) system to self-correct while maintaining Ihsān compliance and preventing runaway recursion.

---

## 2) Scope

### In Scope
- SAPE `/v1/sape/plan` feedback loop
- Bounded retry policy
- Rejection reason schema
- Audit logging requirements
- Test specifications

### Out of Scope
- Core SAPE algorithm changes
- Multi-node federation (deferred to P1.12)
- Autonomous escalation (requires human-in-loop for P1)

---

## 3) Bounded Retry Policy

### 3.1 Retry Limits

| Context | Max Retries | Backoff | Escalation |
|---------|-------------|---------|------------|
| SAPE Plan Generation | 2 | Exponential (1s, 2s) | Human review queue |
| Ihsān Evaluation | 1 | None | Immediate block |
| Evidence Sealing | 3 | Linear (500ms) | Alert + manual seal |

### 3.2 Retry State Machine

```
┌─────────────┐
│  INITIAL    │
└──────┬──────┘
       │ submit(plan)
       ▼
┌─────────────┐
│  ATTEMPT_1  │──────────────────┐
└──────┬──────┘                  │
       │ reject(reason)          │ accept
       ▼                         │
┌─────────────┐                  │
│  ATTEMPT_2  │──────────────────┤
└──────┬──────┘                  │
       │ reject(reason)          │ accept
       ▼                         │
┌─────────────┐                  │
│  ESCALATED  │                  │
└──────┬──────┘                  │
       │ human_override          │
       ▼                         ▼
┌─────────────┐           ┌─────────────┐
│   BLOCKED   │           │  EXECUTED   │
└─────────────┘           └─────────────┘
```

### 3.3 Retry Invariants

```python
# Pseudo-code for retry logic
MAX_RETRIES = 2
BACKOFF_BASE_MS = 1000

def execute_with_retry(plan: SapePlan) -> Result:
    attempts = 0
    while attempts <= MAX_RETRIES:
        result = sape.execute(plan)
        if result.status == "ACCEPTED":
            return Result.success(result.output)
        
        # Log rejection for audit
        audit_log.record(
            event="sape_rejection",
            attempt=attempts + 1,
            plan_id=plan.id,
            rejection_reason=result.rejection_reason,
            ihsan_score=result.ihsan_score
        )
        
        if attempts < MAX_RETRIES:
            sleep(BACKOFF_BASE_MS * (2 ** attempts))
            plan = refine_plan(plan, result.rejection_reason)
            attempts += 1
        else:
            return Result.escalate(plan, result.rejection_reason)
    
    return Result.blocked(plan)
```

---

## 4) Rejection Reason Schema

### 4.1 JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://bizra.io/schemas/rejection_reason_v1.schema.json",
  "title": "SAPE Rejection Reason",
  "type": "object",
  "required": ["code", "category", "message", "timestamp"],
  "properties": {
    "code": {
      "type": "string",
      "pattern": "^RJ-[A-Z]{2}-[0-9]{3}$",
      "description": "Structured rejection code (e.g., RJ-IH-001)"
    },
    "category": {
      "type": "string",
      "enum": [
        "IHSAN_VIOLATION",
        "SOVEREIGNTY_BREACH",
        "RESOURCE_EXHAUSTION",
        "VALIDATION_FAILURE",
        "DEPENDENCY_UNAVAILABLE",
        "POLICY_CONFLICT"
      ]
    },
    "message": {
      "type": "string",
      "maxLength": 500,
      "description": "Human-readable explanation (no secrets)"
    },
    "remediation_hints": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Suggested corrections for next attempt"
    },
    "ihsan_components": {
      "type": "object",
      "properties": {
        "itqan_score": { "type": "number", "minimum": 0, "maximum": 1 },
        "birr_score": { "type": "number", "minimum": 0, "maximum": 1 },
        "adl_score": { "type": "number", "minimum": 0, "maximum": 1 },
        "amanah_score": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "plan_hash": {
      "type": "string",
      "description": "SHA256 of the rejected plan for correlation"
    },
    "attempt_number": {
      "type": "integer",
      "minimum": 1,
      "maximum": 3
    }
  }
}
```

### 4.2 Rejection Code Registry

| Code | Category | Description |
|------|----------|-------------|
| RJ-IH-001 | IHSAN_VIOLATION | Ihsān score below threshold (< 0.95) |
| RJ-IH-002 | IHSAN_VIOLATION | Harm potential detected |
| RJ-IH-003 | IHSAN_VIOLATION | Fairness constraint violated |
| RJ-SV-001 | SOVEREIGNTY_BREACH | External API dependency detected |
| RJ-SV-002 | SOVEREIGNTY_BREACH | Data egress attempted |
| RJ-RE-001 | RESOURCE_EXHAUSTION | Token budget exceeded |
| RJ-RE-002 | RESOURCE_EXHAUSTION | Memory limit approached |
| RJ-VF-001 | VALIDATION_FAILURE | Schema validation failed |
| RJ-VF-002 | VALIDATION_FAILURE | Precondition not met |
| RJ-DU-001 | DEPENDENCY_UNAVAILABLE | Required service offline |
| RJ-PC-001 | POLICY_CONFLICT | Conflicting policies detected |

---

## 5) Callback Contract

### 5.1 Endpoint Specification

**URL:** `POST /v1/sape/plan/feedback`

**Request:**
```json
{
  "plan_id": "uuid",
  "cycle_id": "uuid",
  "feedback_type": "rejection | acceptance | escalation",
  "rejection_reason": { /* RejectionReason object if rejection */ },
  "refined_plan": { /* Optional refined plan for retry */ },
  "metadata": {
    "source": "system | human",
    "timestamp": "ISO8601"
  }
}
```

**Response:**
```json
{
  "status": "acknowledged | retry_queued | blocked | escalated",
  "next_action": {
    "type": "retry | await_human | terminate",
    "deadline_utc": "ISO8601",
    "queue_position": 0
  },
  "audit_reference": "uuid"
}
```

### 5.2 Callback Invariants

1. **Idempotency:** Duplicate feedback for same `plan_id + attempt` is ignored
2. **Ordering:** Feedback processed in timestamp order per plan
3. **Timeout:** Callback must complete within 5s or fail open (log + continue)
4. **Authentication:** Requires valid `BIZRA_API_TOKEN` in header

---

## 6) Logging & Audit Requirements

### 6.1 Mandatory Log Fields

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | UUID | Unique event identifier |
| `event_type` | string | `sape_attempt`, `sape_rejection`, `sape_escalation` |
| `timestamp` | ISO8601 | Event timestamp (UTC) |
| `plan_id` | UUID | Plan being processed |
| `cycle_id` | UUID | Parent SAPE cycle |
| `attempt_number` | int | Current attempt (1-3) |
| `ihsan_score` | float | Ihsān evaluation result |
| `duration_ms` | int | Processing duration |
| `outcome` | string | `accepted`, `rejected`, `escalated`, `blocked` |

### 6.2 Forbidden Log Content (Amānah Compliance)

**NEVER LOG:**
- API tokens or credentials
- Personal identifiable information (PII)
- Raw model prompts containing user data
- Encryption keys or secrets
- Session tokens

### 6.3 Log Retention

| Log Type | Retention | Storage |
|----------|-----------|---------|
| SAPE events | 90 days | Hot storage |
| Escalations | 1 year | Warm storage |
| Ihsān violations | 2 years | Cold storage (compliance) |

---

## 7) Test Requirements

### 7.1 Unit Tests

| Test ID | Description | Acceptance |
|---------|-------------|------------|
| UT-RCL-001 | Retry counter increments correctly | Counter = attempts after N tries |
| UT-RCL-002 | Backoff timing is exponential | Delay(2) = 2 × Delay(1) |
| UT-RCL-003 | Escalation triggers at max retries | State = ESCALATED after MAX+1 |
| UT-RCL-004 | Rejection reason validates against schema | Invalid reasons rejected |
| UT-RCL-005 | Forbidden fields filtered from logs | No secrets in log output |

### 7.2 Integration Tests

| Test ID | Description | Acceptance |
|---------|-------------|------------|
| IT-RCL-001 | Full retry cycle with mock SAPE | 2 rejections → escalation |
| IT-RCL-002 | Feedback endpoint round-trip | POST returns acknowledged |
| IT-RCL-003 | Audit log completeness | All mandatory fields present |
| IT-RCL-004 | Ihsān gate blocks low scores | Score < 0.95 → rejection |
| IT-RCL-005 | Human override flow | Escalated → human accept → executed |

### 7.3 Property-Based Tests

```python
# Hypothesis-style property test
@given(attempts=integers(min_value=0, max_value=10))
def test_retry_never_exceeds_max(attempts):
    """Retry count never exceeds MAX_RETRIES."""
    result = simulate_retry_loop(attempts)
    assert result.actual_attempts <= MAX_RETRIES + 1

@given(plan=sape_plans())
def test_rejection_reason_always_valid(plan):
    """Every rejection produces schema-valid reason."""
    result = execute_plan(plan)
    if result.status == "rejected":
        assert validate_rejection_reason(result.rejection_reason)
```

---

## 8) Implementation Checklist

- [ ] Create `RejectionReason` data class with schema validation
- [ ] Implement retry state machine in `sape.py`
- [ ] Add `/v1/sape/plan/feedback` endpoint to kernel API
- [ ] Configure audit logging with field filtering
- [ ] Write unit tests (UT-RCL-001 through UT-RCL-005)
- [ ] Write integration tests (IT-RCL-001 through IT-RCL-005)
- [ ] Update `sape_runbook.md` with escalation procedures
- [ ] Add Prometheus metrics for retry/escalation rates

---

## 9) Metrics & Alerts

### 9.1 Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `sape_attempts_total` | Counter | Total SAPE attempts |
| `sape_rejections_total` | Counter | Total rejections (by reason code) |
| `sape_escalations_total` | Counter | Total escalations |
| `sape_retry_duration_seconds` | Histogram | Time per retry cycle |
| `ihsan_score_distribution` | Histogram | Ihsān score distribution |

### 9.2 Alert Rules

```yaml
groups:
  - name: sape_rcl_alerts
    rules:
      - alert: HighRejectionRate
        expr: rate(sape_rejections_total[5m]) / rate(sape_attempts_total[5m]) > 0.3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "SAPE rejection rate above 30%"
          
      - alert: EscalationBacklogGrowing
        expr: sape_escalations_total - sape_escalations_resolved_total > 10
        for: 1h
        labels:
          severity: critical
        annotations:
          summary: "More than 10 unresolved escalations"
```

---

## Appendix: Example Rejection Flow

```json
// Attempt 1: Plan submitted
{
  "plan_id": "550e8400-e29b-41d4-a716-446655440000",
  "cycle_id": "660f9511-f30c-52e5-b827-557766551111",
  "phase": "probe",
  "action": "execute_model_inference",
  "parameters": {
    "model": "external-gpt-4",
    "prompt": "..."
  }
}

// Rejection 1: Sovereignty breach
{
  "code": "RJ-SV-001",
  "category": "SOVEREIGNTY_BREACH",
  "message": "External API dependency detected: external-gpt-4",
  "remediation_hints": [
    "Use local model: bizra-7b-planner",
    "Configure sovereignty fallback policy"
  ],
  "ihsan_components": {
    "itqan_score": 0.90,
    "birr_score": 0.95,
    "adl_score": 0.88,
    "amanah_score": 0.70
  },
  "timestamp": "2025-12-19T14:30:00Z",
  "plan_hash": "sha256:abc123...",
  "attempt_number": 1
}

// Attempt 2: Refined plan
{
  "plan_id": "550e8400-e29b-41d4-a716-446655440000",
  "parameters": {
    "model": "bizra-7b-planner",  // Fixed
    "prompt": "..."
  }
}

// Acceptance: Ihsān threshold met
{
  "status": "ACCEPTED",
  "ihsan_score": 0.97,
  "execution_id": "770g0622-g41d-63f6-c938-668877662222"
}
```

---

**BIZRA P1 — Recursive Correction with Integrity**
