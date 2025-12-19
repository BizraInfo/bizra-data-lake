# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA CodeRabbit Review Instructions
# ═══════════════════════════════════════════════════════════════════════════════
# This file provides detailed instructions for AI-powered code review
# Aligned with the BIZRA Elite Implementation Blueprint v4.0
# ═══════════════════════════════════════════════════════════════════════════════

## 🎯 Mission Statement

BIZRA is a decentralized intelligence operating system built on three pillars:
1. **Technical Excellence** (SNR-gated, fail-closed)
2. **Human Impact** (Ihsān, benevolence)  
3. **Ethical Integrity** (Adl, Amānah)

Every code review must evaluate contributions against these pillars.

---

## 📋 Review Checklist

### 1. Fail-Closed Verification
- [ ] All error handlers present (no unhandled exceptions)
- [ ] Errors fail visibly, never silently
- [ ] Fallback behaviors are explicit and auditable
- [ ] No `pass` or `...` in exception handlers without justification

### 2. Receipt-Native Compliance
- [ ] All decisions emit structured receipts
- [ ] Receipts include: action, rationale, confidence, trace_id
- [ ] No mutation without corresponding receipt
- [ ] Receipt tables are append-only (no UPDATE/DELETE)

### 3. SNR Quality Gates
- [ ] SNR calculations use approved formulas
- [ ] Low-SNR decisions don't execute (gated)
- [ ] SNR thresholds documented for each decision type
- [ ] Probe failures trigger FAIL_CLOSED, not silent retry

### 4. Sovereignty First
- [ ] No external network calls without explicit opt-in
- [ ] Default embedders use NullEmbedder (no network)
- [ ] API keys never hardcoded
- [ ] External dependencies clearly documented

### 5. Security Posture
- [ ] Input validation on all user-facing endpoints
- [ ] No SQL injection vulnerabilities
- [ ] No prompt injection vectors
- [ ] Secrets use environment variables
- [ ] Authentication/authorization enforced

---

## 🏛️ Architecture Awareness

### Layer 1: Citadel Kernel
**Files**: `bizra_kernel/**`, `core/**`
**Criticality**: MAXIMUM
**Review Focus**:
- Memory management and cleanup
- Thread safety
- Resource limits
- Panic handlers

### Layer 2: Knowledge Substrate
**Files**: `kg/**`, `migrations/**`
**Criticality**: HIGH
**Review Focus**:
- Schema integrity
- Query performance
- Embedding sovereignty
- Graph traversal limits

### Layer 3: Agent Pool (PAT)
**Files**: `agents/**`
**Criticality**: HIGH
**Review Focus**:
- Agent role compliance
- Reasoning traces
- SAPE probe integration
- Decision receipts

### Layer 4: API Gateway
**Files**: `api/**`
**Criticality**: HIGH
**Review Focus**:
- Authentication
- Rate limiting
- Input sanitization
- Error responses

### Layer 5: Observability
**Files**: `dashboards/**`, `tools/**`
**Criticality**: MEDIUM
**Review Focus**:
- Metrics accuracy
- Log sanitization
- Trace correlation

---

## 📊 Ihsān Scorecard

Rate each PR on these dimensions (1-5 scale):

| Dimension | Question | Weight |
|-----------|----------|--------|
| **Technical Excellence** | Is the code clean, efficient, tested? | 30% |
| **Human Impact** | Does this serve user flourishing? | 25% |
| **Justice** | Are there bias risks? Accessibility gaps? | 25% |
| **Trustworthiness** | Is it auditable, reproducible? | 20% |

**Scoring Guide**:
- 5.0: Exemplary (exceeds all standards)
- 4.0-4.9: Strong (meets all, exceeds some)
- 3.0-3.9: Acceptable (meets minimum)
- 2.0-2.9: Needs Work (gaps to address)
- 1.0-1.9: Reject (significant issues)

**PR Approval Threshold**: ≥3.5 weighted average

---

## 🚨 Immediate Flags

Stop the PR if you find:

### Security (P0 - Block Immediately)
- Hardcoded secrets or API keys
- SQL injection vulnerabilities
- Prompt injection vectors
- Missing authentication
- Unbounded resource consumption

### Integrity (P1 - Require Changes)
- Silent error handling
- Missing receipts for decisions
- Mutable receipt tables
- Unvalidated external input

### Quality (P2 - Strongly Recommend)
- Missing tests for critical paths
- No error scenarios tested
- Documentation drift
- Inconsistent naming

---

## 🔄 PAT Agent Review Guide

When reviewing agent implementations:

| Agent Type | Primary Role | Review Focus |
|------------|--------------|--------------|
| **Guardian** | Safety + Consistency | Verify safety checks, consensus logic |
| **Prime** | Strategy + Synthesis | Check integration patterns, orchestration |
| **Scholar** | Evidence + Rigor | Validate citations, evidence chains |
| **Artisan** | Innovation + Novelty | Review creative bounds, safety rails |
| **Merchant** | Economics + Incentives | Check tokenomics, fairness |
| **Mirror** | User Advocacy | Verify user agency preservation |

---

## 📝 Comment Templates

### Requesting Changes
```markdown
🔴 **FAIL_CLOSED Required**

This code path can fail silently:
```python
# Current (problematic)
try:
    result = risky_operation()
except Exception:
    pass  # Silent failure!
```

Suggested fix:
```python
# BIZRA compliant
try:
    result = risky_operation()
except Exception as e:
    emit_receipt(action="risky_operation", result="FAILED", error=str(e))
    raise OperationFailedError(f"risky_operation failed: {e}") from e
```
```

### Security Issue
```markdown
🚨 **SECURITY: Potential Prompt Injection**

User input flows directly to agent prompt without sanitization:
- File: `api/query.py:45`
- Risk: Malicious input could manipulate agent behavior

Required: Apply `sanitize_user_input()` before prompt construction.
See: `bizra_kernel/security/sanitizers.py`
```

### Positive Feedback
```markdown
✅ **Excellent Receipt Implementation**

This follows BIZRA best practices:
- Structured receipt with all required fields
- Proper trace_id propagation
- Append-only insertion
- Clear audit trail

Ihsān Score: 4.5/5.0
```

---

## 🔗 Related Documents

- [Elite Implementation Blueprint](../BIZRA-Elite-Implementation-Blueprint-v4.0.md)
- [Lexicon Ledger](../BIZRA_Lexicon_Ledger_v0.2.0_SEALED.md)
- [Model Expectations](../BIZRA_Model_Expectations_v1.0.md)
- [Trusted System Map](../BIZRA_Trusted_System_Map_v1.0.1.md)

---

## 📞 Escalation Path

1. **Standard Issues** → CodeRabbit auto-review + PR author
2. **Security Issues** → Tag @security-team
3. **Architecture Decisions** → Tag @sape-council
4. **Governance Changes** → Require SAT Council approval (3-of-7)

---

*"Excellence with benevolence, justice with transparency, trustworthiness by design."*
