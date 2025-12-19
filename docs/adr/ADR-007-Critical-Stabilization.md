# ADR-007: Critical Stabilization Phase

**Status:** Accepted  
**Date:** 2025-01-18  
**Deciders:** BIZRA Core Team  
**Branch:** feature/critical-stabilization

## Context

Following the v1.1.0-blueprint release (250 files, 33,000+ lines), a comprehensive system audit identified 18 gaps across architecture, security, and performance. The most critical finding was **GAP-002**: all five SAT (System Agentic Team) validators returned `true` unconditionally, effectively bypassing the entire safety gate.

### Critical Gaps Identified

| Gap ID | Severity | Component | Issue |
|--------|----------|-----------|-------|
| GAP-002 | 🔴 Critical | SAT | All validators return `true` - no rejection capability |
| GAP-003 | 🔴 Critical | MCP | No tool allowlist or timeout |
| GAP-004 | 🔴 Critical | A2A | No delegation depth limit or agent blocklist |
| GAP-011 | 🟡 Medium | CI/CD | No performance regression gates |

## Decision

### 1. SAT VETO Consensus Model

We adopt a **VETO consensus model** for SAT validation:

- **Any rejection blocks the entire request** (fail-safe default)
- Security and Ethics rejections are **absolute** - no override possible
- Performance and Resource rejections can be waived in dev mode only

```rust
pub enum RejectionCode {
    SecurityThreat,        // Absolute VETO
    EthicsViolation,       // Absolute VETO
    PerformanceBudgetExceeded,
    ConsistencyFailure,
    ResourceConstraintViolated,
    Quarantine,            // Ethics edge case - requires human review
}
```

**Rationale:** The previous model where all validators approved by default created a dangerous illusion of safety. A fail-closed system is essential for agentic AI that may take autonomous actions.

### 2. Security Blocklists

Pattern-based blocklists prevent known dangerous operations:

**SECURITY_BLOCKLIST (15 patterns):**
- `rm -rf`, `sudo`, `chmod 777`
- `eval(`, `exec(`, `__import__`
- SQL injection patterns: `'; DROP`, `OR 1=1`
- XSS patterns: `<script>`, `javascript:`
- Shell exploitation: `curl | sh`, `wget | bash`

**ETHICS_BLOCKLIST (10 patterns):**
- `harm`, `attack`, `exploit`
- `deceive`, `manipulate`, `impersonate`
- `illegal`, `bypass security`

**Rationale:** Defense in depth - even if prompts are sophisticated, explicit blocklists catch obvious violations early.

### 3. MCP Tool Security Controls

```rust
pub const TOOL_BLOCKLIST: &[&str] = &[
    "execute_shell",
    "run_arbitrary_code",
    "access_filesystem_root",
    "modify_system_files",
    "network_raw_access",
];

// Defaults
const DEFAULT_TIMEOUT_MS: u64 = 30_000;      // 30 seconds
const MAX_OUTPUT_SIZE: usize = 1_048_576;    // 1 MB
```

**Allowlist approach:** Only explicitly allowed tools can be called. Unknown tools are blocked by default.

### 4. A2A Delegation Controls

```rust
pub const AGENT_BLOCKLIST: &[&str] = &[
    "untrusted_external_agent",
    "deprecated_legacy_agent",
    "sandbox_escape_agent",
];

pub const MAX_DELEGATION_DEPTH: u8 = 5;
const DEFAULT_TIMEOUT_MS: u64 = 60_000;  // 60 seconds
```

**Depth tracking:** Prevents infinite delegation loops and runaway agent chains.

### 5. CI Performance Gates

Three new gates prevent performance regressions:

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| Compile time | < 120s | Detects unnecessary dependencies |
| Test time | < 60s | Prevents test bloat |
| Binary size | < 50 MB | Prevents feature creep |

### 6. Signed Evidence Receipts

All CI runs on `main` generate cryptographically-signed evidence:

```json
{
  "schema": "bizra-ci-evidence-v1",
  "timestamp": "2025-01-18T...",
  "commit_sha": "...",
  "gates_passed": [...],
  "integrity_hash": "sha256:..."
}
```

## Consequences

### Positive

- **SAT now actively rejects** dangerous requests
- **Defense in depth** via blocklists + allowlists
- **Auditability** via signed evidence
- **15 new tests** proving rejection capability
- **Ihsān compliance** maintained (0.90+ threshold)

### Negative

- Increased complexity in SAT orchestration
- Potential false positives on edge cases (mitigated by Quarantine path)
- Performance overhead from pattern matching (~negligible)

### Neutral

- Existing legitimate requests unaffected
- YAML/JSON configurations unchanged

## Phase 2: PAT↔SAT Runtime Activation

### 7. FATE (Fail-Safe Agentic Trust Escalation)

A new `fate.rs` module handles escalation and quarantine:

```rust
pub enum EscalationLevel {
    Low,       // Auto-resolved
    Medium,    // Logged for review
    High,      // Requires human review
    Critical,  // Immediate block, security notification
}
```

**Key behaviors:**
- Security/Ethics → Critical escalation
- Quarantine → High escalation (human review required)
- Performance/Resource → Low/Medium escalation
- Context sanitization: passwords, secrets, keys are redacted

### 8. Rejection Receipts

Every SAT rejection emits a machine-verifiable receipt:

```json
{
  "schema": "bizra-rejection-receipt-v1",
  "receipt_id": "REJ-20251219-000001",
  "rejection_codes": ["SECURITY_THREAT: Blocked pattern: rm -rf"],
  "escalation_id": "FATE-000001",
  "escalation_level": "CRITICAL",
  "integrity_hash": "sha256:..."
}
```

### 9. Execution Receipts

Successful flows also emit receipts for auditability:

```json
{
  "schema": "bizra-execution-receipt-v1",
  "receipt_id": "EXEC-20251219-000001",
  "synergy_score": 0.87,
  "ihsan_score": 0.92,
  "ihsan_threshold": 0.90,
  "integrity_hash": "sha256:..."
}
```

### 10. Bridge Integration

The `bridge.rs` now integrates FATE and receipts:

1. SAT validates → if rejected → FATE escalates → Receipt emitted → Error returned
2. SAT validates → if approved → PAT executes → SAT evaluates → Ihsān gate → Receipt emitted
3. All flows produce machine-verifiable evidence

## Validation

| Artifact | Verification |
|----------|-------------|
| [tests/sat_rejection_tests.rs](../../tests/sat_rejection_tests.rs) | 15/15 tests passing |
| [tests/pat_sat_runtime_tests.rs](../../tests/pat_sat_runtime_tests.rs) | 13/13 E2E tests passing |
| [src/sat.rs](../../src/sat.rs) | Real validators implemented |
| [src/fate.rs](../../src/fate.rs) | FATE escalation module |
| [src/receipts.rs](../../src/receipts.rs) | Receipt emission module |
| [src/bridge.rs](../../src/bridge.rs) | FATE + receipts integration |
- **Ihsān Constitution:** `constitution/ihsan_v1.yaml`

## References

- OWASP Agentic AI Security Guidelines (2024)
- NIST AI Risk Management Framework
- Google's Secure AI Framework (SAIF)
