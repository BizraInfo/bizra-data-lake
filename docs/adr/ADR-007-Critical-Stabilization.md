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

## Validation

| Artifact | Verification |
|----------|-------------|
| [tests/sat_rejection_tests.rs](../../tests/sat_rejection_tests.rs) | 15/15 tests passing |
| [src/sat.rs](../../src/sat.rs) | Real validators implemented |
| [src/mcp.rs](../../src/mcp.rs) | Allowlist + timeout + size limit |
| [src/a2a.rs](../../src/a2a.rs) | Blocklist + depth limit + timeout |
| [phase0_integrity.yml](../../.github/workflows/phase0_integrity.yml) | Performance gates + evidence |

## Related

- **ADR-001:** Architecture overview
- **ADR-006:** Test coverage framework
- **GAP Analysis:** Internal audit document
- **Ihsān Constitution:** `constitution/ihsan_v1.yaml`

## References

- OWASP Agentic AI Security Guidelines (2024)
- NIST AI Risk Management Framework
- Google's Secure AI Framework (SAIF)
