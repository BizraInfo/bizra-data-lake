# BIZRA CI/CD Pipeline DAG — Post-Sprint-2

**Version**: Sprint 2  
**Date**: 2026-03-19  
**Author**: BIZRA Engineering  
**Workflows affected**: 18 total (16 existing + 2 new)

---

## 1. Complete Pipeline DAG

```
Trigger (push to main / feat/sprint-* | PR to main)
│
├── [PARALLEL: Stage 1 — Lint & Schema]
│   ├── lint-python          ci.yml       Ruff, Black, isort, MyPy ratchet, pip-audit
│   │   ├── SEC-001: BLAKE3 enforcement gate
│   │   ├── SEC-002: Secret hygiene gate
│   │   └── SEC-003: Exception specificity gate
│   ├── lint-rust            ci.yml       cargo fmt, clippy -D warnings
│   └── validate-schemas     ci.yml       JSON/YAML schema validation (SP-009)
│
├── [PARALLEL: Stage 2 — Fast Feedback]
│   ├── walking-skeleton     walking-skeleton.yml   compile + basic test (< 3 min)
│   └── docs-quality         docs-quality-fix.yml   ← FIXED (Sprint 2)
│       ├── Existence-checked markdown lint
│       ├── Optional docs: warn-only
│       └── Lychee offline link check (warn-only in Sprint 2)
│
├── [Stage 3 — Tests]  (needs: lint-python, lint-rust)
│   └── tests.yml
│       ├── unit              cargo test --package bizra-* (unit)
│       ├── integration       cargo test -- integration
│       ├── token             cargo test -- token
│       ├── e2e               cargo test -- e2e
│       └── coverage          tarpaulin → lcov upload
│
├── [Stage 4 — Quality Gates]  (needs: tests)
│   └── quality-spine.yml / quality-management.yml
│       ├── coverage-ratchet  SNR ≥ 0.85, Ihsan ≥ 0.95
│       ├── quality-trend     PR quality delta
│       ├── enforce           Constitutional gate
│       └── pr-summary        PR annotation
│
├── [Stage 5 — Proof Pyramid Gate]  ← NEW (Sprint 2)
│   └── proof-pyramid-gate.yml
│       ├── PP-001: Receipt Chain Integrity
│       │   ├── verify_chain() O(n) assertion
│       │   └── Tamper detection (flip byte → verify fails)
│       ├── PP-002: Sippar Encoding Verification
│       │   ├── chain_length=60 → harmonious (5-smooth)
│       │   ├── chain_length=7  → witness (irregular)
│       │   └── RegularNumber::from_u64 rejects all primes > 5
│       ├── PP-003: SMT-LIB2 Syntax Gate
│       │   ├── Balanced parentheses check
│       │   ├── Valid keywords: set-logic, declare-const, assert, check-sat
│       │   └── Rejects empty assertions
│       ├── PP-004: Fate-Binding Z3 Proofs  (requires libz3-dev)
│       │   ├── Passing scores (ihsan ≥ 0.95) → Satisfiable
│       │   ├── Failing scores (ihsan < 0.95) → Unsatisfiable
│       │   └── All 4 FATE gates tested
│       ├── PP-005: Mission → ProofSpace Bridge
│       │   ├── Terminal missions produce valid submissions
│       │   ├── Non-terminal missions are rejected
│       │   └── Ihsan strict mode enforcement
│       ├── PP-006: E2E Proof Pyramid (Layer 0→5)
│       │   └── All 7 E2E tests pass (walking skeleton for proof system)
│       └── Evidence Bundle → evidence/proof_pyramid_evidence.json
│
├── [Stage 6 — Wire Completeness Audit]  ← NEW (Sprint 2)
│   └── wire-completeness-audit.yml   (weekly + on Cargo.toml changes)
│       ├── W1: bizra-action → bizra-core
│       ├── W2: bizra-proofspace → bizra-core
│       ├── W3: bizra-proofspace → bizra-sippar
│       ├── W4: fate-binding → bizra-core
│       ├── W5: bizra-mission → bizra-core
│       ├── W6: bizra-proofspace → bizra-mission
│       ├── W7: fate-binding → bizra-proofspace
│       └── W8: bizra-action → bizra-proofspace
│           └── wire_audit_report.json artifact
│
├── [Stage 7 — Security]  (needs: tests)
│   └── ci.yml / phase56-security-gate.yml
│       ├── Trivy vulnerability scan → SARIF upload
│       ├── Container signing (cosign)
│       └── SBOM generation (syft)
│
├── [Stage 8 — Performance]  (needs: tests)
│   └── performance.yml
│       ├── Inference latency (P95 < 500 ms gate)
│       ├── Throughput (≥ 10 RPS gate)
│       ├── Memory (< 4 GB gate)
│       ├── Startup time regression
│       ├── Criterion proof pyramid benchmarks  ← NEW (Sprint 2)
│       │   ├── receipt_chain_verify  < 5 ms / 1 000 receipts
│       │   ├── jcs_canonicalize      < 100 µs
│       │   ├── compute_block_id      < 50 µs
│       │   ├── sippar_from_u64       < 1 ms / 1 000 calls
│       │   └── fate_proof_generate   < 10 ms
│       └── Regression gate (10% tolerance vs baseline)
│
├── [Stage 9 — Container Build + Sign]
│   └── ci.yml
│       ├── Docker BuildKit multi-stage build (Dockerfile.node0-omega)
│       ├── Image push to registry
│       └── cosign image signing
│
└── [Stage 10 — Evidence Accumulation → CI Summary]
    └── ci.yml / quality-spine.yml
        ├── Merge all evidence.json fragments
        ├── Proof pyramid evidence appended (Sprint 2)
        └── Annotated CI summary posted to PR
              │
              ▼
        [Deploy Pipeline]  (needs: all Stage 1–10 gates pass)
        └── deploy.yml
            ├── Build production image
            ├── Deploy to staging (Argo CD)
            ├── Smoke tests (POST /health, /version, /readiness)
            ├── Production quality gate (Ihsan ≥ 0.95 on staging)
            ├── Canary rollout (10% → 50% → 100%)
            ├── Verify (5-min observation window)
            └── Rollback (automatic on latency P95 > 500 ms or error rate > 1%)
```

---

## 2. Gate Summary Table

| Stage | Gate ID | Workflow | Type | Blocking | Sprint |
|-------|---------|----------|------|----------|--------|
| 1 | SEC-001 | ci.yml | Security | Yes | S1 |
| 1 | SEC-002 | ci.yml | Security | Yes | S1 |
| 1 | SEC-003 | ci.yml | Security | Yes | S1 |
| 2 | DOCS-Q | docs-quality-fix.yml | Quality | Yes (fixed) | S2 |
| 3 | UNIT | tests.yml | Correctness | Yes | S1 |
| 3 | INT | tests.yml | Correctness | Yes | S1 |
| 3 | E2E | tests.yml | Correctness | Yes | S1 |
| 3 | COV | tests.yml | Quality | Yes | S1 |
| 4 | SNR ≥ 0.85 | quality-spine.yml | Constitutional | Yes | S1 |
| 4 | IHSAN ≥ 0.95 | quality-spine.yml | Constitutional | Yes | S1 |
| **5** | **PP-001** | **proof-pyramid-gate.yml** | **Correctness** | **Yes** | **S2** |
| **5** | **PP-002** | **proof-pyramid-gate.yml** | **Correctness** | **Yes** | **S2** |
| **5** | **PP-003** | **proof-pyramid-gate.yml** | **Correctness** | **Yes** | **S2** |
| **5** | **PP-004** | **proof-pyramid-gate.yml** | **Formal** | **Yes** | **S2** |
| **5** | **PP-005** | **proof-pyramid-gate.yml** | **Integration** | **Yes** | **S2** |
| **5** | **PP-006** | **proof-pyramid-gate.yml** | **E2E** | **Yes** | **S2** |
| **6** | **W1–W8** | **wire-completeness-audit.yml** | **Architecture** | **Yes** | **S2** |
| 7 | TRIVY | phase56-security-gate.yml | Security | Yes | S1 |
| 7 | SBOM | ci.yml | Compliance | Yes | S1 |
| 8 | P95 < 500 ms | performance.yml | Performance | Yes | S1 |
| 8 | RPS ≥ 10 | performance.yml | Performance | Yes | S1 |
| 8 | MEM < 4 GB | performance.yml | Performance | Yes | S1 |
| **8** | **BENCH-PP** | **performance.yml** | **Performance** | **Warn** | **S2** |
| 9 | COSIGN | ci.yml | Supply-chain | Yes | S1 |

**Sprint 2 additions: 9 gates (PP-001–006, W1–W8 audit, BENCH-PP, DOCS-Q fix)**

---

## 3. PMBOK Knowledge Area Mapping

Each CI gate enforces a specific PMBOK 7th-edition performance domain or
knowledge area. This mapping satisfies audit requirements for ISO 33001
process quality assurance.

### 3.1 Quality Management (PMBOK: Project Quality Management)

| Gate | PMBOK Process | Justification |
|------|---------------|---------------|
| SNR ≥ 0.85 | Plan Quality Management | Signal-to-noise ratio is a measurable quality metric; defines the project's quality baseline. |
| IHSAN ≥ 0.95 | Manage Quality | Constitutional quality floor; ensures every increment meets the excellence standard before integration. |
| Coverage ratchet | Control Quality | Prevents regression in test coverage; enforces continuous improvement (Kaizen). |
| PP-001 Receipt Chain | Control Quality | Verifies tamper-evident audit trail integrity — a direct quality control check on the evidence record. |
| PP-003 SMT-LIB2 Syntax | Manage Quality | Formal assertion syntax validation ensures quality of the specification artifacts. |
| DOCS-Q | Manage Quality | Documentation quality is a deliverable quality attribute; missing or malformed docs are defects. |

### 3.2 Risk Management (PMBOK: Project Risk Management)

| Gate | PMBOK Process | Justification |
|------|---------------|---------------|
| SEC-001 BLAKE3 | Identify Risks | Enforces cryptographic integrity on PCI/proof paths; detects supply-chain tampering risk. |
| SEC-002 Secret hygiene | Identify Risks | Credential leak prevention; maps to OWASP A02 and PMBOK risk identification for information security. |
| SEC-003 Exception specificity | Monitor & Control Risks | Broad exception handlers mask runtime failures; specific handlers expose risk signals. |
| Trivy scan | Perform Qualitative Risk Analysis | CVE detection and scoring; direct input to risk register. |
| SBOM generation | Plan Risk Responses | Software bill of materials enables rapid response to newly disclosed vulnerabilities. |
| cosign signing | Implement Risk Responses | Image signing provides tamper evidence; prevents supply-chain substitution attacks. |
| PP-004 Z3 Proofs | Identify Risks | Formal satisfiability proofs identify score ranges that violate constitutional invariants — risk modeling at machine precision. |

### 3.3 Scope Management (PMBOK: Project Scope Management)

| Gate | PMBOK Process | Justification |
|------|---------------|---------------|
| W1–W8 Wire Audit | Validate Scope | The wire map is the architectural scope contract; the audit validates that the implemented inter-crate wiring matches the design WBS. |
| Walking skeleton | Validate Scope | Confirms the minimum viable integration compiles and executes; scope baseline for each sprint. |
| PP-006 E2E Pyramid | Validate Scope | The Layer 0→5 E2E test is the acceptance criterion for the proof pyramid feature; passing = scope delivered. |

### 3.4 Schedule Management (PMBOK: Project Schedule Management)

| Gate | PMBOK Process | Justification |
|------|---------------|---------------|
| P95 latency < 500 ms | Control Schedule | Inference latency directly constrains user-facing SLA; a regression indicates schedule risk for production readiness. |
| Throughput ≥ 10 RPS | Control Schedule | Minimum throughput to meet production load projections; gates deployment schedule. |
| BENCH-PP Criterion | Control Schedule | Proof pyramid benchmark baselines enable early detection of performance schedule risk in Sprint 3+. |
| Startup time regression | Control Schedule | Slow startup increases canary rollout duration; a 10% tolerance prevents schedule creep. |

### 3.5 Integration Management (PMBOK: Project Integration Management)

| Gate | PMBOK Process | Justification |
|------|---------------|---------------|
| PP-005 Mission Bridge | Direct & Manage Work | Validates the Mission → ProofSpace integration contract; ensures work products from WBS 1.2 and 1.3 integrate correctly. |
| Evidence accumulation | Monitor & Control Project Work | The evidence.json bundle is the CI/CD PMBOK work performance information artifact; it records what was done and the results. |
| Deploy pipeline | Perform Integrated Change Control | Staging → canary → production progression enforces formal change control; rollback implements the approved change reversal procedure. |
| PP-001–PP-006 gates | Close Project/Phase | Each sprint's proof pyramid gate must pass before the sprint is considered closed; gates are the sprint acceptance criteria. |

### 3.6 Resource Management (PMBOK: Project Resource Management)

| Gate | PMBOK Process | Justification |
|------|---------------|---------------|
| Memory < 4 GB | Control Resources | Physical memory is a constrained resource; the gate prevents resource overrun. |
| Container multi-stage build | Acquire Resources | The `Dockerfile.node0-omega` test stage is the CI resource definition for proof pyramid test execution. |

### 3.7 Stakeholder Management (PMBOK: Stakeholder Engagement)

| Gate | PMBOK Process | Justification |
|------|---------------|---------------|
| PR Quality Summary | Manage Stakeholder Engagement | Annotated PR comments inform developers of quality state, closing the feedback loop for immediate stakeholder (developer) engagement. |
| CI Summary markdown | Monitor Stakeholder Engagement | GitHub Actions Step Summary provides executive-level gate status without requiring CI log access. |
| DOCS-Q | Plan Stakeholder Engagement | Accurate, up-to-date documentation is the primary communication artifact for external and internal stakeholders. |

---

## 4. Gap Resolution Summary (Sprint 2)

| Gap (from grep verification) | Resolution | Gate |
|------------------------------|------------|------|
| NO receipt chain integrity CI gate | Added PP-001 job in proof-pyramid-gate.yml | PP-001 |
| NO SMT-LIB2 syntax validation gate | Added PP-003 job in proof-pyramid-gate.yml | PP-003 |
| NO Sippar encoding verification gate | Added PP-002 job in proof-pyramid-gate.yml | PP-002 |
| NO proof pyramid integration test in CI | Added PP-006 E2E job; all 7 tests required | PP-006 |
| NO formal verification (Z3 satisfiability) gate | Added PP-004 job using verified libz3-dev | PP-004 |
| NO inter-crate wire completeness audit | Added wire-completeness-audit.yml (W1–W8) | W1–W8 |
| Docs Quality = FAILURE (only failed workflow) | Replaced with docs-quality-fix.yml | DOCS-Q |

---

## 5. Environment Variables (Constitutional Thresholds)

All thresholds are declared as `env:` at workflow level to ensure single
source of truth across all jobs. They mirror `bizra-core/src/lib.rs`.

| Variable | Value | Enforced By |
|----------|-------|-------------|
| `IHSAN_THRESHOLD` | `0.95` | quality-spine.yml, proof-pyramid-gate.yml |
| `SNR_THRESHOLD` | `0.85` | quality-spine.yml, proof-pyramid-gate.yml |
| `ADL_GINI_MAX` | `0.35` | proof-pyramid-gate.yml, PP-004 |
| `P95_LATENCY_THRESHOLD_MS` | `500` | performance.yml |
| `THROUGHPUT_THRESHOLD_RPS` | `10` | performance.yml |
| `MEMORY_THRESHOLD_GB` | `4` | performance.yml |
| `REGRESSION_TOLERANCE_PCT` | `10` | performance.yml |
| `Z3_SYS_Z3_HEADER` | `/usr/include/z3.h` | proof-pyramid-gate.yml (PP-004) |

---

## 6. Artifact Registry

| Artifact Name | Produced By | Consumed By | Retention |
|---------------|-------------|-------------|-----------|
| `pp001-evidence` | PP-001 job | Evidence Bundle | 90 days |
| `pp002-evidence` | PP-002 job | Evidence Bundle | 90 days |
| `pp003-evidence` | PP-003 job | Evidence Bundle | 90 days |
| `pp004-evidence` | PP-004 job | Evidence Bundle | 90 days |
| `pp005-evidence` | PP-005 job | Evidence Bundle | 90 days |
| `pp006-evidence` | PP-006 job | Evidence Bundle | 90 days |
| `proof-pyramid-evidence-bundle` | Evidence Bundle job | evidence.json, audit | 90 days |
| `wire-completeness-audit-report` | Wire Audit job | Architecture review | 90 days |
| `trivy-sarif` | Security job | GitHub Security tab | 30 days |
| `sbom` | Container job | Compliance | 365 days |

---

## 7. New File Inventory

| File | Location in Repo | Purpose |
|------|-----------------|---------|
| `proof-pyramid-gate.yml` | `.github/workflows/` | 6-gate PP workflow (NEW) |
| `wire-completeness-audit.yml` | `.github/workflows/` | W1–W8 wire audit (NEW) |
| `docs-quality-fix.yml` | `.github/workflows/docs-quality.yml` | Replace failing docs gate (FIXED) |
| `ci_proof_pyramid_gate.py` | `scripts/` | Python evidence aggregator (NEW) |
| `Dockerfile.node0-omega` | `bizra-omega/` | Multi-stage builder/runtime/test image (NEW) |
| `performance_criterion_bench.rs` | `bizra-omega/bizra-proofspace/benches/` | Criterion benchmarks (NEW) |
