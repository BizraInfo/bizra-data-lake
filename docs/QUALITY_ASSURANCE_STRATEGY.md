# BIZRA Quality Assurance Strategy

Last updated: 2026-03-06
Status: enterprise QA baseline

## 1. Objective

Establish an executable quality model spanning unit, integration, contract, end-to-end, performance, security, compliance, and operational validation.

## 2. Testing Hierarchy

| Layer | Scope | Minimum standard |
|---|---|---|
| Unit | pure logic, validation, deterministic functions | >= 80% overall |
| Critical unit | auth, security, constitutional algorithms, proofs | >= 95% |
| Integration | module boundaries, persistence, cache, service interactions | all critical interfaces covered |
| Contract | API schemas, public/internal service contracts, Python/Rust boundaries | 100% of published interfaces |
| End-to-end | top user and operator workflows | top 10 journeys |
| Performance | latency, throughput, startup, memory | required for every release candidate |
| Security | code, dependencies, runtime, containers, IaC | required on every PR or release lane |
| Operational | backup/restore, failover, runbook execution | monthly rehearsal |

## 3. Quality Gates

### Pull Request Gate

- Ruff, Black, isort, MyPy ratchet
- unit tests
- targeted integration tests
- dependency audit
- secret scan
- API exposure gate
- dependency governance gate

### Release Candidate Gate

- full integration suite
- contract suite
- end-to-end suite
- performance benchmark suite
- security scans
- SBOM generation
- staging smoke tests

### Production Promotion Gate

- staging deployment green
- smoke tests green
- canary SLOs met
- rollback tested
- release evidence package generated

## 4. Automation Stack

| QA area | Tooling |
|---|---|
| Python tests | pytest, pytest-cov, pytest-asyncio, hypothesis |
| Rust tests | cargo test, clippy, rustfmt |
| API contract | Schemathesis or generated OpenAPI contract suite |
| UI/E2E | Playwright |
| Performance | k6 + existing perf scripts |
| Security | Trivy, pip-audit, cargo-audit, Bandit, Semgrep |
| Coverage | coverage.py, Codecov or internal artifact reporting |
| Docs | docs quality gate |

## 5. Coverage Ratchet

Current-state note: repository still carries a lower fail-under gate. The target enterprise ratchet is:

1. 38% -> 50%
2. 50% -> 60%
3. 60% -> 70%
4. 70% -> 80%
5. constitutional and security core remain >= 95%

## 6. Performance Benchmarks

| Domain | Target |
|---|---|
| API p95 read | < 300 ms |
| API p95 orchestration | < 800 ms |
| Startup | < 5 s |
| Throughput | >= 25 req/s baseline |
| Error rate | < 1% at expected load |
| Memory growth | no unbounded growth under 24h soak |

Protocols:

- baseline benchmark stored per mainline release
- regression threshold 10% max without approved exception
- canary rollback on latency or error SLO breach

## 7. Security Testing Protocol

- SAST on every PR
- dependency audit on every PR
- container scan on release build
- IaC scan before deploy
- authenticated DAST on staging before production promotion
- quarterly penetration test
- annual external review for SOC 2 scope

## 8. Compliance Verification

Evidence required:

- test reports
- scan reports
- release checklist
- approval records
- backup restore proof
- incident drill results
- accessibility audit

## 9. Acceptance Criteria Model

A feature is not complete until:

1. behavior is implemented
2. contracts are documented
3. regression tests exist
4. observability exists
5. rollback path exists
6. runbook impact is documented if operationally relevant
