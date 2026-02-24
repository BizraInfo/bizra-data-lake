# CI/CD Pipeline

> 7 GitHub Actions workflows with constitutional quality gates.

## Workflow Overview

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | push/PR to main | Lint, test, quality gates, security, Docker build |
| `tests.yml` | push/PR to main | Parallel test suites (unit/integration/slow) |
| `release.yml` | tag `v*` | PyPI publish, Docker push, SBOM generation |
| `deploy.yml` | workflow_dispatch | K8s staging/production with canary rollout |
| `performance.yml` | push to main | Latency, throughput, memory, startup benchmarks |
| `alpha100-release-binaries.yml` | tag | Cross-platform binary builds |
| `docs-quality.yml` | push/PR | Documentation linting and link checking |

## Main Pipeline (ci.yml)

### Stage Dependency Graph

```
Lint Python ──┐
Lint Rust  ───┼──→ Test Python (3.11 + 3.12) ──┐
Cross-Lang ───┘    Test Rust                     ├──→ Quality Gates ──→ Security Scan ──→ Docker Build
                   PyO3 Bindings ────────────────┘                                        ↓
                                                                                    Integration Tests
```

### Stage 1: Lint

| Check | Tool | Gate |
|-------|------|------|
| Python format | `black --check` | Hard |
| Python imports | `isort --check-only` | Hard |
| Python lint | `ruff check core/` | Hard |
| Python types | `mypy core/` | Soft (1,555 pre-existing errors) |
| Python security | `bandit -r core/ -ll --confidence-level high` | Hard |
| Rust format | `cargo fmt --all -- --check` | Hard |
| Rust lint | `cargo clippy --workspace -- -D warnings` | Hard (zero warnings) |

### Stage 2: Test

| Suite | Command | Coverage |
|-------|---------|----------|
| Python 3.11 | `pytest tests/ --cov=core` | 65% floor |
| Python 3.12 | `pytest tests/ --cov=core` | 65% floor |
| Rust | `cargo test --workspace --release` | Tracked |
| PyO3 | `maturin develop && pytest tests/e2e_http/test_pyo3_bridge.py` | Tracked |

Coverage ratchet plan: 30% → 55% → 60% → 65% (current) → 95% (Ihsan target).

### Stage 3: Quality Gates

All hard-gated:

- SNR + Ihsan threshold enforcement
- Bridge smoke tests (TCP listener + ping)
- SAPE v1 gap closure validation (8 modules)
- Token ledger integrity check
- Desktop bridge latency SLO (P95 < 200ms)

### Stage 4: Security Scan

| Tool | Scope | Gate |
|------|-------|------|
| `pip-audit --strict` | Python dependencies | Hard |
| `bandit` | Python code | Hard |
| `cargo audit` | Rust dependencies | Hard |
| `trivy` (CRITICAL,HIGH) | Filesystem scan | Hard (exit-code 1) |

**Ignored CVEs**:
- `PYSEC-2024-48`: urllib3 DoS via header regex. Mitigated by httpx usage, no direct urllib3 calls.

### Stage 5: Docker Build

| Image | Dockerfile | Registry |
|-------|-----------|----------|
| `bizra-elite` (Python) | `deploy/Dockerfile.elite` | ghcr.io |
| `bizra-omega` (Rust/CPU) | `bizra-omega/Dockerfile` | ghcr.io |

Built only after security scan passes.

## Test Matrix

### Markers

| Marker | CI Behavior | Use Case |
|--------|-------------|----------|
| `@pytest.mark.slow` | Excluded by default | Weekly schedule |
| `@pytest.mark.integration` | Separate job | External service tests |
| `@pytest.mark.requires_ollama` | Skipped | Needs Ollama at localhost:11434 |
| `@pytest.mark.requires_gpu` | Skipped | Needs CUDA GPU |
| `@pytest.mark.requires_network` | Skipped | Needs internet |

### Collection Guards

- `tests/integration/conftest.py`: Skips `test_live_pipeline.py` and `test_one_human.py` unless `BIZRA_COLLECT_HEAVY=1` (torch/pandas `__spec__` collision)
- `tests/root_legacy/`: Excluded via `pyproject.toml` `addopts`

### Test Suite Size

8,103 tests across 40 modules. Largest: sovereign (3,259), proof_engine (563), integration (413), spearpoint (391).

## Deployment Pipeline (deploy.yml)

1. Build and push images to ghcr.io
2. Deploy to staging namespace
3. Run smoke tests on staging
4. Production quality gate (SNR/Ihsan validation)
5. Canary rollout (10% → 100%)
6. Manual rollback capability

## Performance Pipeline (performance.yml)

4 benchmark jobs with 10% regression tolerance:

| Benchmark | Threshold |
|-----------|-----------|
| Inference latency | P95 < 500ms |
| Throughput | >= 10 req/s |
| Memory peak | < 4 GB |
| Cold start | < 5s |

Baselines auto-update on `main` when gate passes.

## Release Pipeline (release.yml)

Triggered by `v*` tags:

1. Rust binary builds (linux-x86_64, linux-musl)
2. Python wheel + PyO3 wheels (manylinux)
3. SBOM generation (CycloneDX JSON)
4. SHA256 checksums
5. `pip-audit --strict` (hard-gated)
6. GitHub Release creation
7. PyPI publish (non-prerelease only)

## Constitutional Thresholds

All gates import from `core/integration/constants.py`:

| Constant | Value | Gate Context |
|----------|-------|-------------|
| `UNIFIED_IHSAN_THRESHOLD` | 0.95 | Production quality gate |
| `STRICT_IHSAN_THRESHOLD` | 0.99 | Consensus operations |
| `UNIFIED_SNR_THRESHOLD` | 0.85 | Minimum signal quality |
| `SNR_THRESHOLD_T1_HIGH` | 0.95 | High-quality tier |
| `SNR_THRESHOLD_T0_ELITE` | 0.98 | Elite operations |
| `ADL_GINI_THRESHOLD` | 0.35 | Justice/inequality enforcement |

## Local CI Simulation

```bash
# Lint
ruff check core/ && black --check core/ && isort --check-only core/

# Test (fast)
pytest tests/ -m "not slow and not requires_ollama and not requires_gpu" --timeout=60

# Security
bandit -r core/ -ll --confidence-level high
pip-audit --strict --ignore-vuln PYSEC-2024-48

# Rust
cd bizra-omega && cargo fmt --all -- --check && cargo clippy --workspace -- -D warnings && cargo test --workspace
```
