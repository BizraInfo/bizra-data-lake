# BIZRA DevOps Blueprint

Enterprise-grade CI/CD Pipeline Documentation

**Version:** 1.0.0
**Standing on Giants:** GitHub Actions, ArgoCD, Kubernetes, Docker BuildKit

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [GitHub Actions Workflows](#github-actions-workflows)
3. [Quality Gates](#quality-gates)
4. [Container Images](#container-images)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [GitOps with ArgoCD](#gitops-with-argocd)
7. [Monitoring and Alerting](#monitoring-and-alerting)
8. [Rollback Procedures](#rollback-procedures)
9. [Security Considerations](#security-considerations)

---

## Architecture Overview

```
+-------------------------------------------------------------------------+
|                        BIZRA CI/CD Pipeline                              |
+-------------------------------------------------------------------------+
|                                                                          |
|   [Push/PR] --> [Lint] --> [Test Matrix] --> [Quality Gates]            |
|                                    |                                     |
|                                    v                                     |
|   [Security Scan] --> [Build Images] --> [Integration Tests]            |
|                                    |                                     |
|                                    v                                     |
|   [Push to Registry] --> [Deploy Staging] --> [Smoke Tests]             |
|                                    |                                     |
|                                    v                                     |
|   [Production Gate] --> [Canary Deploy] --> [Full Rollout]              |
|                                    |                                     |
|                                    v                                     |
|                         [Verify & Monitor]                               |
+-------------------------------------------------------------------------+
```

### Pipeline Stages

| Stage | Purpose | Blocking |
|-------|---------|----------|
| Lint | Code formatting and style | Yes |
| Test | Unit and integration tests | Yes |
| Quality Gates | SNR/Ihsan validation | Yes |
| Security Scan | Vulnerability detection | Warning |
| Build | Docker image creation | Yes |
| Integration | End-to-end testing | Yes |
| Deploy Staging | Staging environment | Yes |
| Smoke Tests | Basic functionality | Yes |
| Production Gate | Constitutional compliance | Yes |
| Deploy Production | Production environment | Yes |

---

## GitHub Actions Workflows

### CI Workflow (`.github/workflows/ci.yml`)

Runs on every push and pull request. All jobs have `timeout-minutes` set.

```yaml
Triggers:
  - push to: main, master, develop
  - pull_request to: main, master

Concurrency: cancel-in-progress per workflow+ref
Runner OS: ubuntu-24.04 (pinned, no 'latest')
Actions: SHA-pinned to immutable commit hashes

Jobs:
  1. lint-python           # ruff, Black, isort, mypy (15m)
  2. lint-rust             # cargo fmt, clippy (15m)
  3. validate-schemas      # JSON schema validation (10m)
  4. test-python           # pytest matrix (3.11, 3.12) (30m)
  5. test-rust             # cargo test (30m)
  6. test-pyo3-bindings    # PyO3 binding tests (20m)
  7. quality-gates         # SNR/Ihsan validation (30m)
  8. security-scan         # Bandit, pip-audit, Trivy (20m)
  9. build-python-image    # Docker multi-stage build (20m)
  10. build-rust-image     # Rust Docker build (30m)
  11. integration-tests    # Full stack tests with Redis service (20m)
  12. ci-summary           # Final status report (5m)
```

### Deploy Workflow (`.github/workflows/deploy.yml`)

Runs on release or manual dispatch. Has deployment-level concurrency guard.

```yaml
Triggers:
  - release: published
  - workflow_dispatch: staging/production

Concurrency: deploy-{environment}, cancel-in-progress: false
Runner OS: ubuntu-24.04 (pinned)

Jobs:
  1. build-push           # Build and push to GHCR (30m)
  2. deploy-staging       # Deploy to staging cluster (15m)
  3. smoke-tests          # Verify staging health (10m)
  4. production-quality-gate  # Constitutional validation (20m)
  5. deploy-production    # Canary then full rollout (20m)
  6. verify-production    # Final health check (5m)
  7. rollback             # Emergency rollback (manual, 10m)
```

### Release Workflow (`.github/workflows/release.yml`)

Runs on version tags (v*.*.*)

```yaml
Triggers:
  - tag: v[0-9]+.[0-9]+.[0-9]+

Concurrency: release-{ref}, cancel-in-progress: false
Runner OS: ubuntu-24.04 (pinned)
Permissions: contents: read (least privilege)

Jobs:
  1. validate         # Version consistency (5m)
  2. build-artifacts  # Multi-platform binaries (30m)
  3. build-wheel      # Python wheel (15m)
  4. build-pyo3       # PyO3 manylinux wheels (30m)
  5. generate-sbom    # Software Bill of Materials (15m)
  6. create-release   # GitHub release (10m)
  7. publish-pypi     # PyPI publication (10m)
```

---

## CI Hardening Standards

All 7 workflows enforce the following security and reliability standards:

| Standard | Enforcement | Details |
|----------|-------------|---------|
| SHA-pinned Actions | All `uses:` entries | 40-char hex commit hash, version in comment |
| Pinned runner OS | `ubuntu-24.04` | No `ubuntu-latest` anywhere |
| Job timeouts | `timeout-minutes` on every job | Prevents stuck runners |
| Concurrency groups | Per workflow + ref | Prevents parallel builds on same branch |
| Docker image pins | SHA256 digest | Redis, Prometheus, Grafana in CI services |
| Least privilege | `permissions: contents: read` | Added to release workflow |

### Additional Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `tests.yml` | Push/PR | Extended test matrix (unit, integration, token, spearpoint, slow) |
| `performance.yml` | Push to main | Inference latency, throughput, memory benchmarks |
| `native-ci.yml` | Push to `native/` | Rust native crates (fate-binding, iceoryx-bridge) |
| `docs-quality.yml` | Push with doc changes | Markdown lint, link check, policy validation |

---

## Quality Gates

### Constitutional Thresholds

| Environment | Ihsan Threshold | SNR Threshold |
|-------------|-----------------|---------------|
| Production  | 0.95            | 0.95 (T1)     |
| Staging     | 0.95            | 0.90 (T2)     |
| CI          | 0.90            | 0.90 (T2)     |
| Development | 0.80            | 0.80          |

### SNR Tiers

| Tier | Threshold | Use Case |
|------|-----------|----------|
| T0 Elite | >= 0.98 | Production-critical |
| T1 High | >= 0.95 | Production-ready |
| T2 Standard | >= 0.90 | General use |
| T3 Acceptable | >= 0.85 | Limited use |
| Below | < 0.85 | Requires improvement |

### Quality Gate Script

```bash
# Run quality gate locally
python scripts/ci_quality_gate.py --environment ci

# Run with strict (production) thresholds
python scripts/ci_quality_gate.py --strict

# Output JSON for CI integration
python scripts/ci_quality_gate.py --json --output quality_report.json
```

---

## Container Images

### Image Registry

Images are published to GitHub Container Registry (GHCR):

```
ghcr.io/bizra/bizra-data-lake/elite:TAG      # Python Elite
ghcr.io/bizra/bizra-data-lake/omega:TAG      # Rust Omega (CPU)
ghcr.io/bizra/bizra-data-lake/omega:TAG-cuda # Rust Omega (CUDA)
```

### Build Commands

```bash
# Build Python Elite image
docker build -t bizra-elite:local -f deploy/Dockerfile.elite .

# Build Rust Omega image
docker build -t bizra-omega:local -f bizra-omega/Dockerfile bizra-omega/

# Build Unified image (Python + Rust + PyO3)
docker build -t bizra:unified -f deploy/Dockerfile.unified .

# Build with CUDA support
docker build -t bizra:unified-cuda --build-arg CUDA=1 -f deploy/Dockerfile.unified .
```

### Image Security

- Non-root user execution
- Read-only root filesystem
- Minimal base images (slim/distroless)
- No secrets in image layers
- Security scanning with Trivy

---

## Kubernetes Deployment

### Directory Structure

```
deploy/k8s/
  base/                    # Common resources
    namespace.yaml
    configmap.yaml
    secrets.yaml
    rbac.yaml
    storage.yaml
    deployment-elite.yaml
    deployment-omega.yaml
    services.yaml
    ingress.yaml
    hpa.yaml
    kustomization.yaml
  overlays/
    staging/               # Staging-specific patches
      kustomization.yaml
    production/            # Production-specific patches
      kustomization.yaml
      monitoring.yaml
```

### Deploy Commands

```bash
# Deploy to staging
kubectl apply -k deploy/k8s/overlays/staging

# Deploy to production
kubectl apply -k deploy/k8s/overlays/production

# Check deployment status
kubectl rollout status deployment/bizra-elite -n bizra
kubectl rollout status deployment/bizra-omega -n bizra
```

### Resource Requirements

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|-----------|----------------|--------------|
| Elite | 250m | 1000m | 512Mi | 2Gi |
| Omega | 250m | 1000m | 256Mi | 1Gi |
| Omega (GPU) | 500m | 2000m | 1Gi | 4Gi |

---

## GitOps with ArgoCD

### Application Setup

```bash
# Install ArgoCD Application
kubectl apply -f deploy/argocd/application.yaml

# Check sync status
argocd app get bizra-production
argocd app get bizra-staging

# Sync manually (if needed)
argocd app sync bizra-production
```

### Progressive Delivery

Using Argo Rollouts for canary deployments:

1. 5% traffic for 2 minutes
2. 20% traffic for 5 minutes
3. 50% traffic for 5 minutes
4. 80% traffic for 2 minutes
5. 100% traffic (full rollout)

Automatic rollback if:
- Error rate > 1%
- P95 latency > 500ms
- SNR score < 0.85
- Ihsan score < 0.95

---

## Monitoring and Alerting

### Metrics Endpoints

| Service | Port | Path |
|---------|------|------|
| Elite | 9090 | /metrics |
| Omega | 3001 | /metrics |

### Key Metrics

```promql
# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m]))
/ sum(rate(http_requests_total[5m]))

# Latency (P95)
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))

# SNR Score
bizra_snr_score

# Ihsan Score
bizra_ihsan_score
```

### Alerts

| Alert | Severity | Condition |
|-------|----------|-----------|
| HighErrorRate | Critical | > 1% errors for 5m |
| PodRestartLoop | Warning | > 3 restarts/hour |
| QualityGateFailure | Critical | Gate failed for 5m |
| SNRBelowThreshold | Warning | SNR < 0.85 for 10m |
| IhsanBelowThreshold | Critical | Ihsan < 0.95 for 5m |

---

## Rollback Procedures

### Automatic Rollback

Argo Rollouts automatically rolls back when:
- Analysis template fails
- Pod health checks fail
- Custom metrics exceed thresholds

### Manual Rollback

#### Kubernetes Rollback

```bash
# Rollback to previous revision
kubectl rollout undo deployment/bizra-elite -n bizra
kubectl rollout undo deployment/bizra-omega -n bizra

# Rollback to specific revision
kubectl rollout undo deployment/bizra-elite -n bizra --to-revision=2

# Check rollout history
kubectl rollout history deployment/bizra-elite -n bizra
```

#### ArgoCD Rollback

```bash
# List revision history
argocd app history bizra-production

# Rollback to specific revision
argocd app rollback bizra-production REVISION_NUMBER

# Sync to specific commit
argocd app sync bizra-production --revision COMMIT_SHA
```

#### Emergency Rollback Script

```bash
#!/bin/bash
# Emergency rollback procedure

NAMESPACE=${1:-bizra}
DEPLOYMENT=${2:-bizra-elite}

echo "Starting emergency rollback for $DEPLOYMENT in $NAMESPACE"

# 1. Scale down to prevent new traffic
kubectl scale deployment/$DEPLOYMENT -n $NAMESPACE --replicas=0

# 2. Rollback to previous version
kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE

# 3. Wait for rollback
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=300s

# 4. Scale back up
kubectl scale deployment/$DEPLOYMENT -n $NAMESPACE --replicas=3

# 5. Verify health
kubectl wait --for=condition=ready pod -l app=$DEPLOYMENT -n $NAMESPACE --timeout=120s

echo "Rollback complete"
```

### Rollback Checklist

1. **Identify the issue**
   - Check error logs: `kubectl logs -l app=bizra-elite -n bizra --tail=100`
   - Check metrics in Grafana
   - Review recent changes

2. **Initiate rollback**
   - Use kubectl rollback or ArgoCD
   - Monitor rollout progress

3. **Verify recovery**
   - Check pod health: `kubectl get pods -n bizra`
   - Check service health: `curl https://api.bizra.node0/api/v1/health`
   - Verify quality metrics

4. **Post-mortem**
   - Document the incident
   - Identify root cause
   - Update tests/quality gates

---

## Security Considerations

### Secrets Management

- Never commit secrets to Git
- Use External Secrets Operator or similar
- Rotate credentials regularly
- Use minimal permission tokens

### Network Security

- Network policies enforced
- TLS everywhere
- Rate limiting on ingress
- Internal services not exposed

### Container Security

- Non-root execution
- Read-only filesystem
- Dropped capabilities
- Security scanning in CI

### Access Control

- RBAC enforced
- Service accounts with minimal permissions
- Audit logging enabled
- Multi-factor authentication for clusters

---

## Quick Reference

### Environment Variables

```bash
# Constitutional thresholds
IHSAN_THRESHOLD=0.95
SNR_THRESHOLD_T0_ELITE=0.98
SNR_THRESHOLD_T1_HIGH=0.95

# Runtime
BIZRA_ENV=production
RUST_LOG=bizra_api=info
FAIL_CLOSED=true

# Inference
OLLAMA_HOST=http://ollama:11434
DEFAULT_MODEL=qwen2.5:7b
```

### Useful Commands

```bash
# Check CI status
gh run list --workflow=ci.yml

# Trigger deployment
gh workflow run deploy.yml -f environment=staging

# View logs
kubectl logs -f -l app=bizra-elite -n bizra

# Port forward for debugging
kubectl port-forward svc/bizra-elite 8000:80 -n bizra

# Run quality gate
python scripts/ci_quality_gate.py --environment production --strict
```

---

**Constitutional Compliance:** This pipeline enforces BIZRA constitutional thresholds at every stage. No deployment proceeds without SNR >= 0.85 and Ihsan >= 0.95 (production).
