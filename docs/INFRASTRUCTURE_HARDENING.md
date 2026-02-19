# BIZRA Infrastructure Hardening Reference

**Version:** 1.0.0 | **Date:** 2026-02-19
**Standing on Giants:** OWASP (security baselines) · CIS Benchmarks (container hardening) · Kubernetes SIG-Security · Lamport (distributed reliability)

---

## Overview

This document covers all security hardening applied to BIZRA Node0 infrastructure across Docker, Kubernetes, monitoring, and CI/CD layers. Use it as a verification checklist during deployments and audits.

---

## 1. Docker Compose Hardening

### 1.1 Root Compose (`docker-compose.yml`)

All 6 services have enforced resource limits:

| Service | CPU Limit | Memory Limit | Stop Grace |
|---------|-----------|-------------|------------|
| python-api | 2.0 | 4G | 30s |
| rust-api | 2.0 | 2G | 15s |
| desktop-bridge | 1.0 | 1G | 15s |
| redis | 0.5 | 512M | 10s |
| prometheus | 1.0 | 2G | 15s |
| grafana | 1.0 | 1G | 10s |

**Redis authentication:** Required via `${REDIS_PASSWORD:?REDIS_PASSWORD must be set}` with `--requirepass`.

**Healthchecks:** All 6 services include healthchecks. Prometheus uses `wget --spider`, Grafana uses `wget --spider /api/health`.

**Verification:**
```bash
docker compose config --quiet  # Validates syntax
docker compose up -d && docker compose ps  # Confirm all healthy
```

### 1.2 MCP Compose (`deploy/mcp-compose.yaml`)

Shared anchor (`x-mcp-common`) enforces:

| Control | Value |
|---------|-------|
| `security_opt` | `no-new-privileges:true` |
| `cap_drop` | `ALL` |
| `cap_add` | `NET_BIND_SERVICE` |
| `stop_grace_period` | `30s` |
| `logging.max-size` | `50m` (5 files max) |
| `deploy.resources.limits.cpus` | `2.0` |
| `deploy.resources.limits.memory` | `2G` |

**Redis:** Image pinned by digest (`redis:7-alpine@sha256:b9f6...`), authenticated connections (`redis://:${REDIS_PASSWORD}@bizra-redis:6379`).

**Verification:**
```bash
docker compose -f deploy/mcp-compose.yaml config --quiet
# Confirm no-new-privileges:
docker inspect bizra-mcp-gateway --format '{{.HostConfig.SecurityOpt}}'
```

### 1.3 MCP Entrypoint (`deploy/mcp-entrypoint.sh`)

**Graceful shutdown pattern:**

```
SIGTERM received → trap fires → kill -TERM child PID → wait → exit 0
```

- Gateway uses background+wait pattern (not `exec`) so the shell trap fires
- Uvicorn configured with `--timeout-graceful-shutdown 25` to drain in-flight requests
- Other servers use `exec` (they handle SIGTERM natively)

**Verification:**
```bash
# In a running container:
docker exec bizra-mcp-gateway bash -c 'kill -TERM 1 && sleep 2 && echo "graceful"'
```

---

## 2. Kubernetes Hardening

### 2.1 Security Context (All Deployments)

Every pod spec enforces:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault
```

Every container enforces:

```yaml
securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop: [ALL]
```

### 2.2 Startup Probes

All production deployments now include startup probes to prevent liveness kill during slow initialization:

| Deployment | Port | Path | Failure Threshold | Max Startup Time |
|------------|------|------|-------------------|------------------|
| bizra-elite | 8000 | `/v1/health` | 12 | 70s |
| bizra-omega | 3001 | `/api/v1/health` | 12 | 65s |
| mcp-gateway | 8080 | `/health` | 12 | 70s |
| mcp-sovereign | 8081 | `/health` | 12 | 75s |
| mcp-ecosystem | 8082 | `/health` | 12 | 75s |
| mcp-peak | 8084 | `/health` | 18 | 110s |

Peak gets a higher threshold because it loads the GoT + Hypergraph engines.

### 2.3 PodDisruptionBudgets

Prevent simultaneous eviction during node drains:

| PDB | Target | minAvailable |
|-----|--------|-------------|
| `bizra-elite-pdb` | `app: bizra-elite` | 2 |
| `bizra-omega-pdb` | `app: bizra-omega` | 2 |
| `bizra-mcp-gateway-pdb` | `app: bizra-mcp, component: gateway` | 1 |
| `bizra-mcp-sovereign-pdb` | `app: bizra-mcp, component: sovereign` | 1 |
| `bizra-mcp-ecosystem-pdb` | `app: bizra-mcp, component: ecosystem` | 1 |

### 2.4 Horizontal Pod Autoscalers

| HPA | Min | Max | CPU Target | Memory Target |
|-----|-----|-----|-----------|--------------|
| `bizra-elite-hpa` | 3 | 10 | 70% | 80% |
| `bizra-omega-hpa` | 3 | 15 | 70% | 80% |
| `bizra-mcp-gateway-hpa` | 2 | 6 | 70% | 80% |

All HPAs use conservative scale-down (300s stabilization, 10%/min) and aggressive scale-up (0s stabilization, 100%/15s).

### 2.5 Network Policies

Defined in `deploy/k8s/base/networkpolicy.yaml`. Restricts:
- Inter-pod communication to labeled services only
- MCP ports (8080-8085) accessible from gateway and Prometheus
- Redis (6379) accessible only from MCP services

**Verification:**
```bash
kubectl get pdb -n bizra
kubectl get hpa -n bizra
kubectl get networkpolicy -n bizra
```

---

## 3. Monitoring Hardening

### 3.1 PromQL Guard Pattern

All division expressions use the safe pattern:

```promql
# Correct: divide by actual value, filter with "and"
(rate(errors[5m]) / rate(total[5m])) > 0.1 and rate(total[5m]) > 0

# Wrong: divides by boolean (0 or 1)
rate(errors[5m]) / (rate(total[5m]) > 0) > 0.1
```

This pattern is enforced in CI via `scripts/ci_promql_guard.py`.

### 3.2 Constitutional Compliance Recording

Two recording rules track pass/fail compliance as weighted gates:

```promql
# Produces: 0.0 (both fail), 0.4 (SNR pass), 0.6 (Ihsan pass), 1.0 (both pass)
(sovereign_snr_score >= 0.85) * 0.4 + (sovereign_ihsan_score >= 0.95) * 0.6
```

This boolean multiplication is intentional — it creates a discrete compliance indicator, not a continuous score.

---

## 4. CI/CD Governance

### 4.1 Quality Gate Bypass Audit

When `skip_quality_gate` is triggered via `workflow_dispatch`, the `quality-gate-bypass-audit` job records:

| Field | Source |
|-------|--------|
| Actor | `${{ github.actor }}` |
| Timestamp | UTC ISO-8601 |
| Ref | Branch/tag |
| SHA | Commit hash |
| Run ID | Workflow run link |

Output appears in GitHub Step Summary for permanent audit trail.

### 4.2 Staging Bypass Audit

Same pattern for `skip_staging` in the deploy workflow. Records who deployed directly to production and when.

### 4.3 Downstream Integration

Both bypass audit jobs are wired into dependency chains:
- `security-scan` depends on `quality-gates || quality-gate-bypass-audit`
- `build-*-image` depends on both
- `production-quality-gate` depends on `smoke-tests || staging-bypass-audit`

This ensures the pipeline continues even when gates are bypassed, while maintaining a full audit trail.

---

## 5. Python Configuration Hardening

### 5.1 Single Source of Truth

All constitutional thresholds are defined in `core/integration/constants.py`:

| Constant | Value | Usage |
|----------|-------|-------|
| `UNIFIED_IHSAN_THRESHOLD` | 0.95 | Production Ihsan floor |
| `UNIFIED_SNR_THRESHOLD` | 0.85 | Minimum SNR |
| `SNR_THRESHOLD_T1_HIGH` | 0.95 | Tier 1 SNR |
| `SNR_THRESHOLD_T0_ELITE` | 0.98 | Elite tier SNR |
| `STRICT_IHSAN_THRESHOLD` | 0.99 | Consensus/strict ops |
| `ADL_GINI_THRESHOLD` | 0.40 | Justice hard gate |
| `LMSTUDIO_HOST` | env: `LMSTUDIO_HOST` / `192.168.56.1` | LLM backend host |
| `LMSTUDIO_PORT` | env: `LMSTUDIO_PORT` / `1234` | LLM backend port |
| `LMSTUDIO_URL` | env: `LMSTUDIO_URL` / derived | Full LLM endpoint |

**No module may define its own threshold values.** All imports must come from `constants.py`.

### 5.2 Eliminated Patterns

| Anti-Pattern | Status |
|-------------|--------|
| `try: from constants import X; except: X = 0.85` | Eliminated (5 files) |
| Hardcoded `192.168.56.1` in module defaults | Eliminated (12 files) |
| `os.getenv("LMSTUDIO_HOST", "192.168.56.1")` | Replaced with `LMSTUDIO_HOST` import |

---

## Verification Checklist

```bash
# 1. Python — confirm no hardcoded thresholds remain
grep -r "= 0.85" core/ --include="*.py" | grep -v constants.py | grep -v __pycache__
grep -r "= 0.95" core/ --include="*.py" | grep -v constants.py | grep -v __pycache__
grep -r "192.168.56.1" core/ --include="*.py" | grep -v constants.py | grep -v __pycache__

# 2. Docker — validate compose files
docker compose config --quiet
docker compose -f deploy/mcp-compose.yaml config --quiet

# 3. Kubernetes — validate manifests
kubectl apply --dry-run=client -k deploy/k8s/base/

# 4. Tests — full regression
pytest tests/ -m "not requires_ollama and not requires_gpu and not slow" --timeout=60

# 5. PromQL — check for division-by-boolean
grep -rn '/ *(.*> 0)' deploy/monitoring/ --include="*.yaml"  # Should return 0 results
```
