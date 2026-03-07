# Module 10 — Infrastructure

> **Domain:** K8s, CI/CD, resilience gate, monitoring, containers, deployment
> **Source Specs:** ADR-012 (canary strategy), Phase 68-70 (CI/infra)
> **Key Paths:** `deploy/`, `.github/workflows/`, `scripts/`

## 10.1 Kubernetes Base Manifests

**Status:** [x] BUILT
**Path:** `deploy/k8s/base/kustomization.yaml`

Kustomize base with 3 images: elite, omega, mcp from ghcr.io.
Includes Deployment, Service, ConfigMap, ServiceAccount.

---

## 10.2 Kubernetes Overlays (Dev/Staging/Production)

**Status:** [x] BUILT
**Paths:**
- `deploy/k8s/overlays/dev/` — development overlay
- `deploy/k8s/overlays/staging/kustomization.yaml` — staging patches
- `deploy/k8s/overlays/production/kustomization.yaml` — production patches

Each overlay patches replicas, resource limits, and environment variables.

---

## 10.3 HPA (Horizontal Pod Autoscaler)

**Status:** [x] BUILT
**Path:** `deploy/k8s/base/hpa.yaml` + overlay patches

CPU/memory-based autoscaling for bizra-elite pods.

---

## 10.4 PDB (Pod Disruption Budget)

**Status:** [x] BUILT
**Path:** `deploy/k8s/base/` (pdb.yaml)

Ensures minimum availability during voluntary disruptions.

---

## 10.5 NetworkPolicy

**Status:** [x] BUILT
**Path:** `deploy/k8s/base/` (networkpolicy.yaml)

Namespace-level network segmentation. Restricts inter-pod communication
to declared dependencies only.

---

## 10.6 Argo Rollouts (Canary Strategy)

**Status:** [x] BUILT
**Path:** `deploy/argocd/rollouts.yaml`
**ADR:** `docs/adr/ADR-012-release-strategy-unification.md`

Canonical production deployment strategy. Progressive canary:
5% -> 20% -> 50% -> 80% -> 100% with 7-metric SLO analysis.

**Components in rollouts.yaml:**
- `Rollout` (bizra-elite-rollout) — canary spec with Traefik traffic routing
- `AnalysisTemplate` (bizra-slo-gate) — 7 metrics: error-rate, P95, P99, SNR, Ihsan, chaos, E2E
- `TraefikService` (bizra-elite-traffic) — weighted round-robin for traffic splitting
- `ConfigMap` (bizra-chaos-experiments) — pod-kill canary experiment
- `Service` (bizra-elite-stable + bizra-elite-canary) — traffic split targets
- `IngressRoute` — Traefik route via `Host(elite.bizra.node0)`
- `Middleware` (bizra-security-headers) — X-Frame-Options, X-Content-Type-Options

**K3d deployment:** Live on bizra-prod cluster, 2 pods on agent-0 + agent-1

---

## 10.7 ArgoCD Application

**Status:** [x] BUILT
**Path:** `deploy/argocd/application.yaml`

GitOps-driven deployment with auto-sync and self-heal for prod + staging.

---

## 10.8 CI Pipeline (GitHub Actions)

**Status:** [x] BUILT
**Path:** `.github/workflows/ci.yml`

9-stage pipeline, all GREEN:
1. Lint Python (ruff, black, isort, mypy)
2. Lint Rust (cargo fmt, clippy)
3. Cross-Language Sync (constants parity)
4. Schema Validation
5. Build Frontend
6. Test Rust (cargo test)
7. Test PyO3 (maturin build + smoke)
8. Test Python 3.11
9. Test Python 3.12

---

## 10.9 Resilience Gate Workflow

**Status:** [x] BUILT
**Path:** `.github/workflows/resilience-gate.yml`

5 jobs: frontend-vuln, backend-safety, e2e, evidence, verdict.
Post-merge quality gate for deployment authorization.

---

## 10.10 Deploy Workflow

**Status:** [x] BUILT
**Path:** `.github/workflows/deploy.yml`

Deployment automation triggered by CI success.

---

## 10.11 Dependency Lock Workflow

**Status:** [x] BUILT
**Path:** `.github/workflows/lock-deps.yml`

Automated dependency locking and governance.

---

## 10.12 Docker Images

**Status:** [x] BUILT
**Paths:**
- `deploy/Dockerfile.elite` — Python multi-stage (builder->runtime), non-root, healthcheck
- `deploy/Dockerfile.node0-genesis` — v6 constitutional pipeline server (260MB)
- `bizra-omega/Dockerfile` — Rust multi-stage, CPU/CUDA variants

---

## 10.13 Node0 Genesis Deployment

**Status:** [x] BUILT
**Paths:**
- `deploy/node0-genesis-compose.yaml` — Docker Compose with health checks
- `deploy/node0/bizra-node0-genesis.service` — systemd unit (port 7770)
- `deploy/node0/start-genesis-server.sh` — start script with 6 preflight checks
- `deploy/node0/node0-manifest.yaml` — full hardware spec, 7 services

---

## 10.14 Release Evidence Bundle

**Status:** [x] BUILT
**Path:** `scripts/release_evidence_bundle.py`

Persists: git info, coverage, security scan, container status, rollout verdict,
rollback reason into machine-readable release record.

---

## 10.15 Infrastructure Guardian

**Status:** [x] BUILT
**Paths:**
- `scripts/guardian/infra_guardian.py` — OODA loop, 9 probes, Ihsan scoring
- `scripts/guardian/bizra-guardian.service` — systemd unit (256M limit, 10% CPU)
- `scripts/guardian/install_guardian.sh` — syntax check, enable+start

**Probes:** docker_socket, container_health, memory, disk, ext4_errors,
critical_services, journal, banned_svcs, port_collisions, self_eval

---

## 10.16 Deployment Runbook

**Status:** [x] BUILT
**Path:** `deploy/resilience/RUNBOOK.md`

Pre-deploy, monitoring, rollback, post-mortem procedures.

---

## 10.17 Grafana Dashboards & Monitoring

**Status:** [x] BUILT
**Paths:**
- `deploy/monitoring/grafana-dashboard.json` (~2,000 LOC) — versioned dashboard
- `deploy/monitoring/prometheus-config.yaml` (~200 LOC) — ServiceMonitor, PodMonitor, recording rules
- `deploy/monitoring/alerting-rules.yaml` (~100 LOC) — SLO alerts (SNR, Ihsan, error, latency)
- `deploy/monitoring/mcp-alerting-rules.yaml` (~80 LOC) — MCP-specific alerts

Constitutional compliance metric:
`(sovereign_snr_score >= 0.85) * 0.4 + (sovereign_ihsan_score >= 0.95) * 0.6`

---

## 10.18 Chaos Mesh Operator

**Status:** [~] PARTIAL
**Path:** `deploy/argocd/rollouts.yaml` (ConfigMap with experiment definition)
**Gap:** CRDs not installed in cluster. Only ConfigMap exists with pod-kill
experiment YAML. Full Chaos Mesh operator deployment needed.

---

## 10.19 DAST (Dynamic Application Security Testing)

**Status:** [x] BUILT
**Path:** `.github/workflows/ci.yml` — `dast-zap` job

OWASP ZAP Baseline Scan runs against the built frontend dist served locally.
Triggers on every CI run after `build-frontend` completes.
Uses `zaproxy/action-baseline@v0.14.0` with `fail_action: warn`.
Results captured in CI evidence bundle.

---

## 10.20 Container Image Signing

**Status:** [ ] NOT BUILT
**Spec:** Required by security compliance and supply chain integrity
**Gap:** No cosign, notation, or Sigstore integration.

### Pseudocode
```
# In CI after docker build:
cosign sign --key env://COSIGN_PRIVATE_KEY ghcr.io/bizrainfo/bizra-elite:$TAG
cosign verify --key cosign.pub ghcr.io/bizrainfo/bizra-elite:$TAG

# In K8s admission:
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
spec:
  validations:
    - expression: "object.spec.containers.all(c, c.image.startsWith('ghcr.io/bizrainfo/'))"
```

---

## 10.21 Database Backup Automation

**Status:** [ ] NOT BUILT
**Spec:** Required for production data safety
**Gap:** No pg_dump cron, no Redis RDB backup, no Neo4j backup scripts.

### Pseudocode
```
# scripts/backup/db_backup.sh
pg_dump -h localhost -p 5433 -U bizra bizra_db | gzip > backups/pg_$(date +%Y%m%d).sql.gz
redis-cli -p 6379 BGSAVE && cp /var/lib/redis/dump.rdb backups/redis_$(date +%Y%m%d).rdb
neo4j-admin dump --to=backups/neo4j_$(date +%Y%m%d).dump

# Retention: 7 daily, 4 weekly, 12 monthly
find backups/ -name "pg_*.gz" -mtime +7 -delete
```

---

## Completion

| Feature | Status | Coverage |
|---------|--------|----------|
| 10.1 K8s Base | BUILT | Kustomize |
| 10.2 K8s Overlays | BUILT | 3 envs |
| 10.3 HPA | BUILT | CPU/mem |
| 10.4 PDB | BUILT | Min avail |
| 10.5 NetworkPolicy | BUILT | Namespace |
| 10.6 Argo Rollouts | BUILT | 7-metric |
| 10.7 ArgoCD App | BUILT | GitOps |
| 10.8 CI Pipeline | BUILT | 9/9 GREEN |
| 10.9 Resilience Gate | BUILT | 5 jobs |
| 10.10 Deploy Workflow | BUILT | Automation |
| 10.11 Dep Lock | BUILT | Governance |
| 10.12 Docker Images | BUILT | 3 images |
| 10.13 Node0 Genesis | BUILT | Compose+sys |
| 10.14 Evidence Bundle | BUILT | Machine-read |
| 10.15 Guardian | BUILT | 9 probes |
| 10.16 Runbook | BUILT | Full |
| 10.17 Monitoring | BUILT | Dashboard+alerts |
| 10.18 Chaos Mesh | PARTIAL | ConfigMap only |
| 10.19 DAST | BUILT | ZAP in CI |
| 10.20 Container Signing | NOT BUILT | Zero |
| 10.21 DB Backups | NOT BUILT | Zero |
| **TOTAL** | **18/21 + 1P + 2N** | **86%** |
