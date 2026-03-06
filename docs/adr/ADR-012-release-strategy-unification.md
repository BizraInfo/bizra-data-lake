# ADR-012: Release Strategy Unification

## Status
**ACCEPTED** - 2026-03-07

## Context

The BIZRA codebase contains conflicting deployment strategy references:

| Source | Strategy | Evidence |
|--------|----------|----------|
| `deploy/argocd/rollouts.yaml` | **Canary** via Argo Rollouts | 5/20/50/80% weight steps |
| `deploy/k8s/canary/` | **Canary** via NGINX ingress annotations | Manual 10/90 split |
| `deploy/.github/workflows/deploy.yml` | **Canary** then scale-swap | Manual pod health check |
| `docs/ENTERPRISE_IMPLEMENTATION_BLUEPRINT.md` | **Blue-green** | Legacy reference |
| `tools/engines/omega_blueprint.py` | **Blue-green** | Legacy reference |

This drift creates ambiguity: operators do not know which strategy is canonical.
The SAT/PAT DevOps spec requires "progressive, SLO-gated, chaos-tested, self-healing delivery."

### Standing on Giants

- **Lamport (1982)**: Distributed consensus under partial failure
- **Netflix / Kayenta (2018)**: Automated canary analysis at scale
- **Argo Project (2020)**: Progressive delivery as Kubernetes-native CRDs

## Decision

### Production Standard: Argo Rollouts Canary

All production releases use **Argo Rollouts `canary` strategy** with automated analysis.

**Rationale:** Canary _proves_ the new release under partial real traffic with measurable
blast-radius control. Blue-green merely _switches_ traffic, providing no gradual
evidence gathering.

### Fallback: Blue-Green Emergency Recovery

Blue-green remains available as a **disaster-recovery posture only** (full-cluster failover
to a pre-provisioned standby). It is NOT the default production deploy path.

### Canonical Rollout Stages

```
STAGE 1:  5% traffic  → 2 min pause  → analysis starts
STAGE 2: 20% traffic  → 5 min pause  → SLO + chaos validation
STAGE 3: 50% traffic  → 5 min pause  → E2E functional gate
STAGE 4: 80% traffic  → 2 min pause  → final promotion check
STAGE 5: 100%         → promoted      → evidence bundle persisted
```

### Analysis Gates (AnalysisTemplate)

| Metric | SLO | Failure Limit | Source |
|--------|-----|---------------|--------|
| Error rate (5xx/total) | < 1% | 3 | Prometheus |
| P99 latency | < 1000ms | 3 | Prometheus |
| P95 latency | < 500ms | 3 | Prometheus |
| SNR score | >= 0.85 | 1 | Prometheus (constitutional) |
| Ihsan score | >= 0.95 | 1 | Prometheus (constitutional) |
| Chaos resilience | pass | 1 | Chaos Mesh Job |
| E2E smoke | pass | 1 | Playwright Job |

### RACI

| Activity | DevOps/SRE | QA | Security | Engineering |
|----------|------------|-----|----------|-------------|
| Rollout manifest | **R/A** | C | C | I |
| SLO thresholds | R | **A** | C | I |
| Chaos scenarios | R | R | **A** | I |
| Evidence bundle | **R** | I | A | I |
| Promotion/rollback | **R/A** | C | I | I |
| Runbook maintenance | **R/A** | C | C | I |

## Consequences

### Benefits

1. **Single source of truth**: One strategy, one set of manifests
2. **Evidence-driven promotion**: SLO violations auto-rollback; no human judgment needed
3. **Constitutional enforcement**: Ihsan/SNR gates in the rollout path, not just alerting
4. **Blast-radius control**: 5% initial exposure limits damage from bad releases
5. **Audit trail**: Every promotion/rollback recorded with metrics snapshot

### Tradeoffs

1. **Argo Rollouts dependency**: Requires CRD installation on cluster
2. **Traefik ingress**: Traffic splitting uses Traefik (K3s default), not NGINX
3. **Prometheus required**: All SLO checks query live metrics

### Migration

1. Remove blue-green references from `docs/ENTERPRISE_IMPLEMENTATION_BLUEPRINT.md`
2. Remove blue-green references from `tools/engines/omega_blueprint.py`
3. Upgrade `deploy/argocd/rollouts.yaml` with P99 + chaos + E2E analysis
4. Deploy workflow (`deploy.yml`) delegates to Argo Rollouts instead of manual scaling

## Files

- `deploy/argocd/rollouts.yaml` — Canonical Rollout + AnalysisTemplate
- `deploy/argocd/application.yaml` — ArgoCD GitOps application
- `deploy/k8s/canary/` — Legacy manual canary (deprecated by this ADR)
- `deploy/resilience/chaos-canary-job.yaml` — Chaos Mesh experiment
- `deploy/resilience/playwright-canary-job.yaml` — E2E functional gate
- `scripts/release_evidence_bundle.py` — Release evidence generator
- `.github/workflows/resilience-gate.yml` — Pre-merge resilience CI

## References

1. Argo Rollouts Documentation: Progressive Delivery
2. Netflix Kayenta: Automated Canary Analysis
3. Chaos Mesh: Chaos Engineering for Kubernetes
4. ADR-001: Unified Constitutional Engine (Ihsan/SNR thresholds)
5. ADR-011: Gate Ordering and Trust Encoding

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-03-07 | System Integrator | Initial version — resolves blue-green/canary conflict |
