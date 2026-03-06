# BIZRA Canary Deployment Runbook
> ADR-012 | Last updated: 2026-03-07

## Pre-Deployment Checklist

- [ ] All CI gates green (lint, test, security scan, resilience gate)
- [ ] Release evidence bundle generated and reviewed
- [ ] On-call engineer confirmed available
- [ ] Grafana dashboards open: SNR, Ihsan, Latency, Error Rate
- [ ] Prometheus alerts firing = 0 (no active incidents)
- [ ] Previous rollout fully promoted (no in-progress canary)

## Deployment Flow

```
 PR merged → CI builds images → Staging deploy → Smoke tests
       → Production quality gate → Argo Rollouts canary
       → SLO analysis (7 metrics) → Auto-promote or auto-rollback
       → Evidence bundle persisted
```

## Canary Stages (Argo Rollouts)

| Stage | Weight | Pause | Analysis |
|-------|--------|-------|----------|
| 1 | 5% | 2 min | — |
| 2 | 20% | 5 min | SLO + Chaos + E2E start |
| 3 | 50% | 5 min | Continuous SLO check |
| 4 | 80% | 2 min | Final promotion check |
| 5 | 100% | — | Promoted |

## SLO Thresholds (Auto-Rollback Triggers)

| Metric | Threshold | Failure Limit |
|--------|-----------|---------------|
| Error rate | < 1% | 3 consecutive |
| P95 latency | < 500ms | 3 consecutive |
| P99 latency | < 1000ms | 3 consecutive |
| SNR score | >= 0.85 | 1 (constitutional) |
| Ihsan score | >= 0.95 | 1 (constitutional) |
| Chaos resilience | pass | 1 |
| E2E smoke | pass | 1 |

## Manual Operations

### Monitor Rollout
```bash
kubectl argo rollouts get rollout bizra-elite-rollout -n bizra --watch
```

### Force Promote (skip remaining steps)
```bash
kubectl argo rollouts promote bizra-elite-rollout -n bizra
```

### Force Abort (immediate rollback)
```bash
kubectl argo rollouts abort bizra-elite-rollout -n bizra
```

### Retry Failed Rollout
```bash
kubectl argo rollouts retry rollout bizra-elite-rollout -n bizra
```

### Check Analysis Results
```bash
kubectl get analysisrun -n bizra -l rollouts-pod-template-hash \
  --sort-by=.metadata.creationTimestamp
```

## Emergency Rollback

If Argo Rollouts is unresponsive:

```bash
# 1. Scale down canary pods directly
kubectl scale deployment bizra-elite-canary -n bizra --replicas=0

# 2. Remove canary ingress
kubectl delete ingress bizra-elite-ingress-canary -n bizra

# 3. Verify stable is serving
kubectl get pods -n bizra -l app=bizra-elite
curl -f http://elite.bizra.node0/v1/health
```

## Post-Deployment

- [ ] Verify Grafana: no alert regressions for 15 minutes
- [ ] Generate evidence bundle: `python scripts/release_evidence_bundle.py --version <tag> --rollout-verdict promoted`
- [ ] Update release notes if tagged release
- [ ] Close deployment ticket

## Post-Rollback

- [ ] Save canary logs: `kubectl logs -n bizra -l track=canary --tail=5000 > rollback-<date>.log`
- [ ] Generate evidence bundle with `--rollout-verdict rolled-back`
- [ ] Open incident ticket with root cause analysis
- [ ] Update this runbook if new failure mode discovered

## Contacts

| Role | Channel |
|------|---------|
| On-Call SRE | PagerDuty rotation |
| Engineering | #bizra-ops |
| Escalation | Incident Commander |
