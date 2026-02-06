# BIZRA Elite Canary Rollback Procedures

> **Standing on Giants:** Shannon (Signal Verification) | Lamport (Distributed Consistency) | Kubernetes

## Overview

This document describes rollback procedures for BIZRA Elite canary deployments. The canary architecture uses:

- **Stable Deployment**: `bizra-elite-stable` (90% traffic)
- **Canary Deployment**: `bizra-elite-canary` (10% traffic)
- **Traffic Splitting**: NGINX Ingress canary annotations

---

## Quick Reference

| Scenario | Command |
|----------|---------|
| Immediate Rollback | `kubectl delete deployment bizra-elite-canary -n bizra` |
| Scale Down Canary | `kubectl scale deployment bizra-elite-canary -n bizra --replicas=0` |
| Remove Traffic | `kubectl delete ingress bizra-elite-ingress-canary -n bizra` |
| Full Rollback | `./rollback.sh immediate` |
| Verify Status | `kubectl get pods -n bizra -l app=bizra-elite -o wide` |

---

## Rollback Scenarios

### Scenario 1: Immediate Rollback (Emergency)

Use when canary is causing critical issues (crashes, data corruption, security).

```bash
# 1. Remove canary from traffic immediately
kubectl delete ingress bizra-elite-ingress-canary -n bizra

# 2. Delete canary deployment
kubectl delete deployment bizra-elite-canary -n bizra

# 3. Verify only stable pods remain
kubectl get pods -n bizra -l app=bizra-elite

# 4. Check stable service is healthy
kubectl get endpoints bizra-elite-stable -n bizra
```

### Scenario 2: Gradual Rollback (Non-Critical Issues)

Use when canary has issues but not critical (elevated errors, latency).

```bash
# 1. Reduce canary traffic to 0%
kubectl annotate ingress bizra-elite-ingress-canary -n bizra \
  nginx.ingress.kubernetes.io/canary-weight="0" --overwrite

# 2. Monitor for 5 minutes
watch -n 5 "kubectl top pods -n bizra -l app=bizra-elite"

# 3. Scale down canary
kubectl scale deployment bizra-elite-canary -n bizra --replicas=0

# 4. Analyze canary logs
kubectl logs -n bizra -l track=canary --tail=1000 > canary-failure-logs.txt

# 5. Remove canary resources
kubectl delete deployment bizra-elite-canary -n bizra
kubectl delete ingress bizra-elite-ingress-canary -n bizra
```

### Scenario 3: Rollback to Previous Stable Version

Use when you need to rollback the stable deployment itself.

```bash
# 1. Check deployment history
kubectl rollout history deployment/bizra-elite-stable -n bizra

# 2. Rollback to previous revision
kubectl rollout undo deployment/bizra-elite-stable -n bizra

# 3. Or rollback to specific revision
kubectl rollout undo deployment/bizra-elite-stable -n bizra --to-revision=2

# 4. Verify rollback
kubectl rollout status deployment/bizra-elite-stable -n bizra
```

---

## Health Check Commands

### Pod Status
```bash
# All BIZRA Elite pods
kubectl get pods -n bizra -l app=bizra-elite -o wide

# Canary pods only
kubectl get pods -n bizra -l track=canary

# Stable pods only
kubectl get pods -n bizra -l track=stable
```

### Service Endpoints
```bash
# Verify endpoints are healthy
kubectl get endpoints -n bizra | grep bizra-elite

# Describe services
kubectl describe svc bizra-elite-stable -n bizra
kubectl describe svc bizra-elite-canary -n bizra
```

### Logs Analysis
```bash
# Canary logs (recent errors)
kubectl logs -n bizra -l track=canary --tail=100 | grep -i error

# Stable logs for comparison
kubectl logs -n bizra -l track=stable --tail=100 | grep -i error

# Stream canary logs
kubectl logs -n bizra -l track=canary -f
```

### Metrics Validation
```bash
# Port-forward to canary pod for direct metrics access
kubectl port-forward -n bizra svc/bizra-elite-canary 9090:9090

# In another terminal:
curl http://localhost:9090/metrics | grep -E "(snr|ihsan|error)"
```

---

## Automated Rollback Triggers

The following conditions should trigger automatic rollback:

| Metric | Threshold | Action |
|--------|-----------|--------|
| Error Rate | > 1% for 2 minutes | Immediate rollback |
| P95 Latency | > 500ms for 5 minutes | Gradual rollback |
| SNR Score | < 0.85 | Immediate rollback |
| Ihsan Score | < 0.95 | Immediate rollback |
| Pod Crashes | > 2 in 5 minutes | Immediate rollback |

### Prometheus Alerting Rules

Add these to your Prometheus configuration:

```yaml
groups:
- name: bizra-canary-alerts
  rules:
  - alert: CanaryHighErrorRate
    expr: |
      sum(rate(http_requests_total{namespace="bizra",track="canary",status=~"5.."}[2m]))
      /
      sum(rate(http_requests_total{namespace="bizra",track="canary"}[2m])) > 0.01
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "BIZRA Canary error rate > 1%"
      action: "Execute immediate rollback"

  - alert: CanarySNRBelowThreshold
    expr: bizra_snr_score{namespace="bizra",track="canary"} < 0.85
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "BIZRA Canary SNR below threshold"
      action: "Execute immediate rollback"

  - alert: CanaryIhsanViolation
    expr: bizra_ihsan_score{namespace="bizra",track="canary"} < 0.95
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "BIZRA Canary Ihsan constraint violated"
      action: "Execute immediate rollback - constitutional violation"
```

---

## Post-Rollback Checklist

After any rollback, complete these steps:

- [ ] Verify stable pods are healthy: `kubectl get pods -n bizra -l track=stable`
- [ ] Check endpoint health: `curl -s http://elite.bizra.local/v1/health`
- [ ] Verify traffic is flowing: Check ingress controller logs
- [ ] Review canary logs: Save to `canary-failure-$(date +%Y%m%d-%H%M%S).log`
- [ ] Document failure cause in incident report
- [ ] Create follow-up ticket for fix
- [ ] Update runbook if new failure mode discovered

---

## Canary Promotion (Success Path)

If canary is healthy and ready for full rollout:

```bash
# 1. Increase canary weight gradually
kubectl annotate ingress bizra-elite-ingress-canary -n bizra \
  nginx.ingress.kubernetes.io/canary-weight="25" --overwrite

# Wait 5 minutes, monitor metrics

kubectl annotate ingress bizra-elite-ingress-canary -n bizra \
  nginx.ingress.kubernetes.io/canary-weight="50" --overwrite

# Wait 5 minutes, monitor metrics

kubectl annotate ingress bizra-elite-ingress-canary -n bizra \
  nginx.ingress.kubernetes.io/canary-weight="90" --overwrite

# Wait 5 minutes, final verification

# 2. Update stable deployment with canary image
kubectl set image deployment/bizra-elite-stable -n bizra \
  bizra-elite=bizra-elite:v1.0.2

# 3. Wait for stable rollout
kubectl rollout status deployment/bizra-elite-stable -n bizra

# 4. Remove canary (now redundant)
kubectl delete deployment bizra-elite-canary -n bizra
kubectl delete ingress bizra-elite-ingress-canary -n bizra

# 5. Verify final state
kubectl get pods -n bizra -l app=bizra-elite
```

---

## Contact

- **On-Call**: Check PagerDuty rotation
- **Slack**: #bizra-ops
- **Escalation**: See incident response playbook

---

*Last Updated: 2026-02-03*
*Version: 1.0.1*
