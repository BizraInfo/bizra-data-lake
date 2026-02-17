# Phase 36: Production Fortress — Autoscaling + Network Policy + Observability

> Hardens the k8s deployment with vertical/GPU-aware autoscaling, east-west network segmentation, complete observability pipeline, and runtime degradation surfacing.

Standing on Giants: Deming (1950, measure everything, improve continuously) + Lamport (1978, distributed system must report its own health) + Burns et al. (2016, Kubernetes design patterns) + Shannon (1948, channel capacity — know your system limits)

## Context

The k8s deployment runs `bizra-elite:v1.2.0` on a 3-node k3d cluster. HPA and Ingress manifests exist but are not deployed. Missing: VPA (vertical pod autoscaler), KEDA (GPU queue-depth scaling), NetworkPolicy (east-west segmentation), ServiceMonitor/PrometheusRule (in base manifests). The Sovereign runtime has 4 subsystems that silently fall back to stubs when imports fail — this degradation is invisible to operators.

## Gaps Addressed

| Gap | Current State | Target State |
|-----|--------------|--------------|
| VPA | Absent | VerticalPodAutoscaler for memory right-sizing |
| KEDA GPU scaling | Absent | ScaledObject for RTX 4090 queue depth |
| NetworkPolicy | Absent | East-west segmentation between pods |
| ServiceMonitor | Only in production overlay | In base manifests |
| PrometheusRule/Alerts | Only in production overlay | Critical alerts in base |
| Runtime degradation | Silent stub fallback | Health endpoint reports degraded subsystems |
| Ingress deployment | Manifest exists, not applied | Applied with cert-manager |

## 1. VerticalPodAutoscaler

```yaml
# deploy/k8s/base/vpa.yaml

PSEUDOCODE:
  VPA for bizra-elite:
    target: Deployment/bizra-elite
    updateMode: Auto                    # Restarts pods with right-sized requests
    resourcePolicy:
      containerPolicies:
        - containerName: bizra-elite
          minAllowed:
            cpu: 100m
            memory: 256Mi
          maxAllowed:
            cpu: 4000m
            memory: 8Gi               # torch can consume 2-4GB
          controlledResources: [cpu, memory]

  VPA for bizra-omega:
    target: Deployment/bizra-omega
    updateMode: Auto
    resourcePolicy:
      containerPolicies:
        - containerName: omega-runtime
          minAllowed:
            cpu: 50m
            memory: 64Mi
          maxAllowed:
            cpu: 2000m
            memory: 2Gi
          controlledResources: [cpu, memory]

  RATIONALE:
    Deming: "You can't improve what you don't measure."
    VPA continuously measures actual resource usage and adjusts
    requests/limits to eliminate waste and prevent OOM kills.
```

---

## 2. KEDA GPU-Aware Scaling

```yaml
# deploy/k8s/base/keda-scaledobject.yaml

PSEUDOCODE:
  ScaledObject for bizra-elite:
    scaleTargetRef: Deployment/bizra-elite
    minReplicaCount: 1
    maxReplicaCount: 5
    triggers:
      - type: prometheus
        metadata:
          serverAddress: http://prometheus:9090
          metricName: bizra_inference_queue_depth
          query: >
            sum(bizra_inference_queue_depth{namespace="bizra"})
          threshold: "10"             # Scale up when >10 requests queued
          activationThreshold: "3"    # Activate scaling at >3

      - type: prometheus
        metadata:
          metricName: bizra_query_latency_p95
          query: >
            histogram_quantile(0.95,
              rate(bizra_query_duration_seconds_bucket{namespace="bizra"}[5m]))
          threshold: "5"              # Scale up when p95 > 5 seconds

  RATIONALE:
    Shannon: Channel capacity determines throughput.
    When inference queue depth exceeds capacity, KEDA adds replicas.
    When p95 latency breaches SLO, KEDA adds replicas.
    GPU workloads (embedding, LLM inference) are the bottleneck.
```

---

## 3. NetworkPolicy

```yaml
# deploy/k8s/base/network-policy.yaml

PSEUDOCODE:
  NetworkPolicy: bizra-elite-policy
    podSelector: app=bizra-elite
    policyTypes: [Ingress, Egress]

    ingress:
      - from:
          - podSelector: {app: bizra-omega}     # Omega can call Elite
          - podSelector: {app: prometheus}       # Prometheus can scrape
          - namespaceSelector: {name: ingress}   # Ingress controller
        ports: [8000]                             # Only HTTP port

    egress:
      - to:
          - podSelector: {app: bizra-omega}     # Elite can call Omega
          - ipBlock: {cidr: 192.168.56.0/24}    # LM Studio network
        ports: [8000, 1234]
      - to:
          - ipBlock: {cidr: 0.0.0.0/0}         # DNS
        ports: [{port: 53, protocol: UDP}]

  NetworkPolicy: bizra-omega-policy
    podSelector: app=bizra-omega
    policyTypes: [Ingress, Egress]

    ingress:
      - from:
          - podSelector: {app: bizra-elite}
          - podSelector: {app: prometheus}
        ports: [8000]

    egress:
      - to:
          - podSelector: {app: bizra-elite}
        ports: [8000]
      - to:
          - ipBlock: {cidr: 0.0.0.0/0}
        ports: [{port: 53, protocol: UDP}]

  RATIONALE:
    Lamport: Distributed systems must restrict communication channels.
    Default-deny with explicit allow lists prevents lateral movement
    if any pod is compromised.
```

---

## 4. Runtime Degradation Reporting

```
# MODIFY: core/sovereign/api.py — enhance /v1/health

FUNCTION health_endpoint() -> HealthResponse:
  """
  Enhanced health check that reports subsystem degradation.

  Instead of binary healthy/unhealthy, reports per-subsystem status
  so operators and k8s probes can make informed decisions.

  Standing on Giants: Lamport (distributed health must be observable)
  Artifact: core/sovereign/api.py
  """

  subsystems = {}
  runtime = get_runtime()

  # Check each potentially-stubbed subsystem
  FOR name, attr IN [
    ("graph_of_thoughts", "_got_reasoner"),
    ("snr_maximizer", "_snr"),
    ("guardian_council", "_guardian"),
    ("autonomous_loop", "_auto_loop"),
    ("cognitive_fusion", "_cognitive_fusion"),
    ("embedding_service", "_embedding_service"),
  ]:
    instance = getattr(runtime, attr, None)
    IF instance IS None:
      subsystems[name] = "unavailable"
    ELIF hasattr(instance, "__class__") AND "Stub" IN instance.__class__.__name__:
      subsystems[name] = "stub"
    ELSE:
      subsystems[name] = "active"

  # Determine overall status
  stub_count = sum(1 for s in subsystems.values() if s == "stub")
  unavailable_count = sum(1 for s in subsystems.values() if s == "unavailable")

  IF unavailable_count > 2:
    status = "degraded"
  ELIF stub_count > 0:
    status = "partial"
  ELSE:
    status = "healthy"

  RETURN HealthResponse(
    status=status,
    version=ELITE_VERSION,
    uptime_seconds=runtime.uptime(),
    checks={
      "subsystems": subsystems,
      "stub_count": stub_count,
      "unavailable_count": unavailable_count,
    }
  )
```

---

## 5. ServiceMonitor + PrometheusRule (Base)

```yaml
# deploy/k8s/base/monitoring.yaml

PSEUDOCODE:
  ServiceMonitor: bizra-elite-monitor
    selector: app=bizra-elite
    endpoints:
      - port: http
        path: /v1/metrics
        interval: 15s

  PrometheusRule: bizra-critical-alerts
    groups:
      - name: bizra.critical
        rules:
          - alert: ElitePodDown
            expr: up{job="bizra-elite"} == 0
            for: 1m
            labels: {severity: critical}

          - alert: HighQueryLatency
            expr: >
              histogram_quantile(0.95,
                rate(bizra_query_duration_seconds_bucket[5m])) > 10
            for: 5m
            labels: {severity: warning}

          - alert: SNRBelowThreshold
            expr: bizra_snr_score < 0.85
            for: 3m
            labels: {severity: critical}
            annotations:
              description: "SNR score {{ $value }} below UNIFIED_SNR_THRESHOLD (0.85)"

          - alert: IhsanBelowThreshold
            expr: bizra_ihsan_score < 0.95
            for: 3m
            labels: {severity: warning}

          - alert: SubsystemDegraded
            expr: bizra_health_stub_count > 2
            for: 5m
            labels: {severity: warning}
            annotations:
              description: "{{ $value }} subsystems running on stubs"

          - alert: HighMemoryUsage
            expr: >
              container_memory_working_set_bytes{container="bizra-elite"}
              / container_spec_memory_limit_bytes > 0.9
            for: 5m
            labels: {severity: warning}

  RATIONALE:
    Deming: Quality is achieved through continuous measurement.
    Shannon: Constitutional thresholds (SNR, Ihsan) become alert thresholds.
    These are the minimum viable alerts for production operation.
```

---

## 6. Proactive Provider Hardening

```
# MODIFY: core/sovereign/runtime_core.py — _register_proactive_providers()

FUNCTION _register_proactive_providers(self):
  """
  Replace silent ImportError swallowing with explicit degradation tracking.

  Standing on Giants: Lamport (failure must be observable)
  """
  self._degraded_providers = []

  providers = [
    ("opportunity_pipeline", "core.sovereign.opportunity_pipeline", "OpportunityPipeline"),
    ("proactive_scheduler", "core.sovereign.proactive_scheduler", "ProactiveScheduler"),
    ("predictive_monitor", "core.sovereign.predictive_monitor", "PredictiveMonitor"),
  ]

  FOR name, module_path, class_name IN providers:
    TRY:
      module = importlib.import_module(module_path)
      cls = getattr(module, class_name)
      setattr(self, f"_{name}", cls(self))
      self.logger.info(f"Proactive provider registered: {name}")
    EXCEPT ImportError AS e:
      self._degraded_providers.append(name)
      self.logger.warning(f"Proactive provider unavailable: {name} ({e})")
    EXCEPT Exception AS e:
      self._degraded_providers.append(name)
      self.logger.error(f"Proactive provider failed to init: {name} ({e})")
```

---

## 7. TDD Anchors

```
TEST test_health_reports_stub_subsystems:
  runtime = create_runtime_with_stubs(["snr_maximizer", "guardian_council"])
  health = health_endpoint(runtime)
  ASSERT health.status == "partial"
  ASSERT health.checks["subsystems"]["snr_maximizer"] == "stub"
  ASSERT health.checks["subsystems"]["guardian_council"] == "stub"
  ASSERT health.checks["stub_count"] == 2

TEST test_health_reports_healthy_when_all_active:
  runtime = create_runtime_full()
  health = health_endpoint(runtime)
  ASSERT health.status == "healthy"
  ASSERT health.checks["stub_count"] == 0

TEST test_health_reports_degraded:
  runtime = create_runtime_with_stubs(["snr", "guardian", "got"])
  health = health_endpoint(runtime)
  ASSERT health.status == "degraded"

TEST test_degraded_providers_tracked:
  runtime = SovereignRuntime(config)
  # Mock missing providers
  runtime._register_proactive_providers()
  ASSERT "opportunity_pipeline" IN runtime._degraded_providers OR \
         len(runtime._degraded_providers) >= 0

TEST test_networkpolicy_allows_prometheus_scrape:
  # k8s integration: verify prometheus can reach /v1/metrics
  result = kubectl_exec("prometheus-pod", "curl -s bizra-elite:8000/v1/metrics")
  ASSERT "bizra_query_duration" IN result

TEST test_vpa_adjusts_memory:
  # After running with VPA for sufficient recommendation window,
  # verify requests are adjusted
  vpa = kubectl_get("vpa/bizra-elite-vpa")
  ASSERT vpa.status.recommendation IS NOT None

TEST test_keda_scales_on_queue_depth:
  # Simulate queue depth > 10, verify HPA target increases
  push_metric("bizra_inference_queue_depth", 15)
  wait_for_condition(lambda: get_replicas("bizra-elite") > 1, timeout=120)
```

## Deployment Order

1. **NetworkPolicy** — apply first (restrict before exposing)
2. **ServiceMonitor + PrometheusRule** — enable observability
3. **Runtime degradation reporting** — code change + redeploy
4. **VPA** — install VPA controller, apply manifests
5. **KEDA** — install KEDA controller, apply ScaledObject
6. **Ingress** — apply with cert-manager issuer

## Success Criteria

| Metric | Target |
|--------|--------|
| NetworkPolicy | East-west traffic segmented, only explicit allows |
| Health granularity | Per-subsystem status (active/stub/unavailable) |
| Alert coverage | SNR < 0.85, Ihsan < 0.95, pod down, latency p95 > 10s |
| VPA | Memory requests auto-adjusted within 24h |
| KEDA | Scales on queue depth > 10 or p95 > 5s |
| Degraded providers | Tracked in runtime, exposed in health check |
