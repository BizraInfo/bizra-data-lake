# Step 3: Health Endpoint Tiering

## Standing on Giants: Kubernetes (liveness/readiness probes) | Nygard (Release It!, 2007) | Fowler (health check patterns)

## Problem Statement

The SAPE audit finding F8 identified that `/v1/health` performs an 11-subsystem
availability scan on every probe. In production, Kubernetes liveness probes hit
this endpoint every 10-30 seconds. Each probe checks:

```
graph_of_thoughts, snr_maximizer, guardian_council, autonomous_loop,
cognitive_fusion, embedding_service, memory_coordinator, evidence_ledger,
rdve_engine, fate_gate, sat_controller
```

This is O(11) per probe — not O(n) in the data-structure sense, but each
subsystem check can involve I/O (Redis ping, file stat, embedding service
health). Under load, this creates observability overhead that competes with
actual work.

**Solution:** Split the health endpoint into three tiers following Kubernetes
conventions:

| Tier | Path | Latency | Purpose |
|------|------|---------|---------|
| Live | `/v1/health/live` | < 5ms | "Is the process alive?" — returns 200 if Python is running |
| Ready | `/v1/health/ready` | < 50ms | "Can it serve traffic?" — checks critical subsystems only |
| Deep | `/v1/health/deep` | < 500ms | "Is everything healthy?" — full 11-subsystem scan |

The existing `/v1/health` becomes an alias for `/v1/health/ready` for backward
compatibility.

## Target Files

| File | Action |
|------|--------|
| `core/sovereign/api.py` | Update: add tiered health routes |
| `deploy/mcp-compose.yaml` | Update: point liveness probe to `/v1/health/live` |
| `deploy/k8s/base/deployment-mcp.yaml` | Update: separate liveness/readiness probes |
| `tests/core/sovereign/test_api_health_tiering.py` | New: tests for all three tiers |

## Pseudocode

### core/sovereign/api.py — Health Tier Handlers

```pseudocode
# Add to route dispatch (line ~523):

IF path == "/v1/health/live" AND method == "GET":
    RETURN _handle_liveness()

IF path == "/v1/health/ready" AND method == "GET":
    RETURN await _handle_readiness()

IF path == "/v1/health/deep" AND method == "GET":
    RETURN await _handle_deep_health()

IF path == "/v1/health" AND method == "GET":
    RETURN await _handle_readiness()  # backward compat


FUNCTION _handle_liveness() -> Response:
    """Liveness probe — O(1), no I/O, no computation.

    Returns 200 if the Python process is running and can handle HTTP.
    This is the probe Kubernetes uses to decide whether to restart the pod.
    """
    RETURN Response(
        status=200,
        body=json.dumps({
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
        }),
    )


ASYNC FUNCTION _handle_readiness() -> Response:
    """Readiness probe — checks 3 critical subsystems only.

    Returns 200 if the service can serve traffic.
    Returns 503 if any critical subsystem is down.
    Critical subsystems: evidence_ledger, snr_maximizer, fate_gate
    """
    critical_checks = [
        ("evidence_ledger", "_evidence_ledger"),
        ("snr_maximizer", "_snr_optimizer"),
        ("fate_gate", "_ihsan_watchdog"),
    ]

    results = {}
    FOR name, attr IN critical_checks:
        component = getattr(self, attr, None)
        results[name] = component IS NOT None

    all_ready = all(results.values())
    status = 200 IF all_ready ELSE 503

    RETURN Response(
        status=status,
        body=json.dumps({
            "status": "ready" IF all_ready ELSE "not_ready",
            "checks": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }),
    )


ASYNC FUNCTION _handle_deep_health() -> Response:
    """Deep health — full 11-subsystem scan.

    Use this for dashboards and alerting, NOT for K8s probes.
    """
    # Existing implementation from _handle_health()
    RETURN await self._handle_health()
```

### K8s Manifest Update

```pseudocode
# deploy/k8s/base/deployment-mcp.yaml — separate probes:

containers:
  - name: bizra-sovereign
    livenessProbe:
      httpGet:
        path: /v1/health/live    # O(1), no I/O
        port: 8010
      initialDelaySeconds: 5
      periodSeconds: 10
      timeoutSeconds: 2
      failureThreshold: 3

    readinessProbe:
      httpGet:
        path: /v1/health/ready   # 3 critical checks
        port: 8010
      initialDelaySeconds: 10
      periodSeconds: 15
      timeoutSeconds: 5
      failureThreshold: 2

    startupProbe:
      httpGet:
        path: /v1/health/deep    # Full check at boot
        port: 8010
      initialDelaySeconds: 15
      periodSeconds: 10
      timeoutSeconds: 10
      failureThreshold: 30       # Allow 5 min for startup
```

## TDD Anchors

```pseudocode
TEST liveness_returns_200_immediately:
    response = client.get("/v1/health/live")
    ASSERT response.status_code == 200
    data = response.json()
    ASSERT data["status"] == "alive"
    ASSERT "pid" IN data
    ASSERT "timestamp" IN data

TEST liveness_is_fast:
    """Liveness must respond in < 5ms."""
    start = time.monotonic()
    client.get("/v1/health/live")
    elapsed = (time.monotonic() - start) * 1000
    ASSERT elapsed < 50  # generous bound for test envs

TEST readiness_checks_critical_subsystems:
    response = client.get("/v1/health/ready")
    data = response.json()
    ASSERT "checks" IN data
    ASSERT "evidence_ledger" IN data["checks"]
    ASSERT "snr_maximizer" IN data["checks"]
    ASSERT "fate_gate" IN data["checks"]
    # Should NOT check non-critical systems
    ASSERT "cognitive_fusion" NOT IN data["checks"]

TEST readiness_returns_503_when_critical_down:
    """If evidence_ledger is missing, readiness fails."""
    # Mock: set _evidence_ledger to None
    api._evidence_ledger = None
    response = client.get("/v1/health/ready")
    ASSERT response.status_code == 503
    ASSERT response.json()["status"] == "not_ready"

TEST deep_health_checks_all_11_subsystems:
    response = client.get("/v1/health/deep")
    data = response.json()
    # Deep check should include all 11 subsystems
    ASSERT len(data.get("checks", data.get("subsystems", {}))) >= 11

TEST backward_compat_health_routes_to_ready:
    """GET /v1/health should behave identically to /v1/health/ready."""
    r1 = client.get("/v1/health")
    r2 = client.get("/v1/health/ready")
    ASSERT r1.status_code == r2.status_code
    # Both should have same check set (timestamps may differ)
    ASSERT set(r1.json()["checks"].keys()) == set(r2.json()["checks"].keys())

TEST liveness_has_no_side_effects:
    """Liveness probe must not modify any state."""
    # Record state before
    state_before = capture_system_state()
    FOR _ IN range(100):
        client.get("/v1/health/live")
    state_after = capture_system_state()
    ASSERT state_before == state_after
```

## Acceptance Criteria

1. `/v1/health/live` returns 200 in < 5ms with no I/O
2. `/v1/health/ready` checks exactly 3 critical subsystems
3. `/v1/health/deep` performs full 11-subsystem scan (existing behavior)
4. `/v1/health` backward-compatible alias for `/v1/health/ready`
5. K8s manifests updated with separate liveness/readiness/startup probes
6. Docker compose health check updated to use `/v1/health/live`
7. Full test suite GREEN
