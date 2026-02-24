# 03: Observability — Metrics, Structured Logs, Alerts

## Standing on Giants
Shannon (information measurement, 1948) · Beyer/Google SRE (observability, 2016) · Nygard (Release It! stability, 2007)

## Overview

Complete observability for Phase 46 components during canary rollout. Three pillars: Prometheus metrics, structured request logs, and Prometheus alerting rules.

## Metrics

### Metric Definitions

All metrics prefixed with `phase46_` to avoid collision with existing `sovereign_*` metrics.

```
# Search
phase46_search_requests_total          counter    Total search requests routed to Phase 46
phase46_search_errors_total            counter    Search errors (FAISS load fail, dim mismatch, etc.)
phase46_search_latency_seconds         histogram  Search latency (buckets: 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
phase46_search_results_count           histogram  Number of results per search (buckets: 0, 1, 5, 10, 25, 50)
phase46_search_hit_rate                gauge      Fraction of searches returning >= 1 result (5m window)

# GoT Bridge
phase46_got_requests_total             counter    Total GoT bridge invocations
phase46_got_convergence_pass_total     counter    GoT reached convergence SNR threshold
phase46_got_fallback_total             counter    GoT fell back to standard reasoning
phase46_got_latency_seconds            histogram  GoT bridge latency

# HMM
phase46_hmm_observations_total         counter    Total HMM observations accepted
phase46_hmm_drops_total                counter    Total HMM observations dropped (caller isolation)
phase46_hmm_prediction_confidence      summary    HMM prediction confidence (p50, p95, p99)
phase46_hmm_observation_entropy        gauge      Shannon entropy of observation distribution

# Resonance (combined pipeline)
phase46_resonance_requests_total       counter    Total resonance pipeline invocations
phase46_resonance_combined_snr         summary    Combined SNR score (p50, p95, p99)
phase46_resonance_latency_seconds      histogram  Full pipeline latency
phase46_resonance_components_active    gauge      Number of active pipeline components (0-3)
```

### Pseudocode: Metrics Emitter

```
MODULE core/rollout/metrics.py

IMPORT time, math, logging
FROM typing IMPORT Dict, Optional, Any
FROM collections IMPORT Counter

logger = logging.getLogger(__name__)


CLASS Phase46Metrics:
    """In-process metrics collector for Phase 46 observability.

    Emits to Prometheus via the existing exposition endpoint
    or falls back to structured logging when Prometheus unavailable.
    """

    FUNCTION __init__(self):
        self._counters: Dict[str, int] = Counter()
        self._latencies: Dict[str, list] = {}  # component -> [latency_ms, ...]
        self._snr_values: list = []
        self._hmm_confidences: list = []
        self._observation_counts: Dict[str, int] = Counter()  # symbol -> count
        self._prometheus = self._try_init_prometheus()

    FUNCTION _try_init_prometheus(self) -> Optional[Any]:
        """Attempt to import prometheus_client. Non-fatal if absent."""
        TRY:
            IMPORT prometheus_client
            # Register metrics only once (check registry)
            RETURN prometheus_client
        EXCEPT ImportError:
            logger.info("prometheus_client not available; using structured logging")
            RETURN None

    # --- Counter operations ---

    FUNCTION inc(self, metric_name: str, value: int = 1, labels: Dict = None):
        """Increment a counter metric."""
        self._counters[metric_name] += value
        IF self._prometheus:
            self._prometheus_inc(metric_name, value, labels)

    # --- Latency operations ---

    FUNCTION record_latency(self, component: str, latency_ms: float):
        """Record a latency observation."""
        IF component NOT IN self._latencies:
            self._latencies[component] = []
        self._latencies[component].append(latency_ms)
        # Keep bounded (last 10000 observations)
        IF len(self._latencies[component]) > 10000:
            self._latencies[component] = self._latencies[component][-5000:]

    # --- SNR operations ---

    FUNCTION record_snr(self, combined_snr: float):
        """Record a combined_snr value."""
        self._snr_values.append(combined_snr)
        IF len(self._snr_values) > 10000:
            self._snr_values = self._snr_values[-5000:]

    # --- HMM operations ---

    FUNCTION record_hmm_confidence(self, confidence: float):
        """Record HMM prediction confidence."""
        self._hmm_confidences.append(confidence)
        IF len(self._hmm_confidences) > 10000:
            self._hmm_confidences = self._hmm_confidences[-5000:]

    FUNCTION record_hmm_observation(self, symbol: str):
        """Record an HMM observation for entropy calculation."""
        self._observation_counts[symbol] += 1

    FUNCTION observation_entropy(self) -> float:
        """Compute Shannon entropy of observation distribution."""
        total = sum(self._observation_counts.values())
        IF total == 0:
            RETURN 0.0
        entropy = 0.0
        FOR count IN self._observation_counts.values():
            p = count / total
            IF p > 0:
                entropy -= p * math.log2(p)
        RETURN entropy

    # --- Percentile operations ---

    FUNCTION percentile(self, values: list, p: float) -> float:
        """Compute p-th percentile (0-100) of a sorted list."""
        IF NOT values:
            RETURN 0.0
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        IF c >= len(sorted_vals):
            RETURN sorted_vals[-1]
        RETURN sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

    # --- Snapshot for health endpoint ---

    FUNCTION snapshot(self) -> Dict[str, Any]:
        """Full metrics snapshot for mcp_health and alerting."""
        RETURN {
            "counters": dict(self._counters),
            "search": {
                "latency_p50_ms": self.percentile(
                    self._latencies.get("search", []), 50
                ),
                "latency_p95_ms": self.percentile(
                    self._latencies.get("search", []), 95
                ),
                "hit_rate": self._compute_hit_rate(),
            },
            "got_bridge": {
                "convergence_pass_rate": self._compute_rate(
                    "got_convergence_pass", "got_requests"
                ),
                "fallback_rate": self._compute_rate(
                    "got_fallback", "got_requests"
                ),
            },
            "hmm": {
                "confidence_p50": self.percentile(self._hmm_confidences, 50),
                "confidence_p95": self.percentile(self._hmm_confidences, 95),
                "observation_entropy": self.observation_entropy(),
            },
            "resonance": {
                "combined_snr_p50": self.percentile(self._snr_values, 50),
                "combined_snr_p95": self.percentile(self._snr_values, 95),
            },
        }
```

## Structured Logs

### Per-Request Log Schema

Every Phase 46 component emits a structured log entry on each request:

```json
{
    "ts": "2026-02-19T14:30:00Z",
    "component": "search",
    "event": "request",
    "routed": true,
    "canary_percent": 50,
    "fallback_used": false,
    "latency_ms": 42.3,
    "result_count": 7,
    "error_code": null,
    "caller_id": "mcp",
    "request_key": "abc123"
}
```

### Pseudocode: Structured Logger

```
FUNCTION log_phase46_event(
    component: str,
    event: str,
    routed: bool,
    canary_percent: int,
    latency_ms: float,
    fallback_used: bool = False,
    error_code: str = None,
    caller_id: str = None,
    extra: Dict = None,
):
    """Emit structured log for Phase 46 observability."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "event": event,
        "routed": routed,
        "canary_percent": canary_percent,
        "fallback_used": fallback_used,
        "latency_ms": round(latency_ms, 2),
        "error_code": error_code,
        "caller_id": caller_id,
    }
    IF extra:
        entry.update(extra)
    logger.info("phase46_event", extra={"phase46": entry})
```

## Alerting Rules

### Added to `deploy/monitoring/alerting-rules.yaml`

```yaml
# =========================================================================
# Phase 46 — Cognitive Resonance Alerts (Phase 47.1 Canary)
# =========================================================================
- name: bizra.phase46.alerts
  rules:
    # Search error rate > 2% over 15m
    - alert: Phase46SearchErrorRate
      expr: |
        rate(phase46_search_errors_total[15m])
        / rate(phase46_search_requests_total[15m])
        > 0.02
      for: 5m
      labels:
        severity: warning
        team: sovereign
        phase: "46"
      annotations:
        summary: "Phase 46 search error rate exceeds 2%"
        description: |
          Search error rate {{ $value | printf "%.2f" }}% over 15m window.
          Check FAISS index availability and embedding dimension match.

    # GoT fallback rate > 20% over 15m
    - alert: Phase46GoTFallbackRate
      expr: |
        rate(phase46_got_fallback_total[15m])
        / rate(phase46_got_requests_total[15m])
        > 0.20
      for: 5m
      labels:
        severity: warning
        team: sovereign
        phase: "46"
      annotations:
        summary: "Phase 46 GoT fallback rate exceeds 20%"
        description: |
          GoT bridge falling back {{ $value | printf "%.1f" }}% of the time.
          Check inference backend availability and convergence threshold.

    # End-to-end p95 latency delta > 30% over 30m
    - alert: Phase46LatencyRegression
      expr: |
        histogram_quantile(0.95, rate(phase46_resonance_latency_seconds_bucket[30m]))
        > 1.3 * histogram_quantile(0.95, rate(phase46_resonance_latency_seconds_bucket[30m] offset 1h))
      for: 10m
      labels:
        severity: warning
        team: sovereign
        phase: "46"
      annotations:
        summary: "Phase 46 resonance p95 latency regressed > 30%"

    # HMM confidence p50 < 0.55 over 30m
    - alert: Phase46HMMConfidenceLow
      expr: |
        phase46_hmm_prediction_confidence{quantile="0.5"} < 0.55
      for: 30m
      labels:
        severity: warning
        team: sovereign
        phase: "46"
      annotations:
        summary: "Phase 46 HMM prediction confidence below 0.55"
        description: |
          HMM median confidence at {{ $value | printf "%.3f" }}.
          May indicate insufficient observation diversity or model drift.

    # Resonance combined_snr p50 drops > 15% from baseline
    - alert: Phase46ResonanceSNRDrop
      expr: |
        phase46_resonance_combined_snr{quantile="0.5"}
        < 0.85 * phase46_resonance_combined_snr_baseline
      for: 30m
      labels:
        severity: critical
        team: sovereign
        phase: "46"
      annotations:
        summary: "Phase 46 combined SNR dropped > 15% from baseline"
        description: |
          Combined SNR at {{ $value | printf "%.3f" }}, baseline breach.
          Triggers strict rollback evaluation.
```

## TDD Anchors

```python
class TestPhase46Metrics:

    def test_counter_increment(self):
        m = Phase46Metrics()
        m.inc("search_requests")
        m.inc("search_requests")
        assert m.snapshot()["counters"]["search_requests"] == 2

    def test_latency_percentile(self):
        m = Phase46Metrics()
        for i in range(100):
            m.record_latency("search", float(i))
        snap = m.snapshot()
        assert 45 < snap["search"]["latency_p50_ms"] < 55

    def test_observation_entropy_uniform(self):
        """Uniform distribution over N symbols has entropy log2(N)."""
        m = Phase46Metrics()
        for s in ["search", "edit", "test", "deploy"]:
            for _ in range(100):
                m.record_hmm_observation(s)
        entropy = m.observation_entropy()
        assert abs(entropy - 2.0) < 0.01  # log2(4) = 2.0

    def test_observation_entropy_single_symbol(self):
        """Single symbol has entropy 0."""
        m = Phase46Metrics()
        for _ in range(100):
            m.record_hmm_observation("search")
        assert m.observation_entropy() == 0.0

    def test_snr_recording(self):
        m = Phase46Metrics()
        for _ in range(100):
            m.record_snr(0.92)
        snap = m.snapshot()
        assert snap["resonance"]["combined_snr_p50"] == 0.92

    def test_snapshot_structure(self):
        m = Phase46Metrics()
        snap = m.snapshot()
        assert "counters" in snap
        assert "search" in snap
        assert "got_bridge" in snap
        assert "hmm" in snap
        assert "resonance" in snap
```
