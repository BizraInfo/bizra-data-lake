"""Phase 46 observability metrics collector.

In-process metrics for canary rollout monitoring.  Emits to Prometheus
via the existing exposition endpoint or falls back to structured logging.

Standing on Giants: Shannon (information measurement, 1948)
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Phase46Metrics:
    """In-process metrics collector for Phase 46 observability.

    Collects counters, latencies, SNR values, HMM confidence scores,
    and observation symbol distributions.  Provides snapshot for health
    endpoints and percentile computation.
    """

    _MAX_OBSERVATIONS = 10_000
    _TRIM_TO = 5_000

    def __init__(self) -> None:
        self._counters: Counter = Counter()
        self._latencies: Dict[str, List[float]] = {}
        self._snr_values: List[float] = []
        self._hmm_confidences: List[float] = []
        self._observation_counts: Counter = Counter()
        self._start_time = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def inc(self, metric_name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self._counters[metric_name] += value

    def get_counter(self, metric_name: str) -> int:
        return self._counters.get(metric_name, 0)

    # ------------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------------

    def record_latency(self, component: str, latency_ms: float) -> None:
        """Record a latency observation for *component*."""
        buf = self._latencies.setdefault(component, [])
        buf.append(latency_ms)
        if len(buf) > self._MAX_OBSERVATIONS:
            self._latencies[component] = buf[-self._TRIM_TO :]

    # ------------------------------------------------------------------
    # SNR
    # ------------------------------------------------------------------

    def record_snr(self, combined_snr: float) -> None:
        """Record a combined_snr value from the resonance pipeline."""
        self._snr_values.append(combined_snr)
        if len(self._snr_values) > self._MAX_OBSERVATIONS:
            self._snr_values = self._snr_values[-self._TRIM_TO :]

    # ------------------------------------------------------------------
    # HMM
    # ------------------------------------------------------------------

    def record_hmm_confidence(self, confidence: float) -> None:
        """Record HMM prediction confidence."""
        self._hmm_confidences.append(confidence)
        if len(self._hmm_confidences) > self._MAX_OBSERVATIONS:
            self._hmm_confidences = self._hmm_confidences[-self._TRIM_TO :]

    def record_hmm_observation(self, symbol: str) -> None:
        """Record an HMM observation for entropy calculation."""
        self._observation_counts[symbol] += 1

    def observation_entropy(self) -> float:
        """Compute Shannon entropy of observation symbol distribution."""
        total = sum(self._observation_counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in self._observation_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    # ------------------------------------------------------------------
    # Rates
    # ------------------------------------------------------------------

    def compute_rate(self, numerator: str, denominator: str) -> float:
        """Compute ratio of two counters.  Returns 0.0 if denominator is 0."""
        denom = self._counters.get(denominator, 0)
        if denom == 0:
            return 0.0
        return self._counters.get(numerator, 0) / denom

    def compute_hit_rate(self) -> float:
        """Fraction of searches returning >= 1 result."""
        return self.compute_rate("search_hits", "search_requests")

    # ------------------------------------------------------------------
    # Percentile
    # ------------------------------------------------------------------

    @staticmethod
    def percentile(values: List[float], p: float) -> float:
        """Compute *p*-th percentile (0-100) using linear interpolation."""
        if not values:
            return 0.0
        s = sorted(values)
        k = (len(s) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(s) - 1)
        return s[f] + (k - f) * (s[c] - s[f])

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Full metrics snapshot for ``mcp_health`` and alerting."""
        return {
            "counters": dict(self._counters),
            "uptime_seconds": (
                datetime.now(timezone.utc) - self._start_time
            ).total_seconds(),
            "search": {
                "latency_p50_ms": self.percentile(
                    self._latencies.get("search", []), 50
                ),
                "latency_p95_ms": self.percentile(
                    self._latencies.get("search", []), 95
                ),
                "hit_rate": self.compute_hit_rate(),
            },
            "got_bridge": {
                "convergence_pass_rate": self.compute_rate(
                    "got_convergence_pass", "got_requests"
                ),
                "fallback_rate": self.compute_rate("got_fallback", "got_requests"),
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

    # ------------------------------------------------------------------
    # Structured log helper
    # ------------------------------------------------------------------

    @staticmethod
    def log_event(
        component: str,
        event: str,
        routed: bool,
        canary_percent: int,
        latency_ms: float,
        fallback_used: bool = False,
        error_code: Optional[str] = None,
        caller_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit structured log entry for Phase 46 observability."""
        entry: Dict[str, Any] = {
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
        if extra:
            entry.update(extra)
        logger.info("phase46_event %s", entry)
