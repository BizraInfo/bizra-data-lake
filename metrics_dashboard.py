# BIZRA Operational Metrics Dashboard v1.0
# Real-time SNR monitoring, Ihsān compliance tracking, and system health
# Part of SAPE Implementation Blueprint

import json
import os
import time
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque
import threading

# Import unified thresholds from authoritative source
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
    STRICT_IHSAN_THRESHOLD,
)

# Configuration — cross-platform path resolution
_data_lake_root = os.environ.get("BIZRA_DATA_LAKE_ROOT", "/mnt/c/BIZRA-DATA-LAKE")
METRICS_ROOT = Path(_data_lake_root) / "03_INDEXED" / "metrics"

# Use unified thresholds
IHSAN_THRESHOLD = STRICT_IHSAN_THRESHOLD  # 0.99 for operational monitoring
ACCEPTABLE_THRESHOLD = UNIFIED_IHSAN_THRESHOLD  # 0.95 standard threshold
METRICS_RETENTION_HOURS = 24
ALERT_COOLDOWN_SECONDS = 300


class MetricType(Enum):
    """Types of metrics tracked by the dashboard"""
    SNR = "snr"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    GRAPH_OPS = "graph_operations"
    EMBEDDING_OPS = "embedding_operations"
    LLM_CALLS = "llm_calls"
    IHSAN_COMPLIANCE = "ihsan_compliance"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Single metric measurement"""
    timestamp: str
    metric_type: str
    value: float
    context: Dict = field(default_factory=dict)


@dataclass
class Alert:
    """System alert"""
    timestamp: str
    level: str
    metric_type: str
    message: str
    value: float
    threshold: float


@dataclass
class SNRMetrics:
    """Detailed SNR breakdown"""
    overall_snr: float
    signal_strength: float
    information_density: float
    symbolic_grounding: float
    coverage_balance: float
    ihsan_achieved: bool
    timestamp: str


@dataclass
class SystemHealth:
    """Overall system health status"""
    status: str  # healthy, degraded, critical
    snr_average: float
    ihsan_compliance_rate: float
    error_rate: float
    latency_p95: float
    active_alerts: int
    last_check: str


class MetricsCollector:
    """Collects and stores metrics from all BIZRA components"""

    def __init__(self):
        self.metrics_dir = METRICS_ROOT
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # In-memory buffers for real-time metrics
        self._snr_buffer: deque = deque(maxlen=1000)
        self._latency_buffer: deque = deque(maxlen=1000)
        self._error_buffer: deque = deque(maxlen=1000)
        self._alerts: List[Alert] = []
        self._last_alert_time: Dict[str, float] = {}

        # Lock for thread safety
        self._lock = threading.Lock()

        # Load historical metrics
        self._load_historical()

    def _load_historical(self):
        """Load metrics from persistent storage"""
        metrics_file = self.metrics_dir / "metrics_history.jsonl"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    for line in f:
                        if line.strip():
                            point = json.loads(line)
                            # Only load recent metrics
                            ts = datetime.fromisoformat(point["timestamp"])
                            if datetime.now() - ts < timedelta(hours=METRICS_RETENTION_HOURS):
                                if point["metric_type"] == "snr":
                                    self._snr_buffer.append(point)
                                elif point["metric_type"] == "latency":
                                    self._latency_buffer.append(point)
                                elif point["metric_type"] == "error_rate":
                                    self._error_buffer.append(point)
            except Exception as e:
                print(f"⚠️ Error loading historical metrics: {e}")

    def record_snr(self, snr_metrics: SNRMetrics):
        """Record SNR measurement"""
        with self._lock:
            point = MetricPoint(
                timestamp=snr_metrics.timestamp,
                metric_type=MetricType.SNR.value,
                value=snr_metrics.overall_snr,
                context={
                    "signal_strength": snr_metrics.signal_strength,
                    "information_density": snr_metrics.information_density,
                    "symbolic_grounding": snr_metrics.symbolic_grounding,
                    "coverage_balance": snr_metrics.coverage_balance,
                    "ihsan_achieved": snr_metrics.ihsan_achieved
                }
            )
            self._snr_buffer.append(asdict(point))
            self._persist_metric(point)

            # Check for alerts
            if snr_metrics.overall_snr < ACCEPTABLE_THRESHOLD:
                self._trigger_alert(
                    AlertLevel.CRITICAL,
                    MetricType.SNR,
                    f"SNR below acceptable threshold: {snr_metrics.overall_snr:.4f}",
                    snr_metrics.overall_snr,
                    ACCEPTABLE_THRESHOLD
                )
            elif snr_metrics.overall_snr < IHSAN_THRESHOLD:
                self._trigger_alert(
                    AlertLevel.WARNING,
                    MetricType.SNR,
                    f"SNR below Ihsān threshold: {snr_metrics.overall_snr:.4f}",
                    snr_metrics.overall_snr,
                    IHSAN_THRESHOLD
                )

    def record_latency(self, latency_ms: float, operation: str):
        """Record operation latency"""
        with self._lock:
            point = MetricPoint(
                timestamp=datetime.now().isoformat(),
                metric_type=MetricType.LATENCY.value,
                value=latency_ms,
                context={"operation": operation}
            )
            self._latency_buffer.append(asdict(point))
            self._persist_metric(point)

            # Alert on high latency
            if latency_ms > 5000:  # 5 second threshold
                self._trigger_alert(
                    AlertLevel.WARNING,
                    MetricType.LATENCY,
                    f"High latency detected: {latency_ms:.0f}ms for {operation}",
                    latency_ms,
                    5000
                )

    def record_error(self, error_type: str, message: str, recoverable: bool):
        """Record error occurrence"""
        with self._lock:
            point = MetricPoint(
                timestamp=datetime.now().isoformat(),
                metric_type=MetricType.ERROR_RATE.value,
                value=1.0,
                context={
                    "error_type": error_type,
                    "message": message,
                    "recoverable": recoverable
                }
            )
            self._error_buffer.append(asdict(point))
            self._persist_metric(point)

            if not recoverable:
                self._trigger_alert(
                    AlertLevel.CRITICAL,
                    MetricType.ERROR_RATE,
                    f"Non-recoverable error: {error_type} - {message}",
                    1.0,
                    0.0
                )

    def _trigger_alert(self, level: AlertLevel, metric_type: MetricType,
                       message: str, value: float, threshold: float):
        """Trigger an alert with cooldown"""
        alert_key = f"{level.value}_{metric_type.value}"
        now = time.time()

        if alert_key in self._last_alert_time:
            if now - self._last_alert_time[alert_key] < ALERT_COOLDOWN_SECONDS:
                return  # Skip alert due to cooldown

        alert = Alert(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            metric_type=metric_type.value,
            message=message,
            value=value,
            threshold=threshold
        )
        self._alerts.append(alert)
        self._last_alert_time[alert_key] = now
        self._persist_alert(alert)

        # Print alert
        symbol = "🚨" if level == AlertLevel.CRITICAL else "⚠️" if level == AlertLevel.WARNING else "ℹ️"
        print(f"{symbol} [{level.value.upper()}] {message}")

    def _persist_metric(self, point: MetricPoint):
        """Persist metric to file"""
        metrics_file = self.metrics_dir / "metrics_history.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(asdict(point)) + "\n")

    def _persist_alert(self, alert: Alert):
        """Persist alert to file"""
        alerts_file = self.metrics_dir / "alerts.jsonl"
        with open(alerts_file, "a") as f:
            f.write(json.dumps(asdict(alert)) + "\n")

    def get_snr_statistics(self) -> Dict:
        """Get SNR statistics"""
        with self._lock:
            if not self._snr_buffer:
                return {"status": "no_data"}

            values = [p["value"] for p in self._snr_buffer]
            ihsan_count = sum(1 for p in self._snr_buffer
                             if p.get("context", {}).get("ihsan_achieved", False))

            return {
                "count": len(values),
                "average": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "ihsan_compliance_rate": ihsan_count / len(values) if values else 0,
                "below_threshold_count": sum(1 for v in values if v < ACCEPTABLE_THRESHOLD)
            }

    def get_latency_statistics(self) -> Dict:
        """Get latency statistics"""
        with self._lock:
            if not self._latency_buffer:
                return {"status": "no_data"}

            values = sorted([p["value"] for p in self._latency_buffer])

            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "average_ms": statistics.mean(values),
                "median_ms": statistics.median(values),
                "p95_ms": values[int(len(values) * 0.95)] if values else 0,
                "p99_ms": values[int(len(values) * 0.99)] if values else 0,
                "max_ms": max(values)
            }

    def get_error_rate(self) -> float:
        """Calculate recent error rate"""
        with self._lock:
            if not self._snr_buffer:
                return 0.0

            total_ops = len(self._snr_buffer) + len(self._error_buffer)
            if total_ops == 0:
                return 0.0

            return len(self._error_buffer) / total_ops

    def get_active_alerts(self) -> List[Alert]:
        """Get recent active alerts"""
        cutoff = datetime.now() - timedelta(hours=1)
        return [a for a in self._alerts
                if datetime.fromisoformat(a.timestamp) > cutoff]


class MetricsDashboard:
    """Main dashboard interface for BIZRA metrics visualization"""

    def __init__(self):
        self.collector = MetricsCollector()
        self._running = False

    def get_system_health(self) -> SystemHealth:
        """Get overall system health status"""
        snr_stats = self.collector.get_snr_statistics()
        latency_stats = self.collector.get_latency_statistics()
        error_rate = self.collector.get_error_rate()
        active_alerts = self.collector.get_active_alerts()

        # Determine health status
        critical_alerts = [a for a in active_alerts if a.level == "critical"]
        warning_alerts = [a for a in active_alerts if a.level == "warning"]

        if critical_alerts or error_rate > 0.1:
            status = "critical"
        elif warning_alerts or (snr_stats.get("average", 1) < ACCEPTABLE_THRESHOLD):
            status = "degraded"
        else:
            status = "healthy"

        return SystemHealth(
            status=status,
            snr_average=snr_stats.get("average", 0),
            ihsan_compliance_rate=snr_stats.get("ihsan_compliance_rate", 0),
            error_rate=error_rate,
            latency_p95=latency_stats.get("p95_ms", 0),
            active_alerts=len(active_alerts),
            last_check=datetime.now().isoformat()
        )

    def print_dashboard(self):
        """Print formatted dashboard to console"""
        health = self.get_system_health()
        snr_stats = self.collector.get_snr_statistics()
        latency_stats = self.collector.get_latency_statistics()

        # Status symbol
        status_symbol = {
            "healthy": "✅",
            "degraded": "⚠️",
            "critical": "🚨"
        }.get(health.status, "❓")

        print("\n" + "=" * 60)
        print("           BIZRA OPERATIONAL METRICS DASHBOARD")
        print("=" * 60)
        print(f"\n  System Status: {status_symbol} {health.status.upper()}")
        print(f"  Last Check: {health.last_check}")

        # SNR Section
        print("\n  📊 SNR METRICS")
        print("  " + "-" * 40)
        if snr_stats.get("status") != "no_data":
            ihsan_symbol = "✅" if snr_stats["ihsan_compliance_rate"] >= 0.95 else "⚠️"
            print(f"  Average SNR:        {snr_stats['average']:.4f}")
            print(f"  Median SNR:         {snr_stats['median']:.4f}")
            print(f"  Min/Max:            {snr_stats['min']:.4f} / {snr_stats['max']:.4f}")
            print(f"  Std Deviation:      {snr_stats['std_dev']:.4f}")
            print(f"  Ihsān Compliance:   {ihsan_symbol} {snr_stats['ihsan_compliance_rate']*100:.1f}%")
            print(f"  Below Threshold:    {snr_stats['below_threshold_count']} queries")
        else:
            print("  No SNR data collected yet")

        # Latency Section
        print("\n  ⏱️  LATENCY METRICS")
        print("  " + "-" * 40)
        if latency_stats.get("status") != "no_data":
            print(f"  Average:            {latency_stats['average_ms']:.0f}ms")
            print(f"  Median:             {latency_stats['median_ms']:.0f}ms")
            print(f"  P95:                {latency_stats['p95_ms']:.0f}ms")
            print(f"  P99:                {latency_stats['p99_ms']:.0f}ms")
            print(f"  Max:                {latency_stats['max_ms']:.0f}ms")
        else:
            print("  No latency data collected yet")

        # Error Section
        print("\n  ❌ ERROR METRICS")
        print("  " + "-" * 40)
        error_symbol = "✅" if health.error_rate < 0.01 else "⚠️" if health.error_rate < 0.05 else "🚨"
        print(f"  Error Rate:         {error_symbol} {health.error_rate*100:.2f}%")

        # Alerts Section
        alerts = self.collector.get_active_alerts()
        print("\n  🔔 ACTIVE ALERTS")
        print("  " + "-" * 40)
        if alerts:
            for alert in alerts[-5:]:  # Show last 5
                alert_symbol = "🚨" if alert.level == "critical" else "⚠️"
                print(f"  {alert_symbol} [{alert.level}] {alert.message[:40]}...")
        else:
            print("  ✅ No active alerts")

        # Thresholds Reference
        print("\n  📏 THRESHOLDS")
        print("  " + "-" * 40)
        print(f"  Ihsān Threshold:    {IHSAN_THRESHOLD}")
        print(f"  Acceptable Min:     {ACCEPTABLE_THRESHOLD}")
        print(f"  Latency Target:     <1500ms (p95)")
        print(f"  Error Rate Target:  <1%")

        print("\n" + "=" * 60)

    def export_report(self, output_path: Optional[Path] = None) -> Dict:
        """Export metrics report as JSON"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "system_health": asdict(self.get_system_health()),
            "snr_statistics": self.collector.get_snr_statistics(),
            "latency_statistics": self.collector.get_latency_statistics(),
            "error_rate": self.collector.get_error_rate(),
            "active_alerts": [asdict(a) for a in self.collector.get_active_alerts()],
            "thresholds": {
                "ihsan": IHSAN_THRESHOLD,
                "acceptable": ACCEPTABLE_THRESHOLD
            }
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"📊 Report exported to {output_path}")

        return report


# Convenience functions for integration with other engines

_collector: Optional[MetricsCollector] = None


def get_collector() -> MetricsCollector:
    """Get or create singleton collector"""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


def record_snr(overall: float, signal: float, density: float,
               grounding: float, balance: float):
    """Quick SNR recording function"""
    collector = get_collector()
    metrics = SNRMetrics(
        overall_snr=overall,
        signal_strength=signal,
        information_density=density,
        symbolic_grounding=grounding,
        coverage_balance=balance,
        ihsan_achieved=overall >= IHSAN_THRESHOLD,
        timestamp=datetime.now().isoformat()
    )
    collector.record_snr(metrics)


def record_latency(latency_ms: float, operation: str):
    """Quick latency recording function"""
    get_collector().record_latency(latency_ms, operation)


def record_error(error_type: str, message: str, recoverable: bool = True):
    """Quick error recording function"""
    get_collector().record_error(error_type, message, recoverable)


# Main execution
if __name__ == "__main__":
    print("🚀 BIZRA Metrics Dashboard v1.0")
    print("=" * 40)

    # Create dashboard
    dashboard = MetricsDashboard()

    # Simulate some metrics for demonstration
    print("\n📈 Simulating metrics collection...")

    # Simulate SNR measurements
    import random
    for i in range(20):
        snr = random.uniform(0.92, 1.0)
        record_snr(
            overall=snr,
            signal=random.uniform(0.9, 1.0),
            density=random.uniform(0.85, 1.0),
            grounding=random.uniform(0.88, 1.0),
            balance=random.uniform(0.9, 1.0)
        )

    # Simulate latency measurements
    for i in range(20):
        record_latency(random.uniform(100, 2000), "query_processing")

    # Display dashboard
    dashboard.print_dashboard()

    # Export report
    report_path = METRICS_ROOT / "metrics_report.json"
    dashboard.export_report(report_path)

    print("\n✅ Metrics dashboard initialized and operational")
    print(f"📁 Metrics stored at: {METRICS_ROOT}")
