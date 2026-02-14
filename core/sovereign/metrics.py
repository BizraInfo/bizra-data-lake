"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███╗   ███╗███████╗████████╗██████╗ ██╗ ██████╗███████╗                   ║
║   ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║██╔════╝██╔════╝                   ║
║   ██╔████╔██║█████╗     ██║   ██████╔╝██║██║     ███████╗                   ║
║   ██║╚██╔╝██║██╔══╝     ██║   ██╔══██╗██║██║     ╚════██║                   ║
║   ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║██║╚██████╗███████║                   ║
║   ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝╚══════╝                   ║
║                                                                              ║
║                    SOVEREIGN METRICS COLLECTOR v1.0                          ║
║         Real-time System Metrics for Autonomous Decision Making              ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Collects metrics from:                                                     ║
║   • System resources (CPU, memory, GPU)                                      ║
║   • Inference backends (latency, throughput)                                 ║
║   • Query processing (SNR, Ihsān scores)                                     ║
║   • Federation health (peer count, consensus)                                ║
║                                                                              ║
║   Feeds metrics to AutonomousLoop for OODA cycle decisions                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Optional

logger = logging.getLogger("sovereign.metrics")

# =============================================================================
# METRIC TYPES
# =============================================================================


@dataclass
class MetricPoint:
    """A single metric data point."""

    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: dict[str, str] = field(default_factory=dict)
    unit: str = ""

    def to_prometheus(self) -> str:
        """Format as Prometheus metric."""
        labels_str = ""
        if self.labels:
            pairs = [f'{k}="{v}"' for k, v in self.labels.items()]
            labels_str = "{" + ",".join(pairs) + "}"
        return f"{self.name}{labels_str} {self.value}"


@dataclass
class MetricSeries:
    """Time series of metrics."""

    name: str
    points: Deque[MetricPoint] = field(default_factory=lambda: deque(maxlen=1000))
    unit: str = ""

    def add(self, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Add a data point."""
        self.points.append(
            MetricPoint(
                name=self.name,
                value=value,
                labels=labels or {},
                unit=self.unit,
            )
        )

    def latest(self) -> Optional[float]:
        """Get latest value."""
        return self.points[-1].value if self.points else None

    def average(self, window: int = 10) -> float:
        """Get average of last N points."""
        if not self.points:
            return 0.0
        recent = list(self.points)[-window:]
        return sum(p.value for p in recent) / len(recent)

    def trend(self, window: int = 10) -> str:
        """Determine trend direction."""
        if len(self.points) < 2:
            return "stable"
        recent = list(self.points)[-window:]
        if len(recent) < 2:
            return "stable"
        first_half = sum(p.value for p in recent[: len(recent) // 2]) / (
            len(recent) // 2
        )
        second_half = sum(p.value for p in recent[len(recent) // 2 :]) / (
            len(recent) - len(recent) // 2
        )
        diff = second_half - first_half
        if diff > 0.05 * first_half:
            return "increasing"
        elif diff < -0.05 * first_half:
            return "decreasing"
        return "stable"


@dataclass
class SystemSnapshot:
    """Complete system state snapshot."""

    timestamp: datetime = field(default_factory=datetime.now)

    # Resource metrics
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    disk_percent: float = 0.0

    # GPU metrics (if available)
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_temperature: float = 0.0

    # Inference metrics
    inference_latency_ms: float = 0.0
    inference_throughput: float = 0.0
    inference_queue_depth: int = 0

    # Quality metrics
    snr_score: float = 0.0
    ihsan_score: float = 0.0
    error_rate: float = 0.0

    # Federation metrics
    peer_count: int = 0
    consensus_health: float = 0.0

    # Derived metrics
    def resource_pressure(self) -> float:
        """Calculate overall resource pressure (0-1)."""
        weights = {"cpu": 0.3, "memory": 0.4, "gpu": 0.3}
        pressure = (
            weights["cpu"] * self.cpu_percent / 100
            + weights["memory"] * self.memory_percent / 100
            + weights["gpu"] * self.gpu_percent / 100
        )
        return min(1.0, pressure)

    def health_score(self) -> float:
        """Calculate overall health score (0-1)."""
        factors = [
            max(0, 1 - self.resource_pressure()),
            self.snr_score,
            self.ihsan_score,
            max(0, 1 - self.error_rate * 10),
            max(0, 1 - self.inference_latency_ms / 5000),
        ]
        return sum(factors) / len(factors)

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "resources": {
                "cpu_percent": self.cpu_percent,
                "memory_percent": self.memory_percent,
                "memory_used_mb": self.memory_used_mb,
                "gpu_percent": self.gpu_percent,
                "gpu_memory_percent": self.gpu_memory_percent,
            },
            "inference": {
                "latency_ms": self.inference_latency_ms,
                "throughput": self.inference_throughput,
                "queue_depth": self.inference_queue_depth,
            },
            "quality": {
                "snr": self.snr_score,
                "ihsan": self.ihsan_score,
                "error_rate": self.error_rate,
            },
            "federation": {
                "peers": self.peer_count,
                "consensus_health": self.consensus_health,
            },
            "derived": {
                "resource_pressure": self.resource_pressure(),
                "health_score": self.health_score(),
            },
        }


# =============================================================================
# METRICS COLLECTOR
# =============================================================================


class MetricsCollector:
    """
    Collects and aggregates system metrics.

    Usage:
        collector = MetricsCollector()
        await collector.start()

        snapshot = await collector.snapshot()
        print(f"Health: {snapshot.health_score()}")

        await collector.stop()
    """

    def __init__(
        self,
        collection_interval: float = 1.0,
        retention_seconds: int = 3600,
    ):
        self.collection_interval = collection_interval
        self.retention_seconds = retention_seconds

        # Metric series
        self.series: dict[str, MetricSeries] = {
            "cpu_percent": MetricSeries("cpu_percent", unit="%"),
            "memory_percent": MetricSeries("memory_percent", unit="%"),
            "gpu_percent": MetricSeries("gpu_percent", unit="%"),
            "gpu_memory_percent": MetricSeries("gpu_memory_percent", unit="%"),
            "inference_latency_ms": MetricSeries("inference_latency_ms", unit="ms"),
            "snr_score": MetricSeries("snr_score"),
            "ihsan_score": MetricSeries("ihsan_score"),
            "error_rate": MetricSeries("error_rate"),
            "health_score": MetricSeries("health_score"),
        }

        # Snapshots history
        self.snapshots: Deque[SystemSnapshot] = deque(maxlen=3600)

        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._external_metrics: dict[str, float] = {}

        # Callbacks for external metric sources
        self._collectors: list[Callable[[], dict[str, float]]] = []

    async def start(self) -> None:
        """Start collecting metrics."""
        self._running = True
        self._task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collector started")

    async def stop(self) -> None:
        """Stop collecting metrics."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collector stopped")

    def register_collector(self, collector: Callable[[], dict[str, float]]) -> None:
        """Register an external metrics collector."""
        self._collectors.append(collector)

    def update_metric(self, name: str, value: float) -> None:
        """Update a metric from external source."""
        self._external_metrics[name] = value

    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self._running:
            try:
                snapshot = await self._collect()
                self.snapshots.append(snapshot)

                # Update series
                self.series["cpu_percent"].add(snapshot.cpu_percent)
                self.series["memory_percent"].add(snapshot.memory_percent)
                self.series["gpu_percent"].add(snapshot.gpu_percent)
                self.series["gpu_memory_percent"].add(snapshot.gpu_memory_percent)
                self.series["inference_latency_ms"].add(snapshot.inference_latency_ms)
                self.series["snr_score"].add(snapshot.snr_score)
                self.series["ihsan_score"].add(snapshot.ihsan_score)
                self.series["error_rate"].add(snapshot.error_rate)
                self.series["health_score"].add(snapshot.health_score())

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")

            await asyncio.sleep(self.collection_interval)

    async def _collect(self) -> SystemSnapshot:
        """Collect all metrics."""
        snapshot = SystemSnapshot()

        # Collect system resources
        await self._collect_system(snapshot)

        # Collect GPU metrics
        await self._collect_gpu(snapshot)

        # Apply external metrics
        snapshot.inference_latency_ms = self._external_metrics.get(
            "inference_latency_ms", 0
        )
        snapshot.inference_throughput = self._external_metrics.get(
            "inference_throughput", 0
        )
        snapshot.snr_score = self._external_metrics.get("snr_score", 0.9)
        snapshot.ihsan_score = self._external_metrics.get("ihsan_score", 0.9)
        snapshot.error_rate = self._external_metrics.get("error_rate", 0)
        snapshot.peer_count = int(self._external_metrics.get("peer_count", 0))

        # Call registered collectors
        for collector in self._collectors:
            try:
                metrics = collector()
                for k, v in metrics.items():
                    if hasattr(snapshot, k):
                        setattr(snapshot, k, v)
            except Exception as e:
                logger.debug(f"Collector error: {e}")

        return snapshot

    async def _collect_system(self, snapshot: SystemSnapshot) -> None:
        """Collect system resource metrics."""
        try:
            import psutil

            # CPU
            snapshot.cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory
            mem = psutil.virtual_memory()
            snapshot.memory_percent = mem.percent
            snapshot.memory_used_mb = mem.used / (1024 * 1024)
            snapshot.memory_total_mb = mem.total / (1024 * 1024)

            # Disk
            disk = psutil.disk_usage("/")
            snapshot.disk_percent = disk.percent

        except ImportError:
            # Fallback without psutil
            pass
        except Exception as e:
            logger.debug(f"System metrics error: {e}")

    async def _collect_gpu(self, snapshot: SystemSnapshot) -> None:
        """Collect GPU metrics (NVIDIA)."""
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )

            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 4:
                    snapshot.gpu_percent = float(parts[0].strip())
                    snapshot.gpu_memory_used_mb = float(parts[1].strip())
                    total_mb = float(parts[2].strip())
                    snapshot.gpu_memory_percent = (
                        snapshot.gpu_memory_used_mb / total_mb * 100
                        if total_mb > 0
                        else 0
                    )
                    snapshot.gpu_temperature = float(parts[3].strip())

        except FileNotFoundError:
            pass  # nvidia-smi not available
        except Exception as e:
            logger.debug(f"GPU metrics error: {e}")

    async def snapshot(self) -> SystemSnapshot:
        """Get current snapshot."""
        if self.snapshots:
            return self.snapshots[-1]
        return await self._collect()

    def get_series(self, name: str) -> Optional[MetricSeries]:
        """Get a metric series by name."""
        return self.series.get(name)

    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value of a metric."""
        series = self.series.get(name)
        return series.latest() if series else None

    def get_average(self, name: str, window: int = 10) -> float:
        """Get average of a metric over window."""
        series = self.series.get(name)
        return series.average(window) if series else 0.0

    def get_trend(self, name: str) -> str:
        """Get trend of a metric."""
        series = self.series.get(name)
        return series.trend() if series else "unknown"

    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus format."""
        lines = [
            "# HELP sovereign_metrics BIZRA Sovereign Engine metrics",
            "# TYPE sovereign_metrics gauge",
        ]

        for name, series in self.series.items():
            if series.points:
                latest = series.points[-1]
                lines.append(latest.to_prometheus())

        return "\n".join(lines)

    def status(self) -> dict[str, Any]:
        """Get collector status."""
        return {
            "running": self._running,
            "collection_interval": self.collection_interval,
            "snapshots_count": len(self.snapshots),
            "series": {
                name: {
                    "latest": series.latest(),
                    "average": series.average(),
                    "trend": series.trend(),
                    "points": len(series.points),
                }
                for name, series in self.series.items()
            },
        }


# =============================================================================
# AUTONOMY INTEGRATION
# =============================================================================


def create_autonomy_observer(collector: MetricsCollector):
    """
    Create an observer function for AutonomousLoop.

    Usage:
        from core.sovereign.autonomy import AutonomousLoop
        from core.sovereign.metrics import MetricsCollector, create_autonomy_observer

        collector = MetricsCollector()
        loop = AutonomousLoop()
        loop.register_observer(create_autonomy_observer(collector))
    """

    async def observer(metrics) -> None:
        """Update autonomy metrics from system snapshot."""
        snapshot = await collector.snapshot()

        # Map snapshot to autonomy SystemMetrics
        metrics.snr_score = snapshot.snr_score
        metrics.ihsan_score = snapshot.ihsan_score
        metrics.latency_ms = snapshot.inference_latency_ms
        metrics.error_rate = snapshot.error_rate
        metrics.memory_usage = snapshot.memory_percent / 100

    return observer


def create_autonomy_analyzer(collector: MetricsCollector):
    """
    Create an analyzer function for AutonomousLoop.

    Usage:
        loop.register_analyzer(create_autonomy_analyzer(collector))
    """
    from .autonomy import DecisionCandidate, DecisionType

    async def analyzer(metrics) -> list:
        """Analyze metrics and generate decision candidates."""
        candidates = []
        snapshot = await collector.snapshot()

        # High memory pressure
        if snapshot.memory_percent > 85:
            candidates.append(
                DecisionCandidate(
                    decision_type=DecisionType.CORRECTIVE,
                    action="reduce_memory",
                    parameters={"current": snapshot.memory_percent},
                    expected_impact=0.2,
                    risk_score=0.3,
                    confidence=0.85,
                    rationale=f"Memory at {snapshot.memory_percent:.1f}%",
                    rollback_plan="restore_cache",
                )
            )

        # High GPU temperature
        if snapshot.gpu_temperature > 80:
            candidates.append(
                DecisionCandidate(
                    decision_type=DecisionType.PREVENTIVE,
                    action="throttle_inference",
                    parameters={"temperature": snapshot.gpu_temperature},
                    expected_impact=0.15,
                    risk_score=0.2,
                    confidence=0.9,
                    rationale=f"GPU at {snapshot.gpu_temperature}°C",
                    rollback_plan="restore_inference_rate",
                )
            )

        # High inference latency
        if snapshot.inference_latency_ms > 3000:
            candidates.append(
                DecisionCandidate(
                    decision_type=DecisionType.ADAPTIVE,
                    action="switch_inference_tier",
                    parameters={"current_latency": snapshot.inference_latency_ms},
                    expected_impact=0.3,
                    risk_score=0.25,
                    confidence=0.8,
                    rationale=f"Latency at {snapshot.inference_latency_ms:.0f}ms",
                    rollback_plan="revert_tier",
                )
            )

        # SNR degradation
        if snapshot.snr_score < 0.85:
            candidates.append(
                DecisionCandidate(
                    decision_type=DecisionType.CORRECTIVE,
                    action="boost_snr",
                    parameters={"current_snr": snapshot.snr_score},
                    expected_impact=0.25,
                    risk_score=0.15,
                    confidence=0.85,
                    rationale=f"SNR at {snapshot.snr_score:.3f}",
                    rollback_plan="revert_snr_config",
                )
            )

        return candidates

    return analyzer


# =============================================================================
# FACTORY
# =============================================================================


async def create_metrics_collector(
    collection_interval: float = 1.0,
    auto_start: bool = True,
) -> MetricsCollector:
    """Create and optionally start a metrics collector."""
    collector = MetricsCollector(collection_interval=collection_interval)
    if auto_start:
        await collector.start()
    return collector


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MetricsCollector",
    "MetricPoint",
    "MetricSeries",
    "SystemSnapshot",
    "create_metrics_collector",
    "create_autonomy_observer",
    "create_autonomy_analyzer",
]
