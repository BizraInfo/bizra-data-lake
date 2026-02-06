"""
Muraqabah Engine — 24/7 Continuous Vigilance Monitoring
=======================================================
Implements Al-Ghazali's concept of Muraqabah (مراقبة) - constant awareness
and vigilance. Provides domain-specific sensors for proactive monitoring.

Standing on Giants: Al-Ghazali (1058-1111) + Observability Patterns + Event-Driven Architecture
"""

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Deque, Dict, List, Optional

from .event_bus import EventBus, EventPriority, get_event_bus

logger = logging.getLogger(__name__)


class MonitorDomain(str, Enum):
    """Domains monitored by Muraqabah."""

    FINANCIAL = "financial"  # Cash flow, portfolio, market signals
    HEALTH = "health"  # System health, mental indicators
    SOCIAL = "social"  # Relationships, network, conflicts
    COGNITIVE = "cognitive"  # Skills, learning, creativity
    ENVIRONMENTAL = "environmental"  # Energy, maintenance, security


class SensorState(Enum):
    """State of a sensor."""

    ACTIVE = auto()
    INACTIVE = auto()
    ERROR = auto()
    CALIBRATING = auto()


@dataclass
class SensorReading:
    """A reading from a Muraqabah sensor."""

    sensor_id: str = ""
    domain: MonitorDomain = MonitorDomain.ENVIRONMENTAL
    metric_name: str = ""
    value: float = 0.0
    unit: str = ""
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Opportunity:
    """A detected opportunity from Muraqabah analysis."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    domain: MonitorDomain = MonitorDomain.ENVIRONMENTAL
    description: str = ""
    estimated_value: float = 0.0  # Relative value 0-1
    urgency: float = 0.5  # 0=low, 1=urgent
    confidence: float = 0.9
    action_required: str = ""
    expires_at: Optional[datetime] = None
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# Sensor function type
SensorFn = Callable[[], Dict[str, Any]]


class MuraqabahEngine:
    """
    24/7 Continuous Monitoring Engine.

    Muraqabah (مراقبة) is the Islamic spiritual practice of constant
    self-awareness before Allah. Applied here as continuous system
    and opportunity awareness.

    Features:
    - Multi-domain sensor monitoring
    - Constitutional filtering of actions
    - Opportunity detection and scoring
    - Event-driven alerting
    """

    # Default scan intervals per domain (seconds)
    DEFAULT_INTERVALS = {
        MonitorDomain.FINANCIAL: 300,  # 5 minutes
        MonitorDomain.HEALTH: 60,  # 1 minute
        MonitorDomain.SOCIAL: 600,  # 10 minutes
        MonitorDomain.COGNITIVE: 300,  # 5 minutes
        MonitorDomain.ENVIRONMENTAL: 120,  # 2 minutes
    }

    def __init__(
        self,
        ihsan_threshold: float = 0.95,
        event_bus: Optional[EventBus] = None,
    ):
        self.ihsan_threshold = ihsan_threshold
        self.event_bus = event_bus or get_event_bus()

        # Sensors by domain
        self._sensors: Dict[MonitorDomain, Dict[str, SensorFn]] = {
            domain: {} for domain in MonitorDomain
        }
        self._sensor_states: Dict[str, SensorState] = {}
        self._intervals: Dict[MonitorDomain, float] = self.DEFAULT_INTERVALS.copy()

        # PERF FIX: Use deque with maxlen for O(1) bounded storage
        # instead of list trimming which is O(n)
        self._max_readings = 1000
        self._readings: Dict[MonitorDomain, Deque[SensorReading]] = {
            domain: deque(maxlen=self._max_readings) for domain in MonitorDomain
        }

        # Opportunities - use deque for bounded storage
        self._max_opportunities = 100
        self._opportunities: Deque[Opportunity] = deque(maxlen=self._max_opportunities)
        self._opportunity_handlers: List[Callable[[Opportunity], None]] = []

        # State
        self._running = False
        self._scan_count = 0
        self._opportunity_count = 0

        # Register default sensors
        self._register_default_sensors()

    def _register_default_sensors(self) -> None:
        """Register default domain-specific sensors for all five monitoring domains."""

        # =========================================================================
        # ENVIRONMENTAL Domain Sensors - Energy, maintenance, security
        # =========================================================================

        def system_health_sensor() -> Dict[str, Any]:
            """Monitor system resource health (CPU, memory, disk)."""
            try:
                import psutil

                return {
                    "cpu_usage": psutil.cpu_percent() / 100,
                    "memory_usage": psutil.virtual_memory().percent / 100,
                    "disk_usage": psutil.disk_usage("/").percent / 100,
                }
            except ImportError:
                return {"cpu_usage": 0.5, "memory_usage": 0.5, "disk_usage": 0.5}

        def process_sensor() -> Dict[str, Any]:
            """Monitor current process resource usage."""
            try:
                import psutil

                proc = psutil.Process()
                return {
                    "process_memory_mb": proc.memory_info().rss / (1024 * 1024),
                    "process_cpu": proc.cpu_percent() / 100,
                    "threads": proc.num_threads(),
                }
            except ImportError:
                return {"process_memory_mb": 100, "process_cpu": 0.1, "threads": 4}

        def network_io_sensor() -> Dict[str, Any]:
            """Monitor network I/O statistics."""
            try:
                import psutil

                net = psutil.net_io_counters()
                return {
                    "bytes_sent_mb": net.bytes_sent / (1024 * 1024),
                    "bytes_recv_mb": net.bytes_recv / (1024 * 1024),
                    "packets_sent": net.packets_sent,
                    "packets_recv": net.packets_recv,
                }
            except ImportError:
                return {
                    "bytes_sent_mb": 0,
                    "bytes_recv_mb": 0,
                    "packets_sent": 0,
                    "packets_recv": 0,
                }

        def disk_io_sensor() -> Dict[str, Any]:
            """Monitor disk I/O statistics."""
            try:
                import psutil

                disk = psutil.disk_io_counters()
                return {
                    "read_mb": disk.read_bytes / (1024 * 1024) if disk else 0,
                    "write_mb": disk.write_bytes / (1024 * 1024) if disk else 0,
                    "read_count": disk.read_count if disk else 0,
                    "write_count": disk.write_count if disk else 0,
                }
            except (ImportError, AttributeError):
                return {"read_mb": 0, "write_mb": 0, "read_count": 0, "write_count": 0}

        # =========================================================================
        # HEALTH Domain Sensors - System health, latency, error rates
        # =========================================================================

        def latency_sensor() -> Dict[str, Any]:
            """Monitor system response latency."""
            import time

            # Measure simple operation latency
            start = time.perf_counter()
            _ = [i**2 for i in range(100)]
            end = time.perf_counter()
            measured_latency = (end - start) * 1000  # ms

            return {
                "operation_latency_ms": measured_latency,
                "baseline_latency_ms": 0.1,  # Expected baseline
            }

        def error_rate_sensor() -> Dict[str, Any]:
            """Monitor error rates from logging."""
            # Would integrate with actual error tracking
            return {
                "error_rate": 0.0,
                "warning_rate": 0.0,
                "critical_count": 0,
            }

        def uptime_sensor() -> Dict[str, Any]:
            """Monitor system uptime."""
            try:
                import psutil

                boot_time = psutil.boot_time()
                import time

                uptime_hours = (time.time() - boot_time) / 3600
                return {
                    "uptime_hours": uptime_hours,
                    "health_score": min(1.0, uptime_hours / 24),  # Max 1.0 after 24h
                }
            except ImportError:
                return {"uptime_hours": 0, "health_score": 1.0}

        # =========================================================================
        # COGNITIVE Domain Sensors - Learning, task performance, creativity
        # =========================================================================

        def task_queue_sensor() -> Dict[str, Any]:
            """Monitor task queue depth and throughput."""
            # Placeholder - would integrate with actual task system
            return {
                "pending_tasks": 0,
                "active_tasks": 0,
                "failed_tasks": 0,
                "tasks_per_hour": 0,
            }

        def learning_progress_sensor() -> Dict[str, Any]:
            """Monitor learning and adaptation metrics."""
            # Would track actual learning metrics
            return {
                "pattern_discoveries": 0,
                "knowledge_updates": 0,
                "adaptation_score": 0.9,
            }

        def inference_quality_sensor() -> Dict[str, Any]:
            """Monitor inference/reasoning quality metrics."""
            return {
                "ihsan_average": 0.95,
                "snr_average": 0.90,
                "reasoning_depth": 3,
            }

        # =========================================================================
        # FINANCIAL Domain Sensors - Resource costs, efficiency
        # =========================================================================

        def compute_cost_sensor() -> Dict[str, Any]:
            """Monitor computational resource costs."""
            try:
                import psutil

                # Estimate cost based on resource usage
                cpu = psutil.cpu_percent() / 100
                mem = psutil.virtual_memory().percent / 100
                # Simplified cost model
                hourly_cost_estimate = cpu * 0.1 + mem * 0.05  # Arbitrary units
                return {
                    "hourly_cost_estimate": hourly_cost_estimate,
                    "cpu_cost_factor": cpu * 0.1,
                    "memory_cost_factor": mem * 0.05,
                }
            except ImportError:
                return {
                    "hourly_cost_estimate": 0.05,
                    "cpu_cost_factor": 0.03,
                    "memory_cost_factor": 0.02,
                }

        def efficiency_sensor() -> Dict[str, Any]:
            """Monitor system efficiency metrics."""
            return {
                "throughput_efficiency": 0.85,
                "resource_efficiency": 0.90,
                "cost_per_operation": 0.001,
            }

        def budget_sensor() -> Dict[str, Any]:
            """Monitor budget utilization (placeholder)."""
            return {
                "budget_used_percent": 0.0,
                "remaining_budget": 1.0,
                "projected_overage": 0.0,
            }

        # =========================================================================
        # SOCIAL Domain Sensors - Connectivity, collaboration
        # =========================================================================

        def connectivity_sensor() -> Dict[str, Any]:
            """Monitor network connectivity status."""
            import socket

            connected = False
            latency = 0.0
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=3)
                connected = True
                import time

                start = time.perf_counter()
                socket.create_connection(("8.8.8.8", 53), timeout=3).close()
                latency = (time.perf_counter() - start) * 1000
            except (socket.error, OSError):
                pass

            return {
                "internet_connected": 1.0 if connected else 0.0,
                "network_latency_ms": latency,
            }

        def federation_sensor() -> Dict[str, Any]:
            """Monitor federation/peer connectivity."""
            # Would track actual peer connections
            return {
                "connected_peers": 0,
                "message_rate": 0,
                "consensus_health": 1.0,
            }

        def collaboration_sensor() -> Dict[str, Any]:
            """Monitor collaboration metrics."""
            return {
                "active_collaborations": 0,
                "team_synergy_score": 0.85,
                "communication_volume": 0,
            }

        # =========================================================================
        # Register all sensors
        # =========================================================================

        sensor_registrations = [
            # Environmental (5 sensors)
            (MonitorDomain.ENVIRONMENTAL, "system_health", system_health_sensor),
            (MonitorDomain.ENVIRONMENTAL, "process", process_sensor),
            (MonitorDomain.ENVIRONMENTAL, "network_io", network_io_sensor),
            (MonitorDomain.ENVIRONMENTAL, "disk_io", disk_io_sensor),
            # Health (3 sensors)
            (MonitorDomain.HEALTH, "latency", latency_sensor),
            (MonitorDomain.HEALTH, "error_rate", error_rate_sensor),
            (MonitorDomain.HEALTH, "uptime", uptime_sensor),
            # Cognitive (3 sensors)
            (MonitorDomain.COGNITIVE, "task_queue", task_queue_sensor),
            (MonitorDomain.COGNITIVE, "learning", learning_progress_sensor),
            (MonitorDomain.COGNITIVE, "inference_quality", inference_quality_sensor),
            # Financial (3 sensors)
            (MonitorDomain.FINANCIAL, "compute_cost", compute_cost_sensor),
            (MonitorDomain.FINANCIAL, "efficiency", efficiency_sensor),
            (MonitorDomain.FINANCIAL, "budget", budget_sensor),
            # Social (3 sensors)
            (MonitorDomain.SOCIAL, "connectivity", connectivity_sensor),
            (MonitorDomain.SOCIAL, "federation", federation_sensor),
            (MonitorDomain.SOCIAL, "collaboration", collaboration_sensor),
        ]

        for domain, name, sensor_fn in sensor_registrations:
            try:
                self.register_sensor(domain, name, sensor_fn)
            except Exception as e:
                logger.warning(f"Could not register sensor {domain.value}:{name}: {e}")

        logger.info(
            f"Registered {len(sensor_registrations)} default sensors across {len(MonitorDomain)} domains"
        )

    def register_sensor(
        self,
        domain: MonitorDomain,
        name: str,
        sensor_fn: SensorFn,
    ) -> None:
        """Register a sensor for a domain."""
        sensor_id = f"{domain.value}:{name}"
        self._sensors[domain][name] = sensor_fn
        self._sensor_states[sensor_id] = SensorState.ACTIVE
        logger.debug(f"Registered sensor: {sensor_id}")

    def unregister_sensor(self, domain: MonitorDomain, name: str) -> None:
        """Unregister a sensor."""
        if name in self._sensors[domain]:
            del self._sensors[domain][name]
            sensor_id = f"{domain.value}:{name}"
            if sensor_id in self._sensor_states:
                del self._sensor_states[sensor_id]

    def set_interval(self, domain: MonitorDomain, seconds: float) -> None:
        """Set scan interval for a domain."""
        self._intervals[domain] = max(10, seconds)  # Minimum 10 seconds

    async def _scan_domain(self, domain: MonitorDomain) -> List[SensorReading]:
        """Scan all sensors in a domain."""
        readings = []
        loop = asyncio.get_event_loop()

        for name, sensor_fn in self._sensors[domain].items():
            sensor_id = f"{domain.value}:{name}"

            if self._sensor_states.get(sensor_id) != SensorState.ACTIVE:
                continue

            try:
                # PERF FIX #3: Run blocking sensors in thread pool executor
                # This prevents psutil and other blocking calls from blocking the event loop
                result = await loop.run_in_executor(None, sensor_fn)

                for metric_name, value in result.items():
                    if isinstance(value, (int, float)):
                        reading = SensorReading(
                            sensor_id=sensor_id,
                            domain=domain,
                            metric_name=metric_name,
                            value=float(value),
                        )
                        readings.append(reading)
                        # PERF FIX: deque with maxlen auto-discards oldest (O(1))
                        self._readings[domain].append(reading)

            except Exception as e:
                logger.error(f"Sensor {sensor_id} error: {e}")
                self._sensor_states[sensor_id] = SensorState.ERROR

        return readings

    async def _analyze_readings(
        self,
        readings: List[SensorReading],
    ) -> List[Opportunity]:
        """Analyze readings to detect domain-specific opportunities."""
        opportunities = []

        for reading in readings:
            opp = self._analyze_single_reading(reading)
            if opp:
                opportunities.append(opp)

        return opportunities

    def _analyze_single_reading(self, reading: SensorReading) -> Optional[Opportunity]:
        """Analyze a single reading for opportunity detection."""
        metric = reading.metric_name
        value = reading.value
        domain = reading.domain

        # =========================================================================
        # ENVIRONMENTAL Domain Opportunities
        # =========================================================================

        if domain == MonitorDomain.ENVIRONMENTAL:
            # High CPU usage
            if metric == "cpu_usage" and value > 0.8:
                return Opportunity(
                    domain=domain,
                    description="High CPU usage detected - consider scaling or optimization",
                    estimated_value=0.6,
                    urgency=0.7,
                    confidence=reading.confidence,
                    action_required="investigate_cpu_usage",
                    metadata={"metric": metric, "value": value},
                )

            # High memory usage
            if metric == "memory_usage" and value > 0.85:
                return Opportunity(
                    domain=domain,
                    description="High memory usage - consider garbage collection or restart",
                    estimated_value=0.7,
                    urgency=0.8,
                    confidence=reading.confidence,
                    action_required="optimize_memory",
                    metadata={"metric": metric, "value": value},
                )

            # High disk usage
            if metric == "disk_usage" and value > 0.9:
                return Opportunity(
                    domain=domain,
                    description="Disk space critically low - clean up or expand storage",
                    estimated_value=0.8,
                    urgency=0.9,
                    confidence=reading.confidence,
                    action_required="cleanup_disk",
                    metadata={"metric": metric, "value": value},
                )

            # Low CPU utilization (opportunity)
            if metric == "cpu_usage" and value < 0.2:
                return Opportunity(
                    domain=domain,
                    description="Low CPU utilization - can accept more workload",
                    estimated_value=0.4,
                    urgency=0.2,
                    confidence=reading.confidence,
                    action_required="increase_throughput",
                    metadata={"metric": metric, "value": value},
                )

        # =========================================================================
        # HEALTH Domain Opportunities
        # =========================================================================

        elif domain == MonitorDomain.HEALTH:
            # High error rate
            if metric == "error_rate" and value > 0.05:
                return Opportunity(
                    domain=domain,
                    description=f"Error rate elevated ({value:.1%}) - investigate root cause",
                    estimated_value=0.8,
                    urgency=0.85,
                    confidence=reading.confidence,
                    action_required="investigate_errors",
                    metadata={"metric": metric, "value": value},
                )

            # High latency
            if metric == "operation_latency_ms" and value > 100:
                return Opportunity(
                    domain=domain,
                    description=f"High latency detected ({value:.1f}ms) - optimize performance",
                    estimated_value=0.5,
                    urgency=0.6,
                    confidence=reading.confidence,
                    action_required="optimize_latency",
                    metadata={"metric": metric, "value": value},
                )

            # Low health score
            if metric == "health_score" and value < 0.7:
                return Opportunity(
                    domain=domain,
                    description="System health degraded - perform diagnostics",
                    estimated_value=0.7,
                    urgency=0.75,
                    confidence=reading.confidence,
                    action_required="run_diagnostics",
                    metadata={"metric": metric, "value": value},
                )

        # =========================================================================
        # COGNITIVE Domain Opportunities
        # =========================================================================

        elif domain == MonitorDomain.COGNITIVE:
            # Failed tasks accumulating
            if metric == "failed_tasks" and value > 5:
                return Opportunity(
                    domain=domain,
                    description=f"{int(value)} failed tasks accumulated - review and retry",
                    estimated_value=0.6,
                    urgency=0.7,
                    confidence=reading.confidence,
                    action_required="review_failed_tasks",
                    metadata={"metric": metric, "value": value},
                )

            # Low adaptation score
            if metric == "adaptation_score" and value < 0.7:
                return Opportunity(
                    domain=domain,
                    description="Adaptation score low - trigger learning cycle",
                    estimated_value=0.5,
                    urgency=0.4,
                    confidence=reading.confidence,
                    action_required="trigger_learning",
                    metadata={"metric": metric, "value": value},
                )

            # Low Ihsan average
            if metric == "ihsan_average" and value < 0.9:
                return Opportunity(
                    domain=domain,
                    description=f"Ihsan score below target ({value:.2f}) - improve quality",
                    estimated_value=0.7,
                    urgency=0.6,
                    confidence=reading.confidence,
                    action_required="improve_quality",
                    metadata={"metric": metric, "value": value},
                )

        # =========================================================================
        # FINANCIAL Domain Opportunities
        # =========================================================================

        elif domain == MonitorDomain.FINANCIAL:
            # High cost estimate
            if metric == "hourly_cost_estimate" and value > 0.15:
                return Opportunity(
                    domain=domain,
                    description="Resource costs elevated - consider optimization",
                    estimated_value=0.6,
                    urgency=0.5,
                    confidence=reading.confidence,
                    action_required="optimize_costs",
                    metadata={"metric": metric, "value": value},
                )

            # Low efficiency
            if metric == "throughput_efficiency" and value < 0.7:
                return Opportunity(
                    domain=domain,
                    description=f"Throughput efficiency low ({value:.1%}) - improve pipeline",
                    estimated_value=0.5,
                    urgency=0.4,
                    confidence=reading.confidence,
                    action_required="improve_efficiency",
                    metadata={"metric": metric, "value": value},
                )

            # Budget warning
            if metric == "budget_used_percent" and value > 0.8:
                return Opportunity(
                    domain=domain,
                    description=f"Budget {value:.0%} consumed - review spending",
                    estimated_value=0.7,
                    urgency=0.8,
                    confidence=reading.confidence,
                    action_required="review_budget",
                    metadata={"metric": metric, "value": value},
                )

        # =========================================================================
        # SOCIAL Domain Opportunities
        # =========================================================================

        elif domain == MonitorDomain.SOCIAL:
            # Network disconnected
            if metric == "internet_connected" and value < 0.5:
                return Opportunity(
                    domain=domain,
                    description="Network connectivity lost - check connection",
                    estimated_value=0.9,
                    urgency=0.95,
                    confidence=reading.confidence,
                    action_required="restore_connectivity",
                    metadata={"metric": metric, "value": value},
                )

            # Low peer connectivity
            if metric == "connected_peers" and value < 1:
                return Opportunity(
                    domain=domain,
                    description="No connected peers - isolated from federation",
                    estimated_value=0.6,
                    urgency=0.5,
                    confidence=reading.confidence,
                    action_required="reconnect_peers",
                    metadata={"metric": metric, "value": value},
                )

            # Low team synergy
            if metric == "team_synergy_score" and value < 0.6:
                return Opportunity(
                    domain=domain,
                    description="Team synergy below optimal - improve coordination",
                    estimated_value=0.4,
                    urgency=0.3,
                    confidence=reading.confidence,
                    action_required="improve_coordination",
                    metadata={"metric": metric, "value": value},
                )

        return None

    async def _constitutional_filter(self, opportunity: Opportunity) -> bool:
        """Filter opportunity against constitutional constraints (Ihsan)."""
        # Basic filtering - extend with actual constitutional checks
        if opportunity.confidence < self.ihsan_threshold - 0.1:
            return False

        # High urgency with low confidence is risky
        if opportunity.urgency > 0.8 and opportunity.confidence < 0.8:
            return False

        return True

    async def scan(self, domain: Optional[MonitorDomain] = None) -> Dict[str, Any]:
        """Perform a scan of one or all domains."""
        self._scan_count += 1
        domains = [domain] if domain else list(MonitorDomain)
        all_readings = []
        all_opportunities = []

        for d in domains:
            readings = await self._scan_domain(d)
            all_readings.extend(readings)

            opportunities = await self._analyze_readings(readings)
            for opp in opportunities:
                if await self._constitutional_filter(opp):
                    all_opportunities.append(opp)
                    self._opportunities.append(opp)
                    self._opportunity_count += 1

                    # Emit event
                    await self.event_bus.emit(
                        topic=f"muraqabah.opportunity.{d.value}",
                        payload={
                            "id": opp.id,
                            "domain": opp.domain.value,
                            "description": opp.description,
                            "urgency": opp.urgency,
                            "value": opp.estimated_value,
                        },
                        priority=(
                            EventPriority.HIGH
                            if opp.urgency > 0.7
                            else EventPriority.NORMAL
                        ),
                    )

                    # Call handlers
                    for handler in self._opportunity_handlers:
                        try:
                            handler(opp)
                        except Exception as e:
                            logger.error(f"Opportunity handler error: {e}")

        # PERF FIX: deque with maxlen auto-trims (removed manual trimming)

        return {
            "scan_number": self._scan_count,
            "domains_scanned": len(domains),
            "readings": len(all_readings),
            "opportunities": len(all_opportunities),
        }

    async def start_monitoring(self) -> None:
        """Start continuous 24/7 monitoring."""
        self._running = True
        logger.info("Muraqabah engine started - continuous vigilance active")

        # Create scan tasks for each domain with different intervals
        async def domain_scanner(domain: MonitorDomain):
            while self._running:
                await self.scan(domain)
                await asyncio.sleep(self._intervals[domain])

        tasks = [
            asyncio.create_task(domain_scanner(domain)) for domain in MonitorDomain
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._running = False
        logger.info("Muraqabah engine stopped")

    def add_opportunity_handler(self, handler: Callable[[Opportunity], None]) -> None:
        """Add a handler for detected opportunities."""
        self._opportunity_handlers.append(handler)

    def get_recent_opportunities(self, limit: int = 10) -> List[Opportunity]:
        """Get recent opportunities."""
        # Convert deque slice to list for external use
        return list(self._opportunities)[-limit:]

    def stats(self) -> Dict[str, Any]:
        """Get Muraqabah statistics."""
        return {
            "running": self._running,
            "scan_count": self._scan_count,
            "total_opportunities": self._opportunity_count,
            "active_sensors": sum(
                1 for s in self._sensor_states.values() if s == SensorState.ACTIVE
            ),
            "sensors_by_domain": {
                d.value: len(self._sensors[d]) for d in MonitorDomain
            },
            "readings_by_domain": {
                d.value: len(self._readings[d]) for d in MonitorDomain
            },
        }


__all__ = [
    "MonitorDomain",
    "MuraqabahEngine",
    "Opportunity",
    "SensorReading",
    "SensorState",
]
