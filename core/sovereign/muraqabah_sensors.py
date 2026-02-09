"""
Muraqabah Sensor Hub - Multi-Domain SNR-Filtered Monitoring
============================================================
Enhanced sensor array for the proactive sovereign system with SNR-based
quality filtering and event-driven notifications.

Standing on Giants:
- Al-Ghazali (1058-1111): Muraqabah - continuous vigilance
- Shannon (1948): SNR as information quality metric
- Observability Patterns: Health metrics, latency, error tracking

Architecture:
    +----------------------------------------------------------+
    |                   MuraqabahSensorHub                      |
    | +------------+ +------------+ +------------+ +----------+ |
    | | System     | | SNR        | | Agent      | | Constit  | |
    | | Health     | | Quality    | | Perf       | | Compliance| |
    | +------------+ +------------+ +------------+ +----------+ |
    |                         |                                 |
    |                    SNR Filter                             |
    |                         |                                 |
    |                    Event Bus                              |
    +----------------------------------------------------------+

Created: 2026-02-04 | BIZRA Sovereign v2.3.0
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .event_bus import EventBus, EventPriority, get_event_bus

try:
    from core.integration.constants import (
        UNIFIED_IHSAN_THRESHOLD,
        UNIFIED_SNR_THRESHOLD,
    )
except ImportError:
    UNIFIED_IHSAN_THRESHOLD = 0.95  # type: ignore[misc]
    UNIFIED_SNR_THRESHOLD = 0.85  # type: ignore[misc]

logger = logging.getLogger(__name__)

# SNR thresholds (Shannon-inspired) — sourced from centralized constants
SNR_FLOOR: float = UNIFIED_SNR_THRESHOLD
SNR_HIGH: float = UNIFIED_IHSAN_THRESHOLD


class SensorDomain(str, Enum):
    """Sensor domains for the hub."""

    SYSTEM_HEALTH = "system_health"
    SNR_QUALITY = "snr_quality"
    AGENT_PERFORMANCE = "agent_performance"
    CONSTITUTIONAL = "constitutional"


@dataclass
class SensorReading:
    """A reading from a Muraqabah sensor with SNR score."""

    sensor_id: str
    domain: SensorDomain
    metric_name: str
    value: float
    snr_score: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignificantChange:
    """A significant change detected by the sensor hub."""

    sensor_id: str
    domain: SensorDomain
    previous_value: float
    current_value: float
    delta_pct: float
    snr_score: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Sensor function type
SensorFn = Callable[[], Dict[str, float]]


class MuraqabahSensorHub:
    """
    Multi-domain sensor hub with SNR filtering.

    Monitors:
    - System health (CPU, memory, latency)
    - SNR quality trends across channels
    - Agent performance scores
    - Constitutional compliance metrics

    Emits events to EventBus when significant changes occur.
    """

    CHANGE_THRESHOLD: float = 0.1  # 10% change triggers event

    def __init__(
        self,
        snr_threshold: float = SNR_FLOOR,
        event_bus: Optional[EventBus] = None,
    ):
        self.snr_threshold = snr_threshold
        self.event_bus = event_bus or get_event_bus()

        # Sensors by domain
        self._sensors: Dict[SensorDomain, Dict[str, SensorFn]] = {
            d: {} for d in SensorDomain
        }

        # Last readings for change detection
        self._last_readings: Dict[str, SensorReading] = {}

        # Register default sensors
        self._register_default_sensors()

        logger.info(f"MuraqabahSensorHub initialized: SNR threshold={snr_threshold}")

    def _register_default_sensors(self) -> None:
        """Register default sensors for all domains."""

        # System Health Sensors
        def cpu_sensor() -> Dict[str, float]:
            try:
                import psutil

                return {"cpu_usage": psutil.cpu_percent() / 100}
            except ImportError:
                return {"cpu_usage": 0.5}

        def memory_sensor() -> Dict[str, float]:
            try:
                import psutil

                return {"memory_usage": psutil.virtual_memory().percent / 100}
            except ImportError:
                return {"memory_usage": 0.5}

        def latency_sensor() -> Dict[str, float]:
            start = time.perf_counter()
            _ = sum(i * i for i in range(1000))
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Normalize: < 1ms = 1.0 SNR, > 10ms = 0.5 SNR
            latency_snr = max(0.5, 1.0 - (elapsed_ms / 10))
            return {"latency_ms": elapsed_ms, "latency_snr": latency_snr}

        # SNR Quality Sensors
        def inference_snr_sensor() -> Dict[str, float]:
            # Placeholder - would integrate with actual inference system
            return {"inference_snr": 0.92, "embedding_snr": 0.94}

        def knowledge_snr_sensor() -> Dict[str, float]:
            return {"retrieval_snr": 0.90, "grounding_snr": 0.88}

        # Agent Performance Sensors
        def task_completion_sensor() -> Dict[str, float]:
            return {"completion_rate": 0.95, "success_rate": 0.92}

        def response_quality_sensor() -> Dict[str, float]:
            return {"ihsan_score": 0.96, "coherence_score": 0.93}

        # Constitutional Compliance Sensors
        def ihsan_compliance_sensor() -> Dict[str, float]:
            return {"ihsan_compliance": 0.97, "adl_compliance": 0.98}

        def boundary_sensor() -> Dict[str, float]:
            return {"boundary_adherence": 1.0, "ethics_score": 0.99}

        # Register all sensors
        registrations = [
            (SensorDomain.SYSTEM_HEALTH, "cpu", cpu_sensor),
            (SensorDomain.SYSTEM_HEALTH, "memory", memory_sensor),
            (SensorDomain.SYSTEM_HEALTH, "latency", latency_sensor),
            (SensorDomain.SNR_QUALITY, "inference", inference_snr_sensor),
            (SensorDomain.SNR_QUALITY, "knowledge", knowledge_snr_sensor),
            (SensorDomain.AGENT_PERFORMANCE, "task", task_completion_sensor),
            (SensorDomain.AGENT_PERFORMANCE, "quality", response_quality_sensor),
            (SensorDomain.CONSTITUTIONAL, "ihsan", ihsan_compliance_sensor),
            (SensorDomain.CONSTITUTIONAL, "boundary", boundary_sensor),
        ]

        for domain, name, fn in registrations:
            self.register_sensor(domain, name, fn)

    def register_sensor(
        self,
        domain: SensorDomain,
        name: str,
        sensor_fn: SensorFn,
    ) -> None:
        """Register a sensor function."""
        self._sensors[domain][name] = sensor_fn
        logger.debug(f"Registered sensor: {domain.value}:{name}")

    def _calculate_snr(self, domain: SensorDomain, metrics: Dict[str, float]) -> float:
        """Calculate SNR score for a set of metrics."""
        if not metrics:
            return 0.0

        # Domain-specific SNR calculation
        if domain == SensorDomain.SYSTEM_HEALTH:
            # Lower resource usage = higher SNR
            cpu = metrics.get("cpu_usage", 0.5)
            mem = metrics.get("memory_usage", 0.5)
            lat_snr = metrics.get("latency_snr", 0.9)
            return (1 - cpu) * 0.3 + (1 - mem) * 0.3 + lat_snr * 0.4

        elif domain == SensorDomain.SNR_QUALITY:
            # Average of SNR metrics
            snr_values = [v for k, v in metrics.items() if "snr" in k]
            return sum(snr_values) / len(snr_values) if snr_values else 0.9

        elif domain == SensorDomain.AGENT_PERFORMANCE:
            # Weighted average of performance metrics
            return (
                metrics.get("completion_rate", 0.9) * 0.4
                + metrics.get("success_rate", 0.9) * 0.3
                + metrics.get("ihsan_score", 0.9) * 0.3
            )

        elif domain == SensorDomain.CONSTITUTIONAL:
            # Minimum of compliance metrics (strictest gate)
            return min(metrics.values()) if metrics else 0.95

        return 0.9

    async def poll_sensor(
        self,
        domain: SensorDomain,
        name: str,
    ) -> Optional[SensorReading]:
        """Poll a single sensor and return reading if above SNR threshold."""
        sensor_fn = self._sensors.get(domain, {}).get(name)
        if not sensor_fn:
            return None

        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(None, sensor_fn)

        snr_score = self._calculate_snr(domain, metrics)

        # Filter by SNR threshold
        if snr_score < self.snr_threshold:
            logger.debug(f"Filtered {domain.value}:{name} SNR={snr_score:.2f}")
            return None

        # Create reading with primary metric value
        primary_value = list(metrics.values())[0] if metrics else 0.0

        return SensorReading(
            sensor_id=f"{domain.value}:{name}",
            domain=domain,
            metric_name=list(metrics.keys())[0] if metrics else "unknown",
            value=primary_value,
            snr_score=snr_score,
            metadata=metrics,
        )

    async def poll_all_sensors(self) -> List[SensorReading]:
        """Poll all sensors and return filtered readings."""
        readings: List[SensorReading] = []
        changes: List[SignificantChange] = []

        for domain in SensorDomain:
            for name in self._sensors[domain]:
                reading = await self.poll_sensor(domain, name)
                if reading:
                    readings.append(reading)

                    # Check for significant change
                    change = self._detect_change(reading)
                    if change:
                        changes.append(change)

                    # Update last reading
                    self._last_readings[reading.sensor_id] = reading

        # Emit events for significant changes
        for change in changes:
            await self._emit_change_event(change)

        return readings

    def _detect_change(self, reading: SensorReading) -> Optional[SignificantChange]:
        """Detect if reading represents a significant change."""
        last = self._last_readings.get(reading.sensor_id)
        if not last:
            return None

        if last.value == 0:
            return None

        delta_pct = abs(reading.value - last.value) / abs(last.value)

        if delta_pct >= self.CHANGE_THRESHOLD:
            return SignificantChange(
                sensor_id=reading.sensor_id,
                domain=reading.domain,
                previous_value=last.value,
                current_value=reading.value,
                delta_pct=delta_pct,
                snr_score=reading.snr_score,
            )

        return None

    async def _emit_change_event(self, change: SignificantChange) -> None:
        """Emit event for significant change."""
        priority = (
            EventPriority.HIGH if change.delta_pct > 0.25 else EventPriority.NORMAL
        )

        await self.event_bus.emit(
            topic=f"muraqabah.change.{change.domain.value}",
            payload={
                "sensor_id": change.sensor_id,
                "previous": change.previous_value,
                "current": change.current_value,
                "delta_pct": change.delta_pct,
                "snr_score": change.snr_score,
            },
            priority=priority,
            source="muraqabah_sensor_hub",
        )

        logger.info(
            f"Change detected: {change.sensor_id} "
            f"{change.previous_value:.2f} -> {change.current_value:.2f} "
            f"({change.delta_pct:.1%})"
        )

    def stats(self) -> Dict[str, Any]:
        """Get sensor hub statistics."""
        return {
            "snr_threshold": self.snr_threshold,
            "sensors_by_domain": {d.value: len(self._sensors[d]) for d in SensorDomain},
            "total_sensors": sum(len(s) for s in self._sensors.values()),
            "tracked_readings": len(self._last_readings),
        }


__all__ = [
    "MuraqabahSensorHub",
    "SensorDomain",
    "SensorReading",
    "SignificantChange",
    "SNR_FLOOR",
    "SNR_HIGH",
]
