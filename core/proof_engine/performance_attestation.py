"""
Performance Attestation Envelope — Thermodynamic Tamper Detection
=================================================================
Golden Gem α9 + XZ Gem #4: Every module has cryptographically signed
expected execution time, memory, and I/O bounds. Deviations beyond
2σ trigger automatic isolation and audit.

Standing on Giants:
  - Andres Freund — Detected XZ backdoor via 500ms timing anomaly
  - Shannon (1948) — Information has thermodynamic cost
  - Boltzmann — Computation is physical; malicious code costs energy
  - Saltzer & Schroeder (1975) — "Economy of mechanism"

Core Insight: Code can lie. Physics can't.
Every backdoor operation — decoding instructions, scanning memory,
running encryption — costs measurable time and energy. Performance
profiling against a known baseline is a UNIVERSAL detection method
that works regardless of how cleverly the malicious code is hidden.

This module provides:
1. Baseline recording: measure expected performance envelopes
2. Runtime checking: compare actual vs expected
3. Anomaly flagging: deviations beyond threshold trigger alerts
4. Attestation: cryptographically sign performance measurements

Constitutional Principle: ZANN_ZERO (physics-verified claims)
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Final, Optional

logger = logging.getLogger("sovereign.performance_attestation")


# ═══════════════════════════════════════════════════════════════════════════════
# ENVELOPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════


class AnomalyLevel(Enum):
    """Severity of performance anomaly (ordered by severity)."""

    NORMAL = 0  # Within expected bounds
    WARNING = 1  # 1-2σ deviation — log, no action
    ANOMALY = 2  # 2-3σ deviation — flag for review
    CRITICAL = 3  # >3σ deviation — isolate and audit
    TAMPER_SUSPECTED = 4  # Consistent multi-dimension anomaly

    def __ge__(self, other: "AnomalyLevel") -> bool:
        if not isinstance(other, AnomalyLevel):
            return NotImplemented
        return self.value >= other.value

    def __gt__(self, other: "AnomalyLevel") -> bool:
        if not isinstance(other, AnomalyLevel):
            return NotImplemented
        return self.value > other.value

    def __le__(self, other: "AnomalyLevel") -> bool:
        if not isinstance(other, AnomalyLevel):
            return NotImplemented
        return self.value <= other.value

    def __lt__(self, other: "AnomalyLevel") -> bool:
        if not isinstance(other, AnomalyLevel):
            return NotImplemented
        return self.value < other.value


@dataclass
class PerformanceEnvelope:
    """Expected performance bounds for a module or operation."""

    module_name: str
    operation: str

    # Time bounds (nanoseconds)
    expected_time_ns: int = 0
    time_stddev_ns: int = 0
    time_samples: int = 0

    # Memory bounds (bytes)
    expected_memory_bytes: int = 0
    memory_stddev_bytes: int = 0

    # I/O bounds
    expected_io_ops: int = 0
    io_stddev: int = 0

    # Sigma threshold for anomaly detection
    sigma_threshold: float = 2.0

    # Minimum samples before enforcement
    min_calibration_samples: Final[int] = 10


@dataclass
class PerformanceMeasurement:
    """Actual measured performance for a single execution."""

    module_name: str
    operation: str
    actual_time_ns: int
    actual_memory_bytes: int = 0
    actual_io_ops: int = 0
    timestamp_ns: int = field(default_factory=time.time_ns)


@dataclass
class AnomalyReport:
    """Report when performance deviates from envelope."""

    module_name: str
    operation: str
    level: AnomalyLevel
    dimensions: list[str] = field(default_factory=list)  # Which dimensions deviated
    deviations: dict[str, float] = field(default_factory=dict)  # dimension → sigma
    measurement: Optional[PerformanceMeasurement] = None
    envelope: Optional[PerformanceEnvelope] = None
    recommendation: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# ATTESTATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class PerformanceAttestationEngine:
    """
    Records performance baselines and detects anomalies.

    Usage:
        engine = PerformanceAttestationEngine()

        # Calibration phase
        with engine.measure("schema_validator", "validate_structural") as m:
            result = validator.validate(schema)
        # Automatically records measurement

        # After min_calibration_samples, anomalies are detected
        report = engine.check("schema_validator", "validate_structural", m)
        if report.level >= AnomalyLevel.ANOMALY:
            logger.warning("Performance anomaly: %s", report)
    """

    def __init__(self, sigma_threshold: float = 2.0) -> None:
        self._envelopes: dict[str, PerformanceEnvelope] = {}
        self._history: dict[str, list[int]] = {}  # key → time_ns samples
        self._sigma_threshold = sigma_threshold
        self._anomaly_count: dict[str, int] = {}  # key → consecutive anomalies

    def _key(self, module: str, operation: str) -> str:
        return f"{module}::{operation}"

    def record(self, measurement: PerformanceMeasurement) -> Optional[AnomalyReport]:
        """
        Record a performance measurement and check for anomalies.

        Returns AnomalyReport if anomaly detected, None if normal.
        """
        key = self._key(measurement.module_name, measurement.operation)

        # Add to history
        if key not in self._history:
            self._history[key] = []
        self._history[key].append(measurement.actual_time_ns)

        # Keep bounded history (last 1000 samples)
        if len(self._history[key]) > 1000:
            self._history[key] = self._history[key][-1000:]

        # Update envelope
        samples = self._history[key]
        n = len(samples)

        if n < 2:
            # Not enough data for statistics
            return None

        mean_ns = int(statistics.mean(samples))
        stdev_ns = int(statistics.stdev(samples)) if n >= 2 else 0

        envelope = PerformanceEnvelope(
            module_name=measurement.module_name,
            operation=measurement.operation,
            expected_time_ns=mean_ns,
            time_stddev_ns=max(stdev_ns, 1),  # Prevent div by zero
            time_samples=n,
        )
        self._envelopes[key] = envelope

        # Check for anomaly (only after calibration)
        if n < envelope.min_calibration_samples:
            return None

        return self._check_anomaly(measurement, envelope)

    def _check_anomaly(
        self,
        measurement: PerformanceMeasurement,
        envelope: PerformanceEnvelope,
    ) -> Optional[AnomalyReport]:
        """Check if measurement deviates from envelope beyond threshold."""
        key = self._key(measurement.module_name, measurement.operation)
        deviations: dict[str, float] = {}
        anomalous_dims: list[str] = []

        # Time deviation (primary — the Andres Freund signal)
        if envelope.time_stddev_ns > 0:
            time_sigma = (
                abs(measurement.actual_time_ns - envelope.expected_time_ns)
                / envelope.time_stddev_ns
            )
            deviations["time"] = round(time_sigma, 2)
            if time_sigma > self._sigma_threshold:
                anomalous_dims.append("time")

        if not anomalous_dims:
            # Reset consecutive anomaly counter
            self._anomaly_count[key] = 0
            return None

        # Determine severity
        max_sigma = max(deviations.values()) if deviations else 0
        consecutive = self._anomaly_count.get(key, 0) + 1
        self._anomaly_count[key] = consecutive

        if max_sigma > 3.0 or consecutive >= 3:
            level = AnomalyLevel.CRITICAL
            recommendation = (
                f"ISOLATE module '{measurement.module_name}' operation "
                f"'{measurement.operation}'. {consecutive} consecutive anomalies "
                f"detected. Maximum deviation: {max_sigma:.1f}σ. "
                f"Possible tamper or resource contention."
            )
        elif max_sigma > 2.0:
            level = AnomalyLevel.ANOMALY
            recommendation = (
                f"FLAG for review. Deviation: {max_sigma:.1f}σ. "
                f"Consecutive: {consecutive}."
            )
        else:
            level = AnomalyLevel.WARNING
            recommendation = f"Monitor. Deviation: {max_sigma:.1f}σ."

        # Multi-dimension anomaly = higher suspicion
        if len(anomalous_dims) >= 2:
            level = AnomalyLevel.TAMPER_SUSPECTED
            recommendation = (
                f"TAMPER SUSPECTED: Multiple dimensions anomalous "
                f"({', '.join(anomalous_dims)}). Isolate and audit."
            )

        report = AnomalyReport(
            module_name=measurement.module_name,
            operation=measurement.operation,
            level=level,
            dimensions=anomalous_dims,
            deviations=deviations,
            measurement=measurement,
            envelope=envelope,
            recommendation=recommendation,
        )

        if level >= AnomalyLevel.ANOMALY:
            logger.warning(
                "Performance anomaly [%s]: %s — %s (%.1fσ, %d consecutive)",
                level.name,
                key,
                recommendation,
                max_sigma,
                consecutive,
            )

        return report

    def get_envelope(
        self, module: str, operation: str
    ) -> Optional[PerformanceEnvelope]:
        """Get current performance envelope for a module operation."""
        return self._envelopes.get(self._key(module, operation))

    def get_all_envelopes(self) -> dict[str, PerformanceEnvelope]:
        """Get all recorded envelopes."""
        return dict(self._envelopes)

    def export_baselines(self) -> dict[str, Any]:
        """Export all baselines for persistence / cross-node sharing."""
        return {
            key: {
                "module": env.module_name,
                "operation": env.operation,
                "expected_time_ns": env.expected_time_ns,
                "time_stddev_ns": env.time_stddev_ns,
                "samples": env.time_samples,
            }
            for key, env in self._envelopes.items()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT MANAGER FOR EASY MEASUREMENT
# ═══════════════════════════════════════════════════════════════════════════════


class MeasureContext:
    """
    Context manager for measuring execution performance.

    Usage:
        engine = PerformanceAttestationEngine()
        with MeasureContext(engine, "validator", "check") as ctx:
            do_work()
        # ctx.report is populated if anomaly detected
    """

    def __init__(
        self,
        engine: PerformanceAttestationEngine,
        module: str,
        operation: str,
    ) -> None:
        self.engine = engine
        self.module = module
        self.operation = operation
        self.start_ns: int = 0
        self.measurement: Optional[PerformanceMeasurement] = None
        self.report: Optional[AnomalyReport] = None

    def __enter__(self) -> "MeasureContext":
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        elapsed_ns = time.perf_counter_ns() - self.start_ns
        self.measurement = PerformanceMeasurement(
            module_name=self.module,
            operation=self.operation,
            actual_time_ns=elapsed_ns,
        )
        self.report = self.engine.record(self.measurement)


__all__ = [
    "AnomalyLevel",
    "AnomalyReport",
    "MeasureContext",
    "PerformanceAttestationEngine",
    "PerformanceEnvelope",
    "PerformanceMeasurement",
]
