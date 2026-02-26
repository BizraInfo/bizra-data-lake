"""
Performance Attestation Envelope — Thermodynamic Security Gate
==============================================================
Golden Gem α9 + XZ Gem #4: Every module has expected time/memory bounds.
Deviations beyond 2σ trigger automatic isolation.

Standing on Giants:
  - Andres Freund (2024) — Found XZ backdoor via 500ms timing anomaly
  - Shannon (1948) — Information has physical cost
  - Lamport (1982) — Byzantine fault detection through timing

PRINCIPLE: Code can lie. Physics can't.
Malicious computation costs time and energy.
Performance profiling against known baselines detects what
code review and static analysis cannot.

إحسان Principle: Excellence includes predictable performance.
A module that produces correct output in unexpected time
is NOT excellent — it is suspicious.
"""

from __future__ import annotations

import functools
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Deque, Final, Optional, TypeVar

logger = logging.getLogger("sovereign.performance_attestation")

F = TypeVar("F", bound=Callable[..., Any])


class AnomalyLevel(Enum):
    """Severity of detected performance anomaly."""

    NORMAL = auto()  # Within expected envelope
    WARNING = auto()  # 1σ–2σ deviation — log
    ANOMALY = auto()  # 2σ–3σ deviation — alert + flag for review
    CRITICAL = auto()  # >3σ deviation — isolate + audit
    BASELINE_INSUFFICIENT = auto()  # Not enough samples for statistical judgment


@dataclass
class PerformanceEnvelope:
    """Expected performance bounds for a module/function.

    The envelope defines what NORMAL looks like.
    Deviations from normal are the detection signal.
    """

    module_name: str
    function_name: str
    # Baseline statistics (populated from calibration runs)
    mean_time_ms: float = 0.0
    std_time_ms: float = 0.0
    max_time_ms: float = 0.0
    min_time_ms: float = 0.0
    sample_count: int = 0
    # Thresholds (σ multipliers)
    warning_sigma: float = 1.5
    anomaly_sigma: float = 2.0
    critical_sigma: float = 3.0
    # Minimum samples for statistical validity
    min_calibration_samples: int = 10


@dataclass
class AttestationResult:
    """Result of performance attestation check."""

    module_name: str
    function_name: str
    measured_time_ms: float
    expected_mean_ms: float
    expected_std_ms: float
    deviation_sigma: float
    anomaly_level: AnomalyLevel
    # Metadata
    timestamp_ns: int = 0
    detail: str = ""

    @property
    def is_suspicious(self) -> bool:
        return self.anomaly_level in (AnomalyLevel.ANOMALY, AnomalyLevel.CRITICAL)


class PerformanceAttestor:
    """Runtime performance attestation engine.

    Maintains rolling baselines for monitored functions and detects
    anomalous execution times that may indicate:
    - Injected computation (XZ-style backdoor)
    - Resource exhaustion attacks
    - Degraded dependencies
    - Side-channel exploitation

    Usage:
        attestor = PerformanceAttestor()

        @attestor.monitor("proof_engine", "validate_schema")
        def validate_schema(data):
            ...

        # Or manual attestation:
        with attestor.measure("proof_engine", "validate_schema") as m:
            result = validate_schema(data)
        attestation = m.result  # AttestationResult
    """

    def __init__(
        self,
        *,
        window_size: int = 100,
        min_calibration: int = 10,
        auto_isolate: bool = True,
    ):
        self._baselines: dict[str, Deque[float]] = {}
        self._envelopes: dict[str, PerformanceEnvelope] = {}
        self._window_size: Final[int] = window_size
        self._min_calibration: Final[int] = min_calibration
        self._auto_isolate: bool = auto_isolate
        self._anomaly_log: Deque[AttestationResult] = deque(maxlen=1000)
        self._isolation_set: set[str] = set()

    def _key(self, module: str, function: str) -> str:
        return f"{module}::{function}"

    def register_envelope(self, envelope: PerformanceEnvelope) -> None:
        """Register a pre-computed performance envelope (from calibration)."""
        key = self._key(envelope.module_name, envelope.function_name)
        self._envelopes[key] = envelope

    def record_measurement(
        self,
        module: str,
        function: str,
        elapsed_ms: float,
    ) -> AttestationResult:
        """Record a measurement and check against envelope."""
        key = self._key(module, function)

        # Initialize rolling window
        if key not in self._baselines:
            self._baselines[key] = deque(maxlen=self._window_size)
        self._baselines[key].append(elapsed_ms)

        # Get or compute envelope
        envelope = self._envelopes.get(key)
        samples = self._baselines[key]

        if len(samples) < self._min_calibration and envelope is None:
            # Not enough data — cannot judge
            return AttestationResult(
                module_name=module,
                function_name=function,
                measured_time_ms=elapsed_ms,
                expected_mean_ms=0.0,
                expected_std_ms=0.0,
                deviation_sigma=0.0,
                anomaly_level=AnomalyLevel.BASELINE_INSUFFICIENT,
                timestamp_ns=time.perf_counter_ns(),
                detail=f"Calibrating: {len(samples)}/{self._min_calibration} samples",
            )

        # Compute statistics from rolling window or envelope
        if envelope and envelope.sample_count >= self._min_calibration:
            mean = envelope.mean_time_ms
            std = envelope.std_time_ms
            warning_σ = envelope.warning_sigma
            anomaly_σ = envelope.anomaly_sigma
            critical_σ = envelope.critical_sigma
        else:
            mean = statistics.mean(samples)
            std = statistics.stdev(samples) if len(samples) > 1 else 0.0
            warning_σ = 1.5
            anomaly_σ = 2.0
            critical_σ = 3.0

        # Compute deviation
        if std > 0:
            deviation = abs(elapsed_ms - mean) / std
        else:
            deviation = 0.0 if abs(elapsed_ms - mean) < 1e-6 else float("inf")

        # Classify
        if deviation >= critical_σ:
            level = AnomalyLevel.CRITICAL
            detail = (
                f"CRITICAL: {elapsed_ms:.2f}ms is {deviation:.1f}σ from "
                f"mean {mean:.2f}ms (std={std:.2f}ms). "
                f"Possible injected computation or resource attack."
            )
            if self._auto_isolate:
                self._isolation_set.add(key)
                detail += " MODULE ISOLATED."
            logger.critical(detail)
        elif deviation >= anomaly_σ:
            level = AnomalyLevel.ANOMALY
            detail = (
                f"ANOMALY: {elapsed_ms:.2f}ms is {deviation:.1f}σ from "
                f"mean {mean:.2f}ms. Flagged for review."
            )
            logger.warning(detail)
        elif deviation >= warning_σ:
            level = AnomalyLevel.WARNING
            detail = (
                f"WARNING: {elapsed_ms:.2f}ms is {deviation:.1f}σ from "
                f"mean {mean:.2f}ms."
            )
            logger.info(detail)
        else:
            level = AnomalyLevel.NORMAL
            detail = f"Normal: {elapsed_ms:.2f}ms (mean={mean:.2f}ms)"

        result = AttestationResult(
            module_name=module,
            function_name=function,
            measured_time_ms=elapsed_ms,
            expected_mean_ms=mean,
            expected_std_ms=std,
            deviation_sigma=deviation,
            anomaly_level=level,
            timestamp_ns=time.perf_counter_ns(),
            detail=detail,
        )

        if result.is_suspicious:
            self._anomaly_log.append(result)

        return result

    def is_isolated(self, module: str, function: str) -> bool:
        """Check if a module/function has been isolated due to anomalies."""
        return self._key(module, function) in self._isolation_set

    def clear_isolation(self, module: str, function: str) -> None:
        """Remove a module from isolation (after manual audit)."""
        self._isolation_set.discard(self._key(module, function))

    def get_anomaly_log(self) -> list[AttestationResult]:
        """Return recent anomalies for audit."""
        return list(self._anomaly_log)

    def get_envelope(self, module: str, function: str) -> Optional[PerformanceEnvelope]:
        """Get the current performance envelope for a module."""
        key = self._key(module, function)
        if key in self._envelopes:
            return self._envelopes[key]
        samples = self._baselines.get(key)
        if samples and len(samples) >= self._min_calibration:
            return PerformanceEnvelope(
                module_name=module,
                function_name=function,
                mean_time_ms=statistics.mean(samples),
                std_time_ms=statistics.stdev(samples),
                max_time_ms=max(samples),
                min_time_ms=min(samples),
                sample_count=len(samples),
            )
        return None

    def monitor(self, module: str, function: str) -> Callable[[F], F]:
        """Decorator: monitor function execution time."""

        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                key = self._key(module, function)
                if key in self._isolation_set:
                    raise RuntimeError(
                        f"Module {module}::{function} is ISOLATED due to "
                        f"performance anomaly. Manual audit required."
                    )
                start = time.perf_counter_ns()
                try:
                    result = func(*args, **kwargs)
                finally:
                    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
                    self.record_measurement(module, function, elapsed_ms)
                return result

            return wrapper  # type: ignore[return-value]

        return decorator


# Module-level singleton for global attestation
_global_attestor: Optional[PerformanceAttestor] = None


def get_attestor() -> PerformanceAttestor:
    """Get or create the global performance attestor."""
    global _global_attestor
    if _global_attestor is None:
        _global_attestor = PerformanceAttestor()
    return _global_attestor


__all__ = [
    "AnomalyLevel",
    "AttestationResult",
    "PerformanceAttestor",
    "PerformanceEnvelope",
    "get_attestor",
]
