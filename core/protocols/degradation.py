"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA DEGRADATION PROTOCOL                                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Defines the DegradationEvent contract and DegradationEmitter utility       ║
║   for Protocol-optional engines (CognitiveFusion, GoTBridge, Bicameral).     ║
║                                                                              ║
║   When an engine starts with None implementations for Protocol-typed args,   ║
║   it MUST emit a DegradationEvent so downstream consumers can distinguish    ║
║   real computation from default/empty results.                               ║
║                                                                              ║
║   Standing on Giants:                                                        ║
║   - Erlang/OTP (graceful degradation, 1986)                                  ║
║   - Liskov (substitution principle, 1987)                                    ║
║   - Netflix Hystrix (circuit breaker pattern, 2012)                          ║
║                                                                              ║
║   Constitutional: Degradation is transparent — never silent failure.         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Blueprint Reference: Section 3.2 — P1 Graceful Degradation Protocol
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DegradationSeverity(Enum):
    """Severity of a degradation event."""

    PARTIAL = "PARTIAL"  # Some Protocol args are None
    FULL = "FULL"  # ALL Protocol args are None


@dataclass(frozen=True)
class DegradationEvent:
    """Immutable record of a degradation occurrence.

    Emitted when a Protocol-optional engine initializes with missing
    implementations. Consumers MUST check event severity to determine
    if results are from real computation or defaults.

    Attributes:
        engine: Fully qualified class name (e.g. "CognitiveFusionEngine").
        missing: List of Protocol argument names that are None.
        available: List of Protocol argument names that ARE provided.
        severity: PARTIAL if some are present, FULL if all are missing.
        timestamp: UTC timestamp of the degradation detection.
    """

    engine: str
    missing: List[str]
    available: List[str]
    severity: DegradationSeverity
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def degradation_ratio(self) -> float:
        """Fraction of Protocol args that are missing (0.0 = fully healthy)."""
        total = len(self.missing) + len(self.available)
        if total == 0:
            return 0.0
        return len(self.missing) / total

    def to_dict(self) -> Dict:
        return {
            "engine": self.engine,
            "missing": self.missing,
            "available": self.available,
            "severity": self.severity.value,
            "degradation_ratio": self.degradation_ratio,
            "timestamp": self.timestamp.isoformat(),
        }


class DegradationEmitter:
    """Utility for engines to check and emit degradation events.

    Usage in a Protocol-optional engine constructor:

        class CognitiveFusionEngine:
            def __init__(self, moe_router=None, hrm_engine=None, ...):
                emitter = DegradationEmitter("CognitiveFusionEngine")
                emitter.check("moe_router", moe_router)
                emitter.check("hrm_engine", hrm_engine)
                event = emitter.emit()
                if event:
                    self._degraded = True
                    self._degradation_event = event
    """

    def __init__(self, engine_name: str):
        self._engine_name = engine_name
        self._missing: List[str] = []
        self._available: List[str] = []
        self._listeners: List = []

    def check(self, arg_name: str, arg_value: Optional[object]) -> None:
        """Register an argument as missing or available."""
        if arg_value is None:
            self._missing.append(arg_name)
        else:
            self._available.append(arg_name)

    def emit(self) -> Optional[DegradationEvent]:
        """Emit a DegradationEvent if any arguments are missing.

        Returns None if all arguments are available (healthy state).
        Logs at WARNING for FULL degradation, INFO for PARTIAL.
        """
        if not self._missing:
            return None

        severity = (
            DegradationSeverity.FULL
            if not self._available
            else DegradationSeverity.PARTIAL
        )

        event = DegradationEvent(
            engine=self._engine_name,
            missing=list(self._missing),
            available=list(self._available),
            severity=severity,
        )

        if severity == DegradationSeverity.FULL:
            logger.warning(
                "DEGRADATION-FULL: %s initialized with zero implementations "
                "(missing: %s) — all results will be defaults",
                self._engine_name,
                ", ".join(self._missing),
            )
        else:
            logger.info(
                "DEGRADATION-PARTIAL: %s missing %d/%d Protocol args: %s",
                self._engine_name,
                len(self._missing),
                len(self._missing) + len(self._available),
                ", ".join(self._missing),
            )

        return event

    @property
    def is_healthy(self) -> bool:
        """True if no missing arguments were registered."""
        return len(self._missing) == 0

    @property
    def total_checked(self) -> int:
        return len(self._missing) + len(self._available)
