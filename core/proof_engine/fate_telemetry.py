"""
FATE Telemetry — Observability for the governed proof pipeline.

Emits structured telemetry events for each FATE crossing stage:
  PAT execution → Evidence Audit → SAT Verdict → Final Result

Events are written to a JSONL telemetry log for Glass Cockpit consumption.
No external dependencies. Append-only. Fail-silent (never blocks the pipeline).

Standing on Giants:
- Dijkstra (1968): Structured observability over ad-hoc debugging
- Shannon (1948): Signal measurement at every stage
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("fate.telemetry")

TELEMETRY_PATH = Path(
    os.getenv("BIZRA_FATE_TELEMETRY", "/data/bizra/logs/fate-telemetry.jsonl")
)


@dataclass
class FateEvent:
    """A single telemetry event from the FATE pipeline."""

    timestamp: str = ""
    stage: str = ""  # pat_execution | evidence_audit | sat_verdict | fate_result
    duration_ms: float = 0.0
    verdict: str = ""
    ihsan_score: float = 0.0
    evidence_valid: bool = True
    evidence_count: int = 0
    invalid_refs: List[str] = field(default_factory=list)
    model: str = ""
    short_circuited: bool = False
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v or v == 0 or v is False}


class FateTelemetry:
    """Collects and emits telemetry for a single FATE crossing."""

    def __init__(self):
        self._events: List[FateEvent] = []
        self._start = time.perf_counter()

    def record(self, stage: str, **kwargs) -> None:
        """Record a telemetry event for a pipeline stage."""
        event = FateEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            duration_ms=(time.perf_counter() - self._start) * 1000,
            **kwargs,
        )
        self._events.append(event)

    def emit(self) -> None:
        """Write all events to the telemetry log. Fail-silent."""
        try:
            TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(TELEMETRY_PATH, "a") as f:
                for event in self._events:
                    f.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        except OSError as e:
            logger.debug("Telemetry write failed: %s", e)

    @property
    def events(self) -> List[FateEvent]:
        return list(self._events)

    def summary(self) -> Dict[str, Any]:
        """Compact summary for Glass Cockpit display."""
        if not self._events:
            return {"stages": 0}

        final = self._events[-1]
        return {
            "stages": len(self._events),
            "total_ms": round(final.duration_ms, 1),
            "verdict": final.verdict or self._events[-1].stage,
            "ihsan": final.ihsan_score,
            "evidence_valid": all(e.evidence_valid for e in self._events),
            "short_circuited": any(e.short_circuited for e in self._events),
        }
