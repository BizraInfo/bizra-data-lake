"""HMM single-caller isolation gate for Phase 46 staging.

During staging, HMM observations are accepted from exactly ONE allowed
caller.  Other callers can read predictions but cannot mutate HMM state.

Standing on Giants: Rabiner (HMM, 1989) · Lamport (distributed state, 1978)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.integration.constants import (
    HMM_ALLOWED_CALLER_DEFAULT,
    HMM_CALLER_MODE_DEFAULT,
)

logger = logging.getLogger(__name__)


@dataclass
class HMMCallerStats:
    """Telemetry for HMM caller isolation."""

    accepted_count: int = 0
    dropped_count: int = 0
    dropped_callers: Dict[str, int] = field(default_factory=dict)
    last_accepted: Optional[datetime] = None
    last_dropped: Optional[datetime] = None


class HMMCallerGate:
    """Gates HMM observations to a single allowed caller during staging.

    Modes:
        ``"single"``: Only *allowed_caller* can observe.  Others get read-only.
        ``"multi"``:  All callers can observe (production mode).
        ``"disabled"``: No callers can observe (emergency shutoff).
    """

    def __init__(self, hmm_engine: Any) -> None:
        self._engine = hmm_engine
        self._mode = os.getenv("BIZRA_PHASE46_HMM_CALLER_MODE", HMM_CALLER_MODE_DEFAULT)
        self._allowed = os.getenv(
            "BIZRA_PHASE46_HMM_ALLOWED_CALLER", HMM_ALLOWED_CALLER_DEFAULT
        )
        self._stats = HMMCallerStats()

    # ------------------------------------------------------------------
    # Observe (gated)
    # ------------------------------------------------------------------

    def observe(self, symbol: str, caller_id: str) -> Optional[Any]:
        """Gated observation — only allowed caller mutates HMM state.

        Args:
            symbol: HMM observation symbol (e.g. ``"search"``, ``"edit"``).
            caller_id: Identity of the caller (``"mcp"``, ``"proactive"``, …).

        Returns:
            PredictionResult if accepted, ``None`` if dropped.
        """
        if self._mode == "disabled":
            self._record_drop(caller_id)
            return None

        if self._mode == "single" and caller_id != self._allowed:
            self._record_drop(caller_id)
            logger.debug(
                "HMM gate: dropped from %s (allowed: %s)", caller_id, self._allowed
            )
            return None

        # Accepted — mutate HMM state
        self._stats.accepted_count += 1
        self._stats.last_accepted = datetime.now(timezone.utc)
        return self._engine.observe(symbol)

    # ------------------------------------------------------------------
    # Predict (always allowed)
    # ------------------------------------------------------------------

    def predict(self, caller_id: str) -> Optional[Any]:
        """Read-only prediction — always allowed regardless of mode."""
        if self._engine is None:
            return None
        try:
            return self._engine.predict_next()
        except Exception as exc:
            logger.warning("HMM gate predict failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _record_drop(self, caller_id: str) -> None:
        self._stats.dropped_count += 1
        self._stats.dropped_callers[caller_id] = (
            self._stats.dropped_callers.get(caller_id, 0) + 1
        )
        self._stats.last_dropped = datetime.now(timezone.utc)

    @property
    def stats(self) -> Dict[str, Any]:
        """Telemetry snapshot for observability."""
        return {
            "mode": self._mode,
            "allowed_caller": self._allowed,
            "accepted_count": self._stats.accepted_count,
            "dropped_count": self._stats.dropped_count,
            "dropped_callers": dict(self._stats.dropped_callers),
            "last_accepted": (
                self._stats.last_accepted.isoformat()
                if self._stats.last_accepted
                else None
            ),
            "last_dropped": (
                self._stats.last_dropped.isoformat()
                if self._stats.last_dropped
                else None
            ),
        }

    @property
    def mode(self) -> str:
        return self._mode
