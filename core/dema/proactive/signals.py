"""Ambient signals — local, read-only-by-default observations Dema may act on.

Phase A0.6 v0.1 ships a registry of ALLOWED signal kinds. Real file
watchers, OS hooks, browser observers etc. land in later phases under
explicit operator consent. For now the layer is fed by simulated signals
in tests and by future read-only collectors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# All valid signal kinds. Adding a new kind requires an explicit code change
# so the policy layer stays auditable. Kind names are stable identifiers.
VALID_SIGNAL_KINDS: tuple[str, ...] = (
    "downloads_folder_large",
    "stale_files_detected",
    "long_idle_session",
    "unfinished_mission",
    "resource_pressure",
    "duplicate_delete_candidate",
    "format_drive_candidate",
    "credential_exposure_candidate",
    "social_post_candidate",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_known_signal(kind: str) -> bool:
    return kind in VALID_SIGNAL_KINDS


@dataclass
class AmbientSignal:
    """A single ambient observation. Read-only at this layer."""

    kind: str
    confidence: float
    urgency: str = "low"  # low | medium | high
    evidence: dict[str, Any] = field(default_factory=dict)
    source: str = "simulated"
    timestamp: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not is_known_signal(self.kind):
            raise ValueError(
                f"unknown signal kind {self.kind!r}; allowed: {VALID_SIGNAL_KINDS}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")
        if self.urgency not in ("low", "medium", "high"):
            raise ValueError(
                f"urgency must be one of low|medium|high, got {self.urgency!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
