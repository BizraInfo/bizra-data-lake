"""Dema four-state model: Current → Ideal → Gap → Next Admissible Action.

Mirrors §9 of the Origin Manifest. Persists under
sovereign_state/dema/mission_state.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID_TRUTH_LABELS = ("MEASURED", "DERIVED", "PLANNED", "SANDBOX", "UNKNOWN")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class FourStateModel:
    current: str = ""
    ideal: str = ""
    gap: str = ""
    next_admissible_action: str = ""
    truth_label: str = "PLANNED"
    timestamp: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.truth_label not in VALID_TRUTH_LABELS:
            raise ValueError(
                f"truth_label must be one of {VALID_TRUTH_LABELS}, "
                f"got {self.truth_label!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_actionable(self) -> bool:
        """An entry is actionable when next_admissible_action is non-empty."""
        return bool(self.next_admissible_action.strip())


class MissionStateMachine:
    def __init__(self, root: Path, *, create: bool = True) -> None:
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "mission_state.json"

    def get(self) -> FourStateModel:
        if not self.path.exists():
            return FourStateModel(
                current="UNKNOWN — onboarding incomplete or status not yet captured",
                ideal="",
                gap="",
                next_admissible_action="",
                truth_label="UNKNOWN",
            )
        data = json.loads(self.path.read_text(encoding="utf-8"))
        known = {f for f in FourStateModel.__dataclass_fields__}
        return FourStateModel(**{k: v for k, v in data.items() if k in known})

    def update(
        self,
        *,
        current: str | None = None,
        ideal: str | None = None,
        gap: str | None = None,
        next_admissible_action: str | None = None,
        truth_label: str | None = None,
    ) -> FourStateModel:
        existing = self.get()
        updated = FourStateModel(
            current=current if current is not None else existing.current,
            ideal=ideal if ideal is not None else existing.ideal,
            gap=gap if gap is not None else existing.gap,
            next_admissible_action=(
                next_admissible_action
                if next_admissible_action is not None
                else existing.next_admissible_action
            ),
            truth_label=(
                truth_label if truth_label is not None else existing.truth_label
            ),
            timestamp=_utc_now(),
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(updated.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return updated
