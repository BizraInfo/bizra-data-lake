"""Dema daily log — append-only operator log under local state.

Stored under sovereign_state/dema/logs/YYYY-MM-DD.jsonl (gitignored). Only
hashes/labels/summaries land in committed artifacts; raw content stays
local.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

VALID_KINDS = ("tick", "mission", "dream", "onboarding", "import", "action")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class DailyLogEntry:
    timestamp: str
    kind: str
    summary: str
    receipt_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(
                f"kind must be one of {VALID_KINDS}, got {self.kind!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DailyLog:
    def __init__(self, root: Path) -> None:
        self.root = Path(root) / "logs"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, date: str | None = None) -> Path:
        return self.root / f"{date or _today_utc()}.jsonl"

    def append(self, entry: DailyLogEntry) -> Path:
        path = self._path_for()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
        return path

    def read_today(self) -> list[DailyLogEntry]:
        return self.read(_today_utc())

    def read(self, date: str) -> list[DailyLogEntry]:
        path = self._path_for(date)
        if not path.exists():
            return []
        out: list[DailyLogEntry] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            out.append(DailyLogEntry(**data))
        return out

    def iter_dates(self) -> Iterable[str]:
        for p in sorted(self.root.glob("*.jsonl")):
            yield p.stem
