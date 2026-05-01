"""Dema profile — onboarding identity captured per node.

Stored locally only (under sovereign_state/dema/profile.json which is
gitignored). Holds preferred name, mother/work languages, persona tone, and
memory-consent setting. No secrets.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1.0"

VALID_CONSENT = ("off", "local", "private", "shared_candidates")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class DemaProfile:
    preferred_name: str
    mother_language: str
    work_language: str
    persona_tone: str = "pragmatic-mystic"
    memory_consent: str = "local"
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.memory_consent not in VALID_CONSENT:
            raise ValueError(
                f"memory_consent must be one of {VALID_CONSENT}, "
                f"got {self.memory_consent!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProfileStore:
    """Disk-backed persistence for DemaProfile."""

    def __init__(self, root: Path, *, create: bool = True) -> None:
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "profile.json"

    def load(self) -> DemaProfile | None:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        # Ignore unknown future fields rather than failing.
        known = {f for f in DemaProfile.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return DemaProfile(**filtered)

    def save(self, profile: DemaProfile) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        profile.updated_at = _utc_now()
        self.path.write_text(
            json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self.path

    def init_from_env_or_defaults(
        self, *, defaults: dict[str, str] | None = None
    ) -> DemaProfile:
        """Build a profile from env vars, defaults, or sensible fallbacks.

        Env vars (all optional):
          DEMA_PREFERRED_NAME, DEMA_MOTHER_LANGUAGE, DEMA_WORK_LANGUAGE,
          DEMA_PERSONA_TONE, DEMA_MEMORY_CONSENT.
        """
        d = dict(defaults or {})
        return DemaProfile(
            preferred_name=os.environ.get(
                "DEMA_PREFERRED_NAME", d.get("preferred_name", "operator")
            ),
            mother_language=os.environ.get(
                "DEMA_MOTHER_LANGUAGE", d.get("mother_language", "en")
            ),
            work_language=os.environ.get(
                "DEMA_WORK_LANGUAGE", d.get("work_language", "en")
            ),
            persona_tone=os.environ.get(
                "DEMA_PERSONA_TONE", d.get("persona_tone", "pragmatic-mystic")
            ),
            memory_consent=os.environ.get(
                "DEMA_MEMORY_CONSENT", d.get("memory_consent", "local")
            ),
        )
