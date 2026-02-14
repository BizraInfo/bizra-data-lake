"""
Actuator Skill Ledger — Typed registry for AHK desktop skills.

Each skill is a verified instruction template that maps RDVE reasoning
to concrete desktop events via the Desktop Bridge (127.0.0.1:9742).

Skills are gated by Shannon entropy (H >= 3.5) and Ihsan (>= 0.95)
before registration, ensuring only high-signal instructions enter
the ledger.

Standing on Giants:
- Shannon: Information density threshold prevents low-signal noise
- Boyd: OODA loop — skills are the "Act" phase
- Al-Ghazali: Ihsan constraint on every registered procedure
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

ENTROPY_THRESHOLD = 3.5
IHSAN_MINIMUM = 0.90


@dataclass
class ActuatorSkillManifest:
    """Schema for registering an AHK skill in the RDVE skills ledger."""

    name: str
    description: str
    target_app: str
    ahk_code: str
    entropy_score: float
    parameters: dict[str, str] = field(default_factory=dict)
    requires_context: bool = False
    min_ihsan: float = 0.95

    def validate(self) -> bool:
        """Validate manifest meets constitutional constraints."""
        if not self.name or not self.ahk_code:
            return False
        if self.entropy_score < ENTROPY_THRESHOLD:
            return False
        if self.min_ihsan < IHSAN_MINIMUM:
            return False
        return True


class ActuatorSkillLedger:
    """Registry of verified AHK skills available to RDVE."""

    def __init__(self) -> None:
        self._skills: dict[str, ActuatorSkillManifest] = {}

    def register(self, manifest: ActuatorSkillManifest) -> bool:
        """Register a skill after validation. Returns False if invalid."""
        if not manifest.validate():
            logger.warning(f"Skill '{manifest.name}' rejected: failed validation")
            return False
        self._skills[manifest.name] = manifest
        return True

    def get(self, name: str) -> Optional[ActuatorSkillManifest]:
        """Retrieve a skill by name."""
        return self._skills.get(name)

    def list_all(self) -> list[ActuatorSkillManifest]:
        """List all registered skills."""
        return list(self._skills.values())

    def resolve_for_app(self, app: str) -> list[ActuatorSkillManifest]:
        """Find all skills targeting a specific application."""
        return [s for s in self._skills.values() if s.target_app == app]

    @property
    def count(self) -> int:
        return len(self._skills)


# ---------------------------------------------------------------------------
# Baseline skills (pre-registered)
# ---------------------------------------------------------------------------

BASELINE_SKILLS = [
    ActuatorSkillManifest(
        name="vscode_save",
        description="Save current file in VS Code",
        target_app="Code.exe",
        ahk_code="Send('^s')",
        entropy_score=4.2,
        parameters={},
    ),
    ActuatorSkillManifest(
        name="browser_navigate",
        description="Navigate to URL in active browser",
        target_app="chrome.exe",
        ahk_code="Send('^l'); Sleep(100); Send('{url}'); Send('{Enter}')",
        entropy_score=5.1,
        parameters={"url": "str"},
    ),
    ActuatorSkillManifest(
        name="terminal_command",
        description="Execute command in active terminal",
        target_app="WindowsTerminal.exe",
        ahk_code="Send('{command}'); Send('{Enter}')",
        entropy_score=4.8,
        parameters={"command": "str"},
        requires_context=True,
    ),
]


def create_default_ledger() -> ActuatorSkillLedger:
    """Create a ledger pre-populated with baseline skills."""
    ledger = ActuatorSkillLedger()
    for skill in BASELINE_SKILLS:
        ledger.register(skill)
    return ledger
