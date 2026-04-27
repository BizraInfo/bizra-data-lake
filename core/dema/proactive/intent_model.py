"""Intent prediction — converts an AmbientSignal into a labelled intent.

v0.1 is a deterministic rule-based mapper. Each signal kind has a fixed
intent label, default risk class, default reversibility, and a short
explanation. Real ML-driven intent inference lands in a later phase under
the same interface.

The output IntentPrediction feeds the interruption policy unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.dema.proactive.signals import AmbientSignal


@dataclass
class IntentPrediction:
    intent: str
    confidence: float
    urgency: str
    risk: str  # low | medium | high
    reversible: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""

    def __post_init__(self) -> None:
        if self.risk not in ("low", "medium", "high"):
            raise ValueError(f"risk must be one of low|medium|high, got {self.risk!r}")
        if self.urgency not in ("low", "medium", "high"):
            raise ValueError(
                f"urgency must be one of low|medium|high, got {self.urgency!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# (intent_label, default_risk, default_reversible, explanation_template)
_RULES: dict[str, tuple[str, str, bool, str]] = {
    "downloads_folder_large": (
        "suggest_downloads_review",
        "low",
        True,
        "Downloads folder size suggests review/archive may help; no deletion proposed.",
    ),
    "stale_files_detected": (
        "suggest_archive_stale",
        "low",
        True,
        "Stale files detected; suggest archive (no deletion).",
    ),
    "long_idle_session": (
        "suggest_save_or_pause",
        "low",
        True,
        "Long idle suggests session checkpoint; reversible.",
    ),
    "unfinished_mission": (
        "suggest_resume_mission",
        "low",
        True,
        "Mission state has unresolved gap; suggest resume via Mission tab.",
    ),
    "resource_pressure": (
        "suggest_lighten_load",
        "medium",
        True,
        "Local resource pressure; suggest pausing background work.",
    ),
    "duplicate_delete_candidate": (
        "propose_duplicate_review",
        "medium",
        False,
        "Duplicates suspected; deletion is non-reversible at filesystem layer.",
    ),
    "format_drive_candidate": (
        "propose_format_drive",
        "high",
        False,
        "Drive-format suggestion is destructive and irreversible.",
    ),
    "credential_exposure_candidate": (
        "propose_credential_audit",
        "high",
        False,
        "Possible credential exposure; immutable history makes rotation the only safe path.",
    ),
    "social_post_candidate": (
        "propose_social_draft",
        "medium",
        True,
        "Social post draft requires explicit human approval before publish.",
    ),
}


def predict_intent(signal: AmbientSignal) -> IntentPrediction:
    if signal.kind not in _RULES:
        raise ValueError(f"no intent rule for signal {signal.kind!r}")
    intent, risk, reversible, explanation = _RULES[signal.kind]
    return IntentPrediction(
        intent=intent,
        confidence=signal.confidence,
        urgency=signal.urgency,
        risk=risk,
        reversible=reversible,
        evidence=dict(signal.evidence),
        explanation=explanation,
    )
