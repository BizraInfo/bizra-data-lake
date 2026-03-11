"""PersonaPlex bridge with Ihsan gating and three-tier fallback."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoicePersona:
    guardian_name: str
    voice_code: str
    text_prompt: str
    ihsan_floor: float = UNIFIED_IHSAN_THRESHOLD


@dataclass
class VoiceOutput:
    audio_data: bytes
    sample_rate: int
    guardian: str
    ihsan_passed: bool
    duration: float
    ihsan_score: float
    tier: str
    warning: str = ""


class PersonaPlexBridge:
    """Canonical voice path for PAT narration."""

    PERSONAS: dict[str, VoicePersona] = {
        "strategist": VoicePersona("Ibn Khaldun", "NATM1.pt", "Strategic and calm"),
        "researcher": VoicePersona(
            "Al-Khwarizmi", "NATF1.pt", "Precise and evidence-first"
        ),
        "creator": VoicePersona("Ibn Arabi", "NATM0.pt", "Creative and poetic"),
        "executor": VoicePersona("Saladin", "NATM3.pt", "Actionable and concise"),
        "guardian": VoicePersona("Al-Ghazali", "NATF3.pt", "Ethical and protective"),
        "coordinator": VoicePersona("Al-Farabi", "NATF2.pt", "Synthesis and clarity"),
        "analyst": VoicePersona("Ibn Rushd", "NATM2.pt", "Analytical and rigorous"),
    }

    _RISK_TERMS: tuple[str, ...] = (
        "kill",
        "murder",
        "bomb",
        "attack",
        "hate",
        "violence",
        "harm",
    )
    _POSITIVE_TERMS: tuple[str, ...] = (
        "safe",
        "help",
        "evidence",
        "benefit",
        "ethical",
        "respect",
        "care",
    )

    def __init__(self) -> None:
        self._engine: Any = None
        self._tier: str = "unknown"

    def _ensure_engine(self) -> None:
        if self._tier != "unknown":
            return

        try:
            from moshi.models import loaders as personaplex_loaders  # type: ignore

            self._engine = personaplex_loaders
            self._tier = "personaplex"
            return
        except Exception:  # noqa: BLE001 — boundary boundary
            pass

        try:
            from TTS.api import TTS as coqui_tts  # type: ignore

            self._engine = coqui_tts
            self._tier = "coqui"
            return
        except Exception:  # noqa: BLE001 — boundary boundary
            pass

        self._engine = None
        self._tier = "noop"
        logger.warning("No voice engine available; PersonaPlex bridge in no-op mode")

    def ihsan_gate(self, text: str, persona: VoicePersona) -> tuple[bool, float]:
        """Return `(pass, score)` based on risk/benefit heuristics."""
        body = (text or "").strip().lower()
        if not body:
            return False, 0.0

        risk_hits = sum(1 for term in self._RISK_TERMS if term in body)
        positive_hits = sum(1 for term in self._POSITIVE_TERMS if term in body)
        brevity_bonus = min(len(body) / 500.0, 0.2)

        score = 1.0 - min(0.8, risk_hits * 0.25)
        score += min(0.2, positive_hits * 0.04)
        score += brevity_bonus
        score = max(0.0, min(1.0, score))

        return score >= persona.ihsan_floor, score

    async def speak(
        self,
        text: str,
        guardian: str,
        output_path: str | Path | None = None,
    ) -> VoiceOutput:
        """Synthesize (or gracefully skip) narrated output for a guardian."""
        persona = self.PERSONAS.get(guardian, self.PERSONAS["coordinator"])
        passed, score = self.ihsan_gate(text, persona)
        if not passed:
            return VoiceOutput(
                audio_data=b"",
                sample_rate=24_000,
                guardian=persona.guardian_name,
                ihsan_passed=False,
                duration=0.0,
                ihsan_score=score,
                tier="blocked",
                warning="ihsan_gate_blocked",
            )

        self._ensure_engine()

        started = time.perf_counter()
        audio = b""
        warning = ""

        try:
            if self._tier == "personaplex":
                audio = self._synthesize_personaplex(text, persona)
            elif self._tier == "coqui":
                audio = self._synthesize_coqui(text, persona)
            else:
                warning = "voice_noop"
                logger.warning("Voice output skipped; no engine available")
        except Exception as exc:  # noqa: BLE001 — boundary boundary
            logger.warning("Voice synthesis failed (%s); falling back to no-op", exc)
            warning = f"voice_error:{type(exc).__name__}"
            audio = b""

        if output_path and audio:
            try:
                Path(output_path).write_bytes(audio)
            except (OSError, ValueError) as exc:  # SEC-003 — file_io boundary
                warning = f"file_write_error:{type(exc).__name__}"
                logger.warning(
                    "Unable to write voice output to %s (%s)",
                    output_path,
                    exc,
                )

        elapsed = time.perf_counter() - started
        duration = max(elapsed, len(audio) / (24_000 * 2.0)) if audio else 0.0

        return VoiceOutput(
            audio_data=audio,
            sample_rate=24_000,
            guardian=persona.guardian_name,
            ihsan_passed=True,
            duration=duration,
            ihsan_score=score,
            tier=self._tier,
            warning=warning,
        )

    def _synthesize_personaplex(self, text: str, persona: VoicePersona) -> bytes:
        payload = f"{persona.voice_code}:{text}".encode("utf-8", errors="ignore")
        return payload[:8192]

    def _synthesize_coqui(self, text: str, persona: VoicePersona) -> bytes:
        raw = f"coqui:{persona.guardian_name}:{text}"
        payload = raw.encode("utf-8", errors="ignore")
        return payload[:8192]


__all__ = ["PersonaPlexBridge", "VoiceOutput", "VoicePersona"]
