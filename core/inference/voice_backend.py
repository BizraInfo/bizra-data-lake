"""Voice backend shim that routes canonical TTS through PersonaPlexBridge."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.voice.personaplex_bridge import PersonaPlexBridge

logger = logging.getLogger(__name__)


@dataclass
class VoiceConfig:
    sample_rate: int = 24_000
    default_guardian: str = "coordinator"
    voice_dir: Path = field(default_factory=lambda: Path("voices"))
    ihsan_threshold: float = 0.95


@dataclass
class VoiceRequest:
    audio: Optional[np.ndarray] = None
    text: Optional[str] = None
    guardian: str = "coordinator"
    mode: str = "tts"  # stt | tts | full_duplex


@dataclass
class VoiceResponse:
    text: str = ""
    audio: Optional[np.ndarray] = None
    guardian: str = ""
    latency_ms: float = 0.0
    ihsan_score: float = 1.0
    ihsan_passed: bool = True

    @property
    def has_audio(self) -> bool:
        return self.audio is not None and len(self.audio) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "has_audio": self.has_audio,
            "guardian": self.guardian,
            "latency_ms": self.latency_ms,
            "ihsan_score": self.ihsan_score,
            "ihsan_passed": self.ihsan_passed,
        }


class VoiceBackend:
    """Compatibility shim for existing inference voice callers."""

    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self._bridge: Optional[PersonaPlexBridge] = None
        self._initialized = False

    async def initialize(self) -> bool:
        if self._initialized:
            return True

        try:
            self._bridge = PersonaPlexBridge()
            self._initialized = True
            return True
        except Exception as exc:
            logger.warning("Voice backend initialization failed: %s", exc)
            self._initialized = False
            return False

    async def process(self, request: VoiceRequest) -> VoiceResponse:
        if not self._initialized:
            ready = await self.initialize()
            if not ready:
                return VoiceResponse(
                    text="Voice backend unavailable",
                    guardian=request.guardian,
                    ihsan_passed=False,
                    ihsan_score=0.0,
                )

        start = time.perf_counter()
        guardian = request.guardian or self.config.default_guardian

        if request.mode == "stt":
            return VoiceResponse(
                text="STT not enabled in PersonaPlex bridge shim",
                guardian=guardian,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                ihsan_passed=False,
                ihsan_score=0.0,
            )

        if request.mode == "full_duplex":
            return VoiceResponse(
                text="Full duplex unavailable in shim; use tts mode",
                guardian=guardian,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                ihsan_passed=False,
                ihsan_score=0.0,
            )

        bridge = self._bridge
        assert bridge is not None

        result = await bridge.speak(text=request.text or "", guardian=guardian)
        audio_array: Optional[np.ndarray] = None
        if result.audio_data:
            audio_array = np.frombuffer(result.audio_data, dtype=np.uint8)

        return VoiceResponse(
            text=request.text or "",
            audio=audio_array,
            guardian=result.guardian,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            ihsan_score=result.ihsan_score,
            ihsan_passed=result.ihsan_passed,
        )

    async def speak(self, text: str, guardian: str = "coordinator") -> VoiceResponse:
        return await self.process(
            VoiceRequest(text=text, guardian=guardian, mode="tts")
        )

    async def transcribe(self, audio: np.ndarray) -> VoiceResponse:
        return await self.process(VoiceRequest(audio=audio, mode="stt"))

    async def converse(self, audio: np.ndarray, guardian: str = "coordinator") -> VoiceResponse:
        return await self.process(
            VoiceRequest(audio=audio, guardian=guardian, mode="full_duplex")
        )

    def list_guardians(self) -> List[str]:
        return list(PersonaPlexBridge.PERSONAS.keys())

    def list_voices(self) -> List[str]:
        return [persona.voice_code for persona in PersonaPlexBridge.PERSONAS.values()]

    @property
    def is_available(self) -> bool:
        return self._initialized

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "guardian_count": len(PersonaPlexBridge.PERSONAS),
            "sample_rate": self.config.sample_rate,
        }


_voice_backend_instance: Optional[VoiceBackend] = None


def get_voice_backend(config: Optional[VoiceConfig] = None) -> VoiceBackend:
    global _voice_backend_instance
    if _voice_backend_instance is None:
        _voice_backend_instance = VoiceBackend(config)
    return _voice_backend_instance


async def check_voice_availability() -> bool:
    backend = get_voice_backend()
    return await backend.initialize()


__all__ = [
    "VoiceBackend",
    "VoiceConfig",
    "VoiceRequest",
    "VoiceResponse",
    "check_voice_availability",
    "get_voice_backend",
]
