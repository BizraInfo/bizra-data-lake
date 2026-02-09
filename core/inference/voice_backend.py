"""
BIZRA Voice Backend — PersonaPlex Integration
═══════════════════════════════════════════════════════════════════════════════

Bridges the MultiModalRouter to PersonaPlex for voice tasks.
Provides unified interface for speech-to-speech processing.

Capabilities:
- Speech-to-text (STT): Transcribe user audio
- Text-to-speech (TTS): Generate Guardian responses
- Full-duplex: Real-time conversation with interruption support
- Persona control: Route to specific BIZRA Guardian voices

Standing on Giants:
- PersonaPlex (NVIDIA/Roy et al.): Real-time voice personas
- Moshi Architecture (Kyutai): Full-duplex speech
- BIZRA Guardians: 8 specialized AI personas

Created: 2026-02-04 | BIZRA Sovereignty
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VoiceConfig:
    """Configuration for voice backend."""

    # PersonaPlex settings
    personaplex_endpoint: str = "localhost:8998"
    hf_repo: str = "nvidia/personaplex-7b-v1"
    device: str = "cuda"

    # Audio settings
    sample_rate: int = 24000
    frame_rate: float = 12.5  # Moshi default

    # Voice prompts directory
    voice_dir: Path = field(
        default_factory=lambda: Path("/mnt/c/BIZRA-DATA-LAKE/voices")
    )

    # Guardian defaults
    default_guardian: str = "architect"

    # Ihsān settings
    ihsan_threshold: float = 0.95
    require_ihsan_gate: bool = True


@dataclass
class VoiceRequest:
    """Voice processing request."""

    audio: Optional[np.ndarray] = None
    text: Optional[str] = None  # For TTS-only mode
    guardian: str = "architect"
    mode: str = "full_duplex"  # "stt", "tts", "full_duplex"
    voice_prompt: Optional[str] = None  # Override default voice


@dataclass
class VoiceResponse:
    """Voice processing response."""

    text: str = ""
    audio: Optional[np.ndarray] = None
    guardian: str = ""
    latency_ms: float = 0.0
    ihsan_score: float = 1.0
    ihsan_passed: bool = True

    @property
    def has_audio(self) -> bool:
        return self.audio is not None and len(self.audio) > 0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "has_audio": self.has_audio,
            "guardian": self.guardian,
            "latency_ms": self.latency_ms,
            "ihsan_score": self.ihsan_score,
            "ihsan_passed": self.ihsan_passed,
        }


class VoiceBackend:
    """
    Voice backend that routes to PersonaPlex.

    This provides a unified interface for voice processing that integrates
    with the MultiModalRouter. When a task is detected as VOICE capability,
    it routes here instead of to LM Studio.

    Usage:
        backend = VoiceBackend()
        await backend.initialize()

        # Full-duplex conversation
        response = await backend.process(VoiceRequest(
            audio=audio_data,
            guardian="ethics"
        ))

        # Text-to-speech only
        response = await backend.speak(
            text="Hello, I am the Ethics Guardian",
            guardian="ethics"
        )
    """

    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self._personaplex: Any = None
        self._initialized = False
        self._guardians_registered = False

    async def initialize(self) -> bool:
        """Initialize PersonaPlex engine."""
        if self._initialized:
            return True

        try:
            # Import PersonaPlex engine
            from ..personaplex.engine import BIZRAPersonaPlex, PersonaPlexConfig
            from ..personaplex.guardians import BIZRA_GUARDIANS

            # Create PersonaPlex config
            pp_config = PersonaPlexConfig(
                hf_repo=self.config.hf_repo,
                device=self.config.device,
                sample_rate=self.config.sample_rate,
                frame_rate=self.config.frame_rate,
                voice_dir=self.config.voice_dir,
                ihsan_threshold=self.config.ihsan_threshold,
                require_ihsan_gate=self.config.require_ihsan_gate,
                server_port=int(self.config.personaplex_endpoint.split(":")[-1]),
            )

            # Initialize engine
            self._personaplex = BIZRAPersonaPlex(pp_config)
            await self._personaplex.initialize()

            # Register all BIZRA Guardians
            for guardian in BIZRA_GUARDIANS.values():
                self._personaplex.register_guardian(guardian)

            self._guardians_registered = True
            self._initialized = True
            logger.info("Voice backend initialized with PersonaPlex")
            return True

        except ImportError as e:
            logger.warning(f"PersonaPlex not available: {e}")
            self._initialized = False
            return False
        except Exception as e:
            logger.error(f"Failed to initialize voice backend: {e}")
            self._initialized = False
            return False

    async def process(self, request: VoiceRequest) -> VoiceResponse:
        """
        Process a voice request.

        Routes to appropriate PersonaPlex method based on mode:
        - stt: Speech-to-text transcription only
        - tts: Text-to-speech synthesis only
        - full_duplex: Full conversation with audio in/out
        """
        if not self._initialized:
            if not await self.initialize():
                return VoiceResponse(
                    text="Voice backend not available",
                    ihsan_passed=False,
                )

        import time

        start = time.time()

        try:
            if request.mode == "stt":
                # Speech-to-text only
                result = await self._personaplex.transcribe(request.audio)
                return VoiceResponse(
                    text=result.get("text", ""),
                    guardian=request.guardian,
                    latency_ms=(time.time() - start) * 1000,
                )

            elif request.mode == "tts":
                # Text-to-speech only
                result = await self._personaplex.synthesize(
                    text=request.text or "",
                    guardian_name=request.guardian,
                )
                return VoiceResponse(
                    text=request.text or "",
                    audio=result.audio if hasattr(result, "audio") else None,
                    guardian=request.guardian,
                    latency_ms=(time.time() - start) * 1000,
                    ihsan_score=1.0,
                    ihsan_passed=True,
                )

            else:
                # Full-duplex conversation
                result = await self._personaplex.process_audio(
                    guardian_name=request.guardian,
                    audio_input=request.audio,
                )
                return VoiceResponse(
                    text=result.text,
                    audio=result.audio,
                    guardian=result.guardian_name,
                    latency_ms=result.latency_ms,
                    ihsan_score=1.0 if result.ihsan_passed else 0.5,
                    ihsan_passed=result.ihsan_passed,
                )

        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            return VoiceResponse(
                text=f"Error: {str(e)}",
                guardian=request.guardian,
                latency_ms=(time.time() - start) * 1000,
                ihsan_passed=False,
            )

    async def speak(self, text: str, guardian: str = "architect") -> VoiceResponse:
        """Convenience method for text-to-speech."""
        return await self.process(
            VoiceRequest(
                text=text,
                guardian=guardian,
                mode="tts",
            )
        )

    async def transcribe(self, audio: np.ndarray) -> VoiceResponse:
        """Convenience method for speech-to-text."""
        return await self.process(
            VoiceRequest(
                audio=audio,
                mode="stt",
            )
        )

    async def converse(
        self,
        audio: np.ndarray,
        guardian: str = "architect",
    ) -> VoiceResponse:
        """Convenience method for full-duplex conversation."""
        return await self.process(
            VoiceRequest(
                audio=audio,
                guardian=guardian,
                mode="full_duplex",
            )
        )

    def list_guardians(self) -> List[str]:
        """List available Guardian personas."""
        return [
            "architect",  # System architecture, design
            "security",  # Security, privacy, protection
            "ethics",  # Ethical reasoning, values
            "reasoning",  # Logical analysis, proof
            "knowledge",  # Information, memory, retrieval
            "creative",  # Innovation, synthesis
            "integration",  # Coordination, connection
            "nucleus",  # Core orchestration
        ]

    def list_voices(self) -> List[str]:
        """List available voice prompts."""
        return [
            # Natural voices
            "NATF0",
            "NATF1",
            "NATF2",
            "NATF3",  # Natural female
            "NATM0",
            "NATM1",
            "NATM2",
            "NATM3",  # Natural male
            # Variety voices
            "VARF0",
            "VARF1",
            "VARF2",
            "VARF3",
            "VARF4",  # Variety female
            "VARM0",
            "VARM1",
            "VARM2",
            "VARM3",
            "VARM4",  # Variety male
        ]

    @property
    def is_available(self) -> bool:
        """Check if voice backend is available."""
        return self._initialized

    @property
    def status(self) -> Dict[str, Any]:
        """Get backend status."""
        return {
            "initialized": self._initialized,
            "guardians_registered": self._guardians_registered,
            "personaplex_available": self._personaplex is not None,
            "device": self.config.device,
            "endpoint": self.config.personaplex_endpoint,
            "ihsan_threshold": self.config.ihsan_threshold,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY & SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_voice_backend_instance: Optional[VoiceBackend] = None


def get_voice_backend(config: Optional[VoiceConfig] = None) -> VoiceBackend:
    """Get singleton voice backend instance."""
    global _voice_backend_instance
    if _voice_backend_instance is None:
        _voice_backend_instance = VoiceBackend(config)
    return _voice_backend_instance


async def check_voice_availability() -> bool:
    """Check if voice processing is available."""
    backend = get_voice_backend()
    return await backend.initialize()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio

    async def test_voice_backend():
        print("═" * 75)
        print("    BIZRA VOICE BACKEND — PersonaPlex Integration")
        print("═" * 75)

        backend = get_voice_backend()

        print("\n📋 Configuration:")
        print(f"   Endpoint: {backend.config.personaplex_endpoint}")
        print(f"   Device: {backend.config.device}")
        print(f"   Ihsān Threshold: {backend.config.ihsan_threshold}")

        print("\n🎙️ Available Guardians:")
        for g in backend.list_guardians():
            print(f"   • {g}")

        print("\n🔊 Available Voices:")
        voices = backend.list_voices()
        print(f"   {len(voices)} voice prompts available")

        print("\n⏳ Initializing PersonaPlex...")
        if await backend.initialize():
            print("   ✅ Voice backend ready")

            # Test TTS
            print("\n🗣️ Testing TTS...")
            response = await backend.speak(
                text="Hello, I am the Architect Guardian. How may I assist you?",
                guardian="architect",
            )
            print(f"   Response: {response.text[:50]}...")
            print(f"   Has audio: {response.has_audio}")
            print(f"   Latency: {response.latency_ms:.1f}ms")
        else:
            print("   ⚠️ Voice backend not available (PersonaPlex not installed)")
            print("   To enable: pip install personaplex")

        print("\n" + "═" * 75)

    asyncio.run(test_voice_backend())
