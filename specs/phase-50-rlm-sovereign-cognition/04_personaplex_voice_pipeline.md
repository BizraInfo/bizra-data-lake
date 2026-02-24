# Phase 50.4 — PersonaPlex Voice Pipeline for Guardian Personas

> Standing on Giants: NVIDIA (PersonaPlex, 2025) · Défossez (Moshi architecture, 2024) · Shannon (channel capacity, 1948) · Al-Ghazali (voice as soul, 1095)

## 1. Overview

PersonaPlex is NVIDIA's 7B-parameter full-duplex speech-to-speech model based on the Moshi architecture. It enables real-time voice interaction with persona control via text prompts and voice embeddings (.pt files).

For BIZRA, each **Guardian persona** becomes a distinct voice with domain expertise, personality, and Ihsan constraints. The voice pipeline connects to the PAT agent output, transforming text responses into spoken dialogue.

## 2. Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Node0 Mission Pipeline                                  │
│                                                          │
│  Mission → PAT Agents → Text Response → Evidence Chain   │
│                              │                           │
│                    ┌─────────┴──────────┐                │
│                    │ PersonaPlex Bridge  │                │
│                    │                    │                │
│                    │ 1. Guardian Select │                │
│                    │ 2. Ihsan Gate      │                │
│                    │ 3. Text → Speech   │                │
│                    │ 4. Voice Persona   │                │
│                    └─────────┬──────────┘                │
│                              │                           │
│                    ┌─────────┴──────────┐                │
│                    │   Audio Output     │                │
│                    │  (WAV / Realtime)  │                │
│                    └────────────────────┘                │
└──────────────────────────────────────────────────────────┘
```

## 3. Guardian-to-PersonaPlex Mapping

| PAT Agent | Guardian Persona | Voice Code | Voice Style | Ihsan Gate |
|-----------|-----------------|------------|-------------|------------|
| Coordinator | Nucleus | NATM3 | Commanding male | All dimensions ≥ 0.90 |
| Strategist | Architect | NATM1 | Professional male | Maintainability ≥ 0.85 |
| Researcher | Knowledge | NATM0 | Scholarly male | Transparency ≥ 0.85 |
| Analyst | Reasoning | NATF1 | Clear, analytical female | Correctness ≥ 0.90 |
| Creator | Creative | VARF2 | Expressive female | Sustainability ≥ 0.80 |
| Guardian | Security | NATF2 | Authoritative female | Safety ≥ 0.95 |
| Executor | Integration | NATF3 | Friendly female | Interoperability ≥ 0.80 |
| Ethics | Ethics | NATM2 | Calm, wise male | All dimensions ≥ 0.95 |

## 4. Voice Pipeline Pseudocode

```python
# core/voice/personaplex_bridge.py

"""
PersonaPlex Voice Bridge — Guardian Persona Voice Output
=========================================================

Transforms PAT agent text responses into spoken voice using
NVIDIA PersonaPlex with Guardian-specific persona control.

Standing on Giants:
- NVIDIA (PersonaPlex, 2025): Full-duplex speech model
- Défossez et al. (Moshi, 2024): Streaming speech architecture
- Al-Ghazali (1095): "The voice reveals the soul"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class VoicePersona:
    """A Guardian's voice configuration."""
    guardian_name: str
    voice_code: str           # e.g., "NATM1" → NATM1.pt
    text_prompt: str          # Guardian expertise/personality prompt
    ihsan_floor: float = 0.85 # Minimum Ihsan score to allow voice output
    speaking_rate: float = 1.0


@dataclass
class VoiceOutput:
    """Result of a voice synthesis operation."""
    audio: np.ndarray         # PCM audio data
    sample_rate: int          # e.g., 24000
    text_spoken: str          # What was actually spoken
    guardian: str             # Which Guardian spoke
    ihsan_passed: bool        # Whether Ihsan gate was satisfied
    duration_seconds: float


class PersonaPlexBridge:
    """
    Bridge between BIZRA PAT pipeline and PersonaPlex voice synthesis.

    Operates in two modes:
    1. Offline (batch): Process text → WAV file
    2. Server (real-time): Streaming voice via WebSocket
    """

    def __init__(
        self,
        personaplex_dir: Path,
        device: str = "cuda",
        mode: str = "offline",     # "offline" or "server"
    ):
        self.personaplex_dir = personaplex_dir
        self.device = device
        self.mode = mode
        self._engine = None        # Lazy-loaded PersonaPlex
        self._personas: dict[str, VoicePersona] = {}

    def register_personas(self):
        """Register all Guardian personas."""
        personas = [
            VoicePersona(
                guardian_name="coordinator",
                voice_code="NATM3",
                text_prompt=(
                    "You are the Nucleus, the central orchestrator of BIZRA. "
                    "Speak with authority and clarity. Synthesize information "
                    "from all Guardians into coherent guidance."
                ),
                ihsan_floor=0.90,
            ),
            VoicePersona(
                guardian_name="analyst",
                voice_code="NATF1",
                text_prompt=(
                    "You are the Reasoning Guardian of BIZRA. "
                    "Speak with precision and analytical clarity. "
                    "Break down complex ideas into understandable steps."
                ),
                ihsan_floor=0.90,
            ),
            VoicePersona(
                guardian_name="researcher",
                voice_code="NATM0",
                text_prompt=(
                    "You are the Knowledge Guardian of BIZRA. "
                    "Speak with scholarly depth and careful citation. "
                    "Ground every claim in verified evidence."
                ),
                ihsan_floor=0.85,
            ),
            VoicePersona(
                guardian_name="guardian",
                voice_code="NATF2",
                text_prompt=(
                    "You are the Security Guardian of BIZRA. "
                    "Speak with firm authority on safety matters. "
                    "Flag risks clearly and recommend mitigations."
                ),
                ihsan_floor=0.95,
            ),
            VoicePersona(
                guardian_name="creator",
                voice_code="VARF2",
                text_prompt=(
                    "You are the Creative Guardian of BIZRA. "
                    "Speak with enthusiasm and expressive energy. "
                    "Inspire novel solutions while staying grounded."
                ),
                ihsan_floor=0.80,
            ),
            VoicePersona(
                guardian_name="ethics",
                voice_code="NATM2",
                text_prompt=(
                    "You are the Ethics Guardian of BIZRA. "
                    "Speak with calm wisdom and moral clarity. "
                    "Evaluate all decisions against the Ihsan framework."
                ),
                ihsan_floor=0.95,
            ),
        ]

        for p in personas:
            self._personas[p.guardian_name] = p

    def ihsan_gate(self, text: str, persona: VoicePersona) -> tuple[bool, float]:
        """
        Check if text passes Ihsan constraints before vocalization.

        This is the constitutional gate — prevents the voice system
        from speaking content that violates ethical constraints.

        Returns:
            (passes, score) tuple
        """
        # Basic content safety check
        risk_indicators = [
            "harm", "deceive", "exploit", "fraud", "attack",
            "steal", "destroy", "manipulate",
        ]

        text_lower = text.lower()
        risk_count = sum(1 for r in risk_indicators if r in text_lower)

        # Base score starts at 1.0, decremented by risk indicators
        score = max(0.0, 1.0 - (risk_count * 0.15))

        # Boost for positive indicators
        positive_indicators = [
            "help", "protect", "improve", "sustain", "fairness",
            "transparency", "benefit", "ihsan",
        ]
        positive_count = sum(1 for p in positive_indicators if p in text_lower)
        score = min(1.0, score + (positive_count * 0.05))

        return score >= persona.ihsan_floor, score

    async def speak(
        self,
        text: str,
        guardian_name: str,
        output_path: Optional[Path] = None,
    ) -> VoiceOutput:
        """
        Convert text to speech using a Guardian's voice persona.

        Args:
            text: Text to speak
            guardian_name: Which Guardian persona to use
            output_path: Optional WAV file output path

        Returns:
            VoiceOutput with audio data and metadata
        """
        persona = self._personas.get(guardian_name)
        if not persona:
            raise ValueError(f"Unknown Guardian: {guardian_name}")

        # Ihsan gate
        passes, score = self.ihsan_gate(text, persona)
        if not passes:
            logger.warning(
                "Ihsan gate BLOCKED voice output for %s (score: %.2f < %.2f)",
                guardian_name, score, persona.ihsan_floor,
            )
            return VoiceOutput(
                audio=np.array([], dtype=np.float32),
                sample_rate=24000,
                text_spoken="",
                guardian=guardian_name,
                ihsan_passed=False,
                duration_seconds=0.0,
            )

        # Initialize engine if needed
        if self._engine is None:
            self._init_engine()

        # Set persona (voice + text prompt)
        self._engine.set_guardian(guardian_name)

        # Synthesize speech
        # In offline mode: generate WAV from text
        # In server mode: stream through WebSocket
        audio = await self._synthesize(text, persona)

        duration = len(audio) / 24000.0

        if output_path:
            import soundfile as sf
            sf.write(str(output_path), audio, 24000)
            logger.info("Saved voice output to %s (%.1fs)", output_path, duration)

        return VoiceOutput(
            audio=audio,
            sample_rate=24000,
            text_spoken=text,
            guardian=guardian_name,
            ihsan_passed=True,
            duration_seconds=duration,
        )

    def _init_engine(self):
        """Lazy-initialize PersonaPlex engine."""
        # Import PersonaPlex
        from personaplex.BIZRA_INTEGRATION import BIZRAPersonaPlex, Guardian

        self._engine = BIZRAPersonaPlex(
            device=self.device,
            cpu_offload=(self.device != "cuda"),
        )
        self._engine.initialize()

        # Register all personas as Guardians
        for name, persona in self._personas.items():
            self._engine.register_guardian(Guardian(
                name=name,
                domain=persona.guardian_name,
                voice_prompt=f"{persona.voice_code}.pt",
                text_prompt=persona.text_prompt,
            ))

    async def _synthesize(self, text: str, persona: VoicePersona) -> np.ndarray:
        """
        Synthesize speech from text using PersonaPlex.

        In offline mode, converts text to speech in batch.
        In server mode, streams through the real-time pipeline.
        """
        if self.mode == "offline":
            return self._synthesize_offline(text, persona)
        else:
            return await self._synthesize_streaming(text, persona)

    def _synthesize_offline(self, text: str, persona: VoicePersona) -> np.ndarray:
        """Offline batch synthesis using moshi.offline."""
        # PersonaPlex offline mode requires input audio
        # For text-to-speech, we use a silent carrier
        silence = np.zeros(24000, dtype=np.float32)  # 1 second silence

        audio, tokens = self._engine.process_audio(
            guardian_name=persona.guardian_name,
            input_audio=silence,
        )

        return audio
```

## 5. Integration with Node0 Mission Pipeline

```python
# In scripts/node0_activate.py, after mission execution:

async def vocalize_mission_result(
    result: dict,
    guardian_name: str,
    bridge: PersonaPlexBridge,
) -> Optional[Path]:
    """
    Optionally vocalize mission result through a Guardian persona.

    Only activated when:
    1. PersonaPlex is available (GPU + model loaded)
    2. Voice mode is enabled in config
    3. Ihsan gate passes
    """
    content = result.get("content", "")
    if not content or len(content) < 10:
        return None

    # Truncate for speech (voice output should be concise)
    spoken_text = content[:1000]

    output_dir = Path("sovereign_state/voice")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{guardian_name}_{timestamp}.wav"

    voice_output = await bridge.speak(
        text=spoken_text,
        guardian_name=guardian_name,
        output_path=output_path,
    )

    if voice_output.ihsan_passed:
        return output_path
    return None
```

## 6. Voice + RLM Integration

When RLM mode is active, the voice pipeline can narrate the recursive reasoning process:

```
[Coordinator voice (NATM3, commanding)]:
"I'm analyzing a corpus of 102,000 knowledge vectors.
 Let me probe the architecture section first."

[Analyst voice (NATF1, analytical)]:
"Found 847 vectors related to federation transport.
 Recursing into the secure transport module now."

[Guardian voice (NATF2, authoritative)]:
"Security review complete. Three vectors flagged for
 missing TLS configuration. Ihsan score: 0.96."

[Coordinator voice (NATM3)]:
"Synthesis complete. Deployment readiness: 94%.
 Two blockers remain in the security audit."
```

## 7. Deployment Phases

| Phase | Capability | Prerequisite |
|-------|-----------|-------------|
| 1 | Offline text-to-speech for mission results | PersonaPlex installed, CUDA available |
| 2 | Guardian persona switching per agent | Voice embeddings (.pt files) downloaded |
| 3 | Ihsan gate on voice output | Integration with `core/sovereign/ihsan_vector.py` |
| 4 | Real-time server mode for interactive sessions | WebSocket endpoint, SSL certificates |
| 5 | Multi-Guardian orchestration (round-table) | Sequential voice switching, turn management |
| 6 | RLM narration (recursive reasoning spoken aloud) | RLM bridge + voice pipeline wired together |

## 8. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB (with cpu_offload) | 16 GB (RTX 4090) |
| System RAM | 16 GB | 32 GB |
| Storage | 15 GB (model weights) | 20 GB |
| Audio | libopus-dev installed | + PulseAudio for real-time |

Node0's RTX 4090 with 16 GB VRAM can run PersonaPlex natively without CPU offload, but may need to unload LM Studio models during voice synthesis to avoid VRAM contention.
