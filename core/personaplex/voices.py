"""
Voice Prompt Library for PersonaPlex Integration

Each voice prompt is a pre-trained embedding that conditions the model's
vocal characteristics: timbre, pitch, speaking style, and cadence.

Voice Categories:
- NAT (Natural): Professional, clear voices suitable for assistants
- VAR (Variety): More expressive, varied voices for creative applications
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
from pathlib import Path


class VoiceGender(Enum):
    FEMALE = "female"
    MALE = "male"


class VoiceStyle(Enum):
    WARM = "warm"
    CLEAR = "clear"
    AUTHORITATIVE = "authoritative"
    FRIENDLY = "friendly"
    SCHOLARLY = "scholarly"
    PROFESSIONAL = "professional"
    CALM = "calm"
    COMMANDING = "commanding"
    EXPRESSIVE = "expressive"
    VARIED = "varied"


@dataclass
class VoicePrompt:
    """A voice prompt configuration."""
    code: str
    filename: str
    gender: VoiceGender
    style: VoiceStyle
    description: str
    recommended_for: list

    @property
    def path(self) -> str:
        return f"{self.filename}"


# Voice Library - All available PersonaPlex voices
VOICE_LIBRARY: Dict[str, VoicePrompt] = {
    # Natural Female Voices
    "NATF0": VoicePrompt(
        code="NATF0",
        filename="NATF0.pt",
        gender=VoiceGender.FEMALE,
        style=VoiceStyle.WARM,
        description="Warm female voice with approachable tone",
        recommended_for=["customer_service", "onboarding", "support"]
    ),
    "NATF1": VoicePrompt(
        code="NATF1",
        filename="NATF1.pt",
        gender=VoiceGender.FEMALE,
        style=VoiceStyle.CLEAR,
        description="Clear, analytical female voice for precise communication",
        recommended_for=["reasoning", "analysis", "technical_explanation"]
    ),
    "NATF2": VoicePrompt(
        code="NATF2",
        filename="NATF2.pt",
        gender=VoiceGender.FEMALE,
        style=VoiceStyle.AUTHORITATIVE,
        description="Authoritative female voice conveying expertise",
        recommended_for=["security", "compliance", "leadership"]
    ),
    "NATF3": VoicePrompt(
        code="NATF3",
        filename="NATF3.pt",
        gender=VoiceGender.FEMALE,
        style=VoiceStyle.FRIENDLY,
        description="Friendly female voice for approachable interaction",
        recommended_for=["integration", "collaboration", "casual"]
    ),

    # Natural Male Voices
    "NATM0": VoicePrompt(
        code="NATM0",
        filename="NATM0.pt",
        gender=VoiceGender.MALE,
        style=VoiceStyle.SCHOLARLY,
        description="Scholarly male voice with educational tone",
        recommended_for=["knowledge", "teaching", "documentation"]
    ),
    "NATM1": VoicePrompt(
        code="NATM1",
        filename="NATM1.pt",
        gender=VoiceGender.MALE,
        style=VoiceStyle.PROFESSIONAL,
        description="Professional male voice for business contexts",
        recommended_for=["architect", "business", "presentations"]
    ),
    "NATM2": VoicePrompt(
        code="NATM2",
        filename="NATM2.pt",
        gender=VoiceGender.MALE,
        style=VoiceStyle.CALM,
        description="Calm, wise male voice for thoughtful discourse",
        recommended_for=["ethics", "counseling", "meditation"]
    ),
    "NATM3": VoicePrompt(
        code="NATM3",
        filename="NATM3.pt",
        gender=VoiceGender.MALE,
        style=VoiceStyle.COMMANDING,
        description="Commanding male voice for leadership and authority",
        recommended_for=["nucleus", "orchestration", "decisions"]
    ),

    # Variety Female Voices
    "VARF0": VoicePrompt(
        code="VARF0",
        filename="VARF0.pt",
        gender=VoiceGender.FEMALE,
        style=VoiceStyle.VARIED,
        description="Varied female voice 0",
        recommended_for=["creative", "storytelling"]
    ),
    "VARF1": VoicePrompt(
        code="VARF1",
        filename="VARF1.pt",
        gender=VoiceGender.FEMALE,
        style=VoiceStyle.VARIED,
        description="Varied female voice 1",
        recommended_for=["creative", "entertainment"]
    ),
    "VARF2": VoicePrompt(
        code="VARF2",
        filename="VARF2.pt",
        gender=VoiceGender.FEMALE,
        style=VoiceStyle.EXPRESSIVE,
        description="Expressive female voice for creative applications",
        recommended_for=["creative", "innovation", "brainstorming"]
    ),
    "VARF3": VoicePrompt(
        code="VARF3",
        filename="VARF3.pt",
        gender=VoiceGender.FEMALE,
        style=VoiceStyle.VARIED,
        description="Varied female voice 3",
        recommended_for=["roleplay", "characters"]
    ),
    "VARF4": VoicePrompt(
        code="VARF4",
        filename="VARF4.pt",
        gender=VoiceGender.FEMALE,
        style=VoiceStyle.VARIED,
        description="Varied female voice 4",
        recommended_for=["diverse", "multilingual"]
    ),

    # Variety Male Voices
    "VARM0": VoicePrompt(
        code="VARM0",
        filename="VARM0.pt",
        gender=VoiceGender.MALE,
        style=VoiceStyle.VARIED,
        description="Varied male voice 0",
        recommended_for=["creative", "narration"]
    ),
    "VARM1": VoicePrompt(
        code="VARM1",
        filename="VARM1.pt",
        gender=VoiceGender.MALE,
        style=VoiceStyle.VARIED,
        description="Varied male voice 1",
        recommended_for=["creative", "podcasts"]
    ),
    "VARM2": VoicePrompt(
        code="VARM2",
        filename="VARM2.pt",
        gender=VoiceGender.MALE,
        style=VoiceStyle.VARIED,
        description="Varied male voice 2",
        recommended_for=["roleplay", "games"]
    ),
    "VARM3": VoicePrompt(
        code="VARM3",
        filename="VARM3.pt",
        gender=VoiceGender.MALE,
        style=VoiceStyle.VARIED,
        description="Varied male voice 3",
        recommended_for=["characters", "drama"]
    ),
    "VARM4": VoicePrompt(
        code="VARM4",
        filename="VARM4.pt",
        gender=VoiceGender.MALE,
        style=VoiceStyle.VARIED,
        description="Varied male voice 4",
        recommended_for=["diverse", "experimental"]
    ),
}


def get_voice(code: str) -> Optional[VoicePrompt]:
    """Get a voice prompt by code."""
    return VOICE_LIBRARY.get(code.upper())


def get_voices_by_gender(gender: VoiceGender) -> list:
    """Get all voices of a specific gender."""
    return [v for v in VOICE_LIBRARY.values() if v.gender == gender]


def get_voices_by_style(style: VoiceStyle) -> list:
    """Get all voices of a specific style."""
    return [v for v in VOICE_LIBRARY.values() if v.style == style]


def get_recommended_voice(use_case: str) -> Optional[VoicePrompt]:
    """Get a recommended voice for a specific use case."""
    use_case_lower = use_case.lower()
    for voice in VOICE_LIBRARY.values():
        if use_case_lower in voice.recommended_for:
            return voice
    return None


def resolve_voice_path(code: str, base_dir: Optional[Path] = None) -> Path:
    """Resolve the full path to a voice prompt file."""
    voice = get_voice(code)
    if not voice:
        raise ValueError(f"Unknown voice code: {code}")

    if base_dir is None:
        # Default to BIZRA-DATA-LAKE/voices
        base_dir = Path("/mnt/c/BIZRA-DATA-LAKE/voices")

    return base_dir / voice.filename
