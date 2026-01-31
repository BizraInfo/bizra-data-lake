"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA PERSONAPLEX INTEGRATION — VOICE-ENABLED GUARDIANS                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Standing on Giants:                                                        ║
║   • NVIDIA PersonaPlex (Roy et al., 2026)                                    ║
║   • Moshi Full-Duplex Speech Model (Kyutai)                                  ║
║   • BIZRA Guardian System (Node0 Genesis)                                    ║
║                                                                              ║
║   "The voice is the soul. The role is the purpose. The Ihsān is the gate."  ║
║                                                                              ║
║   Created: 2026-01-30 | BIZRA Sovereignty + PersonaPlex Integration          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from .guardians import (
    Guardian,
    IhsanVector,
    BIZRA_GUARDIANS,
    get_guardian,
    list_guardians,
)
from .engine import (
    BIZRAPersonaPlex,
    PersonaPlexConfig,
    VoiceResponse,
)
from .voices import (
    VoicePrompt,
    VOICE_LIBRARY,
    get_voice,
)

__all__ = [
    # Guardians
    "Guardian",
    "IhsanVector",
    "BIZRA_GUARDIANS",
    "get_guardian",
    "list_guardians",
    # Engine
    "BIZRAPersonaPlex",
    "PersonaPlexConfig",
    "VoiceResponse",
    # Voices
    "VoicePrompt",
    "VOICE_LIBRARY",
    "get_voice",
]

__version__ = "1.0.0"
__author__ = "BIZRA Node0 + NVIDIA PersonaPlex Integration"
