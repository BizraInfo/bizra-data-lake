"""Re-export from canonical location: core.sovereign.collective_synthesizer"""
# Canonical implementation is in core/sovereign/ (uses centralized constants)
from core.sovereign.collective_synthesizer import *  # noqa: F401,F403
from core.sovereign.collective_synthesizer import (
    AgentOutput,
    CollectiveSynthesizer,
    ConflictStrategy,
    ResolvedOutput,
    SynthesizedResult,
)
