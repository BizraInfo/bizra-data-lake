"""
BIZRA Memory Auto-Coder -- Cognitive Pattern Synthesis

Distills accumulated agent memory into a reusable codebook of cognitive
patterns, accelerating future reasoning through compressed experience.

Exports:
    MemorySynthesizer  -- PDCA cycle that clusters memories into patterns
    PatternCodebook    -- Indexed collection of synthesized patterns
    SynthesizedPattern -- A single reusable cognitive pattern
"""

from __future__ import annotations

from .memory_synthesizer import MemorySynthesizer, MemoryRecord
from .pattern_codebook import PatternCodebook, SynthesizedPattern

__all__ = [
    "MemorySynthesizer",
    "MemoryRecord",
    "PatternCodebook",
    "SynthesizedPattern",
]

__version__ = "1.0.0"
