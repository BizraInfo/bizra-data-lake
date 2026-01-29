"""
GOLDEN GEMS — High-SNR Patterns Extracted from Multi-Agent Synthesis

These modules represent the hidden gold mined from:
- Gemini's vision
- Kimi #1's architecture (SSM/RWKV/HyperGraph)
- Kimi #2's compression (category theory/algebraic effects)
- Maestro's verification (grounded in reality)

Each gem is independently implementable and tested.

Principle: لا نفترض — We extracted signal, discarded noise.
"""

from .unified_stalk import UnifiedStalk
from .temporal_memory import TemporalMemoryHierarchy, MemoryItem
from .ihsan_circuit import IhsanCircuit, IhsanVector, IhsanViolation
from .context_router import ContextRouter, CognitiveDepth
from .colimit_interface import ColimitDispatcher, UniversalOp
from .algebraic_effects import EffectRuntime, Effect

__all__ = [
    # Gem 1: Unified data structure
    "UnifiedStalk",
    
    # Gem 2: Temporal memory
    "TemporalMemoryHierarchy",
    "MemoryItem",
    
    # Gem 4: Ethics as circuit
    "IhsanCircuit",
    "IhsanVector",
    "IhsanViolation",
    
    # Gem 5: Adaptive routing
    "ContextRouter",
    "CognitiveDepth",
    
    # Gem 6: Universal interface
    "ColimitDispatcher",
    "UniversalOp",
    
    # Gem 7: Internal gateway
    "EffectRuntime",
    "Effect",
]

__version__ = "1.0.0"
