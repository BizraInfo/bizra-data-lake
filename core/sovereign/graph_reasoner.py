"""
Graph-of-Thoughts Reasoning Engine — Public Facade
===================================================
This module provides the unified entry point for the Graph-of-Thoughts engine.

Standing on Giants:
- Graph of Thoughts (Besta et al., 2024): "Solving elaborate problems with LLMs"
- Tree of Thoughts (Yao et al., 2023): Deliberate problem solving
- Chain of Thought (Wei et al., 2022): Step-by-step reasoning
- BIZRA ARTE Engine: Symbolic-neural bridge

"The Graph of Thoughts paradigm enables LLMs to pursue and combine
 multiple independent lines of reasoning, moving beyond sequential
 chains to rich, networked cognitive structures."

Module Structure (SPARC refinement):
- graph_types.py      — Enums, data classes (ThoughtNode, ThoughtEdge, etc.)
- graph_operations.py — Core graph operations (add, aggregate, refine, validate, prune)
- graph_search.py     — Search algorithms (find_best_path, backtrack)
- graph_reasoning.py  — High-level reasoning API (reason method)
- graph_core.py       — GraphOfThoughts class (composes mixins)
- graph_reasoner.py   — Public facade (this file)
"""

from __future__ import annotations

import logging

# Main class (composed from mixins)
from .graph_core import GraphOfThoughts

# Types, enums, data classes
from .graph_types import (  # Enums; Data classes
    EdgeType,
    ReasoningPath,
    ReasoningResult,
    ReasoningStrategy,
    ThoughtEdge,
    ThoughtNode,
    ThoughtType,
)

# =============================================================================
# PUBLIC API - Re-export from modular components
# =============================================================================


logger = logging.getLogger(__name__)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ThoughtType",
    "EdgeType",
    "ReasoningStrategy",
    # Data classes
    "ThoughtNode",
    "ThoughtEdge",
    "ReasoningPath",
    "ReasoningResult",
    # Main class
    "GraphOfThoughts",
]
