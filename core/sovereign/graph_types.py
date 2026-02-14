"""
Graph-of-Thoughts Types — Enums and Data Classes
=================================================
type definitions for the Graph-of-Thoughts reasoning engine.

Standing on Giants:
- Graph of Thoughts (Besta et al., 2024): "Solving elaborate problems with LLMs"
- Tree of Thoughts (Yao et al., 2023): Deliberate problem solving
- Chain of Thought (Wei et al., 2022): Step-by-step reasoning
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD
from core.proof_engine.canonical import hex_digest


class ThoughtType(Enum):
    """Types of thought nodes in the graph."""

    HYPOTHESIS = "hypothesis"  # Initial conjectures
    EVIDENCE = "evidence"  # Supporting/refuting data
    REASONING = "reasoning"  # Logical deduction steps
    SYNTHESIS = "synthesis"  # Merged conclusions
    REFINEMENT = "refinement"  # Improved versions
    VALIDATION = "validation"  # Quality checks
    CONCLUSION = "conclusion"  # Final answers
    QUESTION = "question"  # Sub-questions to explore
    COUNTERPOINT = "counterpoint"  # Alternative perspectives


class EdgeType(Enum):
    """Types of edges connecting thoughts."""

    SUPPORTS = "supports"  # Evidence supports hypothesis
    REFUTES = "refutes"  # Evidence contradicts
    DERIVES = "derives"  # Logical derivation
    SYNTHESIZES = "synthesizes"  # Aggregation relationship
    REFINES = "refines"  # Improvement relationship
    QUESTIONS = "questions"  # Raises question
    VALIDATES = "validates"  # Quality check relationship


class ReasoningStrategy(Enum):
    """High-level reasoning strategies."""

    BREADTH_FIRST = "breadth_first"  # Explore widely first
    DEPTH_FIRST = "depth_first"  # Explore deeply first
    BEST_FIRST = "best_first"  # Follow highest SNR paths
    BEAM_SEARCH = "beam_search"  # Keep top-k paths
    MCTS = "mcts"  # Monte Carlo Tree Search
    ADAPTIVE = "adaptive"  # Switch strategies dynamically


@dataclass
class ThoughtNode:
    """A node in the Graph of Thoughts."""

    id: str
    content: str
    thought_type: ThoughtType
    confidence: float = 0.5  # 0-1 confidence score
    snr_score: float = 0.5  # Signal-to-noise ratio
    depth: int = 0  # Depth in reasoning tree
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Ihsan dimensions
    correctness: float = 0.5
    groundedness: float = 0.5
    coherence: float = 0.5

    @property
    def content_hash(self) -> str:
        """Content-addressed hash (BLAKE3 of canonical content, SEC-001).

        Standing on: Merkle (1979) — content-addressed integrity.
        Makes each node independently verifiable.
        """
        canonical = json.dumps(
            {
                "content": self.content,
                "type": self.thought_type.value,
                "depth": self.depth,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hex_digest(canonical)  # SEC-001: BLAKE3 for Rust interop

    @property
    def ihsan_score(self) -> float:
        """Composite Ihsan score (geometric mean)."""
        scores = [
            max(self.correctness, 1e-10),
            max(self.groundedness, 1e-10),
            max(self.coherence, 1e-10),
            max(self.confidence, 1e-10),
        ]
        return math.exp(sum(math.log(s) for s in scores) / len(scores))

    @property
    def passes_ihsan(self) -> bool:
        """Check if thought passes Ihsan threshold."""
        return self.ihsan_score >= UNIFIED_IHSAN_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": (
                self.content[:200] + "..." if len(self.content) > 200 else self.content
            ),
            "type": self.thought_type.value,
            "confidence": self.confidence,
            "snr": self.snr_score,
            "ihsan": self.ihsan_score,
            "depth": self.depth,
            "content_hash": self.content_hash,
        }


@dataclass
class ThoughtEdge:
    """An edge connecting thought nodes."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0  # Edge importance
    reasoning: str = ""  # Why this connection exists

    def to_dict(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.value,
            "weight": self.weight,
        }


@dataclass
class ReasoningPath:
    """A path through the thought graph."""

    nodes: list[str]  # Node IDs in order
    total_snr: float = 0.0
    total_confidence: float = 0.0

    @property
    def length(self) -> int:
        return len(self.nodes)

    @property
    def average_snr(self) -> float:
        return self.total_snr / max(self.length, 1)


@dataclass
class ReasoningResult:
    """Result from the high-level reason() method."""

    thoughts: list[str]
    conclusion: str
    confidence: float
    depth_reached: int
    snr_score: float
    ihsan_score: float
    passes_threshold: bool
    graph_stats: dict[str, int] = field(default_factory=dict)
    graph_hash: Optional[str] = None  # BLAKE3 hash of the full graph artifact (SEC-001)
    signature: Optional[str] = None  # Ed25519 signature of graph_hash

    def to_dict(self) -> dict[str, Any]:
        result = {
            "thoughts": self.thoughts,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "depth_reached": self.depth_reached,
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
            "passes_threshold": self.passes_threshold,
            "graph_stats": self.graph_stats,
        }
        if self.graph_hash:
            result["graph_hash"] = self.graph_hash
        if self.signature:
            result["signature"] = self.signature
        return result


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
]
