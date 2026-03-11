"""
Sci-Reasoning Thinking Patterns — Native BIZRA Pattern Library
==============================================================

Maps the 15 thinking patterns discovered by Sci-Reasoning (Li et al., 2025)
into BIZRA's Graph-of-Thoughts type system. These patterns represent the
cognitive moves behind 3,819 top-tier ML papers (NeurIPS, ICML, ICLR 2023-2025).

Standing on Giants:
- Li et al. (2025): "Sci-Reasoning: structured scientific innovation patterns"
- Besta et al. (2024): Graph of Thoughts
- Shannon (1948): SNR as pattern signal quality
- Boyd (1976): OODA loop (Observe-Orient-Decide-Act)

Source: https://github.com/AmberLJC/Sci-Reasoning
Dataset: https://huggingface.co/datasets/AmberLJC/Sci-Reasoning
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from core.sovereign.graph_types import EdgeType, ThoughtType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sci_reasoning"


class PatternID(str, Enum):
    """The 15 canonical thinking patterns from Sci-Reasoning."""

    P01 = "P01"  # Gap-Driven Reframing
    P02 = "P02"  # Cross-Domain Synthesis
    P03 = "P03"  # Representation Shift & Primitive Recasting
    P04 = "P04"  # Modular Pipeline Composition
    P05 = "P05"  # Data & Evaluation Engineering
    P06 = "P06"  # Principled Probabilistic Modeling & Uncertainty
    P07 = "P07"  # Formal-Experimental Tightening
    P08 = "P08"  # Approximation Engineering for Scalability
    P09 = "P09"  # Inference-Time Control & Guided Sampling
    P10 = "P10"  # Inject Structural Inductive Bias
    P11 = "P11"  # Multiscale & Hierarchical Modeling
    P12 = "P12"  # Mechanistic Decomposition & Causal Localization
    P13 = "P13"  # Adversary Modeling & Defensive Repurposing
    P14 = "P14"  # Numerics & Systems Co-design
    P15 = "P15"  # Data-Centric Optimization & Active Sampling


class PredecessorRole(str, Enum):
    """Intellectual predecessor relationship types from Sci-Reasoning."""

    BASELINE = "Baseline"
    INSPIRATION = "Inspiration"
    GAP_IDENTIFICATION = "Gap Identification"
    FOUNDATION = "Foundation"
    EXTENSION = "Extension"
    RELATED_PROBLEM = "Related Problem"


# ---------------------------------------------------------------------------
# Role → GoT EdgeType mapping
# ---------------------------------------------------------------------------

ROLE_TO_EDGE_TYPE: dict[PredecessorRole, EdgeType] = {
    PredecessorRole.BASELINE: EdgeType.REFINES,  # Improved upon
    PredecessorRole.INSPIRATION: EdgeType.DERIVES,  # Derived from
    PredecessorRole.GAP_IDENTIFICATION: EdgeType.QUESTIONS,  # Gap raised
    PredecessorRole.FOUNDATION: EdgeType.SUPPORTS,  # Built upon
    PredecessorRole.EXTENSION: EdgeType.REFINES,  # Extended
    PredecessorRole.RELATED_PROBLEM: EdgeType.SYNTHESIZES,  # Cross-problem
}


def role_to_edge_type(role: str) -> EdgeType:
    """Map a predecessor role string to a GoT EdgeType.

    Handles both enum values and raw strings from JSON data.
    Fail-closed: unknown roles map to DERIVES (weakest assumption).
    """
    normalized = role.strip().title()
    for pr in PredecessorRole:
        if pr.value == normalized:
            return ROLE_TO_EDGE_TYPE[pr]
    return EdgeType.DERIVES


# ---------------------------------------------------------------------------
# Pattern → ThoughtType mapping
# ---------------------------------------------------------------------------

PATTERN_TO_THOUGHT_TYPE: dict[PatternID, ThoughtType] = {
    PatternID.P01: ThoughtType.QUESTION,  # Reframing starts with questions
    PatternID.P02: ThoughtType.SYNTHESIS,  # Cross-domain = synthesis
    PatternID.P03: ThoughtType.REFINEMENT,  # Representation shift = refine
    PatternID.P04: ThoughtType.REASONING,  # Pipeline = structured reasoning
    PatternID.P05: ThoughtType.EVIDENCE,  # Data/eval = evidence engineering
    PatternID.P06: ThoughtType.HYPOTHESIS,  # Probabilistic = hypothesis
    PatternID.P07: ThoughtType.VALIDATION,  # Formal-experimental = validation
    PatternID.P08: ThoughtType.REFINEMENT,  # Approximation = refinement
    PatternID.P09: ThoughtType.REFINEMENT,  # Inference-time = refinement
    PatternID.P10: ThoughtType.REASONING,  # Inductive bias = reasoning
    PatternID.P11: ThoughtType.REASONING,  # Hierarchical = reasoning
    PatternID.P12: ThoughtType.EVIDENCE,  # Mechanistic = evidence
    PatternID.P13: ThoughtType.COUNTERPOINT,  # Adversary = counterpoint
    PatternID.P14: ThoughtType.REFINEMENT,  # Systems co-design = refinement
    PatternID.P15: ThoughtType.EVIDENCE,  # Data-centric = evidence
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThinkingPattern:
    """A single thinking pattern from the Sci-Reasoning taxonomy."""

    id: PatternID
    name: str
    category: str
    description: str
    cognitive_move: str
    key_indicators: tuple[str, ...] = ()
    example: str = ""
    learnable_insight: str = ""

    @property
    def thought_type(self) -> ThoughtType:
        """Map to BIZRA GoT ThoughtType."""
        return PATTERN_TO_THOUGHT_TYPE.get(self.id, ThoughtType.REASONING)

    def matches_text(self, text: str) -> float:
        """Score how well a text matches this pattern's indicators (0-1)."""
        if not text:
            return 0.0
        text_lower = text.lower()
        hits = sum(1 for kw in self.key_indicators if kw.lower() in text_lower)
        return min(1.0, hits / max(len(self.key_indicators), 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id.value,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "cognitive_move": self.cognitive_move,
            "key_indicators": list(self.key_indicators),
            "thought_type": self.thought_type.value,
        }


@dataclass(frozen=True)
class ClassifiedPaper:
    """A paper with its thinking pattern classification."""

    title: str
    conference: str
    year: int
    presentation_type: str
    primary_pattern: PatternID
    secondary_patterns: tuple[PatternID, ...] = ()
    confidence: str = "high"
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "conference": self.conference,
            "year": self.year,
            "presentation_type": self.presentation_type,
            "primary_pattern": self.primary_pattern.value,
            "secondary_patterns": [p.value for p in self.secondary_patterns],
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass(frozen=True)
class PriorWork:
    """An intellectual predecessor of a paper."""

    title: str
    authors: str
    year: int
    role: PredecessorRole
    relationship_sentence: str
    arxiv_id: Optional[str] = None

    @property
    def edge_type(self) -> EdgeType:
        """Map to BIZRA GoT EdgeType."""
        return ROLE_TO_EDGE_TYPE.get(self.role, EdgeType.DERIVES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "role": self.role.value,
            "edge_type": self.edge_type.value,
            "relationship_sentence": self.relationship_sentence,
            "arxiv_id": self.arxiv_id,
        }


@dataclass
class PaperLineage:
    """Complete intellectual lineage for a paper."""

    title: str
    conference: str
    year: int
    prior_works: list[PriorWork] = field(default_factory=list)
    synthesis_narrative: str = ""
    primary_pattern: Optional[PatternID] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "conference": self.conference,
            "year": self.year,
            "prior_works": [pw.to_dict() for pw in self.prior_works],
            "synthesis_narrative": self.synthesis_narrative[:500],
            "primary_pattern": (
                self.primary_pattern.value if self.primary_pattern else None
            ),
        }


# ---------------------------------------------------------------------------
# Pattern Taxonomy Loader
# ---------------------------------------------------------------------------


class PatternTaxonomy:
    """Loads and queries the 15-pattern taxonomy.

    Standing on: Li et al. (2025) — Sci-Reasoning structured innovation patterns.
    """

    def __init__(self, taxonomy_path: Optional[Path] = None):
        self._path = taxonomy_path or (DATA_DIR / "pattern_taxonomy.json")
        self._patterns: dict[PatternID, ThinkingPattern] = {}
        self._cooccurrence: dict[str, dict[str, int]] = {}
        self._loaded = False

    def load(self) -> None:
        """Load pattern taxonomy from JSON."""
        if not self._path.exists():
            logger.warning(f"Pattern taxonomy not found: {self._path}")
            return

        with open(self._path) as f:
            data = json.load(f)

        for entry in data.get("taxonomy", []):
            pid = PatternID(entry["id"])
            self._patterns[pid] = ThinkingPattern(
                id=pid,
                name=entry["name"],
                category=entry["category"],
                description=entry["description"],
                cognitive_move=entry["cognitive_move"],
                key_indicators=tuple(entry.get("key_indicators", [])),
                example=entry.get("example", ""),
                learnable_insight=entry.get("learnable_insight", ""),
            )

        # Load co-occurrence if available
        analysis_path = self._path.parent / "analysis_results.json"
        if analysis_path.exists():
            with open(analysis_path) as f:
                analysis = json.load(f)
            self._cooccurrence = analysis.get("cooccurrence", {})

        self._loaded = True
        logger.info(f"Loaded {len(self._patterns)} thinking patterns")

    def ensure_loaded(self) -> None:
        """Load if not already loaded."""
        if not self._loaded:
            self.load()

    def get(self, pattern_id: PatternID) -> Optional[ThinkingPattern]:
        """Get a specific pattern."""
        self.ensure_loaded()
        return self._patterns.get(pattern_id)

    def all_patterns(self) -> list[ThinkingPattern]:
        """Get all 15 patterns."""
        self.ensure_loaded()
        return list(self._patterns.values())

    def resolve_pattern(self, text: str) -> list[tuple[ThinkingPattern, float]]:
        """Score text against all patterns and return ranked matches.

        Returns list of (pattern, score) sorted by score descending.
        """
        self.ensure_loaded()
        scored = []
        for pattern in self._patterns.values():
            score = pattern.matches_text(text)
            if score > 0:
                scored.append((pattern, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def cooccurrence(self, pattern_id: PatternID) -> dict[str, int]:
        """Get co-occurrence counts for a pattern."""
        self.ensure_loaded()
        return self._cooccurrence.get(pattern_id.value, {})

    def suggest_complementary(
        self, pattern_id: PatternID, top_k: int = 3
    ) -> list[tuple[PatternID, int]]:
        """Suggest complementary patterns based on co-occurrence.

        Standing on: Shannon — high co-occurrence = high mutual information.
        """
        self.ensure_loaded()
        cooc = self.cooccurrence(pattern_id)
        ranked = sorted(cooc.items(), key=lambda x: x[1], reverse=True)
        result = []
        for pid_str, count in ranked[:top_k]:
            try:
                result.append((PatternID(pid_str), count))
            except ValueError:
                continue
        return result

    def get_statistics(self) -> dict[str, Any]:
        """Get taxonomy statistics."""
        self.ensure_loaded()
        return {
            "total_patterns": len(self._patterns),
            "categories": list({p.category for p in self._patterns.values()}),
            "has_cooccurrence": bool(self._cooccurrence),
            "loaded": self._loaded,
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_taxonomy: Optional[PatternTaxonomy] = None


def get_taxonomy() -> PatternTaxonomy:
    """Get the default pattern taxonomy (singleton)."""
    global _default_taxonomy
    if _default_taxonomy is None:
        _default_taxonomy = PatternTaxonomy()
    return _default_taxonomy


__all__ = [
    "PatternID",
    "PredecessorRole",
    "ThinkingPattern",
    "ClassifiedPaper",
    "PriorWork",
    "PaperLineage",
    "PatternTaxonomy",
    "ROLE_TO_EDGE_TYPE",
    "PATTERN_TO_THOUGHT_TYPE",
    "role_to_edge_type",
    "get_taxonomy",
]
