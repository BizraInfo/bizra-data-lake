"""
Sci-Reasoning Bridge — Intellectual Lineage Graphs for BIZRA
=============================================================

Loads the Sci-Reasoning dataset (3,819 papers, 15 thinking patterns,
intellectual lineage graphs) and converts into BIZRA's Graph-of-Thoughts
type system for use by the hypothesis generator, Spearpoint orchestrator,
and living memory.

Standing on Giants:
- Li et al. (2025): Sci-Reasoning structured innovation patterns
- Besta et al. (2024): Graph of Thoughts
- Shannon (1948): Signal-to-noise as information quality
- Lamport (1978): Happens-before ordering of intellectual lineage

Source: https://github.com/AmberLJC/Sci-Reasoning
Dataset: https://huggingface.co/datasets/AmberLJC/Sci-Reasoning
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from core.proof_engine.canonical import hex_digest
from core.sovereign.graph_types import ThoughtEdge, ThoughtNode, ThoughtType

from .sci_reasoning_patterns import (
    ClassifiedPaper,
    PaperLineage,
    PatternID,
    PatternTaxonomy,
    PredecessorRole,
    PriorWork,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sci_reasoning"


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------


def _safe_pattern_id(raw: str) -> Optional[PatternID]:
    """Parse a pattern ID string, returning None on failure."""
    try:
        return PatternID(raw)
    except ValueError:
        return None


def _safe_role(raw: str) -> PredecessorRole:
    """Parse a role string with normalization. Fail-closed: RELATED_PROBLEM."""
    normalized = raw.strip().title()
    for role in PredecessorRole:
        if role.value == normalized:
            return role
    return PredecessorRole.RELATED_PROBLEM


def load_classified_papers(
    path: Optional[Path] = None,
) -> list[ClassifiedPaper]:
    """Load classified papers from JSON.

    Returns up to 3,291 papers with their thinking pattern classifications.
    """
    fpath = path or (DATA_DIR / "classified_papers.json")
    if not fpath.exists():
        logger.warning(f"Classified papers not found: {fpath}")
        return []

    with open(fpath) as f:
        raw = json.load(f)

    papers = []
    for entry in raw:
        cls = entry.get("classification", {})
        primary = _safe_pattern_id(cls.get("primary_pattern", ""))
        if primary is None:
            continue

        secondary = tuple(
            p
            for s in cls.get("secondary_patterns", [])
            if (p := _safe_pattern_id(s)) is not None
        )

        papers.append(
            ClassifiedPaper(
                title=entry.get("title", ""),
                conference=entry.get("conference", ""),
                year=entry.get("year", 0),
                presentation_type=entry.get("presentation_type", ""),
                primary_pattern=primary,
                secondary_patterns=secondary,
                confidence=cls.get("confidence", "low"),
                reasoning=cls.get("reasoning", ""),
            )
        )

    logger.info(f"Loaded {len(papers)} classified papers")
    return papers


def load_prior_works(
    base_dir: Optional[Path] = None,
) -> dict[str, PaperLineage]:
    """Load prior work analyses from per-paper JSON files.

    Returns a dict mapping paper filename (OpenReview ID) to PaperLineage.
    """
    base = base_dir or (DATA_DIR / "prior_works")
    if not base.exists():
        logger.warning(f"Prior works directory not found: {base}")
        return {}

    lineages: dict[str, PaperLineage] = {}

    for conf_dir in sorted(base.iterdir()):
        if not conf_dir.is_dir():
            continue
        conference = conf_dir.name  # e.g., "ICLR_2024"

        for json_file in conf_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            target = data.get("target_paper", {})
            prior_works_raw = data.get("prior_works", [])

            priors = []
            for pw in prior_works_raw:
                priors.append(
                    PriorWork(
                        title=pw.get("title", ""),
                        authors=pw.get("authors", ""),
                        year=pw.get("year", 0),
                        role=_safe_role(pw.get("role", "Related Problem")),
                        relationship_sentence=pw.get("relationship_sentence", ""),
                        arxiv_id=pw.get("arxiv_id"),
                    )
                )

            lineage = PaperLineage(
                title=target.get("title", json_file.stem),
                conference=target.get("conference", conference.split("_")[0]),
                year=target.get("year", 0),
                prior_works=priors,
                synthesis_narrative=data.get("synthesis_narrative", ""),
            )

            lineages[json_file.stem] = lineage

    logger.info(f"Loaded {len(lineages)} paper lineages")
    return lineages


# ---------------------------------------------------------------------------
# Graph Conversion — Paper Lineage → GoT ThoughtNodes + ThoughtEdges
# ---------------------------------------------------------------------------


def _paper_node_id(title: str) -> str:
    """Deterministic node ID from paper title (BLAKE3 content-addressed)."""
    return f"paper_{hex_digest(title.encode('utf-8'))[:16]}"


def lineage_to_graph(
    lineage: PaperLineage,
) -> tuple[list[ThoughtNode], list[ThoughtEdge]]:
    """Convert a PaperLineage into GoT ThoughtNodes and ThoughtEdges.

    The target paper becomes a SYNTHESIS node (it synthesized prior works).
    Each predecessor becomes a node typed by its role.
    Edges encode the intellectual relationship.
    """
    nodes: list[ThoughtNode] = []
    edges: list[ThoughtEdge] = []

    # Target paper as synthesis node
    target_id = _paper_node_id(lineage.title)
    target_node = ThoughtNode(
        id=target_id,
        content=f"[{lineage.conference} {lineage.year}] {lineage.title}",
        thought_type=ThoughtType.SYNTHESIS,
        confidence=0.95,
        snr_score=0.95,
        depth=0,
        metadata={
            "source": "sci_reasoning",
            "conference": lineage.conference,
            "year": lineage.year,
            "primary_pattern": (
                lineage.primary_pattern.value if lineage.primary_pattern else None
            ),
        },
    )
    nodes.append(target_node)

    # Predecessor nodes
    for pw in lineage.prior_works:
        pw_id = _paper_node_id(pw.title)

        # Map predecessor role to thought type
        role_thought_map = {
            PredecessorRole.BASELINE: ThoughtType.EVIDENCE,
            PredecessorRole.INSPIRATION: ThoughtType.HYPOTHESIS,
            PredecessorRole.GAP_IDENTIFICATION: ThoughtType.QUESTION,
            PredecessorRole.FOUNDATION: ThoughtType.EVIDENCE,
            PredecessorRole.EXTENSION: ThoughtType.REFINEMENT,
            PredecessorRole.RELATED_PROBLEM: ThoughtType.REASONING,
        }

        pw_node = ThoughtNode(
            id=pw_id,
            content=f"[{pw.year}] {pw.title}",
            thought_type=role_thought_map.get(pw.role, ThoughtType.REASONING),
            confidence=0.85,
            snr_score=0.90,
            depth=1,
            metadata={
                "source": "sci_reasoning",
                "role": pw.role.value,
                "authors": pw.authors,
                "arxiv_id": pw.arxiv_id,
            },
        )
        nodes.append(pw_node)

        # Edge: predecessor → target (the intellectual contribution flows forward)
        edge = ThoughtEdge(
            source_id=pw_id,
            target_id=target_id,
            edge_type=pw.edge_type,
            weight=0.9,
            reasoning=pw.relationship_sentence,
        )
        edges.append(edge)

    return nodes, edges


# ---------------------------------------------------------------------------
# Main Bridge Class
# ---------------------------------------------------------------------------


class SciReasoningBridge:
    """Bridge between Sci-Reasoning dataset and BIZRA's reasoning engine.

    Provides:
    1. Pattern taxonomy queries (15 thinking patterns)
    2. Paper classification lookups (3,291 papers)
    3. Intellectual lineage graph conversion (prior works → GoT)
    4. Pattern-based hypothesis seeding for Spearpoint

    Usage:
        bridge = SciReasoningBridge()
        bridge.load()

        # Query patterns
        patterns = bridge.taxonomy.resolve_pattern("we reframe the problem as...")

        # Get paper lineage as GoT graph
        nodes, edges = bridge.get_lineage_graph("ICLR_2024", "06lrITXVAx")

        # Seed hypotheses from pattern
        seeds = bridge.seed_hypotheses(PatternID.P01, top_k=5)
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self._data_dir = data_dir or DATA_DIR
        self.taxonomy = PatternTaxonomy(self._data_dir / "pattern_taxonomy.json")
        self._papers: list[ClassifiedPaper] = []
        self._lineages: dict[str, PaperLineage] = {}
        self._loaded = False

    def load(self) -> None:
        """Load all Sci-Reasoning data."""
        self.taxonomy.load()
        self._papers = load_classified_papers(self._data_dir / "classified_papers.json")
        self._lineages = load_prior_works(self._data_dir / "prior_works")
        self._loaded = True
        logger.info(
            f"SciReasoningBridge loaded: {len(self._papers)} papers, "
            f"{len(self._lineages)} lineages, "
            f"{self.taxonomy.get_statistics()['total_patterns']} patterns"
        )

    def ensure_loaded(self) -> None:
        """Load if not already loaded."""
        if not self._loaded:
            self.load()

    # --- Paper queries ---

    def papers_by_pattern(
        self,
        pattern_id: PatternID,
        conference: Optional[str] = None,
        min_confidence: str = "low",
    ) -> list[ClassifiedPaper]:
        """Get papers classified under a specific thinking pattern."""
        self.ensure_loaded()
        confidence_order = {"high": 3, "medium": 2, "low": 1}
        min_level = confidence_order.get(min_confidence, 0)

        results = []
        for paper in self._papers:
            if paper.primary_pattern != pattern_id:
                continue
            if conference and paper.conference != conference:
                continue
            if confidence_order.get(paper.confidence, 0) < min_level:
                continue
            results.append(paper)
        return results

    def papers_by_conference(
        self,
        conference: str,
        year: Optional[int] = None,
    ) -> list[ClassifiedPaper]:
        """Get papers from a specific conference."""
        self.ensure_loaded()
        return [
            p
            for p in self._papers
            if p.conference == conference and (year is None or p.year == year)
        ]

    # --- Lineage queries ---

    def get_lineage(self, paper_id: str) -> Optional[PaperLineage]:
        """Get the intellectual lineage for a paper by its OpenReview ID."""
        self.ensure_loaded()
        return self._lineages.get(paper_id)

    def get_lineage_graph(
        self,
        paper_id: str,
    ) -> Optional[tuple[list[ThoughtNode], list[ThoughtEdge]]]:
        """Get a paper's lineage as GoT ThoughtNodes and ThoughtEdges."""
        lineage = self.get_lineage(paper_id)
        if lineage is None:
            return None
        return lineage_to_graph(lineage)

    # --- Hypothesis seeding ---

    def seed_hypotheses(
        self,
        pattern_id: PatternID,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate hypothesis seeds based on a thinking pattern.

        Returns structured hypothesis data suitable for the autopoiesis
        hypothesis generator or Spearpoint researcher.

        Standing on: Boyd (OODA) — Orient phase uses patterns to frame hypotheses.
        """
        self.ensure_loaded()
        pattern = self.taxonomy.get(pattern_id)
        if pattern is None:
            return []

        # Get exemplar papers for this pattern
        exemplars = self.papers_by_pattern(pattern_id, min_confidence="high")[:top_k]

        # Get complementary patterns
        complementary = self.taxonomy.suggest_complementary(pattern_id, top_k=3)

        seeds = []
        for paper in exemplars:
            seeds.append(
                {
                    "hypothesis_type": "pattern_exemplar",
                    "pattern_id": pattern_id.value,
                    "pattern_name": pattern.name,
                    "cognitive_move": pattern.cognitive_move,
                    "exemplar_title": paper.title,
                    "exemplar_conference": paper.conference,
                    "exemplar_year": paper.year,
                    "exemplar_reasoning": paper.reasoning,
                    "complementary_patterns": [
                        {"id": pid.value, "cooccurrence": count}
                        for pid, count in complementary
                    ],
                    "learnable_insight": pattern.learnable_insight,
                }
            )

        return seeds

    def pattern_distribution(
        self,
        conference: Optional[str] = None,
        year: Optional[int] = None,
    ) -> dict[str, int]:
        """Get pattern distribution across papers, optionally filtered."""
        self.ensure_loaded()
        counts: dict[str, int] = {}
        for paper in self._papers:
            if conference and paper.conference != conference:
                continue
            if year and paper.year != year:
                continue
            pid = paper.primary_pattern.value
            counts[pid] = counts.get(pid, 0) + 1
        return counts

    # --- Statistics ---

    def get_statistics(self) -> dict[str, Any]:
        """Get bridge statistics."""
        self.ensure_loaded()
        return {
            "total_papers": len(self._papers),
            "total_lineages": len(self._lineages),
            "taxonomy": self.taxonomy.get_statistics(),
            "conferences": sorted({p.conference for p in self._papers}),
            "years": sorted({p.year for p in self._papers}),
            "pattern_distribution": self.pattern_distribution(),
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_bridge: Optional[SciReasoningBridge] = None


def get_bridge() -> SciReasoningBridge:
    """Get the default Sci-Reasoning bridge (singleton)."""
    global _default_bridge
    if _default_bridge is None:
        _default_bridge = SciReasoningBridge()
    return _default_bridge


__all__ = [
    "SciReasoningBridge",
    "load_classified_papers",
    "load_prior_works",
    "lineage_to_graph",
    "get_bridge",
]
