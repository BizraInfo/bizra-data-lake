"""
Sci-Reasoning Bridge Integration Tests
=======================================

Tests the full integration between the Sci-Reasoning dataset
(3,819 ML papers, 15 thinking patterns, intellectual lineage graphs)
and BIZRA's Graph-of-Thoughts type system.

Standing on Giants: Li et al. (2025), Besta et al. (2024)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.bridges.sci_reasoning_patterns import (
    ClassifiedPaper,
    PaperLineage,
    PatternID,
    PatternTaxonomy,
    PredecessorRole,
    PriorWork,
    ThinkingPattern,
    role_to_edge_type,
    PATTERN_TO_THOUGHT_TYPE,
    ROLE_TO_EDGE_TYPE,
)
from core.sovereign.graph_types import EdgeType, ThoughtType

# Data directory
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "sci_reasoning"
HAS_DATA = (DATA_DIR / "pattern_taxonomy.json").exists()


# ---------------------------------------------------------------------------
# Unit Tests — Pattern Types
# ---------------------------------------------------------------------------


class TestPatternID:
    """PatternID enum covers all 15 patterns."""

    def test_count(self):
        assert len(PatternID) == 15

    def test_ids_sequential(self):
        for i in range(1, 16):
            assert PatternID(f"P{i:02d}") is not None

    def test_p01_is_gap_driven(self):
        assert PatternID.P01.value == "P01"

    def test_p15_is_data_centric(self):
        assert PatternID.P15.value == "P15"


class TestPredecessorRole:
    """PredecessorRole enum covers all 6 roles."""

    def test_count(self):
        assert len(PredecessorRole) == 6

    def test_baseline_exists(self):
        assert PredecessorRole.BASELINE.value == "Baseline"

    def test_gap_identification_exists(self):
        assert PredecessorRole.GAP_IDENTIFICATION.value == "Gap Identification"


class TestRoleToEdgeType:
    """Every predecessor role maps to a GoT EdgeType."""

    def test_all_roles_have_mapping(self):
        for role in PredecessorRole:
            assert role in ROLE_TO_EDGE_TYPE

    def test_baseline_maps_to_refines(self):
        assert ROLE_TO_EDGE_TYPE[PredecessorRole.BASELINE] == EdgeType.REFINES

    def test_foundation_maps_to_supports(self):
        assert ROLE_TO_EDGE_TYPE[PredecessorRole.FOUNDATION] == EdgeType.SUPPORTS

    def test_gap_maps_to_questions(self):
        assert ROLE_TO_EDGE_TYPE[PredecessorRole.GAP_IDENTIFICATION] == EdgeType.QUESTIONS

    def test_inspiration_maps_to_derives(self):
        assert ROLE_TO_EDGE_TYPE[PredecessorRole.INSPIRATION] == EdgeType.DERIVES

    def test_role_to_edge_type_function(self):
        assert role_to_edge_type("Baseline") == EdgeType.REFINES
        assert role_to_edge_type("Foundation") == EdgeType.SUPPORTS

    def test_unknown_role_fails_closed(self):
        """Unknown roles map to DERIVES (fail-closed, weakest assumption)."""
        assert role_to_edge_type("Unknown Role") == EdgeType.DERIVES


class TestPatternToThoughtType:
    """Every pattern maps to a GoT ThoughtType."""

    def test_all_patterns_have_mapping(self):
        for pid in PatternID:
            assert pid in PATTERN_TO_THOUGHT_TYPE

    def test_p01_gap_is_question(self):
        assert PATTERN_TO_THOUGHT_TYPE[PatternID.P01] == ThoughtType.QUESTION

    def test_p02_synthesis_is_synthesis(self):
        assert PATTERN_TO_THOUGHT_TYPE[PatternID.P02] == ThoughtType.SYNTHESIS

    def test_p13_adversary_is_counterpoint(self):
        assert PATTERN_TO_THOUGHT_TYPE[PatternID.P13] == ThoughtType.COUNTERPOINT


# ---------------------------------------------------------------------------
# Unit Tests — Data Classes
# ---------------------------------------------------------------------------


class TestThinkingPattern:
    """ThinkingPattern data class."""

    def test_creation(self):
        p = ThinkingPattern(
            id=PatternID.P01,
            name="Gap-Driven Reframing",
            category="Problem Diagnosis & Reframing",
            description="Reframe the problem...",
            cognitive_move="Turn a specific failure into a constraint",
            key_indicators=("limitation", "gap", "reframed as"),
        )
        assert p.id == PatternID.P01
        assert p.thought_type == ThoughtType.QUESTION

    def test_matches_text_positive(self):
        p = ThinkingPattern(
            id=PatternID.P01,
            name="Gap-Driven Reframing",
            category="test",
            description="test",
            cognitive_move="test",
            key_indicators=("limitation", "gap", "reframed as"),
        )
        score = p.matches_text("We identified a gap in the existing limitation")
        assert score > 0

    def test_matches_text_empty(self):
        p = ThinkingPattern(
            id=PatternID.P01, name="t", category="t",
            description="t", cognitive_move="t",
            key_indicators=("x", "y"),
        )
        assert p.matches_text("") == 0.0

    def test_matches_text_no_match(self):
        p = ThinkingPattern(
            id=PatternID.P01, name="t", category="t",
            description="t", cognitive_move="t",
            key_indicators=("quantum", "photon"),
        )
        assert p.matches_text("unrelated text about cooking") == 0.0

    def test_to_dict(self):
        p = ThinkingPattern(
            id=PatternID.P02,
            name="Cross-Domain Synthesis",
            category="Synthesis & Transfer",
            description="Combine ideas from distinct fields",
            cognitive_move="Map components across boundaries",
            key_indicators=("borrow from", "combine"),
        )
        d = p.to_dict()
        assert d["id"] == "P02"
        assert d["thought_type"] == "synthesis"


class TestClassifiedPaper:

    def test_creation(self):
        p = ClassifiedPaper(
            title="Test Paper",
            conference="ICLR",
            year=2024,
            presentation_type="oral",
            primary_pattern=PatternID.P01,
            secondary_patterns=(PatternID.P03, PatternID.P08),
            confidence="high",
        )
        assert p.primary_pattern == PatternID.P01
        assert len(p.secondary_patterns) == 2

    def test_to_dict(self):
        p = ClassifiedPaper(
            title="Test", conference="NeurIPS", year=2024,
            presentation_type="spotlight", primary_pattern=PatternID.P02,
        )
        d = p.to_dict()
        assert d["primary_pattern"] == "P02"
        assert d["secondary_patterns"] == []


class TestPriorWork:

    def test_edge_type_property(self):
        pw = PriorWork(
            title="Prior Paper", authors="A. Author",
            year=2020, role=PredecessorRole.FOUNDATION,
            relationship_sentence="Core framework",
        )
        assert pw.edge_type == EdgeType.SUPPORTS

    def test_to_dict(self):
        pw = PriorWork(
            title="Prior", authors="Auth", year=2019,
            role=PredecessorRole.GAP_IDENTIFICATION,
            relationship_sentence="Addresses limitation",
            arxiv_id="1912.12345",
        )
        d = pw.to_dict()
        assert d["role"] == "Gap Identification"
        assert d["edge_type"] == "questions"
        assert d["arxiv_id"] == "1912.12345"


class TestPaperLineage:

    def test_to_dict(self):
        lineage = PaperLineage(
            title="Target Paper", conference="ICML", year=2024,
            prior_works=[
                PriorWork(
                    title="Prior 1", authors="A", year=2020,
                    role=PredecessorRole.BASELINE,
                    relationship_sentence="Main baseline",
                ),
            ],
            synthesis_narrative="A narrative about intellectual evolution.",
        )
        d = lineage.to_dict()
        assert len(d["prior_works"]) == 1
        assert d["prior_works"][0]["role"] == "Baseline"


# ---------------------------------------------------------------------------
# Integration Tests — Taxonomy Loading
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_DATA, reason="Sci-Reasoning data not available")
class TestPatternTaxonomyLoading:
    """Tests that require the actual data files."""

    def test_load_taxonomy(self):
        tax = PatternTaxonomy(DATA_DIR / "pattern_taxonomy.json")
        tax.load()
        assert len(tax.all_patterns()) == 15

    def test_get_pattern_by_id(self):
        tax = PatternTaxonomy(DATA_DIR / "pattern_taxonomy.json")
        tax.load()
        p01 = tax.get(PatternID.P01)
        assert p01 is not None
        assert "Gap" in p01.name

    def test_resolve_pattern(self):
        tax = PatternTaxonomy(DATA_DIR / "pattern_taxonomy.json")
        tax.load()
        matches = tax.resolve_pattern("We identified a gap and reframed the limitation")
        assert len(matches) > 0
        # P01 (Gap-Driven) should rank high
        top_pattern = matches[0][0]
        assert top_pattern.id == PatternID.P01

    def test_cooccurrence_loaded(self):
        tax = PatternTaxonomy(DATA_DIR / "pattern_taxonomy.json")
        tax.load()
        cooc = tax.cooccurrence(PatternID.P01)
        assert len(cooc) > 0

    def test_suggest_complementary(self):
        tax = PatternTaxonomy(DATA_DIR / "pattern_taxonomy.json")
        tax.load()
        suggestions = tax.suggest_complementary(PatternID.P01, top_k=3)
        assert len(suggestions) == 3
        # Each suggestion is (PatternID, count)
        for pid, count in suggestions:
            assert isinstance(pid, PatternID)
            assert count > 0

    def test_statistics(self):
        tax = PatternTaxonomy(DATA_DIR / "pattern_taxonomy.json")
        tax.load()
        stats = tax.get_statistics()
        assert stats["total_patterns"] == 15
        assert stats["loaded"] is True
        assert len(stats["categories"]) > 0


# ---------------------------------------------------------------------------
# Integration Tests — Data Bridge
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_DATA, reason="Sci-Reasoning data not available")
class TestSciReasoningBridge:
    """Full bridge integration tests."""

    def test_load(self):
        from core.bridges.sci_reasoning_bridge import SciReasoningBridge
        bridge = SciReasoningBridge(DATA_DIR)
        bridge.load()
        stats = bridge.get_statistics()
        assert stats["total_papers"] > 0
        assert stats["total_lineages"] > 0

    def test_papers_by_pattern(self):
        from core.bridges.sci_reasoning_bridge import SciReasoningBridge
        bridge = SciReasoningBridge(DATA_DIR)
        bridge.load()
        papers = bridge.papers_by_pattern(PatternID.P01)
        assert len(papers) > 0
        assert all(p.primary_pattern == PatternID.P01 for p in papers)

    def test_papers_by_conference(self):
        from core.bridges.sci_reasoning_bridge import SciReasoningBridge
        bridge = SciReasoningBridge(DATA_DIR)
        bridge.load()
        papers = bridge.papers_by_conference("ICLR")
        assert len(papers) > 0
        assert all(p.conference == "ICLR" for p in papers)

    def test_pattern_distribution(self):
        from core.bridges.sci_reasoning_bridge import SciReasoningBridge
        bridge = SciReasoningBridge(DATA_DIR)
        bridge.load()
        dist = bridge.pattern_distribution()
        assert "P01" in dist
        assert dist["P01"] > 0

    def test_seed_hypotheses(self):
        from core.bridges.sci_reasoning_bridge import SciReasoningBridge
        bridge = SciReasoningBridge(DATA_DIR)
        bridge.load()
        seeds = bridge.seed_hypotheses(PatternID.P01, top_k=3)
        assert len(seeds) > 0
        assert seeds[0]["pattern_id"] == "P01"
        assert "cognitive_move" in seeds[0]
        assert "complementary_patterns" in seeds[0]


@pytest.mark.skipif(not HAS_DATA, reason="Sci-Reasoning data not available")
class TestLineageGraphConversion:
    """Tests for GoT graph conversion."""

    def test_lineage_to_graph(self):
        from core.bridges.sci_reasoning_bridge import SciReasoningBridge
        bridge = SciReasoningBridge(DATA_DIR)
        bridge.load()

        # Get a lineage with data
        lineages = bridge._lineages
        if not lineages:
            pytest.skip("No lineage data available")

        paper_id = next(iter(lineages))
        result = bridge.get_lineage_graph(paper_id)
        assert result is not None

        nodes, edges = result
        assert len(nodes) > 0
        assert len(edges) > 0

        # Target node is SYNTHESIS
        target = nodes[0]
        assert target.thought_type == ThoughtType.SYNTHESIS
        assert target.metadata["source"] == "sci_reasoning"

        # All edges point to target
        for edge in edges:
            assert edge.target_id == target.id
            assert isinstance(edge.edge_type, EdgeType)
            assert len(edge.reasoning) > 0

    def test_graph_nodes_are_content_addressed(self):
        """Verify BLAKE3 content hashing on graph nodes (SEC-001)."""
        from core.bridges.sci_reasoning_bridge import SciReasoningBridge
        bridge = SciReasoningBridge(DATA_DIR)
        bridge.load()

        lineages = bridge._lineages
        if not lineages:
            pytest.skip("No lineage data available")

        paper_id = next(iter(lineages))
        nodes, _ = bridge.get_lineage_graph(paper_id)

        for node in nodes:
            h = node.content_hash
            assert len(h) == 64  # BLAKE3 hex digest
            assert all(c in "0123456789abcdef" for c in h)

    def test_nonexistent_lineage_returns_none(self):
        from core.bridges.sci_reasoning_bridge import SciReasoningBridge
        bridge = SciReasoningBridge(DATA_DIR)
        bridge.load()
        assert bridge.get_lineage_graph("nonexistent_paper_id") is None


# ---------------------------------------------------------------------------
# Unit Tests — Data Loaders (synthetic data)
# ---------------------------------------------------------------------------


class TestLoadClassifiedPapersSynthetic:
    """Test paper loading with synthetic data (no data files needed)."""

    def test_load_from_synthetic(self, tmp_path):
        from core.bridges.sci_reasoning_bridge import load_classified_papers

        data = [
            {
                "title": "Test Paper 1",
                "conference": "ICLR",
                "year": 2024,
                "presentation_type": "oral",
                "classification": {
                    "paper_index": 1,
                    "primary_pattern": "P01",
                    "secondary_patterns": ["P03"],
                    "confidence": "high",
                    "reasoning": "Test reasoning",
                },
            }
        ]
        fpath = tmp_path / "classified.json"
        fpath.write_text(json.dumps(data))

        papers = load_classified_papers(fpath)
        assert len(papers) == 1
        assert papers[0].primary_pattern == PatternID.P01
        assert papers[0].secondary_patterns == (PatternID.P03,)

    def test_invalid_pattern_id_skipped(self, tmp_path):
        from core.bridges.sci_reasoning_bridge import load_classified_papers

        data = [
            {
                "title": "Bad Paper",
                "classification": {"primary_pattern": "INVALID"},
            }
        ]
        fpath = tmp_path / "bad.json"
        fpath.write_text(json.dumps(data))

        papers = load_classified_papers(fpath)
        assert len(papers) == 0  # Skipped invalid

    def test_missing_file_returns_empty(self, tmp_path):
        from core.bridges.sci_reasoning_bridge import load_classified_papers
        papers = load_classified_papers(tmp_path / "nonexistent.json")
        assert papers == []


class TestLoadPriorWorksSynthetic:
    """Test prior work loading with synthetic data."""

    def test_load_from_synthetic(self, tmp_path):
        from core.bridges.sci_reasoning_bridge import load_prior_works

        conf_dir = tmp_path / "ICLR_2024"
        conf_dir.mkdir()

        data = {
            "target_paper": {
                "title": "Target Paper",
                "conference": "ICLR",
                "year": 2024,
            },
            "prior_works": [
                {
                    "title": "Prior 1",
                    "authors": "Author A",
                    "year": 2020,
                    "role": "Foundation",
                    "relationship_sentence": "Core framework",
                    "arxiv_id": "2020.12345",
                }
            ],
            "synthesis_narrative": "A synthesis narrative.",
        }
        (conf_dir / "test_paper.json").write_text(json.dumps(data))

        lineages = load_prior_works(tmp_path)
        assert "test_paper" in lineages
        lineage = lineages["test_paper"]
        assert lineage.title == "Target Paper"
        assert len(lineage.prior_works) == 1
        assert lineage.prior_works[0].role == PredecessorRole.FOUNDATION

    def test_lineage_to_graph_synthetic(self, tmp_path):
        from core.bridges.sci_reasoning_bridge import lineage_to_graph

        lineage = PaperLineage(
            title="Synthetic Target",
            conference="NeurIPS",
            year=2025,
            prior_works=[
                PriorWork(
                    title="Foundation Paper",
                    authors="Auth A",
                    year=2020,
                    role=PredecessorRole.FOUNDATION,
                    relationship_sentence="Introduced the framework",
                ),
                PriorWork(
                    title="Inspiration Paper",
                    authors="Auth B",
                    year=2022,
                    role=PredecessorRole.INSPIRATION,
                    relationship_sentence="Sparked the key idea",
                ),
            ],
            synthesis_narrative="Test narrative.",
        )

        nodes, edges = lineage_to_graph(lineage)
        assert len(nodes) == 3  # 1 target + 2 priors
        assert len(edges) == 2

        # Target is SYNTHESIS
        assert nodes[0].thought_type == ThoughtType.SYNTHESIS

        # Foundation → SUPPORTS edge
        foundation_edge = [e for e in edges if "Introduced" in e.reasoning][0]
        assert foundation_edge.edge_type == EdgeType.SUPPORTS

        # Inspiration → DERIVES edge
        inspiration_edge = [e for e in edges if "Sparked" in e.reasoning][0]
        assert inspiration_edge.edge_type == EdgeType.DERIVES
