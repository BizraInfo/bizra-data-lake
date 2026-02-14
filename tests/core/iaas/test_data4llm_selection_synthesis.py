"""
Tests for DATA4LLM Selection and Synthesis Engines

Validates:
- SimilaritySelector: Lexicon, BoW, Cosine selection
- OptimizationSelector: DataModels, LESS, Influence selection
- ModelBasedSelector: Quality, Relevance, DEITA scoring
- DataSelectionPipeline: Ensemble selection with Ihsān constraint
- RephrasingSynthesizer: Multi-style corpus generation
- InstructionSynthesizer: QA pair generation
- ReasoningSynthesizer: Chain-of-thought synthesis
- AgenticSynthesizer: Tool-use trajectory generation
- DomainSynthesizer: GLAN-style domain content
- DataSynthesisPipeline: Multi-strategy synthesis
"""

import pytest
import numpy as np
from typing import List, Dict

# Import selection components
from core.iaas.selection import (
    SimilaritySelector,
    OptimizationSelector,
    ModelBasedSelector,
    DataSelectionPipeline,
    SelectionResult,
)

# Import synthesis components
from core.iaas.synthesis import (
    RephrasingSynthesizer,
    InstructionSynthesizer,
    ReasoningSynthesizer,
    AgenticSynthesizer,
    DomainSynthesizer,
    DataSynthesisPipeline,
    SynthesisStrategy,
    SynthesisResult,
)


# ============================================================================
# SELECTION TESTS
# ============================================================================

class TestSimilaritySelector:
    """Tests for similarity-based data selection."""

    @pytest.fixture
    def sample_texts(self) -> List[str]:
        return [
            "Machine learning enables computers to learn from data.",
            "Deep neural networks power modern AI systems.",
            "Natural language processing helps understand text.",
            "Computer vision recognizes images and objects.",
            "Reinforcement learning optimizes through rewards.",
        ]

    @pytest.fixture
    def target_domain(self) -> str:
        return "Machine learning and artificial intelligence research."

    def test_lexicon_overlap_selection(self, sample_texts, target_domain):
        """Test lexicon overlap similarity selection."""
        selector = SimilaritySelector(
            similarity_metric="lexicon",
            threshold=0.05,  # Low threshold for testing
        )

        result = selector.select(sample_texts, target_domain)

        assert isinstance(result, SelectionResult)
        assert result.original_count == 5
        assert result.selected_count > 0
        assert len(result.selected_indices) > 0
        assert result.method == "similarity_lexicon"

    def test_bow_similarity_selection(self, sample_texts, target_domain):
        """Test bag-of-words similarity selection."""
        selector = SimilaritySelector(
            similarity_metric="bow",
            threshold=0.01,
        )

        result = selector.select(sample_texts, target_domain)

        assert isinstance(result, SelectionResult)
        assert result.original_count == 5
        # BOW should find matches for ML-related texts
        assert len(result.scores) == 5

    def test_cosine_similarity_selection(self, sample_texts, target_domain):
        """Test cosine similarity selection."""
        selector = SimilaritySelector(
            similarity_metric="cosine",
            top_k=3,  # Select top 3
        )

        result = selector.select(sample_texts, target_domain)

        assert result.selected_count == 3
        assert len(result.selected_indices) == 3
        assert result.method == "similarity_cosine"

    def test_top_k_selection(self, sample_texts, target_domain):
        """Test top-k selection mode."""
        selector = SimilaritySelector(
            similarity_metric="lexicon",
            top_k=2,
        )

        result = selector.select(sample_texts, target_domain)

        assert result.selected_count == 2
        assert len(result.selected_indices) == 2


class TestOptimizationSelector:
    """Tests for optimization-based data selection."""

    @pytest.fixture
    def sample_texts(self) -> List[str]:
        return [
            "This is a comprehensive guide to machine learning algorithms.",
            "Short text.",
            "The detailed analysis of neural network architectures provides insights.",
            "x",
            "Optimization techniques for deep learning models include gradient descent.",
        ]

    def test_datamodels_selection(self, sample_texts):
        """Test DataModels-based selection."""
        selector = OptimizationSelector(
            method="datamodels",
            selection_ratio=0.4,
        )

        result = selector.select(sample_texts)

        assert isinstance(result, SelectionResult)
        assert result.original_count == 5
        assert result.selected_count == 2  # 40% of 5 = 2
        assert result.method == "optimization_datamodels"

    def test_less_projection_selection(self, sample_texts):
        """Test LESS-style projection selection."""
        selector = OptimizationSelector(
            method="less",
            selection_ratio=0.6,
        )

        result = selector.select(sample_texts)

        assert result.selected_count == 3  # 60% of 5 = 3
        assert result.method == "optimization_less"

    def test_influence_selection(self, sample_texts):
        """Test influence-based selection (fallback mode)."""
        selector = OptimizationSelector(
            method="influence",
            selection_ratio=0.4,
        )

        validation = ["Test validation sample for machine learning."]
        result = selector.select(sample_texts, validation_samples=validation)

        assert result.selected_count >= 1
        assert result.method == "optimization_influence"

    def test_longer_texts_preferred(self, sample_texts):
        """Test that longer, more informative texts score higher."""
        selector = OptimizationSelector(
            method="datamodels",
            selection_ratio=0.4,
        )

        result = selector.select(sample_texts)

        # Short/trivial texts should have lower scores
        assert result.scores[1] < result.scores[0]  # "Short text" < comprehensive guide
        assert result.scores[3] < result.scores[2]  # "x" < detailed analysis


class TestModelBasedSelector:
    """Tests for model-based quality/relevance selection."""

    @pytest.fixture
    def sample_texts(self) -> List[str]:
        return [
            "The implementation of transformer architectures revolutionized NLP. This breakthrough enables better understanding of language.",
            "bad txt no punct",
            "A comprehensive study of reinforcement learning algorithms demonstrates their effectiveness in game playing and robotics.",
            "",
            "Machine learning models require careful tuning of hyperparameters for optimal performance.",
        ]

    def test_quality_scoring(self, sample_texts):
        """Test quality-based selection."""
        selector = ModelBasedSelector(
            scoring_method="quality",
            threshold=0.3,
        )

        result = selector.select(sample_texts)

        assert isinstance(result, SelectionResult)
        # Well-formed texts should be selected
        assert result.selected_count >= 1
        # Empty text should have lowest score
        assert result.scores[3] == 0.0

    def test_relevance_scoring(self, sample_texts):
        """Test relevance-based selection with keywords."""
        selector = ModelBasedSelector(
            scoring_method="relevance",
            threshold=0.1,
        )

        keywords = {"machine", "learning", "transformer", "neural"}
        result = selector.select(sample_texts, domain_keywords=keywords)

        assert result.method == "model_relevance"
        # Texts mentioning keywords should score higher
        assert result.scores[0] > result.scores[3]

    def test_deita_scoring(self, sample_texts):
        """Test DEITA dual (complexity + quality) scoring."""
        selector = ModelBasedSelector(
            scoring_method="deita",
            threshold=0.2,
        )

        result = selector.select(sample_texts)

        assert result.method == "model_deita"
        # Well-structured texts should pass
        assert 0 in result.selected_indices or 2 in result.selected_indices


class TestDataSelectionPipeline:
    """Tests for unified selection pipeline."""

    @pytest.fixture
    def sample_texts(self) -> List[str]:
        return [
            "Machine learning algorithms process data to make predictions.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing enables text understanding.",
            "x",
            "Computer vision systems recognize visual patterns.",
        ]

    def test_ensemble_selection(self, sample_texts):
        """Test ensemble selection combining all methods."""
        pipeline = DataSelectionPipeline(ihsan_threshold=0.95)

        target = "Machine learning and AI research."
        result = pipeline.select(sample_texts, target=target, method="ensemble")

        assert isinstance(result, SelectionResult)
        assert result.method == "ensemble"
        # Should select at least some high-quality samples
        assert result.selected_count >= 1

    def test_similarity_only_selection(self, sample_texts):
        """Test pipeline with similarity-only method."""
        pipeline = DataSelectionPipeline()

        target = "Neural networks and deep learning."
        result = pipeline.select(sample_texts, target=target, method="similarity")

        assert result.method.startswith("similarity")

    def test_optimization_only_selection(self, sample_texts):
        """Test pipeline with optimization-only method."""
        pipeline = DataSelectionPipeline()

        result = pipeline.select(sample_texts, method="optimization")

        assert result.method.startswith("optimization")

    def test_model_only_selection(self, sample_texts):
        """Test pipeline with model-based method."""
        pipeline = DataSelectionPipeline()

        result = pipeline.select(sample_texts, method="model")

        assert result.method.startswith("model")


# ============================================================================
# SYNTHESIS TESTS
# ============================================================================

class TestRephrasingSynthesizer:
    """Tests for multi-style rephrasing synthesis."""

    @pytest.fixture
    def source_texts(self) -> List[str]:
        return [
            "Machine learning models learn patterns from data.",
            "Neural networks process information in layers.",
        ]

    def test_basic_rephrasing(self, source_texts):
        """Test basic rephrasing with default styles."""
        synthesizer = RephrasingSynthesizer()

        result = synthesizer.synthesize(source_texts)

        assert isinstance(result, SynthesisResult)
        assert result.original_count == 2
        # Each text gets original + rephrased versions
        assert result.total_count > 2
        assert result.strategy == "rephrasing"

    def test_style_diversity(self, source_texts):
        """Test that different styles produce different outputs."""
        synthesizer = RephrasingSynthesizer(
            styles=["formal", "conversational", "technical"]
        )

        result = synthesizer.synthesize(source_texts)

        # Should have original + 3 styles per text
        assert result.total_count == 2 + (2 * 3)

        # Check style diversity in samples
        sources = {s["source"] for s in result.samples}
        assert "original" in sources
        assert "rephrased_formal" in sources
        assert "rephrased_conversational" in sources
        assert "rephrased_technical" in sources

    def test_quality_scores(self, source_texts):
        """Test that quality scores are assigned."""
        synthesizer = RephrasingSynthesizer(styles=["formal"])

        result = synthesizer.synthesize(source_texts)

        for sample in result.samples:
            assert "quality_score" in sample
            assert 0 <= sample["quality_score"] <= 1


class TestInstructionSynthesizer:
    """Tests for instruction-response pair synthesis."""

    @pytest.fixture
    def source_texts(self) -> List[str]:
        return [
            "Transformers use attention mechanisms to process sequences.",
            "Gradient descent optimizes neural network parameters.",
        ]

    def test_qa_generation(self, source_texts):
        """Test QA pair generation from texts."""
        synthesizer = InstructionSynthesizer(qa_per_text=3)

        result = synthesizer.synthesize(source_texts)

        assert isinstance(result, SynthesisResult)
        assert result.synthesized_count >= 2  # At least 1 QA per text
        assert result.strategy == "instruction"

    def test_qa_structure(self, source_texts):
        """Test that generated QA has proper structure."""
        synthesizer = InstructionSynthesizer(qa_per_text=2)

        result = synthesizer.synthesize(source_texts)

        for sample in result.samples:
            assert "instruction" in sample
            assert "response" in sample
            assert "### Instruction:" in sample["text"]
            assert "### Response:" in sample["text"]


class TestReasoningSynthesizer:
    """Tests for chain-of-thought reasoning synthesis."""

    @pytest.fixture
    def problems(self) -> List[str]:
        return [
            "How does backpropagation work in neural networks?",
            "Explain the attention mechanism in transformers.",
        ]

    def test_reasoning_chain_generation(self, problems):
        """Test reasoning chain generation."""
        synthesizer = ReasoningSynthesizer(chain_length=3)

        result = synthesizer.synthesize(problems)

        assert isinstance(result, SynthesisResult)
        assert result.synthesized_count == 2
        assert result.strategy == "reasoning"

    def test_chain_structure(self, problems):
        """Test that reasoning chains have proper structure."""
        synthesizer = ReasoningSynthesizer()

        result = synthesizer.synthesize(problems)

        for sample in result.samples:
            assert "problem" in sample
            # Check for reasoning markers
            text = sample["text"]
            assert "step by step" in text.lower() or "think through" in text.lower()


class TestAgenticSynthesizer:
    """Tests for tool-use trajectory synthesis."""

    @pytest.fixture
    def tasks(self) -> List[str]:
        return [
            "Find information about machine learning libraries.",
            "Calculate the accuracy of a model.",
        ]

    def test_trajectory_generation(self, tasks):
        """Test multi-turn trajectory generation."""
        synthesizer = AgenticSynthesizer(max_turns=3)

        result = synthesizer.synthesize(tasks)

        assert isinstance(result, SynthesisResult)
        assert result.synthesized_count == 2
        assert result.strategy == "agentic"

    def test_trajectory_structure(self, tasks):
        """Test that trajectories have proper structure."""
        synthesizer = AgenticSynthesizer(max_turns=3)

        result = synthesizer.synthesize(tasks)

        for sample in result.samples:
            assert "trajectory" in sample
            assert len(sample["trajectory"]) >= 2  # At least start and end

            # Check turn structure
            for turn in sample["trajectory"]:
                assert "turn" in turn
                assert "thought" in turn


class TestDomainSynthesizer:
    """Tests for GLAN-style domain synthesis."""

    def test_domain_content_generation(self):
        """Test domain-specific content generation."""
        synthesizer = DomainSynthesizer(
            domain="machine_learning",
            topics=["fundamentals", "applications"],
        )

        result = synthesizer.synthesize()

        assert isinstance(result, SynthesisResult)
        assert result.synthesized_count == 2  # One per topic
        assert result.strategy == "domain"

    def test_topic_coverage(self):
        """Test that all topics are covered."""
        topics = ["basics", "advanced", "applications"]
        synthesizer = DomainSynthesizer(
            domain="data_science",
            topics=topics,
        )

        result = synthesizer.synthesize()

        generated_topics = {s["topic"] for s in result.samples}
        assert generated_topics == set(topics)


class TestDataSynthesisPipeline:
    """Tests for unified synthesis pipeline."""

    @pytest.fixture
    def source_texts(self) -> List[str]:
        return [
            "Neural networks learn hierarchical representations.",
            "Transformers use self-attention mechanisms.",
        ]

    def test_single_strategy(self, source_texts):
        """Test pipeline with single strategy."""
        pipeline = DataSynthesisPipeline()

        result = pipeline.synthesize(
            source_texts,
            strategies=[SynthesisStrategy.REPHRASING],
        )

        assert isinstance(result, SynthesisResult)
        assert result.strategy == "pipeline"
        assert result.synthesized_count > 0

    def test_multi_strategy(self, source_texts):
        """Test pipeline with multiple strategies."""
        pipeline = DataSynthesisPipeline()

        result = pipeline.synthesize(
            source_texts,
            strategies=[
                SynthesisStrategy.REPHRASING,
                SynthesisStrategy.INSTRUCTION,
            ],
        )

        # Should have samples from both strategies
        strategies_used = {s["strategy"] for s in result.samples}
        assert "rephrasing" in strategies_used
        assert "instruction" in strategies_used

    def test_ihsan_filtering(self, source_texts):
        """Test that Ihsān filtering is applied."""
        pipeline = DataSynthesisPipeline(ihsan_threshold=0.95)

        result = pipeline.synthesize(source_texts)

        # All samples should pass soft quality threshold (0.95 * 0.8 = 0.76)
        # Synthesis uses softer threshold since samples undergo further validation
        for sample in result.samples:
            assert sample["quality_score"] >= 0.76

    def test_default_strategies(self, source_texts):
        """Test default strategy selection."""
        pipeline = DataSynthesisPipeline()

        result = pipeline.synthesize(source_texts)

        # Default is rephrasing + instruction
        assert result.synthesized_count > 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSelectionSynthesisIntegration:
    """Integration tests for selection + synthesis pipeline."""

    def test_select_then_synthesize(self):
        """Test selecting data then synthesizing from selection."""
        # Source pool
        texts = [
            "Deep learning revolutionized computer vision applications.",
            "Random noise text without much value.",
            "Natural language processing enables chatbots and translation.",
            "More random stuff here.",
            "Reinforcement learning trains agents through rewards.",
        ]

        # Step 1: Select high-quality samples
        selection_pipeline = DataSelectionPipeline()
        selection_result = selection_pipeline.select(texts, method="model")

        selected_texts = [texts[i] for i in selection_result.selected_indices]

        # Step 2: Synthesize from selected samples
        synthesis_pipeline = DataSynthesisPipeline()
        synthesis_result = synthesis_pipeline.synthesize(
            selected_texts,
            strategies=[SynthesisStrategy.INSTRUCTION],
        )

        # Should have generated instruction pairs
        assert synthesis_result.synthesized_count > 0
        for sample in synthesis_result.samples:
            assert sample["strategy"] == "instruction"

    def test_full_data4llm_pipeline(self):
        """Test full DATA4LLM-inspired pipeline."""
        # Simulate raw data
        raw_texts = [
            "Machine learning models require training data to learn patterns.",
            "x",
            "Deep neural networks can approximate complex functions.",
            "",
            "Transfer learning enables knowledge reuse across domains.",
        ]

        # Selection (DATA4LLM: Data Selection)
        selector = ModelBasedSelector(scoring_method="quality", threshold=0.3)
        selection_result = selector.select(raw_texts)

        assert selection_result.selected_count >= 2
        assert 1 not in selection_result.selected_indices  # "x" not selected
        assert 3 not in selection_result.selected_indices  # "" not selected

        selected = [raw_texts[i] for i in selection_result.selected_indices]

        # Synthesis (DATA4LLM: Data Synthesis)
        synthesizer = DataSynthesisPipeline()
        synthesis_result = synthesizer.synthesize(
            selected,
            strategies=[
                SynthesisStrategy.REPHRASING,
                SynthesisStrategy.INSTRUCTION,
            ],
        )

        # Verify augmentation
        assert synthesis_result.total_count > len(selected)
        assert synthesis_result.augmentation_ratio > 1.0
