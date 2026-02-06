"""
SDPO Integration Tests — Verify SAPE-SDPO Cognitive Architecture
===============================================================================

Tests the full SDPO integration with BIZRA's cognitive architecture.
Genesis Strict Synthesis v2.2.2
"""

import pytest
import asyncio
from typing import Dict, Any

from core.sdpo import (
    SDPO_LEARNING_RATE,
    SDPO_ADVANTAGE_THRESHOLD,
    SAPE_WISDOM_SNR,
)
from core.sdpo.optimization import (
    SDPOAdvantage,
    SDPOAdvantageCalculator,
    SDPOFeedback,
    BIZRAFeedbackGenerator,
)
from core.sdpo.cosmos import (
    SAPE_SDPO_Fusion,
    SDPO_SAPE_Result,
    SAPELayerOutput,
    DefaultSAPEProcessor,
    ImplicitPRM,
)
from core.sdpo.agents import (
    PAT_SDPO_Learner,
    PAT_SDPO_Config,
    ContextCompressionEngine,
)
from core.sdpo.discovery import (
    SDPOTestTimeDiscovery,
    DiscoveryConfig,
    NoveltyScorer,
)
from core.sdpo.training import (
    BIZRASDPOTrainer,
    TrainingConfig,
    TrainingBatch,
)
from core.sdpo.validation import (
    SDPOABTestFramework,
    ABTestConfig,
    QualityValidator,
)


class TestSDPOOptimization:
    """Test SDPO advantage calculation."""

    @pytest.fixture
    def calculator(self):
        return SDPOAdvantageCalculator()

    @pytest.mark.asyncio
    async def test_advantage_calculation_heuristic(self, calculator):
        """Test heuristic-based advantage calculation."""
        advantage = await calculator.calculate_advantages(
            question="What is sovereignty?",
            failed_attempt="Sovereignty is about control and power.",
            feedback="Your response lacks precision. Be more specific about legal authority.",
            corrected_attempt="Sovereignty is the supreme legal authority within a territory.",
        )

        assert isinstance(advantage, SDPOAdvantage)
        assert len(advantage.token_advantages) > 0
        assert 0 <= advantage.positive_ratio <= 1

    @pytest.mark.asyncio
    async def test_advantage_beneficial_detection(self, calculator):
        """Test detection of beneficial feedback."""
        # Good feedback case
        good_advantage = await calculator.calculate_advantages(
            question="Calculate 2+2",
            failed_attempt="2+2 equals 5",
            feedback="Basic arithmetic error. 2+2=4, not 5. Improve calculation accuracy.",
            corrected_attempt="2+2 equals 4",
        )

        # The corrected attempt should show improvement
        assert good_advantage.overall_advantage >= 0


class TestBIZRAFeedbackGenerator:
    """Test BIZRA feedback generation."""

    @pytest.fixture
    def generator(self):
        return BIZRAFeedbackGenerator()

    def test_feedback_for_low_ihsan(self, generator):
        """Test feedback generation for low Ihsān score."""
        quality_check = {
            "passes": False,
            "ihsan_score": 0.85,
            "snr": 0.95,
            "fate_compliant": True,
        }

        feedback = generator.generate_feedback(quality_check)

        assert isinstance(feedback, SDPOFeedback)
        assert "Ihsān" in feedback.text
        assert feedback.confidence > 0

    def test_feedback_for_fate_violation(self, generator):
        """Test feedback for FATE violation."""
        quality_check = {
            "passes": False,
            "ihsan_score": 0.98,
            "snr": 0.95,
            "fate_compliant": False,
            "fate_violation": "bias detected",
            "fate_correction": "remove biased language",
        }

        feedback = generator.generate_feedback(quality_check)

        assert "FATE" in feedback.text
        assert "ethical_compliance" in feedback.improvement_areas


class TestSAPESDPOFusion:
    """Test SAPE-SDPO cognitive fusion."""

    @pytest.fixture
    def fusion(self):
        return SAPE_SDPO_Fusion()

    @pytest.mark.asyncio
    async def test_full_pipeline(self, fusion):
        """Test full SAPE-SDPO pipeline."""
        result = await fusion.process("What is the meaning of sovereignty?")

        assert isinstance(result, SDPO_SAPE_Result)
        assert len(result.layer_outputs) == 4  # data, info, knowledge, wisdom
        assert result.total_snr > 0
        assert result.ihsan_score > 0

    @pytest.mark.asyncio
    async def test_layer_progression(self, fusion):
        """Test SNR increases through layers."""
        result = await fusion.process("Test input for layer analysis.")

        snr_by_layer = {lo.layer: lo.snr_score for lo in result.layer_outputs}

        # Wisdom should have highest SNR
        assert snr_by_layer["wisdom"] >= snr_by_layer["knowledge"]
        assert snr_by_layer["knowledge"] >= snr_by_layer["information"]
        assert snr_by_layer["information"] >= snr_by_layer["data"]


class TestImplicitPRM:
    """Test Implicit Process Reward Model."""

    def test_store_and_lookup(self):
        """Test storing and looking up feedback."""
        prm = ImplicitPRM(max_cache_size=10)

        feedback = SDPOFeedback(
            text="Improve clarity",
            source="test",
            confidence=0.9,
        )

        prm.store("error_sig_1", feedback)
        retrieved = prm.lookup("error_sig_1")

        assert retrieved is not None
        assert retrieved.text == "Improve clarity"

    def test_cache_eviction(self):
        """Test FIFO cache eviction."""
        prm = ImplicitPRM(max_cache_size=3)

        for i in range(5):
            feedback = SDPOFeedback(text=f"Feedback {i}", source="test", confidence=0.9)
            prm.store(f"sig_{i}", feedback)

        # First two should be evicted
        assert prm.lookup("sig_0") is None
        assert prm.lookup("sig_1") is None
        assert prm.lookup("sig_4") is not None


class TestPATSDPOLearner:
    """Test PAT Agent SDPO learning."""

    @pytest.fixture
    def learner(self):
        return PAT_SDPO_Learner()

    @pytest.mark.asyncio
    async def test_context_compression(self, learner):
        """Test context compression."""
        context = """
        This is a test context with multiple sentences.
        It contains information about sovereignty and governance.
        The BIZRA ecosystem handles decentralized systems.
        Each node contributes to the overall network.
        """

        compressed, ratio = await learner.compress_context(
            context,
            preserve=["sovereignty", "BIZRA"],
        )

        assert len(compressed) <= len(context)
        assert ratio <= 1.0

    @pytest.mark.asyncio
    async def test_learning_from_interaction(self, learner):
        """Test learning from interaction feedback."""
        result = await learner.learn_from_interaction(
            task="answer question",
            response="This is the answer.",
            feedback="Be more specific.",
            quality_score=0.85,
        )

        # Should return advantage when quality below threshold
        # (default threshold is 0.95)
        assert result is not None or learner.state.total_interactions == 0

        stats = learner.get_learning_stats()
        assert stats["total_interactions"] >= 0


class TestNoveltyScorer:
    """Test novelty scoring."""

    def test_first_solution_is_novel(self):
        """First solution should be maximally novel."""
        scorer = NoveltyScorer()
        score = scorer.score_novelty("This is a completely new solution.")
        assert score == 1.0

    def test_similar_solution_less_novel(self):
        """Similar solutions should have lower novelty."""
        scorer = NoveltyScorer()

        scorer.add_to_archive("The quick brown fox jumps over the lazy dog.")
        score = scorer.score_novelty("The quick brown fox runs over the lazy dog.")

        assert score < 1.0  # Should be less novel


class TestSDPOTestTimeDiscovery:
    """Test test-time discovery engine."""

    @pytest.fixture
    def discovery(self):
        return SDPOTestTimeDiscovery()

    @pytest.mark.asyncio
    async def test_discovery_with_initial_solutions(self, discovery):
        """Test discovery with provided initial solutions."""
        result = await discovery.discover(
            query="How to optimize database performance?",
            initial_solutions=[
                "Add indexes to frequently queried columns.",
                "Use connection pooling.",
            ],
        )

        assert result.total_explorations >= 2
        assert len(result.best_solution) > 0


class TestBIZRASDPOTrainer:
    """Test BIZRA-SDPO training loop."""

    @pytest.fixture
    def trainer(self):
        return BIZRASDPOTrainer(
            config=TrainingConfig(
                max_epochs=1,
                checkpoint_interval=1000,  # Don't checkpoint during tests
            )
        )

    @pytest.mark.asyncio
    async def test_training_batch(self, trainer):
        """Test training on a batch."""
        batch = TrainingBatch(
            questions=["What is 2+2?"],
            failed_attempts=["2+2 equals 5"],
            feedbacks=["Incorrect. 2+2=4."],
            corrected_attempts=["2+2 equals 4"],
            quality_scores=[0.95],
        )

        result = await trainer.train([batch])

        assert result.total_epochs_completed == 1
        assert result.total_steps >= 1

    @pytest.mark.asyncio
    async def test_evaluation(self, trainer):
        """Test evaluation on validation data."""
        batch = TrainingBatch(
            questions=["What is 3+3?"],
            failed_attempts=["3+3 equals 7"],
            feedbacks=["Incorrect arithmetic."],
            corrected_attempts=["3+3 equals 6"],
            quality_scores=[0.96],
        )

        eval_result = await trainer.evaluate([batch])

        assert "eval_loss" in eval_result
        assert "eval_ihsan" in eval_result


class TestSDPOABTestFramework:
    """Test A/B testing framework."""

    @pytest.fixture
    def framework(self):
        return SDPOABTestFramework()

    def test_create_experiment(self, framework):
        """Test experiment creation."""
        exp_id = framework.create_experiment(
            name="SDPO v1 vs v2",
            control_desc="Baseline",
            treatment_desc="Enhanced",
        )

        assert exp_id is not None
        status = framework.get_experiment_status(exp_id)
        assert status["status"] == "running"

    def test_add_samples(self, framework):
        """Test adding samples."""
        exp_id = framework.create_experiment(
            name="Test",
            control_desc="Control",
            treatment_desc="Treatment",
        )

        for _ in range(10):
            framework.add_sample(exp_id, "control", 0.90)
            framework.add_sample(exp_id, "treatment", 0.95)

        status = framework.get_experiment_status(exp_id)
        assert status["control_samples"] == 10
        assert status["treatment_samples"] == 10

    def test_simulate_experiment(self, framework):
        """Test experiment simulation."""
        result = framework.simulate_experiment(
            control_mean=0.90,
            treatment_mean=0.95,
            std=0.05,
            n_samples=100,
        )

        assert result.analysis is not None
        assert result.winner is not None or result.recommendation != ""


class TestQualityValidator:
    """Test quality gate validation."""

    def test_ihsan_validation_pass(self):
        """Test passing Ihsān validation."""
        from core.sdpo.validation import ExperimentArm

        validator = QualityValidator()
        arm = ExperimentArm(name="test", description="test arm")
        for _ in range(10):
            arm.add_sample(0.96)

        passes, details = validator.validate(arm, "ihsan")
        assert passes
        assert details["margin"] > 0

    def test_ihsan_validation_fail(self):
        """Test failing Ihsān validation."""
        from core.sdpo.validation import ExperimentArm

        validator = QualityValidator()
        arm = ExperimentArm(name="test", description="test arm")
        for _ in range(10):
            arm.add_sample(0.80)

        passes, details = validator.validate(arm, "ihsan")
        assert not passes
        assert details["margin"] < 0


# Run quick integration test
if __name__ == "__main__":
    print("Running SDPO Integration Tests...")

    async def quick_test():
        # Test SAPE-SDPO Fusion
        fusion = SAPE_SDPO_Fusion()
        result = await fusion.process("What is sovereignty?")
        print(f"SAPE-SDPO Result: SNR={result.total_snr:.3f}, Ihsān={result.ihsan_score:.3f}")

        # Test A/B Framework
        framework = SDPOABTestFramework()
        ab_result = framework.simulate_experiment(0.90, 0.95, 0.05, 100)
        print(f"A/B Test: Winner={ab_result.winner}, Significant={ab_result.analysis.is_significant}")

        print("All quick tests passed!")

    asyncio.run(quick_test())
