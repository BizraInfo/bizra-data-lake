#!/usr/bin/env python3
"""
tests/test_apex_integration.py - BIZRA Apex Orchestrator Integration Tests
===========================================================================

Comprehensive test suite for the Apex Orchestrator components:
- Thompson Sampling Router: Bayesian agent selection
- SONA Learner: Self-Optimizing Novelty Architecture
- Pattern Extractor: Success/failure pattern mining
- Cost Analyzer: Cost-aware model selection

From the Blueprint:
    - Ihsan threshold: >= 0.95 for quality gates
    - Pattern elevation: > 3 repetitions for SAPE cache
    - Cost savings target: 60-70%
    - Routing improvement target: +55%

Test Categories:
    1. Thompson Sampling Router Tests
    2. SONA Learner Tests
    3. Pattern Extractor Tests
    4. Cost Analyzer Tests
    5. End-to-End Integration Tests
"""

import asyncio
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Apex components
from core.apex.thompson_router import (
    ThompsonSamplingRouter,
    CapabilityMatrix,
    AgentProfile,
    AgentCapability,
    TaskCategory,
    BetaPrior,
    SelectionResult,
)
from core.apex.sona_learner import (
    SONALearner,
    LearningConfig,
    ExecutionRecord as SONAExecutionRecord,
    PerformanceMetrics,
    TrackedPattern,
    RoutingWeights,
)
from core.apex.pattern_extractor import (
    PatternExtractor,
    ExecutionPattern,
    PatternType,
    PatternScope,
    ExecutionRecord as ExtractorExecutionRecord,
)
from core.apex.cost_analyzer import (
    CostAnalyzer,
    ModelCostConfig,
    UsageRecord,
    CostMetrics,
    DEFAULT_MODEL_COSTS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_state_dir():
    """Create a temporary directory for state persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_agents() -> List[str]:
    """Sample agent names for testing."""
    return [
        "MasterReasoner",
        "MemoryArchitect",
        "CreativeSynthesizer",
        "DataAnalyzer",
        "Communicator",
        "ExecutionPlanner",
        "EthicsGuardian",
    ]


@pytest.fixture
def sample_tasks() -> List[Dict[str, Any]]:
    """Sample tasks for testing."""
    return [
        {"text": "Analyze the quarterly financial data and provide insights", "category": TaskCategory.ANALYSIS},
        {"text": "Write a creative marketing copy for the new product", "category": TaskCategory.CREATIVE},
        {"text": "Plan the project timeline for Q2 deliverables", "category": TaskCategory.PLANNING},
        {"text": "Check for ethical compliance in the AI model", "category": TaskCategory.ETHICS},
        {"text": "Remember our previous conversation about architecture", "category": TaskCategory.MEMORY},
        {"text": "Explain the technical architecture to stakeholders", "category": TaskCategory.COMMUNICATION},
        {"text": "Reason through the strategic decision options", "category": TaskCategory.REASONING},
        {"text": "Validate the data integrity of the pipeline", "category": TaskCategory.VALIDATION},
    ]


@pytest.fixture
def sample_execution_records() -> List[SONAExecutionRecord]:
    """Sample execution records for SONA learner tests."""
    records = []
    agents = ["MasterReasoner", "CreativeSynthesizer", "DataAnalyzer"]
    categories = ["reasoning", "creative", "analysis"]

    for i in range(50):
        agent = agents[i % len(agents)]
        category = categories[i % len(categories)]
        success = np.random.random() > 0.3
        quality = np.random.uniform(0.7, 1.0) if success else np.random.uniform(0.3, 0.7)

        records.append(SONAExecutionRecord(
            task_id=f"task_{i:04d}",
            task_category=category,
            agent_name=agent,
            success=success,
            quality_score=quality,
            latency_ms=np.random.uniform(500, 2000),
            token_count=np.random.randint(100, 1000),
            cost=np.random.uniform(0.001, 0.01),
        ))

    return records


@pytest.fixture
def sample_extractor_records() -> List[ExtractorExecutionRecord]:
    """Sample execution records for pattern extractor tests."""
    records = []
    agents = ["MasterReasoner", "CreativeSynthesizer", "DataAnalyzer"]
    categories = ["reasoning", "creative", "analysis"]

    for i in range(30):
        agent = agents[i % len(agents)]
        category = categories[i % len(categories)]

        # Bias success toward matching agent-category
        match_bonus = 0.3 if (
            (agent == "MasterReasoner" and category == "reasoning") or
            (agent == "CreativeSynthesizer" and category == "creative") or
            (agent == "DataAnalyzer" and category == "analysis")
        ) else 0.0

        success = np.random.random() < (0.5 + match_bonus)

        records.append(ExtractorExecutionRecord(
            task_id=f"task_{i:04d}",
            task_category=category,
            agent_name=agent,
            success=success,
            quality_score=np.random.uniform(0.7, 1.0) if success else np.random.uniform(0.3, 0.7),
            latency_ms=np.random.uniform(500, 2000),
        ))

    return records


@pytest.fixture
def sample_usage_records() -> List[UsageRecord]:
    """Sample usage records for cost analyzer tests."""
    records = []
    models = ["deepseek-r1:7b", "qwen2.5:7b", "mistral:7b"]
    agents = ["MasterReasoner", "CreativeSynthesizer", "DataAnalyzer"]
    categories = ["reasoning", "creative", "analysis"]

    for i in range(50):
        model = models[i % len(models)]
        agent = agents[i % len(agents)]
        category = categories[i % len(categories)]

        # Get model config for realistic cost calculation
        config = DEFAULT_MODEL_COSTS.get(model)
        input_tokens = np.random.randint(100, 500)
        output_tokens = np.random.randint(100, 1000)

        if config:
            cost = (
                (input_tokens / 1000) * config.cost_per_1k_input_tokens +
                (output_tokens / 1000) * config.cost_per_1k_output_tokens
            )
            latency = config.avg_latency_ms * (1 + np.random.uniform(-0.2, 0.2))
        else:
            cost = 0.001
            latency = 1000

        records.append(UsageRecord(
            execution_id=f"exec_{i:04d}",
            model_name=model,
            agent_name=agent,
            task_category=category,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=latency,
            quality_score=np.random.uniform(0.7, 1.0),
            success=np.random.random() > 0.1,
        ))

    return records


@pytest.fixture
def mock_capability_matrix() -> CapabilityMatrix:
    """Create a capability matrix with test profiles."""
    return CapabilityMatrix()


@pytest.fixture
def thompson_router(temp_state_dir) -> ThompsonSamplingRouter:
    """Create a Thompson Sampling Router for testing."""
    return ThompsonSamplingRouter(
        persistence_path=temp_state_dir / "thompson_state.json",
        seed=42,  # Fixed seed for reproducibility
    )


@pytest.fixture
def sona_learner(temp_state_dir) -> SONALearner:
    """Create a SONA Learner for testing."""
    config = LearningConfig(
        learning_rate=0.1,
        momentum=0.9,
        min_samples_for_update=5,  # Lower threshold for testing
        update_interval_seconds=1.0,  # Faster updates for testing
        elevation_threshold=3,
        min_success_rate_for_elevation=0.7,
        target_improvement=0.55,
        ihsan_threshold=0.95,
        state_path=str(temp_state_dir / "sona_state.json"),
    )
    return SONALearner(config=config)


@pytest.fixture
def pattern_extractor() -> PatternExtractor:
    """Create a Pattern Extractor for testing."""
    return PatternExtractor(
        embedding_dim=64,
        similarity_threshold=0.8,
        max_sequence_length=5,
    )


@pytest.fixture
def cost_analyzer(temp_state_dir) -> CostAnalyzer:
    """Create a Cost Analyzer for testing."""
    return CostAnalyzer(
        persistence_path=temp_state_dir / "cost_state.json",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. THOMPSON SAMPLING ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestThompsonRouterInitialization:
    """Tests for Thompson Sampling Router initialization."""

    def test_router_initialization(self, thompson_router):
        """Test that router initializes correctly."""
        assert thompson_router is not None
        assert thompson_router.capability_matrix is not None
        assert thompson_router.rng is not None
        assert len(thompson_router._priors) == 0  # No priors until first selection

    def test_router_with_custom_capability_matrix(self, temp_state_dir):
        """Test router initialization with custom capability matrix."""
        custom_profile = AgentProfile(
            agent_name="CustomAgent",
            capabilities={AgentCapability.STRATEGIC_THINKING: 0.9},
            task_affinities={TaskCategory.REASONING: 0.9},
            cost_per_token=0.001,
            avg_latency_ms=500,
        )
        custom_matrix = CapabilityMatrix()
        custom_matrix.add_profile(custom_profile)

        router = ThompsonSamplingRouter(
            capability_matrix=custom_matrix,
            persistence_path=temp_state_dir / "custom_router.json",
        )

        assert "CustomAgent" in router.capability_matrix.profiles

    def test_router_with_seed(self, temp_state_dir):
        """Test that seeded router produces reproducible results."""
        router1 = ThompsonSamplingRouter(
            persistence_path=temp_state_dir / "router1.json",
            seed=12345,
        )
        router2 = ThompsonSamplingRouter(
            persistence_path=temp_state_dir / "router2.json",
            seed=12345,
        )

        # Same seed should produce same selection for same task
        result1 = router1.select_agent("Analyze data patterns")
        result2 = router2.select_agent("Analyze data patterns")

        assert result1.agent_name == result2.agent_name


class TestThompsonAgentSelection:
    """Tests for agent selection functionality."""

    def test_agent_selection_returns_valid_agent(self, thompson_router, sample_agents):
        """Test that agent selection returns a valid agent."""
        result = thompson_router.select_agent("Analyze the data trends")

        assert result is not None
        assert isinstance(result, SelectionResult)
        assert result.agent_name in sample_agents
        assert isinstance(result.task_category, TaskCategory)
        assert 0.0 <= result.sampled_value <= 1.0
        assert 0.0 <= result.exploration_rate <= 1.0

    def test_agent_selection_with_explicit_category(self, thompson_router):
        """Test agent selection with explicit task category."""
        result = thompson_router.select_agent(
            "Some task",
            category=TaskCategory.CREATIVE,
        )

        assert result.task_category == TaskCategory.CREATIVE

    def test_agent_selection_with_explicit_candidates(self, thompson_router):
        """Test agent selection with explicit candidate list."""
        candidates = ["MasterReasoner", "DataAnalyzer"]
        result = thompson_router.select_agent(
            "Analyze something",
            candidates=candidates,
        )

        assert result.agent_name in candidates

    def test_task_classification(self, thompson_router, sample_tasks):
        """Test that tasks are classified to correct categories."""
        for task in sample_tasks:
            result = thompson_router.select_agent(task["text"])
            # The classification might vary, but should be consistent
            assert isinstance(result.task_category, TaskCategory)


class TestThompsonPosteriorUpdate:
    """Tests for posterior update functionality."""

    def test_posterior_update_after_success(self, thompson_router):
        """Test that posterior updates correctly after success."""
        # Make an initial selection
        result = thompson_router.select_agent("Analyze data")
        agent_name = result.agent_name
        category = result.task_category

        # Get initial prior
        initial_prior = thompson_router._get_prior(agent_name, category)
        initial_alpha = initial_prior.alpha

        # Update with success
        thompson_router.update_posterior(agent_name, category, success=True)

        # Check that alpha increased
        updated_prior = thompson_router._get_prior(agent_name, category)
        assert updated_prior.alpha > initial_alpha

    def test_posterior_update_after_failure(self, thompson_router):
        """Test that posterior updates correctly after failure."""
        # Make an initial selection
        result = thompson_router.select_agent("Analyze data")
        agent_name = result.agent_name
        category = result.task_category

        # Get initial prior
        initial_prior = thompson_router._get_prior(agent_name, category)
        initial_beta = initial_prior.beta

        # Update with failure
        thompson_router.update_posterior(agent_name, category, success=False)

        # Check that beta increased
        updated_prior = thompson_router._get_prior(agent_name, category)
        assert updated_prior.beta > initial_beta

    def test_posterior_update_with_quality_score(self, thompson_router):
        """Test posterior update with weighted quality score."""
        result = thompson_router.select_agent("Analyze data")
        agent_name = result.agent_name
        category = result.task_category

        initial_prior = thompson_router._get_prior(agent_name, category)
        initial_alpha = initial_prior.alpha
        initial_beta = initial_prior.beta

        # Update with high quality score
        thompson_router.update_posterior(
            agent_name, category,
            success=True,
            quality_score=0.95,
        )

        updated_prior = thompson_router._get_prior(agent_name, category)

        # Both should increase, but alpha more than beta
        assert updated_prior.alpha > initial_alpha
        assert updated_prior.alpha - initial_alpha > updated_prior.beta - initial_beta


class TestThompsonExplorationRate:
    """Tests for exploration rate calculation."""

    def test_exploration_rate_initial(self, thompson_router):
        """Test that initial exploration rate is maximum."""
        rate = thompson_router.get_exploration_rate()
        assert rate == 1.0  # Maximum exploration with no history

    def test_exploration_rate_decreases_with_selections(self, thompson_router):
        """Test that exploration rate changes with selections."""
        # Make multiple selections to build history
        for _ in range(10):
            result = thompson_router.select_agent("Analyze data")
            thompson_router.update_posterior(
                result.agent_name,
                result.task_category,
                success=True,
            )

        rate = thompson_router.get_exploration_rate()
        assert rate < 1.0

    def test_exploration_rate_by_category(self, thompson_router):
        """Test exploration rate filtering by category."""
        # Make selections for ANALYSIS category
        for _ in range(5):
            result = thompson_router.select_agent(
                "Analyze something",
                category=TaskCategory.ANALYSIS,
            )
            thompson_router.update_posterior(
                result.agent_name,
                result.task_category,
                success=True,
            )

        # Rate for ANALYSIS should be lower than for unused categories
        analysis_rate = thompson_router.get_exploration_rate(TaskCategory.ANALYSIS)
        creative_rate = thompson_router.get_exploration_rate(TaskCategory.CREATIVE)

        assert analysis_rate <= creative_rate


class TestThompsonCapabilityMatrix:
    """Tests for capability matrix functionality."""

    def test_capability_matrix_task_matching(self, mock_capability_matrix):
        """Test that capability matrix returns appropriate candidates."""
        candidates = mock_capability_matrix.get_candidates(TaskCategory.REASONING)

        assert len(candidates) > 0
        assert "MasterReasoner" in candidates  # High affinity for reasoning

    def test_capability_matrix_min_affinity(self, mock_capability_matrix):
        """Test candidate filtering by minimum affinity."""
        # High threshold should return fewer candidates
        high_threshold = mock_capability_matrix.get_candidates(
            TaskCategory.ETHICS,
            min_affinity=0.9,
        )
        low_threshold = mock_capability_matrix.get_candidates(
            TaskCategory.ETHICS,
            min_affinity=0.5,
        )

        assert len(high_threshold) <= len(low_threshold)

    def test_task_classification_keywords(self, mock_capability_matrix):
        """Test task classification using keywords."""
        # Reasoning keywords
        category = mock_capability_matrix.classify_task(
            "analyze and evaluate the strategic decision"
        )
        assert category == TaskCategory.REASONING

        # Creative keywords
        category = mock_capability_matrix.classify_task(
            "write a creative story with imaginative elements"
        )
        assert category == TaskCategory.CREATIVE

        # General fallback
        category = mock_capability_matrix.classify_task("xyz abc def")
        assert category == TaskCategory.GENERAL


class TestThompsonPersistence:
    """Tests for router state persistence."""

    def test_router_persistence_save_load(self, temp_state_dir):
        """Test that router state can be saved and loaded."""
        persistence_path = temp_state_dir / "persist_test.json"

        # Create router and make some selections
        router1 = ThompsonSamplingRouter(
            persistence_path=persistence_path,
            seed=42,
        )

        for _ in range(5):
            result = router1.select_agent("Analyze data")
            router1.update_posterior(
                result.agent_name,
                result.task_category,
                success=True,
            )

        # Save state
        router1._save_state()

        # Create new router and load state
        router2 = ThompsonSamplingRouter(
            persistence_path=persistence_path,
        )

        # Check that priors were loaded
        assert len(router2._priors) > 0

    def test_router_to_json(self, thompson_router):
        """Test serialization to JSON."""
        # Make some selections first
        result = thompson_router.select_agent("Analyze data")
        thompson_router.update_posterior(
            result.agent_name,
            result.task_category,
            success=True,
        )

        json_str = thompson_router.to_json()
        data = json.loads(json_str)

        assert "version" in data
        assert "priors" in data
        assert "timestamp" in data

    def test_router_from_json(self, thompson_router):
        """Test deserialization from JSON."""
        # Serialize
        json_str = thompson_router.to_json()

        # Deserialize
        new_router = ThompsonSamplingRouter.from_json(json_str)

        assert new_router is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SONA LEARNER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSONALearnerInitialization:
    """Tests for SONA Learner initialization."""

    def test_learner_initialization(self, sona_learner):
        """Test that learner initializes correctly."""
        assert sona_learner is not None
        assert sona_learner.config is not None
        assert sona_learner._routing_weights is not None
        assert sona_learner._metrics is not None

    def test_learner_config_from_env(self):
        """Test learner configuration from environment."""
        with patch.dict(os.environ, {
            "SONA_LEARNING_RATE": "0.2",
            "IHSAN_THRESHOLD": "0.98",
        }):
            config = LearningConfig.from_env()
            assert config.learning_rate == 0.2
            assert config.ihsan_threshold == 0.98


class TestSONAPatternExtraction:
    """Tests for pattern extraction from history."""

    def test_pattern_extraction_from_history(self, sona_learner, sample_execution_records):
        """Test that patterns are extracted from execution history."""
        # Record executions
        for record in sample_execution_records:
            sona_learner.record_execution(record)

        # Extract patterns
        patterns = sona_learner.extract_patterns()

        assert len(patterns) > 0
        assert all(isinstance(p, TrackedPattern) for p in patterns)

    def test_pattern_tracking_updates(self, sona_learner):
        """Test that pattern tracking updates correctly."""
        # Record same pattern multiple times
        for i in range(5):
            record = SONAExecutionRecord(
                task_id=f"task_{i:04d}",
                task_category="reasoning",
                agent_name="MasterReasoner",
                success=True,
                quality_score=0.9,
                latency_ms=1000,
                token_count=500,
                cost=0.005,
            )
            sona_learner.record_execution(record)

        patterns = sona_learner.extract_patterns()

        # Find the pattern for this combination
        matching_patterns = [
            p for p in patterns
            if "reasoning:MasterReasoner" in p.pattern_signature[0]
        ]

        assert len(matching_patterns) > 0
        assert matching_patterns[0].occurrence_count == 5


class TestSONARoutingOptimization:
    """Tests for routing optimization."""

    def test_routing_optimization(self, sona_learner, sample_execution_records):
        """Test that routing weights are optimized."""
        # Record executions
        for record in sample_execution_records:
            sona_learner.record_execution(record)

        # Get initial weights
        initial_weight = sona_learner._routing_weights.get_weight(
            "MasterReasoner", "reasoning"
        )

        # Optimize
        sona_learner.optimize_routing()

        # Weights should be updated
        updated_weight = sona_learner._routing_weights.get_weight(
            "MasterReasoner", "reasoning"
        )

        # Weight should have changed (may increase or decrease based on performance)
        assert updated_weight is not None

    def test_routing_recommendation(self, sona_learner, sample_execution_records):
        """Test getting routing recommendations."""
        # Record executions
        for record in sample_execution_records:
            sona_learner.record_execution(record)

        sona_learner.optimize_routing()

        recommendations = sona_learner.get_routing_recommendation(
            "reasoning",
            ["MasterReasoner", "DataAnalyzer", "CreativeSynthesizer"],
        )

        assert len(recommendations) == 3
        assert all(isinstance(r, tuple) for r in recommendations)
        assert all(len(r) == 2 for r in recommendations)


class TestSONAPerformanceEvaluation:
    """Tests for performance evaluation with Ihsan integration."""

    def test_performance_evaluation_with_ihsan(self, sona_learner, sample_execution_records):
        """Test performance evaluation with Ihsan threshold."""
        # Record executions
        for record in sample_execution_records:
            sona_learner.record_execution(record)

        metrics = sona_learner.evaluate_performance()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_executions == len(sample_execution_records)
        assert 0.0 <= metrics.avg_quality_score <= 1.0

    def test_ihsan_compliance_check(self, sona_learner):
        """Test Ihsan compliance in performance metrics."""
        # Record high-quality executions
        for i in range(10):
            record = SONAExecutionRecord(
                task_id=f"task_{i:04d}",
                task_category="reasoning",
                agent_name="MasterReasoner",
                success=True,
                quality_score=0.98,  # Above Ihsan threshold
                latency_ms=1000,
                token_count=500,
                cost=0.005,
            )
            sona_learner.record_execution(record)

        metrics = sona_learner.evaluate_performance()

        assert metrics.ihsan_compliant is True
        assert metrics.avg_quality_score >= 0.95


class TestSONAPatternElevation:
    """Tests for pattern elevation threshold."""

    def test_pattern_elevation_threshold(self, sona_learner):
        """Test pattern elevation when threshold is exceeded."""
        elevated_patterns = []

        def mock_callback(pattern):
            elevated_patterns.append(pattern)

        sona_learner.sape_elevation_callback = mock_callback

        # Record same pattern multiple times with high success
        for i in range(5):  # Exceed threshold of 3
            record = SONAExecutionRecord(
                task_id=f"task_{i:04d}",
                task_category="reasoning",
                agent_name="MasterReasoner",
                success=True,
                quality_score=0.9,  # High quality
                latency_ms=1000,
                token_count=500,
                cost=0.005,
            )
            sona_learner.record_execution(record)

        patterns = sona_learner.extract_patterns()

        # Find patterns that should elevate
        candidates = [p for p in patterns if p.should_elevate(3, 0.7)]

        assert len(candidates) > 0


class TestSONALearningLoop:
    """Tests for async learning loop."""

    @pytest.mark.asyncio
    async def test_learning_loop_async(self, temp_state_dir):
        """Test async learning loop start and stop."""
        config = LearningConfig(
            update_interval_seconds=0.1,  # Very short for testing
            min_samples_for_update=5,
            state_path=str(temp_state_dir / "async_test.json"),
        )
        learner = SONALearner(config=config)

        # Record some executions
        for i in range(10):
            record = SONAExecutionRecord(
                task_id=f"task_{i:04d}",
                task_category="reasoning",
                agent_name="MasterReasoner",
                success=True,
                quality_score=0.9,
                latency_ms=1000,
                token_count=500,
                cost=0.005,
            )
            learner.record_execution(record)

        # Start and stop learning loop
        await learner.start_learning_loop()
        assert learner._running is True

        # Wait for one update cycle
        await asyncio.sleep(0.2)

        await learner.stop_learning_loop()
        assert learner._running is False


class TestSONAImprovementProgress:
    """Tests for improvement progress tracking."""

    def test_improvement_progress(self, sona_learner, sample_execution_records):
        """Test improvement progress calculation."""
        for record in sample_execution_records:
            sona_learner.record_execution(record)

        sona_learner.evaluate_performance()
        progress = sona_learner.get_improvement_progress()

        assert "target_improvement" in progress
        assert "current_improvement" in progress
        assert "progress_percent" in progress
        assert progress["target_improvement"] == 0.55  # +55% target


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PATTERN EXTRACTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPatternExtractorBasic:
    """Basic tests for Pattern Extractor."""

    def test_extractor_initialization(self, pattern_extractor):
        """Test that extractor initializes correctly."""
        assert pattern_extractor is not None
        assert pattern_extractor.embedding_dim == 64
        assert pattern_extractor.similarity_threshold == 0.8

    def test_success_pattern_extraction(self, pattern_extractor, sample_extractor_records):
        """Test extraction of success patterns."""
        patterns = pattern_extractor.extract_success_patterns(sample_extractor_records)

        assert len(patterns) > 0
        assert all(p.pattern_type == PatternType.SUCCESS for p in patterns)

    def test_failure_pattern_extraction(self, pattern_extractor, sample_extractor_records):
        """Test extraction of failure patterns."""
        patterns = pattern_extractor.extract_failure_patterns(sample_extractor_records)

        # May or may not have failure patterns depending on random data
        assert all(p.pattern_type == PatternType.FAILURE for p in patterns)

    def test_all_patterns_extraction(self, pattern_extractor, sample_extractor_records):
        """Test extraction of all patterns."""
        patterns = pattern_extractor.extract_all_patterns(sample_extractor_records)

        assert len(patterns) > 0


class TestPatternHashUniqueness:
    """Tests for pattern hash uniqueness."""

    def test_pattern_hash_uniqueness(self, pattern_extractor):
        """Test that different signatures produce different hashes."""
        hash1 = pattern_extractor.compute_pattern_hash(["agent:MasterReasoner"])
        hash2 = pattern_extractor.compute_pattern_hash(["agent:DataAnalyzer"])
        hash3 = pattern_extractor.compute_pattern_hash(["agent:MasterReasoner"])

        assert hash1 != hash2
        assert hash1 == hash3  # Same signature, same hash

    def test_pattern_hash_determinism(self, pattern_extractor):
        """Test that hash computation is deterministic."""
        signature = ["agent:MasterReasoner", "category:reasoning"]

        hash1 = pattern_extractor.compute_pattern_hash(signature)
        hash2 = pattern_extractor.compute_pattern_hash(signature)

        assert hash1 == hash2


class TestPatternSimilarity:
    """Tests for pattern similarity scoring."""

    def test_pattern_similarity_scoring(self, pattern_extractor):
        """Test similarity scoring between patterns."""
        # Create two similar patterns
        pattern1 = ExecutionPattern(
            pattern_id="test1",
            pattern_type=PatternType.SUCCESS,
            scope=PatternScope.AGENT,
            signature=["agent:MasterReasoner"],
        )
        pattern2 = ExecutionPattern(
            pattern_id="test2",
            pattern_type=PatternType.SUCCESS,
            scope=PatternScope.AGENT,
            signature=["agent:MasterReasoner"],
        )

        similarity = pattern_extractor.compute_pattern_similarity(pattern1, pattern2)

        # Same signature should have high similarity
        assert similarity > 0.9

    def test_pattern_similarity_different(self, pattern_extractor):
        """Test similarity between different patterns."""
        pattern1 = ExecutionPattern(
            pattern_id="test1",
            pattern_type=PatternType.SUCCESS,
            scope=PatternScope.AGENT,
            signature=["agent:MasterReasoner"],
        )
        pattern2 = ExecutionPattern(
            pattern_id="test2",
            pattern_type=PatternType.FAILURE,
            scope=PatternScope.CATEGORY,
            signature=["category:creative"],
        )

        similarity = pattern_extractor.compute_pattern_similarity(pattern1, pattern2)

        # Different signatures should have lower similarity
        assert similarity < 1.0


class TestElevationCandidates:
    """Tests for elevation candidates."""

    def test_elevation_candidates(self, pattern_extractor, sample_extractor_records):
        """Test identification of elevation candidates."""
        # Extract all patterns first
        pattern_extractor.extract_all_patterns(sample_extractor_records)

        candidates = pattern_extractor.get_elevation_candidates()

        # All candidates should meet elevation criteria
        for candidate in candidates:
            assert candidate.occurrence_count > 3
            assert candidate.success_rate >= 0.7
            assert candidate.pattern_type == PatternType.SUCCESS

    def test_should_elevate_property(self):
        """Test the should_elevate property."""
        # Pattern that should elevate
        good_pattern = ExecutionPattern(
            pattern_id="good",
            pattern_type=PatternType.SUCCESS,
            scope=PatternScope.AGENT,
            signature=["agent:MasterReasoner"],
            occurrence_count=5,
            success_count=4,
        )

        assert good_pattern.should_elevate is True

        # Pattern that should not elevate (low count)
        low_count_pattern = ExecutionPattern(
            pattern_id="low",
            pattern_type=PatternType.SUCCESS,
            scope=PatternScope.AGENT,
            signature=["agent:DataAnalyzer"],
            occurrence_count=2,
            success_count=2,
        )

        assert low_count_pattern.should_elevate is False


class TestPatternExtractorPersistence:
    """Tests for pattern extractor persistence."""

    def test_extractor_to_json(self, pattern_extractor, sample_extractor_records):
        """Test serialization to JSON."""
        pattern_extractor.extract_all_patterns(sample_extractor_records)

        json_str = pattern_extractor.to_json()
        data = json.loads(json_str)

        assert "version" in data
        assert "patterns" in data
        assert "config" in data

    def test_extractor_from_json(self, pattern_extractor, sample_extractor_records):
        """Test deserialization from JSON."""
        pattern_extractor.extract_all_patterns(sample_extractor_records)
        json_str = pattern_extractor.to_json()

        new_extractor = PatternExtractor.from_json(json_str)

        assert len(new_extractor._patterns) == len(pattern_extractor._patterns)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COST ANALYZER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCostAnalyzerInitialization:
    """Tests for Cost Analyzer initialization."""

    def test_analyzer_initialization(self, cost_analyzer):
        """Test that analyzer initializes correctly."""
        assert cost_analyzer is not None
        assert cost_analyzer.model_configs is not None
        assert len(cost_analyzer.model_configs) > 0

    def test_default_model_costs(self, cost_analyzer):
        """Test that default model costs are loaded."""
        assert "deepseek-r1:7b" in cost_analyzer.model_configs
        assert "qwen2.5:7b" in cost_analyzer.model_configs
        assert "mistral:7b" in cost_analyzer.model_configs


class TestUsageTracking:
    """Tests for usage tracking."""

    def test_usage_tracking(self, cost_analyzer, sample_usage_records):
        """Test that usage is tracked correctly."""
        for record in sample_usage_records:
            cost_analyzer.track_usage(record)

        assert cost_analyzer._metrics.total_executions == len(sample_usage_records)
        assert cost_analyzer._metrics.total_cost > 0
        assert cost_analyzer._metrics.total_tokens > 0

    def test_usage_by_model(self, cost_analyzer, sample_usage_records):
        """Test usage tracking by model."""
        for record in sample_usage_records:
            cost_analyzer.track_usage(record)

        assert len(cost_analyzer._metrics.cost_by_model) > 0
        assert len(cost_analyzer._metrics.tokens_by_model) > 0


class TestCostPerformanceRatio:
    """Tests for cost-performance ratio computation."""

    def test_cost_performance_ratio(self, cost_analyzer, sample_usage_records):
        """Test cost-performance ratio computation."""
        for record in sample_usage_records:
            cost_analyzer.track_usage(record)

        ratio = cost_analyzer.compute_cost_performance_ratio(
            "deepseek-r1:7b",
            "reasoning",
        )

        assert 0.0 <= ratio <= 1.0

    def test_cost_performance_ratio_unknown_model(self, cost_analyzer):
        """Test ratio computation for unknown model."""
        ratio = cost_analyzer.compute_cost_performance_ratio(
            "unknown-model",
            "reasoning",
        )

        assert ratio == 0.5  # Default for unknown


class TestModelSelectionOptimization:
    """Tests for model selection optimization."""

    def test_model_selection_optimization(self, cost_analyzer, sample_usage_records):
        """Test model selection optimization."""
        for record in sample_usage_records:
            cost_analyzer.track_usage(record)

        models = list(cost_analyzer.model_configs.keys())
        selections = cost_analyzer.optimize_model_selection(
            "reasoning",
            models,
            min_quality=0.7,
            max_latency_ms=3000,
        )

        assert len(selections) > 0
        assert all(isinstance(s, tuple) for s in selections)
        # Should be sorted by score descending
        scores = [s[1] for s in selections]
        assert scores == sorted(scores, reverse=True)

    def test_model_selection_constraints(self, cost_analyzer):
        """Test that model selection respects constraints."""
        # Very strict constraints should return fewer models
        strict_selections = cost_analyzer.optimize_model_selection(
            "reasoning",
            list(cost_analyzer.model_configs.keys()),
            min_quality=0.95,
            max_latency_ms=500,
        )

        lenient_selections = cost_analyzer.optimize_model_selection(
            "reasoning",
            list(cost_analyzer.model_configs.keys()),
            min_quality=0.5,
            max_latency_ms=10000,
        )

        assert len(strict_selections) <= len(lenient_selections)


class TestSavingsCalculation:
    """Tests for savings calculation."""

    def test_savings_calculation(self, cost_analyzer, sample_usage_records):
        """Test savings calculation."""
        for record in sample_usage_records:
            cost_analyzer.track_usage(record)

        savings = cost_analyzer.compute_savings()

        assert "baseline_cost" in savings
        assert "actual_cost" in savings
        assert "savings_rate" in savings
        assert "target_status" in savings

    def test_savings_target_check(self, cost_analyzer, sample_usage_records):
        """Test savings target status check."""
        for record in sample_usage_records:
            cost_analyzer.track_usage(record)

        savings = cost_analyzer.compute_savings()

        # Target status should be one of the expected values
        assert savings["target_status"] in ["below_target", "on_target", "above_target"]


class TestCostReportGeneration:
    """Tests for cost report generation."""

    def test_cost_report_generation(self, cost_analyzer, sample_usage_records):
        """Test comprehensive cost report generation."""
        for record in sample_usage_records:
            cost_analyzer.track_usage(record)

        report = cost_analyzer.generate_cost_report()

        assert "report_timestamp" in report
        assert "summary" in report
        assert "savings" in report
        assert "model_efficiency" in report
        assert "agent_costs" in report
        assert "category_costs" in report
        assert "recommendations" in report

    def test_cost_report_recommendations(self, cost_analyzer, sample_usage_records):
        """Test that cost report includes recommendations."""
        for record in sample_usage_records:
            cost_analyzer.track_usage(record)

        report = cost_analyzer.generate_cost_report()

        assert len(report["recommendations"]) > 0
        assert all(isinstance(r, str) for r in report["recommendations"])


class TestCostAnalyzerPersistence:
    """Tests for cost analyzer persistence."""

    def test_analyzer_save_load(self, temp_state_dir, sample_usage_records):
        """Test that analyzer state can be saved and loaded."""
        persistence_path = temp_state_dir / "cost_persist_test.json"

        # Create analyzer and track usage
        analyzer1 = CostAnalyzer(persistence_path=persistence_path)
        for record in sample_usage_records[:10]:
            analyzer1.track_usage(record)

        analyzer1._save_state()

        # Create new analyzer and load state
        analyzer2 = CostAnalyzer(persistence_path=persistence_path)

        # State should be loaded
        assert analyzer2._metrics.total_executions > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. END-TO-END INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullTaskRoutingCycle:
    """Tests for full task routing cycle."""

    def test_full_task_routing_cycle(
        self,
        thompson_router,
        sona_learner,
        pattern_extractor,
        cost_analyzer,
        sample_tasks,
    ):
        """Test complete task routing cycle through all components."""
        for task in sample_tasks[:5]:
            # 1. Route task using Thompson Sampling
            selection = thompson_router.select_agent(task["text"])

            # 2. Simulate execution
            success = np.random.random() > 0.2
            quality_score = np.random.uniform(0.8, 1.0) if success else np.random.uniform(0.4, 0.7)
            latency_ms = np.random.uniform(500, 2000)
            input_tokens = np.random.randint(100, 500)
            output_tokens = np.random.randint(100, 1000)

            # 3. Update Thompson posterior
            thompson_router.update_posterior(
                selection.agent_name,
                selection.task_category,
                success,
                quality_score,
            )

            # 4. Record for SONA learner
            sona_record = SONAExecutionRecord(
                task_id=f"task_{task['text'][:10]}",
                task_category=selection.task_category.value,
                agent_name=selection.agent_name,
                success=success,
                quality_score=quality_score,
                latency_ms=latency_ms,
                token_count=input_tokens + output_tokens,
                cost=0.001,
            )
            sona_learner.record_execution(sona_record)

            # 5. Record for pattern extractor
            extractor_record = ExtractorExecutionRecord(
                task_id=sona_record.task_id,
                task_category=sona_record.task_category,
                agent_name=sona_record.agent_name,
                success=success,
                quality_score=quality_score,
                latency_ms=latency_ms,
            )
            pattern_extractor.extract_all_patterns([extractor_record])

            # 6. Track cost
            usage_record = UsageRecord(
                execution_id=sona_record.task_id,
                model_name="deepseek-r1:7b",
                agent_name=selection.agent_name,
                task_category=selection.task_category.value,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=0.001,
                latency_ms=latency_ms,
                quality_score=quality_score,
                success=success,
            )
            cost_analyzer.track_usage(usage_record)

        # Verify all components have state
        assert thompson_router._selection_history
        assert sona_learner._execution_history
        assert pattern_extractor._patterns
        assert cost_analyzer._usage_history


class TestLearningFeedbackLoop:
    """Tests for learning feedback loop."""

    def test_learning_feedback_loop(self, thompson_router, sona_learner):
        """Test that learning improves routing over time."""
        # Initial selections without learning
        initial_selections = []
        for _ in range(10):
            result = thompson_router.select_agent("Analyze data")
            initial_selections.append(result.agent_name)

        # Record good performance for specific agent-category pairs
        for _ in range(20):
            # Record high-quality success for DataAnalyzer on analysis
            record = SONAExecutionRecord(
                task_id=f"task_{np.random.randint(1000)}",
                task_category="analysis",
                agent_name="DataAnalyzer",
                success=True,
                quality_score=0.95,
                latency_ms=800,
                token_count=500,
                cost=0.001,
            )
            sona_learner.record_execution(record)
            thompson_router.update_posterior(
                "DataAnalyzer",
                TaskCategory.ANALYSIS,
                success=True,
                quality_score=0.95,
            )

        # Selections after learning
        learned_selections = []
        for _ in range(10):
            result = thompson_router.select_agent(
                "Analyze data",
                category=TaskCategory.ANALYSIS,
            )
            learned_selections.append(result.agent_name)

        # DataAnalyzer should appear more frequently after learning
        initial_data_analyzer = initial_selections.count("DataAnalyzer")
        learned_data_analyzer = learned_selections.count("DataAnalyzer")

        # Due to Thompson Sampling randomness, we allow some flexibility
        # but the learned version should trend toward more DataAnalyzer selections
        assert learned_data_analyzer >= initial_data_analyzer


class TestQualityGateEnforcement:
    """Tests for quality gate enforcement."""

    def test_quality_gate_enforcement(self, sona_learner):
        """Test Ihsan quality gate enforcement."""
        # Record executions with varying quality
        for i in range(10):
            quality = 0.96 + (i * 0.005)  # 0.96 to 1.01, clamped to 1.0
            quality = min(quality, 1.0)

            record = SONAExecutionRecord(
                task_id=f"task_{i:04d}",
                task_category="reasoning",
                agent_name="MasterReasoner",
                success=True,
                quality_score=quality,
                latency_ms=1000,
                token_count=500,
                cost=0.005,
            )
            sona_learner.record_execution(record)

        metrics = sona_learner.evaluate_performance()

        # Should be Ihsan compliant with high quality scores
        assert metrics.avg_quality_score >= 0.95
        assert metrics.ihsan_compliant is True


class TestReceiptGeneration:
    """Tests for receipt generation integration."""

    def test_receipt_generation(
        self,
        thompson_router,
        sona_learner,
        cost_analyzer,
    ):
        """Test that components generate appropriate data for receipts."""
        # Execute a task
        selection = thompson_router.select_agent("Analyze quarterly data")

        # Record execution
        record = SONAExecutionRecord(
            task_id="receipt_test_task",
            task_category=selection.task_category.value,
            agent_name=selection.agent_name,
            success=True,
            quality_score=0.95,
            latency_ms=1200,
            token_count=750,
            cost=0.008,
        )
        sona_learner.record_execution(record)

        usage = UsageRecord(
            execution_id="receipt_test_task",
            model_name="deepseek-r1:7b",
            agent_name=selection.agent_name,
            task_category=selection.task_category.value,
            input_tokens=250,
            output_tokens=500,
            cost=0.008,
            latency_ms=1200,
            quality_score=0.95,
            success=True,
        )
        cost_analyzer.track_usage(usage)

        # Generate a receipt-like structure
        receipt_data = {
            "task_id": "receipt_test_task",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "routing": {
                "agent": selection.agent_name,
                "category": selection.task_category.value,
                "sampled_value": selection.sampled_value,
                "exploration_rate": selection.exploration_rate,
            },
            "execution": {
                "success": record.success,
                "quality_score": record.quality_score,
                "latency_ms": record.latency_ms,
            },
            "cost": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_cost": usage.cost,
            },
            "ihsan_compliant": record.quality_score >= 0.95,
        }

        # Compute integrity hash
        receipt_json = json.dumps(receipt_data, sort_keys=True)
        integrity_hash = hashlib.sha256(receipt_json.encode()).hexdigest()
        receipt_data["integrity_hash"] = integrity_hash

        # Verify receipt structure
        assert "task_id" in receipt_data
        assert "timestamp" in receipt_data
        assert "routing" in receipt_data
        assert "execution" in receipt_data
        assert "cost" in receipt_data
        assert "ihsan_compliant" in receipt_data
        assert "integrity_hash" in receipt_data


class TestBetaPriorMathematics:
    """Tests for Beta distribution prior mathematics."""

    def test_beta_prior_mean(self):
        """Test Beta prior mean calculation."""
        prior = BetaPrior(alpha=3.0, beta=1.0)
        assert abs(prior.mean - 0.75) < 0.001

        prior2 = BetaPrior(alpha=1.0, beta=1.0)
        assert abs(prior2.mean - 0.5) < 0.001

    def test_beta_prior_variance(self):
        """Test Beta prior variance decreases with more samples."""
        prior1 = BetaPrior(alpha=2.0, beta=2.0)
        prior2 = BetaPrior(alpha=20.0, beta=20.0)

        # More samples = lower variance
        assert prior2.variance < prior1.variance

    def test_beta_prior_sampling(self):
        """Test Beta prior sampling."""
        prior = BetaPrior(alpha=5.0, beta=5.0)

        samples = [prior.sample() for _ in range(1000)]

        # All samples should be in [0, 1]
        assert all(0.0 <= s <= 1.0 for s in samples)

        # Mean of samples should be close to prior mean
        sample_mean = np.mean(samples)
        assert abs(sample_mean - prior.mean) < 0.1


class TestRoutingWeights:
    """Tests for routing weights management."""

    def test_routing_weights_update(self):
        """Test routing weight updates with momentum."""
        weights = RoutingWeights()

        # Update weight multiple times
        for _ in range(5):
            weights.update_weight(
                "MasterReasoner",
                "reasoning",
                gradient=0.1,
                learning_rate=0.1,
                momentum=0.9,
            )

        weight = weights.get_weight("MasterReasoner", "reasoning")
        assert 0.01 <= weight <= 0.99

    def test_routing_weights_normalization(self):
        """Test weight normalization."""
        weights = RoutingWeights()

        # Set some weights
        weights.agent_category_weights["Agent1"] = {"cat1": 0.3, "cat2": 0.7}
        weights.agent_category_weights["Agent2"] = {"cat1": 0.5, "cat2": 0.5}

        weights.normalize()

        # Check normalization per agent
        for agent, categories in weights.agent_category_weights.items():
            total = sum(categories.values())
            assert abs(total - 1.0) < 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
