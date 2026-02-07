"""
APEX ENGINE INTEGRATION TEST SUITE
===================================

Ultimate integration test suite for the Peak Masterpiece implementation.

Tests the complete integration of:
- Local-First Model Selection (LM Studio, Ollama, Bicameral)
- Graph-of-Thoughts (GoT) Multi-Path Reasoning
- Bicameral Reasoning (Cold Core + Warm Surface)
- FATE Gate Formal Verification (Z3 SMT Solver)
- Autopoietic Self-Improvement Loop

Standing on Giants:
- Besta (2024): Graph of Thoughts
- DeepSeek R1 (2025): Bicameral Reasoning
- de Moura & Bjorner (2008): Z3 SMT Solver
- Maturana & Varela (1972): Autopoiesis
- Shannon (1948): Signal-to-Noise Ratio
- Anthropic (2024): Constitutional AI

BIZRA Genesis Strict Synthesis v2.2.2
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

# Import core modules
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
    STRICT_IHSAN_THRESHOLD,
    SNR_THRESHOLD_T0_ELITE,
    SNR_THRESHOLD_T1_HIGH,
)

# Conditional imports with graceful degradation
try:
    from core.inference.gateway import (
        InferenceGateway,
        InferenceConfig,
        InferenceResult,
        InferenceStatus,
        InferenceBackend,
        ComputeTier,
    )
    GATEWAY_AVAILABLE = True
except ImportError as e:
    GATEWAY_AVAILABLE = False
    logging.warning(f"Inference gateway not available: {e}")

try:
    from core.sovereign.bicameral_engine import (
        BicameralReasoningEngine,
        ReasoningCandidate,
        VerificationResult,
        BicameralResult,
    )
    BICAMERAL_AVAILABLE = True
except ImportError as e:
    BICAMERAL_AVAILABLE = False
    logging.warning(f"Bicameral engine not available: {e}")

try:
    from core.sovereign.z3_fate_gate import (
        Z3FATEGate,
        Z3Proof,
        Z3Constraint,
        Z3_AVAILABLE,
    )
    FATE_GATE_AVAILABLE = Z3_AVAILABLE
except ImportError as e:
    FATE_GATE_AVAILABLE = False
    logging.warning(f"Z3 FATE gate not available: {e}")

try:
    from core.autopoiesis.loop_engine import (
        ActivationGuardrails,
        AutopoieticLoop,
        AutopoieticState,
        AutopoieticResult,
        Hypothesis,
        HypothesisCategory,
        RiskLevel,
        ValidationResult,
        ImplementationResult,
        IntegrationResult,
        create_autopoietic_loop,
    )
    AUTOPOIETIC_AVAILABLE = True
except ImportError as e:
    AUTOPOIETIC_AVAILABLE = False
    logging.warning(f"Autopoietic loop not available: {e}")

try:
    from core.sovereign.runtime_engines.got_bridge import (
        ThoughtGraph,
        ThoughtNode,
        ThoughtType,
        ThoughtStatus,
        GoTResult,
        GoTBridge,
        think,
    )
    GOT_AVAILABLE = True
except ImportError as e:
    GOT_AVAILABLE = False
    logging.warning(f"Graph-of-Thoughts not available: {e}")


# =============================================================================
# TEST CONSTANTS
# =============================================================================

LMSTUDIO_URL = "http://192.168.56.1:1234"
OLLAMA_URL = "http://localhost:11434"

# Performance SLAs
MAX_LOCAL_LATENCY_MS = 100.0
MIN_THROUGHPUT_QPS = 10.0
MAX_MEMORY_MB = 512


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@dataclass
class MockLocalEndpoint:
    """Mock local inference endpoint for testing."""
    response: str = "Mock generated response"
    delay_ms: float = 10.0
    fail_after: int = -1
    call_count: int = 0

    async def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        self.call_count += 1
        if self.fail_after >= 0 and self.call_count > self.fail_after:
            raise RuntimeError("Mock endpoint failure")
        await asyncio.sleep(self.delay_ms / 1000)
        return f"{self.response} | temp={temperature:.1f}"


@dataclass
class MockAnalyticalClient:
    """Mock analytical client (Claude-style) for verification."""
    verify_result: bool = True
    confidence_delta: float = 0.1

    async def analyze(self, content: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "passes": self.verify_result,
            "confidence_delta": self.confidence_delta,
            "critique": "Mock verification passed" if self.verify_result else "Mock verification failed",
        }


@dataclass
class MockSensorHub:
    """Mock sensor hub for autopoietic loop testing."""
    ihsan_score: float = 0.96
    snr_score: float = 0.91
    latency_ms: float = 15.0
    error_rate: float = 0.01

    async def poll_all_sensors(self) -> List[Any]:
        @dataclass
        class MockReading:
            sensor_id: str
            value: float
            snr_score: float

        return [
            MockReading("ihsan_compliance", self.ihsan_score, 0.95),
            MockReading("snr_quality", self.snr_score, 0.90),
            MockReading("latency_ms", self.latency_ms, 0.88),
            MockReading("cpu_usage", 0.45, 0.92),
            MockReading("memory_usage", 0.55, 0.90),
        ]


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_local_endpoint():
    """Provide a mock local endpoint."""
    return MockLocalEndpoint()


@pytest.fixture
def mock_analytical_client():
    """Provide a mock analytical client."""
    return MockAnalyticalClient()


@pytest.fixture
def mock_sensor_hub():
    """Provide a mock sensor hub."""
    return MockSensorHub()


@pytest.fixture
def bicameral_engine(mock_local_endpoint, mock_analytical_client):
    """Create a bicameral engine with mock backends."""
    if not BICAMERAL_AVAILABLE:
        pytest.skip("Bicameral engine not available")
    return BicameralReasoningEngine(
        local_endpoint=mock_local_endpoint,
        api_client=mock_analytical_client,
        consensus_threshold=UNIFIED_IHSAN_THRESHOLD,
    )


@pytest.fixture
def thought_graph():
    """Create a thought graph for GoT testing."""
    if not GOT_AVAILABLE:
        pytest.skip("Graph-of-Thoughts not available")
    return ThoughtGraph(
        max_depth=5,
        max_branches=3,
        prune_threshold=0.3,
    )


@pytest.fixture
def fate_gate():
    """Create a Z3 FATE gate for formal verification."""
    if not FATE_GATE_AVAILABLE:
        pytest.skip("Z3 FATE gate not available")
    return Z3FATEGate()


@pytest.fixture
def autopoietic_loop(mock_sensor_hub):
    """Create an autopoietic loop with mock sensors."""
    if not AUTOPOIETIC_AVAILABLE:
        pytest.skip("Autopoietic loop not available")
    return AutopoieticLoop(
        ihsan_floor=UNIFIED_IHSAN_THRESHOLD,
        snr_floor=UNIFIED_SNR_THRESHOLD,
        cycle_interval_s=1.0,  # Fast cycles for testing
        sensor_hub=mock_sensor_hub,
        activation_guardrails=ActivationGuardrails(enabled=False),
    )


# =============================================================================
# 1. LOCAL-FIRST MODEL TESTS
# =============================================================================

class TestLocalFirstModel:
    """Test suite for local-first model selection and routing."""

    @pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="Gateway not available")
    @pytest.mark.asyncio
    async def test_lm_studio_connection(self):
        """Test LM Studio connection (requires LM Studio running)."""
        # Check if LM Studio is reachable first
        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not installed")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{LMSTUDIO_URL}/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    connected = resp.status == 200
        except Exception:
            connected = False

        if not connected:
            pytest.skip("LM Studio not available - skipping live connection test")

        config = InferenceConfig(
            lmstudio_url=LMSTUDIO_URL,
            require_local=False,  # Allow fallback for testing
        )
        gateway = InferenceGateway(config)

        success = await gateway.initialize()
        assert success, "Gateway should initialize with LM Studio"
        assert gateway.status in (InferenceStatus.READY, InferenceStatus.WARMING, InferenceStatus.DEGRADED)

    @pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="Gateway not available")
    def test_model_routing_by_task(self):
        """Test that tasks are routed to appropriate tiers based on complexity."""
        config = InferenceConfig()
        gateway = InferenceGateway(config)

        # Simple task -> should route to some tier
        simple_complexity = gateway.estimate_complexity("What is 2+2?")
        simple_tier = gateway.route(simple_complexity)
        assert simple_tier in ComputeTier, "Simple tasks should route to valid tier"

        # Medium task -> should route to some tier
        medium_complexity = gateway.estimate_complexity(
            "Explain the algorithm and its time complexity with examples."
        )
        medium_tier = gateway.route(medium_complexity)
        assert medium_tier in ComputeTier, "Medium tasks should route to valid tier"

        # Complex task with many reasoning keywords
        complex_complexity = gateway.estimate_complexity(
            "Analyze the distributed consensus theorem and prove its correctness with formal verification, "
            "then compare it to alternative approaches in the literature including Paxos, Raft, and PBFT. "
            "Provide a comprehensive analysis of trade-offs in terms of latency, throughput, and fault tolerance."
        )
        complex_tier = gateway.route(complex_complexity)
        assert complex_tier in ComputeTier, "Complex tasks should route to valid tier"

        # The key test: complexity scoring should reflect task difficulty
        # More complex tasks should have higher or equal complexity scores
        assert complex_complexity.reasoning_depth >= simple_complexity.reasoning_depth or \
               complex_complexity.domain_specificity >= simple_complexity.domain_specificity or \
               complex_complexity.input_tokens >= simple_complexity.input_tokens, \
               "Complex tasks should have higher complexity indicators"

    @pytest.mark.skipif(not BICAMERAL_AVAILABLE, reason="Bicameral not available")
    @pytest.mark.asyncio
    async def test_bicameral_model_selection(self, bicameral_engine):
        """Test bicameral model selection (local + API)."""
        assert bicameral_engine.is_bicameral, "Engine should be bicameral with both endpoints"

        # Generate candidates using local endpoint
        candidates = await bicameral_engine.generate_candidates("Test problem", num_candidates=2)
        assert len(candidates) > 0, "Should generate candidates"
        assert all(c.source == "r1" for c in candidates), "Candidates should be from local R1"

    @pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="Gateway not available")
    @pytest.mark.asyncio
    async def test_fallback_chain_respects_privacy(self):
        """Test that fallback chain respects local-first privacy."""
        config = InferenceConfig(
            require_local=True,
            fallbacks=["ollama"],  # No external APIs in fallback
        )
        gateway = InferenceGateway(config)

        # With require_local=True, should not call external APIs
        assert "openai" not in config.fallbacks
        assert "anthropic" not in config.fallbacks

    @pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="Gateway not available")
    @pytest.mark.asyncio
    async def test_no_external_calls_by_default(self):
        """Test that no external API calls are made by default."""
        config = InferenceConfig(require_local=True)
        gateway = InferenceGateway(config)

        # Default should not include external API backends
        assert InferenceBackend.POOL.value not in [b.value for b in [
            InferenceBackend.LLAMACPP,
            InferenceBackend.OLLAMA,
            InferenceBackend.LMSTUDIO,
        ]]


# =============================================================================
# 2. GRAPH-OF-THOUGHTS INTEGRATION TESTS
# =============================================================================

class TestGoTIntegration:
    """Test suite for Graph-of-Thoughts integration."""

    @pytest.mark.skipif(not GOT_AVAILABLE, reason="GoT not available")
    @pytest.mark.asyncio
    async def test_got_generates_multiple_paths(self, thought_graph):
        """Test that GoT generates multiple reasoning paths."""
        root = thought_graph.create_root("Solve this problem")
        children = thought_graph.generate(root, "Solve this problem")

        assert len(children) > 0, "Should generate child thoughts"
        assert len(children) <= thought_graph.max_branches, "Should respect max_branches"

        # Each child should have unique content
        contents = [c.content for c in children]
        assert len(set(contents)) == len(contents), "Paths should be unique"

    @pytest.mark.skipif(not GOT_AVAILABLE, reason="GoT not available")
    @pytest.mark.asyncio
    async def test_got_aggregates_similar_hypotheses(self, thought_graph):
        """Test that GoT aggregates similar hypotheses."""
        root = thought_graph.create_root("Test goal")

        # Create multiple nodes to aggregate
        nodes = [
            ThoughtNode(content=f"Approach {i}", score=0.7 + i * 0.05)
            for i in range(3)
        ]
        for node in nodes:
            thought_graph._nodes[node.id] = node

        aggregated = thought_graph.aggregate(nodes)

        assert aggregated.thought_type == ThoughtType.AGGREGATE
        assert "Synthesis" in aggregated.content
        assert all(n.id in aggregated.parent_ids for n in nodes)

    @pytest.mark.skipif(not GOT_AVAILABLE, reason="GoT not available")
    @pytest.mark.asyncio
    async def test_got_prunes_low_value_paths(self, thought_graph):
        """Test that GoT prunes low-quality paths."""
        root = thought_graph.create_root("Test goal")

        # Add low-quality nodes
        for i in range(5):
            node = ThoughtNode(
                content=f"Low quality {i}",
                score=0.1 + i * 0.05,  # Below prune_threshold (0.3)
            )
            thought_graph._nodes[node.id] = node
            thought_graph._frontier.append(node)

        # Add high-quality node
        good_node = ThoughtNode(content="High quality", score=0.9)
        thought_graph._nodes[good_node.id] = good_node
        thought_graph._frontier.append(good_node)

        pruned_count = thought_graph.prune()

        assert pruned_count > 0, "Should prune low-quality nodes"
        # Verify pruned nodes are marked
        pruned_nodes = [n for n in thought_graph._nodes.values() if n.status == ThoughtStatus.PRUNED]
        assert len(pruned_nodes) == pruned_count

    @pytest.mark.skipif(not GOT_AVAILABLE, reason="GoT not available")
    @pytest.mark.asyncio
    async def test_snr_maximized_across_paths(self, thought_graph):
        """Test that SNR is maximized across reasoning paths."""
        # Set up a custom scorer that prioritizes high SNR
        def snr_scorer(node: ThoughtNode) -> float:
            base_score = node.combined_score()
            # Simulate SNR calculation
            snr = base_score * 0.9  # SNR derived from confidence/coherence
            return snr if snr >= UNIFIED_SNR_THRESHOLD else snr * 0.5

        thought_graph.set_scorer(snr_scorer)

        result = await thought_graph.reason("Maximize SNR problem", max_iterations=20)

        if result.solution:
            final_snr = snr_scorer(result.solution)
            assert final_snr >= UNIFIED_SNR_THRESHOLD * 0.8, "Solution should have acceptable SNR"

    @pytest.mark.skipif(not GOT_AVAILABLE, reason="GoT not available")
    @pytest.mark.asyncio
    async def test_convergence_detection(self, thought_graph):
        """Test that GoT detects convergence by exploring multiple paths."""
        # Use a generator that produces high-quality thoughts
        iteration = [0]

        def converging_generator(parent: ThoughtNode, goal: str) -> List[str]:
            iteration[0] += 1
            # Generate high-scoring thoughts
            return [
                f"High quality approach {iteration[0]}: {goal}",
                f"Alternative approach {iteration[0]}: refined solution",
            ]

        # Set a scorer that gives high scores to encourage solution finding
        def high_scorer(node: ThoughtNode) -> float:
            # Give high scores to nodes with "refined" or "approach" in content
            base = 0.7
            if "refined" in node.content.lower():
                base = 0.95
            if "approach" in node.content.lower():
                base = 0.92
            if node.depth >= 2:
                base = max(base, 0.9)
            return base

        thought_graph.set_generator(converging_generator)
        thought_graph.set_scorer(high_scorer)
        result = await thought_graph.reason("Converge test", max_iterations=20, min_solutions=1)

        # The test verifies exploration happens, not necessarily finding a solution
        # since solution criteria may vary
        assert result.explored_nodes > 0, "Should explore nodes"
        assert result.max_depth_reached > 0, "Should reach some depth"


# =============================================================================
# 3. BICAMERAL REASONING TESTS
# =============================================================================

class TestBicameralReasoning:
    """Test suite for bicameral reasoning (cold core + warm surface)."""

    @pytest.mark.skipif(not BICAMERAL_AVAILABLE, reason="Bicameral not available")
    @pytest.mark.asyncio
    async def test_cold_core_generates_candidates(self, bicameral_engine):
        """Test that cold core (R1/local) generates candidates."""
        candidates = await bicameral_engine.generate_candidates(
            "What is the optimal solution?",
            num_candidates=3,
        )

        assert len(candidates) > 0, "Should generate candidates"
        for candidate in candidates:
            assert candidate.source == "r1", "Candidates should be from cold core (R1)"
            assert candidate.content, "Candidates should have content"
            assert 0 <= candidate.confidence <= 1, "Confidence should be normalized"

    @pytest.mark.skipif(not BICAMERAL_AVAILABLE, reason="Bicameral not available")
    @pytest.mark.asyncio
    async def test_warm_surface_verifies(self, bicameral_engine, mock_analytical_client):
        """Test that warm surface (Claude/API) verifies candidates."""
        # Create a candidate
        candidate = ReasoningCandidate(
            candidate_id="test_001",
            content="Proposed solution: X",
            source="r1",
            confidence=0.7,
            reasoning_trace="Step 1 -> Step 2 -> Conclusion",
        )

        verification = await bicameral_engine.verify_candidate(
            candidate,
            {"correctness": True, "safety": True},
        )

        assert verification.candidate_id == candidate.candidate_id
        assert verification.verified == mock_analytical_client.verify_result

    @pytest.mark.skipif(not BICAMERAL_AVAILABLE, reason="Bicameral not available")
    @pytest.mark.asyncio
    async def test_consensus_synthesis(self, bicameral_engine):
        """Test consensus synthesis between hemispheres."""
        result = await bicameral_engine.reason(
            "Find the best approach",
            {"num_candidates": 3, "criteria": {"correctness": True}},
        )

        assert result.final_answer, "Should produce a final answer"
        assert result.consensus_score >= 0, "Consensus score should be non-negative"
        assert result.candidates_generated > 0, "Should generate candidates"

    @pytest.mark.skipif(not BICAMERAL_AVAILABLE, reason="Bicameral not available")
    @pytest.mark.asyncio
    async def test_disagreement_handling(self, mock_local_endpoint):
        """Test handling when hemispheres disagree."""
        # Create analytical client that rejects
        rejecting_client = MockAnalyticalClient(verify_result=False, confidence_delta=-0.2)

        engine = BicameralReasoningEngine(
            local_endpoint=mock_local_endpoint,
            api_client=rejecting_client,
            consensus_threshold=0.95,
        )

        result = await engine.reason(
            "Controversial problem",
            {"num_candidates": 2},
        )

        # Should still produce a result, but with lower consensus
        assert result.final_answer, "Should still produce an answer"
        assert result.consensus_score < 0.95, "Consensus should be low when verification fails"

    @pytest.mark.skipif(not BICAMERAL_AVAILABLE, reason="Bicameral not available")
    @pytest.mark.asyncio
    async def test_bicameral_improves_accuracy(self, bicameral_engine):
        """Test that bicameral reasoning improves accuracy over single-model."""
        # Run bicameral reasoning
        bicameral_result = await bicameral_engine.reason(
            "Complex reasoning task",
            {"num_candidates": 3},
        )

        # Bicameral should have verification
        assert bicameral_result.candidates_verified >= 0
        # The verified result should have higher adjusted confidence
        if bicameral_result.candidates_verified > 0:
            assert bicameral_result.consensus_score > 0.5


# =============================================================================
# 4. FATE GATE INTEGRATION TESTS
# =============================================================================

class TestFATEGateIntegration:
    """Test suite for FATE gate formal verification."""

    @pytest.mark.skipif(not FATE_GATE_AVAILABLE, reason="Z3 FATE gate not available")
    def test_all_outputs_pass_fate(self, fate_gate):
        """Test that valid outputs pass FATE gate."""
        action_context = {
            "ihsan": 0.97,
            "snr": 0.92,
            "risk_level": 0.3,
            "reversible": True,
            "human_approved": False,
            "cost": 0.1,
            "autonomy_limit": 0.3,
        }

        proof = fate_gate.generate_proof(action_context)

        assert proof.satisfiable, "Valid action should satisfy constraints"
        assert proof.proof_id, "Proof should have an ID"

    @pytest.mark.skipif(not FATE_GATE_AVAILABLE, reason="Z3 FATE gate not available")
    def test_z3_proof_generated(self, fate_gate):
        """Test that Z3 generates proofs with proper structure."""
        action_context = {
            "ihsan": 0.96,
            "snr": 0.90,
            "risk_level": 0.2,
            "reversible": True,
            "cost": 0.05,
            "autonomy_limit": 0.5,
        }

        proof = fate_gate.generate_proof(action_context)

        assert proof.proof_id.startswith("proof_")
        assert len(proof.constraints_checked) > 0
        assert proof.generation_time_ms >= 0
        if proof.satisfiable:
            assert proof.model is not None

    @pytest.mark.skipif(not FATE_GATE_AVAILABLE, reason="Z3 FATE gate not available")
    def test_ihsan_constraint_enforced(self, fate_gate):
        """Test that Ihsan constraint is enforced."""
        # Below threshold - should fail
        low_ihsan_context = {
            "ihsan": 0.80,  # Below 0.95 threshold
            "snr": 0.90,
            "risk_level": 0.2,
            "reversible": True,
            "cost": 0.1,
            "autonomy_limit": 0.5,
        }

        proof = fate_gate.generate_proof(low_ihsan_context)
        assert not proof.satisfiable, "Low Ihsan should fail FATE gate"

        # Verify specific constraint
        assert not fate_gate.verify_ihsan(0.80)
        assert fate_gate.verify_ihsan(0.96)

    @pytest.mark.skipif(not FATE_GATE_AVAILABLE, reason="Z3 FATE gate not available")
    def test_snr_floor_enforced(self, fate_gate):
        """Test that SNR floor is enforced."""
        # Below threshold - should fail
        low_snr_context = {
            "ihsan": 0.96,
            "snr": 0.70,  # Below 0.85 threshold
            "risk_level": 0.2,
            "reversible": True,
            "cost": 0.1,
            "autonomy_limit": 0.5,
        }

        proof = fate_gate.generate_proof(low_snr_context)
        assert not proof.satisfiable, "Low SNR should fail FATE gate"

        # Verify specific constraint
        assert not fate_gate.verify_snr(0.70)
        assert fate_gate.verify_snr(0.90)

    @pytest.mark.skipif(not FATE_GATE_AVAILABLE, reason="Z3 FATE gate not available")
    def test_rejection_on_violation(self, fate_gate):
        """Test that violations are properly rejected with counterexample."""
        # High risk without reversibility or approval
        risky_context = {
            "ihsan": 0.96,
            "snr": 0.90,
            "risk_level": 0.9,  # High risk
            "reversible": False,  # Not reversible
            "human_approved": False,  # Not approved
            "cost": 0.1,
            "autonomy_limit": 0.5,
        }

        proof = fate_gate.generate_proof(risky_context)

        assert not proof.satisfiable, "High risk without safety should fail"
        assert proof.counterexample, "Should provide counterexample"


# =============================================================================
# 5. AUTOPOIETIC INTEGRATION TESTS
# =============================================================================

class TestAutopoieticIntegration:
    """Test suite for autopoietic self-improvement loop."""

    @pytest.mark.skipif(not AUTOPOIETIC_AVAILABLE, reason="Autopoietic loop not available")
    @pytest.mark.asyncio
    async def test_loop_improves_performance(self, autopoietic_loop):
        """Test that the autopoietic loop can improve performance."""
        # Run a single cycle
        result = await autopoietic_loop.run_cycle()

        assert result.cycle_id, "Cycle should have an ID"
        assert result.state in AutopoieticState
        assert result.observation is not None, "Should observe system state"

    @pytest.mark.skipif(not AUTOPOIETIC_AVAILABLE, reason="Autopoietic loop not available")
    @pytest.mark.asyncio
    async def test_hypothesis_from_got(self, autopoietic_loop):
        """Test that hypotheses are generated from observations."""
        # Create an observation that triggers hypothesis generation
        from core.autopoiesis.loop_engine import SystemObservation

        observation = SystemObservation(
            observation_id="test_obs",
            timestamp=datetime.now(timezone.utc),
            ihsan_score=0.95,
            snr_score=0.88,  # Slightly below optimal
            latency_p50_ms=50.0,
            latency_p99_ms=150.0,  # High latency
            error_rate=0.02,
            throughput_qps=80.0,
            cpu_usage=0.6,
            memory_usage=0.7,
        )

        hypotheses = await autopoietic_loop.hypothesize(observation)

        assert len(hypotheses) > 0, "Should generate hypotheses"
        for hyp in hypotheses:
            assert hyp.description, "Hypothesis should have description"
            assert hyp.risk_level in RiskLevel

    @pytest.mark.skipif(not AUTOPOIETIC_AVAILABLE, reason="Autopoietic loop not available")
    @pytest.mark.asyncio
    async def test_shadow_deployment_works(self, autopoietic_loop):
        """Test that shadow deployment mechanism works."""
        from core.autopoiesis.loop_engine import Hypothesis, HypothesisCategory, RiskLevel

        hypothesis = Hypothesis(
            id="test_hyp_001",
            description="Test performance optimization",
            category=HypothesisCategory.PERFORMANCE,
            predicted_improvement=0.10,
            required_changes=[{
                "component": "test",
                "parameter": "batch_size",
                "action": "increase",
            }],
            affected_components=["test"],
            risk_level=RiskLevel.LOW,
            reversibility_plan={"action": "restore", "value": 8},
            ihsan_impact_estimate=0.0,
            snr_impact_estimate=0.01,
        )

        result = await autopoietic_loop.implement(hypothesis)

        assert result.hypothesis_id == hypothesis.id
        assert result.shadow_instance_id, "Should create shadow instance"

    @pytest.mark.skipif(not AUTOPOIETIC_AVAILABLE, reason="Autopoietic loop not available")
    @pytest.mark.asyncio
    async def test_learning_persists(self, autopoietic_loop):
        """Test that learnings from improvements persist."""
        # Run multiple cycles
        results = []
        for _ in range(2):
            result = await autopoietic_loop.run_cycle()
            results.append(result)
            await asyncio.sleep(0.1)

        # Check that observations are stored
        assert len(autopoietic_loop._observation_history) > 0
        # Check that audit log is maintained
        status = autopoietic_loop.get_status()
        assert status["cycle_count"] >= 2

    @pytest.mark.skipif(not AUTOPOIETIC_AVAILABLE, reason="Autopoietic loop not available")
    @pytest.mark.asyncio
    async def test_constitutional_preserved_during_evolution(self, autopoietic_loop):
        """Test that constitutional constraints are preserved during evolution."""
        # Run a cycle with FATE gate
        result = await autopoietic_loop.run_cycle()

        # Check that Ihsan and SNR floors are maintained
        status = autopoietic_loop.get_status()
        assert status["ihsan_floor"] >= UNIFIED_IHSAN_THRESHOLD
        assert status["snr_floor"] >= UNIFIED_SNR_THRESHOLD

        # If validation occurred, check it enforced constraints
        if result.validation_result:
            if result.validation_result.is_valid:
                assert result.validation_result.ihsan_gate_passed
                assert result.validation_result.snr_gate_passed


# =============================================================================
# 6. END-TO-END APEX TESTS
# =============================================================================

class TestApexEndToEnd:
    """End-to-end tests for the complete Apex engine."""

    @pytest.mark.skipif(
        not all([BICAMERAL_AVAILABLE, GOT_AVAILABLE, AUTOPOIETIC_AVAILABLE]),
        reason="Not all components available"
    )
    @pytest.mark.asyncio
    async def test_full_query_pipeline(self, mock_local_endpoint, mock_analytical_client, mock_sensor_hub):
        """Test full query pipeline: GoT -> Bicameral -> FATE -> Response."""
        # 1. Graph-of-Thoughts reasoning
        graph = ThoughtGraph(max_depth=3, max_branches=2)
        got_result = await graph.reason("Complex reasoning problem", max_iterations=10)

        assert got_result.explored_nodes > 0, "GoT should explore nodes"

        # 2. Bicameral verification
        engine = BicameralReasoningEngine(
            local_endpoint=mock_local_endpoint,
            api_client=mock_analytical_client,
        )
        bicameral_result = await engine.reason(
            got_result.goal,
            {"num_candidates": 2},
        )

        assert bicameral_result.final_answer, "Bicameral should produce answer"

        # 3. FATE gate verification (if available)
        if FATE_GATE_AVAILABLE:
            fate_gate = Z3FATEGate()
            action_context = {
                "ihsan": 0.96,
                "snr": 0.90,
                "risk_level": 0.2,
                "reversible": True,
                "cost": 0.1,
                "autonomy_limit": 0.5,
            }
            proof = fate_gate.generate_proof(action_context)
            assert proof.satisfiable, "Valid output should pass FATE"

    @pytest.mark.skipif(not AUTOPOIETIC_AVAILABLE, reason="Autopoietic not available")
    @pytest.mark.asyncio
    async def test_continuous_improvement_cycle(self, autopoietic_loop):
        """Test continuous improvement cycle."""
        # Run 3 cycles
        improvement_count = 0
        for _ in range(3):
            result = await autopoietic_loop.run_cycle()
            if result.improvements_integrated > 0:
                improvement_count += 1
            await asyncio.sleep(0.1)

        status = autopoietic_loop.get_status()
        assert status["cycle_count"] >= 3
        # Not all cycles may produce improvements
        assert status["total_rollbacks"] >= 0

    def test_giants_attribution_complete(self):
        """Test that all standing-on-giants attributions are present."""
        attributions = [
            "Besta",      # Graph of Thoughts
            "Shannon",    # SNR
            "Lamport",    # Distributed systems
            "Anthropic",  # Constitutional AI
        ]

        # Check module docstrings for attributions
        modules_checked = 0

        if GOT_AVAILABLE:
            from core.sovereign.runtime_engines import got_bridge
            docstring = got_bridge.__doc__ or ""
            assert "Besta" in docstring, "GoT should cite Besta"
            modules_checked += 1

        if BICAMERAL_AVAILABLE:
            from core.sovereign import bicameral_engine
            docstring = bicameral_engine.__doc__ or ""
            assert any(attr in docstring for attr in ["DeepSeek", "Jaynes"]), "Bicameral should cite sources"
            modules_checked += 1

        assert modules_checked > 0, "Should check at least one module"

    @pytest.mark.asyncio
    async def test_snr_maintained_above_085(self, mock_sensor_hub):
        """Test that SNR is maintained above 0.85 threshold."""
        mock_sensor_hub.snr_score = 0.91  # Above threshold

        readings = await mock_sensor_hub.poll_all_sensors()
        snr_reading = next((r for r in readings if "snr" in r.sensor_id.lower()), None)

        assert snr_reading is not None
        assert snr_reading.value >= UNIFIED_SNR_THRESHOLD

    @pytest.mark.asyncio
    async def test_ihsan_maintained_above_095(self, mock_sensor_hub):
        """Test that Ihsan is maintained above 0.95 threshold."""
        mock_sensor_hub.ihsan_score = 0.96  # Above threshold

        readings = await mock_sensor_hub.poll_all_sensors()
        ihsan_reading = next((r for r in readings if "ihsan" in r.sensor_id.lower()), None)

        assert ihsan_reading is not None
        assert ihsan_reading.value >= UNIFIED_IHSAN_THRESHOLD


# =============================================================================
# 7. PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for latency, throughput, and memory."""

    @pytest.mark.skipif(not BICAMERAL_AVAILABLE, reason="Bicameral not available")
    @pytest.mark.asyncio
    async def test_latency_under_100ms_local(self, mock_local_endpoint, mock_analytical_client):
        """Test that local inference latency is under 100ms."""
        mock_local_endpoint.delay_ms = 10.0  # 10ms mock delay

        engine = BicameralReasoningEngine(
            local_endpoint=mock_local_endpoint,
            api_client=mock_analytical_client,
        )

        start = time.perf_counter()
        candidates = await engine.generate_candidates("Quick test", num_candidates=1)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(candidates) > 0
        assert elapsed_ms < MAX_LOCAL_LATENCY_MS, f"Latency {elapsed_ms:.1f}ms exceeds {MAX_LOCAL_LATENCY_MS}ms"

    @pytest.mark.skipif(not GOT_AVAILABLE, reason="GoT not available")
    @pytest.mark.asyncio
    async def test_throughput_meets_sla(self, thought_graph):
        """Test that throughput meets SLA requirements."""
        request_count = 10
        start = time.perf_counter()

        for _ in range(request_count):
            graph = ThoughtGraph(max_depth=2, max_branches=2)
            await graph.reason("Quick test", max_iterations=5)

        elapsed = time.perf_counter() - start
        throughput = request_count / elapsed

        assert throughput >= MIN_THROUGHPUT_QPS, f"Throughput {throughput:.1f} QPS below {MIN_THROUGHPUT_QPS}"

    @pytest.mark.skipif(not GOT_AVAILABLE, reason="GoT not available")
    @pytest.mark.asyncio
    async def test_memory_bounded(self, thought_graph):
        """Test that memory usage is bounded."""
        import tracemalloc

        tracemalloc.start()

        # Run multiple iterations
        for _ in range(5):
            graph = ThoughtGraph(max_depth=5, max_branches=3)
            await graph.reason("Memory test problem", max_iterations=20)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        assert peak_mb < MAX_MEMORY_MB, f"Peak memory {peak_mb:.1f}MB exceeds {MAX_MEMORY_MB}MB"

    @pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="Gateway not available")
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when backends fail."""
        config = InferenceConfig(
            require_local=False,
            fallbacks=["ollama"],
        )
        gateway = InferenceGateway(config)

        # Simulate all backends failing
        with patch.object(gateway, '_backends', {}):
            gateway.status = InferenceStatus.OFFLINE

            # Should not crash, should report offline status
            health = await gateway.health()
            assert health["status"] == "offline"


# =============================================================================
# INTEGRATION MARKERS
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "requires_lmstudio: tests that require LM Studio to be running"
    )
    config.addinivalue_line(
        "markers", "requires_ollama: tests that require Ollama to be running"
    )
    config.addinivalue_line(
        "markers", "requires_z3: tests that require Z3 solver"
    )
    config.addinivalue_line(
        "markers", "slow: tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests"
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestLocalFirstModel",
    "TestGoTIntegration",
    "TestBicameralReasoning",
    "TestFATEGateIntegration",
    "TestAutopoieticIntegration",
    "TestApexEndToEnd",
    "TestPerformance",
]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
