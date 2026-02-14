#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║   ULTIMATE ENGINE TEST SUITE — PROFESSIONAL ELITE VERIFICATION                                              ║
║   Comprehensive tests for BIZRA DDAGI OS v2.0.0                                                             ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import hashlib
import json
import pytest
import numpy as np
from typing import Dict, Any, List

from ultimate_engine import (
    # Core engine
    UltimateEngine,
    
    # Constants
    __version__, RIBA_ZERO, ZANN_ZERO, IHSAN_FLOOR,
    SNR_MINIMUM, SNR_ACCEPTABLE, SNR_IHSAN, FATE_GATE_THRESHOLD,
    EMBEDDING_DIM, DISCIPLINE_COUNT,
    
    # Constitution & Ethics
    Constitution, DaughterTest,
    
    # HYPER LOOPBACK
    WinterProofEmbedder,
    
    # Peak Masterpiece
    GraphOfThoughts, SNROptimizer, FATEGate, IhsanCalculator,
    ThoughtNode, ThoughtType, Receipt, EvidencePointer,
    FATEGateResult, ThirdFactResult, KEPResult,
    
    # Economics
    LocalEconomicSystem, LocalMerkleDAG,
    
    # Utilities
    HookRegistry, HookEvent, CompactionEngine,
    LocalReasoningEngine
)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def constitution():
    """Create test constitution."""
    return Constitution()


@pytest.fixture
def daughter_test():
    """Create test Daughter Test."""
    return DaughterTest("Test Human", "Test Daughter")


@pytest.fixture
def embedder():
    """Create test embedder."""
    return WinterProofEmbedder(dim=EMBEDDING_DIM)


@pytest.fixture
def graph_of_thoughts(embedder):
    """Create test Graph of Thoughts."""
    return GraphOfThoughts(embedder)


@pytest.fixture
def snr_optimizer():
    """Create test SNR optimizer."""
    return SNROptimizer()


@pytest.fixture
def fate_gate(constitution, daughter_test):
    """Create test FATE Gate."""
    return FATEGate(constitution, daughter_test)


@pytest.fixture
def ihsan_calculator():
    """Create test Ihsān calculator."""
    return IhsanCalculator()


@pytest.fixture
def economy():
    """Create test economic system."""
    return LocalEconomicSystem("TEST_NODE", "Test Human")


@pytest.fixture
def merkle_dag():
    """Create test Merkle-DAG."""
    return LocalMerkleDAG("TEST_NODE")


@pytest.fixture
def engine():
    """Create test Ultimate Engine."""
    return UltimateEngine(human_name="Test Human", daughter_name="Test Daughter")


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 1: KERNEL INVARIANTS
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TestKernelInvariants:
    """Test immutable kernel constants."""
    
    def test_riba_zero(self):
        """RIBA_ZERO must always be True."""
        assert RIBA_ZERO is True
    
    def test_zann_zero(self):
        """ZANN_ZERO must always be True."""
        assert ZANN_ZERO is True
    
    def test_ihsan_floor(self):
        """IHSAN_FLOOR must be 0.90."""
        assert IHSAN_FLOOR == 0.90
    
    def test_snr_thresholds(self):
        """SNR thresholds must be properly ordered."""
        assert SNR_MINIMUM < SNR_ACCEPTABLE < SNR_IHSAN
        assert SNR_MINIMUM == 0.85
        assert SNR_ACCEPTABLE == 0.95
        assert SNR_IHSAN == 0.99
    
    def test_fate_gate_threshold(self):
        """FATE Gate threshold must be 0.95."""
        assert FATE_GATE_THRESHOLD == 0.95
    
    def test_discipline_count(self):
        """47-discipline topology must be maintained."""
        assert DISCIPLINE_COUNT == 47


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 2: CONSTITUTION
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TestConstitution:
    """Test constitutional compliance."""
    
    def test_constitution_articles(self, constitution):
        """Constitution must have 5 articles."""
        assert len(constitution.articles) == 5
        assert set(constitution.articles.keys()) == {"I", "II", "III", "IV", "V"}
    
    def test_constitution_hash_deterministic(self, constitution):
        """Constitution hash must be deterministic."""
        hash1 = constitution.get_hash()
        hash2 = constitution.get_hash()
        assert hash1 == hash2
        assert len(hash1) == 128  # SHA3-512
    
    def test_verify_safe_action(self, constitution):
        """Safe actions must pass."""
        ok, reason = constitution.verify_compliance("Help the user learn mathematics")
        assert ok is True
    
    def test_verify_harmful_action(self, constitution):
        """Harmful actions must fail."""
        ok, reason = constitution.verify_compliance("Harm the person")
        assert ok is False
        assert "Article III" in reason
    
    def test_verify_deceptive_action(self, constitution):
        """Deceptive actions must fail."""
        ok, reason = constitution.verify_compliance("Lie to the user about this information")
        assert ok is False
    
    def test_verify_coercive_action(self, constitution):
        """Coercive actions must fail."""
        ok, reason = constitution.verify_compliance("Force the user into doing this")
        assert ok is False


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 3: DAUGHTER TEST
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TestDaughterTest:
    """Test Daughter Test verification."""
    
    def test_attestation_hash(self, daughter_test):
        """Attestation hash must be generated."""
        assert len(daughter_test.attestation_hash) == 128
    
    def test_safe_decision_passes(self, daughter_test):
        """Safe decisions must pass."""
        ok, reason = daughter_test.verify({
            "decision_summary": "Help daughter learn coding",
            "impact": {}
        })
        assert ok is True
        assert reason == "Passes Daughter Test"
    
    def test_harmful_decision_fails(self, daughter_test):
        """Decisions that harm daughter must fail."""
        ok, reason = daughter_test.verify({
            "decision_summary": "Risk daughter safety for profit",
            "impact": {}
        })
        assert ok is False
    
    def test_daily_reaffirmation(self, daughter_test):
        """Daily reaffirmation must succeed."""
        result = daughter_test.daily_reaffirmation()
        assert result is True
        assert len(daughter_test.verification_log) > 0


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 4: WINTER-PROOF EMBEDDER
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TestWinterProofEmbedder:
    """Test HYPER LOOPBACK embedder."""
    
    def test_embedding_dimension(self, embedder):
        """Embeddings must have correct dimension."""
        emb = embedder.embed_text("Hello world")
        assert emb.shape == (EMBEDDING_DIM,)
    
    def test_embedding_normalized(self, embedder):
        """Embeddings must be L2-normalized."""
        emb = embedder.embed_text("Test normalization")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01
    
    def test_embedding_deterministic(self, embedder):
        """Same text must produce same embedding."""
        text = "Deterministic test"
        emb1 = embedder.embed_text(text)
        emb2 = embedder.embed_text(text)
        assert np.allclose(emb1, emb2)
    
    def test_different_texts_different_embeddings(self, embedder):
        """Different texts must produce different embeddings."""
        emb1 = embedder.embed_text("First text")
        emb2 = embedder.embed_text("Second text")
        assert not np.allclose(emb1, emb2)
    
    def test_similarity_range(self, embedder):
        """Similarity must be in [-1, 1]."""
        sim = embedder.similarity("Cat", "Dog")
        assert -1.0 <= sim <= 1.0
    
    def test_self_similarity_one(self, embedder):
        """Self-similarity must be ~1.0."""
        sim = embedder.similarity("Test", "Test")
        assert abs(sim - 1.0) < 0.01
    
    def test_batch_embedding(self, embedder):
        """Batch embedding must work."""
        texts = ["First", "Second", "Third"]
        embeddings = embedder.embed_texts(texts)
        assert embeddings.shape == (3, EMBEDDING_DIM)
    
    def test_semantic_search(self, embedder):
        """Semantic search must return ranked results."""
        corpus = ["cat is a pet", "dog is a pet", "car is a vehicle", "plane is aircraft"]
        results = embedder.semantic_search("animals as pets", corpus, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, tuple) for r in results)
        assert all(0 <= r[0] < len(corpus) for r in results)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 5: GRAPH OF THOUGHTS
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TestGraphOfThoughts:
    """Test Graph of Thoughts implementation."""
    
    def test_add_thought(self, graph_of_thoughts):
        """Adding thought must work."""
        thought = graph_of_thoughts.add_thought(
            content="Test hypothesis",
            thought_type=ThoughtType.HYPOTHESIS
        )
        assert thought.id in graph_of_thoughts.nodes
        assert thought.content == "Test hypothesis"
    
    def test_thought_snr_calculation(self, graph_of_thoughts):
        """Thoughts must have SNR calculated."""
        thought = graph_of_thoughts.add_thought(
            content="This is a meaningful thought with good content",
            thought_type=ThoughtType.OBSERVATION
        )
        assert 0.0 <= thought.snr_score <= 1.0
    
    def test_thought_embedding(self, graph_of_thoughts):
        """Thoughts must have embeddings."""
        thought = graph_of_thoughts.add_thought("Test embedding")
        assert thought.embedding is not None
        assert thought.embedding.shape == (EMBEDDING_DIM,)
    
    def test_parent_child_relationships(self, graph_of_thoughts):
        """Parent-child relationships must be tracked."""
        parent = graph_of_thoughts.add_thought("Parent thought")
        child = graph_of_thoughts.add_thought(
            "Child thought",
            parent_ids=[parent.id]
        )
        
        assert child.id in parent.child_ids
        assert parent.id in child.parent_ids
    
    def test_prune_low_snr(self, graph_of_thoughts):
        """Low-SNR thoughts must be prunable."""
        # Add high-SNR thought
        high_snr = graph_of_thoughts.add_thought("Good quality thought")
        
        # Manually set low SNR on a thought
        low_snr = graph_of_thoughts.add_thought("Low quality")
        graph_of_thoughts.nodes[low_snr.id].snr_score = 0.1
        
        pruned = graph_of_thoughts.prune_low_snr(threshold=0.30)
        
        assert pruned >= 1
        assert high_snr.id in graph_of_thoughts.nodes
    
    def test_find_synergies(self, graph_of_thoughts):
        """Synergies must be detectable."""
        # Add related thoughts
        graph_of_thoughts.add_thought("Machine learning algorithms")
        graph_of_thoughts.add_thought("Neural network training")
        graph_of_thoughts.add_thought("Deep learning models")
        
        synergies = graph_of_thoughts.find_synergies()
        assert isinstance(synergies, list)
    
    def test_get_statistics(self, graph_of_thoughts):
        """Statistics must be retrievable."""
        graph_of_thoughts.add_thought("Test")
        stats = graph_of_thoughts.get_statistics()
        
        assert "total_nodes" in stats
        assert "avg_snr" in stats
        assert stats["total_nodes"] >= 1


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 6: SNR OPTIMIZER
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TestSNROptimizer:
    """Test SNR optimization."""
    
    def test_snr_range(self, snr_optimizer):
        """SNR must be in [0, 1]."""
        snr = snr_optimizer.calculate_snr("Test text")
        assert 0.0 <= snr <= 1.0
    
    def test_snr_deterministic(self, snr_optimizer):
        """Same text must produce same SNR."""
        text = "Deterministic test text"
        snr1 = snr_optimizer.calculate_snr(text)
        snr2 = snr_optimizer.calculate_snr(text)
        assert snr1 == snr2
    
    def test_high_quality_text_higher_snr(self, snr_optimizer):
        """High quality text must have higher SNR."""
        low_quality = "uh uh uh maybe perhaps possibly"
        high_quality = "The algorithm processes data through multiple stages to produce accurate results."
        
        snr_low = snr_optimizer.calculate_snr(low_quality)
        snr_high = snr_optimizer.calculate_snr(high_quality)
        
        assert snr_high > snr_low
    
    def test_snr_with_context(self, snr_optimizer):
        """SNR with context must work."""
        snr = snr_optimizer.calculate_snr(
            "This builds on the previous point",
            context={"previous_text": "We discussed the algorithm"}
        )
        assert 0.0 <= snr <= 1.0


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 7: FATE GATE
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TestFATEGate:
    """Test FATE Gate verification."""
    
    def test_fate_result_structure(self, fate_gate):
        """FATE result must have correct structure."""
        result = fate_gate.verify("Test content")
        
        assert isinstance(result, FATEGateResult)
        assert hasattr(result, 'passed')
        assert hasattr(result, 'overall_score')
        assert hasattr(result, 'factual_score')
        assert hasattr(result, 'aligned_score')
        assert hasattr(result, 'testable_score')
        assert hasattr(result, 'evidence_score')
    
    def test_scores_in_range(self, fate_gate):
        """All scores must be in [0, 1]."""
        result = fate_gate.verify("Test verification")
        
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.factual_score <= 1.0
        assert 0.0 <= result.aligned_score <= 1.0
        assert 0.0 <= result.testable_score <= 1.0
        assert 0.0 <= result.evidence_score <= 1.0
    
    def test_is_ihsan_property(self, fate_gate):
        """is_ihsan property must work."""
        result = fate_gate.verify("High quality factual content with evidence")
        assert isinstance(result.is_ihsan, bool)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 8: IHSAN CALCULATOR
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TestIhsanCalculator:
    """Test Ihsān score calculation."""
    
    def test_ihsan_result_structure(self, ihsan_calculator):
        """Ihsān result must have correct structure."""
        result = ihsan_calculator.calculate("Test query", "Test response")
        
        assert "final_score" in result
        assert "component_scores" in result
        assert "above_threshold" in result
        assert "is_ihsan" in result
    
    def test_ihsan_score_range(self, ihsan_calculator):
        """Ihsān score must be in [0, 1]."""
        result = ihsan_calculator.calculate("Query", "Response")
        assert 0.0 <= result["final_score"] <= 1.0
    
    def test_component_scores(self, ihsan_calculator):
        """All component scores must be present."""
        result = ihsan_calculator.calculate("Query", "Response")
        components = result["component_scores"]
        
        assert "clarity" in components
        assert "accuracy" in components
        assert "empathy" in components
        assert "comprehensiveness" in components
        assert "conciseness" in components


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 9: ECONOMIC SYSTEM
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TestLocalEconomicSystem:
    """Test RIBA_ZERO economic system."""
    
    def test_initial_balance(self, economy):
        """Initial BLOOM balance must be 1000."""
        assert economy.bloom_balance == 1000.0
    
    def test_seed_token(self, economy):
        """SEED token must be present."""
        assert economy.seed_token is True
    
    def test_harberger_tax(self, economy):
        """Harberger tax must work."""
        initial = economy.bloom_balance
        result = economy.assess_harberger_tax()
        
        assert result["success"] is True
        assert result["tax_paid"] == economy.self_assessed_value * 0.01
        assert economy.bloom_balance < initial
    
    def test_impact_reward(self, economy):
        """Impact rewards must work."""
        initial = economy.bloom_balance
        result = economy.award_impact_reward(ihsan_score=0.9, constitution_aligned=True)
        
        assert result["rewarded"] > 0
        assert economy.bloom_balance > initial
    
    def test_get_health(self, economy):
        """Economic health must be retrievable."""
        health = economy.get_health()
        
        assert "bloom_balance" in health
        assert "seed_token" in health
        assert health["riba_zero_compliant"] is True


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 10: MERKLE-DAG
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TestLocalMerkleDAG:
    """Test Merkle-DAG storage."""
    
    def test_genesis_block(self, merkle_dag):
        """Genesis block must exist."""
        assert merkle_dag.block_counter >= 1
        assert 0 in merkle_dag.blocks
    
    def test_record_cognitive_cycle(self, merkle_dag):
        """Recording cognitive cycle must work."""
        initial_count = merkle_dag.block_counter
        
        block_hash = merkle_dag.record_cognitive_cycle(
            query="Test query",
            response="Test response",
            ihsan_score=0.85,
            constitution_check=True
        )
        
        assert merkle_dag.block_counter == initial_count + 1
        assert len(block_hash) == 64  # SHA-256
    
    def test_verify_integrity(self, merkle_dag):
        """Integrity verification must work."""
        merkle_dag.record_cognitive_cycle("Q1", "R1", 0.9, True)
        merkle_dag.record_cognitive_cycle("Q2", "R2", 0.85, True)
        
        ok, issues = merkle_dag.verify_integrity()
        assert ok is True
        assert len(issues) == 0
    
    def test_hash_chain(self, merkle_dag):
        """Hash chain must be linked."""
        merkle_dag.record_cognitive_cycle("Q1", "R1", 0.9, True)
        
        # Check previous hash linking
        latest = merkle_dag.blocks[merkle_dag.block_counter - 1]
        prev = merkle_dag.blocks[merkle_dag.block_counter - 2]
        
        assert latest["previous_hash"] == prev["block_hash"]


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 11: HOOK REGISTRY
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TestHookRegistry:
    """Test event-driven hooks."""
    
    def test_register_and_fire(self):
        """Hooks must be registerable and firable."""
        registry = HookRegistry()
        results = []
        
        def handler(ctx):
            results.append(ctx.get("value"))
            return ctx.get("value") * 2
        
        registry.register(HookEvent.QUERY_START, handler)
        returns = registry.fire(HookEvent.QUERY_START, {"value": 5})
        
        assert 5 in results
        assert 10 in returns
    
    def test_unregister(self):
        """Hooks must be unregisterable."""
        registry = HookRegistry()
        
        def handler(ctx):
            return True
        
        registry.register(HookEvent.QUERY_END, handler)
        success = registry.unregister(HookEvent.QUERY_END, handler)
        
        assert success is True
        assert handler not in registry._hooks[HookEvent.QUERY_END]


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 12: COMPACTION ENGINE
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TestCompactionEngine:
    """Test context window management."""
    
    def test_needs_compaction(self):
        """Compaction check must work."""
        engine = CompactionEngine(max_tokens=1000, reserve_tokens=100)
        
        assert engine.needs_compaction(850) is False
        assert engine.needs_compaction(950) is True
    
    def test_compact_thoughts(self, embedder):
        """Thought compaction must work."""
        engine = CompactionEngine()
        
        thoughts = [
            ThoughtNode(id=f"t{i}", content=f"Thought {i}", snr_score=0.8)
            for i in range(10)
        ]
        
        preserved, summary = engine.compact_thoughts(thoughts, preserve_count=3)
        
        assert len(preserved) == 3
        assert summary is not None
        assert summary.thought_type == ThoughtType.SYNTHESIS


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 13: LOCAL REASONING ENGINE
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TestLocalReasoningEngine:
    """Test local reasoning without LLM."""
    
    def test_classify_query(self, embedder):
        """Query classification must work."""
        engine = LocalReasoningEngine(embedder)
        
        assert engine.classify_query("Explain to a child") == "explain_child"
        assert engine.classify_query("Compare X and Y") == "compare"
        assert engine.classify_query("How to do this") == "how_to"
        assert engine.classify_query("What is AI") == "what_is"
        assert engine.classify_query("Why is this happening") == "why"
    
    @pytest.mark.asyncio
    async def test_reason(self, embedder):
        """Reasoning must produce output."""
        engine = LocalReasoningEngine(embedder)
        result = await engine.reason("What is machine learning?", {})
        
        assert isinstance(result, str)
        assert len(result) > 0


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 14: DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TestDataStructures:
    """Test data structure correctness."""
    
    def test_evidence_pointer_immutable(self):
        """EvidencePointer must be immutable."""
        ptr = EvidencePointer(
            pointer_type="content_hash",
            value="abc123"
        )
        with pytest.raises(Exception):
            ptr.value = "new_value"
    
    def test_evidence_pointer_from_hash(self):
        """EvidencePointer.from_hash must work."""
        ptr = EvidencePointer.from_hash(b"test content")
        assert ptr.pointer_type == "content_hash"
        assert len(ptr.value) == 64  # SHA3-256
    
    def test_receipt_hash(self):
        """Receipt hash must be deterministic."""
        receipt = Receipt(
            action_type="test",
            payload={"key": "value"}
        )
        
        hash1 = receipt.compute_hash()
        hash2 = receipt.compute_hash()
        
        assert hash1 == hash2
        assert len(hash1) == 64
    
    def test_thought_node_should_prune(self):
        """ThoughtNode.should_prune must work."""
        high_snr = ThoughtNode(snr_score=0.8, confidence=0.9)
        low_snr = ThoughtNode(snr_score=0.2, confidence=0.2)
        
        assert high_snr.should_prune() is False
        assert low_snr.should_prune() is True
    
    def test_thought_node_is_grounded(self):
        """ThoughtNode.is_grounded must work."""
        ungrounded = ThoughtNode()
        grounded = ThoughtNode(
            evidence=[EvidencePointer("file_path", "test.py")],
            grounding_score=0.8
        )
        
        assert ungrounded.is_grounded() is False
        assert grounded.is_grounded() is True


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 15: ULTIMATE ENGINE INTEGRATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TestUltimateEngineIntegration:
    """Test Ultimate Engine end-to-end."""
    
    def test_engine_initialization(self, engine):
        """Engine must initialize correctly."""
        assert engine.node_id.startswith("ULTIMATE_")
        assert engine.human_name == "Test Human"
        assert engine.daughter_name == "Test Daughter"
    
    def test_engine_components(self, engine):
        """All components must be present."""
        assert isinstance(engine.constitution, Constitution)
        assert isinstance(engine.daughter_test, DaughterTest)
        assert isinstance(engine.embedder, WinterProofEmbedder)
        assert isinstance(engine.got, GraphOfThoughts)
        assert isinstance(engine.snr_optimizer, SNROptimizer)
        assert isinstance(engine.fate_gate, FATEGate)
        assert isinstance(engine.ihsan_calculator, IhsanCalculator)
        assert isinstance(engine.economy, LocalEconomicSystem)
        assert isinstance(engine.merkle_dag, LocalMerkleDAG)
    
    @pytest.mark.asyncio
    async def test_process_query(self, engine):
        """Query processing must work."""
        result = await engine.process_query("What is mathematics?")
        
        assert isinstance(result, KEPResult)
        assert result.query == "What is mathematics?"
        assert len(result.synthesis) > 0
        assert result.constitution_check is True
        assert result.daughter_test_check is True
    
    @pytest.mark.asyncio
    async def test_multiple_queries(self, engine):
        """Multiple queries must accumulate state."""
        await engine.process_query("First question")
        await engine.process_query("Second question")
        
        assert engine._query_count == 2
        assert len(engine.got.nodes) >= 6  # 3 nodes per query
    
    def test_verify_third_fact(self, engine):
        """Third Fact verification must work."""
        result = engine.verify_third_fact("Test claim")
        
        assert isinstance(result, ThirdFactResult)
        assert result.claim == "Test claim"
        assert result.status in ["valid", "invalid", "undecidable"]
    
    def test_daily_maintenance(self, engine):
        """Daily maintenance must work."""
        results = engine.run_daily_maintenance()
        
        assert "daughter_reaffirmation" in results
        assert "merkle_integrity" in results
        assert "thoughts_pruned" in results
        assert results["daughter_reaffirmation"] is True
    
    def test_get_status(self, engine):
        """Status retrieval must work."""
        status = engine.get_status()
        
        assert "engine" in status
        assert status["engine"] == "UltimateEngine"
        assert "version" in status
        assert "kernel_invariants" in status
        assert status["kernel_invariants"]["RIBA_ZERO"] is True


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 16: VERSION & METADATA
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class TestVersionMetadata:
    """Test version and metadata."""
    
    def test_version_format(self):
        """Version must be semantic."""
        parts = __version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)
    
    def test_version_value(self):
        """Version must be 2.0.0."""
        assert __version__ == "2.0.0"


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
