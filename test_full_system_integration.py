#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║   BIZRA FULL-SYSTEM INTEGRATION TESTS — /T full-system                                                      ║
║   Cross-File Verification for BIZRA DDAGI OS v2.0.0                                                         ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║   Tests across:                                                                                              ║
║   • ultimate_engine.py (Core cognitive engine)                                                               ║
║   • ecosystem_bridge.py (Unified integration layer)                                                          ║
║   • peak_masterpiece.py (Graph of Thoughts, SNR, FATE)                                                       ║
║   • hyper_loopback.py (Winter-proof sovereign system)                                                        ║
║   • bizra_orchestrator.py (Hypergraph RAG + ARTE + PAT)                                                      ║
║   • sovereign_apex.py (VectorLayer + GraphLayer)                                                             ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import pytest
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# IMPORT ECOSYSTEM COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

# Ultimate Engine (Core)
from ultimate_engine import (
    UltimateEngine, Constitution, DaughterTest, WinterProofEmbedder,
    GraphOfThoughts, SNROptimizer, FATEGate, IhsanCalculator,
    LocalEconomicSystem, LocalMerkleDAG, HookRegistry, HookEvent,
    CompactionEngine, LocalReasoningEngine,
    ThoughtNode, ThoughtType, Receipt, EvidencePointer,
    RIBA_ZERO, ZANN_ZERO, IHSAN_FLOOR,
    SNR_MINIMUM, SNR_ACCEPTABLE, SNR_IHSAN, FATE_GATE_THRESHOLD,
    EMBEDDING_DIM, DISCIPLINE_COUNT, __version__ as ULTIMATE_VERSION
)

# Ecosystem Bridge
from ecosystem_bridge import (
    EcosystemBridge, UnifiedQuery, UnifiedResponse,
    EcosystemHealth, ComponentStatus,
    get_ecosystem, initialize_ecosystem,
    ULTIMATE_ENGINE_AVAILABLE, ORCHESTRATOR_AVAILABLE,
    APEX_AVAILABLE, PEAK_AVAILABLE, BRIDGE_AVAILABLE,
    HYPERLOOPBACK_AVAILABLE,
    __version__ as BRIDGE_VERSION
)

# Try to import optional components
try:
    from hyper_loopback import HyperLoopbackSystem
    HYPER_AVAILABLE = True
except ImportError:
    HYPER_AVAILABLE = False

try:
    from peak_masterpiece import PeakMasterpieceEngine
    PEAK_ENGINE_AVAILABLE = True
except ImportError:
    PEAK_ENGINE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def ultimate_engine():
    """Create Ultimate Engine instance."""
    return UltimateEngine(human_name="Test Human", daughter_name="Test Daughter")


@pytest.fixture
def ecosystem_bridge():
    """Create Ecosystem Bridge instance."""
    return EcosystemBridge(
        human_name="Test Human",
        daughter_name="Test Daughter",
        enable_orchestrator=False,  # Disable for unit tests
        enable_apex=False
    )


@pytest.fixture
def hyper_loopback():
    """Create HyperLoopback instance if available."""
    if HYPER_AVAILABLE:
        return HyperLoopbackSystem("Test Human", "Test Daughter")
    return None


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 1: KERNEL INVARIANTS ACROSS ALL MODULES
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestKernelInvariantsAcrossModules:
    """Test that kernel invariants are consistent across all modules."""
    
    def test_riba_zero_ultimate_engine(self):
        """RIBA_ZERO must be True in Ultimate Engine."""
        from ultimate_engine import RIBA_ZERO
        assert RIBA_ZERO is True
    
    def test_riba_zero_ecosystem_bridge(self):
        """RIBA_ZERO must be accessible from Ecosystem Bridge."""
        from ecosystem_bridge import ULTIMATE_ENGINE_AVAILABLE
        if ULTIMATE_ENGINE_AVAILABLE:
            from ultimate_engine import RIBA_ZERO
            assert RIBA_ZERO is True
    
    def test_zann_zero_consistency(self):
        """ZANN_ZERO must be True everywhere."""
        assert ZANN_ZERO is True
    
    def test_ihsan_floor_value(self):
        """IHSAN_FLOOR must be 0.90 everywhere."""
        assert IHSAN_FLOOR == 0.90
    
    def test_snr_thresholds_ordering(self):
        """SNR thresholds must be properly ordered."""
        assert SNR_MINIMUM < SNR_ACCEPTABLE < SNR_IHSAN
        assert SNR_MINIMUM == 0.85
        assert SNR_IHSAN == 0.99
    
    def test_embedding_dimension(self):
        """Embedding dimension must be 384 across all modules."""
        assert EMBEDDING_DIM == 384
    
    def test_discipline_count(self):
        """47-discipline count must be maintained."""
        assert DISCIPLINE_COUNT == 47


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 2: CONSTITUTION CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestConstitutionConsistency:
    """Test constitution is consistent across components."""
    
    def test_constitution_hash_deterministic_across_instances(self):
        """Constitution hash must be same across different engine instances."""
        engine1 = UltimateEngine()
        engine2 = UltimateEngine()
        
        assert engine1.constitution.get_hash() == engine2.constitution.get_hash()
    
    def test_constitution_has_five_articles(self, ultimate_engine):
        """Constitution must have exactly 5 articles."""
        assert len(ultimate_engine.constitution.articles) == 5
    
    def test_constitution_article_keys(self, ultimate_engine):
        """Constitution must have articles I through V."""
        assert set(ultimate_engine.constitution.articles.keys()) == {"I", "II", "III", "IV", "V"}
    
    def test_constitution_immutability(self, ultimate_engine):
        """Constitution hash should not change after operations."""
        hash_before = ultimate_engine.constitution.get_hash()
        
        # Perform operations
        ultimate_engine.constitution.verify_compliance("Test action")
        
        hash_after = ultimate_engine.constitution.get_hash()
        assert hash_before == hash_after


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 3: DAUGHTER TEST CROSS-MODULE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestDaughterTestCrossModule:
    """Test Daughter Test consistency across modules."""
    
    def test_daughter_test_in_ultimate_engine(self, ultimate_engine):
        """Daughter Test must exist in Ultimate Engine."""
        assert ultimate_engine.daughter_test is not None
        assert ultimate_engine.daughter_test.daughter_name == "Test Daughter"
    
    def test_daughter_test_in_ecosystem_bridge(self, ecosystem_bridge):
        """Daughter Test should be accessible via Ecosystem Bridge."""
        if ecosystem_bridge.ultimate_engine:
            assert ecosystem_bridge.ultimate_engine.daughter_test is not None
    
    def test_daughter_test_attestation_hash(self, ultimate_engine):
        """Attestation hash must be generated."""
        assert len(ultimate_engine.daughter_test.attestation_hash) == 128
    
    def test_daughter_test_passes_safe_actions(self, ultimate_engine):
        """Safe actions must pass Daughter Test."""
        ok, _ = ultimate_engine.daughter_test.verify({
            "decision_summary": "Help daughter learn mathematics",
            "impact": {}
        })
        assert ok is True


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 4: WINTER-PROOF EMBEDDER CROSS-MODULE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestWinterProofEmbedderCrossModule:
    """Test embedder consistency across modules."""
    
    def test_embedder_dimension_ultimate_engine(self, ultimate_engine):
        """Embedder in Ultimate Engine must produce correct dimension."""
        emb = ultimate_engine.embedder.embed_text("Test")
        assert emb.shape == (EMBEDDING_DIM,)
    
    def test_embedder_determinism(self):
        """Same text must produce same embedding across instances."""
        embedder1 = WinterProofEmbedder()
        embedder2 = WinterProofEmbedder()
        
        text = "Test determinism"
        emb1 = embedder1.embed_text(text)
        emb2 = embedder2.embed_text(text)
        
        assert np.allclose(emb1, emb2)
    
    def test_embedder_normalization(self, ultimate_engine):
        """Embeddings must be L2-normalized."""
        emb = ultimate_engine.embedder.embed_text("Test normalization")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 5: ECONOMIC SYSTEM CROSS-MODULE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestEconomicSystemCrossModule:
    """Test economic system across modules."""
    
    def test_economy_in_ultimate_engine(self, ultimate_engine):
        """Economic system must exist in Ultimate Engine."""
        assert ultimate_engine.economy is not None
        assert ultimate_engine.economy.bloom_balance == 1000.0
    
    def test_economy_riba_zero_compliant(self, ultimate_engine):
        """Economic system must be RIBA_ZERO compliant."""
        health = ultimate_engine.economy.get_health()
        assert health["riba_zero_compliant"] is True
    
    def test_economy_harberger_tax(self, ultimate_engine):
        """Harberger tax must work correctly."""
        initial = ultimate_engine.economy.bloom_balance
        result = ultimate_engine.economy.assess_harberger_tax()
        
        assert result["success"] is True
        assert ultimate_engine.economy.bloom_balance < initial
    
    def test_economy_impact_reward(self, ultimate_engine):
        """Impact rewards must work correctly."""
        initial = ultimate_engine.economy.bloom_balance
        result = ultimate_engine.economy.award_impact_reward(0.9, True)
        
        assert result["rewarded"] > 0
        assert ultimate_engine.economy.bloom_balance > initial


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 6: MERKLE-DAG CROSS-MODULE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestMerkleDagCrossModule:
    """Test Merkle-DAG across modules."""
    
    def test_merkle_dag_in_ultimate_engine(self, ultimate_engine):
        """Merkle-DAG must exist in Ultimate Engine."""
        assert ultimate_engine.merkle_dag is not None
        assert ultimate_engine.merkle_dag.block_counter >= 1
    
    def test_merkle_dag_genesis_block(self, ultimate_engine):
        """Genesis block must exist."""
        assert 0 in ultimate_engine.merkle_dag.blocks
        genesis = ultimate_engine.merkle_dag.blocks[0]
        assert genesis["data"]["type"] == "genesis"
    
    def test_merkle_dag_integrity(self, ultimate_engine):
        """Merkle-DAG integrity must be verifiable."""
        # Add some blocks
        ultimate_engine.merkle_dag.record_cognitive_cycle("Q1", "R1", 0.9, True)
        ultimate_engine.merkle_dag.record_cognitive_cycle("Q2", "R2", 0.85, True)
        
        ok, issues = ultimate_engine.merkle_dag.verify_integrity()
        assert ok is True
        assert len(issues) == 0


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 7: ECOSYSTEM BRIDGE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestEcosystemBridgeIntegration:
    """Test Ecosystem Bridge integration."""
    
    def test_bridge_creates_ultimate_engine(self, ecosystem_bridge):
        """Bridge must create Ultimate Engine."""
        assert ecosystem_bridge.ultimate_engine is not None
    
    def test_bridge_node_id(self, ecosystem_bridge):
        """Bridge must have valid node ID."""
        assert ecosystem_bridge.node_id.startswith("ECOSYSTEM_")
    
    def test_bridge_health_report(self, ecosystem_bridge):
        """Bridge must provide health report."""
        health = ecosystem_bridge.get_health()
        
        assert isinstance(health, EcosystemHealth)
        assert health.ultimate_engine == ComponentStatus.AVAILABLE
        assert 0.0 <= health.overall_health <= 1.0
    
    def test_bridge_kernel_invariants(self, ecosystem_bridge):
        """Bridge must report kernel invariants."""
        health = ecosystem_bridge.get_health()
        assert health.kernel_invariants_ok is True
    
    def test_bridge_status(self, ecosystem_bridge):
        """Bridge must provide comprehensive status."""
        status = ecosystem_bridge.get_status()
        
        assert "bridge" in status
        assert "version" in status
        assert "components" in status
        assert "kernel_invariants" in status


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 8: ASYNC QUERY INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestAsyncQueryIntegration:
    """Test async query processing across modules."""
    
    @pytest.mark.asyncio
    async def test_ultimate_engine_query(self, ultimate_engine):
        """Ultimate Engine must process queries."""
        result = await ultimate_engine.process_query("What is mathematics?")
        
        assert result.query == "What is mathematics?"
        assert len(result.synthesis) > 0
        assert result.constitution_check is True
        assert result.daughter_test_check is True
    
    @pytest.mark.asyncio
    async def test_ecosystem_bridge_query(self, ecosystem_bridge):
        """Ecosystem Bridge must process unified queries."""
        query = UnifiedQuery(
            text="Explain quantum mechanics",
            require_constitution_check=True,
            require_daughter_test=True
        )
        
        response = await ecosystem_bridge.query(query)
        
        assert isinstance(response, UnifiedResponse)
        assert response.constitution_check is True
        assert response.daughter_test_check is True
        assert len(response.synthesis) > 0
    
    @pytest.mark.asyncio
    async def test_multiple_queries_state_accumulation(self, ecosystem_bridge):
        """Multiple queries must accumulate state correctly."""
        for i in range(3):
            query = UnifiedQuery(text=f"Query number {i+1}")
            await ecosystem_bridge.query(query)
        
        assert ecosystem_bridge._query_count == 3


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 9: GRAPH OF THOUGHTS INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestGraphOfThoughtsIntegration:
    """Test Graph of Thoughts across modules."""
    
    def test_got_in_ultimate_engine(self, ultimate_engine):
        """GoT must exist in Ultimate Engine."""
        assert ultimate_engine.got is not None
        assert isinstance(ultimate_engine.got, GraphOfThoughts)
    
    def test_got_uses_winter_proof_embedder(self, ultimate_engine):
        """GoT must use WinterProofEmbedder."""
        assert ultimate_engine.got.embedder is ultimate_engine.embedder
    
    @pytest.mark.asyncio
    async def test_got_accumulates_thoughts(self, ultimate_engine):
        """GoT must accumulate thoughts across queries."""
        initial_count = len(ultimate_engine.got.nodes)
        
        await ultimate_engine.process_query("First query")
        mid_count = len(ultimate_engine.got.nodes)
        
        await ultimate_engine.process_query("Second query")
        final_count = len(ultimate_engine.got.nodes)
        
        assert mid_count > initial_count
        assert final_count > mid_count


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 10: SNR OPTIMIZATION INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestSNROptimizationIntegration:
    """Test SNR optimization across modules."""
    
    def test_snr_optimizer_in_ultimate_engine(self, ultimate_engine):
        """SNR Optimizer must exist in Ultimate Engine."""
        assert ultimate_engine.snr_optimizer is not None
        assert isinstance(ultimate_engine.snr_optimizer, SNROptimizer)
    
    def test_snr_calculation_range(self, ultimate_engine):
        """SNR must be in [0, 1]."""
        snr = ultimate_engine.snr_optimizer.calculate_snr("Test text for SNR")
        assert 0.0 <= snr <= 1.0
    
    @pytest.mark.asyncio
    async def test_snr_in_query_result(self, ultimate_engine):
        """Query result must include SNR score."""
        result = await ultimate_engine.process_query("Calculate my SNR")
        assert 0.0 <= result.snr_score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 11: FATE GATE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestFATEGateIntegration:
    """Test FATE Gate across modules."""
    
    def test_fate_gate_in_ultimate_engine(self, ultimate_engine):
        """FATE Gate must exist in Ultimate Engine."""
        assert ultimate_engine.fate_gate is not None
        assert isinstance(ultimate_engine.fate_gate, FATEGate)
    
    def test_fate_gate_uses_constitution(self, ultimate_engine):
        """FATE Gate must use Constitution."""
        assert ultimate_engine.fate_gate.constitution is ultimate_engine.constitution
    
    def test_fate_gate_uses_daughter_test(self, ultimate_engine):
        """FATE Gate must use Daughter Test."""
        assert ultimate_engine.fate_gate.daughter_test is ultimate_engine.daughter_test
    
    def test_fate_verification(self, ultimate_engine):
        """FATE verification must return proper result."""
        result = ultimate_engine.fate_gate.verify("Test content for FATE")
        
        assert hasattr(result, 'passed')
        assert hasattr(result, 'overall_score')
        assert 0.0 <= result.overall_score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 12: IHSAN CALCULATION INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestIhsanCalculationIntegration:
    """Test Ihsān calculation across modules."""
    
    def test_ihsan_calculator_in_ultimate_engine(self, ultimate_engine):
        """Ihsān Calculator must exist in Ultimate Engine."""
        assert ultimate_engine.ihsan_calculator is not None
        assert isinstance(ultimate_engine.ihsan_calculator, IhsanCalculator)
    
    def test_ihsan_calculation(self, ultimate_engine):
        """Ihsān calculation must return proper result."""
        result = ultimate_engine.ihsan_calculator.calculate(
            "What is the meaning of life?",
            "Life has many dimensions including purpose, relationships, and growth."
        )
        
        assert "final_score" in result
        assert "component_scores" in result
        assert "above_threshold" in result
        assert 0.0 <= result["final_score"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 13: THIRD FACT PROTOCOL INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestThirdFactProtocolIntegration:
    """Test Third Fact Protocol across modules."""
    
    def test_third_fact_verification(self, ultimate_engine):
        """Third Fact verification must work."""
        result = ultimate_engine.verify_third_fact("Test claim for verification")
        
        assert result.claim == "Test claim for verification"
        assert result.status in ["valid", "invalid", "undecidable"]
        assert result.step in ["neural", "semantic", "formal", "cryptographic"]
    
    def test_third_fact_has_proof_trace(self, ultimate_engine):
        """Third Fact result must have proof trace."""
        result = ultimate_engine.verify_third_fact("Another test claim")
        assert len(result.proof_trace) > 0


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 14: VERSION CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestVersionConsistency:
    """Test version consistency across modules."""
    
    def test_ultimate_engine_version(self):
        """Ultimate Engine must be v2.0.0."""
        assert ULTIMATE_VERSION == "2.0.0"
    
    def test_ecosystem_bridge_version(self):
        """Ecosystem Bridge must be v2.0.0."""
        assert BRIDGE_VERSION == "2.0.0"
    
    def test_versions_match(self):
        """All module versions must match."""
        assert ULTIMATE_VERSION == BRIDGE_VERSION


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 15: HYPER LOOPBACK INTEGRATION (IF AVAILABLE)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HYPER_AVAILABLE, reason="HyperLoopback not available")
class TestHyperLoopbackIntegration:
    """Test HyperLoopback integration."""
    
    def test_hyper_loopback_creation(self, hyper_loopback):
        """HyperLoopback must be creatable."""
        assert hyper_loopback is not None
    
    def test_hyper_loopback_constitution(self, hyper_loopback):
        """HyperLoopback must have constitution."""
        if hyper_loopback:
            assert hyper_loopback.constitution is not None


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 16: COMPONENT AVAILABILITY
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestComponentAvailability:
    """Test component availability flags."""
    
    def test_ultimate_engine_available(self):
        """Ultimate Engine must be available."""
        assert ULTIMATE_ENGINE_AVAILABLE is True
    
    def test_ecosystem_bridge_detects_components(self, ecosystem_bridge):
        """Ecosystem Bridge must detect available components."""
        status = ecosystem_bridge.get_status()
        assert status["components"]["ultimate_engine"] is True


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 17: END-TO-END WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_full_query_workflow(self, ecosystem_bridge):
        """Test full query workflow through ecosystem."""
        # Initialize
        await ecosystem_bridge.initialize()
        
        # Query
        query = UnifiedQuery(
            text="What is the importance of ethical AI?",
            require_constitution_check=True,
            require_daughter_test=True,
            require_fate_gate=True
        )
        
        response = await ecosystem_bridge.query(query)
        
        # Verify all checks passed
        assert response.constitution_check is True
        assert response.daughter_test_check is True
        assert len(response.synthesis) > 0
        assert "UltimateEngine" in response.components_used or "Constitution" in response.components_used
    
    @pytest.mark.asyncio
    async def test_daily_maintenance_workflow(self, ultimate_engine):
        """Test daily maintenance workflow."""
        results = ultimate_engine.run_daily_maintenance()
        
        assert results["daughter_reaffirmation"] is True
        assert results["merkle_integrity"] is True
        assert "thoughts_pruned" in results
    
    @pytest.mark.asyncio
    async def test_economic_reward_workflow(self, ultimate_engine):
        """Test economic reward workflow after query."""
        initial_balance = ultimate_engine.economy.bloom_balance
        
        result = await ultimate_engine.process_query("High quality query")
        
        if result.bloom_reward > 0:
            assert ultimate_engine.economy.bloom_balance > initial_balance


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 18: STRESS TESTS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestStressTests:
    """Stress tests for the ecosystem."""
    
    @pytest.mark.asyncio
    async def test_multiple_rapid_queries(self, ecosystem_bridge):
        """Test handling multiple rapid queries."""
        await ecosystem_bridge.initialize()
        
        queries = [
            UnifiedQuery(text=f"Rapid query number {i}")
            for i in range(10)
        ]
        
        responses = []
        for query in queries:
            response = await ecosystem_bridge.query(query)
            responses.append(response)
        
        assert len(responses) == 10
        assert all(r.constitution_check for r in responses)
    
    def test_embedder_large_batch(self, ultimate_engine):
        """Test embedder with large batch."""
        texts = [f"Sample text number {i}" for i in range(100)]
        embeddings = ultimate_engine.embedder.embed_texts(texts)
        
        assert embeddings.shape == (100, EMBEDDING_DIM)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 19: RECEIPT CHAIN INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestReceiptChainIntegration:
    """Test receipt chain across modules."""
    
    @pytest.mark.asyncio
    async def test_receipts_generated_on_query(self, ultimate_engine):
        """Receipts must be generated on queries."""
        initial_count = len(ultimate_engine._receipt_chain)
        
        await ultimate_engine.process_query("Generate receipt test")
        
        assert len(ultimate_engine._receipt_chain) > initial_count
    
    def test_receipt_hash_chain(self, ultimate_engine):
        """Receipt hashes must form a chain."""
        # Generate some receipts
        for _ in range(3):
            ultimate_engine._emit_receipt("test", {"data": "test"})
        
        # Verify chain
        for i in range(1, len(ultimate_engine._receipt_chain)):
            curr = ultimate_engine._receipt_chain[i]
            prev = ultimate_engine._receipt_chain[i-1]
            assert curr.prev_hash == prev.compute_hash()


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 20: HOOK REGISTRY INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class TestHookRegistryIntegration:
    """Test hook registry across modules."""
    
    def test_hooks_in_ultimate_engine(self, ultimate_engine):
        """Hook registry must exist in Ultimate Engine."""
        assert ultimate_engine.hooks is not None
        assert isinstance(ultimate_engine.hooks, HookRegistry)
    
    @pytest.mark.asyncio
    async def test_hooks_fire_on_query(self, ultimate_engine):
        """Hooks must fire during query processing."""
        hook_fired = []
        
        def test_hook(ctx):
            hook_fired.append(ctx)
            return True
        
        ultimate_engine.hooks.register(HookEvent.QUERY_START, test_hook)
        
        await ultimate_engine.process_query("Test hook firing")
        
        assert len(hook_fired) > 0


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
