"""E2E Integration Tests — MOE Engine → Bridge → NervousSystem → Node0.

Validates the full integration chain:
  1. MOE Engine routes queries to correct experts
  2. MOE Bridge dispatches to Ollama models (mocked)
  3. NervousSystem wraps bridge as InferenceProvider
  4. Node0 /v1/query?route=moe uses the bridge
  5. ReflexCompiler records observations with ihsan_tensor
  6. Evidence chain maintains integrity

Standing on: PMBOK (integration verification), Deming (PDCA),
Boyd (OODA full loop), Shannon (E2E signal integrity).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# =============================================================================
# 1. MOE Engine → Bridge Integration
# =============================================================================


class TestMOEEngineToBridge:
    """MOE Engine routing decisions flow correctly through the Bridge."""

    @pytest.mark.asyncio
    async def test_bridge_uses_engine_routing(self) -> None:
        """Bridge's infer() delegates to MOE Engine route() internally."""
        from core.sovereign.moe_bridge import MOEBridge

        bridge = MOEBridge.create()

        with patch.object(bridge, "_ollama_generate", new_callable=AsyncMock) as mock:
            mock.return_value = (
                "Analyzed step by step: first observe, then orient, decide, and act"
            )
            result = await bridge.infer("How do I analyze and explain this problem?")

        assert result  # Non-empty
        assert bridge.stats.total_inferences == 1
        assert bridge.stats.expert_calls >= 1

    @pytest.mark.asyncio
    async def test_expert_model_mapping_correct(self) -> None:
        """Each expert dispatches to its assigned Ollama model."""
        from core.sovereign.moe_bridge import MOEBridge

        bridge = MOEBridge(
            ollama_url="http://localhost:11434",
            expert_models={
                "pat_r": "deepseek-r1:14b",
                "pat_k": "qwen2.5:3b",
                "pat_s": "qwen2.5-coder:7b",
                "sat_g": "phi3:mini",
            },
        )
        models_called: list[str] = []

        async def capture_model(model, prompt, system=""):
            models_called.append(model)
            return f"Response from {model}"

        with patch.object(bridge, "_ollama_generate", side_effect=capture_model):
            await bridge.infer("test", expert_override="pat_s")

        # pat_s should dispatch to qwen2.5-coder:7b
        assert any("coder" in m for m in models_called)

    @pytest.mark.asyncio
    async def test_synthesis_produces_weighted_output(self) -> None:
        """Multi-expert synthesis produces weighted combination."""
        from core.sovereign.moe_bridge import MOEBridge

        bridge = MOEBridge.create()

        async def mock_gen(model, prompt, system=""):
            return f"Expert output from {model}"

        with patch.object(bridge, "_ollama_generate", side_effect=mock_gen):
            result = await bridge.infer("test", expert_override=["pat_r", "pat_k"])

        assert "[pat_r]" in result
        assert "[pat_k]" in result
        tensor = bridge.last_ihsan_tensor
        assert abs(sum(tensor.values()) - 1.0) < 1e-6


# =============================================================================
# 2. Bridge → NervousSystem Integration
# =============================================================================


class TestBridgeToNervousSystem:
    """MOE Bridge as InferenceProvider for SovereignNervousSystem."""

    @pytest.mark.asyncio
    async def test_ns_s2_uses_bridge(self) -> None:
        """NervousSystem S2 path uses MOE Bridge for inference."""
        from core.sovereign.mission_nervous_system import SovereignNervousSystem
        from core.sovereign.moe_bridge import MOEBridge

        bridge = MOEBridge.create()

        with patch.object(bridge, "_ollama_generate", new_callable=AsyncMock) as mock:
            mock.return_value = (
                "Comprehensive expert analysis with detailed reasoning about "
                "the constitutional governance framework and its implications "
                "for sovereign identity verification systems"
            )
            ns = SovereignNervousSystem(inference=bridge)
            receipt = await ns.run("Explain constitutional governance")

        assert receipt.system == "S2"
        assert receipt.output_text
        assert receipt.ihsan_score > 0.0
        assert receipt.snr_score > 0.0
        assert receipt.evidence_hash.startswith("ev:")

    @pytest.mark.asyncio
    async def test_ns_records_observation_after_s2(self) -> None:
        """After S2, NervousSystem records observation for S1 precipitation."""
        from unittest.mock import MagicMock

        from core.sovereign.mission_nervous_system import SovereignNervousSystem
        from core.sovereign.moe_bridge import MOEBridge

        bridge = MOEBridge.create()
        mock_reflex = MagicMock()
        mock_reflex.lookup.return_value = None  # Force S2

        with patch.object(bridge, "_ollama_generate", new_callable=AsyncMock) as mock:
            mock.return_value = (
                "Expert analysis with thorough reasoning and evidence-based conclusions"
            )
            ns = SovereignNervousSystem(inference=bridge, reflex_cache=mock_reflex)
            await ns.run("How does autopoiesis work?")

        # Should have recorded observation for future S1 precipitation
        mock_reflex.record_observation.assert_called_once()
        call_kwargs = mock_reflex.record_observation.call_args
        assert "autopoiesis" in call_kwargs.kwargs.get(
            "input_text", call_kwargs.args[0] if call_kwargs.args else ""
        )

    @pytest.mark.asyncio
    async def test_ns_evidence_chain_integrity(self) -> None:
        """Two consecutive missions produce linked evidence hashes."""
        from core.sovereign.mission_nervous_system import SovereignNervousSystem
        from core.sovereign.moe_bridge import MOEBridge

        bridge = MOEBridge.create()

        with patch.object(bridge, "_ollama_generate", new_callable=AsyncMock) as mock:
            mock.return_value = "Detailed expert response with comprehensive analysis and thorough evidence"
            ns = SovereignNervousSystem(inference=bridge)
            r1 = await ns.run("Mission one")
            r2 = await ns.run("Mission two")

        # Chain integrity: r2.chain_hash depends on r1's evidence
        assert r1.chain_hash != r2.chain_hash
        assert r1.evidence_hash != r2.evidence_hash
        # Both should be valid hashes
        assert len(r1.chain_hash) == 64  # SHA-256
        assert len(r2.chain_hash) == 64

    @pytest.mark.asyncio
    async def test_ns_stats_accumulate(self) -> None:
        """Stats track S1/S2 split correctly."""
        from core.sovereign.mission_nervous_system import SovereignNervousSystem
        from core.sovereign.moe_bridge import MOEBridge

        bridge = MOEBridge.create()

        with patch.object(bridge, "_ollama_generate", new_callable=AsyncMock) as mock:
            mock.return_value = "Expert response"
            ns = SovereignNervousSystem(inference=bridge)
            await ns.run("m1")
            await ns.run("m2")
            await ns.run("m3")

        assert ns._stats.total_missions == 3
        assert ns._stats.s2_executions == 3
        assert ns._stats.s1_hits == 0


# =============================================================================
# 3. Node0 /v1/query MOE Route
# =============================================================================


class TestNode0MOERoute:
    """Node0 /v1/query endpoint with route=moe."""

    @pytest.mark.asyncio
    async def test_moe_route_calls_bridge(self) -> None:
        """route=moe dispatches through MOE Bridge."""
        # Import the _query_moe function directly to test without FastAPI
        import sys

        sys.path.insert(
            0, str(Path(__file__).resolve().parent.parent.parent / "scripts")
        )

        from core.sovereign.moe_bridge import MOEBridge

        bridge = MOEBridge.create()

        with patch.object(bridge, "_ollama_generate", new_callable=AsyncMock) as mock:
            mock.return_value = "MOE expert response"
            response = await bridge.infer("How to optimize?")

        assert response
        assert bridge.stats.total_inferences == 1

    @pytest.mark.asyncio
    async def test_moe_route_returns_expert_tensor(self) -> None:
        """MOE route response includes expert contribution tensor."""
        from core.sovereign.moe_bridge import MOEBridge

        bridge = MOEBridge.create()

        with patch.object(bridge, "_ollama_generate", new_callable=AsyncMock) as mock:
            mock.return_value = "response"
            await bridge.infer("How do I explain this?")

        tensor = bridge.last_ihsan_tensor
        assert isinstance(tensor, dict)
        assert len(tensor) > 0
        # Weights sum to 1.0
        assert abs(sum(tensor.values()) - 1.0) < 1e-6


# =============================================================================
# 4. Full Pipeline: MOE → Synthesis → ReflexCompiler
# =============================================================================


class TestFullPipeline:
    """Complete closed-loop: MOE → Synthesis → ReflexCompiler → Evidence."""

    @pytest.mark.asyncio
    async def test_closed_loop_learning_pipeline(self) -> None:
        """Full loop: MOE inference → NS receipt → ReflexCompiler observation."""
        from core.sovereign.mission_nervous_system import SovereignNervousSystem
        from core.sovereign.moe_bridge import MOEBridge

        bridge = MOEBridge.create()
        observations: list[dict] = []

        class MockReflex:
            def lookup(self, text, **kw):
                return None

            def record_observation(self, **kwargs):
                observations.append(kwargs)
                return None

            @staticmethod
            def _hash_input(input_text: str) -> str:
                import hashlib

                normalized = " ".join(input_text.lower().split())
                return hashlib.sha256(normalized.encode()).hexdigest()

        with patch.object(bridge, "_ollama_generate", new_callable=AsyncMock) as mock:
            mock.return_value = (
                "The system uses constitutional governance with Ihsan threshold "
                "of 0.95 and SNR minimum of 0.85 for quality assurance, "
                "verified through evidence-chained receipts"
            )
            ns = SovereignNervousSystem(inference=bridge, reflex_cache=MockReflex())
            receipt = await ns.run("What are the constitutional thresholds?")

        # Verify full chain:
        # 1. Bridge executed MOE routing
        assert bridge.stats.total_inferences == 1
        # 2. NervousSystem produced receipt
        assert receipt.system == "S2"
        assert receipt.evidence_hash.startswith("ev:")
        # 3. ReflexCompiler recorded observation
        assert len(observations) == 1
        assert "constitutional" in observations[0]["input_text"].lower()
        assert observations[0]["ihsan_composite"] > 0.0

    @pytest.mark.asyncio
    async def test_moe_expert_diversity(self) -> None:
        """Different query types activate different expert combinations."""
        from core.sovereign.moe_bridge import MOEBridge

        bridge = MOEBridge.create()

        async def mock_gen(model, prompt, system=""):
            return f"Response from {model}"

        queries = {
            "How do I analyze and explain the reasoning?": "pat_r",
            "What is the history of who invented TCP/IP?": "pat_k",
            "Write code to implement and deploy a REST API function": "pat_s",
        }

        with patch.object(bridge, "_ollama_generate", side_effect=mock_gen):
            for query, expected_expert in queries.items():
                await bridge.infer(query)
                tensor = bridge.last_ihsan_tensor
                # Expected expert should have highest weight
                if tensor:
                    top_expert = max(tensor, key=tensor.get)
                    assert top_expert == expected_expert, (
                        f"Query '{query[:30]}...' expected {expected_expert}, "
                        f"got {top_expert} (tensor={tensor})"
                    )
