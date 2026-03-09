"""MOE Bridge Tests — Expert-to-Model Dispatch Integration.

Validates that the MOE Bridge:
1. Implements the InferenceProvider protocol
2. Routes queries to correct experts via MOE Engine
3. Dispatches each expert to its Ollama model
4. Synthesizes multi-expert responses
5. Tracks ihsan_tensor for ReflexCompiler learning
6. Handles expert failures gracefully
7. Hot-swaps models at runtime
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.sovereign.moe_bridge import (
    MOEBridge,
    MOEBridgeStats,
    ExpertCallResult,
    _EXPERT_MODEL_MAP,
    _EXPERT_SYSTEM_PROMPTS,
)


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def bridge() -> MOEBridge:
    """Bridge with default config (no actual Ollama calls)."""
    return MOEBridge(ollama_url="http://localhost:11434")


@pytest.fixture
def mock_ollama_response() -> dict:
    """Standard Ollama /api/generate response."""
    return {"response": "This is a test response from the model."}


# ═══════════════════════════════════════════════════════════════════
# 1. PROTOCOL COMPLIANCE
# ═══════════════════════════════════════════════════════════════════


class TestProtocolCompliance:
    """MOEBridge must implement InferenceProvider protocol."""

    def test_has_infer_method(self, bridge: MOEBridge) -> None:
        assert hasattr(bridge, "infer")
        assert callable(bridge.infer)

    @pytest.mark.asyncio
    async def test_infer_returns_string(self, bridge: MOEBridge) -> None:
        with patch.object(bridge, "_ollama_generate", new_callable=AsyncMock) as mock:
            mock.return_value = "Expert response"
            result = await bridge.infer("test query")
            assert isinstance(result, str)
            assert len(result) > 0

    def test_factory_create(self) -> None:
        bridge = MOEBridge.create(ollama_url="http://test:11434")
        assert bridge._ollama_url == "http://test:11434"


# ═══════════════════════════════════════════════════════════════════
# 2. ROUTING INTEGRATION
# ═══════════════════════════════════════════════════════════════════


class TestRoutingIntegration:
    """MOE Engine routes correctly through the bridge."""

    @pytest.mark.asyncio
    async def test_reasoning_query_routes_to_pat_r(self, bridge: MOEBridge) -> None:
        calls: list[str] = []

        async def mock_generate(model, prompt, system=""):
            calls.append(model)
            return f"Response from {model}"

        with patch.object(bridge, "_ollama_generate", side_effect=mock_generate):
            await bridge.infer("How do I analyze and explain the reason for this?")

        # pat_r (deepseek-r1:14b) should be in the calls
        assert any("deepseek" in c for c in calls) or any("r1" in c for c in calls)

    @pytest.mark.asyncio
    async def test_knowledge_query_routes_to_pat_k(self, bridge: MOEBridge) -> None:
        calls: list[str] = []

        async def mock_generate(model, prompt, system=""):
            calls.append(model)
            return f"Response from {model}"

        with patch.object(bridge, "_ollama_generate", side_effect=mock_generate):
            await bridge.infer("What is the history of who invented this?")

        assert any("qwen2.5:3b" in c for c in calls)

    @pytest.mark.asyncio
    async def test_code_query_routes_to_pat_s(self, bridge: MOEBridge) -> None:
        calls: list[str] = []

        async def mock_generate(model, prompt, system=""):
            calls.append(model)
            return f"Response from {model}"

        with patch.object(bridge, "_ollama_generate", side_effect=mock_generate):
            await bridge.infer("Write code to implement and deploy this function")

        assert any("coder" in c for c in calls)

    @pytest.mark.asyncio
    async def test_expert_override(self, bridge: MOEBridge) -> None:
        calls: list[str] = []

        async def mock_generate(model, prompt, system=""):
            calls.append(model)
            return f"Response from {model}"

        with patch.object(bridge, "_ollama_generate", side_effect=mock_generate):
            await bridge.infer("test", expert_override="sat_g")

        # Should call sat_g's model (phi3:mini)
        assert len(calls) == 1
        assert "phi3" in calls[0]


# ═══════════════════════════════════════════════════════════════════
# 3. SYNTHESIS
# ═══════════════════════════════════════════════════════════════════


class TestSynthesis:
    """Multi-expert output synthesis."""

    @pytest.mark.asyncio
    async def test_single_expert_no_prefix(self, bridge: MOEBridge) -> None:
        """Single expert result should not have [expert_id] prefix."""
        bridge._top_k = 1
        bridge._engine = None  # Reset lazy engine

        async def mock_generate(model, prompt, system=""):
            return "Clean response"

        with patch.object(bridge, "_ollama_generate", side_effect=mock_generate):
            result = await bridge.infer("test", expert_override="pat_r")

        assert "[pat_r]" not in result
        assert "Clean response" in result

    @pytest.mark.asyncio
    async def test_multi_expert_has_prefixes(self, bridge: MOEBridge) -> None:
        """Multi-expert results should have [expert_id] prefixes."""
        async def mock_generate(model, prompt, system=""):
            return f"Response from model {model}"

        with patch.object(bridge, "_ollama_generate", side_effect=mock_generate):
            result = await bridge.infer("test", expert_override=["pat_r", "pat_k"])

        assert "[pat_r]" in result
        assert "[pat_k]" in result


# ═══════════════════════════════════════════════════════════════════
# 4. IHSAN TENSOR
# ═══════════════════════════════════════════════════════════════════


class TestIhsanTensor:
    """ihsan_tensor tracking for ReflexCompiler learning."""

    @pytest.mark.asyncio
    async def test_tensor_populated_after_inference(self, bridge: MOEBridge) -> None:
        with patch.object(bridge, "_ollama_generate", new_callable=AsyncMock) as mock:
            mock.return_value = "response"
            await bridge.infer("How do I analyze this?")

        tensor = bridge.last_ihsan_tensor
        assert isinstance(tensor, dict)
        assert len(tensor) > 0
        assert all(0.0 <= v <= 1.0 for v in tensor.values())

    @pytest.mark.asyncio
    async def test_tensor_weights_match_assignments(self, bridge: MOEBridge) -> None:
        with patch.object(bridge, "_ollama_generate", new_callable=AsyncMock) as mock:
            mock.return_value = "response"
            await bridge.infer("test", expert_override=["pat_r", "pat_k"])

        tensor = bridge.last_ihsan_tensor
        # Two experts with equal override weights
        assert len(tensor) == 2
        assert abs(sum(tensor.values()) - 1.0) < 1e-6

    def test_tensor_empty_before_first_call(self, bridge: MOEBridge) -> None:
        assert bridge.last_ihsan_tensor == {}


# ═══════════════════════════════════════════════════════════════════
# 5. FAILURE HANDLING
# ═══════════════════════════════════════════════════════════════════


class TestFailureHandling:
    """Graceful degradation on expert failures."""

    @pytest.mark.asyncio
    async def test_single_expert_failure_returns_error(self, bridge: MOEBridge) -> None:
        async def failing_generate(model, prompt, system=""):
            raise RuntimeError("Model not loaded")

        with patch.object(bridge, "_ollama_generate", side_effect=failing_generate):
            result = await bridge.infer("test", expert_override="pat_r")

        assert "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_partial_failure_uses_successful(self, bridge: MOEBridge) -> None:
        call_count = 0

        async def partial_generate(model, prompt, system=""):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("First expert crashed")
            return "Second expert succeeded"

        with patch.object(bridge, "_ollama_generate", side_effect=partial_generate):
            result = await bridge.infer("test", expert_override=["pat_r", "pat_k"])

        assert "succeeded" in result
        assert bridge.stats.expert_failures == 1

    @pytest.mark.asyncio
    async def test_failure_stats_tracked(self, bridge: MOEBridge) -> None:
        async def failing(model, prompt, system=""):
            raise ConnectionError("Ollama down")

        with patch.object(bridge, "_ollama_generate", side_effect=failing):
            await bridge.infer("test", expert_override="pat_r")

        assert bridge.stats.expert_failures >= 1
        assert bridge.stats.expert_calls >= 1


# ═══════════════════════════════════════════════════════════════════
# 6. MODEL MANAGEMENT
# ═══════════════════════════════════════════════════════════════════


class TestModelManagement:
    """Expert-to-model mapping and hot-swap."""

    def test_default_model_map(self, bridge: MOEBridge) -> None:
        assert bridge.get_expert_model("pat_r") == _EXPERT_MODEL_MAP["pat_r"]
        assert bridge.get_expert_model("pat_k") == _EXPERT_MODEL_MAP["pat_k"]
        assert bridge.get_expert_model("pat_s") == _EXPERT_MODEL_MAP["pat_s"]
        assert bridge.get_expert_model("sat_g") == _EXPERT_MODEL_MAP["sat_g"]
        assert bridge.get_expert_model("sat_v") == _EXPERT_MODEL_MAP["sat_v"]

    def test_hot_swap_model(self, bridge: MOEBridge) -> None:
        bridge.set_expert_model("pat_r", "llama3:8b")
        assert bridge.get_expert_model("pat_r") == "llama3:8b"

    def test_unknown_expert_gets_phi3(self, bridge: MOEBridge) -> None:
        assert bridge.get_expert_model("nonexistent") == "phi3:mini"

    def test_system_prompts_exist_for_all_experts(self) -> None:
        for expert_id in ["pat_r", "pat_k", "pat_s", "sat_g", "sat_v"]:
            assert expert_id in _EXPERT_SYSTEM_PROMPTS
            assert len(_EXPERT_SYSTEM_PROMPTS[expert_id]) > 10

    @pytest.mark.asyncio
    async def test_model_usage_tracked(self, bridge: MOEBridge) -> None:
        with patch.object(bridge, "_ollama_generate", new_callable=AsyncMock) as mock:
            mock.return_value = "response"
            await bridge.infer("test", expert_override="pat_r")

        assert len(bridge.stats.model_usage) > 0


# ═══════════════════════════════════════════════════════════════════
# 7. STATS
# ═══════════════════════════════════════════════════════════════════


class TestStats:
    """Telemetry and observability."""

    @pytest.mark.asyncio
    async def test_inference_count(self, bridge: MOEBridge) -> None:
        with patch.object(bridge, "_ollama_generate", new_callable=AsyncMock) as mock:
            mock.return_value = "r"
            await bridge.infer("test1", expert_override="pat_r")
            await bridge.infer("test2", expert_override="pat_k")

        assert bridge.stats.total_inferences == 2

    @pytest.mark.asyncio
    async def test_avg_latency_tracked(self, bridge: MOEBridge) -> None:
        with patch.object(bridge, "_ollama_generate", new_callable=AsyncMock) as mock:
            mock.return_value = "r"
            await bridge.infer("test", expert_override="pat_r")

        assert bridge.stats.avg_latency_ms > 0

    def test_initial_stats_zero(self, bridge: MOEBridge) -> None:
        s = bridge.stats
        assert s.total_inferences == 0
        assert s.expert_calls == 0
        assert s.expert_failures == 0


# ═══════════════════════════════════════════════════════════════════
# 8. NERVOUS SYSTEM INTEGRATION
# ═══════════════════════════════════════════════════════════════════


class TestNervousSystemIntegration:
    """MOEBridge as drop-in InferenceProvider for SovereignNervousSystem."""

    @pytest.mark.asyncio
    async def test_drop_in_replacement(self) -> None:
        """MOEBridge works as InferenceProvider in NervousSystem."""
        from core.sovereign.mission_nervous_system import SovereignNervousSystem

        bridge = MOEBridge.create()

        with patch.object(bridge, "_ollama_generate", new_callable=AsyncMock) as mock:
            mock.return_value = "Expert analysis of the topic with comprehensive details and thorough reasoning based on evidence"

            ns = SovereignNervousSystem(inference=bridge)
            receipt = await ns.run("Explain the concept of autopoiesis")

        assert receipt.system == "S2"  # No reflex cache, so S2
        assert receipt.output_text  # Non-empty
        assert receipt.ihsan_score > 0.0

    @pytest.mark.asyncio
    async def test_ihsan_tensor_available_after_ns_run(self) -> None:
        """After NervousSystem run, bridge has ihsan_tensor for learning."""
        from core.sovereign.mission_nervous_system import SovereignNervousSystem

        bridge = MOEBridge.create()

        with patch.object(bridge, "_ollama_generate", new_callable=AsyncMock) as mock:
            mock.return_value = "Response"
            ns = SovereignNervousSystem(inference=bridge)
            await ns.run("test")

        # Bridge should have tensor from the inference call
        tensor = bridge.last_ihsan_tensor
        assert isinstance(tensor, dict)
