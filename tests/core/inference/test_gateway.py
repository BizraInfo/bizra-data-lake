"""
BIZRA Inference Gateway Test Suite

Tests for the tiered inference gateway with fail-closed semantics.
Target: 70% coverage of core/inference/gateway.py (705 lines)
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path (works across platforms)
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from core.inference.gateway import (
    InferenceGateway,
    InferenceConfig,
    InferenceResult,
    InferenceStatus,
    InferenceBackend,
    ComputeTier,
    TaskComplexity,
    LlamaCppBackend,
    OllamaBackend,
    get_inference_gateway,
    TIER_CONFIGS,
)


@pytest.fixture
def config():
    """Create a test configuration."""
    return InferenceConfig(
        model_path="/tmp/test_model.gguf",
        require_local=True,
        fallbacks=["ollama"],
    )


@pytest.fixture
def gateway(config):
    """Create a gateway instance."""
    return InferenceGateway(config)


class TestInferenceConfig:
    """Tests for InferenceConfig dataclass."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = InferenceConfig()

        assert config.context_length == 8192
        assert config.max_tokens == 2048
        assert config.default_tier == ComputeTier.LOCAL
        assert config.require_local is True
        assert "ollama" in config.fallbacks

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = InferenceConfig(
            context_length=4096,
            max_tokens=1024,
            require_local=False,
        )

        assert config.context_length == 4096
        assert config.max_tokens == 1024
        assert config.require_local is False

    def test_lmstudio_url_default(self):
        """LM Studio URL should default to 192.168.56.1:1234."""
        config = InferenceConfig()

        assert config.lmstudio_url == "http://192.168.56.1:1234"


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_result_creation(self):
        """Result should contain all required fields."""
        result = InferenceResult(
            content="Hello, world!",
            model="test-model",
            backend=InferenceBackend.LLAMACPP,
            tier=ComputeTier.LOCAL,
            tokens_generated=10,
            tokens_per_second=50.0,
            latency_ms=200.0,
        )

        assert result.content == "Hello, world!"
        assert result.model == "test-model"
        assert result.backend == InferenceBackend.LLAMACPP
        assert result.tier == ComputeTier.LOCAL

    def test_auto_timestamp(self):
        """Timestamp should be auto-generated if not provided."""
        result = InferenceResult(
            content="test",
            model="test",
            backend=InferenceBackend.LLAMACPP,
            tier=ComputeTier.LOCAL,
        )

        assert result.timestamp != ""
        assert "T" in result.timestamp  # ISO format


class TestTaskComplexity:
    """Tests for task complexity estimation."""

    def test_simple_task_low_score(self):
        """Simple tasks should have low complexity score."""
        complexity = TaskComplexity(
            input_tokens=50,
            estimated_output_tokens=100,
            reasoning_depth=0.1,
            domain_specificity=0.1,
        )

        assert complexity.score < 0.3

    def test_complex_task_high_score(self):
        """Complex tasks should have high complexity score."""
        complexity = TaskComplexity(
            input_tokens=2000,
            estimated_output_tokens=2000,
            reasoning_depth=0.9,
            domain_specificity=0.9,
        )

        assert complexity.score > 0.7

    def test_score_bounded(self):
        """Score should be bounded between 0 and 1."""
        # Extreme values
        complexity = TaskComplexity(
            input_tokens=10000,
            estimated_output_tokens=10000,
            reasoning_depth=1.0,
            domain_specificity=1.0,
        )

        assert 0.0 <= complexity.score <= 1.0


class TestGatewayInitialization:
    """Tests for gateway initialization."""

    @pytest.mark.asyncio
    async def test_cold_status_initially(self, gateway):
        """Gateway should start in COLD status."""
        assert gateway.status == InferenceStatus.COLD

    @pytest.mark.asyncio
    async def test_offline_mode_denies_inference(self, gateway):
        """Offline gateway should deny inference requests."""
        gateway.status = InferenceStatus.OFFLINE

        with pytest.raises(RuntimeError, match="no backend available"):
            await gateway.infer("test prompt")

    @pytest.mark.asyncio
    async def test_no_backend_denies_inference(self, gateway):
        """Gateway without active backend should deny requests."""
        gateway.status = InferenceStatus.READY
        gateway._active_backend = None

        with pytest.raises(RuntimeError, match="no active backend"):
            await gateway.infer("test prompt")


class TestComplexityRouting:
    """Tests for complexity-based tier routing."""

    def test_route_simple_to_edge(self, gateway):
        """Low complexity should route to EDGE tier."""
        complexity = TaskComplexity(
            input_tokens=20,
            estimated_output_tokens=50,
            reasoning_depth=0.05,
            domain_specificity=0.05,
        )

        tier = gateway.route(complexity)

        assert tier == ComputeTier.EDGE

    def test_route_medium_to_local(self, gateway):
        """Medium complexity should route to LOCAL tier."""
        complexity = TaskComplexity(
            input_tokens=500,
            estimated_output_tokens=500,
            reasoning_depth=0.5,
            domain_specificity=0.4,
        )

        tier = gateway.route(complexity)

        assert tier == ComputeTier.LOCAL

    def test_route_complex_to_pool(self, gateway):
        """High complexity should route to POOL tier."""
        complexity = TaskComplexity(
            input_tokens=3000,
            estimated_output_tokens=1000,
            reasoning_depth=0.9,
            domain_specificity=0.8,
        )

        tier = gateway.route(complexity)

        assert tier == ComputeTier.POOL


class TestComplexityEstimation:
    """Tests for prompt complexity estimation."""

    def test_estimate_simple_prompt(self, gateway):
        """Simple prompt should have low complexity."""
        prompt = "What is 2 + 2?"

        complexity = gateway.estimate_complexity(prompt)

        assert complexity.reasoning_depth < 0.3

    def test_estimate_reasoning_prompt(self, gateway):
        """Prompt with reasoning keywords should have higher depth."""
        prompt = "Explain why the algorithm works and how it compares to alternatives."

        complexity = gateway.estimate_complexity(prompt)

        assert complexity.reasoning_depth > 0

    def test_estimate_technical_prompt(self, gateway):
        """Technical prompt should have higher domain specificity."""
        prompt = "Explain the protocol architecture and theorem behind the algorithm."

        complexity = gateway.estimate_complexity(prompt)

        assert complexity.domain_specificity > 0


class TestRequireLocalBlocking:
    """Tests for require_local fail-closed behavior."""

    @pytest.mark.asyncio
    async def test_require_local_blocks_fallback(self):
        """With require_local=True, fallbacks should not be tried when all backends fail."""
        config = InferenceConfig(require_local=True)
        gateway = InferenceGateway(config)

        # Mock all backends to fail initialization
        with patch('core.inference.gateway.LMSTUDIO_AVAILABLE', False):
            with patch.object(LlamaCppBackend, 'initialize', new_callable=AsyncMock, return_value=False):
                with patch.object(OllamaBackend, 'initialize', new_callable=AsyncMock, return_value=False):
                    success = await gateway.initialize()

        assert not success
        assert gateway.status == InferenceStatus.OFFLINE

    @pytest.mark.asyncio
    async def test_fallback_allowed_when_not_required(self):
        """With require_local=False, fallbacks should be tried."""
        config = InferenceConfig(require_local=False, fallbacks=["ollama"])
        gateway = InferenceGateway(config)

        # Mock LM Studio to fail, Ollama to succeed
        with patch('core.inference.gateway.LMSTUDIO_AVAILABLE', False):
            with patch.object(LlamaCppBackend, 'initialize', new_callable=AsyncMock, return_value=False):
                with patch.object(OllamaBackend, 'initialize', new_callable=AsyncMock, return_value=True):
                    success = await gateway.initialize()

        assert success
        assert gateway.status == InferenceStatus.DEGRADED


class TestMetrics:
    """Tests for metrics tracking."""

    @pytest.mark.asyncio
    async def test_metrics_updated_after_inference(self, gateway):
        """Metrics should update after inference."""
        # Setup mock backend
        mock_backend = MagicMock()
        mock_backend.generate = AsyncMock(return_value="Generated response")
        mock_backend.backend_type = InferenceBackend.LLAMACPP
        mock_backend.get_loaded_model = MagicMock(return_value="test-model")

        gateway._active_backend = mock_backend
        gateway._backends[ComputeTier.LOCAL] = mock_backend
        gateway.status = InferenceStatus.READY

        initial_requests = gateway._total_requests

        await gateway.infer("Test prompt")

        assert gateway._total_requests == initial_requests + 1
        assert gateway._total_tokens > 0
        assert gateway._total_latency_ms > 0


class TestHealthCheck:
    """Tests for gateway health status."""

    @pytest.mark.asyncio
    async def test_health_returns_status(self, gateway):
        """Health check should return current status."""
        gateway.status = InferenceStatus.READY

        health = await gateway.health()

        assert health["status"] == "ready"

    @pytest.mark.asyncio
    async def test_health_includes_stats(self, gateway):
        """Health check should include statistics."""
        gateway.status = InferenceStatus.READY
        gateway._total_requests = 100
        gateway._total_tokens = 5000
        gateway._total_latency_ms = 10000.0

        health = await gateway.health()

        assert health["stats"]["total_requests"] == 100
        assert health["stats"]["total_tokens"] == 5000
        assert health["stats"]["avg_latency_ms"] == 100.0


class TestBackendInterface:
    """Tests for backend abstract interface."""

    def test_llamacpp_backend_type(self, config):
        """LlamaCpp should report correct backend type."""
        backend = LlamaCppBackend(config)

        assert backend.backend_type == InferenceBackend.LLAMACPP

    def test_ollama_backend_type(self, config):
        """Ollama should report correct backend type."""
        backend = OllamaBackend(config)

        assert backend.backend_type == InferenceBackend.OLLAMA


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_inference_gateway_returns_same_instance(self):
        """get_inference_gateway should return same instance."""
        # Reset singleton for test
        import core.inference.gateway as gw_module
        gw_module._gateway_instance = None

        gw1 = get_inference_gateway()
        gw2 = get_inference_gateway()

        assert gw1 is gw2


class TestTierConfigs:
    """Tests for tier configuration constants."""

    def test_edge_tier_config(self):
        """EDGE tier should have CPU-only settings."""
        edge = TIER_CONFIGS["EDGE"]

        assert edge["n_gpu_layers"] == 0
        assert edge["target_speed"] == 12

    def test_local_tier_config(self):
        """LOCAL tier should have GPU settings."""
        local = TIER_CONFIGS["LOCAL"]

        assert local["n_gpu_layers"] == -1
        assert local["target_speed"] == 35

    def test_pool_tier_config(self):
        """POOL tier should be for federated compute."""
        pool = TIER_CONFIGS["POOL"]

        assert pool["default_model"] is None  # Federated


class TestInferenceBackendEnum:
    """Tests for InferenceBackend enum."""

    def test_backend_values(self):
        """Backend enum should have expected values."""
        assert InferenceBackend.LLAMACPP.value == "llamacpp"
        assert InferenceBackend.OLLAMA.value == "ollama"
        assert InferenceBackend.LMSTUDIO.value == "lmstudio"
        assert InferenceBackend.POOL.value == "pool"
        assert InferenceBackend.OFFLINE.value == "offline"


class TestInferenceStatusEnum:
    """Tests for InferenceStatus enum."""

    def test_status_values(self):
        """Status enum should have expected values."""
        assert InferenceStatus.COLD.value == "cold"
        assert InferenceStatus.WARMING.value == "warming"
        assert InferenceStatus.READY.value == "ready"
        assert InferenceStatus.DEGRADED.value == "degraded"
        assert InferenceStatus.OFFLINE.value == "offline"
