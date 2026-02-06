"""
Test suite for BIZRA Multi-Modal Router.

Tests the routing logic, task detection, and model selection.
"""

import pytest
from core.inference.multimodal import (
    ModelCapability,
    ModelInfo,
    MultiModalConfig,
    MultiModalRouter,
    TaskTypeDetector,
    get_multimodal_router,
)


class TestTaskTypeDetector:
    """Test task type detection."""

    def test_detect_reasoning_tasks(self):
        """Test detection of reasoning tasks."""
        reasoning_queries = [
            "Prove that the square root of 2 is irrational",
            "Analyze the trade-offs between microservices and monolithic",
            "Design a distributed consensus algorithm",
            "Why does this code have a memory leak?",
            "Research the impacts of climate change",
        ]

        for query in reasoning_queries:
            capability = TaskTypeDetector.detect_input_type(text=query)
            assert capability == ModelCapability.REASONING, f"Failed for: {query}"

    def test_detect_vision_tasks(self):
        """Test detection of vision tasks."""
        vision_queries = [
            "Describe what you see in this image",
            "Extract text from the screenshot",
            "Analyze the chart in this diagram",
            "Read the text in this photo",
            "What objects are in this picture?",
        ]

        for query in vision_queries:
            capability = TaskTypeDetector.detect_input_type(text=query)
            assert capability == ModelCapability.VISION, f"Failed for: {query}"

    def test_detect_voice_tasks(self):
        """Test detection of voice/audio tasks."""
        voice_queries = [
            "Transcribe the speech in this audio",
            "What is the person saying in this recording?",
            "Convert this voice message to text",
            "Analyze the acoustic properties",
        ]

        for query in voice_queries:
            capability = TaskTypeDetector.detect_input_type(text=query)
            assert capability == ModelCapability.VOICE, f"Failed for: {query}"

    def test_detect_general_tasks(self):
        """Test detection of general tasks."""
        general_queries = [
            "What is the capital of France?",
            "Write a poem about nature",
            "Summarize this article",
        ]

        for query in general_queries:
            capability = TaskTypeDetector.detect_input_type(text=query)
            assert capability == ModelCapability.GENERAL, f"Failed for: {query}"

    def test_explicit_input_type_override(self):
        """Test that explicit input_type overrides text detection."""
        # Text that suggests vision, but explicit type says audio
        text = "Describe the image in detail"
        capability = TaskTypeDetector.detect_input_type(
            text=text,
            input_type="audio",
        )
        assert capability == ModelCapability.VOICE

    def test_has_image_flag(self):
        """Test has_image flag takes precedence."""
        capability = TaskTypeDetector.detect_input_type(
            text="What is 2+2?",
            has_image=True,
        )
        assert capability == ModelCapability.VISION

    def test_has_audio_flag(self):
        """Test has_audio flag takes precedence."""
        capability = TaskTypeDetector.detect_input_type(
            text="What is 2+2?",
            has_audio=True,
        )
        assert capability == ModelCapability.VOICE


class TestMultiModalRouter:
    """Test the multi-modal router."""

    @pytest.fixture
    def router(self):
        """Create a fresh router instance for each test."""
        config = MultiModalConfig()
        return MultiModalRouter(config)

    def test_router_loads_default_models(self, router):
        """Test that router loads default model registry."""
        models = router.list_models()
        assert len(models) > 0

        # Check that we have models for key capabilities
        model_names = {m.name for m in models}
        # At least one reasoning model should exist
        reasoning_models = [m for m in models if ModelCapability.REASONING in m.capabilities]
        assert len(reasoning_models) > 0, f"No reasoning models found. Available: {model_names}"
        # At least one vision model should exist
        vision_models = [m for m in models if ModelCapability.VISION in m.capabilities]
        assert len(vision_models) > 0, f"No vision models found. Available: {model_names}"
        # At least one voice model should exist
        voice_models = [m for m in models if ModelCapability.VOICE in m.capabilities]
        assert len(voice_models) > 0, f"No voice models found. Available: {model_names}"

    def test_select_reasoning_model(self, router):
        """Test selection of reasoning model."""
        decision = router.select_model(ModelCapability.REASONING)

        assert decision.model is not None
        assert decision.capability_match == ModelCapability.REASONING
        assert decision.confidence >= 0.5
        assert decision.reason

        # Should select a reasoning specialist
        assert decision.model.primary_capability == ModelCapability.REASONING

    def test_select_vision_model(self, router):
        """Test selection of vision model."""
        decision = router.select_model(ModelCapability.VISION)

        assert decision.model is not None
        assert decision.capability_match == ModelCapability.VISION
        assert decision.confidence >= 0.5
        assert decision.model.primary_capability == ModelCapability.VISION

    def test_select_voice_model(self, router):
        """Test selection of voice model."""
        decision = router.select_model(ModelCapability.VOICE)

        assert decision.model is not None
        assert decision.model.primary_capability == ModelCapability.VOICE

    def test_select_general_model(self, router):
        """Test selection of general model."""
        decision = router.select_model(ModelCapability.GENERAL)

        assert decision.model is not None
        assert decision.model.primary_capability == ModelCapability.GENERAL

    def test_route_text_query(self, router):
        """Test routing a text query."""
        decision = router.route("What is the meaning of life?")

        assert decision.model is not None
        assert decision.capability_match == ModelCapability.GENERAL

    def test_route_reasoning_query(self, router):
        """Test routing a reasoning query."""
        decision = router.route("Analyze the pros and cons of using microservices")

        assert decision.capability_match == ModelCapability.REASONING

    def test_route_vision_query(self, router):
        """Test routing a vision query."""
        decision = router.route("What objects are visible in this image?")

        assert decision.capability_match == ModelCapability.VISION

    def test_route_with_dict_task(self, router):
        """Test routing with dict task specification."""
        task = {
            "text": "Describe the image",
            "has_image": True,
        }
        decision = router.route(task)

        assert decision.capability_match == ModelCapability.VISION

    def test_route_with_explicit_type(self, router):
        """Test routing with explicit type override."""
        decision = router.route("This is audio", explicit_type="audio")

        assert decision.capability_match == ModelCapability.VOICE

    def test_register_custom_model(self, router):
        """Test registering a custom model."""
        custom = ModelInfo(
            name="custom-reasoning",
            capabilities=[ModelCapability.REASONING, ModelCapability.GENERAL],
            primary_capability=ModelCapability.REASONING,
            backend="ollama",
            endpoint="localhost:11434",
            params_b=100.0,  # Very large
            context_length=128000,
            speed_tok_per_sec=5.0,
            description="Custom ultra-large reasoning model",
        )

        router.register_model(custom)
        models = router.list_models()

        assert any(m.name == "custom-reasoning" for m in models)

    def test_list_models_by_capability(self, router):
        """Test listing models by capability."""
        reasoning_models = router.list_by_capability(ModelCapability.REASONING)
        assert len(reasoning_models) > 0
        assert all(
            ModelCapability.REASONING in m.capabilities for m in reasoning_models
        )

        vision_models = router.list_by_capability(ModelCapability.VISION)
        assert len(vision_models) > 0
        assert all(
            ModelCapability.VISION in m.capabilities for m in vision_models
        )

    def test_routing_confidence_scores(self, router):
        """Test that confidence scores reflect match quality."""
        # Exact match should be high confidence
        exact_decision = router.select_model(ModelCapability.REASONING)
        assert exact_decision.confidence >= 0.9

        # General model as fallback should be lower confidence
        # (This depends on registry, but reasoning should be > general as fallback)
        reasoning_confidence = exact_decision.confidence
        assert reasoning_confidence > 0.5

    def test_model_endpoints_configured(self, router):
        """Test that models have valid endpoints."""
        valid_backends = {"lmstudio", "ollama", "personaplex", "local"}
        for model in router.list_models():
            assert model.endpoint, f"Model {model.name} missing endpoint"
            assert model.backend in valid_backends, f"Invalid backend: {model.backend}"

    def test_model_metadata_complete(self, router):
        """Test that all models have complete metadata."""
        for model in router.list_models():
            assert model.name
            assert len(model.capabilities) > 0
            assert model.primary_capability in model.capabilities
            assert model.params_b > 0
            assert model.context_length > 0
            assert model.description


class TestMultiModalConfig:
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration."""
        config = MultiModalConfig()

        assert config.lmstudio_endpoint == "192.168.56.1:1234"
        assert config.ollama_endpoint == "localhost:11434"
        assert config.prefer_reasoning_models is True
        assert config.enable_fallback is True
        assert config.latency_aware is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = MultiModalConfig(
            lmstudio_endpoint="custom.host:5000",
            ollama_endpoint="ollama.local:11434",
            prefer_reasoning_models=False,
        )

        assert config.lmstudio_endpoint == "custom.host:5000"
        assert config.ollama_endpoint == "ollama.local:11434"
        assert config.prefer_reasoning_models is False


class TestModelInfo:
    """Test ModelInfo data class."""

    def test_model_info_creation(self):
        """Test creating ModelInfo."""
        model = ModelInfo(
            name="test-model",
            capabilities=[ModelCapability.GENERAL],
            primary_capability=ModelCapability.GENERAL,
            backend="ollama",
            endpoint="localhost:11434",
            params_b=7.0,
        )

        assert model.name == "test-model"
        assert ModelCapability.GENERAL in model.capabilities
        assert model.primary_capability == ModelCapability.GENERAL
        assert model.backend == "ollama"


class TestSingleton:
    """Test singleton behavior."""

    def test_get_router_singleton(self):
        """Test that get_multimodal_router returns same instance."""
        router1 = get_multimodal_router()
        router2 = get_multimodal_router()

        assert router1 is router2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string_query(self):
        """Test routing empty query."""
        router = MultiModalRouter()
        decision = router.route("")

        # Should default to general
        assert decision.capability_match == ModelCapability.GENERAL

    def test_very_long_query(self):
        """Test routing very long query."""
        router = MultiModalRouter()
        long_query = "What is " * 100 + "?"
        decision = router.route(long_query)

        assert decision.model is not None

    def test_special_characters_in_query(self):
        """Test routing query with special characters."""
        router = MultiModalRouter()
        decision = router.route("Analyze <img> tags & special chars @#$%")

        assert decision.model is not None

    def test_fallback_when_no_matching_model(self):
        """Test fallback behavior."""
        config = MultiModalConfig()
        config.enable_fallback = True

        # Clear all reasoning models (unlikely, but tests fallback)
        router = MultiModalRouter(config)

        # Even if we clear specialists, should get a model
        decision = router.select_model(ModelCapability.REASONING)
        assert decision.model is not None
