"""
BIZRA MULTI-MODAL ROUTER
═══════════════════════════════════════════════════════════════════════════════

Intelligent task routing to the right local model based on input modality and
task requirements. Routes to reasoning, vision, voice, or general models.

Architecture:
- ModelCapability: What each model can do (reasoning, vision, voice, general)
- MultiModalConfig: Registry of available models with their capabilities
- MultiModalRouter: Routes tasks to optimal model based on input analysis

Giants Protocol:
- Shazeer (Mixture of Experts): Sparse, dynamic routing
- Graves (Adaptive Computation): Route to right model for right task
- Vaswani (Attention): Multi-modal attention over available models

Created: 2026-02-04 | BIZRA Sovereignty
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CAPABILITIES
# ═══════════════════════════════════════════════════════════════════════════════


class ModelCapability(str, Enum):
    """Model capability classification."""

    REASONING = "reasoning"  # Chain-of-thought, complex thinking, mathematical proof
    VISION = "vision"  # Image understanding, visual analysis, OCR
    VOICE = "voice"  # Speech recognition, audio processing
    AGENTIC = "agentic"  # Workflow planning, task decomposition, orchestration
    GENERAL = "general"  # Text generation, Q&A, summarization


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelInfo:
    """Information about a registered model."""

    name: str
    capabilities: List[ModelCapability]
    primary_capability: ModelCapability
    backend: str  # "lmstudio" or "ollama"
    endpoint: str  # e.g., "192.168.56.1:1234"
    params_b: float  # Model size in billions
    context_length: int = 4096
    speed_tok_per_sec: float = 0.0  # Approximate
    description: str = ""


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    model: ModelInfo
    capability_match: ModelCapability
    confidence: float  # 0.0-1.0, how confident in this choice
    reason: str
    alternatives: List[ModelInfo] = field(default_factory=list)


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal routing."""

    model_registry: Dict[str, ModelInfo] = field(default_factory=dict)

    # Default backend endpoints
    lmstudio_endpoint: str = "192.168.56.1:1234"
    ollama_endpoint: str = "localhost:11434"

    # Routing preferences
    prefer_reasoning_models: bool = True  # Use deepseek-r1, qwq for complex tasks
    enable_fallback: bool = True  # Fallback to general model if specific not available
    latency_aware: bool = True  # Consider model speed


# ═══════════════════════════════════════════════════════════════════════════════
# TASK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


class TaskTypeDetector:
    """Detects required capability from input/task."""

    # Patterns indicating reasoning/thinking tasks
    REASONING_PATTERNS = [
        r"(?i)(prove|derive|analyze|synthesize|explain why|research)",
        r"(?i)(compare|evaluate|trade-off|pros and cons)",
        r"(?i)(design|architect|algorithm|mathematical)",
        r"(?i)(debug|troubleshoot|diagnose|root cause)",
        r"(?i)(step by step|chain of thought|reasoning)",
        r"(?i)(why does|why is|why do|how does|how is|how do)",
        r"(?i)(memory leak|performance issue|bottleneck|optimization)",
    ]

    # Patterns indicating vision tasks
    VISION_PATTERNS = [
        r"(?i)(image|photo|picture|screenshot|diagram)",
        r"(?i)(visual|see|look at|describe.*image)",
        r"(?i)(ocr|read.*text|extract.*image)",
        r"(?i)(analyze.*figure|chart|graph)",
    ]

    # Patterns indicating voice tasks
    VOICE_PATTERNS = [
        r"(?i)(audio|sound|voice|speech)",
        r"(?i)(transcribe|transcription|speech.*text)",
        r"(?i)(listen|hearing|acoustic)",
        r"(?i)(recording|spoken|saying)",
    ]

    # Patterns indicating agentic/planning tasks
    AGENTIC_PATTERNS = [
        r"(?i)(plan|planning|workflow|orchestrat)",
        r"(?i)(decompose|break.*down|split.*task)",
        r"(?i)(multi.*step|sequence|pipeline)",
        r"(?i)(coordinate|schedule|prioritize)",
        r"(?i)(agent|swarm|delegate|assign)",
        r"(?i)(automate|automation|batch.*process)",
    ]

    @staticmethod
    def detect_input_type(
        text: Optional[str] = None,
        has_image: bool = False,
        has_audio: bool = False,
        input_type: Optional[str] = None,
    ) -> ModelCapability:
        """
        Detect required capability from input characteristics.

        Args:
            text: Text description or query
            has_image: Whether input contains image
            has_audio: Whether input contains audio
            input_type: Explicit type hint ("image", "audio", "text")

        Returns:
            Required ModelCapability
        """
        # Direct input type takes precedence
        if input_type == "image" or has_image:
            return ModelCapability.VISION
        if input_type == "audio" or has_audio:
            return ModelCapability.VOICE

        # Pattern matching on text
        if text:
            text_lower = text.lower()

            for pattern in TaskTypeDetector.VOICE_PATTERNS:
                if re.search(pattern, text_lower):
                    return ModelCapability.VOICE

            for pattern in TaskTypeDetector.VISION_PATTERNS:
                if re.search(pattern, text_lower):
                    return ModelCapability.VISION

            for pattern in TaskTypeDetector.AGENTIC_PATTERNS:
                if re.search(pattern, text_lower):
                    return ModelCapability.AGENTIC

            for pattern in TaskTypeDetector.REASONING_PATTERNS:
                if re.search(pattern, text_lower):
                    return ModelCapability.REASONING

        # Default to general
        return ModelCapability.GENERAL


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-MODAL ROUTER
# ═══════════════════════════════════════════════════════════════════════════════


class MultiModalRouter:
    """
    Routes tasks to the right local model based on capability requirements.

    Uses Mixture of Experts approach: sparse, dynamic routing to specialized
    models for best performance per capability.
    """

    def __init__(self, config: Optional[MultiModalConfig] = None):
        self.config = config or MultiModalConfig()
        self._load_default_registry()

    def _load_default_registry(self):
        """Load default model registry with actual LM Studio models."""

        # ═══════════════════════════════════════════════════════════════════════════
        # REASONING MODELS (Complex thinking, CoT, mathematical reasoning)
        # ═══════════════════════════════════════════════════════════════════════════
        self.config.model_registry["deepseek/deepseek-r1-0528-qwen3-8b"] = ModelInfo(
            name="deepseek/deepseek-r1-0528-qwen3-8b",
            capabilities=[ModelCapability.REASONING, ModelCapability.GENERAL],
            primary_capability=ModelCapability.REASONING,
            backend="lmstudio",
            endpoint=self.config.lmstudio_endpoint,
            params_b=8.0,
            context_length=131072,  # 131K context!
            speed_tok_per_sec=25.0,
            description="DeepSeek R1 (Qwen3 8B): Advanced reasoning with 131K context",
        )

        self.config.model_registry["mistralai/ministral-3-14b-reasoning"] = ModelInfo(
            name="mistralai/ministral-3-14b-reasoning",
            capabilities=[ModelCapability.REASONING, ModelCapability.GENERAL],
            primary_capability=ModelCapability.REASONING,
            backend="lmstudio",
            endpoint=self.config.lmstudio_endpoint,
            params_b=14.0,
            context_length=32768,
            speed_tok_per_sec=15.0,
            description="Ministral 14B: Large-scale reasoning specialist",
        )

        self.config.model_registry["qwen/qwen3-4b-thinking-2507"] = ModelInfo(
            name="qwen/qwen3-4b-thinking-2507",
            capabilities=[ModelCapability.REASONING, ModelCapability.GENERAL],
            primary_capability=ModelCapability.REASONING,
            backend="lmstudio",
            endpoint=self.config.lmstudio_endpoint,
            params_b=4.0,
            context_length=32768,
            speed_tok_per_sec=35.0,
            description="Qwen3 4B Thinking: Fast reasoning model",
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # AGENTIC MODELS (Workflow planning, task decomposition, orchestration)
        # ═══════════════════════════════════════════════════════════════════════════
        self.config.model_registry["agentflow-planner-7b-i1"] = ModelInfo(
            name="agentflow-planner-7b-i1",
            capabilities=[
                ModelCapability.AGENTIC,
                ModelCapability.REASONING,
                ModelCapability.GENERAL,
            ],
            primary_capability=ModelCapability.AGENTIC,
            backend="lmstudio",
            endpoint=self.config.lmstudio_endpoint,
            params_b=7.0,
            context_length=32768,
            speed_tok_per_sec=25.0,
            description="AgentFlow Planner: Task decomposition, workflow planning",
        )

        self.config.model_registry[
            "llama-3.2-8x3b-moe-dark-champion-instruct-uncensored-abliterated-18.4b"
        ] = ModelInfo(
            name="llama-3.2-8x3b-moe-dark-champion-instruct-uncensored-abliterated-18.4b",
            capabilities=[ModelCapability.AGENTIC, ModelCapability.GENERAL],
            primary_capability=ModelCapability.AGENTIC,
            backend="lmstudio",
            endpoint=self.config.lmstudio_endpoint,
            params_b=18.4,
            context_length=32768,
            speed_tok_per_sec=20.0,
            description="Dark Champion MoE 18.4B: Powerful uncensored agentic model",
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # VISION MODELS (Image understanding, OCR, visual analysis)
        # ═══════════════════════════════════════════════════════════════════════════
        self.config.model_registry["qwen/qwen3-vl-8b"] = ModelInfo(
            name="qwen/qwen3-vl-8b",
            capabilities=[ModelCapability.VISION, ModelCapability.GENERAL],
            primary_capability=ModelCapability.VISION,
            backend="lmstudio",
            endpoint=self.config.lmstudio_endpoint,
            params_b=8.0,
            context_length=32768,
            speed_tok_per_sec=20.0,
            description="Qwen3 VL 8B: High-quality vision-language model",
        )

        self.config.model_registry["qwen/qwen3-vl-4b"] = ModelInfo(
            name="qwen/qwen3-vl-4b",
            capabilities=[ModelCapability.VISION, ModelCapability.GENERAL],
            primary_capability=ModelCapability.VISION,
            backend="lmstudio",
            endpoint=self.config.lmstudio_endpoint,
            params_b=4.0,
            context_length=32768,
            speed_tok_per_sec=30.0,
            description="Qwen3 VL 4B: Fast vision model for real-time tasks",
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # EMBEDDING MODEL (Vector embeddings for RAG, search)
        # ═══════════════════════════════════════════════════════════════════════════
        self.config.model_registry["text-embedding-nomic-embed-text-v1.5"] = ModelInfo(
            name="text-embedding-nomic-embed-text-v1.5",
            capabilities=[ModelCapability.GENERAL],  # Special: embedding only
            primary_capability=ModelCapability.GENERAL,
            backend="lmstudio",
            endpoint=self.config.lmstudio_endpoint,
            params_b=0.1,
            context_length=8192,
            speed_tok_per_sec=100.0,
            description="Nomic Embed Text v1.5: Fast text embeddings",
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # GENERAL MODELS (Default, text generation, Q&A, summarization)
        # ═══════════════════════════════════════════════════════════════════════════
        self.config.model_registry["qwen2.5-14b_uncensored_instruct"] = ModelInfo(
            name="qwen2.5-14b_uncensored_instruct",
            capabilities=[ModelCapability.GENERAL, ModelCapability.REASONING],
            primary_capability=ModelCapability.GENERAL,
            backend="lmstudio",
            endpoint=self.config.lmstudio_endpoint,
            params_b=14.0,
            context_length=32768,
            speed_tok_per_sec=15.0,
            description="Qwen 2.5 14B Uncensored: Large general-purpose model",
        )

        self.config.model_registry["chuanli11_-_llama-3.2-3b-instruct-uncensored"] = (
            ModelInfo(
                name="chuanli11_-_llama-3.2-3b-instruct-uncensored",
                capabilities=[ModelCapability.GENERAL],
                primary_capability=ModelCapability.GENERAL,
                backend="lmstudio",
                endpoint=self.config.lmstudio_endpoint,
                params_b=3.0,
                context_length=8192,
                speed_tok_per_sec=40.0,
                description="Llama 3.2 3B Uncensored: Fast general model",
            )
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # NANO MODELS (Ultra-fast, edge deployment)
        # ═══════════════════════════════════════════════════════════════════════════
        self.config.model_registry["qwen2.5-0.5b-instruct"] = ModelInfo(
            name="qwen2.5-0.5b-instruct",
            capabilities=[ModelCapability.GENERAL],
            primary_capability=ModelCapability.GENERAL,
            backend="lmstudio",
            endpoint=self.config.lmstudio_endpoint,
            params_b=0.5,
            context_length=32768,
            speed_tok_per_sec=80.0,
            description="Qwen 2.5 0.5B: Ultra-fast nano model",
        )

        self.config.model_registry["nvidia/nemotron-3-nano"] = ModelInfo(
            name="nvidia/nemotron-3-nano",
            capabilities=[ModelCapability.GENERAL],
            primary_capability=ModelCapability.GENERAL,
            backend="lmstudio",
            endpoint=self.config.lmstudio_endpoint,
            params_b=1.0,
            context_length=4096,
            speed_tok_per_sec=70.0,
            description="Nemotron 3 Nano: NVIDIA's fast nano model",
        )

        self.config.model_registry["liquid/lfm2.5-1.2b"] = ModelInfo(
            name="liquid/lfm2.5-1.2b",
            capabilities=[ModelCapability.GENERAL],
            primary_capability=ModelCapability.GENERAL,
            backend="lmstudio",
            endpoint=self.config.lmstudio_endpoint,
            params_b=1.2,
            context_length=4096,
            speed_tok_per_sec=60.0,
            description="Liquid LFM 1.2B: Efficient small model",
        )

        self.config.model_registry["ibm/granite-4-h-tiny"] = ModelInfo(
            name="ibm/granite-4-h-tiny",
            capabilities=[ModelCapability.GENERAL],
            primary_capability=ModelCapability.GENERAL,
            backend="lmstudio",
            endpoint=self.config.lmstudio_endpoint,
            params_b=0.5,
            context_length=4096,
            speed_tok_per_sec=85.0,
            description="IBM Granite 4 Tiny: Enterprise-grade nano model",
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # VOICE MODEL (Full-duplex speech-to-speech via PersonaPlex)
        # ═══════════════════════════════════════════════════════════════════════════
        self.config.model_registry["nvidia/personaplex-7b-v1"] = ModelInfo(
            name="nvidia/personaplex-7b-v1",
            capabilities=[ModelCapability.VOICE, ModelCapability.GENERAL],
            primary_capability=ModelCapability.VOICE,
            backend="personaplex",  # Special backend: routes to BIZRAPersonaPlex
            endpoint="localhost:8998",  # PersonaPlex WebSocket server
            params_b=7.0,
            context_length=4096,
            speed_tok_per_sec=50.0,  # Real-time voice @ 12.5 FPS
            description="PersonaPlex 7B: Full-duplex speech-to-speech with 16 voices and 8 BIZRA Guardians",
        )

    def detect_task_type(
        self,
        text: Optional[str] = None,
        has_image: bool = False,
        has_audio: bool = False,
        input_type: Optional[str] = None,
    ) -> ModelCapability:
        """
        Detect required capability from input.

        Args:
            text: Text query or description
            has_image: Whether input contains image
            has_audio: Whether input contains audio
            input_type: Explicit type hint ("image", "audio", "text", "reasoning")

        Returns:
            Required ModelCapability
        """
        return TaskTypeDetector.detect_input_type(
            text=text,
            has_image=has_image,
            has_audio=has_audio,
            input_type=input_type,
        )

    def select_model(self, capability: ModelCapability) -> RoutingDecision:
        """
        Select best available model for capability.

        Selection algorithm (Mixture of Experts):
        1. Filter models supporting the capability
        2. Prefer models with it as primary capability
        3. Consider latency (speed) for real-time tasks
        4. Fall back to general model if needed

        Args:
            capability: Required ModelCapability

        Returns:
            RoutingDecision with selected model and reasoning
        """
        # Find exact matches (primary capability)
        exact_matches = [
            m
            for m in self.config.model_registry.values()
            if m.primary_capability == capability
        ]

        if exact_matches:
            # Prefer faster model for real-time capabilities
            if capability in [
                ModelCapability.VISION,
                ModelCapability.VOICE,
                ModelCapability.AGENTIC,
            ]:
                best = max(exact_matches, key=lambda m: m.speed_tok_per_sec)
                confidence = 0.95
            else:
                # For reasoning, size and capability matter more than speed
                best = max(exact_matches, key=lambda m: m.params_b)
                confidence = 0.95

            return RoutingDecision(
                model=best,
                capability_match=capability,
                confidence=confidence,
                reason=f"Routed to {best.name} ({best.primary_capability.value} specialist)",
                alternatives=exact_matches[1:3],
            )

        # Find secondary capability matches
        secondary_matches = [
            m
            for m in self.config.model_registry.values()
            if capability in m.capabilities
        ]

        if secondary_matches:
            best = secondary_matches[0]
            return RoutingDecision(
                model=best,
                capability_match=capability,
                confidence=0.75,
                reason=f"No {capability.value} specialist, using {best.name} with secondary capability",
                alternatives=secondary_matches[1:3],
            )

        # Fallback to general model
        if self.config.enable_fallback:
            general_models = [
                m
                for m in self.config.model_registry.values()
                if m.primary_capability == ModelCapability.GENERAL
            ]

            if general_models:
                best = general_models[0]
                return RoutingDecision(
                    model=best,
                    capability_match=ModelCapability.GENERAL,
                    confidence=0.5,
                    reason=f"No {capability.value} model, falling back to {best.name}",
                    alternatives=general_models[1:3],
                )

        # Last resort: return smallest available model
        all_models = list(self.config.model_registry.values())
        if all_models:
            best = min(all_models, key=lambda m: m.params_b)
            return RoutingDecision(
                model=best,
                capability_match=ModelCapability.GENERAL,
                confidence=0.25,
                reason=f"Routing to smallest available model: {best.name}",
            )

        raise ValueError("No models available in registry")

    def route(
        self,
        task: Union[str, Dict[str, Any]],
        explicit_type: Optional[str] = None,
    ) -> RoutingDecision:
        """
        Route a task to the appropriate model.

        Args:
            task: Task description (string) or task dict with metadata
            explicit_type: Optional explicit type hint to override detection

        Returns:
            RoutingDecision with selected model

        Example:
            decision = router.route("Analyze this image for text")
            print(f"Using: {decision.model.name}")
            print(f"Endpoint: {decision.model.endpoint}")
        """
        # Parse task
        if isinstance(task, str):
            text = task
            has_image = False
            has_audio = False
        elif isinstance(task, dict):
            text = task.get("text", task.get("query", ""))
            has_image = task.get("has_image", False)
            has_audio = task.get("has_audio", False)
        else:
            raise TypeError(f"Task must be str or dict, got {type(task)}")

        # Detect capability
        capability = self.detect_task_type(
            text=text,
            has_image=has_image,
            has_audio=has_audio,
            input_type=explicit_type,
        )

        # Select model
        return self.select_model(capability)

    def register_model(self, model: ModelInfo):
        """Register a custom model."""
        self.config.model_registry[model.name] = model

    def list_models(self) -> List[ModelInfo]:
        """List all registered models."""
        return list(self.config.model_registry.values())

    def list_by_capability(self, capability: ModelCapability) -> List[ModelInfo]:
        """List all models supporting a capability."""
        return [
            m
            for m in self.config.model_registry.values()
            if capability in m.capabilities
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY & SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_router_instance: Optional[MultiModalRouter] = None


def get_multimodal_router(
    config: Optional[MultiModalConfig] = None,
) -> MultiModalRouter:
    """Get singleton multi-modal router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = MultiModalRouter(config)
    return _router_instance


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 75)
    print("    BIZRA MULTI-MODAL ROUTER")
    print("═" * 75)

    router = get_multimodal_router()

    # Test cases
    test_cases = [
        ("What is 2+2?", None, "simple math"),
        (
            "Prove that every even number > 2 is the sum of two primes",
            None,
            "reasoning",
        ),
        ("Describe what you see in this image", True, "vision"),
        ("Transcribe the speech in this audio clip", False, "voice"),
        (
            {
                "text": "Analyze the trade-offs between monolithic vs microservices",
                "has_image": False,
            },
            None,
            "complex reasoning",
        ),
    ]

    for task, has_image, description in test_cases:
        if isinstance(task, dict):
            decision = router.route(task)
        else:
            decision = router.route(task, explicit_type=None)

        print(f"\n📋 Task: {description}")
        print(
            f"   Input: {task if isinstance(task, str) else task.get('text', '')[:50]}"
        )
        print(f"   Detected capability: {decision.capability_match.value}")
        print(f"   Selected model: {decision.model.name}")
        print(f"   Backend: {decision.model.backend} @ {decision.model.endpoint}")
        print(f"   Confidence: {decision.confidence:.0%}")
        print(f"   Reason: {decision.reason}")

    print("\n" + "═" * 75)
    print("Available models by capability:")
    for cap in ModelCapability:
        models = router.list_by_capability(cap)
        if models:
            print(f"\n  {cap.value.upper()}:")
            for m in models:
                print(f"    • {m.name} ({m.params_b}B) - {m.description}")

    print("\n" + "═" * 75)
