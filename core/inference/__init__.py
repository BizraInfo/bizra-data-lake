"""
BIZRA INFERENCE ENGINE
===============================================================================

Tiered local inference for sovereign AI.

Components:
- gateway: Core inference gateway with tiered backends
- selector: Adaptive model selection based on task complexity
- unified: Complete inference system with routing + tracking
- lmstudio_backend: LM Studio v1 API integration (primary backend)
- local_first_config: Bicameral mind architecture (Cold Core + Warm Surface)

Tiers:
- EDGE/NANO: Always-on, low-power (0.5B-1.5B models)
- LOCAL/MEDIUM: On-demand, high-power (7B models, RTX 4090)
- POOL/LARGE: Federated URP compute (70B+ models)

Backends:
- LM Studio (192.168.56.1:1234) - Primary, native v1 API with MCP + stateful chats
- Ollama (localhost:11434) - Fallback

Bicameral Architecture (Jaynes, 1976):
- Cold Core: Logical reasoning (DeepSeek-R1, Qwen-Coder)
- Warm Surface: Creative verification (Mistral-Nemo)
- Generate-Verify Loop (Karpathy, 2024)

Created: 2026-01-29 | BIZRA Sovereignty
Updated: 2026-01-30 | Added selector + unified system
Updated: 2026-02-01 | Added LM Studio v1 API backend
Updated: 2026-02-05 | Added Bicameral Mind local-first config
Updated: 2026-02-05 | Added response_utils for DeepSeek R1 think token stripping
"""

from ._backends import LlamaCppBackend, OllamaBackend
from .gateway import (  # type: ignore[attr-defined]
    ComputeTier,
    InferenceConfig,
    InferenceGateway,
    InferenceResult,
)
from .local_first import (
    BackendStatus,
    LocalBackend,
    LocalFirstDetector,
    get_local_first_backend,
)
from .multimodal import (
    ModelCapability,
)
from .multimodal import ModelInfo as MultiModalModelInfo
from .multimodal import (
    MultiModalConfig,
    MultiModalRouter,
    RoutingDecision,
    TaskTypeDetector,
    get_multimodal_router,
)
from .response_utils import (
    extract_think_content,
    has_think_tokens,
    normalize_response,
    strip_all_reasoning_tokens,
    strip_reasoning_tokens,
    strip_think_tokens,
)
from .selector import (
    AdaptiveModelSelector,
    LatencyClass,
    ModelTier,
    TaskAnalyzer,
    TaskComplexity,
    get_model_selector,
    get_task_analyzer,
)
from .unified import (
    UnifiedInferenceResult,
    UnifiedInferenceSystem,
    get_inference_system,
)

# Bicameral Mind Local-First Configuration (requires httpx)
_BICAMERAL_AVAILABLE = False
try:
    from .local_first_config import (
        LOCAL_MODELS,
        TASK_TO_ROLE,
        BicameralOrchestrator,
        BicameralResult,
        FallbackLevel,
        HealthMonitor,
        HealthReport,
        HealthStatus,
        LocalFirstManager,
        ModelConfig,
        ModelRole,
        ModelRouter,
    )
    from .local_first_config import (
        RoutingDecision as BicameralRoutingDecision,  # Enums; Data models; Registry; Classes; Convenience functions
    )
    from .local_first_config import (
        TaskType,
        get_bicameral_orchestrator,
        get_local_first_manager,
        get_model_router,
    )

    _BICAMERAL_AVAILABLE = True
except ImportError:
    # Define placeholder classes for when httpx is not available
    class ModelRole:  # type: ignore
        """Placeholder - requires httpx."""

        pass

    class TaskType:  # type: ignore
        """Placeholder - requires httpx."""

        pass

    class FallbackLevel:  # type: ignore
        """Placeholder - requires httpx."""

        pass

    class HealthStatus:  # type: ignore
        """Placeholder - requires httpx."""

        pass

    class ModelConfig:  # type: ignore
        """Placeholder - requires httpx."""

        pass

    class HealthReport:  # type: ignore
        """Placeholder - requires httpx."""

        pass

    class BicameralRoutingDecision:  # type: ignore
        """Placeholder - requires httpx."""

        pass

    class BicameralResult:  # type: ignore
        """Placeholder - requires httpx."""

        pass

    LOCAL_MODELS = {}  # type: ignore

    TASK_TO_ROLE = {}  # type: ignore

    class ModelRouter:  # type: ignore
        """Placeholder - requires httpx."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Bicameral config requires httpx. Install with: pip install httpx"
            )

    class HealthMonitor:  # type: ignore
        """Placeholder - requires httpx."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Bicameral config requires httpx. Install with: pip install httpx"
            )

    class BicameralOrchestrator:  # type: ignore
        """Placeholder - requires httpx."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Bicameral config requires httpx. Install with: pip install httpx"
            )

    class LocalFirstManager:  # type: ignore
        """Placeholder - requires httpx."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Bicameral config requires httpx. Install with: pip install httpx"
            )

    async def get_local_first_manager():  # type: ignore
        raise ImportError(
            "Bicameral config requires httpx. Install with: pip install httpx"
        )

    def get_model_router():  # type: ignore
        raise ImportError(
            "Bicameral config requires httpx. Install with: pip install httpx"
        )

    def get_bicameral_orchestrator():  # type: ignore
        raise ImportError(
            "Bicameral config requires httpx. Install with: pip install httpx"
        )


# Voice backend (requires personaplex)
_VOICE_AVAILABLE = False
try:
    from .voice_backend import (
        VoiceBackend,
        VoiceConfig,
        VoiceRequest,
        VoiceResponse,
        check_voice_availability,
        get_voice_backend,
    )

    _VOICE_AVAILABLE = True
except ImportError:
    # Define placeholder classes
    class VoiceBackend:  # type: ignore
        """Placeholder - requires personaplex."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Voice backend requires personaplex. Install with: pip install personaplex"
            )

    class VoiceConfig:  # type: ignore
        """Placeholder - requires personaplex."""

        pass

    class VoiceRequest:  # type: ignore
        """Placeholder - requires personaplex."""

        pass

    class VoiceResponse:  # type: ignore
        """Placeholder - requires personaplex."""

        pass

    def get_voice_backend(*args, **kwargs):  # type: ignore
        raise ImportError(
            "Voice backend requires personaplex. Install with: pip install personaplex"
        )

    async def check_voice_availability():  # type: ignore
        return False


# LM Studio backend (optional - requires httpx)
_LMSTUDIO_AVAILABLE = False
try:
    from .lmstudio_backend import (
        ChatMessage,
        ChatResponse,
        LMStudioBackend,
        LMStudioConfig,
        LMStudioEndpoint,
        ModelInfo,
        create_lmstudio_backend,
    )

    _LMSTUDIO_AVAILABLE = True
except ImportError:
    # Define placeholder classes
    class LMStudioBackend:  # type: ignore
        """Placeholder - requires httpx."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LM Studio backend requires httpx. Install with: pip install httpx"
            )

    class LMStudioConfig:  # type: ignore
        """Placeholder - requires httpx."""

        pass

    class LMStudioEndpoint:  # type: ignore
        """Placeholder - requires httpx."""

        pass

    class ModelInfo:  # type: ignore
        """Placeholder - requires httpx."""

        pass

    class ChatMessage:  # type: ignore
        """Placeholder - requires httpx."""

        pass

    class ChatResponse:  # type: ignore
        """Placeholder - requires httpx."""

        pass

    def create_lmstudio_backend(*args, **kwargs):  # type: ignore
        raise ImportError(
            "LM Studio backend requires httpx. Install with: pip install httpx"
        )


__all__ = [
    # Response Utilities (DeepSeek R1 think token handling)
    "strip_think_tokens",
    "strip_reasoning_tokens",
    "strip_all_reasoning_tokens",
    "normalize_response",
    "extract_think_content",
    "has_think_tokens",
    # Gateway
    "InferenceGateway",
    "ComputeTier",
    "InferenceConfig",
    "InferenceResult",
    # Backends
    "LlamaCppBackend",
    "OllamaBackend",
    "LMStudioBackend",
    "LMStudioConfig",
    "LMStudioEndpoint",
    "create_lmstudio_backend",
    # LM Studio types
    "ModelInfo",
    "ChatMessage",
    "ChatResponse",
    # Selector
    "AdaptiveModelSelector",
    "TaskAnalyzer",
    "ModelTier",
    "TaskComplexity",
    "LatencyClass",
    "get_model_selector",
    "get_task_analyzer",
    # Unified
    "UnifiedInferenceSystem",
    "UnifiedInferenceResult",
    "get_inference_system",
    # Local-First (Zero-Token Operation)
    "LocalFirstDetector",
    "LocalBackend",
    "BackendStatus",
    "get_local_first_backend",
    # Multi-Modal Router
    "MultiModalRouter",
    "MultiModalConfig",
    "ModelCapability",
    "MultiModalModelInfo",
    "RoutingDecision",
    "TaskTypeDetector",
    "get_multimodal_router",
    # Voice Backend (PersonaPlex)
    "VoiceBackend",
    "VoiceConfig",
    "VoiceRequest",
    "VoiceResponse",
    "get_voice_backend",
    "check_voice_availability",
    # Bicameral Mind Local-First Config
    "ModelRole",
    "TaskType",
    "FallbackLevel",
    "HealthStatus",
    "ModelConfig",
    "HealthReport",
    "BicameralRoutingDecision",
    "BicameralResult",
    "LOCAL_MODELS",
    "TASK_TO_ROLE",
    "ModelRouter",
    "HealthMonitor",
    "BicameralOrchestrator",
    "LocalFirstManager",
    "get_local_first_manager",
    "get_model_router",
    "get_bicameral_orchestrator",
]
