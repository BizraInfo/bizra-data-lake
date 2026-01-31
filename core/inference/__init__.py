"""
BIZRA INFERENCE ENGINE
═══════════════════════════════════════════════════════════════════════════════

Tiered local inference for sovereign AI.

Components:
- gateway: Core inference gateway with tiered backends
- selector: Adaptive model selection based on task complexity
- unified: Complete inference system with routing + tracking
- lmstudio_backend: LM Studio v1 API integration (primary backend)

Tiers:
- EDGE/NANO: Always-on, low-power (0.5B-1.5B models)
- LOCAL/MEDIUM: On-demand, high-power (7B models, RTX 4090)
- POOL/LARGE: Federated URP compute (70B+ models)

Backends:
- LM Studio (192.168.56.1:1234) - Primary, native v1 API with MCP + stateful chats
- Ollama (localhost:11434) - Fallback

Created: 2026-01-29 | BIZRA Sovereignty
Updated: 2026-01-30 | Added selector + unified system
Updated: 2026-02-01 | Added LM Studio v1 API backend
"""

from .gateway import InferenceGateway, ComputeTier, InferenceConfig, InferenceResult
from .backends import LlamaCppBackend, OllamaBackend
from .selector import (
    AdaptiveModelSelector,
    TaskAnalyzer,
    ModelTier,
    TaskComplexity,
    LatencyClass,
    get_model_selector,
    get_task_analyzer,
)
from .unified import UnifiedInferenceSystem, UnifiedInferenceResult, get_inference_system

# LM Studio backend (optional - requires httpx)
_LMSTUDIO_AVAILABLE = False
try:
    from .lmstudio_backend import (
        LMStudioBackend,
        LMStudioConfig,
        LMStudioEndpoint,
        ModelInfo,
        ChatMessage,
        ChatResponse,
        create_lmstudio_backend,
    )
    _LMSTUDIO_AVAILABLE = True
except ImportError:
    # Define placeholder classes
    class LMStudioBackend:  # type: ignore
        """Placeholder - requires httpx."""
        def __init__(self, *args, **kwargs):
            raise ImportError("LM Studio backend requires httpx. Install with: pip install httpx")

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
        raise ImportError("LM Studio backend requires httpx. Install with: pip install httpx")

__all__ = [
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
]
