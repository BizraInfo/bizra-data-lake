"""
BIZRA Cognitive Module - LLM Integration Layer

T1.2 Hook Integration per Blueprint v1.0

Components:
- direct_client: Lightweight LM Studio/Ollama client
- hooks: FATE-gated cognitive hooks with Ihsān scoring
- rlm_adapter: RLM library adapter (for recursive reasoning)

Usage:
    from core.cognitive import invoke, get_hook

    # Quick invocation
    result = invoke("What is 2+2?")

    # Named hooks
    from core.cognitive import get_hook
    planner = get_hook("planner")
    result = planner.invoke("Create a project plan for X")
"""

from core.cognitive.direct_client import (
    BizraDirectClient,
    CompletionResult,
    MODEL_SLOTS,
    quick_completion,
    get_client,
)

from core.cognitive.hooks import (
    CognitiveHook,
    HookConfig,
    HookResult,
    HookType,
    register_hook,
    get_hook,
    invoke,
    IHSAN_THRESHOLD,
)

__all__ = [
    # Direct Client
    "BizraDirectClient",
    "CompletionResult",
    "MODEL_SLOTS",
    "quick_completion",
    "get_client",
    # Hooks
    "CognitiveHook",
    "HookConfig",
    "HookResult",
    "HookType",
    "register_hook",
    "get_hook",
    "invoke",
    "IHSAN_THRESHOLD",
]
