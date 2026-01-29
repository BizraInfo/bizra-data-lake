"""
BIZRA INFERENCE ENGINE
═══════════════════════════════════════════════════════════════════════════════

Tiered local inference for sovereign AI.

Components:
- gateway: Core inference gateway with tiered backends
- selector: Adaptive model selection based on task complexity
- unified: Complete inference system with routing + tracking

Tiers:
- EDGE/NANO: Always-on, low-power (0.5B-1.5B models)
- LOCAL/MEDIUM: On-demand, high-power (7B models, RTX 4090)
- POOL/LARGE: Federated URP compute (70B+ models)

Created: 2026-01-29 | BIZRA Sovereignty
Updated: 2026-01-30 | Added selector + unified system
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

__all__ = [
    # Gateway
    "InferenceGateway",
    "ComputeTier", 
    "InferenceConfig",
    "InferenceResult",
    
    # Backends
    "LlamaCppBackend",
    "OllamaBackend",
    
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
