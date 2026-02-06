"""
+==============================================================================+
|   BIZRA BRIDGES -- Cross-System Integration                                   |
+==============================================================================+
|   Bridge implementations for connecting BIZRA systems with external services |
|   and between internal components.                                           |
|                                                                              |
|   Components:                                                                |
|   - sovereign_bridge: Main orchestration bridge                              |
|   - dual_agentic_bridge: Bicameral reasoning bridge                          |
|   - local_inference_bridge: LLM inference integration                        |
|   - swarm_knowledge_bridge: Agent-to-knowledge interface                     |
|   - iceoryx2_bridge: Zero-copy IPC (L2 Synapse)                              |
|   - rust_lifecycle: Python <-> Rust bridge                                   |
|                                                                              |
|   All bridges implement core.protocols.BridgeProtocol                        |
|                                                                              |
|   Standing on Giants: Gang of Four Bridge Pattern (1994)                     |
+==============================================================================+

Created: 2026-02-05 | SAPE Sovereign Module Decomposition
Migrated: 2026-02-05 | Files now in dedicated bridges package
"""

# Import protocol for type checking
from core.protocols.bridge import BridgeDirection, BridgeHealth, BridgeProtocol

# --------------------------------------------------------------------------
# PHASE 1: Safe imports (no cross-package dependencies)
# --------------------------------------------------------------------------
from .bridge import SovereignBridge
from .iceoryx2_bridge import Iceoryx2Bridge
from .knowledge_integrator import KnowledgeIntegrator
from .local_inference_bridge import LocalInferenceBridge
from .rust_lifecycle import (
    RustLifecycle,
    RustLifecycleManager,
    RustProcessManager,
)

# --------------------------------------------------------------------------
# PHASE 2: Lazy imports for modules with cross-package dependencies.
# dual_agentic_bridge and swarm_knowledge_bridge import from
# core.orchestration, which may not be fully initialized yet.
# --------------------------------------------------------------------------
_LAZY_MODULES = {
    "DualAgenticBridge": (".dual_agentic_bridge", "DualAgenticBridge"),
    "SwarmKnowledgeBridge": (".swarm_knowledge_bridge", "SwarmKnowledgeBridge"),
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        module_path, attr_name = _LAZY_MODULES[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        value = getattr(mod, attr_name)
        globals()[name] = value  # Cache for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Bridges
    "SovereignBridge",
    "DualAgenticBridge",
    "LocalInferenceBridge",
    "SwarmKnowledgeBridge",
    "Iceoryx2Bridge",
    # Rust Integration
    "RustLifecycle",
    "RustLifecycleManager",
    "RustProcessManager",
    # Knowledge
    "KnowledgeIntegrator",
    # Protocol (for type checking)
    "BridgeProtocol",
    "BridgeHealth",
    "BridgeDirection",
]
