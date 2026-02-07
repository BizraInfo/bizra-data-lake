"""Re-export from canonical location: core.bridges.bridge"""

# Backwards compatibility — canonical implementation is in core/bridges/
from core.bridges.bridge import *  # noqa: F401,F403
from core.bridges.bridge import (
    A2AConnector,
    BridgeMessage,
    FederationConnector,
    InferenceConnector,
    InferenceRequest,
    InferenceResponse,
    InferenceTier,
    MemoryConnector,
    MessagePriority,
    SovereignBridge,
    SubsystemConnector,
    SubsystemHealth,
    SubsystemStatus,
    create_bridge,
)
