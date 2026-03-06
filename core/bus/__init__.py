"""
BIZRA Bus Architecture — Phase 68 Nervous System
═════════════════════════════════════════════════

CQRS command pipeline with constitutional gates.
ActionBus (commands) complements EventBus (facts).

Standing on Giants:
- Fowler (2005): CQRS pattern
- Hewitt (1973): Actor model
- Thompson (1984): Capability-based security
"""

from core.bus.action_bus import ActionBus
from core.bus.capsule import (
    CapsuleManifest,
    CapsuleRegistry,
    CapsuleResult,
    CapsuleRuntime,
)
from core.bus.channels import ChannelExecutor, ChannelResult
from core.bus.omega import OmegaLoopController, OmegaStatus
from core.bus.telescript import TeleScriptEngine, TeleScriptVerdict
from core.bus.topics import TopicRegistry, TopicTier
from core.bus.sovereign_wiring import BusWiringState, wire_all
from core.bus.types import ActionBudget, ActionEnvelope, ActionStatus, BusActionReceipt

__all__ = [
    "ActionBudget",
    "ActionBus",
    "ActionEnvelope",
    "ActionStatus",
    "BusActionReceipt",
    "CapsuleManifest",
    "CapsuleRegistry",
    "CapsuleResult",
    "CapsuleRuntime",
    "ChannelExecutor",
    "ChannelResult",
    "OmegaLoopController",
    "OmegaStatus",
    "TeleScriptEngine",
    "TeleScriptVerdict",
    "TopicRegistry",
    "TopicTier",
    "BusWiringState",
    "wire_all",
]
