"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA A2A — AGENT-TO-AGENT PROTOCOL                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Sovereign agent communication with:                                        ║
║   - PCI-signed message envelopes                                             ║
║   - Capability-based task routing                                            ║
║   - Ihsān integrity verification                                             ║
║   - Federation P2P transport                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from .engine import A2AEngine
from .schema import (
    A2AMessage,
    AgentCard,
    Capability,
    MessageType,
    TaskCard,
    TaskStatus,
)
from .tasks import TaskManager
from .transport import A2ATransport

__all__ = [
    "AgentCard",
    "TaskCard",
    "TaskStatus",
    "Capability",
    "A2AMessage",
    "MessageType",
    "A2AEngine",
    "TaskManager",
    "A2ATransport",
]

__version__ = "1.0.0"
