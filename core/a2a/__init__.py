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

from .schema import (
    AgentCard,
    TaskCard,
    TaskStatus,
    Capability,
    A2AMessage,
    MessageType,
)
from .engine import A2AEngine
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
