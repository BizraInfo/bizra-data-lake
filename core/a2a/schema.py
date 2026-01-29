"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA A2A — PROTOCOL SCHEMA                                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Core data structures for Agent-to-Agent communication:                     ║
║   - AgentCard: Identity and capability manifest                              ║
║   - TaskCard: Structured task representation                                 ║
║   - A2AMessage: PCI-signed message envelope                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class MessageType(str, Enum):
    """A2A message types."""
    # Discovery
    DISCOVER = "discover"               # Request agent capabilities
    ANNOUNCE = "announce"               # Broadcast agent presence
    
    # Task lifecycle
    TASK_REQUEST = "task_request"       # Request task execution
    TASK_ACCEPT = "task_accept"         # Accept task
    TASK_REJECT = "task_reject"         # Reject task (with reason)
    TASK_STATUS = "task_status"         # Status update
    TASK_RESULT = "task_result"         # Final result
    TASK_CANCEL = "task_cancel"         # Cancel request
    
    # Streaming
    ARTIFACT_STREAM = "artifact_stream" # Streaming artifact chunk
    
    # Health
    PING = "ping"
    PONG = "pong"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CapabilityType(str, Enum):
    """Capability classification."""
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    CODE_EXECUTION = "code_execution"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    DATA_ANALYSIS = "data_analysis"
    VISION = "vision"
    AUDIO = "audio"
    ORCHESTRATION = "orchestration"
    SECURITY = "security"
    DESIGN = "design"
    FORMATTING = "formatting"
    CUSTOM = "custom"


# ═══════════════════════════════════════════════════════════════════════════════
# CAPABILITY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Capability:
    """
    Agent capability declaration.
    
    Follows capability-based security model:
    - name: Unique identifier (e.g., "code.python.execute")
    - type: Classification for routing
    - description: Human-readable description
    - parameters: JSON Schema for required inputs
    - ihsan_floor: Minimum Ihsān for invocation
    """
    name: str
    type: CapabilityType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    ihsan_floor: float = 0.95
    version: str = "1.0.0"
    
    def matches(self, query: str) -> bool:
        """Check if capability matches a query string."""
        query_lower = query.lower()
        return (
            query_lower in self.name.lower() or
            query_lower in self.description.lower()
        )
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Capability':
        d = d.copy()
        d["type"] = CapabilityType(d["type"])
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CARD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentCard:
    """
    Agent identity and capability manifest.
    
    Inspired by Google A2A spec, enhanced with:
    - PCI public key for verification
    - Ihsān score for trust
    - Federation address for P2P
    """
    # Identity
    agent_id: str
    name: str
    description: str
    version: str = "1.0.0"
    
    # Cryptographic identity (PCI)
    public_key: str = ""
    
    # Network
    endpoint: str = ""                  # HTTP/WebSocket endpoint
    federation_address: str = ""        # UDP gossip address
    
    # Capabilities
    capabilities: List[Capability] = field(default_factory=list)
    
    # Trust metrics
    ihsan_score: float = 0.95
    tasks_completed: int = 0
    success_rate: float = 1.0
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def has_capability(self, name: str) -> bool:
        """Check if agent has a specific capability."""
        return any(c.name == name for c in self.capabilities)
    
    def find_capabilities(self, query: str) -> List[Capability]:
        """Find capabilities matching a query."""
        return [c for c in self.capabilities if c.matches(query)]
    
    def get_capability(self, name: str) -> Optional[Capability]:
        """Get a specific capability by name."""
        for c in self.capabilities:
            if c.name == name:
                return c
        return None
    
    def card_hash(self) -> str:
        """Generate deterministic hash of agent card."""
        data = json.dumps({
            "agent_id": self.agent_id,
            "public_key": self.public_key,
            "capabilities": [c.name for c in self.capabilities],
            "version": self.version
        }, sort_keys=True)
        return hashlib.blake2b(data.encode(), digest_size=16).hexdigest()
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["capabilities"] = [c.to_dict() if isinstance(c, Capability) else c for c in self.capabilities]
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'AgentCard':
        d = d.copy()
        d["capabilities"] = [
            Capability.from_dict(c) if isinstance(c, dict) else c 
            for c in d.get("capabilities", [])
        ]
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK CARD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskCard:
    """
    Structured task representation for agent delegation.
    
    Features:
    - Unique task ID for tracking
    - Input/output schema
    - Parent task for hierarchical delegation
    - Artifacts for streaming results
    """
    # Identity
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Routing
    capability_required: str = ""       # Required capability name
    target_agent: str = ""              # Specific agent (optional)
    
    # Request
    prompt: str = ""                    # Natural language request
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Response
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    artifacts: List[Dict] = field(default_factory=list)
    
    # Hierarchy
    parent_task_id: Optional[str] = None
    child_task_ids: List[str] = field(default_factory=list)
    
    # Metadata
    requester_id: str = ""
    assignee_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Priority (1-10, 10 = highest)
    priority: int = 5
    
    # Timeout in seconds
    timeout: int = 300
    
    def add_artifact(self, name: str, content: Any, mime_type: str = "text/plain"):
        """Add an artifact to the task."""
        self.artifacts.append({
            "name": name,
            "content": content,
            "mime_type": mime_type,
            "created_at": datetime.now(timezone.utc).isoformat()
        })
    
    def mark_started(self, assignee_id: str):
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.assignee_id = assignee_id
        self.started_at = datetime.now(timezone.utc).isoformat()
    
    def mark_completed(self, result: Any):
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now(timezone.utc).isoformat()
    
    def mark_failed(self, error: str):
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TaskCard':
        d = d.copy()
        d["status"] = TaskStatus(d.get("status", "pending"))
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════════════════
# A2A MESSAGE ENVELOPE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class A2AMessage:
    """
    PCI-signed message envelope for A2A communication.
    
    Security:
    - All messages are signed with sender's Ed25519 key
    - Signature covers: type + sender + recipient + payload + timestamp
    - Ihsān score included for trust assessment
    """
    # Header
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.PING
    
    # Routing
    sender_id: str = ""
    sender_public_key: str = ""
    recipient_id: str = ""              # Empty = broadcast
    
    # Payload
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Security
    signature: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ihsan_score: float = 0.95
    
    # Protocol
    protocol_version: str = "1.0.0"
    
    def signing_content(self) -> bytes:
        """Get content to be signed."""
        content = {
            "type": self.message_type.value,
            "sender": self.sender_id,
            "recipient": self.recipient_id,
            "payload": self.payload,
            "timestamp": self.timestamp
        }
        return json.dumps(content, sort_keys=True).encode('utf-8')
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for transport."""
        return json.dumps(self.to_dict(), default=str).encode('utf-8')
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["message_type"] = self.message_type.value
        return d
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'A2AMessage':
        d = json.loads(data.decode('utf-8'))
        return cls.from_dict(d)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'A2AMessage':
        d = d.copy()
        d["message_type"] = MessageType(d.get("message_type", "ping"))
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_task_request(
    requester_id: str,
    capability: str,
    prompt: str,
    parameters: Optional[Dict] = None,
    target_agent: Optional[str] = None
) -> TaskCard:
    """Create a new task request."""
    return TaskCard(
        capability_required=capability,
        target_agent=target_agent or "",
        prompt=prompt,
        parameters=parameters or {},
        requester_id=requester_id
    )


def create_agent_card(
    agent_id: str,
    name: str,
    description: str,
    capabilities: List[Dict],
    public_key: str = "",
    endpoint: str = ""
) -> AgentCard:
    """Create an agent card from capability definitions."""
    caps = [
        Capability(
            name=c["name"],
            type=CapabilityType(c.get("type", "custom")),
            description=c.get("description", ""),
            parameters=c.get("parameters", {}),
            ihsan_floor=c.get("ihsan_floor", 0.95)
        )
        for c in capabilities
    ]
    return AgentCard(
        agent_id=agent_id,
        name=name,
        description=description,
        public_key=public_key,
        endpoint=endpoint,
        capabilities=caps
    )
