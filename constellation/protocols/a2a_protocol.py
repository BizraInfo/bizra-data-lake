# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Constellation - Agent-to-Agent (A2A) Protocol v1.0
# ═══════════════════════════════════════════════════════════════════════════════
"""
Inter-agent communication protocol for:
- Direct agent-to-agent messaging
- Delegation and handoff
- Collaborative reasoning
- Knowledge sharing
- Dispute resolution
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any, Callable, Awaitable
from enum import Enum
from collections import defaultdict


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MESSAGE TYPES
# ─────────────────────────────────────────────────────────────────────────────

class MessageType(str, Enum):
    """Types of A2A messages."""
    # Basic communication
    QUERY = "query"              # Ask a question
    RESPONSE = "response"        # Answer to query
    INFORM = "inform"            # Share information
    ACKNOWLEDGE = "acknowledge"  # Confirm receipt
    
    # Delegation
    DELEGATE = "delegate"        # Delegate a task
    ACCEPT = "accept"           # Accept delegation
    REFUSE = "refuse"           # Refuse delegation
    COMPLETE = "complete"        # Report task completion
    
    # Collaboration
    PROPOSE = "propose"          # Propose an idea/solution
    COUNTER = "counter"          # Counter-proposal
    AGREE = "agree"             # Agreement
    DISAGREE = "disagree"        # Disagreement with reasoning
    
    # Knowledge
    SHARE = "share"             # Share knowledge
    REQUEST = "request"          # Request knowledge
    VERIFY = "verify"           # Request verification
    CONFIRM = "confirm"          # Confirm as verified
    CHALLENGE = "challenge"      # Challenge a claim
    
    # System
    PING = "ping"               # Health check
    PONG = "pong"               # Health response
    ERROR = "error"             # Error notification


class MessagePriority(str, Enum):
    """Priority levels for messages."""
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class DeliveryStatus(str, Enum):
    """Status of message delivery."""
    PENDING = "pending"
    DELIVERED = "delivered"
    READ = "read"
    REPLIED = "replied"
    FAILED = "failed"
    EXPIRED = "expired"


# ─────────────────────────────────────────────────────────────────────────────
# MESSAGE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class A2AMessage:
    """An agent-to-agent message."""
    id: str
    type: MessageType
    sender: str  # Agent slug
    recipient: str  # Agent slug or "broadcast"
    
    # Content
    subject: str
    content: dict
    
    # Metadata
    priority: MessagePriority = MessagePriority.NORMAL
    session_id: Optional[str] = None
    thread_id: Optional[str] = None  # For conversations
    in_reply_to: Optional[str] = None  # Message ID
    
    # Timing
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: Optional[str] = None
    
    # Delivery
    status: DeliveryStatus = DeliveryStatus.PENDING
    delivered_at: Optional[str] = None
    read_at: Optional[str] = None
    
    # Evidence
    claims: list[dict] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    snr_score: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "subject": self.subject,
            "content": self.content,
            "priority": self.priority.value,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "in_reply_to": self.in_reply_to,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status.value,
            "delivered_at": self.delivered_at,
            "claims": self.claims,
            "evidence_refs": self.evidence_refs,
            "snr_score": self.snr_score,
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> "A2AMessage":
        return cls(
            id=data["id"],
            type=MessageType(data["type"]),
            sender=data["sender"],
            recipient=data["recipient"],
            subject=data["subject"],
            content=data["content"],
            priority=MessagePriority(data.get("priority", "normal")),
            session_id=data.get("session_id"),
            thread_id=data.get("thread_id"),
            in_reply_to=data.get("in_reply_to"),
            created_at=data.get("created_at", ""),
            expires_at=data.get("expires_at"),
            status=DeliveryStatus(data.get("status", "pending")),
            delivered_at=data.get("delivered_at"),
            read_at=data.get("read_at"),
            claims=data.get("claims", []),
            evidence_refs=data.get("evidence_refs", []),
            snr_score=data.get("snr_score"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# MESSAGE HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

MessageHandler = Callable[[A2AMessage], Awaitable[Optional[A2AMessage]]]


@dataclass
class AgentMailbox:
    """Mailbox for an agent to receive messages."""
    agent_slug: str
    inbox: asyncio.Queue = field(default_factory=asyncio.Queue)
    handlers: dict[MessageType, list[MessageHandler]] = field(default_factory=lambda: defaultdict(list))
    unread_count: int = 0
    
    async def receive(self, timeout: Optional[float] = None) -> Optional[A2AMessage]:
        """Receive next message from inbox."""
        try:
            if timeout:
                msg = await asyncio.wait_for(self.inbox.get(), timeout)
            else:
                msg = await self.inbox.get()
            msg.status = DeliveryStatus.READ
            msg.read_at = datetime.now(timezone.utc).isoformat()
            self.unread_count = max(0, self.unread_count - 1)
            return msg
        except asyncio.TimeoutError:
            return None
            
    def register_handler(
        self,
        msg_type: MessageType,
        handler: MessageHandler,
    ) -> None:
        """Register a handler for a message type."""
        self.handlers[msg_type].append(handler)
        
    async def process_message(self, message: A2AMessage) -> list[A2AMessage]:
        """Process a message through registered handlers."""
        responses = []
        
        for handler in self.handlers[message.type]:
            try:
                response = await handler(message)
                if response:
                    responses.append(response)
            except Exception as e:
                logger.error(f"Handler error for {message.type}: {e}")
                
        return responses


# ─────────────────────────────────────────────────────────────────────────────
# A2A ROUTER
# ─────────────────────────────────────────────────────────────────────────────

class A2ARouter:
    """
    Routes messages between agents.
    
    Provides:
    - Message delivery
    - Broadcast support
    - Priority queuing
    - Message persistence
    - Delivery tracking
    """
    
    def __init__(self):
        self._mailboxes: dict[str, AgentMailbox] = {}
        self._message_log: list[A2AMessage] = []
        self._threads: dict[str, list[str]] = defaultdict(list)  # thread_id -> message_ids
        
    def register_agent(self, agent_slug: str) -> AgentMailbox:
        """Register an agent for A2A communication."""
        if agent_slug not in self._mailboxes:
            self._mailboxes[agent_slug] = AgentMailbox(agent_slug=agent_slug)
            logger.debug(f"Registered A2A mailbox for: {agent_slug}")
        return self._mailboxes[agent_slug]
        
    def unregister_agent(self, agent_slug: str) -> bool:
        """Unregister an agent."""
        if agent_slug in self._mailboxes:
            del self._mailboxes[agent_slug]
            return True
        return False
        
    async def send(self, message: A2AMessage) -> DeliveryStatus:
        """Send a message to recipient(s)."""
        # Log message
        self._message_log.append(message)
        
        # Track thread
        if message.thread_id:
            self._threads[message.thread_id].append(message.id)
            
        # Handle broadcast
        if message.recipient == "broadcast":
            return await self._broadcast(message)
            
        # Direct delivery
        return await self._deliver(message)
        
    async def _deliver(self, message: A2AMessage) -> DeliveryStatus:
        """Deliver message to specific recipient."""
        mailbox = self._mailboxes.get(message.recipient)
        
        if not mailbox:
            message.status = DeliveryStatus.FAILED
            logger.warning(f"No mailbox for: {message.recipient}")
            return DeliveryStatus.FAILED
            
        # Add to inbox
        await mailbox.inbox.put(message)
        mailbox.unread_count += 1
        
        message.status = DeliveryStatus.DELIVERED
        message.delivered_at = datetime.now(timezone.utc).isoformat()
        
        logger.debug(f"Delivered {message.type.value} from {message.sender} to {message.recipient}")
        return DeliveryStatus.DELIVERED
        
    async def _broadcast(self, message: A2AMessage) -> DeliveryStatus:
        """Broadcast message to all agents except sender."""
        delivered = 0
        
        for agent_slug, mailbox in self._mailboxes.items():
            if agent_slug != message.sender:
                # Create copy for each recipient
                msg_copy = A2AMessage(
                    id=f"{message.id}_{agent_slug}",
                    type=message.type,
                    sender=message.sender,
                    recipient=agent_slug,
                    subject=message.subject,
                    content=message.content,
                    priority=message.priority,
                    session_id=message.session_id,
                    thread_id=message.thread_id,
                    created_at=message.created_at,
                    claims=message.claims,
                    evidence_refs=message.evidence_refs,
                    snr_score=message.snr_score,
                )
                
                await mailbox.inbox.put(msg_copy)
                mailbox.unread_count += 1
                delivered += 1
                
        message.status = DeliveryStatus.DELIVERED
        message.delivered_at = datetime.now(timezone.utc).isoformat()
        
        logger.debug(f"Broadcast from {message.sender} delivered to {delivered} agents")
        return DeliveryStatus.DELIVERED
        
    def get_mailbox(self, agent_slug: str) -> Optional[AgentMailbox]:
        """Get an agent's mailbox."""
        return self._mailboxes.get(agent_slug)
        
    def get_thread(self, thread_id: str) -> list[A2AMessage]:
        """Get all messages in a thread."""
        message_ids = self._threads.get(thread_id, [])
        return [m for m in self._message_log if m.id in message_ids]
        
    def get_conversation(
        self,
        agent1: str,
        agent2: str,
        limit: int = 50,
    ) -> list[A2AMessage]:
        """Get conversation between two agents."""
        messages = [
            m for m in self._message_log
            if (m.sender == agent1 and m.recipient == agent2) or
               (m.sender == agent2 and m.recipient == agent1)
        ]
        return sorted(messages, key=lambda m: m.created_at)[-limit:]


# ─────────────────────────────────────────────────────────────────────────────
# MESSAGE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class MessageBuilder:
    """Fluent builder for A2A messages."""
    
    def __init__(self, sender: str):
        self.sender = sender
        self._type: MessageType = MessageType.INFORM
        self._recipient: str = "broadcast"
        self._subject: str = ""
        self._content: dict = {}
        self._priority: MessagePriority = MessagePriority.NORMAL
        self._session_id: Optional[str] = None
        self._thread_id: Optional[str] = None
        self._in_reply_to: Optional[str] = None
        self._claims: list[dict] = []
        self._evidence_refs: list[str] = []
        self._snr_score: Optional[float] = None
        
    def query(self, question: str) -> "MessageBuilder":
        """Create a query message."""
        self._type = MessageType.QUERY
        self._content["question"] = question
        return self
        
    def respond(self, answer: str) -> "MessageBuilder":
        """Create a response message."""
        self._type = MessageType.RESPONSE
        self._content["answer"] = answer
        return self
        
    def inform(self, info: str) -> "MessageBuilder":
        """Create an inform message."""
        self._type = MessageType.INFORM
        self._content["information"] = info
        return self
        
    def delegate(self, task: str, context: Optional[dict] = None) -> "MessageBuilder":
        """Create a delegation message."""
        self._type = MessageType.DELEGATE
        self._content["task"] = task
        self._content["context"] = context or {}
        return self
        
    def share(self, knowledge: dict) -> "MessageBuilder":
        """Create a knowledge sharing message."""
        self._type = MessageType.SHARE
        self._content["knowledge"] = knowledge
        return self
        
    def propose(self, proposal: str, details: Optional[dict] = None) -> "MessageBuilder":
        """Create a proposal message."""
        self._type = MessageType.PROPOSE
        self._content["proposal"] = proposal
        self._content["details"] = details or {}
        return self
        
    def challenge(self, claim: str, reason: str) -> "MessageBuilder":
        """Create a challenge message."""
        self._type = MessageType.CHALLENGE
        self._content["challenged_claim"] = claim
        self._content["reason"] = reason
        return self
        
    def to(self, recipient: str) -> "MessageBuilder":
        """Set recipient."""
        self._recipient = recipient
        return self
        
    def broadcast(self) -> "MessageBuilder":
        """Set as broadcast."""
        self._recipient = "broadcast"
        return self
        
    def subject(self, subject: str) -> "MessageBuilder":
        """Set subject."""
        self._subject = subject
        return self
        
    def priority(self, priority: MessagePriority) -> "MessageBuilder":
        """Set priority."""
        self._priority = priority
        return self
        
    def urgent(self) -> "MessageBuilder":
        """Set as urgent priority."""
        self._priority = MessagePriority.URGENT
        return self
        
    def in_thread(self, thread_id: str) -> "MessageBuilder":
        """Set thread ID."""
        self._thread_id = thread_id
        return self
        
    def replying_to(self, message_id: str) -> "MessageBuilder":
        """Set as reply to another message."""
        self._in_reply_to = message_id
        return self
        
    def with_claims(self, claims: list[dict]) -> "MessageBuilder":
        """Add claims."""
        self._claims = claims
        return self
        
    def with_evidence(self, refs: list[str]) -> "MessageBuilder":
        """Add evidence references."""
        self._evidence_refs = refs
        return self
        
    def with_snr(self, score: float) -> "MessageBuilder":
        """Set SNR score."""
        self._snr_score = score
        return self
        
    def in_session(self, session_id: str) -> "MessageBuilder":
        """Set session ID."""
        self._session_id = session_id
        return self
        
    def build(self) -> A2AMessage:
        """Build the message."""
        return A2AMessage(
            id=f"msg_{uuid.uuid4().hex[:12]}",
            type=self._type,
            sender=self.sender,
            recipient=self._recipient,
            subject=self._subject,
            content=self._content,
            priority=self._priority,
            session_id=self._session_id,
            thread_id=self._thread_id,
            in_reply_to=self._in_reply_to,
            claims=self._claims,
            evidence_refs=self._evidence_refs,
            snr_score=self._snr_score,
        )


# ─────────────────────────────────────────────────────────────────────────────
# A2A PROTOCOL INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class A2AProtocol:
    """
    High-level A2A protocol interface for agents.
    
    Provides convenient methods for common communication patterns.
    """
    
    def __init__(self, agent_slug: str, router: A2ARouter):
        self.agent_slug = agent_slug
        self.router = router
        self.mailbox = router.register_agent(agent_slug)
        
    def message(self) -> MessageBuilder:
        """Start building a new message."""
        return MessageBuilder(self.agent_slug)
        
    async def send(self, message: A2AMessage) -> DeliveryStatus:
        """Send a message."""
        return await self.router.send(message)
        
    async def query(
        self,
        recipient: str,
        question: str,
        wait_for_response: bool = True,
        timeout: float = 30.0,
    ) -> Optional[A2AMessage]:
        """Send a query and optionally wait for response."""
        msg = self.message().query(question).to(recipient).build()
        await self.send(msg)
        
        if wait_for_response:
            # Wait for response
            start = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start < timeout:
                response = await self.mailbox.receive(timeout=1.0)
                if response and response.in_reply_to == msg.id:
                    return response
                elif response:
                    # Put back if not our response
                    await self.mailbox.inbox.put(response)
                    
        return None
        
    async def delegate(
        self,
        recipient: str,
        task: str,
        context: Optional[dict] = None,
    ) -> DeliveryStatus:
        """Delegate a task to another agent."""
        msg = self.message().delegate(task, context).to(recipient).build()
        return await self.send(msg)
        
    async def share_knowledge(
        self,
        knowledge: dict,
        recipients: Optional[list[str]] = None,
    ) -> None:
        """Share knowledge with other agents."""
        if recipients:
            for recipient in recipients:
                msg = self.message().share(knowledge).to(recipient).build()
                await self.send(msg)
        else:
            msg = self.message().share(knowledge).broadcast().build()
            await self.send(msg)
            
    async def propose_solution(
        self,
        proposal: str,
        details: dict,
        recipients: Optional[list[str]] = None,
    ) -> DeliveryStatus:
        """Propose a solution to other agents."""
        msg = self.message().propose(proposal, details)
        if recipients:
            # Send to first, could extend to multi-send
            msg = msg.to(recipients[0])
        else:
            msg = msg.broadcast()
        return await self.send(msg.build())
        
    async def challenge_claim(
        self,
        recipient: str,
        claim: str,
        reason: str,
    ) -> DeliveryStatus:
        """Challenge another agent's claim."""
        msg = self.message().challenge(claim, reason).to(recipient).urgent().build()
        return await self.send(msg)
        
    async def receive(self, timeout: Optional[float] = None) -> Optional[A2AMessage]:
        """Receive next message."""
        return await self.mailbox.receive(timeout)
        
    def register_handler(
        self,
        msg_type: MessageType,
        handler: MessageHandler,
    ) -> None:
        """Register a message handler."""
        self.mailbox.register_handler(msg_type, handler)
        
    def get_unread_count(self) -> int:
        """Get count of unread messages."""
        return self.mailbox.unread_count


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

_router: Optional[A2ARouter] = None


def get_a2a_router() -> A2ARouter:
    """Get the global A2A router."""
    global _router
    if _router is None:
        _router = A2ARouter()
    return _router


def get_a2a_protocol(agent_slug: str) -> A2AProtocol:
    """Get A2A protocol interface for an agent."""
    return A2AProtocol(agent_slug, get_a2a_router())
