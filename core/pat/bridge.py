"""
PAT Bridge — Personal AI Team Integration Layer

Connects BIZRA to the PAT (Personal AI Team) ecosystem:
- WebSocket gateway communication
- Multi-channel message routing
- Skill/tool invocation
- Constitutional filtering

Standing on Giants: OpenClaw + A2A Protocol + Constitutional AI
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
)
from core.living_memory.core import LivingMemoryCore, MemoryType

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """PAT message types."""

    TEXT = "text"
    COMMAND = "command"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"
    ERROR = "error"


class ChannelType(str, Enum):
    """Supported communication channels."""

    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"
    SIGNAL = "signal"
    MATRIX = "matrix"
    WEBCHAT = "webchat"
    INTERNAL = "internal"


@dataclass
class PATMessage:
    """A message in the PAT system."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    channel: ChannelType = ChannelType.INTERNAL
    sender: str = "user"
    content: str = ""
    message_type: MessageType = MessageType.TEXT
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Metadata
    reply_to: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Constitutional
    ihsan_validated: bool = False
    ihsan_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "channel": self.channel.value,
            "sender": self.sender,
            "content": self.content,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "reply_to": self.reply_to,
            "attachments": self.attachments,
            "metadata": self.metadata,
            "ihsan_validated": self.ihsan_validated,
            "ihsan_score": self.ihsan_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PATMessage":
        msg = cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
        )
        if "channel" in data:
            msg.channel = ChannelType(data["channel"])
        if "sender" in data:
            msg.sender = data["sender"]
        if "message_type" in data:
            msg.message_type = MessageType(data["message_type"])
        if "timestamp" in data:
            msg.timestamp = datetime.fromisoformat(data["timestamp"])
        msg.reply_to = data.get("reply_to")
        msg.attachments = data.get("attachments", [])
        msg.metadata = data.get("metadata", {})
        return msg


@dataclass
class PATSession:
    """A conversation session with a channel."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    channel: ChannelType = ChannelType.INTERNAL
    user_id: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)


class PATBridge:
    """
    Bridge between BIZRA and PAT (Personal AI Team).

    Handles:
    - WebSocket connection to PAT gateway
    - Message routing across channels
    - Tool/skill invocation
    - Constitutional validation
    """

    def __init__(
        self,
        gateway_url: str = "ws://127.0.0.1:18789",
        memory: Optional[LivingMemoryCore] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
    ):
        self.gateway_url = gateway_url
        self.memory = memory
        self.llm_fn = llm_fn
        self.ihsan_threshold = ihsan_threshold

        # Connection state
        self._connected = False
        self._websocket = None

        # Sessions
        self._sessions: Dict[str, PATSession] = {}
        self._active_channel: Optional[ChannelType] = None

        # Message handling
        self._message_handlers: Dict[MessageType, List[Callable]] = {
            t: [] for t in MessageType
        }
        self._message_queue: asyncio.Queue = asyncio.Queue()

        # Tools/Skills
        self._tools: Dict[str, Callable] = {}

        # Metrics
        self._total_messages: int = 0
        self._ihsan_rejections: int = 0

    def register_tool(
        self,
        name: str,
        fn: Callable,
        description: str = "",
    ) -> None:
        """Register a tool for PAT to invoke."""
        self._tools[name] = fn
        logger.debug(f"Registered PAT tool: {name}")

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable,
    ) -> None:
        """Register a message handler."""
        self._message_handlers[message_type].append(handler)

    async def connect(self) -> bool:
        """Connect to PAT gateway."""
        try:
            # Try to import websockets (optional dependency)
            try:
                import websockets
            except ImportError:
                logger.warning(
                    "websockets not installed, PAT bridge running in simulation mode"
                )
                self._connected = False
                return False

            self._websocket = await websockets.connect(self.gateway_url)
            self._connected = True

            # Start message receiver
            asyncio.create_task(self._receive_loop())

            logger.info(f"Connected to PAT gateway: {self.gateway_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to PAT gateway: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from PAT gateway."""
        if self._websocket:
            await self._websocket.close()
        self._connected = False
        logger.info("Disconnected from PAT gateway")

    async def _receive_loop(self) -> None:
        """Background loop to receive messages."""
        while self._connected and self._websocket:
            try:
                raw = await self._websocket.recv()
                data = json.loads(raw)
                message = PATMessage.from_dict(data)

                await self._handle_incoming(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error receiving PAT message: {e}")

    async def _handle_incoming(self, message: PATMessage) -> None:
        """Handle an incoming message."""
        self._total_messages += 1

        # Update session
        session = self._get_or_create_session(message.channel, message.sender)
        session.last_activity = datetime.now(timezone.utc)
        session.message_count += 1

        # Constitutional validation
        if not await self._validate_ihsan(message):
            self._ihsan_rejections += 1
            await self._send_rejection(message)
            return

        message.ihsan_validated = True

        # Store in memory
        if self.memory:
            await self._store_message(message)

        # Route to handlers
        handlers = self._message_handlers.get(message.message_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Handler error: {e}")

        # Handle tool calls
        if message.message_type == MessageType.TOOL_CALL:
            await self._handle_tool_call(message)

    async def _validate_ihsan(self, message: PATMessage) -> bool:
        """Validate message against Ihsān constraints."""
        # Basic safety checks
        dangerous_patterns = [
            "delete all",
            "rm -rf",
            "format disk",
            "sudo rm",
        ]

        content_lower = message.content.lower()
        for pattern in dangerous_patterns:
            if pattern in content_lower:
                message.ihsan_score = 0.0
                return False

        # LLM-based validation if available
        if self.llm_fn:
            try:
                prompt = f"""Evaluate this message for safety:
Message: {message.content[:500]}

Is this message safe and appropriate? Answer YES or NO."""

                response = self.llm_fn(prompt)
                if "NO" in response.upper():
                    message.ihsan_score = 0.3
                    return False
            except Exception as e:
                logger.warning(f"Ihsan validation failed: {e}")
                # Fail open for simple messages
                if len(message.content) < 100:
                    return True
                return False

        message.ihsan_score = 0.95
        return True

    async def _send_rejection(self, original: PATMessage) -> None:
        """Send rejection message."""
        response = PATMessage(
            channel=original.channel,
            sender="bizra",
            content="I cannot process this request as it may violate safety guidelines.",
            message_type=MessageType.ERROR,
            reply_to=original.id,
        )
        await self.send(response)

    async def _handle_tool_call(self, message: PATMessage) -> None:
        """Handle a tool call message."""
        tool_name = message.metadata.get("tool")
        tool_input = message.metadata.get("input", {})

        if tool_name not in self._tools:
            response = PATMessage(
                channel=message.channel,
                sender="bizra",
                content=f"Unknown tool: {tool_name}",
                message_type=MessageType.ERROR,
                reply_to=message.id,
            )
            await self.send(response)
            return

        try:
            tool_fn = self._tools[tool_name]
            if asyncio.iscoroutinefunction(tool_fn):
                result = await tool_fn(**tool_input)
            else:
                result = tool_fn(**tool_input)

            response = PATMessage(
                channel=message.channel,
                sender="bizra",
                content=str(result),
                message_type=MessageType.TOOL_RESULT,
                reply_to=message.id,
                metadata={"tool": tool_name, "success": True},
            )
            await self.send(response)

        except Exception as e:
            response = PATMessage(
                channel=message.channel,
                sender="bizra",
                content=f"Tool error: {e}",
                message_type=MessageType.ERROR,
                reply_to=message.id,
                metadata={"tool": tool_name, "success": False},
            )
            await self.send(response)

    async def _store_message(self, message: PATMessage) -> None:
        """Store message in living memory."""
        content = f"[{message.channel.value}] {message.sender}: {message.content}"
        await self.memory.encode(
            content=content,
            memory_type=MemoryType.EPISODIC,
            source=f"pat:{message.channel.value}",
            importance=0.6,
        )

    def _get_or_create_session(
        self,
        channel: ChannelType,
        user_id: str,
    ) -> PATSession:
        """Get or create a session."""
        session_key = f"{channel.value}:{user_id}"
        if session_key not in self._sessions:
            self._sessions[session_key] = PATSession(
                channel=channel,
                user_id=user_id,
            )
        return self._sessions[session_key]

    async def send(self, message: PATMessage) -> bool:
        """Send a message through PAT."""
        if not self._connected or not self._websocket:
            logger.warning("Cannot send: not connected to PAT gateway")
            return False

        try:
            await self._websocket.send(json.dumps(message.to_dict()))
            return True
        except Exception as e:
            logger.error(f"Failed to send PAT message: {e}")
            return False

    async def process_local(
        self,
        content: str,
        channel: ChannelType = ChannelType.INTERNAL,
        sender: str = "user",
    ) -> Optional[str]:
        """
        Process a message locally (without gateway).

        Useful for testing or when gateway is unavailable.
        """
        message = PATMessage(
            channel=channel,
            sender=sender,
            content=content,
            message_type=MessageType.TEXT,
        )

        # Validate
        if not await self._validate_ihsan(message):
            return "Message blocked by safety filter."

        # Store
        if self.memory:
            await self._store_message(message)

        # Generate response with LLM
        if self.llm_fn:
            try:
                # Get context from memory
                context = ""
                if self.memory:
                    memories = await self.memory.retrieve(
                        query=content,
                        top_k=5,
                    )
                    context = "\n".join(m.content for m in memories)

                prompt = f"""You are BIZRA, a helpful AI assistant.

Context:
{context[:1000]}

User: {content}

Respond helpfully and concisely."""

                response = self.llm_fn(prompt)

                # Store response
                if self.memory:
                    await self.memory.encode(
                        content=f"[bizra response] {response[:500]}",
                        memory_type=MemoryType.EPISODIC,
                        source="pat:internal",
                    )

                return response

            except Exception as e:
                return f"Error generating response: {e}"

        return "LLM not available for response generation."

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            "connected": self._connected,
            "gateway_url": self.gateway_url,
            "active_sessions": len(self._sessions),
            "total_messages": self._total_messages,
            "ihsan_rejections": self._ihsan_rejections,
            "registered_tools": list(self._tools.keys()),
        }
