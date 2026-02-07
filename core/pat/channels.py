"""
BIZRA Channel Adapter Protocol — Abstract Gateway Interface

Every messaging channel (Telegram, WhatsApp, Signal, etc.) implements
this protocol. The adapter translates platform-specific messages into
PATMessage objects and routes them through the sovereign runtime.

Article IX § 9.6: "The Gateway is open-source — any community can
build, operate, or customize one."

Standing on Giants: Adapter Pattern (GoF) + Hexagonal Architecture (Cockburn)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Optional

from .bridge import ChannelType, PATSession

logger = logging.getLogger(__name__)


class GatewayState(str, Enum):
    """Gateway lifecycle states."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class GatewayMetrics:
    """Metrics collected by the gateway."""

    messages_received: int = 0
    messages_sent: int = 0
    messages_rejected: int = 0
    errors: int = 0
    unique_users: int = 0
    started_at: Optional[str] = None
    uptime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "messages_rejected": self.messages_rejected,
            "errors": self.errors,
            "unique_users": self.unique_users,
            "started_at": self.started_at,
            "uptime_seconds": self.uptime_seconds,
        }


# Type for the query function: takes content string, returns response string
QueryFn = Callable[[str, Optional[Dict[str, Any]]], Coroutine[Any, Any, str]]


class ChannelAdapter(ABC):
    """
    Abstract base for messaging channel adapters.

    Each adapter:
    1. Connects to the platform (Telegram, WhatsApp, etc.)
    2. Receives messages and translates to PATMessage
    3. Routes through the query function (SovereignRuntime.query)
    4. Sends response back through the platform

    Subclasses implement:
    - start() — Connect and begin polling/webhook
    - stop() — Graceful shutdown
    - send_response() — Send message back to user

    The adapter does NOT implement query logic — it delegates to
    the query_fn provided at construction.
    """

    def __init__(
        self,
        channel_type: ChannelType,
        query_fn: QueryFn,
        node_id: str = "",
    ):
        self._channel_type = channel_type
        self._query_fn = query_fn
        self._node_id = node_id
        self._state = GatewayState.STOPPED
        self._metrics = GatewayMetrics()
        self._sessions: Dict[str, PATSession] = {}
        self._seen_users: set = set()

    @property
    def channel_type(self) -> ChannelType:
        return self._channel_type

    @property
    def state(self) -> GatewayState:
        return self._state

    @property
    def metrics(self) -> GatewayMetrics:
        return self._metrics

    @abstractmethod
    async def start(self) -> None:
        """Start the adapter (connect to platform, begin receiving)."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the adapter gracefully."""
        ...

    @abstractmethod
    async def send_response(
        self, platform_user_id: str, content: str, **kwargs: Any
    ) -> bool:
        """Send a response back to the user on the platform."""
        ...

    async def handle_incoming(
        self,
        platform_user_id: str,
        content: str,
        platform_message_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Process an incoming message from the platform.

        This is the core routing logic — translates platform message
        to PATMessage, queries the sovereign runtime, and returns
        the response text.

        Args:
            platform_user_id: User ID on the platform (e.g. Telegram chat_id)
            content: Message text
            platform_message_id: Platform-specific message ID
            metadata: Optional platform-specific metadata

        Returns:
            Response text from the sovereign runtime
        """
        self._metrics.messages_received += 1

        if platform_user_id not in self._seen_users:
            self._seen_users.add(platform_user_id)
            self._metrics.unique_users = len(self._seen_users)

        # Track session
        session = self._get_or_create_session(platform_user_id)
        session.message_count += 1
        session.last_activity = datetime.now(timezone.utc)

        # Build context for the query
        context = {
            "channel": self._channel_type.value,
            "user_id": platform_user_id,
            "session_id": session.id,
            "message_count": session.message_count,
        }
        if metadata:
            context["platform"] = metadata

        try:
            response = await self._query_fn(content, context)
            self._metrics.messages_sent += 1
            return response
        except Exception as e:
            self._metrics.errors += 1
            logger.error(f"Query error for {platform_user_id}: {e}")
            return "I encountered an error processing your request. Please try again."

    def _get_or_create_session(self, platform_user_id: str) -> PATSession:
        """Get or create a session for this user."""
        if platform_user_id not in self._sessions:
            self._sessions[platform_user_id] = PATSession(
                channel=self._channel_type,
                user_id=platform_user_id,
            )
        return self._sessions[platform_user_id]

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        return {
            "channel": self._channel_type.value,
            "state": self._state.value,
            "node_id": self._node_id,
            "metrics": self._metrics.to_dict(),
            "active_sessions": len(self._sessions),
        }
