#!/usr/bin/env python3
"""
BIZRA Trinity Synapse - Agent-to-Agent Communication Layer
============================================================
Fixes F-ARCH-001: Trinity Disconnect

Provides shared runtime memory and messaging between agents via Redis:
- Pub/Sub channels for real-time agent communication
- Shared state store for coordination
- Event sourcing for audit trail
- Presence detection for agent discovery

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          TRINITY SYNAPSE                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
    │   │  PAT Agent  │     │  PAT Agent  │     │  SAT Agent  │               │
    │   │  (Reasoner) │     │  (Memory)   │     │  (Evidence) │               │
    │   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘               │
    │          │                   │                   │                       │
    │          └───────────────────┼───────────────────┘                       │
    │                              │                                           │
    │                              ▼                                           │
    │   ┌──────────────────────────────────────────────────────────────────┐  │
    │   │                      SYNAPSE BUS                                  │  │
    │   │                                                                   │  │
    │   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │  │
    │   │   │   CHANNELS   │  │    STATE     │  │   EVENTS     │           │  │
    │   │   │              │  │              │  │              │           │  │
    │   │   │ agent:*      │  │ state:*      │  │ events:*     │           │  │
    │   │   │ broadcast    │  │ locks:*      │  │ (Stream)     │           │  │
    │   │   │ direct:{id}  │  │ presence:*   │  │              │           │  │
    │   │   │              │  │              │  │              │           │  │
    │   │   └──────────────┘  └──────────────┘  └──────────────┘           │  │
    │   │                                                                   │  │
    │   └──────────────────────────────────────────────────────────────────┘  │
    │                              │                                           │
    │                              ▼                                           │
    │   ┌──────────────────────────────────────────────────────────────────┐  │
    │   │                      REDIS (synapse)                              │  │
    │   │                      redis://synapse:6379                         │  │
    │   └──────────────────────────────────────────────────────────────────┘  │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘

Channels:
    - bizra:broadcast        → All agents (system announcements)
    - bizra:agent:{id}       → Direct to specific agent
    - bizra:team:pat         → All PAT agents
    - bizra:team:sat         → All SAT agents
    - bizra:task:{task_id}   → Task-specific coordination
"""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import uuid4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("synapse")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SYNAPSE_URL = os.getenv("SYNAPSE_URL", "redis://127.0.0.1:6379")
SYNAPSE_PREFIX = os.getenv("SYNAPSE_PREFIX", "bizra")
PRESENCE_TTL = int(os.getenv("SYNAPSE_PRESENCE_TTL", "30"))  # seconds
EVENT_STREAM_MAXLEN = int(os.getenv("SYNAPSE_EVENT_MAXLEN", "10000"))


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class MessageType(Enum):
    """Types of synapse messages."""
    # Agent lifecycle
    AGENT_ONLINE = "agent.online"
    AGENT_OFFLINE = "agent.offline"
    AGENT_HEARTBEAT = "agent.heartbeat"
    
    # Task coordination
    TASK_ASSIGNED = "task.assigned"
    TASK_ACCEPTED = "task.accepted"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_DELEGATED = "task.delegated"
    
    # Knowledge sharing
    KNOWLEDGE_QUERY = "knowledge.query"
    KNOWLEDGE_RESPONSE = "knowledge.response"
    KNOWLEDGE_UPDATE = "knowledge.update"
    
    # Coordination
    CONSENSUS_REQUEST = "consensus.request"
    CONSENSUS_VOTE = "consensus.vote"
    CONSENSUS_RESULT = "consensus.result"
    
    # System
    SYSTEM_ANNOUNCEMENT = "system.announcement"
    SYSTEM_SHUTDOWN = "system.shutdown"
    
    # Direct communication
    DIRECT_MESSAGE = "direct.message"
    DIRECT_REQUEST = "direct.request"
    DIRECT_RESPONSE = "direct.response"


@dataclass
class SynapseMessage:
    """A message transmitted through the synapse."""
    id: str
    type: MessageType
    sender_id: str
    sender_name: str
    payload: Dict[str, Any]
    timestamp: str
    correlation_id: Optional[str] = None  # For request/response pairing
    ttl_ms: Optional[int] = None  # Message expiry
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "ttl_ms": self.ttl_ms
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SynapseMessage':
        return cls(
            id=data["id"],
            type=MessageType(data["type"]),
            sender_id=data["sender_id"],
            sender_name=data["sender_name"],
            payload=data.get("payload", {}),
            timestamp=data["timestamp"],
            correlation_id=data.get("correlation_id"),
            ttl_ms=data.get("ttl_ms")
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SynapseMessage':
        return cls.from_dict(json.loads(json_str))


@dataclass
class AgentPresence:
    """Agent presence information."""
    agent_id: str
    agent_name: str
    agent_type: str  # "PAT" or "SAT"
    status: str  # "online", "busy", "away"
    capabilities: List[str]
    last_seen: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "status": self.status,
            "capabilities": self.capabilities,
            "last_seen": self.last_seen,
            "metadata": self.metadata
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SYNAPSE CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════

class SynapseConnection:
    """
    Redis connection manager for the Synapse.
    Handles connection pooling and reconnection.
    """
    
    _instance: Optional['SynapseConnection'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'SynapseConnection':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._redis = None
        self._pubsub = None
        self._connected = False
        self._url = SYNAPSE_URL
        
        self._initialized = True
    
    def connect(self) -> bool:
        """Establish Redis connection."""
        if self._connected:
            return True
        
        try:
            import redis
            self._redis = redis.from_url(
                self._url,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
            # Test connection
            self._redis.ping()
            self._connected = True
            logger.info(f"Synapse connected: {self._url}")
            return True
        except ImportError:
            logger.error("redis package not installed. Run: pip install redis")
            return False
        except Exception as e:
            logger.error(f"Synapse connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close Redis connection."""
        if self._pubsub:
            self._pubsub.close()
        if self._redis:
            self._redis.close()
        self._connected = False
        logger.info("Synapse disconnected")
    
    @property
    def redis(self):
        """Get Redis client (auto-connect)."""
        if not self._connected:
            self.connect()
        return self._redis
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def health_check(self) -> Dict[str, Any]:
        """Check synapse health."""
        try:
            if not self._connected:
                return {"status": "disconnected", "url": self._url}
            
            start = time.monotonic()
            self._redis.ping()
            latency_ms = (time.monotonic() - start) * 1000
            
            info = self._redis.info("server")
            return {
                "status": "healthy",
                "url": self._url,
                "latency_ms": round(latency_ms, 2),
                "redis_version": info.get("redis_version"),
                "uptime_seconds": info.get("uptime_in_seconds")
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


def get_synapse() -> SynapseConnection:
    """Get the global synapse connection."""
    return SynapseConnection()


# ═══════════════════════════════════════════════════════════════════════════════
# SYNAPSE BUS
# ═══════════════════════════════════════════════════════════════════════════════

MessageHandler = Callable[[SynapseMessage], None]


class SynapseBus:
    """
    Message bus for agent communication.
    
    Features:
    - Pub/Sub messaging
    - Shared state storage
    - Presence tracking
    - Event sourcing
    """
    
    def __init__(self, agent_id: str, agent_name: str, agent_type: str = "PAT"):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.conn = get_synapse()
        
        self._handlers: Dict[MessageType, List[MessageHandler]] = {}
        self._subscriptions: Set[str] = set()
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False
        self._pubsub = None
    
    def _key(self, *parts: str) -> str:
        """Build namespaced Redis key."""
        return f"{SYNAPSE_PREFIX}:{':'.join(parts)}"
    
    def _channel(self, *parts: str) -> str:
        """Build channel name."""
        return self._key("channel", *parts)
    
    # ─────────────────────────────────────────────────────────────────────────
    # PRESENCE
    # ─────────────────────────────────────────────────────────────────────────
    
    def register_presence(self, capabilities: List[str] = None, metadata: Dict = None) -> None:
        """Register agent presence in the synapse."""
        presence = AgentPresence(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            agent_type=self.agent_type,
            status="online",
            capabilities=capabilities or [],
            last_seen=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {}
        )
        
        key = self._key("presence", self.agent_id)
        self.conn.redis.setex(key, PRESENCE_TTL, json.dumps(presence.to_dict()))
        
        # Announce online
        self.publish_broadcast(MessageType.AGENT_ONLINE, {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "capabilities": capabilities or []
        })
        
        logger.info(f"Presence registered: {self.agent_name}")
    
    def update_presence(self, status: str = "online") -> None:
        """Update presence heartbeat."""
        key = self._key("presence", self.agent_id)
        data = self.conn.redis.get(key)
        if data:
            presence = json.loads(data)
            presence["status"] = status
            presence["last_seen"] = datetime.now(timezone.utc).isoformat()
            self.conn.redis.setex(key, PRESENCE_TTL, json.dumps(presence))
    
    def unregister_presence(self) -> None:
        """Remove agent presence."""
        key = self._key("presence", self.agent_id)
        self.conn.redis.delete(key)
        
        # Announce offline
        self.publish_broadcast(MessageType.AGENT_OFFLINE, {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name
        })
    
    def get_online_agents(self, agent_type: str = None) -> List[AgentPresence]:
        """Get all online agents."""
        pattern = self._key("presence", "*")
        agents = []
        
        for key in self.conn.redis.scan_iter(pattern):
            data = self.conn.redis.get(key)
            if data:
                presence = json.loads(data)
                if agent_type is None or presence.get("agent_type") == agent_type:
                    agents.append(AgentPresence(**presence))
        
        return agents
    
    def find_agent_by_capability(self, capability: str) -> Optional[AgentPresence]:
        """Find an online agent with a specific capability."""
        for agent in self.get_online_agents():
            if capability in agent.capabilities:
                return agent
        return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # MESSAGING
    # ─────────────────────────────────────────────────────────────────────────
    
    def _create_message(
        self,
        msg_type: MessageType,
        payload: Dict[str, Any],
        correlation_id: str = None
    ) -> SynapseMessage:
        """Create a new message."""
        return SynapseMessage(
            id=f"msg-{uuid4().hex[:12]}",
            type=msg_type,
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            payload=payload,
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id=correlation_id
        )
    
    def publish(self, channel: str, message: SynapseMessage) -> int:
        """Publish message to a channel."""
        return self.conn.redis.publish(channel, message.to_json())
    
    def publish_broadcast(self, msg_type: MessageType, payload: Dict[str, Any]) -> int:
        """Broadcast to all agents."""
        message = self._create_message(msg_type, payload)
        channel = self._channel("broadcast")
        count = self.publish(channel, message)
        self._record_event(message)
        return count
    
    def publish_to_team(self, team: str, msg_type: MessageType, payload: Dict[str, Any]) -> int:
        """Publish to a team (PAT or SAT)."""
        message = self._create_message(msg_type, payload)
        channel = self._channel("team", team.lower())
        return self.publish(channel, message)
    
    def publish_to_agent(
        self,
        target_id: str,
        msg_type: MessageType,
        payload: Dict[str, Any],
        correlation_id: str = None
    ) -> int:
        """Send direct message to specific agent."""
        message = self._create_message(msg_type, payload, correlation_id)
        channel = self._channel("agent", target_id)
        return self.publish(channel, message)
    
    def publish_to_task(self, task_id: str, msg_type: MessageType, payload: Dict[str, Any]) -> int:
        """Publish to task coordination channel."""
        message = self._create_message(msg_type, payload)
        message.payload["task_id"] = task_id
        channel = self._channel("task", task_id)
        return self.publish(channel, message)
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUBSCRIPTIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def on(self, msg_type: MessageType, handler: MessageHandler) -> None:
        """Register a message handler."""
        if msg_type not in self._handlers:
            self._handlers[msg_type] = []
        self._handlers[msg_type].append(handler)
    
    def subscribe(self, *channels: str) -> None:
        """Subscribe to channels."""
        for channel in channels:
            self._subscriptions.add(channel)
        
        if self._pubsub:
            self._pubsub.subscribe(*channels)
    
    def subscribe_defaults(self) -> None:
        """Subscribe to default channels for this agent."""
        channels = [
            self._channel("broadcast"),
            self._channel("agent", self.agent_id),
            self._channel("team", self.agent_type.lower()),
        ]
        self.subscribe(*channels)
    
    def _message_listener(self) -> None:
        """Background listener for pub/sub messages."""
        self._pubsub = self.conn.redis.pubsub()
        self._pubsub.subscribe(*self._subscriptions)
        
        while self._running:
            try:
                message = self._pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    try:
                        synapse_msg = SynapseMessage.from_json(message["data"])
                        
                        # Don't process own messages
                        if synapse_msg.sender_id == self.agent_id:
                            continue
                        
                        # Dispatch to handlers
                        handlers = self._handlers.get(synapse_msg.type, [])
                        for handler in handlers:
                            try:
                                handler(synapse_msg)
                            except Exception as e:
                                logger.error(f"Handler error: {e}")
                    except Exception as e:
                        logger.warning(f"Message parse error: {e}")
            except Exception as e:
                logger.error(f"Listener error: {e}")
                time.sleep(1.0)
    
    def start_listening(self) -> None:
        """Start background message listener."""
        if self._running:
            return
        
        self._running = True
        self._listener_thread = threading.Thread(target=self._message_listener, daemon=True)
        self._listener_thread.start()
        logger.info(f"Listener started for {self.agent_name}")
    
    def stop_listening(self) -> None:
        """Stop background listener."""
        self._running = False
        if self._listener_thread:
            self._listener_thread.join(timeout=5.0)
        if self._pubsub:
            self._pubsub.close()
    
    # ─────────────────────────────────────────────────────────────────────────
    # SHARED STATE
    # ─────────────────────────────────────────────────────────────────────────
    
    def set_state(self, key: str, value: Any, ttl: int = None) -> None:
        """Store shared state."""
        full_key = self._key("state", key)
        data = json.dumps(value)
        if ttl:
            self.conn.redis.setex(full_key, ttl, data)
        else:
            self.conn.redis.set(full_key, data)
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Retrieve shared state."""
        full_key = self._key("state", key)
        data = self.conn.redis.get(full_key)
        if data:
            return json.loads(data)
        return default
    
    def delete_state(self, key: str) -> bool:
        """Delete shared state."""
        full_key = self._key("state", key)
        return self.conn.redis.delete(full_key) > 0
    
    def acquire_lock(self, lock_name: str, ttl: int = 30) -> bool:
        """Acquire a distributed lock."""
        key = self._key("lock", lock_name)
        return self.conn.redis.set(key, self.agent_id, nx=True, ex=ttl) is not None
    
    def release_lock(self, lock_name: str) -> bool:
        """Release a distributed lock (only if we own it)."""
        key = self._key("lock", lock_name)
        if self.conn.redis.get(key) == self.agent_id:
            return self.conn.redis.delete(key) > 0
        return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # EVENT SOURCING
    # ─────────────────────────────────────────────────────────────────────────
    
    def _record_event(self, message: SynapseMessage) -> str:
        """Record event to the event stream."""
        stream_key = self._key("events")
        event_id = self.conn.redis.xadd(
            stream_key,
            message.to_dict(),
            maxlen=EVENT_STREAM_MAXLEN
        )
        return event_id
    
    def get_recent_events(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent events from the stream."""
        stream_key = self._key("events")
        events = self.conn.redis.xrevrange(stream_key, count=count)
        return [{"id": e[0], **e[1]} for e in events]
    
    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────
    
    def connect(self, capabilities: List[str] = None) -> bool:
        """Connect to the synapse."""
        if not self.conn.connect():
            return False
        
        self.subscribe_defaults()
        self.start_listening()
        self.register_presence(capabilities)
        
        return True
    
    def disconnect(self) -> None:
        """Disconnect from the synapse."""
        self.unregister_presence()
        self.stop_listening()
        logger.info(f"Agent {self.agent_name} disconnected from synapse")
    
    def heartbeat(self) -> None:
        """Send heartbeat to maintain presence."""
        self.update_presence()
        self.publish_broadcast(MessageType.AGENT_HEARTBEAT, {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name
        })


# ═══════════════════════════════════════════════════════════════════════════════
# SYNAPSE FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_synapse_for_agent(agent_id: str, agent_name: str, agent_type: str = "PAT") -> SynapseBus:
    """Create a synapse bus for an agent."""
    return SynapseBus(agent_id, agent_name, agent_type)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / TEST
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Test the Trinity Synapse."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BIZRA Trinity Synapse")
    parser.add_argument("--agent", type=str, default="TestAgent", help="Agent name")
    parser.add_argument("--type", type=str, default="PAT", choices=["PAT", "SAT"], help="Agent type")
    parser.add_argument("--health", action="store_true", help="Check synapse health")
    parser.add_argument("--list", action="store_true", help="List online agents")
    parser.add_argument("--events", action="store_true", help="Show recent events")
    parser.add_argument("--test", action="store_true", help="Run test scenario")
    
    args = parser.parse_args()
    
    if args.health:
        conn = get_synapse()
        health = conn.health_check()
        print("\n" + "═" * 50)
        print("  SYNAPSE HEALTH")
        print("═" * 50)
        for k, v in health.items():
            print(f"  {k}: {v}")
        print("═" * 50 + "\n")
        return
    
    # Create agent synapse
    agent_id = f"agent-{uuid4().hex[:8]}"
    bus = create_synapse_for_agent(agent_id, args.agent, args.type)
    
    if args.list:
        if not bus.conn.connect():
            print("❌ Failed to connect")
            return
        
        agents = bus.get_online_agents()
        print("\n" + "═" * 50)
        print("  ONLINE AGENTS")
        print("═" * 50)
        if not agents:
            print("  No agents online")
        for agent in agents:
            print(f"  {agent.agent_type}: {agent.agent_name}")
            print(f"    ID: {agent.agent_id}")
            print(f"    Status: {agent.status}")
            print(f"    Capabilities: {', '.join(agent.capabilities) or 'None'}")
            print()
        print("═" * 50 + "\n")
        return
    
    if args.events:
        if not bus.conn.connect():
            print("❌ Failed to connect")
            return
        
        events = bus.get_recent_events(20)
        print("\n" + "═" * 50)
        print("  RECENT EVENTS")
        print("═" * 50)
        if not events:
            print("  No events")
        for event in events[:10]:
            print(f"  [{event.get('type')}] {event.get('sender_name')}")
            print(f"    ID: {event.get('id')}")
            print()
        print("═" * 50 + "\n")
        return
    
    if args.test:
        print("\n🧪 SYNAPSE TEST SCENARIO\n")
        
        # Connect
        print("1. Connecting to synapse...")
        if not bus.connect(capabilities=["reasoning", "analysis"]):
            print("   ❌ Connection failed")
            return
        print(f"   ✅ Connected as {args.agent}")
        
        # Register handler
        print("\n2. Registering message handlers...")
        
        def on_broadcast(msg: SynapseMessage):
            print(f"   📨 Broadcast from {msg.sender_name}: {msg.type.value}")
        
        bus.on(MessageType.SYSTEM_ANNOUNCEMENT, on_broadcast)
        print("   ✅ Handlers registered")
        
        # Set shared state
        print("\n3. Testing shared state...")
        bus.set_state("test:counter", 42)
        value = bus.get_state("test:counter")
        print(f"   ✅ State set/get: {value}")
        
        # Acquire lock
        print("\n4. Testing distributed lock...")
        if bus.acquire_lock("test:resource", ttl=10):
            print("   ✅ Lock acquired")
            bus.release_lock("test:resource")
            print("   ✅ Lock released")
        
        # List agents
        print("\n5. Checking online agents...")
        agents = bus.get_online_agents()
        print(f"   ✅ Online: {len(agents)} agent(s)")
        
        # Broadcast message
        print("\n6. Broadcasting message...")
        count = bus.publish_broadcast(MessageType.SYSTEM_ANNOUNCEMENT, {
            "message": "Test broadcast from CLI"
        })
        print(f"   ✅ Broadcast sent to {count} subscriber(s)")
        
        # Wait for any responses
        print("\n7. Listening for 3 seconds...")
        time.sleep(3)
        
        # Disconnect
        print("\n8. Disconnecting...")
        bus.disconnect()
        print("   ✅ Disconnected")
        
        print("\n✅ Synapse test complete!\n")
        return
    
    # Interactive mode
    print(f"\n🔗 Connecting {args.agent} to synapse...")
    if not bus.connect(capabilities=["test"]):
        print("❌ Failed to connect")
        return
    
    print(f"✅ Connected! Listening for messages...")
    print("   Press Ctrl+C to disconnect\n")
    
    def on_any(msg: SynapseMessage):
        print(f"📨 [{msg.type.value}] from {msg.sender_name}: {json.dumps(msg.payload)[:100]}")
    
    for msg_type in MessageType:
        bus.on(msg_type, on_any)
    
    try:
        while True:
            bus.heartbeat()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n\n👋 Disconnecting...")
        bus.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
