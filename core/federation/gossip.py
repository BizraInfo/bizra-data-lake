"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA PATTERN FEDERATION — GOSSIP PROTOCOL                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Enables peer-to-peer node discovery and health monitoring.                 ║
║   Based on SWIM (Scalable Weakly-consistent Infection-style Membership)      ║
║                                                                              ║
║   Network Effect: Value ∝ n² (Metcalfe's Law)                                ║
║   At 1000 nodes → Self-sustaining                                            ║
║   At 10000 nodes → BIZRA becomes infrastructure                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GOSSIP_INTERVAL_MS = 1000  # Heartbeat every 1s
SUSPICION_TIMEOUT_MS = 5000  # Mark suspect after 5s silence
DEAD_TIMEOUT_MS = 15000  # Mark dead after 15s silence
MAX_FANOUT = 3  # Number of peers to gossip to per round
PROTOCOL_VERSION = "1.0.0"

# Security Hardening S-1: Replay Protection
MAX_MESSAGE_AGE_SECONDS = 300  # 5 minute window for message acceptance
MAX_FUTURE_TIMESTAMP_SECONDS = 30  # Reject messages >30s in future
MAX_RATE_PER_PEER_PER_SECOND = 10  # Rate limiting for DoS protection

# ═══════════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════════


class NodeState(str, Enum):
    ALIVE = "ALIVE"
    SUSPECT = "SUSPECT"
    DEAD = "DEAD"
    LEFT = "LEFT"


class MessageType(str, Enum):
    PING = "PING"
    PING_ACK = "PING_ACK"
    PING_REQ = "PING_REQ"  # Indirect ping via intermediary
    ANNOUNCE = "ANNOUNCE"  # New node joining
    LEAVE = "LEAVE"  # Graceful departure
    PATTERN_SHARE = "PATTERN_SHARE"
    PATTERN_ACK = "PATTERN_ACK"
    PROPOSE = "PROPOSE"
    VOTE = "VOTE"
    COMMIT = "COMMIT"


@dataclass
class NodeInfo:
    """Information about a peer node."""

    node_id: str
    address: str  # host:port
    public_key: str
    state: NodeState = NodeState.ALIVE
    incarnation: int = 0  # Lamport-style counter for state changes
    last_seen: float = field(default_factory=time.time)
    ihsan_average: float = 0.95  # Node's average Ihsān score
    patterns_contributed: int = 0

    def __post_init__(self):
        """SECURITY (SEC-017): Mandatory public key validation."""
        if not self.public_key or len(self.public_key) < 64:
            raise ValueError(
                "NodeInfo requires valid 64-char hex public_key. "
                "Nodes without cryptographic identity cannot participate in BIZRA federation."
            )

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "address": self.address,
            "public_key": self.public_key,
            "state": self.state.value,
            "incarnation": self.incarnation,
            "ihsan_average": self.ihsan_average,
            "patterns_contributed": self.patterns_contributed,
        }


@dataclass
class GossipMessage:
    """Wire format for gossip protocol messages."""

    msg_type: MessageType
    sender_id: str
    sender_address: str
    sequence: int
    timestamp: str
    payload: Dict[str, Any]
    piggyback: List[Dict] = field(default_factory=list)  # Bundled state updates
    signature: str = ""  # Ed25519 signature (SEC-016)

    def _signable_dict(self) -> Dict[str, Any]:
        """Return dict of fields to be signed (excludes signature itself)."""
        return {
            "msg_type": (
                self.msg_type.value
                if isinstance(self.msg_type, MessageType)
                else self.msg_type
            ),
            "sender_id": self.sender_id,
            "sender_address": self.sender_address,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "piggyback": self.piggyback,
        }

    def sign(self, private_key_hex: str) -> "GossipMessage":
        """
        Sign the message with Ed25519 (SEC-016).

        Returns self for method chaining.
        """
        from core.pci.crypto import (
            canonical_json,
            domain_separated_digest,
            sign_message,
        )

        digest = domain_separated_digest(canonical_json(self._signable_dict()))
        self.signature = sign_message(digest, private_key_hex)
        return self

    def verify_signature(self, public_key_hex: str) -> bool:
        """
        Verify Ed25519 signature (SEC-016).

        Returns True if signature is valid, False otherwise.
        """
        from core.pci.crypto import (
            canonical_json,
            domain_separated_digest,
            verify_signature,
        )

        if not self.signature or not public_key_hex:
            return False
        try:
            digest = domain_separated_digest(canonical_json(self._signable_dict()))
            return verify_signature(digest, self.signature, public_key_hex)
        except (ValueError, TypeError, KeyError):
            # Signature verification failed - reject message
            # ValueError: Invalid signature format
            # TypeError: Invalid key type
            # KeyError: Missing required field in signable dict
            return False

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), default=str).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "GossipMessage":
        d = json.loads(data.decode("utf-8"))
        d["msg_type"] = MessageType(d["msg_type"])
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════════════════
# GOSSIP ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class GossipEngine:
    """
    SWIM-style gossip protocol for BIZRA node federation.

    Key Features:
    - Failure detection via ping/ping-ack
    - Protocol-piggybacked membership updates
    - Incarnation numbers to handle state conflicts
    - Suspicion mechanism to reduce false positives
    """

    def __init__(
        self,
        node_id: str,
        address: str,
        public_key: str,
        private_key: str = "",  # SEC-016: Required for signing outgoing messages
        on_node_joined: Optional[Callable[[NodeInfo], None]] = None,
        on_node_left: Optional[Callable[[NodeInfo], None]] = None,
        on_pattern_received: Optional[Callable[[Dict], None]] = None,
    ):
        self.self_node = NodeInfo(
            node_id=node_id,
            address=address,
            public_key=public_key,
            state=NodeState.ALIVE,
        )
        # SECURITY (SEC-016): Store private key for signing outgoing messages
        self._private_key = private_key

        self.peers: Dict[str, NodeInfo] = {}
        self.sequence = 0
        self.running = False

        # Callbacks
        self.on_node_joined = on_node_joined
        self.on_node_left = on_node_left
        self.on_pattern_received = on_pattern_received

        # Consensus Integration
        self.on_consensus_msg: Optional[Callable[[GossipMessage], None]] = None

        # Pending indirect pings
        self._pending_pings: Dict[str, float] = {}

        # Message deduplication
        self._seen_messages: Set[str] = set()
        self._max_seen = 10000

        # SECURITY (SEC-018): Rate limiting per peer - DoS protection
        # Maps sender_id to list of message timestamps (last N seconds)
        self._rate_limit_window: Dict[str, List[float]] = {}
        self._rate_limit_window_seconds = 1.0  # 1 second window

    # ─────────────────────────────────────────────────────────────────────────
    # MEMBERSHIP
    # ─────────────────────────────────────────────────────────────────────────

    def add_seed_node(self, address: str, node_id: Optional[str] = None, public_key: str = ""):
        """
        Add a bootstrap/seed node to connect to.

        SECURITY (SEC-017): public_key is now required for all nodes.
        Raises ValueError if public_key is invalid.
        """
        if not public_key or len(public_key) < 64:
            raise ValueError(
                "add_seed_node requires valid 64-char hex public_key. "
                "Seed nodes must have cryptographic identity."
            )
        nid = node_id or f"seed_{hashlib.sha256(address.encode()).hexdigest()[:16]}"
        self.peers[nid] = NodeInfo(
            node_id=nid, address=address, public_key=public_key, state=NodeState.ALIVE
        )

    def get_alive_peers(self) -> List[NodeInfo]:
        """Return all peers currently considered alive."""
        return [p for p in self.peers.values() if p.state == NodeState.ALIVE]

    def get_network_size(self) -> int:
        """Total nodes in the network (including self)."""
        return 1 + len(self.get_alive_peers())

    def calculate_network_multiplier(self) -> float:
        """
        Metcalfe's Law multiplier for network value.
        M = 1 + (log₁₀(n + 1) / 10) × D × I

        Where:
        - n = node_count
        - D = decentralization factor (simplified to 1.0)
        - I = average Ihsān score
        """
        import math

        n = self.get_network_size()
        avg_ihsan = sum(p.ihsan_average for p in self.peers.values()) / max(
            1, len(self.peers)
        )
        from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

        avg_ihsan = max(UNIFIED_IHSAN_THRESHOLD, avg_ihsan)  # Floor at Ihsān minimum

        multiplier = 1.0 + (math.log10(n + 1) / 10.0) * 1.0 * avg_ihsan
        return round(multiplier, 4)

    # ─────────────────────────────────────────────────────────────────────────
    # MESSAGE HANDLING
    # ─────────────────────────────────────────────────────────────────────────

    def _next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    def _create_message(self, msg_type: MessageType, payload: Dict) -> GossipMessage:
        """Create and sign a gossip message (SEC-016)."""
        # Ensure all messages carry sender public key for trust bootstrapping
        safe_payload = dict(payload) if payload is not None else {}
        if "public_key" not in safe_payload:
            safe_payload["public_key"] = self.self_node.public_key

        msg = GossipMessage(
            msg_type=msg_type,
            sender_id=self.self_node.node_id,
            sender_address=self.self_node.address,
            sequence=self._next_sequence(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=safe_payload,
            piggyback=self._collect_piggyback(),
        )
        # SECURITY (SEC-016): Sign all outgoing messages
        if self._private_key:
            msg.sign(self._private_key)
        return msg

    def _collect_piggyback(self, max_items: int = 5) -> List[Dict]:
        """Collect recent state changes to piggyback on messages."""
        # Priority: recently changed nodes
        updates = []
        for peer in sorted(
            self.peers.values(), key=lambda p: p.last_seen, reverse=True
        )[:max_items]:
            updates.append(peer.to_dict())
        return updates

    def _message_id(self, msg: GossipMessage) -> str:
        return f"{msg.sender_id}:{msg.sequence}"

    def _is_duplicate(self, msg: GossipMessage) -> bool:
        mid = self._message_id(msg)
        if mid in self._seen_messages:
            return True
        self._seen_messages.add(mid)
        if len(self._seen_messages) > self._max_seen:
            # Evict oldest (simple approach - could use LRU)
            self._seen_messages = set(list(self._seen_messages)[-5000:])
        return False

    def _get_sender_public_key(self, sender_id: str) -> str:
        """Get the public key for a sender node."""
        if sender_id in self.peers:
            return self.peers[sender_id].public_key
        return ""

    def _check_rate_limit(self, sender_id: str) -> bool:
        """
        SECURITY (SEC-018): Check if sender exceeds rate limit.
        Returns True if rate limit exceeded (should reject message).

        Standing on Giants: Token Bucket / Sliding Window algorithm
        """
        now = time.time()
        window_start = now - self._rate_limit_window_seconds

        # Get or create timestamp list for this sender
        if sender_id not in self._rate_limit_window:
            self._rate_limit_window[sender_id] = []

        # Remove timestamps outside the window
        self._rate_limit_window[sender_id] = [
            ts for ts in self._rate_limit_window[sender_id] if ts > window_start
        ]

        # Check if rate exceeded
        if len(self._rate_limit_window[sender_id]) >= MAX_RATE_PER_PEER_PER_SECOND:
            return True  # Rate limit exceeded

        # Record this message timestamp
        self._rate_limit_window[sender_id].append(now)

        # Periodic cleanup of stale sender entries (every 100 checks)
        if len(self._rate_limit_window) > 1000:
            stale_senders = [
                sid
                for sid, times in self._rate_limit_window.items()
                if not times or max(times) < window_start
            ]
            for sid in stale_senders[:100]:  # Limit cleanup per call
                del self._rate_limit_window[sid]

        return False

    async def handle_message(self, data: bytes) -> Optional[bytes]:
        """
        Process incoming gossip message.
        Returns response bytes if applicable.

        SECURITY (SEC-016): Validates Ed25519 signature BEFORE any other processing.
        This prevents cache poisoning attacks where unsigned messages poison the
        deduplication cache and cause legitimate signed messages to be rejected.

        SECURITY (S-1): Validates timestamp to prevent replay and time-travel attacks.
        Standing on Giants: Lamport (1982) - "Time, Clocks, and Ordering of Events"

        SECURITY (SEC-018): Rate limiting per peer to prevent DoS flooding.
        """
        try:
            msg = GossipMessage.from_bytes(data)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            # Specific exceptions for malformed messages
            return None

        # SECURITY (SEC-018): Rate limiting - check BEFORE any expensive processing
        # This prevents DoS by rejecting flood traffic early
        if self._check_rate_limit(msg.sender_id):
            return None  # Rate limit exceeded, silent drop

        # SECURITY (S-1): Validate timestamp FIRST - prevents replay attacks
        # This MUST happen before signature verification to avoid wasting CPU
        # on replayed messages with valid signatures
        if hasattr(msg, "timestamp") and msg.timestamp:
            try:
                msg_time = datetime.fromisoformat(msg.timestamp.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                age_seconds = (now - msg_time).total_seconds()

                # Reject messages too old (replay attack)
                if age_seconds > MAX_MESSAGE_AGE_SECONDS:
                    return None  # Silent drop

                # Reject messages too far in future (time-travel attack)
                if age_seconds < -MAX_FUTURE_TIMESTAMP_SECONDS:
                    return None  # Silent drop
            except (ValueError, AttributeError):
                return None  # Invalid timestamp format

        # SECURITY (SEC-016): Verify signature FIRST, before deduplication check.
        # This prevents cache poisoning DoS where attacker sends unsigned message
        # with valid message ID, poisoning _seen_messages and causing real signed
        # message to be rejected as duplicate.
        sender_pubkey = self._get_sender_public_key(msg.sender_id)
        if not sender_pubkey:
            # Check if public_key is in payload (for new node announcements)
            sender_pubkey = msg.payload.get("public_key", "")

        if not msg.verify_signature(sender_pubkey):
            # Reject unsigned or invalid signature messages
            # Silent rejection to avoid amplification attacks
            # NOTE: Do NOT add to _seen_messages - this is intentional to prevent
            # cache poisoning attacks
            return None

        # Only check for duplicates AFTER signature verification passes
        if self._is_duplicate(msg):
            return None

        # Process piggybacked state updates
        for update in msg.piggyback:
            self._merge_node_state(update)

        # Update sender's last_seen
        if msg.sender_id in self.peers:
            self.peers[msg.sender_id].last_seen = time.time()
            self.peers[msg.sender_id].state = NodeState.ALIVE
        else:
            # New peer discovered!
            self._add_peer_from_message(msg)

        # Handle by type
        if msg.msg_type == MessageType.PING:
            return self._handle_ping(msg)
        elif msg.msg_type == MessageType.PING_ACK:
            self._handle_ping_ack(msg)
            return None
        elif msg.msg_type == MessageType.ANNOUNCE:
            return self._handle_announce(msg)
        elif msg.msg_type == MessageType.LEAVE:
            self._handle_leave(msg)
            return None
        elif msg.msg_type == MessageType.PATTERN_SHARE:
            return self._handle_pattern_share(msg)
        elif msg.msg_type in [
            MessageType.PROPOSE,
            MessageType.VOTE,
            MessageType.COMMIT,
        ]:
            if self.on_consensus_msg:
                self.on_consensus_msg(msg)
            return None

        return None

    def _add_peer_from_message(self, msg: GossipMessage):
        """Add a newly discovered peer."""
        public_key = msg.payload.get("public_key", "")
        if not public_key or len(public_key) < 64:
            # Do not add peers without cryptographic identity
            return
        peer = NodeInfo(
            node_id=msg.sender_id,
            address=msg.sender_address,
            public_key=public_key,
            state=NodeState.ALIVE,
        )
        self.peers[msg.sender_id] = peer
        if self.on_node_joined:
            self.on_node_joined(peer)

    def _merge_node_state(self, update: Dict):
        """Merge a piggybacked state update using incarnation numbers."""
        node_id = update.get("node_id")
        if not node_id or node_id == self.self_node.node_id:
            return

        new_incarnation = update.get("incarnation", 0)
        new_state = NodeState(update.get("state", "ALIVE"))

        if node_id in self.peers:
            existing = self.peers[node_id]
            # Only accept if incarnation is higher
            if new_incarnation > existing.incarnation:
                existing.incarnation = new_incarnation
                existing.state = new_state
                existing.ihsan_average = update.get(
                    "ihsan_average", existing.ihsan_average
                )
        else:
            # New peer from gossip
            public_key = update.get("public_key", "")
            if not public_key or len(public_key) < 64:
                return
            self.peers[node_id] = NodeInfo(
                node_id=node_id,
                address=update.get("address", "unknown"),
                public_key=public_key,
                state=new_state,
                incarnation=new_incarnation,
                ihsan_average=update.get("ihsan_average", 0.95),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # MESSAGE TYPE HANDLERS
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_ping(self, msg: GossipMessage) -> bytes:
        """Respond with PING_ACK."""
        ack = self._create_message(
            MessageType.PING_ACK,
            {"in_response_to": msg.sequence, "node_count": self.get_network_size()},
        )
        return ack.to_bytes()

    def _handle_ping_ack(self, msg: GossipMessage) -> None:
        """Clear pending ping for sender."""
        if msg.sender_id in self._pending_pings:
            del self._pending_pings[msg.sender_id]
        return None

    def _handle_announce(self, msg: GossipMessage) -> bytes:
        """New node announcing itself."""
        # Already added in _add_peer_from_message
        # Send back our peer list as welcome
        ack = self._create_message(
            MessageType.PING_ACK,
            {
                "welcome": True,
                "network_size": self.get_network_size(),
                "known_peers": [p.to_dict() for p in self.get_alive_peers()[:10]],
            },
        )
        return ack.to_bytes()

    def _handle_leave(self, msg: GossipMessage) -> None:
        """Node gracefully leaving."""
        if msg.sender_id in self.peers:
            self.peers[msg.sender_id].state = NodeState.LEFT
            if self.on_node_left:
                self.on_node_left(self.peers[msg.sender_id])
        return None

    def _handle_pattern_share(self, msg: GossipMessage) -> bytes:
        """Receive a shared pattern."""
        if self.on_pattern_received:
            self.on_pattern_received(msg.payload)

        ack = self._create_message(
            MessageType.PATTERN_ACK,
            {"pattern_id": msg.payload.get("pattern_id"), "accepted": True},
        )
        return ack.to_bytes()

    # ─────────────────────────────────────────────────────────────────────────
    # OUTBOUND MESSAGES
    # ─────────────────────────────────────────────────────────────────────────

    def create_announce_message(self) -> bytes:
        """Create announcement for joining the network."""
        msg = self._create_message(
            MessageType.ANNOUNCE,
            {
                "public_key": self.self_node.public_key,
                "ihsan_average": self.self_node.ihsan_average,
            },
        )
        return msg.to_bytes()

    def create_ping_message(self) -> bytes:
        """Create a ping message."""
        msg = self._create_message(MessageType.PING, {})
        return msg.to_bytes()

    def create_leave_message(self) -> bytes:
        """Create graceful leave message."""
        msg = self._create_message(MessageType.LEAVE, {"reason": "graceful_shutdown"})
        return msg.to_bytes()

    def create_pattern_share_message(self, pattern_data: Dict) -> bytes:
        """Create message to share a pattern."""
        msg = self._create_message(MessageType.PATTERN_SHARE, pattern_data)
        return msg.to_bytes()

    def broadcast_pattern(self, pattern_data: Dict):
        """
        Broadcast a pattern to gossip targets.
        Returns the message bytes for external handling.
        """
        msg_bytes = self.create_pattern_share_message(pattern_data)
        # In a real implementation, this would send to gossip targets
        # For now, return the bytes for the caller to handle
        return msg_bytes

    # ─────────────────────────────────────────────────────────────────────────
    # FAILURE DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def check_peer_health(self):
        """
        Check all peers and update their states based on last_seen.
        Called periodically by the gossip loop.
        """
        now = time.time()
        for peer in self.peers.values():
            if peer.state == NodeState.LEFT:
                continue

            silence_ms = (now - peer.last_seen) * 1000

            if silence_ms > DEAD_TIMEOUT_MS:
                if peer.state != NodeState.DEAD:
                    peer.state = NodeState.DEAD
                    if self.on_node_left:
                        self.on_node_left(peer)
            elif silence_ms > SUSPICION_TIMEOUT_MS:
                peer.state = NodeState.SUSPECT

    def select_gossip_targets(self) -> List[NodeInfo]:
        """Select random peers to gossip to (fanout)."""
        alive = self.get_alive_peers()
        if len(alive) <= MAX_FANOUT:
            return alive
        return random.sample(alive, MAX_FANOUT)

    # ─────────────────────────────────────────────────────────────────────────
    # STATISTICS
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get gossip engine statistics."""
        return {
            "node_id": self.self_node.node_id,
            "network_size": self.get_network_size(),
            "alive_peers": len(self.get_alive_peers()),
            "suspect_peers": len(
                [p for p in self.peers.values() if p.state == NodeState.SUSPECT]
            ),
            "dead_peers": len(
                [p for p in self.peers.values() if p.state == NodeState.DEAD]
            ),
            "network_multiplier": self.calculate_network_multiplier(),
            "sequence": self.sequence,
            "running": self.running,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # NETWORKING LAYER — UDP Transport
    # ═══════════════════════════════════════════════════════════════════════════

    async def start(self):
        """
        Start the gossip engine with UDP networking.
        Binds to the configured address and begins listening.
        """
        if self.running:
            return

        host, port = self.self_node.address.split(":")
        port = int(port)

        # Create UDP socket
        loop = asyncio.get_event_loop()

        class GossipProtocol(asyncio.DatagramProtocol):
            def __init__(self, engine):
                self.engine = engine
                self.transport = None

            def connection_made(self, transport):
                self.transport = transport

            def datagram_received(self, data, addr):
                # Schedule async handling
                asyncio.create_task(self._handle(data, addr))

            async def _handle(self, data, addr):
                response = await self.engine.handle_message(data)
                if response and self.transport:
                    self.transport.sendto(response, addr)

        # Bind UDP socket
        self._transport, self._protocol = await loop.create_datagram_endpoint(
            lambda: GossipProtocol(self), local_addr=(host, port)
        )

        self.running = True

        # Start background loops
        self._gossip_task = asyncio.create_task(self._gossip_loop())
        self._health_task = asyncio.create_task(self._health_check_loop())

        print(f"🌐 GossipEngine started on {host}:{port}")

    async def stop(self):
        """Gracefully shutdown the gossip engine."""
        if not self.running:
            return

        self.running = False

        # Send LEAVE to all peers
        leave_msg = self.create_leave_message()
        for peer in self.get_alive_peers():
            await self._send_to(peer.address, leave_msg)

        # Cancel background tasks
        if hasattr(self, "_gossip_task"):
            self._gossip_task.cancel()
        if hasattr(self, "_health_task"):
            self._health_task.cancel()

        # Close transport
        if hasattr(self, "_transport") and self._transport:
            self._transport.close()

        print("🔌 GossipEngine stopped")

    async def join_network(self, host: str, port: int):
        """
        Join the network by announcing to a seed node.
        """
        addr = f"{host}:{port}"
        announce = self.create_announce_message()
        await self._send_to(addr, announce)
        print(f"📢 Announced to seed node {addr}")

    async def _send_to(self, address: str, data: bytes):
        """Send UDP datagram to an address."""
        if not hasattr(self, "_transport") or not self._transport:
            return

        try:
            host, port = address.split(":")
            self._transport.sendto(data, (host, int(port)))
        except Exception as e:
            print(f"⚠️ Failed to send to {address}: {e}")

    async def _gossip_loop(self):
        """
        Main gossip loop - periodically ping random peers.
        Runs every GOSSIP_INTERVAL_MS.
        """
        while self.running:
            try:
                await asyncio.sleep(GOSSIP_INTERVAL_MS / 1000.0)

                targets = self.select_gossip_targets()
                ping = self.create_ping_message()

                for peer in targets:
                    self._pending_pings[peer.node_id] = time.time()
                    await self._send_to(peer.address, ping)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️ Gossip loop error: {e}")

    async def _health_check_loop(self):
        """
        Periodically check peer health based on last_seen times.
        """
        while self.running:
            try:
                await asyncio.sleep(SUSPICION_TIMEOUT_MS / 1000.0)
                self.check_peer_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️ Health check error: {e}")

    async def broadcast_pattern_async(self, pattern_data: Dict):
        """
        Broadcast a pattern to all gossip targets over the network.
        """
        msg = self.create_pattern_share_message(pattern_data)
        targets = self.select_gossip_targets()
        for peer in targets:
            await self._send_to(peer.address, msg)
        return len(targets)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    sys.path.insert(0, "c:\\BIZRA-DATA-LAKE")
    from core.pci import generate_keypair  # type: ignore[attr-defined]

    print("=" * 70)
    print("BIZRA GOSSIP PROTOCOL — Simulation")
    print("=" * 70)

    # Create 5 simulated nodes
    nodes = []
    for i in range(5):
        priv, pub = generate_keypair()
        engine = GossipEngine(
            node_id=f"node_{i:03d}",
            address=f"127.0.0.1:{8800 + i}",
            public_key=pub,
            on_node_joined=lambda n: print(f"  🟢 Node joined: {n.node_id}"),
            on_pattern_received=lambda p: print(
                f"  📦 Pattern received: {p.get('pattern_id', 'unknown')}"
            ),
        )
        nodes.append(engine)

    # Node 0 is the bootstrap
    for i in range(1, 5):
        nodes[i].add_seed_node("127.0.0.1:8800", "node_000")

    # Simulate gossip round
    print("\n[Round 1] Node 1 announces to Node 0...")
    announce = nodes[1].create_announce_message()
    response = asyncio.run(nodes[0].handle_message(announce))

    print(f"  Node 0 received announcement, responded with {len(response or b'')} bytes")

    # Node 0 processes response
    if response:
        asyncio.run(nodes[1].handle_message(response))

    # Check stats
    print("\n[Stats]")
    for i, node in enumerate(nodes):
        stats = node.get_stats()
        print(
            f"  Node {i}: {stats['alive_peers']} peers, multiplier={stats['network_multiplier']}"
        )

    # Simulate pattern sharing
    print("\n[Round 2] Node 1 shares a pattern...")
    pattern_msg = nodes[1].create_pattern_share_message(
        {
            "pattern_id": "sape_001",
            "logic": "IF snr < 0.8 THEN apply_refinement()",
            "impact_score": 0.92,
        }
    )
    asyncio.run(nodes[0].handle_message(pattern_msg))

    print("\n" + "=" * 70)
    print("✅ Gossip Protocol Demo Complete")
    print("=" * 70)
