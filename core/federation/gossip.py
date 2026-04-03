# core/federation/gossip.py - Gossip Protocol for Pattern Propagation
#
# Epidemic-style gossip protocol for efficient pattern distribution.
# Uses fanout-based broadcasting with exponential backoff.
#
# Protocol Features:
# - Peer discovery via mDNS (local) or bootstrap nodes (WAN)
# - Lazy pull for bandwidth efficiency (announce → request → response)
# - Reputation tracking for peer quality
# - Rate limiting to prevent spam

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.federation.protocol import (
    GossipMessage,
    GossipMessageType,
    PatternEnvelope,
    generate_keypair,
)


logger = logging.getLogger("federation.gossip")


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Gossip parameters
GOSSIP_FANOUT = 3  # Number of peers to gossip to per round
GOSSIP_INTERVAL_SEC = 5.0  # Seconds between gossip rounds
HEARTBEAT_INTERVAL_SEC = 30.0  # Heartbeat frequency
PEER_TIMEOUT_SEC = 90.0  # Peer considered dead after this
MAX_PEERS = 50  # Maximum tracked peers
MAX_PATTERNS_PER_MIN = 100  # Rate limit

# Bootstrap nodes (for WAN discovery)
BOOTSTRAP_NODES = [
    # ("node0.bizra.network", 9999),  # Genesis node (future)
]


# ═══════════════════════════════════════════════════════════════════════════════
# PEER MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class PeerState(Enum):
    """Peer connection states."""
    UNKNOWN = "unknown"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    BANNED = "banned"


@dataclass
class PeerInfo:
    """Information about a peer node."""
    
    node_id: str
    host: str
    port: int
    state: PeerState = PeerState.UNKNOWN
    
    # Reputation
    reputation: float = 1.0  # [0-1], starts at 1
    patterns_received: int = 0
    patterns_rejected: int = 0
    
    # Timing
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_heartbeat: float = 0.0
    
    # Rate limiting
    patterns_this_minute: int = 0
    minute_window_start: float = field(default_factory=time.time)
    
    def update_rate_limit(self) -> bool:
        """Check and update rate limit. Returns True if allowed."""
        now = time.time()
        
        # Reset window if minute has passed
        if now - self.minute_window_start > 60:
            self.minute_window_start = now
            self.patterns_this_minute = 0
        
        if self.patterns_this_minute >= MAX_PATTERNS_PER_MIN:
            return False
        
        self.patterns_this_minute += 1
        return True
    
    def record_pattern(self, accepted: bool):
        """Record pattern reception for reputation."""
        self.patterns_received += 1
        if not accepted:
            self.patterns_rejected += 1
        
        # Update reputation (EMA)
        success_rate = 1.0 - (self.patterns_rejected / max(1, self.patterns_received))
        self.reputation = 0.9 * self.reputation + 0.1 * success_rate
    
    def is_alive(self) -> bool:
        """Check if peer is considered alive."""
        return (
            self.state == PeerState.CONNECTED and
            time.time() - self.last_seen < PEER_TIMEOUT_SEC
        )
    
    @property
    def address(self) -> Tuple[str, int]:
        return (self.host, self.port)


# ═══════════════════════════════════════════════════════════════════════════════
# GOSSIP PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════════

class GossipProtocol:
    """
    Epidemic gossip protocol for pattern federation.
    
    Uses a lazy-pull approach:
    1. PATTERN_ANNOUNCE: Announce pattern existence (header only)
    2. PATTERN_REQUEST: Peer requests full pattern if interested
    3. PATTERN_RESPONSE: Send full pattern data
    
    This minimizes bandwidth for patterns peers already have.
    """
    
    def __init__(
        self,
        node_id: str,
        host: str = "0.0.0.0",
        port: int = 9999,
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        
        # Crypto
        self._private_key, self._public_key = generate_keypair()
        
        # Peer management
        self.peers: Dict[str, PeerInfo] = {}
        self.banned_nodes: Set[str] = set()
        
        # Pattern tracking
        self.known_patterns: Set[str] = set()  # Pattern IDs we have
        self.pending_requests: Dict[str, float] = {}  # pattern_id → request_time
        
        # Message deduplication
        self.seen_messages: Set[str] = set()
        self.seen_message_times: Dict[str, float] = {}
        
        # Callbacks
        self._on_pattern_received: Optional[Callable[[PatternEnvelope], None]] = None
        self._on_peer_connected: Optional[Callable[[PeerInfo], None]] = None
        
        # Async infrastructure
        self._server: Optional[asyncio.Server] = None
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Statistics
        self.stats = GossipStats()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────
    
    async def start(self):
        """Start gossip protocol."""
        if self._running:
            return
        
        self._running = True
        
        # Start TCP server
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
        )
        
        logger.info(f"🔗 Gossip server started on {self.host}:{self.port}")
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._gossip_loop()),
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._cleanup_loop()),
        ]
        
        # Connect to bootstrap nodes
        for host, port in BOOTSTRAP_NODES:
            asyncio.create_task(self.connect_to_peer(host, port))
    
    async def stop(self):
        """Stop gossip protocol."""
        self._running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        # Close server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        
        logger.info("🔌 Gossip server stopped")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Connection Handling
    # ─────────────────────────────────────────────────────────────────────────
    
    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        """Handle incoming peer connection."""
        peer_addr = writer.get_extra_info("peername")
        logger.debug(f"New connection from {peer_addr}")
        
        try:
            while self._running:
                # Read message length prefix (4 bytes, big endian)
                length_bytes = await reader.readexactly(4)
                length = int.from_bytes(length_bytes, "big")
                
                if length > 1_000_000:  # 1MB max message
                    logger.warning(f"Message too large from {peer_addr}: {length}")
                    break
                
                # Read message
                data = await reader.readexactly(length)
                
                # Parse and handle
                try:
                    msg = GossipMessage.from_wire(data)
                    response = await self._handle_message(msg, writer)
                    
                    if response:
                        await self._send_message(writer, response)
                        
                except Exception as e:
                    logger.error(f"Error handling message from {peer_addr}: {e}")
                    
        except asyncio.IncompleteReadError:
            pass  # Connection closed
        except Exception as e:
            logger.error(f"Connection error from {peer_addr}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _send_message(self, writer: asyncio.StreamWriter, msg: GossipMessage):
        """Send message to peer."""
        data = msg.to_wire()
        length = len(data).to_bytes(4, "big")
        writer.write(length + data)
        await writer.drain()
        self.stats.messages_sent += 1
    
    async def connect_to_peer(self, host: str, port: int) -> Optional[PeerInfo]:
        """Connect to a peer."""
        try:
            reader, writer = await asyncio.open_connection(host, port)
            
            # Send HELLO
            hello = GossipMessage.create(
                GossipMessageType.HELLO,
                self.node_id,
                {
                    "port": self.port,
                    "public_key": self._public_key.hex(),
                    "patterns_count": len(self.known_patterns),
                },
                self._private_key,
            )
            
            await self._send_message(writer, hello)
            
            # Wait for response
            length_bytes = await asyncio.wait_for(reader.readexactly(4), timeout=10.0)
            length = int.from_bytes(length_bytes, "big")
            data = await reader.readexactly(length)
            
            response = GossipMessage.from_wire(data)
            
            if response.msg_type == GossipMessageType.HELLO:
                peer_id = response.sender_id
                peer = PeerInfo(
                    node_id=peer_id,
                    host=host,
                    port=port,
                    state=PeerState.CONNECTED,
                )
                self.peers[peer_id] = peer
                
                logger.info(f"✅ Connected to peer {peer_id} at {host}:{port}")
                
                if self._on_peer_connected:
                    self._on_peer_connected(peer)
                
                # Start handling this connection
                asyncio.create_task(self._peer_handler(peer, reader, writer))
                
                return peer
                
        except Exception as e:
            logger.warning(f"Failed to connect to {host}:{port}: {e}")
            return None
    
    async def _peer_handler(
        self,
        peer: PeerInfo,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        """Handle ongoing peer connection."""
        try:
            while self._running and peer.state == PeerState.CONNECTED:
                try:
                    length_bytes = await asyncio.wait_for(
                        reader.readexactly(4),
                        timeout=PEER_TIMEOUT_SEC,
                    )
                    length = int.from_bytes(length_bytes, "big")
                    data = await reader.readexactly(length)
                    
                    msg = GossipMessage.from_wire(data)
                    peer.last_seen = time.time()
                    
                    response = await self._handle_message(msg, writer)
                    if response:
                        await self._send_message(writer, response)
                        
                except asyncio.TimeoutError:
                    logger.debug(f"Peer {peer.node_id} timed out")
                    break
                    
        except Exception as e:
            logger.error(f"Peer handler error for {peer.node_id}: {e}")
        finally:
            peer.state = PeerState.DISCONNECTED
            writer.close()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Message Handling
    # ─────────────────────────────────────────────────────────────────────────
    
    async def _handle_message(
        self,
        msg: GossipMessage,
        writer: asyncio.StreamWriter,
    ) -> Optional[GossipMessage]:
        """Handle incoming gossip message."""
        
        # Check for banned sender
        if msg.sender_id in self.banned_nodes:
            return None
        
        # Deduplicate
        if msg.msg_id in self.seen_messages:
            return None
        self.seen_messages.add(msg.msg_id)
        self.seen_message_times[msg.msg_id] = time.time()
        
        self.stats.messages_received += 1
        
        # Route by type
        if msg.msg_type == GossipMessageType.HELLO:
            return await self._handle_hello(msg)
            
        elif msg.msg_type == GossipMessageType.HEARTBEAT:
            return await self._handle_heartbeat(msg)
            
        elif msg.msg_type == GossipMessageType.PATTERN_ANNOUNCE:
            return await self._handle_pattern_announce(msg)
            
        elif msg.msg_type == GossipMessageType.PATTERN_REQUEST:
            return await self._handle_pattern_request(msg)
            
        elif msg.msg_type == GossipMessageType.PATTERN_RESPONSE:
            return await self._handle_pattern_response(msg)
            
        elif msg.msg_type == GossipMessageType.PEER_LIST:
            return await self._handle_peer_list(msg)
            
        elif msg.msg_type == GossipMessageType.BAN_ANNOUNCE:
            return await self._handle_ban_announce(msg)
        
        return None
    
    async def _handle_hello(self, msg: GossipMessage) -> GossipMessage:
        """Handle HELLO handshake."""
        peer_id = msg.sender_id
        
        if peer_id not in self.peers:
            # New peer
            self.peers[peer_id] = PeerInfo(
                node_id=peer_id,
                host=msg.payload.get("host", "unknown"),
                port=msg.payload.get("port", 9999),
                state=PeerState.CONNECTED,
            )
        
        self.peers[peer_id].last_seen = time.time()
        self.peers[peer_id].state = PeerState.CONNECTED
        
        # Respond with our HELLO
        return GossipMessage.create(
            GossipMessageType.HELLO,
            self.node_id,
            {
                "port": self.port,
                "public_key": self._public_key.hex(),
                "patterns_count": len(self.known_patterns),
            },
            self._private_key,
        )
    
    async def _handle_heartbeat(self, msg: GossipMessage) -> Optional[GossipMessage]:
        """Handle heartbeat."""
        if msg.sender_id in self.peers:
            self.peers[msg.sender_id].last_heartbeat = time.time()
            self.peers[msg.sender_id].last_seen = time.time()
        return None
    
    async def _handle_pattern_announce(self, msg: GossipMessage) -> Optional[GossipMessage]:
        """Handle pattern announcement."""
        pattern_id = msg.payload.get("pattern_id")
        
        if not pattern_id:
            return None
        
        # Already have it?
        if pattern_id in self.known_patterns:
            return None
        
        # Rate limit check
        peer = self.peers.get(msg.sender_id)
        if peer and not peer.update_rate_limit():
            logger.warning(f"Rate limit exceeded for peer {msg.sender_id}")
            return None
        
        # Request full pattern
        self.pending_requests[pattern_id] = time.time()
        
        return GossipMessage.create(
            GossipMessageType.PATTERN_REQUEST,
            self.node_id,
            {"pattern_id": pattern_id},
        )
    
    async def _handle_pattern_request(self, msg: GossipMessage) -> Optional[GossipMessage]:
        """Handle pattern request."""
        pattern_id = msg.payload.get("pattern_id")
        
        if not pattern_id or pattern_id not in self.known_patterns:
            return None
        
        # Retrieve pattern (caller must set this callback)
        if hasattr(self, "_get_pattern_callback") and self._get_pattern_callback:
            envelope = self._get_pattern_callback(pattern_id)
            if envelope:
                return GossipMessage.create(
                    GossipMessageType.PATTERN_RESPONSE,
                    self.node_id,
                    {"envelope": envelope.model_dump()},
                )
        
        return None
    
    async def _handle_pattern_response(self, msg: GossipMessage) -> None:
        """Handle pattern response."""
        envelope_data = msg.payload.get("envelope")
        if not envelope_data:
            return
        
        try:
            envelope = PatternEnvelope.model_validate(envelope_data)
            
            # Verify
            valid, reason = envelope.verify()
            
            if not valid:
                logger.warning(f"Invalid pattern from {msg.sender_id}: {reason}")
                if msg.sender_id in self.peers:
                    self.peers[msg.sender_id].record_pattern(False)
                self.stats.patterns_rejected += 1
                return
            
            # Accept pattern
            self.known_patterns.add(envelope.metadata.pattern_id)
            self.pending_requests.pop(envelope.metadata.pattern_id, None)
            
            if msg.sender_id in self.peers:
                self.peers[msg.sender_id].record_pattern(True)
            
            self.stats.patterns_received += 1
            
            # Notify callback
            if self._on_pattern_received:
                self._on_pattern_received(envelope)
            
            # Re-gossip to other peers (with hop increment)
            envelope.hop_count += 1
            if envelope.hop_count < 10:  # Max 10 hops
                await self.gossip_pattern_announce(envelope.metadata.pattern_id)
                
        except Exception as e:
            logger.error(f"Error processing pattern response: {e}")
    
    async def _handle_peer_list(self, msg: GossipMessage) -> None:
        """Handle peer list sharing."""
        peers = msg.payload.get("peers", [])
        
        for peer_info in peers[:10]:  # Max 10 new peers per message
            node_id = peer_info.get("node_id")
            host = peer_info.get("host")
            port = peer_info.get("port")
            
            if not all([node_id, host, port]):
                continue
            
            if node_id in self.peers or node_id in self.banned_nodes:
                continue
            
            if node_id == self.node_id:
                continue
            
            if len(self.peers) < MAX_PEERS:
                # Try to connect
                asyncio.create_task(self.connect_to_peer(host, port))
    
    async def _handle_ban_announce(self, msg: GossipMessage) -> None:
        """Handle ban announcement."""
        banned_id = msg.payload.get("node_id")
        reason = msg.payload.get("reason", "unspecified")
        
        if banned_id and banned_id != self.node_id:
            self.banned_nodes.add(banned_id)
            if banned_id in self.peers:
                self.peers[banned_id].state = PeerState.BANNED
            logger.warning(f"Node {banned_id} banned: {reason}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Broadcasting
    # ─────────────────────────────────────────────────────────────────────────
    
    async def broadcast_pattern(self, envelope: PatternEnvelope):
        """Broadcast new pattern to network."""
        self.known_patterns.add(envelope.metadata.pattern_id)
        await self.gossip_pattern_announce(envelope.metadata.pattern_id)
        self.stats.patterns_broadcast += 1
    
    async def gossip_pattern_announce(self, pattern_id: str):
        """Gossip pattern announcement to random peers."""
        alive_peers = [p for p in self.peers.values() if p.is_alive()]
        
        if not alive_peers:
            return
        
        # Select random subset (fanout)
        selected = random.sample(alive_peers, min(GOSSIP_FANOUT, len(alive_peers)))
        
        announce = GossipMessage.create(
            GossipMessageType.PATTERN_ANNOUNCE,
            self.node_id,
            {"pattern_id": pattern_id},
        )
        
        for peer in selected:
            try:
                reader, writer = await asyncio.open_connection(peer.host, peer.port)
                await self._send_message(writer, announce)
                writer.close()
                await writer.wait_closed()
            except Exception as e:
                logger.debug(f"Failed to gossip to {peer.node_id}: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Background Tasks
    # ─────────────────────────────────────────────────────────────────────────
    
    async def _gossip_loop(self):
        """Periodic gossip round."""
        while self._running:
            try:
                await asyncio.sleep(GOSSIP_INTERVAL_SEC)
                
                # Share peer list with random peers
                alive_peers = [p for p in self.peers.values() if p.is_alive()]
                
                if alive_peers and len(alive_peers) > 1:
                    target = random.choice(alive_peers)
                    peer_list = [
                        {"node_id": p.node_id, "host": p.host, "port": p.port}
                        for p in alive_peers if p.node_id != target.node_id
                    ][:5]
                    
                    msg = GossipMessage.create(
                        GossipMessageType.PEER_LIST,
                        self.node_id,
                        {"peers": peer_list},
                    )
                    
                    try:
                        reader, writer = await asyncio.open_connection(
                            target.host, target.port
                        )
                        await self._send_message(writer, msg)
                        writer.close()
                    except Exception:
                        pass
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Gossip loop error: {e}")
    
    async def _heartbeat_loop(self):
        """Periodic heartbeat."""
        while self._running:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
                
                heartbeat = GossipMessage.create(
                    GossipMessageType.HEARTBEAT,
                    self.node_id,
                    {"patterns_count": len(self.known_patterns)},
                )
                
                for peer in list(self.peers.values()):
                    if peer.state == PeerState.CONNECTED:
                        try:
                            reader, writer = await asyncio.open_connection(
                                peer.host, peer.port
                            )
                            await self._send_message(writer, heartbeat)
                            writer.close()
                        except Exception:
                            peer.state = PeerState.DISCONNECTED
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of stale data."""
        while self._running:
            try:
                await asyncio.sleep(60.0)  # Every minute
                
                now = time.time()
                
                # Clean seen messages older than 5 minutes
                stale_msgs = [
                    msg_id for msg_id, ts in self.seen_message_times.items()
                    if now - ts > 300
                ]
                for msg_id in stale_msgs:
                    self.seen_messages.discard(msg_id)
                    self.seen_message_times.pop(msg_id, None)
                
                # Clean stale pending requests
                stale_requests = [
                    pid for pid, ts in self.pending_requests.items()
                    if now - ts > 60
                ]
                for pid in stale_requests:
                    self.pending_requests.pop(pid, None)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────
    
    def on_pattern_received(self, callback: Callable[[PatternEnvelope], None]):
        """Register callback for received patterns."""
        self._on_pattern_received = callback
    
    def on_peer_connected(self, callback: Callable[[PeerInfo], None]):
        """Register callback for new peer connections."""
        self._on_peer_connected = callback
    
    def set_pattern_getter(self, callback: Callable[[str], Optional[PatternEnvelope]]):
        """Set callback to retrieve patterns by ID."""
        self._get_pattern_callback = callback


@dataclass
class GossipStats:
    """Gossip protocol statistics."""
    
    messages_sent: int = 0
    messages_received: int = 0
    patterns_received: int = 0
    patterns_rejected: int = 0
    patterns_broadcast: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "patterns_received": self.patterns_received,
            "patterns_rejected": self.patterns_rejected,
            "patterns_broadcast": self.patterns_broadcast,
        }
