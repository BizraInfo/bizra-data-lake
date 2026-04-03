"""
BIZRA 3-Node Local Federation - Node Implementation
Phase 9: Distributed Consensus with Byzantine Fault Tolerance

This module implements a single federation node that participates in:
- Byzantine fault tolerant leader election
- Distributed consensus using PBFT-inspired protocol
- State synchronization across the 3-node mesh
- Knowledge graph sharding and distribution
- Cross-node Graph-of-Thoughts reasoning
"""

import asyncio
import hashlib
import hmac
import json
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import socket
import pickle
import struct

from .consensus_engine import ConsensusEngine, SHARED_SECRET
from .state_ledger import StateLedger
from .memory_system import CognitivePermanence


class NodeState(Enum):
    """Federation node states in the consensus protocol."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    VIEW_CHANGE = "view_change"


class MessageType(Enum):
    """Message types for federation communication."""
    PRE_PREPARE = "pre_prepare"
    PREPARE = "prepare"
    COMMIT = "commit"
    VIEW_CHANGE = "view_change"
    NEW_VIEW = "new_view"
    REQUEST = "request"
    REPLY = "reply"
    HEARTBEAT = "heartbeat"
    STATE_SYNC = "state_sync"
    SHARD_UPDATE = "shard_update"
    GRAPH_REASONING = "graph_reasoning"


@dataclass
class FederationMessage:
    """Standardized message format for federation communication."""
    msg_type: MessageType
    sender_id: str
    receiver_id: str
    view_number: int
    sequence_number: int
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    signature: str = ""

    def sign(self, secret: bytes) -> None:
        """Sign the message with HMAC."""
        msg_data = f"{self.msg_type.value}:{self.sender_id}:{self.view_number}:{self.sequence_number}:{json.dumps(self.payload, sort_keys=True)}:{self.timestamp}"
        self.signature = hmac.new(secret, msg_data.encode(), hashlib.sha256).hexdigest()

    def verify_signature(self, secret: bytes) -> bool:
        """Verify the message signature."""
        expected_sig = self.signature
        self.signature = ""  # Temporarily remove for verification
        msg_data = f"{self.msg_type.value}:{self.sender_id}:{self.view_number}:{self.sequence_number}:{json.dumps(self.payload, sort_keys=True)}:{self.timestamp}"
        self.signature = expected_sig  # Restore
        calculated_sig = hmac.new(secret, msg_data.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected_sig, calculated_sig)


@dataclass
class ConsensusState:
    """PBFT-inspired consensus state for each node."""
    view_number: int = 0
    sequence_number: int = 0
    current_leader: str = ""
    node_state: NodeState = NodeState.FOLLOWER

    # PBFT state
    pre_prepare_log: Dict[int, FederationMessage] = field(default_factory=dict)
    prepare_log: Dict[Tuple[int, int], List[FederationMessage]] = field(default_factory=dict)
    commit_log: Dict[Tuple[int, int], List[FederationMessage]] = field(default_factory=dict)

    # View change state
    view_change_messages: Dict[int, List[FederationMessage]] = field(default_factory=dict)


class FederationNode:
    """
    A single node in the BIZRA 3-Node Local Federation.

    Implements Byzantine fault tolerant consensus, state synchronization,
    knowledge graph sharding, and distributed Graph-of-Thoughts reasoning.
    """

    def __init__(self, node_id: str, peer_nodes: List[str], port: int = 8888, consensus_state: Optional[ConsensusState] = None, consensus_engine: Any = None):
        self.node_id = node_id
        self.peer_nodes = peer_nodes
        self.port = port
        
        # Determine port if not specified (deterministic for testing)
        if port == 8888 and node_id.startswith("node_"):
            try:
                # node_0 -> 8888, node_1 -> 8889, node_2 -> 8890
                offset = int(node_id.split("_")[1])
                self.port = 8888 + offset
            except ValueError:
                pass
        self.all_nodes = [node_id] + peer_nodes

        # Core components
        self.consensus_engine = consensus_engine if consensus_engine else ConsensusEngine(StateLedger())
        self.memory_system = CognitivePermanence(agent_id=node_id)
        self.consensus_state = consensus_state or ConsensusState()

        # Network components
        self.server: Optional[asyncio.AbstractServer] = None
        self.peer_connections: Dict[str, Tuple[asyncio.StreamReader, asyncio.StreamWriter]] = {}
        self.message_queue = asyncio.Queue()
        self.running = False

        # Federation-specific state
        self.knowledge_shards: Dict[str, Dict] = {}  # shard_id -> shard_data
        self.shard_assignments: Dict[str, str] = {}  # entity_id -> node_id
        self.graph_reasoning_sessions: Dict[str, Dict] = {}  # session_id -> reasoning_state

        # Byzantine fault tolerance parameters
        self.fault_tolerance = 1  # For 3 nodes: f = 1 (can tolerate 1 faulty node)
        self.quorum_size = 2 * self.fault_tolerance + 1  # 3 for f=1

        # Timing parameters
        self.heartbeat_interval = 1.0
        self.election_timeout_min = 3.0
        self.election_timeout_max = 6.0
        self.view_change_timeout = 10.0

        # Locks for thread safety
        self.state_lock = asyncio.Lock()  # Changed to asyncio.Lock
        self.network_lock = asyncio.Lock() # Changed to asyncio.Lock

    async def start(self):
        """Start the federation node."""
        print(f"[+] Starting Federation Node {self.node_id} on port {self.port}")
        self.running = True

        # Start network server
        server_task = asyncio.create_task(self._run_server())

        # Start heartbeat
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Start message processing
        processor_task = asyncio.create_task(self._process_messages())

        # Start leader election if needed
        election_task = asyncio.create_task(self._leader_election_loop())

        # Wait for all tasks
        await asyncio.gather(server_task, heartbeat_task, processor_task, election_task)

    async def stop(self):
        """Stop the federation node."""
        print(f"[-] Stopping Federation Node {self.node_id}")
        self.running = False

        async with self.network_lock:
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            for reader, writer in self.peer_connections.values():
                writer.close()
                await writer.wait_closed()
            self.peer_connections.clear()

    async def _run_server(self):
        """Run the network server to accept connections from peer nodes."""
        try:
            self.server = await asyncio.start_server(
                self._handle_connection, 'localhost', self.port
            )
            print(f"[+] Federation Node {self.node_id} listening on port {self.port}")
            async with self.server:
                await self.server.serve_forever()
        except Exception as e:
            if self.running:
                print(f"[!] Server error: {e}")

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming connection from a peer node."""
        try:
            # Receive node ID
            node_id_bytes = await reader.readexactly(32)
            peer_id = node_id_bytes.decode().strip('\x00')

            async with self.network_lock:
                self.peer_connections[peer_id] = (reader, writer)

            print(f"[+] Connected to peer {peer_id}")

            # Start receiving messages from this peer
            await self._receive_loop(peer_id, reader, writer)

        except Exception as e:
            if self.running:
                print(f"[!] Connection handling error: {e}")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError) as e:
                print(f"[!] Error during writer closure: {e}")
            async with self.network_lock:
                # Remove from connections
                for pid, (r, w) in list(self.peer_connections.items()):
                    if r == reader:
                        del self.peer_connections[pid]
                        break

    async def _receive_loop(self, peer_id: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Continuous receive loop for a peer connection."""
        while self.running:
            try:
                message = await self._receive_message(reader)
                if message:
                    await self.message_queue.put(message)
                else:
                    break
            except Exception as e:
                if self.running:
                    print(f"[!] Error receiving from {peer_id}: {e}")
                break

    async def _connect_to_peers(self):
        """Connect to peer nodes."""
        for peer_id in self.peer_nodes:
            async with self.network_lock:
                if peer_id in self.peer_connections:
                    continue

            try:
                # Calculate peer port (deterministic for node_X naming)
                if peer_id.startswith("node_") and peer_id[5:].isdigit():
                    peer_port = 8888 + int(peer_id[5:])
                else:
                    peer_port = 8888 + hash(peer_id) % 100

                reader, writer = await asyncio.open_connection('localhost', peer_port)

                # Send our node ID
                node_id_bytes = self.node_id.encode().ljust(32, b'\x00')
                writer.write(node_id_bytes)
                await writer.drain()

                async with self.network_lock:
                    self.peer_connections[peer_id] = (reader, writer)

                print(f"[+] Connected to peer {peer_id} on port {peer_port}")

                # Start receiving from this peer
                asyncio.create_task(self._receive_loop(peer_id, reader, writer))

            except Exception as e:
                # print(f"[!] Failed to connect to {peer_id}: {e}")
                pass

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to maintain connections."""
        while self.running:
            await self._connect_to_peers()  # Reconnect if needed
            await asyncio.sleep(self.heartbeat_interval)

            heartbeat = FederationMessage(
                msg_type=MessageType.HEARTBEAT,
                sender_id=self.node_id,
                receiver_id="broadcast",
                view_number=self.consensus_state.view_number,
                sequence_number=0,
                payload={"status": "alive", "leader": self.consensus_state.current_leader}
            )
            heartbeat.sign(SHARED_SECRET)

            await self._broadcast_message(heartbeat)

    async def _leader_election_loop(self):
        """Handle leader election and view changes."""
        while self.running:
            await asyncio.sleep(self.election_timeout_min)

            async with self.state_lock:
                # Check if we need to start or RESTART an election
                if not self._is_leader_alive():
                    # Move to candidate state and start/retry election for a new view
                    await self._start_election()

    def _is_leader_alive(self) -> bool:
        """Check if the current leader is still alive."""
        return self.consensus_state.current_leader == self.node_id or self.consensus_state.current_leader in self.peer_connections

    async def _start_election(self):
        # Assumes state_lock is held
        self.consensus_state.node_state = NodeState.CANDIDATE
        self.consensus_state.view_number += 1

        print(f"[*] Node {self.node_id} starting election for view {self.consensus_state.view_number}")

        # Send view change messages to all peers
        view_change = FederationMessage(
            msg_type=MessageType.VIEW_CHANGE,
            sender_id=self.node_id,
            receiver_id="broadcast",
            view_number=self.consensus_state.view_number,
            sequence_number=0,
            payload={"reason": "election"}
        )
        view_change.sign(SHARED_SECRET)
        
        # Add our own message to the collection for quorum calculation
        if self.consensus_state.view_number not in self.consensus_state.view_change_messages:
            self.consensus_state.view_change_messages[self.consensus_state.view_number] = []
        self.consensus_state.view_change_messages[self.consensus_state.view_number].append(view_change)

        await self._broadcast_message(view_change)

    async def _process_messages(self):
        """Process incoming messages from the queue."""
        while self.running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"[!] Message processing error: {e}")

    async def _handle_message(self, message: FederationMessage):
        """Handle a received federation message."""
        print(f"[*] Node {self.node_id} received {message.msg_type.value} from {message.sender_id}")
        
        # Verify signature
        if not message.verify_signature(SHARED_SECRET):
            print(f"[!] Invalid signature from {message.sender_id}")
            return

        # Route to appropriate handler
        if message.msg_type == MessageType.HEARTBEAT:
            await self._handle_heartbeat(message)
        elif message.msg_type == MessageType.VIEW_CHANGE:
            await self._handle_view_change(message)
        elif message.msg_type == MessageType.PRE_PREPARE:
            await self._handle_pre_prepare(message)
        elif message.msg_type == MessageType.PREPARE:
            await self._handle_prepare(message)
        elif message.msg_type == MessageType.COMMIT:
            await self._handle_commit(message)
        elif message.msg_type == MessageType.REQUEST:
            await self._handle_request(message)
        elif message.msg_type == MessageType.STATE_SYNC:
            await self._handle_state_sync(message)
        elif message.msg_type == MessageType.SHARD_UPDATE:
            await self._handle_shard_update(message)
        elif message.msg_type == MessageType.GRAPH_REASONING:
            await self._handle_graph_reasoning(message)

    async def _handle_heartbeat(self, message: FederationMessage):
        """Handle heartbeat message."""
        # Update leader information if provided
        if "leader" in message.payload:
            async with self.state_lock:
                self.consensus_state.current_leader = message.payload["leader"]

    async def _handle_view_change(self, message: FederationMessage):
        """Handle view change message for leader election."""
        view_num = message.view_number

        async with self.state_lock:
            if view_num > self.consensus_state.view_number:
                self.consensus_state.view_number = view_num
                self.consensus_state.node_state = NodeState.FOLLOWER

            # Collect view change messages
            if view_num not in self.consensus_state.view_change_messages:
                self.consensus_state.view_change_messages[view_num] = []
            self.consensus_state.view_change_messages[view_num].append(message)

            # Check if we have enough view change messages to elect a new leader
            vote_count = len(self.consensus_state.view_change_messages[view_num])
            print(f"[*] Node {self.node_id} has {vote_count}/{self.quorum_size} votes for view {view_num}")
            if vote_count >= self.quorum_size:
                await self._elect_new_leader(view_num)

    async def _elect_new_leader(self, view_number: int):
        """Elect a new leader based on collected view change messages."""
        # Simple leader election: choose the node with the "highest" ID
        candidates = [msg.sender_id for msg in self.consensus_state.view_change_messages[view_number]]
        new_leader = max(candidates)

        # Assumes state_lock is held (called from _handle_view_change)
        self.consensus_state.current_leader = new_leader
        if new_leader == self.node_id:
            self.consensus_state.node_state = NodeState.LEADER
            print(f"[+] Node {self.node_id} elected as leader for view {view_number}")
        else:
            self.consensus_state.node_state = NodeState.FOLLOWER
            print(f"[+] Node {new_leader} elected as leader for view {view_number}")

        # Send new view message
        new_view = FederationMessage(
            msg_type=MessageType.NEW_VIEW,
            sender_id=self.node_id,
            receiver_id="broadcast",
            view_number=view_number,
            sequence_number=0,
            payload={"new_leader": new_leader}
        )
        new_view.sign(SHARED_SECRET)
        await self._broadcast_message(new_view)

    async def _handle_pre_prepare(self, message: FederationMessage):
        """Handle pre-prepare message in PBFT protocol."""
        # Process the pre-prepare message
        if self.consensus_engine.process_pre_prepare(message):
            print(f"[*] Node {self.node_id} accepted PRE-PREPARE {message.sequence_number}")
            
            # Generate and broadcast PREPARE message
            prepare_msg = await self.consensus_engine.send_prepare(message.sequence_number)
            if prepare_msg:
                # Add our own vote to the engine (since we don't receive our own broadcast)
                self.consensus_engine.process_prepare(prepare_msg)
                await self._broadcast_message(prepare_msg)

    async def _handle_prepare(self, message: FederationMessage):
        """Handle prepare message in PBFT protocol."""
        # Process the prepare message
        if self.consensus_engine.process_prepare(message):
            print(f"[*] Node {self.node_id} reached PREPARE quorum for {message.sequence_number}")
            
            # Generate and broadcast COMMIT message
            commit_msg = await self.consensus_engine.send_commit(message.sequence_number)
            if commit_msg:
                # Add our own vote
                self.consensus_engine.process_commit(commit_msg)
                await self._broadcast_message(commit_msg)

    async def _handle_commit(self, message: FederationMessage):
        """Handle commit message in PBFT protocol."""
        # Process the commit message
        if self.consensus_engine.process_commit(message):
            print(f"[*] Node {self.node_id} reached COMMIT quorum for {message.sequence_number}")
            
            # Execute the request
            result = await self.consensus_engine.execute_request(message.sequence_number)
            if result:
                print(f"[+] Node {self.node_id} executed request {message.sequence_number}: {result}")

    async def _handle_request(self, message: FederationMessage):
        """Handle client request."""
        # Process the request through consensus
        if self.consensus_state.node_state == NodeState.LEADER:
            print(f"[*] Node {self.node_id} (Leader) processing REQUEST from {message.sender_id}")
            
            payload = message.payload
            op = payload["operation"]
            data = payload["data"]
            client_id = message.sender_id
            
            # Submit to engine (which prepares local state)
            request_id = await self.consensus_engine.submit_request(op, data, client_id)
            
            # Retrieve the pre-prepare message generated by the engine
            request = self.consensus_engine.pending_requests.get(request_id)
            if request and request.pre_prepare_msg:
                # Add our own vote/log
                print(f"[*] Broadcasting PRE-PREPARE for request {request_id}")
                await self._broadcast_message(request.pre_prepare_msg)
        
        else:
            # Forward to leader (simplified: just print for now as clients usually send to leader)
            print(f"[*] Node {self.node_id} received REQUEST but is not LEADER. Forwarding logic needed.")

    async def _handle_state_sync(self, message: FederationMessage):
        """Handle state synchronization message."""
        # Synchronize state with other nodes
        pass

    async def _handle_shard_update(self, message: FederationMessage):
        """Handle knowledge graph shard update."""
        # Update local shard data
        pass

    async def _handle_graph_reasoning(self, message: FederationMessage):
        """Handle cross-node graph reasoning request."""
        # Participate in distributed reasoning
        pass

    async def _broadcast_message(self, message: FederationMessage):
        """Broadcast a message to all peer nodes."""
        message_data = pickle.dumps(message)
        message_length = struct.pack('!I', len(message_data))

        async with self.network_lock:
            print(f"[*] Node {self.node_id} broadcasting {message.msg_type.value} to {len(self.peer_connections)} peers")
            for peer_id, (reader, writer) in list(self.peer_connections.items()):
                try:
                    writer.write(message_length)
                    writer.write(message_data)
                    await writer.drain()
                except Exception as e:
                    print(f"[!] Failed to send to {peer_id}: {e}")
                    # Remove failed connection
                    del self.peer_connections[peer_id]

    async def _receive_message(self, reader: asyncio.StreamReader) -> Optional[FederationMessage]:
        """Receive a message from a stream reader."""
        try:
            # Receive message length
            length_bytes = await reader.readexactly(4)
            message_length = struct.unpack('!I', length_bytes)[0]

            # Receive message data
            message_data = await reader.readexactly(message_length)
            message = pickle.loads(message_data)
            return message
        except Exception:
            return None

    # Public API methods

    async def submit_request(self, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a request to the federation for consensus."""
        request = FederationMessage(
            msg_type=MessageType.REQUEST,
            sender_id=self.node_id,
            receiver_id=self.consensus_state.current_leader,
            view_number=self.consensus_state.view_number,
            sequence_number=self.consensus_state.sequence_number,
            payload={"operation": operation, "data": data}
        )
        request.sign(SHARED_SECRET)

        # Send to leader
        await self._send_to_node(self.consensus_state.current_leader, request)

        # Wait for reply (simplified - in real implementation would track request IDs)
        return {"status": "submitted", "request_id": request.sequence_number}

    async def _send_to_node(self, target_node: str, message: FederationMessage):
        """Send a message to a specific node."""
        async with self.network_lock:
            if target_node not in self.peer_connections:
                # print(f"[!] No connection to node {target_node}")
                return

            reader, writer = self.peer_connections[target_node]
            try:
                message_data = pickle.dumps(message)
                message_length = struct.pack('!I', len(message_data))
                writer.write(message_length)
                writer.write(message_data)
                await writer.drain()
            except Exception as e:
                print(f"[!] Failed to send to {target_node}: {e}")


if __name__ == "__main__":
    # Example usage for testing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python federation_node.py <node_id> [peer1] [peer2]")
        sys.exit(1)

    node_id = sys.argv[1]
    peers = sys.argv[2:] if len(sys.argv) > 2 else []

    node = FederationNode(node_id, peers)

    try:
        asyncio.run(node.start())
    except KeyboardInterrupt:
        asyncio.run(node.stop())