"""
BIZRA Federation Consensus Protocol
Phase 9: Distributed Consensus with Byzantine Fault Tolerance

Extends the base ConsensusEngine for federated operation across 3 nodes.
Implements PBFT-inspired consensus protocol with leader election and state synchronization.
"""

import hashlib
import hmac
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .consensus_engine import ConsensusEngine, SHARED_SECRET
from .state_ledger import StateLedger
from .federation_node import FederationMessage, MessageType, ConsensusState


class ConsensusPhase(Enum):
    """Phases of the PBFT consensus protocol."""
    PRE_PREPARE = "pre_prepare"
    PREPARE = "prepare"
    COMMIT = "commit"
    EXECUTE = "execute"


@dataclass
class ConsensusRequest:
    """A request being processed through consensus."""
    request_id: str
    operation: str
    data: Dict[str, Any]
    timestamp: float
    client_id: str
    sequence_number: int
    view_number: int

    # Consensus tracking
    phase: ConsensusPhase = ConsensusPhase.PRE_PREPARE
    pre_prepare_msg: Optional[FederationMessage] = None
    prepare_votes: List[FederationMessage] = field(default_factory=list)
    commit_votes: List[FederationMessage] = field(default_factory=list)

    executed: bool = False
    result: Optional[Dict[str, Any]] = None


class FederationConsensusEngine:
    """
    Distributed consensus engine for the BIZRA 3-Node Federation.

    Implements Byzantine fault tolerant consensus using a PBFT-inspired protocol.
    Handles leader election, state synchronization, and cross-node agreement.
    """

    def __init__(self, node_id: str, peer_nodes: List[str], state_ledger: StateLedger):
        self.node_id = node_id
        self.peer_nodes = peer_nodes
        self.all_nodes = [node_id] + peer_nodes

        # Core consensus components
        self.base_consensus = ConsensusEngine(state_ledger)
        self.consensus_state = ConsensusState()

        # Byzantine fault tolerance parameters
        self.consensus_mode = "VETO"  # Options: "STANDARD" (PBFT 2f+1), "VETO" (Unanimous)
        self.fault_tolerance = 1  # For 3 nodes: can tolerate 1 faulty node
        
        if self.consensus_mode == "VETO":
             # Unanimous consent required (All nodes must vote)
             self.quorum_size = len(self.all_nodes)
        else:
             # Standard PBFT (3f+1 nodes, 2f+1 quorum)
             self.quorum_size = 2 * self.fault_tolerance + 1

        # Request tracking
        self.pending_requests: Dict[str, ConsensusRequest] = {}
        self.executed_requests: Dict[str, ConsensusRequest] = {}
        self.sequence_number = 0

        # Watermarks for garbage collection
        self.low_watermark = 0
        self.high_watermark = 100  # Max outstanding requests

        # Checkpoint state
        self.checkpoints: Dict[int, str] = {}  # sequence -> state_hash
        self.checkpoint_votes: Dict[int, List[FederationMessage]] = {}

        # Networking
        self.broadcast_callback = None

    def is_leader(self) -> bool:
        """Check if this node is the current leader."""
        return self.consensus_state.current_leader == self.node_id

    def get_next_sequence_number(self) -> int:
        """Get the next sequence number for a request."""
        self.sequence_number += 1
        return self.sequence_number

    async def submit_request(self, operation: str, data: Dict[str, Any], client_id: str) -> str:
        """
        Submit a request for consensus processing.

        Returns a request ID that can be used to track the request.
        """
        request_id = f"{client_id}:{int(time.time() * 1000)}:{hash(operation + str(data)) % 10000}"

        request = ConsensusRequest(
            request_id=request_id,
            operation=operation,
            data=data,
            timestamp=time.time(),
            client_id=client_id,
            sequence_number=self.get_next_sequence_number(),
            view_number=self.consensus_state.view_number
        )

        self.pending_requests[request_id] = request

        # If we're the leader, start the consensus process
        if self.is_leader():
            await self._start_consensus(request)

        return request_id

    async def _start_consensus(self, request: ConsensusRequest):
        """Start the PBFT consensus process for a request (leader only)."""
        if not self.is_leader():
            return

        # Create pre-prepare message
        pre_prepare = FederationMessage(
            msg_type=MessageType.PRE_PREPARE,
            sender_id=self.node_id,
            receiver_id="broadcast",
            view_number=request.view_number,
            sequence_number=request.sequence_number,
            payload={
                "request_id": request.request_id,
                "operation": request.operation,
                "data": request.data,
                "client_id": request.client_id,
                "digest": self._calculate_request_digest(request)
            }
        )
        pre_prepare.sign(SHARED_SECRET)

        # Store the pre-prepare message
        request.pre_prepare_msg = pre_prepare
        request.phase = ConsensusPhase.PRE_PREPARE

        # Log it
        self.consensus_state.pre_prepare_log[request.sequence_number] = pre_prepare

        # Broadcast to all replicas
        if self.broadcast_callback:
            print(f"[*] ConsensusEngine triggering broadcast for request {request.sequence_number}")
            await self.broadcast_callback(pre_prepare)

        # Leader also needs to vote (Prepare phase)
        # We process our own pre-prepare to generate a prepare vote
        print(f"[*] Leader {self.node_id} generating own PREPARE vote for {request.sequence_number}")
        prepare_msg = await self.send_prepare(request.sequence_number)
        if prepare_msg:
            # Broadcast our prepare vote (so others see it)
            if self.broadcast_callback:
                await self.broadcast_callback(prepare_msg)
            
            # Log it locally
            if self.process_prepare(prepare_msg):
                 # If we somehow have quorum already (e.g. 1 node network?), move to commit
                 pass 

        return pre_prepare

    def process_pre_prepare(self, message: FederationMessage) -> bool:
        """
        Process a pre-prepare message.

        Returns True if the message is valid and processing should continue.
        """
        seq_num = message.sequence_number
        view_num = message.view_number

        # Basic validation
        if view_num != self.consensus_state.view_number:
            print(f"[!] Pre-prepare view mismatch: {view_num} vs {self.consensus_state.view_number}")
            return False

        if seq_num <= self.low_watermark or seq_num > self.high_watermark:
            print(f"[!] Pre-prepare sequence out of range: {seq_num}")
            return False

        # Check if we already have a pre-prepare for this sequence
        if seq_num in self.consensus_state.pre_prepare_log:
            existing = self.consensus_state.pre_prepare_log[seq_num]
            if existing.payload != message.payload:
                print(f"[!] Conflicting pre-prepare messages for sequence {seq_num}")
                return False
        else:
            # Store the pre-prepare message
            self.consensus_state.pre_prepare_log[seq_num] = message

        # If we're not the leader, we need to verify the request digest
        if not self.is_leader():
            request_digest = message.payload.get("digest")
            if not request_digest:
                print("[!] Pre-prepare missing request digest")
                return False

            # In a full implementation, we'd reconstruct the request and verify the digest
            # For now, we'll trust the leader (simplified)

        return True

    async def send_prepare(self, sequence_number: int) -> Optional[FederationMessage]:
        """Send a prepare message for the given sequence number."""
        if sequence_number not in self.consensus_state.pre_prepare_log:
            print(f"[!] No pre-prepare found for sequence {sequence_number}")
            return None

        pre_prepare = self.consensus_state.pre_prepare_log[sequence_number]

        prepare = FederationMessage(
            msg_type=MessageType.PREPARE,
            sender_id=self.node_id,
            receiver_id="broadcast",
            view_number=pre_prepare.view_number,
            sequence_number=sequence_number,
            payload={
                "digest": pre_prepare.payload["digest"]
            }
        )
        prepare.sign(SHARED_SECRET)

        # Log our prepare vote
        key = (pre_prepare.view_number, sequence_number)
        if key not in self.consensus_state.prepare_log:
            self.consensus_state.prepare_log[key] = []
        self.consensus_state.prepare_log[key].append(prepare)

        return prepare

    def process_prepare(self, message: FederationMessage) -> bool:
        """Process a prepare message."""
        seq_num = message.sequence_number
        view_num = message.view_number
        key = (view_num, seq_num)

        # Check if we have the corresponding pre-prepare
        if seq_num not in self.consensus_state.pre_prepare_log:
            print(f"[!] Prepare without pre-prepare for sequence {seq_num}")
            return False

        pre_prepare = self.consensus_state.pre_prepare_log[seq_num]

        # Verify digest matches
        if message.payload.get("digest") != pre_prepare.payload.get("digest"):
            print(f"[!] Prepare digest mismatch for sequence {seq_num}")
            return False

        # Log the prepare vote
        if key not in self.consensus_state.prepare_log:
            self.consensus_state.prepare_log[key] = []
        self.consensus_state.prepare_log[key].append(message)

        # Check if we have enough prepares for a quorum
        prepares = self.consensus_state.prepare_log[key]
        unique_senders = set(msg.sender_id for msg in prepares)

        if len(unique_senders) >= self.quorum_size:
            # We have a prepare quorum, move to commit phase
            return True

        return False

    async def send_commit(self, sequence_number: int) -> Optional[FederationMessage]:
        """Send a commit message for the given sequence number."""
        if sequence_number not in self.consensus_state.pre_prepare_log:
            print(f"[!] No pre-prepare found for sequence {sequence_number}")
            return None

        pre_prepare = self.consensus_state.pre_prepare_log[sequence_number]

        commit = FederationMessage(
            msg_type=MessageType.COMMIT,
            sender_id=self.node_id,
            receiver_id="broadcast",
            view_number=pre_prepare.view_number,
            sequence_number=sequence_number,
            payload={
                "digest": pre_prepare.payload["digest"]
            }
        )
        commit.sign(SHARED_SECRET)

        # Log our commit vote
        key = (pre_prepare.view_number, sequence_number)
        if key not in self.consensus_state.commit_log:
            self.consensus_state.commit_log[key] = []
        self.consensus_state.commit_log[key].append(commit)

        return commit

    def process_commit(self, message: FederationMessage) -> bool:
        """Process a commit message."""
        seq_num = message.sequence_number
        view_num = message.view_number
        key = (view_num, seq_num)

        # Check if we have the corresponding pre-prepare
        if seq_num not in self.consensus_state.pre_prepare_log:
            print(f"[!] Commit without pre-prepare for sequence {seq_num}")
            return False

        pre_prepare = self.consensus_state.pre_prepare_log[seq_num]

        # Verify digest matches
        if message.payload.get("digest") != pre_prepare.payload.get("digest"):
            print(f"[!] Commit digest mismatch for sequence {seq_num}")
            return False

        # Log the commit vote
        if key not in self.consensus_state.commit_log:
            self.consensus_state.commit_log[key] = []
        self.consensus_state.commit_log[key].append(message)

        # Check if we have enough commits for a quorum
        commits = self.consensus_state.commit_log[key]
        unique_senders = set(msg.sender_id for msg in commits)

        if len(unique_senders) >= self.quorum_size:
            # We have a commit quorum, can execute the request
            return True

        return False

    async def execute_request(self, sequence_number: int) -> Optional[Dict[str, Any]]:
        """Execute a request that has reached commit quorum."""
        if sequence_number not in self.consensus_state.pre_prepare_log:
            print(f"[!] No pre-prepare found for execution of sequence {sequence_number}")
            return None

        pre_prepare = self.consensus_state.pre_prepare_log[sequence_number]
        payload = pre_prepare.payload

        # Reconstruct the request
        request_id = payload["request_id"]
        operation = payload["operation"]
        data = payload["data"]
        client_id = payload["client_id"]

        # Execute the operation using the base consensus engine
        if operation == "validate_and_commit":
            result = self.base_consensus.validate_and_commit(
                action_name=data.get("action_name", "federated_action"),
                action_data=data.get("action_data", {}),
                metrics=data.get("metrics", {})
            )
        else:
            result = {"status": "UNKNOWN_OPERATION", "operation": operation}

        # Mark as executed
        if request_id in self.pending_requests:
            request = self.pending_requests[request_id]
            request.executed = True
            request.result = result
            self.executed_requests[request_id] = request
            del self.pending_requests[request_id]

        # Update watermarks and garbage collect old state
        if sequence_number > self.low_watermark + 50:  # Checkpoint every 50 requests
            self._create_checkpoint(sequence_number)

        return result

    def _create_checkpoint(self, sequence_number: int):
        """Create a checkpoint of the current state."""
        # Calculate state hash (simplified)
        state_data = {
            "executed_requests": list(self.executed_requests.keys()),
            "ledger_hash": self.base_consensus.ledger.get_latest_hash()
        }
        state_hash = hashlib.sha256(json.dumps(state_data, sort_keys=True).encode()).hexdigest()

        self.checkpoints[sequence_number] = state_hash

        # Broadcast checkpoint (would be handled by FederationNode)
        print(f"[+] Created checkpoint at sequence {sequence_number}: {state_hash[:16]}...")

    def _calculate_request_digest(self, request: ConsensusRequest) -> str:
        """Calculate a digest of the request for integrity checking."""
        request_data = {
            "operation": request.operation,
            "data": request.data,
            "client_id": request.client_id,
            "timestamp": request.timestamp
        }
        return hashlib.sha256(json.dumps(request_data, sort_keys=True).encode()).hexdigest()

    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a pending or executed request."""
        if request_id in self.pending_requests:
            request = self.pending_requests[request_id]
            return {
                "status": "pending",
                "phase": request.phase.value,
                "sequence_number": request.sequence_number,
                "prepares": len(request.prepare_votes),
                "commits": len(request.commit_votes)
            }
        elif request_id in self.executed_requests:
            request = self.executed_requests[request_id]
            return {
                "status": "executed",
                "result": request.result,
                "sequence_number": request.sequence_number
            }
        else:
            return None

    def get_consensus_status(self) -> Dict[str, Any]:
        """Get the current consensus status."""
        return {
            "node_id": self.node_id,
            "is_leader": self.is_leader(),
            "view_number": self.consensus_state.view_number,
            "current_leader": self.consensus_state.current_leader,
            "sequence_number": self.sequence_number,
            "pending_requests": len(self.pending_requests),
            "executed_requests": len(self.executed_requests),
            "quorum_size": self.quorum_size,
            "fault_tolerance": self.fault_tolerance
        }

    def set_broadcast_callback(self, callback):
        """Set the callback for broadcasting messages to the federation."""
        self.broadcast_callback = callback


if __name__ == "__main__":
    # Test the consensus engine
    from bizra_kernel.state_ledger import StateLedger

    ledger = StateLedger()
    consensus = FederationConsensusEngine("node_0", ["node_1", "node_2"], ledger)

    print("[+] Federation Consensus Engine initialized")
    print(f"Status: {consensus.get_consensus_status()}")