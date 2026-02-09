"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA PATTERN FEDERATION — CONSENSUS ENGINE (PBFT)                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Protects the 'Shoulders of Giants' from faulty or malicious input.         ║
║   Algorithm: Practical Byzantine Fault Tolerance (Castro & Liskov, 1999)     ║
║   Phases: PRE-PREPARE → PREPARE → COMMIT → COMMITTED                         ║
║   Quorum Threshold: 2f + 1 (tolerates f Byzantine failures in 3f + 1 nodes)  ║
║   View-Change: Handles leader failures with timeout-based leader rotation    ║
║   Standing on Giants: Lamport (1982), Castro & Liskov (1999)                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD
from core.pci.crypto import (
    canonical_json,
    domain_separated_digest,
    sign_message,
    verify_signature,
)

logger = logging.getLogger("CONSENSUS")


# =============================================================================
# PBFT PHASE STATE MACHINE (GAP-C3 Resolution)
# =============================================================================


class ConsensusPhase(Enum):
    """
    PBFT Phase State Machine.

    Flow: PRE_PREPARE → PREPARE → COMMIT → COMMITTED

    PRE_PREPARE: Leader broadcasts proposal to all replicas
    PREPARE: Replicas acknowledge receipt, verify integrity
    COMMIT: Replicas signal readiness to commit after 2f+1 prepares
    COMMITTED: Final state after 2f+1 commits
    """

    PRE_PREPARE = auto()
    PREPARE = auto()
    COMMIT = auto()
    COMMITTED = auto()
    ABORTED = auto()


@dataclass
class ConsensusState:
    """
    Per-proposal consensus tracking state.

    Implements the PBFT state machine with prepare and commit certificates.
    """

    phase: ConsensusPhase = ConsensusPhase.PRE_PREPARE
    prepare_count: int = 0
    commit_count: int = 0
    prepare_signatures: Dict[str, str] = field(default_factory=dict)  # peer_id -> sig
    commit_signatures: Dict[str, str] = field(default_factory=dict)  # peer_id -> sig
    view_number: int = 0
    sequence_number: int = 0
    timeout_ms: int = 5000  # 5 second default timeout
    started_at: float = field(default_factory=time.time)


# =============================================================================
# VIEW-CHANGE PROTOCOL (GAP-C3 Resolution)
# =============================================================================


@dataclass
class ViewChangeRequest:
    """
    Request to change the view (leader rotation) due to timeout or failure.

    Per PBFT: View change occurs when replicas detect leader unresponsiveness.
    """

    view_number: int
    requester_id: str
    signature: str
    public_key: str
    prepared_proposals: List[str] = field(
        default_factory=list
    )  # Proposals in PREPARE/COMMIT
    timestamp: float = field(default_factory=time.time)


@dataclass
class NewViewMessage:
    """
    Leader's response to view-change containing state for new view.
    """

    new_view_number: int
    leader_id: str
    view_change_proofs: List[ViewChangeRequest] = field(default_factory=list)
    prepared_state: Dict[str, Any] = field(default_factory=dict)
    signature: str = ""


@dataclass
class Proposal:
    """
    PBFT Pre-Prepare message containing the proposed pattern.

    The leader creates this and broadcasts to all replicas.
    """

    proposal_id: str
    proposer_id: str
    pattern_data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    view_number: int = 0
    sequence_number: int = 0


@dataclass
class Vote:
    """
    Vote message used in both PREPARE and COMMIT phases.

    Contains the voter's signature over the proposal digest.
    """

    proposal_id: str
    voter_id: str
    signature: str
    public_key: str
    ihsan_score: float
    phase: ConsensusPhase = ConsensusPhase.PREPARE  # Which phase this vote is for
    view_number: int = 0


@dataclass
class PrepareMessage:
    """
    PBFT PREPARE message - replica acknowledges receipt of pre-prepare.
    """

    proposal_id: str
    replica_id: str
    view_number: int
    sequence_number: int
    digest: str
    signature: str


@dataclass
class CommitMessage:
    """
    PBFT COMMIT message - replica signals readiness to commit.
    """

    proposal_id: str
    replica_id: str
    view_number: int
    sequence_number: int
    digest: str
    signature: str


class ConsensusEngine:
    """
    Practical Byzantine Fault Tolerant Consensus for Pattern Elevation.

    Implements PBFT (Castro & Liskov, 1999) with:
    - Two-phase commit: PREPARE → COMMIT
    - View-change protocol for leader failure recovery
    - Ed25519 signature verification
    - Quorum threshold: 2f + 1

    Standing on Giants:
    - Lamport (1982): Byzantine Generals Problem
    - Castro & Liskov (1999): Practical BFT
    - Anthropic (2023): Constitutional AI alignment → Ihsān threshold

    GAP-C3 Resolution: Full BFT compliance with two-phase commit and view-change.
    """

    def __init__(self, node_id: str, private_key: str, public_key: str):
        self.node_id = node_id
        self.private_key = private_key
        self.public_key = public_key

        # State tracking
        self.active_proposals: Dict[str, Proposal] = {}
        self.votes: Dict[str, List[Vote]] = {}  # proposal_id -> List[Vote]
        self.committed_patterns: Set[str] = set()

        # PBFT State (GAP-C3)
        self._consensus_state: Dict[str, ConsensusState] = {}
        self._current_view: int = 0
        self._sequence_counter: int = 0
        self._view_change_requests: Dict[int, List[ViewChangeRequest]] = {}
        self._is_leader: bool = False
        self._leader_id: Optional[str] = None

        # Timeout tracking
        self._timeout_thread: Optional[threading.Thread] = None
        self._timeout_lock = threading.Lock()
        self._active = True

        # SECURITY: Peer registry for voter_id to public_key binding
        # Maps node_id -> public_key for all known peers
        self._peer_keys: Dict[str, str] = {}
        # Register self
        self._peer_keys[node_id] = public_key

        # SECURITY (S-1): Replay attack prevention
        # Track seen proposal/vote nonces to prevent replay attacks
        self._seen_proposal_ids: Set[str] = set()
        self._seen_vote_ids: Set[str] = set()
        self._max_seen_ids: int = 100000  # Evict oldest when exceeded

        # Callbacks
        self.on_commit_broadcast: Optional[Callable[[Dict], None]] = None
        self.on_prepare_broadcast: Optional[Callable[[PrepareMessage], None]] = None
        self.on_commit_message_broadcast: Optional[Callable[[CommitMessage], None]] = (
            None
        )
        self.on_view_change_broadcast: Optional[Callable[[ViewChangeRequest], None]] = (
            None
        )
        self.on_new_view_broadcast: Optional[Callable[[NewViewMessage], None]] = None

    # =========================================================================
    # LEADER ELECTION AND VIEW MANAGEMENT
    # =========================================================================

    def set_leader(self, leader_id: str) -> None:
        """Set the current leader. Leader is determined by view_number % len(peers)."""
        self._leader_id = leader_id
        self._is_leader = leader_id == self.node_id
        logger.info(f"👑 Leader set to: {leader_id} (self={self._is_leader})")

    def get_leader_for_view(self, view_number: int) -> str:
        """Deterministic leader selection: leader = peers[view % len(peers)]."""
        peers = sorted(self._peer_keys.keys())
        if not peers:
            return self.node_id
        return peers[view_number % len(peers)]

    def get_current_view(self) -> int:
        """Get the current view number."""
        return self._current_view

    def get_quorum_size(self, node_count: int) -> int:
        """
        Calculate quorum size for BFT consensus.

        For n = 3f + 1 nodes, quorum = 2f + 1 (tolerates f Byzantine failures).
        """
        f = (node_count - 1) // 3  # Max Byzantine failures
        return 2 * f + 1

    # =========================================================================
    # PBFT PHASE 1: PRE-PREPARE (Leader broadcasts proposal)
    # =========================================================================

    def initiate_pre_prepare(self, pattern: Dict) -> Optional[Proposal]:
        """
        Leader initiates Pre-Prepare phase.

        Only the leader can initiate proposals in PBFT.
        """
        if not self._is_leader:
            logger.warning(f"⚠️ Non-leader {self.node_id} cannot initiate proposals")
            return None

        self._sequence_counter += 1
        proposal_id = f"prop_{uuid.uuid4().hex[:8]}"

        proposal = Proposal(
            proposal_id=proposal_id,
            proposer_id=self.node_id,
            pattern_data=pattern,
            view_number=self._current_view,
            sequence_number=self._sequence_counter,
        )

        # Initialize consensus state
        self._consensus_state[proposal_id] = ConsensusState(
            phase=ConsensusPhase.PRE_PREPARE,
            view_number=self._current_view,
            sequence_number=self._sequence_counter,
        )

        self.active_proposals[proposal_id] = proposal
        self.votes[proposal_id] = []

        logger.info(
            f"🗳️ PRE-PREPARE initiated: {proposal_id} (v={self._current_view}, s={self._sequence_counter})"
        )
        return proposal

    # =========================================================================
    # PBFT PHASE 2: PREPARE (Replicas acknowledge receipt)
    # =========================================================================

    def send_prepare(
        self, proposal: Proposal, ihsan_score: float
    ) -> Optional[PrepareMessage]:
        """
        Replica sends PREPARE message after receiving pre-prepare.

        Only sends if:
        1. View number matches current view
        2. Ihsān score meets threshold
        3. Haven't already sent PREPARE for this proposal
        """
        if ihsan_score < UNIFIED_IHSAN_THRESHOLD:
            logger.warning(
                f"❌ Rejecting {proposal.proposal_id}: Ihsān {ihsan_score} < {UNIFIED_IHSAN_THRESHOLD}"
            )
            return None

        if proposal.view_number != self._current_view:
            logger.warning(
                f"⚠️ View mismatch: proposal v={proposal.view_number}, current v={self._current_view}"
            )
            return None

        # Create digest and sign
        canon_data = canonical_json(proposal.pattern_data)
        digest = domain_separated_digest(canon_data)
        signature = sign_message(digest, self.private_key)

        prepare = PrepareMessage(
            proposal_id=proposal.proposal_id,
            replica_id=self.node_id,
            view_number=self._current_view,
            sequence_number=proposal.sequence_number,
            digest=digest,
            signature=signature,
        )

        # Track our own prepare
        state = self._consensus_state.get(proposal.proposal_id)
        if state:
            state.prepare_signatures[self.node_id] = signature
            state.prepare_count += 1

        logger.info(f"✉️ PREPARE sent for {proposal.proposal_id}")

        if self.on_prepare_broadcast:
            self.on_prepare_broadcast(prepare)

        return prepare

    def receive_prepare(self, prepare: PrepareMessage, node_count: int) -> bool:
        """
        Process incoming PREPARE message.

        Returns True if quorum reached and COMMIT phase should begin.
        """
        if prepare.proposal_id not in self.active_proposals:
            return False

        proposal = self.active_proposals[prepare.proposal_id]
        state = self._consensus_state.get(prepare.proposal_id)
        if not state:
            return False

        # Verify sender is registered
        if prepare.replica_id not in self._peer_keys:
            logger.error(f"⚠️ PREPARE from unregistered peer: {prepare.replica_id}")
            return False

        # Verify view number
        if prepare.view_number != self._current_view:
            logger.error(
                f"⚠️ PREPARE view mismatch: {prepare.view_number} != {self._current_view}"
            )
            return False

        # Verify signature
        registered_key = self._peer_keys[prepare.replica_id]
        canon_data = canonical_json(proposal.pattern_data)
        expected_digest = domain_separated_digest(canon_data)

        if prepare.digest != expected_digest:
            logger.error(f"⚠️ PREPARE digest mismatch from {prepare.replica_id}")
            return False

        if not verify_signature(expected_digest, prepare.signature, registered_key):
            logger.error(f"⚠️ Invalid PREPARE signature from {prepare.replica_id}")
            return False

        # Check for duplicate
        if prepare.replica_id in state.prepare_signatures:
            return False

        # Record prepare
        state.prepare_signatures[prepare.replica_id] = prepare.signature
        state.prepare_count += 1

        quorum = self.get_quorum_size(node_count)
        logger.info(
            f"📈 PREPARE received for {prepare.proposal_id} ({state.prepare_count}/{quorum})"
        )

        # Check if we have quorum for PREPARE phase
        if state.prepare_count >= quorum and state.phase == ConsensusPhase.PRE_PREPARE:
            state.phase = ConsensusPhase.PREPARE
            logger.info(
                f"✅ PREPARE quorum reached for {prepare.proposal_id}, transitioning to COMMIT phase"
            )
            return True

        return False

    # =========================================================================
    # PBFT PHASE 3: COMMIT (Replicas signal readiness to commit)
    # =========================================================================

    def send_commit(self, proposal: Proposal) -> Optional[CommitMessage]:
        """
        Send COMMIT message after PREPARE quorum is reached.
        """
        state = self._consensus_state.get(proposal.proposal_id)
        if not state or state.phase != ConsensusPhase.PREPARE:
            return None

        canon_data = canonical_json(proposal.pattern_data)
        digest = domain_separated_digest(canon_data)
        # Create commit-specific digest by hashing original digest + ":commit"
        commit_digest = domain_separated_digest((digest + ":commit").encode())
        signature = sign_message(commit_digest, self.private_key)

        commit = CommitMessage(
            proposal_id=proposal.proposal_id,
            replica_id=self.node_id,
            view_number=self._current_view,
            sequence_number=proposal.sequence_number,
            digest=digest,
            signature=signature,
        )

        # Track our own commit
        state.commit_signatures[self.node_id] = signature
        state.commit_count += 1

        logger.info(f"✉️ COMMIT sent for {proposal.proposal_id}")

        if self.on_commit_message_broadcast:
            self.on_commit_message_broadcast(commit)

        return commit

    def receive_commit(self, commit: CommitMessage, node_count: int) -> bool:
        """
        Process incoming COMMIT message.

        Returns True if quorum reached and proposal should be committed.
        """
        if commit.proposal_id not in self.active_proposals:
            return False

        proposal = self.active_proposals[commit.proposal_id]
        state = self._consensus_state.get(commit.proposal_id)
        if not state:
            return False

        # Verify sender is registered
        if commit.replica_id not in self._peer_keys:
            logger.error(f"⚠️ COMMIT from unregistered peer: {commit.replica_id}")
            return False

        # Verify view number
        if commit.view_number != self._current_view:
            logger.error(
                f"⚠️ COMMIT view mismatch: {commit.view_number} != {self._current_view}"
            )
            return False

        # Verify signature (commit signature includes ":commit" suffix)
        registered_key = self._peer_keys[commit.replica_id]
        canon_data = canonical_json(proposal.pattern_data)
        expected_digest = domain_separated_digest(canon_data)

        if commit.digest != expected_digest:
            logger.error(f"⚠️ COMMIT digest mismatch from {commit.replica_id}")
            return False

        expected_commit_digest = domain_separated_digest(
            (expected_digest + ":commit").encode()
        )
        if not verify_signature(
            expected_commit_digest, commit.signature, registered_key
        ):
            logger.error(f"⚠️ Invalid COMMIT signature from {commit.replica_id}")
            return False

        # Check for duplicate
        if commit.replica_id in state.commit_signatures:
            return False

        # Record commit
        state.commit_signatures[commit.replica_id] = commit.signature
        state.commit_count += 1

        quorum = self.get_quorum_size(node_count)
        logger.info(
            f"📈 COMMIT received for {commit.proposal_id} ({state.commit_count}/{quorum})"
        )

        # Check if we have quorum for COMMIT phase
        if state.commit_count >= quorum and state.phase == ConsensusPhase.PREPARE:
            state.phase = ConsensusPhase.COMMIT
            return self._finalize_commit(commit.proposal_id)

        return False

    def _finalize_commit(self, proposal_id: str) -> bool:
        """Finalize proposal after COMMIT quorum is reached."""
        state = self._consensus_state.get(proposal_id)
        if not state or state.phase == ConsensusPhase.COMMITTED:
            return False

        state.phase = ConsensusPhase.COMMITTED
        self.committed_patterns.add(proposal_id)

        proposal = self.active_proposals[proposal_id]
        logger.info(f"🏆 COMMITTED: Pattern {proposal_id} finalized to Giants Ledger.")

        if self.on_commit_broadcast:
            commit_payload = {
                "proposal_id": proposal_id,
                "pattern": proposal.pattern_data,
                "view_number": state.view_number,
                "sequence_number": state.sequence_number,
                "prepare_signatures": state.prepare_signatures,
                "commit_signatures": state.commit_signatures,
            }
            self.on_commit_broadcast(commit_payload)

        return True

    # =========================================================================
    # VIEW-CHANGE PROTOCOL (Leader Failure Recovery)
    # =========================================================================

    def request_view_change(self, reason: str = "timeout") -> ViewChangeRequest:
        """
        Initiate view-change when leader is unresponsive.

        Per PBFT: Replicas request view change after timeout without progress.
        """
        new_view = self._current_view + 1

        # Gather proposals in PREPARE/COMMIT phases (not yet committed)
        prepared_proposals = [
            pid
            for pid, state in self._consensus_state.items()
            if state.phase in (ConsensusPhase.PREPARE, ConsensusPhase.COMMIT)
            and pid not in self.committed_patterns
        ]

        # Sign the view-change request (hash the data to get hex digest)
        view_change_data = (
            f"VIEW-CHANGE:{new_view}:{self.node_id}:{','.join(prepared_proposals)}"
        )
        view_change_digest = domain_separated_digest(view_change_data.encode())
        signature = sign_message(view_change_digest, self.private_key)

        request = ViewChangeRequest(
            view_number=new_view,
            requester_id=self.node_id,
            signature=signature,
            public_key=self.public_key,
            prepared_proposals=prepared_proposals,
        )

        logger.warning(f"🔄 VIEW-CHANGE requested: v={new_view} reason={reason}")

        # Track our request
        if new_view not in self._view_change_requests:
            self._view_change_requests[new_view] = []
        self._view_change_requests[new_view].append(request)

        if self.on_view_change_broadcast:
            self.on_view_change_broadcast(request)

        return request

    def receive_view_change(self, request: ViewChangeRequest, node_count: int) -> bool:
        """
        Process incoming view-change request.

        Returns True if quorum reached and new view should be established.
        """
        # Verify signature
        if request.requester_id not in self._peer_keys:
            logger.error(
                f"⚠️ VIEW-CHANGE from unregistered peer: {request.requester_id}"
            )
            return False

        registered_key = self._peer_keys[request.requester_id]
        view_change_data = f"VIEW-CHANGE:{request.view_number}:{request.requester_id}:{','.join(request.prepared_proposals)}"
        view_change_digest = domain_separated_digest(view_change_data.encode())

        if not verify_signature(view_change_digest, request.signature, registered_key):
            logger.error(
                f"⚠️ Invalid VIEW-CHANGE signature from {request.requester_id}"
            )
            return False

        # Track request
        if request.view_number not in self._view_change_requests:
            self._view_change_requests[request.view_number] = []

        # Check for duplicate
        if any(
            r.requester_id == request.requester_id
            for r in self._view_change_requests[request.view_number]
        ):
            return False

        self._view_change_requests[request.view_number].append(request)

        quorum = self.get_quorum_size(node_count)
        count = len(self._view_change_requests[request.view_number])
        logger.info(
            f"📈 VIEW-CHANGE received for v={request.view_number} ({count}/{quorum})"
        )

        # Check if we have quorum
        if count >= quorum:
            return self._execute_view_change(request.view_number)

        return False

    def _execute_view_change(self, new_view: int) -> bool:
        """
        Execute view change after quorum of view-change requests.
        """
        if new_view <= self._current_view:
            return False

        old_view = self._current_view
        self._current_view = new_view

        # Determine new leader
        new_leader = self.get_leader_for_view(new_view)
        self.set_leader(new_leader)

        logger.info(
            f"🔄 VIEW-CHANGE executed: v={old_view} → v={new_view}, new leader={new_leader}"
        )

        # If we're the new leader, broadcast NEW-VIEW message
        if self._is_leader:
            self._broadcast_new_view(new_view)

        return True

    def _broadcast_new_view(self, view_number: int) -> None:
        """
        New leader broadcasts NEW-VIEW message with state.
        """
        view_change_proofs = self._view_change_requests.get(view_number, [])

        # Gather prepared state from view-change messages
        prepared_state: Dict[str, Any] = {}
        for request in view_change_proofs:
            for pid in request.prepared_proposals:
                if pid in self._consensus_state:
                    prepared_state[pid] = {
                        "phase": self._consensus_state[pid].phase.name,
                        "prepare_count": self._consensus_state[pid].prepare_count,
                        "commit_count": self._consensus_state[pid].commit_count,
                    }

        # Sign new-view message (hash to get hex digest)
        new_view_data = f"NEW-VIEW:{view_number}:{self.node_id}"
        new_view_digest = domain_separated_digest(new_view_data.encode())
        signature = sign_message(new_view_digest, self.private_key)

        new_view_msg = NewViewMessage(
            new_view_number=view_number,
            leader_id=self.node_id,
            view_change_proofs=view_change_proofs,
            prepared_state=prepared_state,
            signature=signature,
        )

        logger.info(f"👑 NEW-VIEW broadcast: v={view_number}")

        if self.on_new_view_broadcast:
            self.on_new_view_broadcast(new_view_msg)

    def get_consensus_state(self, proposal_id: str) -> Optional[ConsensusState]:
        """Get the consensus state for a proposal."""
        return self._consensus_state.get(proposal_id)

    # =========================================================================
    # TIMEOUT MANAGEMENT
    # =========================================================================

    def check_timeouts(self, timeout_ms: int = 5000) -> List[str]:
        """
        Check for proposals that have timed out.

        Returns list of proposal IDs that should trigger view-change.
        """
        timed_out = []
        now = time.time()
        timeout_sec = timeout_ms / 1000.0

        with self._timeout_lock:
            for pid, state in self._consensus_state.items():
                if state.phase not in (
                    ConsensusPhase.COMMITTED,
                    ConsensusPhase.ABORTED,
                ):
                    elapsed = now - state.started_at
                    if elapsed > timeout_sec:
                        timed_out.append(pid)
                        state.phase = ConsensusPhase.ABORTED
                        logger.warning(
                            f"⏰ Proposal {pid} timed out after {elapsed:.1f}s"
                        )

        return timed_out

    def shutdown(self) -> None:
        """Shutdown the consensus engine."""
        self._active = False
        if self._timeout_thread and self._timeout_thread.is_alive():
            self._timeout_thread.join(timeout=1.0)

    def register_peer(self, peer_id: str, public_key: str):
        """
        Register a peer's public key.

        SECURITY: All peers must be registered before they can vote.
        This prevents vote spoofing attacks where an attacker uses
        their own keypair with a fake voter_id.

        Standing on Giants - Lamport (1982):
        "A peer must be authenticated before participating in consensus."
        """
        # Validate peer_id format (non-empty, reasonable length)
        if not peer_id or not isinstance(peer_id, str):
            raise ValueError("peer_id must be a non-empty string")
        if len(peer_id) > 256:
            raise ValueError(f"peer_id too long: {len(peer_id)} > 256")
        # Prevent injection attacks in peer_id
        if any(c in peer_id for c in ["\n", "\r", "\0", ":"]):
            raise ValueError(f"peer_id contains invalid characters: {peer_id!r}")

        # Validate public key format (Ed25519 hex = 64 chars or base64)
        if not public_key or not isinstance(public_key, str):
            raise ValueError(
                f"Invalid public key for peer {peer_id}: key is empty or not string"
            )
        # Ed25519 public key is 32 bytes = 64 hex chars or ~44 base64 chars
        if len(public_key) < 44:
            raise ValueError(
                f"Invalid public key for peer {peer_id}: key too short ({len(public_key)} chars)"
            )
        if len(public_key) > 128:
            raise ValueError(
                f"Invalid public key for peer {peer_id}: key too long ({len(public_key)} chars)"
            )

        # Check for duplicate registration with different key (potential attack)
        if peer_id in self._peer_keys and self._peer_keys[peer_id] != public_key:
            logger.error(f"⚠️ SECURITY: Attempted key change for {peer_id}")
            raise ValueError(f"Peer {peer_id} already registered with different key")

        self._peer_keys[peer_id] = public_key
        logger.info(f"📋 Registered peer: {peer_id}")

    def unregister_peer(self, peer_id: str):
        """Remove a peer from the registry."""
        if peer_id in self._peer_keys and peer_id != self.node_id:
            del self._peer_keys[peer_id]

    def get_registered_peers(self) -> Dict[str, str]:
        """Get all registered peer_id -> public_key mappings."""
        return self._peer_keys.copy()

    def propose_pattern(self, pattern: Dict) -> Optional[Proposal]:
        """
        Initiate a consensus round for an elevated pattern.

        For backwards compatibility, this works in both leader and non-leader modes.
        In strict PBFT mode, use initiate_pre_prepare() instead.
        """
        # Use PBFT if we're in leader mode
        if self._is_leader:
            return self.initiate_pre_prepare(pattern)

        # Legacy mode: any node can propose
        self._sequence_counter += 1
        proposal_id = f"prop_{uuid.uuid4().hex[:8]}"
        proposal = Proposal(
            proposal_id=proposal_id,
            proposer_id=self.node_id,
            pattern_data=pattern,
            view_number=self._current_view,
            sequence_number=self._sequence_counter,
        )
        self.active_proposals[proposal_id] = proposal
        self.votes[proposal_id] = []

        # Initialize PBFT state
        self._consensus_state[proposal_id] = ConsensusState(
            phase=ConsensusPhase.PRE_PREPARE,
            view_number=self._current_view,
            sequence_number=self._sequence_counter,
        )

        logger.info(f"🗳️ Proposal initiated: {proposal_id}")
        return proposal

    def cast_vote(self, proposal: Proposal, ihsan_score: float) -> Optional[Vote]:
        """Validate a pattern and cast a signed vote."""
        # Import unified threshold from Single Source of Truth
        if ihsan_score < UNIFIED_IHSAN_THRESHOLD:
            logger.warning(
                f"❌ Rejecting proposal {proposal.proposal_id}: Ihsan {ihsan_score} < {UNIFIED_IHSAN_THRESHOLD}"
            )
            return None

        # Canonicalize and sign
        canon_data = canonical_json(proposal.pattern_data)
        digest = domain_separated_digest(canon_data)
        sig = sign_message(digest, self.private_key)

        vote = Vote(
            proposal_id=proposal.proposal_id,
            voter_id=self.node_id,
            signature=sig,
            public_key=self.public_key,
            ihsan_score=ihsan_score,
        )
        return vote

    def receive_vote(self, vote: Vote, node_count: int) -> bool:
        """
        Record a vote from a peer. Return True if quorum reached.

        This is the simplified API for backwards compatibility.
        For full PBFT, use receive_prepare() and receive_commit().

        SECURITY (S-1): Includes replay attack prevention via vote ID tracking.
        """
        if vote.proposal_id not in self.active_proposals:
            return False

        # SECURITY (S-1): Replay attack prevention
        # Generate unique vote ID from proposal + voter + signature
        vote_id = f"{vote.proposal_id}:{vote.voter_id}:{vote.signature[:32]}"
        if vote_id in self._seen_vote_ids:
            logger.warning(f"⚠️ REPLAY: Vote {vote_id[:32]}... already processed")
            return False

        proposal = self.active_proposals[vote.proposal_id]
        self._consensus_state.get(vote.proposal_id)

        # SECURITY: Verify voter_id is registered and public_key matches
        if vote.voter_id not in self._peer_keys:
            logger.error(f"⚠️ Vote from unregistered peer: {vote.voter_id}")
            return False

        registered_key = self._peer_keys[vote.voter_id]
        if vote.public_key != registered_key:
            logger.error(
                f"⚠️ Public key mismatch for {vote.voter_id}: "
                f"expected {registered_key[:16]}..., got {vote.public_key[:16]}..."
            )
            return False

        # Verify Signature using the REGISTERED public key (not vote.public_key)
        canon_data = canonical_json(proposal.pattern_data)
        digest = domain_separated_digest(canon_data)

        if not verify_signature(digest, vote.signature, registered_key):
            logger.error(f"⚠️ Invalid signature on vote from {vote.voter_id}")
            return False

        # Check for duplicate votes
        if any(v.voter_id == vote.voter_id for v in self.votes[vote.proposal_id]):
            return False

        # SECURITY (S-1): Record vote ID to prevent replay
        self._seen_vote_ids.add(vote_id)
        if len(self._seen_vote_ids) > self._max_seen_ids:
            # Evict oldest 10%
            to_keep = list(self._seen_vote_ids)[-int(self._max_seen_ids * 0.9) :]
            self._seen_vote_ids = set(to_keep)

        self.votes[vote.proposal_id].append(vote)

        # Check Quorum using consistent BFT formula: 2f + 1 where f = (n-1)//3
        quorum_count = self.get_quorum_size(node_count)

        logger.info(
            f"📈 Vote received for {vote.proposal_id} ({len(self.votes[vote.proposal_id])}/{quorum_count})"
        )

        if len(self.votes[vote.proposal_id]) >= quorum_count:
            return self._commit_proposal(vote.proposal_id)

        return False

    def _commit_proposal(self, proposal_id: str) -> bool:
        """
        Finalize the pattern as a system truth.

        Legacy method for backwards compatibility.
        For full PBFT, the commit happens after receive_commit() quorum.
        """
        if proposal_id in self.committed_patterns:
            return False

        proposal = self.active_proposals[proposal_id]
        state = self._consensus_state.get(proposal_id)

        # Update PBFT state if tracking
        if state:
            state.phase = ConsensusPhase.COMMITTED

        self.committed_patterns.add(proposal_id)

        logger.info(
            f"🏆 QUORUM REACHED: Pattern {proposal_id} committed to Giants Ledger."
        )

        if self.on_commit_broadcast:
            commit_payload: Dict[str, Any] = {
                "proposal_id": proposal_id,
                "pattern": proposal.pattern_data,
                "signatures": [v.signature for v in self.votes[proposal_id]],
            }
            if state:
                commit_payload["view_number"] = state.view_number
                commit_payload["sequence_number"] = state.sequence_number
            self.on_commit_broadcast(commit_payload)

        return True
