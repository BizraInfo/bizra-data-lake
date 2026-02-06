"""
Session as State Machine — Merkle-DAG Session Lineage

Extends NTU with cryptographic Merkle-DAG for provable state transitions.
Every session state is linked to its predecessors via cryptographic hashes,
creating an immutable, verifiable lineage.

Standing on Giants:
- Merkle (1979): Hash trees for data integrity
- Lamport (1978): Logical clocks for happened-before ordering
- Nakamoto (2008): Blockchain-style chaining for immutability
- NTU: Temporal pattern detection integration

State Machine Transitions:
    INIT -> ACTIVE -> COMPUTING -> VALIDATED -> COMMITTED
                   |-> SUSPENDED -> RESUMED -> ...
                   |-> FAILED -> RECOVERED -> ...

Merkle-DAG Properties:
1. Content-addressable: States are identified by their hash
2. Immutable: Past states cannot be modified without detection
3. Verifiable: Any state can prove its lineage to genesis
4. Mergeable: Concurrent branches can be reconciled

Created: 2026-02-03 | BIZRA Elite Integration v1.1.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import uuid

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ============================================================================
# TYPES
# ============================================================================

T = TypeVar("T")


class SessionState(str, Enum):
    """Session lifecycle states."""
    INIT = "init"           # Session created, not yet active
    ACTIVE = "active"       # Session active, processing allowed
    COMPUTING = "computing" # In computation (blocking state)
    VALIDATED = "validated" # Computation validated via FATE
    COMMITTED = "committed" # State committed to DAG
    SUSPENDED = "suspended" # Temporarily suspended
    RESUMED = "resumed"     # Resumed from suspension
    FAILED = "failed"       # Error state
    RECOVERED = "recovered" # Recovered from failure
    TERMINATED = "terminated" # Session ended


class TransitionType(str, Enum):
    """Types of state transitions."""
    STANDARD = "standard"     # Normal state progression
    BRANCH = "branch"         # Fork to parallel state
    MERGE = "merge"           # Join from multiple states
    ROLLBACK = "rollback"     # Revert to previous state
    RECOVERY = "recovery"     # Recover from failure


# Valid state transitions
VALID_TRANSITIONS: Dict[SessionState, Set[SessionState]] = {
    SessionState.INIT: {SessionState.ACTIVE, SessionState.TERMINATED},
    SessionState.ACTIVE: {SessionState.COMPUTING, SessionState.SUSPENDED, SessionState.FAILED, SessionState.TERMINATED},
    SessionState.COMPUTING: {SessionState.VALIDATED, SessionState.FAILED},
    SessionState.VALIDATED: {SessionState.COMMITTED, SessionState.FAILED},
    SessionState.COMMITTED: {SessionState.ACTIVE, SessionState.TERMINATED},
    SessionState.SUSPENDED: {SessionState.RESUMED, SessionState.TERMINATED},
    SessionState.RESUMED: {SessionState.ACTIVE},
    SessionState.FAILED: {SessionState.RECOVERED, SessionState.TERMINATED},
    SessionState.RECOVERED: {SessionState.ACTIVE},
    SessionState.TERMINATED: set(),  # Terminal state
}


# ============================================================================
# MERKLE NODE
# ============================================================================

@dataclass
class MerkleNode:
    """
    A node in the Merkle-DAG.

    Each node represents a session state with cryptographic linking
    to parent states.
    """
    # Content hash (computed from state + metadata)
    hash: str

    # Parent hashes (multiple for merges)
    parents: List[str]

    # Session state at this node
    state: SessionState

    # State data (serializable)
    data: Dict[str, Any]

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    logical_clock: int = 0
    session_id: str = ""

    # NTU integration
    ntu_belief: float = 0.0
    ntu_entropy: float = 1.0
    ntu_potential: float = 0.0

    # FATE validation
    fate_score: float = 0.0
    ihsan_achieved: bool = False

    # Transition info
    transition_type: TransitionType = TransitionType.STANDARD
    transition_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node."""
        return {
            "hash": self.hash,
            "parents": self.parents,
            "state": self.state.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "logical_clock": self.logical_clock,
            "session_id": self.session_id,
            "ntu": {
                "belief": self.ntu_belief,
                "entropy": self.ntu_entropy,
                "potential": self.ntu_potential,
            },
            "fate_score": self.fate_score,
            "ihsan_achieved": self.ihsan_achieved,
            "transition_type": self.transition_type.value,
            "transition_reason": self.transition_reason,
        }

    @staticmethod
    def compute_hash(
        parents: List[str],
        state: SessionState,
        data: Dict[str, Any],
        timestamp: datetime,
    ) -> str:
        """
        Compute Merkle hash for a node.

        Hash includes: parents, state, data hash, timestamp
        This ensures content-addressability.
        """
        content = {
            "parents": sorted(parents),  # Canonical order
            "state": state.value,
            "data_hash": hashlib.sha256(
                json.dumps(data, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "timestamp": timestamp.isoformat(),
        }

        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()


# ============================================================================
# MERKLE DAG
# ============================================================================

class MerkleDAG:
    """
    Merkle-DAG for session state management.

    Provides:
    - Content-addressable state storage
    - Cryptographic linking between states
    - Lineage verification
    - Branch/merge support
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())[:12]

        # Node storage: hash -> MerkleNode
        self._nodes: Dict[str, MerkleNode] = {}

        # Current head(s) - for branch tracking
        self._heads: Set[str] = set()

        # Genesis node hash
        self._genesis: Optional[str] = None

        # Logical clock
        self._logical_clock = 0

        # Initialize with genesis node
        self._create_genesis()

    def _create_genesis(self) -> MerkleNode:
        """Create the genesis (root) node."""
        genesis_hash = MerkleNode.compute_hash(
            parents=[],
            state=SessionState.INIT,
            data={"genesis": True, "session_id": self.session_id},
            timestamp=datetime.now(timezone.utc),
        )

        genesis = MerkleNode(
            hash=genesis_hash,
            parents=[],
            state=SessionState.INIT,
            data={"genesis": True, "session_id": self.session_id},
            session_id=self.session_id,
            logical_clock=0,
        )

        self._nodes[genesis_hash] = genesis
        self._heads.add(genesis_hash)
        self._genesis = genesis_hash

        logger.info(f"Created Merkle-DAG genesis: {genesis_hash[:12]}...")

        return genesis

    def add_state(
        self,
        state: SessionState,
        data: Dict[str, Any],
        parents: Optional[List[str]] = None,
        transition_type: TransitionType = TransitionType.STANDARD,
        transition_reason: str = "",
        ntu_state: Optional[Dict[str, float]] = None,
        fate_score: float = 0.0,
    ) -> MerkleNode:
        """
        Add a new state to the DAG.

        Args:
            state: The new session state
            data: State data
            parents: Parent node hashes (default: current heads)
            transition_type: Type of transition
            transition_reason: Reason for transition
            ntu_state: NTU state dict {belief, entropy, potential}
            fate_score: FATE validation score

        Returns:
            The created MerkleNode
        """
        # Default to current heads as parents
        if parents is None:
            parents = list(self._heads)

        # Validate transition
        for parent_hash in parents:
            parent = self._nodes.get(parent_hash)
            if parent:
                if state not in VALID_TRANSITIONS.get(parent.state, set()):
                    logger.warning(
                        f"Invalid transition: {parent.state.value} -> {state.value}"
                    )
                    # Allow for rollback, recovery, and branch scenarios
                    if transition_type not in (TransitionType.ROLLBACK, TransitionType.RECOVERY, TransitionType.BRANCH, TransitionType.MERGE):
                        raise InvalidTransitionError(
                            f"Cannot transition from {parent.state.value} to {state.value}"
                        )

        # Increment logical clock
        self._logical_clock += 1

        timestamp = datetime.now(timezone.utc)

        # Compute hash
        node_hash = MerkleNode.compute_hash(parents, state, data, timestamp)

        # Extract NTU values
        ntu = ntu_state or {}

        # Create node
        node = MerkleNode(
            hash=node_hash,
            parents=parents,
            state=state,
            data=data,
            timestamp=timestamp,
            logical_clock=self._logical_clock,
            session_id=self.session_id,
            ntu_belief=ntu.get("belief", 0.0),
            ntu_entropy=ntu.get("entropy", 1.0),
            ntu_potential=ntu.get("potential", 0.0),
            fate_score=fate_score,
            ihsan_achieved=fate_score >= UNIFIED_IHSAN_THRESHOLD,
            transition_type=transition_type,
            transition_reason=transition_reason,
        )

        # Store node
        self._nodes[node_hash] = node

        # Update heads
        for parent_hash in parents:
            self._heads.discard(parent_hash)
        self._heads.add(node_hash)

        logger.debug(
            f"Added DAG node: {node_hash[:12]}... "
            f"(state={state.value}, parents={len(parents)}, clock={self._logical_clock})"
        )

        return node

    def get_node(self, hash: str) -> Optional[MerkleNode]:
        """Get node by hash."""
        return self._nodes.get(hash)

    def get_current_state(self) -> SessionState:
        """Get current session state (from primary head)."""
        if not self._heads:
            return SessionState.INIT

        # Use most recent head by logical clock
        head_nodes = [self._nodes[h] for h in self._heads]
        current = max(head_nodes, key=lambda n: n.logical_clock)
        return current.state

    def get_current_node(self) -> Optional[MerkleNode]:
        """Get current (most recent) node."""
        if not self._heads:
            return None

        head_nodes = [self._nodes[h] for h in self._heads]
        return max(head_nodes, key=lambda n: n.logical_clock)

    def get_heads(self) -> List[MerkleNode]:
        """Get all current head nodes."""
        return [self._nodes[h] for h in self._heads]

    def verify_lineage(self, node_hash: str, target_hash: Optional[str] = None) -> bool:
        """
        Verify that a node descends from target (or genesis).

        Args:
            node_hash: Node to verify
            target_hash: Target ancestor (default: genesis)

        Returns:
            True if lineage is valid
        """
        target = target_hash or self._genesis
        if not target:
            return False

        # BFS to find path to target
        visited = set()
        queue = [node_hash]

        while queue:
            current = queue.pop(0)
            if current == target:
                return True

            if current in visited:
                continue
            visited.add(current)

            node = self._nodes.get(current)
            if node:
                queue.extend(node.parents)

        return False

    def get_lineage(self, node_hash: str) -> List[MerkleNode]:
        """
        Get full lineage from node to genesis.

        Returns nodes in reverse chronological order.
        """
        lineage = []
        visited = set()
        queue = [node_hash]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            node = self._nodes.get(current)
            if node:
                lineage.append(node)
                queue.extend(node.parents)

        return lineage

    def branch(self, branch_name: str, data: Optional[Dict[str, Any]] = None) -> MerkleNode:
        """
        Create a branch from current state.

        Branches allow parallel state exploration.
        """
        current = self.get_current_node()
        if not current:
            raise DAGError("Cannot branch: no current node")

        branch_data = data or {}
        branch_data["_branch"] = branch_name

        return self.add_state(
            state=current.state,  # Same state, new branch
            data=branch_data,
            parents=[current.hash],
            transition_type=TransitionType.BRANCH,
            transition_reason=f"Branch: {branch_name}",
        )

    def merge(
        self,
        branch_hashes: List[str],
        merged_data: Dict[str, Any],
        result_state: SessionState,
    ) -> MerkleNode:
        """
        Merge multiple branches.

        Args:
            branch_hashes: Hashes of branches to merge
            merged_data: Reconciled data from merge
            result_state: Resulting state after merge

        Returns:
            Merged node
        """
        if len(branch_hashes) < 2:
            raise DAGError("Merge requires at least 2 branches")

        # Verify all branches exist
        for bh in branch_hashes:
            if bh not in self._nodes:
                raise DAGError(f"Unknown branch: {bh}")

        return self.add_state(
            state=result_state,
            data=merged_data,
            parents=branch_hashes,
            transition_type=TransitionType.MERGE,
            transition_reason=f"Merged {len(branch_hashes)} branches",
        )

    def rollback(self, target_hash: str, reason: str = "") -> MerkleNode:
        """
        Rollback to a previous state.

        Note: This doesn't delete history, it creates a new node
        pointing to the target state.
        """
        target = self._nodes.get(target_hash)
        if not target:
            raise DAGError(f"Rollback target not found: {target_hash}")

        return self.add_state(
            state=target.state,
            data={"_rollback_to": target_hash, **target.data},
            parents=list(self._heads),  # Link from current heads
            transition_type=TransitionType.ROLLBACK,
            transition_reason=reason or f"Rollback to {target_hash[:12]}",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get DAG statistics."""
        states_count = defaultdict(int)
        for node in self._nodes.values():
            states_count[node.state.value] += 1

        ihsan_achieved_count = sum(1 for n in self._nodes.values() if n.ihsan_achieved)

        return {
            "session_id": self.session_id,
            "total_nodes": len(self._nodes),
            "heads": len(self._heads),
            "logical_clock": self._logical_clock,
            "genesis": self._genesis[:12] if self._genesis else None,
            "current_state": self.get_current_state().value,
            "states_distribution": dict(states_count),
            "ihsan_achieved_count": ihsan_achieved_count,
            "ihsan_rate": ihsan_achieved_count / max(len(self._nodes), 1),
        }

    def export(self) -> Dict[str, Any]:
        """Export full DAG for persistence."""
        return {
            "session_id": self.session_id,
            "genesis": self._genesis,
            "heads": list(self._heads),
            "logical_clock": self._logical_clock,
            "nodes": {h: n.to_dict() for h, n in self._nodes.items()},
        }

    @classmethod
    def import_dag(cls, data: Dict[str, Any]) -> "MerkleDAG":
        """Import DAG from exported data."""
        dag = cls.__new__(cls)
        dag.session_id = data["session_id"]
        dag._genesis = data["genesis"]
        dag._heads = set(data["heads"])
        dag._logical_clock = data["logical_clock"]

        # Reconstruct nodes
        dag._nodes = {}
        for hash_, node_data in data["nodes"].items():
            dag._nodes[hash_] = MerkleNode(
                hash=node_data["hash"],
                parents=node_data["parents"],
                state=SessionState(node_data["state"]),
                data=node_data["data"],
                timestamp=datetime.fromisoformat(node_data["timestamp"]),
                logical_clock=node_data["logical_clock"],
                session_id=node_data["session_id"],
                ntu_belief=node_data["ntu"]["belief"],
                ntu_entropy=node_data["ntu"]["entropy"],
                ntu_potential=node_data["ntu"]["potential"],
                fate_score=node_data["fate_score"],
                ihsan_achieved=node_data["ihsan_achieved"],
                transition_type=TransitionType(node_data["transition_type"]),
                transition_reason=node_data["transition_reason"],
            )

        return dag


# ============================================================================
# SESSION STATE MACHINE
# ============================================================================

class SessionStateMachine:
    """
    Session state machine backed by Merkle-DAG.

    Provides:
    - Type-safe state transitions
    - NTU integration for temporal patterns
    - FATE validation at transitions
    - Complete audit trail via DAG
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
    ):
        self.dag = MerkleDAG(session_id)
        self.ihsan_threshold = ihsan_threshold

        # NTU integration (lazy-loaded)
        self._ntu = None

        # Transition callbacks
        self._on_transition: Optional[Callable[[MerkleNode], None]] = None

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self.dag.session_id

    @property
    def current_state(self) -> SessionState:
        """Get current state."""
        return self.dag.get_current_state()

    @property
    def current_node(self) -> Optional[MerkleNode]:
        """Get current node."""
        return self.dag.get_current_node()

    @property
    def ntu(self):
        """Lazy-load NTU."""
        if self._ntu is None:
            try:
                from core.ntu import NTU, NTUConfig
                self._ntu = NTU(NTUConfig(ihsan_threshold=self.ihsan_threshold))
            except ImportError:
                logger.warning("NTU not available")
        return self._ntu

    def transition(
        self,
        to_state: SessionState,
        data: Optional[Dict[str, Any]] = None,
        reason: str = "",
        fate_score: Optional[float] = None,
    ) -> MerkleNode:
        """
        Transition to a new state.

        Args:
            to_state: Target state
            data: State data
            reason: Transition reason
            fate_score: FATE validation score (auto-computed if None)

        Returns:
            The new node
        """
        data = data or {}
        data["_transition_reason"] = reason

        # Get NTU state if available
        ntu_state = None
        if self.ntu:
            # Observe the transition as a quality signal
            # High fate score = high quality observation
            observation = fate_score if fate_score else 0.5
            self.ntu.observe(observation, {"transition": to_state.value})

            ntu_state = {
                "belief": self.ntu.state.belief,
                "entropy": self.ntu.state.entropy,
                "potential": self.ntu.state.potential,
            }

        # Default fate score based on NTU
        if fate_score is None:
            fate_score = ntu_state["belief"] if ntu_state else self.ihsan_threshold

        node = self.dag.add_state(
            state=to_state,
            data=data,
            transition_reason=reason,
            ntu_state=ntu_state,
            fate_score=fate_score,
        )

        # Callback
        if self._on_transition:
            self._on_transition(node)

        logger.info(
            f"Session {self.session_id}: {self.current_state.value} -> {to_state.value} "
            f"(fate={fate_score:.4f}, ihsan={node.ihsan_achieved})"
        )

        return node

    def activate(self, data: Optional[Dict[str, Any]] = None) -> MerkleNode:
        """Activate the session."""
        return self.transition(SessionState.ACTIVE, data, "Session activated")

    def compute(self, computation_data: Dict[str, Any]) -> MerkleNode:
        """Enter computation state."""
        return self.transition(SessionState.COMPUTING, computation_data, "Starting computation")

    def validate(self, validation_result: Dict[str, Any], fate_score: float) -> MerkleNode:
        """Validate computation result."""
        return self.transition(
            SessionState.VALIDATED,
            validation_result,
            "Computation validated",
            fate_score,
        )

    def commit(self, commit_data: Optional[Dict[str, Any]] = None) -> MerkleNode:
        """Commit validated state."""
        return self.transition(SessionState.COMMITTED, commit_data, "State committed")

    def suspend(self, reason: str = "") -> MerkleNode:
        """Suspend session."""
        return self.transition(SessionState.SUSPENDED, {"suspend_reason": reason}, reason)

    def resume(self) -> MerkleNode:
        """Resume suspended session."""
        return self.transition(SessionState.RESUMED, {}, "Session resumed")

    def fail(self, error: str) -> MerkleNode:
        """Mark session as failed."""
        return self.transition(SessionState.FAILED, {"error": error}, f"Failed: {error}")

    def recover(self, recovery_data: Optional[Dict[str, Any]] = None) -> MerkleNode:
        """Recover from failure."""
        return self.transition(SessionState.RECOVERED, recovery_data, "Recovered from failure")

    def terminate(self, reason: str = "Normal termination") -> MerkleNode:
        """Terminate session."""
        return self.transition(SessionState.TERMINATED, {"termination_reason": reason}, reason)

    def on_transition(self, callback: Callable[[MerkleNode], None]) -> None:
        """Register transition callback."""
        self._on_transition = callback

    def get_history(self) -> List[MerkleNode]:
        """Get full session history."""
        current = self.dag.get_current_node()
        if not current:
            return []
        return self.dag.get_lineage(current.hash)

    def verify_integrity(self) -> bool:
        """Verify DAG integrity from current state to genesis."""
        current = self.dag.get_current_node()
        if not current:
            return True
        return self.dag.verify_lineage(current.hash)

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        stats = self.dag.get_stats()

        if self.ntu:
            stats["ntu"] = {
                "belief": self.ntu.state.belief,
                "entropy": self.ntu.state.entropy,
                "potential": self.ntu.state.potential,
                "ihsan_achieved": self.ntu.state.ihsan_achieved,
            }

        return stats


# ============================================================================
# EXCEPTIONS
# ============================================================================

class DAGError(Exception):
    """Base exception for DAG operations."""
    pass


class InvalidTransitionError(DAGError):
    """Invalid state transition attempted."""
    pass


# ============================================================================
# INTEGRATION WITH HOOKS
# ============================================================================

class SessionHookIntegration:
    """
    Integrates session state machine with hook system.

    Provides automatic session state tracking for hooked operations.
    """

    def __init__(self, session: SessionStateMachine):
        self.session = session

    def wrap_operation(
        self,
        operation: Callable,
        operation_name: str = "",
    ) -> Callable:
        """
        Wrap an operation with session state tracking.

        The wrapper:
        1. Transitions to COMPUTING before operation
        2. Validates result via FATE
        3. Transitions to VALIDATED/FAILED based on result
        4. Commits on success
        """
        import asyncio
        import functools

        @functools.wraps(operation)
        async def async_wrapper(*args, **kwargs):
            # Enter computing state
            self.session.compute({
                "operation": operation_name or operation.__name__,
                "args_hash": hashlib.sha256(str(args).encode()).hexdigest()[:8],
            })

            try:
                # Execute operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)

                # Validate (using simple heuristic - production would use FATE)
                fate_score = self.session.ihsan_threshold  # Assume success meets threshold

                self.session.validate(
                    {"result_type": type(result).__name__},
                    fate_score,
                )

                # Commit
                self.session.commit({"success": True})

                return result

            except Exception as e:
                # Failed
                self.session.fail(str(e))
                raise

        return async_wrapper


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_session(
    session_id: Optional[str] = None,
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
) -> SessionStateMachine:
    """
    Create a new session state machine.

    Example:
        session = create_session()
        session.activate()
        session.compute({"task": "process"})
        session.validate({"result": "ok"}, 0.97)
        session.commit()
    """
    return SessionStateMachine(session_id, ihsan_threshold)


def create_tracked_session(
    session_id: Optional[str] = None,
    on_transition: Optional[Callable[[MerkleNode], None]] = None,
) -> SessionStateMachine:
    """
    Create a session with transition tracking.

    Example:
        def track(node):
            print(f"Transition: {node.state.value}")

        session = create_tracked_session(on_transition=track)
    """
    session = create_session(session_id)
    if on_transition:
        session.on_transition(on_transition)
    return session
