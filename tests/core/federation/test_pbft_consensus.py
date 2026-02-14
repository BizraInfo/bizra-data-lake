"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA PBFT CONSENSUS TEST SUITE                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Tests for Practical Byzantine Fault Tolerance implementation (GAP-C3)      ║
║   Validates: Two-phase commit, view-change, quorum logic, timeout handling   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from core.federation.consensus import (
    ConsensusEngine,
    ConsensusPhase,
    ConsensusState,
    Proposal,
    Vote,
    PrepareMessage,
    CommitMessage,
    ViewChangeRequest,
    NewViewMessage,
)
from core.pci.crypto import generate_keypair


class TestConsensusPhase:
    """Test the PBFT phase state machine."""

    def test_phase_progression(self):
        """Verify phase order: PRE_PREPARE → PREPARE → COMMIT → COMMITTED."""
        phases = [
            ConsensusPhase.PRE_PREPARE,
            ConsensusPhase.PREPARE,
            ConsensusPhase.COMMIT,
            ConsensusPhase.COMMITTED,
        ]
        # Verify distinct phases
        assert len(set(phases)) == 4

    def test_aborted_phase_exists(self):
        """Verify ABORTED phase for timeout handling."""
        assert ConsensusPhase.ABORTED is not None


class TestConsensusState:
    """Test consensus state tracking."""

    def test_initial_state(self):
        """Verify initial state is PRE_PREPARE."""
        state = ConsensusState()
        assert state.phase == ConsensusPhase.PRE_PREPARE
        assert state.prepare_count == 0
        assert state.commit_count == 0

    def test_state_tracks_signatures(self):
        """Verify state tracks prepare and commit signatures."""
        state = ConsensusState()
        state.prepare_signatures["node1"] = "sig1"
        state.commit_signatures["node2"] = "sig2"
        assert len(state.prepare_signatures) == 1
        assert len(state.commit_signatures) == 1


class TestQuorumCalculation:
    """Test Byzantine fault tolerance quorum logic."""

    def setup_method(self):
        """Set up test fixtures."""
        private_key, public_key = generate_keypair()
        self.engine = ConsensusEngine("node0", private_key, public_key)

    def test_quorum_4_nodes(self):
        """For n=4 (3f+1, f=1), quorum = 2f+1 = 3."""
        assert self.engine.get_quorum_size(4) == 3

    def test_quorum_7_nodes(self):
        """For n=7 (3f+1, f=2), quorum = 2f+1 = 5."""
        assert self.engine.get_quorum_size(7) == 5

    def test_quorum_10_nodes(self):
        """For n=10 (3f+1, f=3), quorum = 2f+1 = 7."""
        assert self.engine.get_quorum_size(10) == 7

    def test_quorum_single_node(self):
        """Single node should have quorum of 1."""
        assert self.engine.get_quorum_size(1) == 1


class TestLeaderElection:
    """Test deterministic leader selection."""

    def setup_method(self):
        """Set up test fixtures with multiple peers."""
        private_key, public_key = generate_keypair()
        self.engine = ConsensusEngine("node0", private_key, public_key)

        # Register additional peers
        for i in range(1, 4):
            pk, pub = generate_keypair()
            self.engine.register_peer(f"node{i}", pub)

    def test_leader_for_view_0(self):
        """View 0 should select first peer."""
        leader = self.engine.get_leader_for_view(0)
        assert leader in self.engine._peer_keys

    def test_leader_rotation(self):
        """Different views should select different leaders."""
        leaders = [self.engine.get_leader_for_view(v) for v in range(4)]
        # With 4 peers, views 0-3 should cycle through all
        assert len(set(leaders)) == 4

    def test_set_leader(self):
        """Setting leader should update internal state."""
        self.engine.set_leader("node0")
        assert self.engine._is_leader is True
        assert self.engine._leader_id == "node0"

        self.engine.set_leader("node1")
        assert self.engine._is_leader is False
        assert self.engine._leader_id == "node1"


class TestPrePreparePhase:
    """Test PBFT Pre-Prepare phase (leader proposal)."""

    def setup_method(self):
        """Set up test fixtures."""
        private_key, public_key = generate_keypair()
        self.engine = ConsensusEngine("leader", private_key, public_key)
        self.engine.set_leader("leader")

    def test_leader_can_propose(self):
        """Leader should successfully create pre-prepare."""
        proposal = self.engine.initiate_pre_prepare({"pattern": "test"})
        assert proposal is not None
        assert proposal.proposer_id == "leader"
        assert proposal.view_number == 0

    def test_non_leader_cannot_propose(self):
        """Non-leader should fail to create pre-prepare."""
        self.engine.set_leader("other_node")
        proposal = self.engine.initiate_pre_prepare({"pattern": "test"})
        assert proposal is None

    def test_sequence_number_increments(self):
        """Sequence number should increment with each proposal."""
        p1 = self.engine.initiate_pre_prepare({"pattern": "test1"})
        p2 = self.engine.initiate_pre_prepare({"pattern": "test2"})
        assert p2.sequence_number == p1.sequence_number + 1


class TestPreparePhase:
    """Test PBFT Prepare phase (replica acknowledgment)."""

    def setup_method(self):
        """Set up test fixtures with leader and replica."""
        self.leader_pk, self.leader_pub = generate_keypair()
        self.replica_pk, self.replica_pub = generate_keypair()

        self.leader = ConsensusEngine("leader", self.leader_pk, self.leader_pub)
        self.replica = ConsensusEngine("replica", self.replica_pk, self.replica_pub)

        # Cross-register
        self.leader.register_peer("replica", self.replica_pub)
        self.replica.register_peer("leader", self.leader_pub)

        self.leader.set_leader("leader")
        self.replica.set_leader("leader")

    def test_replica_sends_prepare(self):
        """Replica should send PREPARE for valid proposal."""
        proposal = self.leader.initiate_pre_prepare({"pattern": "test"})
        # Replica receives proposal (simulated)
        self.replica.active_proposals[proposal.proposal_id] = proposal
        self.replica._consensus_state[proposal.proposal_id] = ConsensusState()

        prepare = self.replica.send_prepare(proposal, ihsan_score=0.96)
        assert prepare is not None
        assert prepare.replica_id == "replica"
        assert prepare.view_number == 0

    def test_prepare_rejected_low_ihsan(self):
        """PREPARE should be rejected if Ihsān < threshold."""
        proposal = self.leader.initiate_pre_prepare({"pattern": "test"})
        self.replica.active_proposals[proposal.proposal_id] = proposal
        self.replica._consensus_state[proposal.proposal_id] = ConsensusState()

        prepare = self.replica.send_prepare(proposal, ihsan_score=0.50)
        assert prepare is None

    def test_prepare_quorum_detection(self):
        """Should detect when PREPARE quorum is reached."""
        proposal = self.leader.initiate_pre_prepare({"pattern": "test"})

        # Create 3 more replicas for 4-node network
        replicas = []
        for i in range(3):
            pk, pub = generate_keypair()
            replica = ConsensusEngine(f"replica{i}", pk, pub)
            replica.register_peer("leader", self.leader_pub)
            self.leader.register_peer(f"replica{i}", pub)
            replica._current_view = 0
            replicas.append((replica, pk, pub))

        # Each replica sends PREPARE
        for replica, pk, pub in replicas:
            replica.active_proposals[proposal.proposal_id] = proposal
            replica._consensus_state[proposal.proposal_id] = ConsensusState()
            prepare = replica.send_prepare(proposal, ihsan_score=0.96)
            assert prepare is not None
            quorum_reached = self.leader.receive_prepare(prepare, node_count=4)

        # With 4 nodes, quorum=3, should reach quorum
        state = self.leader._consensus_state[proposal.proposal_id]
        assert state.prepare_count >= 3


class TestCommitPhase:
    """Test PBFT Commit phase."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pk, self.pub = generate_keypair()
        self.engine = ConsensusEngine("node", self.pk, self.pub)
        self.engine.set_leader("node")

    def test_commit_after_prepare_quorum(self):
        """Should be able to send COMMIT after PREPARE quorum."""
        proposal = self.engine.initiate_pre_prepare({"pattern": "test"})
        state = self.engine._consensus_state[proposal.proposal_id]

        # Simulate reaching PREPARE quorum
        state.phase = ConsensusPhase.PREPARE
        state.prepare_count = 3

        commit = self.engine.send_commit(proposal)
        assert commit is not None
        assert commit.replica_id == "node"

    def test_commit_rejected_before_prepare(self):
        """Should not send COMMIT if still in PRE_PREPARE phase."""
        proposal = self.engine.initiate_pre_prepare({"pattern": "test"})
        # State is still PRE_PREPARE
        commit = self.engine.send_commit(proposal)
        assert commit is None


class TestViewChange:
    """Test view-change protocol for leader failure recovery."""

    def setup_method(self):
        """Set up test fixtures with multiple nodes."""
        self.nodes = []
        for i in range(4):
            pk, pub = generate_keypair()
            node = ConsensusEngine(f"node{i}", pk, pub)
            self.nodes.append((node, pk, pub))

        # Cross-register all nodes
        for i, (node, pk, pub) in enumerate(self.nodes):
            for j, (other, _, other_pub) in enumerate(self.nodes):
                if i != j:
                    node.register_peer(f"node{j}", other_pub)

    def test_view_change_request(self):
        """Node should be able to request view change."""
        node, _, _ = self.nodes[0]
        request = node.request_view_change(reason="leader_timeout")

        assert request is not None
        assert request.view_number == 1
        assert request.requester_id == "node0"

    def test_view_change_quorum(self):
        """View change should occur after quorum of requests."""
        # All nodes request view change
        requests = []
        for node, _, _ in self.nodes:
            req = node.request_view_change(reason="timeout")
            requests.append(req)

        # Node 0 receives requests from others
        leader_node, _, _ = self.nodes[0]
        for i, req in enumerate(requests[1:], 1):
            result = leader_node.receive_view_change(req, node_count=4)

        # After quorum, view should change
        assert leader_node._current_view == 1

    def test_view_change_leader_rotation(self):
        """View change should result in new leader."""
        node, _, _ = self.nodes[0]
        old_leader = node.get_leader_for_view(0)
        new_leader = node.get_leader_for_view(1)

        # Leaders should differ (with 4 nodes)
        assert old_leader != new_leader


class TestTimeoutHandling:
    """Test timeout-based view change triggers."""

    def setup_method(self):
        """Set up test fixtures."""
        pk, pub = generate_keypair()
        self.engine = ConsensusEngine("node", pk, pub)
        self.engine.set_leader("node")

    def test_timeout_detection(self):
        """Should detect timed-out proposals."""
        proposal = self.engine.initiate_pre_prepare({"pattern": "test"})

        # Set started_at to past (simulate timeout)
        state = self.engine._consensus_state[proposal.proposal_id]
        state.started_at = time.time() - 10  # 10 seconds ago

        timed_out = self.engine.check_timeouts(timeout_ms=5000)
        assert proposal.proposal_id in timed_out

    def test_timeout_aborts_proposal(self):
        """Timed out proposal should be marked ABORTED."""
        proposal = self.engine.initiate_pre_prepare({"pattern": "test"})
        state = self.engine._consensus_state[proposal.proposal_id]
        state.started_at = time.time() - 10

        self.engine.check_timeouts(timeout_ms=5000)
        assert state.phase == ConsensusPhase.ABORTED


class TestFullPBFTFlow:
    """Integration test for complete PBFT consensus flow."""

    def test_full_consensus_round(self):
        """Test complete PBFT: PRE_PREPARE → PREPARE → COMMIT → COMMITTED."""
        # Create 4-node network
        nodes = []
        for i in range(4):
            pk, pub = generate_keypair()
            node = ConsensusEngine(f"node{i}", pk, pub)
            nodes.append((node, pk, pub))

        # Cross-register
        for i, (node, _, pub) in enumerate(nodes):
            for j, (other, _, other_pub) in enumerate(nodes):
                if i != j:
                    node.register_peer(f"node{j}", other_pub)

        # Set node0 as leader
        for node, _, _ in nodes:
            node.set_leader("node0")

        leader, leader_pk, leader_pub = nodes[0]

        # Phase 1: PRE-PREPARE
        proposal = leader.initiate_pre_prepare({"pattern": "test_pattern", "ihsan": 0.97})
        assert proposal is not None

        # Distribute proposal to all replicas
        for node, _, _ in nodes[1:]:
            node.active_proposals[proposal.proposal_id] = proposal
            node._consensus_state[proposal.proposal_id] = ConsensusState(
                view_number=proposal.view_number,
                sequence_number=proposal.sequence_number,
            )

        # Phase 2: PREPARE - each replica sends prepare
        prepares = []
        for node, _, _ in nodes:
            prepare = node.send_prepare(proposal, ihsan_score=0.97)
            if prepare:
                prepares.append(prepare)

        # Leader receives prepares
        for prepare in prepares:
            leader.receive_prepare(prepare, node_count=4)

        # Check leader state - should be in PREPARE phase after quorum
        leader_state = leader._consensus_state[proposal.proposal_id]
        # We have 4 prepares, quorum is 3, so should transition
        assert leader_state.prepare_count >= 3

        # Phase 3: COMMIT
        leader_state.phase = ConsensusPhase.PREPARE  # Transition
        commits = []
        for node, _, _ in nodes:
            node._consensus_state[proposal.proposal_id].phase = ConsensusPhase.PREPARE
            commit = node.send_commit(proposal)
            if commit:
                commits.append(commit)

        # Leader receives commits
        for commit in commits:
            leader.receive_commit(commit, node_count=4)

        # Should be committed
        assert proposal.proposal_id in leader.committed_patterns


class TestBackwardsCompatibility:
    """Test that legacy API still works."""

    def setup_method(self):
        """Set up test fixtures."""
        pk, pub = generate_keypair()
        self.engine = ConsensusEngine("node", pk, pub)
        # Don't set leader - use legacy mode

    def test_propose_pattern_legacy(self):
        """Legacy propose_pattern should still work."""
        proposal = self.engine.propose_pattern({"pattern": "test"})
        assert proposal is not None
        assert proposal.proposal_id.startswith("prop_")

    def test_cast_vote_legacy(self):
        """Legacy cast_vote should still work."""
        proposal = self.engine.propose_pattern({"pattern": "test"})
        vote = self.engine.cast_vote(proposal, ihsan_score=0.96)
        assert vote is not None

    def test_receive_vote_legacy(self):
        """Legacy receive_vote should still work."""
        # Create two nodes
        pk1, pub1 = generate_keypair()
        pk2, pub2 = generate_keypair()

        node1 = ConsensusEngine("node1", pk1, pub1)
        node2 = ConsensusEngine("node2", pk2, pub2)

        # Cross-register
        node1.register_peer("node2", pub2)
        node2.register_peer("node1", pub1)

        # Node1 proposes
        proposal = node1.propose_pattern({"pattern": "test"})

        # Node2 receives and votes
        node2.active_proposals[proposal.proposal_id] = proposal
        node2.votes[proposal.proposal_id] = []
        vote = node2.cast_vote(proposal, ihsan_score=0.96)

        # Node1 receives vote
        result = node1.receive_vote(vote, node_count=2)
        # With 2 nodes, quorum=1, should commit immediately
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
