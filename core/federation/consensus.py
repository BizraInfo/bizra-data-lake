"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA PATTERN FEDERATION — CONSENSUS ENGINE (BFT)                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Protects the 'Shoulders of Giants' from faulty or malicious input.         ║
║   Algorithm: Simplified 2-Phase Commit with Ed25519 Signatures.              ║
║   Quorum Threshold: 2f + 1                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Callable
from core.pci.crypto import (
    domain_separated_digest,
    sign_message,
    verify_signature,
    canonical_json,
)

logger = logging.getLogger("CONSENSUS")


@dataclass
class Proposal:
    proposal_id: str
    proposer_id: str
    pattern_data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class Vote:
    proposal_id: str
    voter_id: str
    signature: str
    public_key: str
    ihsan_score: float


class ConsensusEngine:
    """
    Byzantine Fault Tolerant Consensus for Pattern Elevation.
    """

    def __init__(self, node_id: str, private_key: str, public_key: str):
        self.node_id = node_id
        self.private_key = private_key
        self.public_key = public_key

        # State tracking
        self.active_proposals: Dict[str, Proposal] = {}
        self.votes: Dict[str, List[Vote]] = {}  # proposal_id -> List[Vote]
        self.committed_patterns: Set[str] = set()

        # Callbacks
        self.on_commit_broadcast: Optional[Callable[[Dict], None]] = None

    def propose_pattern(self, pattern: Dict) -> Proposal:
        """Initiate a consensus round for an elevated pattern."""
        proposal_id = f"prop_{uuid.uuid4().hex[:8]}"
        proposal = Proposal(
            proposal_id=proposal_id, proposer_id=self.node_id, pattern_data=pattern
        )
        self.active_proposals[proposal_id] = proposal
        self.votes[proposal_id] = []

        logger.info(f"🗳️ Proposal initiated: {proposal_id}")
        return proposal

    def cast_vote(self, proposal: Proposal, ihsan_score: float) -> Optional[Vote]:
        """Validate a pattern and cast a signed vote."""
        # Unified Ihsan threshold (0.95) - consistent with A2A, PCI, propagation
        if ihsan_score < 0.95:
            logger.warning(
                f"❌ Rejecting proposal {proposal.proposal_id}: Ihsan {ihsan_score} < 0.95"
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
        """Record a vote from a peer. Return True if quorum reached."""
        if vote.proposal_id not in self.active_proposals:
            return False

        proposal = self.active_proposals[vote.proposal_id]

        # Verify Signature
        canon_data = canonical_json(proposal.pattern_data)
        digest = domain_separated_digest(canon_data)

        if not verify_signature(digest, vote.signature, vote.public_key):
            logger.error(f"⚠️ Invalid signature on vote from {vote.voter_id}")
            return False

        # Check for duplicate votes
        if any(v.voter_id == vote.voter_id for v in self.votes[vote.proposal_id]):
            return False

        self.votes[vote.proposal_id].append(vote)

        # Check Quorum (Simplified: 2n/3 + 1)
        # f = (node_count - 1) // 3
        # quorum = 2 * f + 1
        quorum_count = (2 * node_count // 3) + 1

        logger.info(
            f"📈 Vote received for {vote.proposal_id} ({len(self.votes[vote.proposal_id])}/{quorum_count})"
        )

        if len(self.votes[vote.proposal_id]) >= quorum_count:
            return self._commit_proposal(vote.proposal_id)

        return False

    def _commit_proposal(self, proposal_id: str) -> bool:
        """Finalize the pattern as a system truth."""
        if proposal_id in self.committed_patterns:
            return False

        proposal = self.active_proposals[proposal_id]
        self.committed_patterns.add(proposal_id)

        logger.info(
            f"🏆 QUORUM REACHED: Pattern {proposal_id} committed to Giants Ledger."
        )

        if self.on_commit_broadcast:
            commit_payload = {
                "proposal_id": proposal_id,
                "pattern": proposal.pattern_data,
                "signatures": [v.signature for v in self.votes[proposal_id]],
            }
            self.on_commit_broadcast(commit_payload)

        return True
