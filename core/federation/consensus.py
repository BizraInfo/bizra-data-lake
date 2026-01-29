"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA PATTERN FEDERATION — IMPACT CONSENSUS                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Distributed consensus on pattern impact scores.                            ║
║                                                                              ║
║   Consensus Model: Weighted voting based on node reputation                  ║
║   - Each node votes on pattern impact                                        ║
║   - Votes weighted by node's Ihsān average and contribution count            ║
║   - Pattern accepted if weighted consensus ≥ 0.67                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# CONSENSUS MODEL DECLARATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# HONEST LABELING — This is NOT PBFT/BFT
#
# Model: WEIGHTED_VOTING_IHSAN
# - Votes weighted by node reputation (Ihsan score × contribution factor)
# - Assumes honest participants (no Byzantine fault tolerance)
# - No network partition testing has been performed
# - Single-operator genesis mode
#
# Future: Tendermint BFT at Node count >= 4 with partition testing
#
# ═══════════════════════════════════════════════════════════════════════════════

CONSENSUS_MODEL = "WEIGHTED_VOTING_IHSAN"  # NOT "PBFT" — we don't claim what we haven't tested

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

QUORUM_THRESHOLD = 0.67          # 67% weighted consensus required
MIN_VOTERS = 3                   # Minimum validators for consensus
VOTE_TIMEOUT_SECONDS = 300       # 5 minute voting window
MAX_IMPACT_VARIANCE = 0.2        # Max allowed variance in impact scores
BFT_ENABLED = False              # True only after multi-node partition testing

# ═══════════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class VoteType(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"

@dataclass
class Vote:
    """A single validator's vote on a pattern."""
    voter_node_id: str
    pattern_id: str
    vote: VoteType
    local_impact_score: float    # Voter's measured impact
    signature: str               # Ed25519 signature
    timestamp: str
    
    # Voter's credentials (for weighting)
    voter_ihsan: float = 0.95
    voter_contributions: int = 0
    
    def compute_weight(self) -> float:
        """
        Calculate vote weight based on voter reputation.
        Weight = ihsan × (1 + log(1 + contributions) / 10)
        """
        import math
        contribution_factor = 1 + math.log10(1 + self.voter_contributions) / 10
        return self.voter_ihsan * contribution_factor

@dataclass
class ConsensusRound:
    """A single consensus round for a pattern."""
    pattern_id: str
    pattern_hash: str
    proposer_node_id: str
    proposed_impact: float
    
    start_time: float = field(default_factory=time.time)
    votes: List[Vote] = field(default_factory=list)
    
    finalized: bool = False
    accepted: bool = False
    final_impact: float = 0.0
    
    def add_vote(self, vote: Vote) -> bool:
        """Add a vote to this round. Returns False if duplicate."""
        if any(v.voter_node_id == vote.voter_node_id for v in self.votes):
            return False
        self.votes.append(vote)
        return True
    
    def is_expired(self) -> bool:
        return time.time() - self.start_time > VOTE_TIMEOUT_SECONDS
    
    def compute_consensus(self) -> Tuple[bool, float, str]:
        """
        Compute weighted consensus.
        Returns (accepted, final_impact, reason)
        """
        if len(self.votes) < MIN_VOTERS:
            return False, 0.0, f"Insufficient voters ({len(self.votes)} < {MIN_VOTERS})"
        
        # Calculate weighted votes
        total_weight = 0.0
        accept_weight = 0.0
        impact_sum = 0.0
        impact_weights = 0.0
        
        for vote in self.votes:
            weight = vote.compute_weight()
            total_weight += weight
            
            if vote.vote == VoteType.ACCEPT:
                accept_weight += weight
                impact_sum += vote.local_impact_score * weight
                impact_weights += weight
        
        if total_weight == 0:
            return False, 0.0, "No valid votes"
        
        accept_ratio = accept_weight / total_weight
        
        if accept_ratio < QUORUM_THRESHOLD:
            return False, 0.0, f"Quorum not reached ({accept_ratio:.2%} < {QUORUM_THRESHOLD:.0%})"
        
        # Calculate weighted average impact
        final_impact = impact_sum / impact_weights if impact_weights > 0 else 0.0
        
        # Check variance
        impacts = [v.local_impact_score for v in self.votes if v.vote == VoteType.ACCEPT]
        if impacts:
            variance = max(impacts) - min(impacts)
            if variance > MAX_IMPACT_VARIANCE:
                return False, 0.0, f"Impact variance too high ({variance:.2f} > {MAX_IMPACT_VARIANCE})"
        
        return True, final_impact, "Consensus reached"
    
    def finalize(self) -> Tuple[bool, float]:
        """Finalize the consensus round."""
        accepted, impact, reason = self.compute_consensus()
        self.finalized = True
        self.accepted = accepted
        self.final_impact = impact
        return accepted, impact


# ═══════════════════════════════════════════════════════════════════════════════
# CONSENSUS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ConsensusEngine:
    """
    Manages distributed consensus for pattern validation.
    """
    
    def __init__(self, node_id: str, ihsan_score: float = 0.95, contributions: int = 0):
        self.node_id = node_id
        self.ihsan_score = ihsan_score
        self.contributions = contributions
        
        self.active_rounds: Dict[str, ConsensusRound] = {}
        self.completed_rounds: Dict[str, ConsensusRound] = {}
        
        # Cache of our votes to avoid double-voting
        self._my_votes: Set[str] = set()
    
    def propose_pattern(self, pattern_id: str, pattern_hash: str, impact: float) -> ConsensusRound:
        """
        Start a new consensus round for a pattern.
        """
        round_key = f"{pattern_id}:{pattern_hash}"
        
        if round_key in self.active_rounds:
            return self.active_rounds[round_key]
        
        round = ConsensusRound(
            pattern_id=pattern_id,
            pattern_hash=pattern_hash,
            proposer_node_id=self.node_id,
            proposed_impact=impact
        )
        
        self.active_rounds[round_key] = round
        print(f"📋 Consensus round started for pattern {pattern_id}")
        return round
    
    def cast_vote(
        self, 
        pattern_id: str, 
        pattern_hash: str,
        vote_type: VoteType,
        local_impact: float,
        sign_fn=None
    ) -> Optional[Vote]:
        """
        Cast our vote on a pattern.
        """
        round_key = f"{pattern_id}:{pattern_hash}"
        
        if round_key in self._my_votes:
            print(f"⚠️ Already voted on {pattern_id}")
            return None
        
        # Create signature (mock if no sign_fn provided)
        signature = "mock_signature"
        if sign_fn:
            vote_data = f"{pattern_id}:{vote_type.value}:{local_impact}"
            signature = sign_fn(vote_data)
        
        vote = Vote(
            voter_node_id=self.node_id,
            pattern_id=pattern_id,
            vote=vote_type,
            local_impact_score=local_impact,
            signature=signature,
            timestamp=datetime.now(timezone.utc).isoformat(),
            voter_ihsan=self.ihsan_score,
            voter_contributions=self.contributions
        )
        
        self._my_votes.add(round_key)
        
        # Add to active round if exists
        if round_key in self.active_rounds:
            self.active_rounds[round_key].add_vote(vote)
        
        return vote
    
    def receive_vote(self, vote: Vote) -> bool:
        """
        Receive a vote from another node.
        """
        round_key = f"{vote.pattern_id}:{vote.pattern_id}"  # Simplified key
        
        # Find or create round
        if round_key not in self.active_rounds:
            # Create round if we're receiving votes for unknown pattern
            self.active_rounds[round_key] = ConsensusRound(
                pattern_id=vote.pattern_id,
                pattern_hash=vote.pattern_id,
                proposer_node_id=vote.voter_node_id,
                proposed_impact=vote.local_impact_score
            )
        
        return self.active_rounds[round_key].add_vote(vote)
    
    def check_and_finalize(self) -> List[Tuple[str, bool, float]]:
        """
        Check all active rounds and finalize those ready.
        Returns list of (pattern_id, accepted, final_impact)
        """
        results = []
        to_remove = []
        
        for round_key, round in self.active_rounds.items():
            if round.finalized:
                continue
            
            if round.is_expired() or len(round.votes) >= MIN_VOTERS:
                accepted, impact = round.finalize()
                results.append((round.pattern_id, accepted, impact))
                
                self.completed_rounds[round_key] = round
                to_remove.append(round_key)
                
                status = "✅ ACCEPTED" if accepted else "❌ REJECTED"
                print(f"{status} Pattern {round.pattern_id} (impact={impact:.3f})")
        
        for key in to_remove:
            del self.active_rounds[key]
        
        return results
    
    def get_stats(self) -> Dict:
        return {
            "active_rounds": len(self.active_rounds),
            "completed_rounds": len(self.completed_rounds),
            "accepted_patterns": sum(1 for r in self.completed_rounds.values() if r.accepted),
            "rejected_patterns": sum(1 for r in self.completed_rounds.values() if not r.accepted),
            "my_votes_cast": len(self._my_votes)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("BIZRA IMPACT CONSENSUS — Simulation")
    print("=" * 70)
    
    # Create 5 validator nodes
    validators = [
        ConsensusEngine(f"validator_{i}", ihsan_score=0.95 + i*0.01, contributions=i*10)
        for i in range(5)
    ]
    
    # Node 0 proposes a pattern
    proposer = validators[0]
    pattern_id = "sape_test_001"
    pattern_hash = hashlib.sha256(pattern_id.encode()).hexdigest()[:16]
    
    print(f"\n[Proposer] Starting consensus for pattern {pattern_id}")
    round = proposer.propose_pattern(pattern_id, pattern_hash, impact=0.85)
    
    # Other nodes vote
    print("\n[Validators] Casting votes...")
    for i, validator in enumerate(validators[1:], 1):
        # Simulate slight variation in measured impact
        local_impact = 0.85 + (i - 2) * 0.02  # 0.83, 0.85, 0.87, 0.89
        vote = validator.cast_vote(pattern_id, pattern_hash, VoteType.ACCEPT, local_impact)
        
        if vote:
            round.add_vote(vote)
            print(f"  Validator {i}: ACCEPT (impact={local_impact:.2f}, weight={vote.compute_weight():.3f})")
    
    # Finalize
    print("\n[Consensus] Finalizing...")
    accepted, final_impact = round.finalize()
    
    if accepted:
        print(f"✅ Pattern ACCEPTED with consensus impact: {final_impact:.3f}")
    else:
        _, _, reason = round.compute_consensus()
        print(f"❌ Pattern REJECTED: {reason}")
    
    # Show stats
    print("\n[Stats]")
    print(f"  Votes cast: {len(round.votes)}")
    print(f"  Total weight: {sum(v.compute_weight() for v in round.votes):.3f}")
    
    print("\n" + "=" * 70)
    print("✅ Impact Consensus Demo Complete")
    print("=" * 70)
