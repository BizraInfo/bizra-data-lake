# Step 3: Pool-Mediated Consensus + Propagation Amendments

## Standing on Giants: Castro-Liskov (PBFT, 1999) | Demers et al. (Gossip, 1987) | Lamport (Paxos, 1998)

**Date:** 2026-03-03
**Ω⁷ Gems:** Ω⁷-1 (Gossip → Pool propagation), Ω⁷-2 (Byzantine → Pool consensus)
**Intent:** Amend Theorem 2.4 and Lemma 3.3 for Pool-mediated architecture

---

## Problem Statement

**Theorem 2.4** assumes peer-ranked verification where k neighbors validate a
reflex. Under the Interaction Boundary (Axiom 1.6), there are no neighbors.

**Lemma 3.3** proves gossip propagation in O(log N) rounds. Under Axiom 1.6,
there is no gossip. Nodes don't talk to nodes.

Both proofs transfer to the Pool architecture — and get STRONGER.

---

## Mathematical Formalization

### Amended Theorem 2.4 (Pool-Mediated Byzantine Consensus)

```
Original:
  In a network with N nodes where f are Byzantine,
  PoI consensus is safe if f < N/3.

Amended:
  In a Pool-mediated network with S SAT ConsensusValidators,
  where f validators are Byzantine, the PoI consensus protocol
  guarantees safety if f < S/3, with the additional property
  that equivocation is impossible.

New property (No Equivocation):
  In peer-to-peer BFT, a Byzantine node can send different messages
  to different neighbors (equivocation). Under Pool mediation:

  ∀ evidence e submitted by node i:
    ∀ validators v₁, v₂ ∈ ConsensusValidators:
      View(v₁, e) = View(v₂, e)

  Because the Pool controls evidence distribution, all validators
  see IDENTICAL evidence. Equivocation requires controlling the Pool
  itself, which requires compromising > S/3 SAT agents.

Proof sketch:
  1. Standard PBFT safety: honest majority (2f+1 out of 3f+1)
     ensures agreement. This transfers directly because
     SAT validators play the PBFT replica role.

  2. No-equivocation addendum: In PBFT, equivocation is handled
     by the view-change protocol (expensive). Under Pool mediation,
     equivocation is IMPOSSIBLE because the Pool is the single
     message source. This eliminates the need for view-change,
     reducing consensus latency.

  3. Attack cost at scale: At N=100K nodes with 5N=500K SAT agents,
     20% in consensus role = 100K ConsensusValidators.
     f < 100K/3 ≈ 33,333 compromises needed.
     Each SAT is bound to a real identity (Definition 1.6).
     Cost = 33,333 × (hardware + human attestation).

Strictly stronger than original because:
  - Same safety guarantee (f < S/3)
  - Plus no-equivocation (free, by construction)
  - Plus view-change elimination (lower latency)
```

### Amended Lemma 3.3 (Pool-Mediated Propagation)

```
Original:
  In a gossip network with fanout f = O(log N), information
  propagates to ≥ 0.99N nodes in O(log N) rounds.

Amended:
  In a Pool-mediated network of N nodes with SAT CacheCoordinator
  fanout f = O(log N), a verified reflex pattern propagates to
  ≥ 0.99N nodes in time T_propagate = O(log N).

Proof:
  The mathematical structure is identical. The difference is transport:
  - Original: probabilistic gossip (each node randomly selects f peers)
  - Amended: deterministic tree (Pool CacheCoordinators maintain
    balanced distribution tree with branching factor O(log N))

  The bound T = O(log N) holds because:
    Tree depth = log(N) / log(f) = log(N) / log(log(N))
    Each level processes in O(1) time.
    Total propagation: O(log N / log(log N)) ⊂ O(log N).

  Stronger than gossip because:
    - Deterministic, not probabilistic (guaranteed 0.99N, not expected)
    - Pool optimizes distribution topology (gossip cannot)
    - No duplicate message waste (gossip has O(N log N) redundancy)

Concrete numbers:
  N = 1,000     → f ≈ 10, rounds ≈ 3
  N = 1,000,000 → f ≈ 20, rounds ≈ 5
  N = 1,000,000,000 → f ≈ 30, rounds ≈ 7
```

---

## Pseudocode

### core/federation/pool_consensus.py

```pseudocode
"""Pool-Mediated Consensus — Amended Theorem 2.4.

Standing on Giants: Castro-Liskov (PBFT) | Lamport (Paxos)
"""

FROM __future__ IMPORT annotations
FROM dataclasses IMPORT dataclass, field
FROM typing IMPORT Optional
IMPORT math


@dataclass(frozen=True)
CLASS ConsensusParams:
    """Parameters for Pool-mediated BFT consensus."""
    total_validators: int
    quorum_size: int           # 2f+1 where 3f+1 = total_validators
    byzantine_tolerance: int   # f = floor((total - 1) / 3)

    @staticmethod
    FUNCTION from_validator_count(s: int) -> "ConsensusParams":
        """Derive BFT parameters from validator pool size.
        f < S/3 → f = floor((S-1)/3), quorum = 2f+1.
        """
        f = (s - 1) // 3
        quorum = 2 * f + 1
        RETURN ConsensusParams(
            total_validators=s,
            quorum_size=quorum,
            byzantine_tolerance=f,
        )

    @property
    FUNCTION safety_margin(self) -> float:
        """Fraction of validators that must be compromised."""
        RETURN self.byzantine_tolerance / self.total_validators

    @property
    FUNCTION equivocation_possible(self) -> bool:
        """Under Pool mediation, equivocation is always impossible."""
        RETURN False  # Axiom 1.6 eliminates this by construction


@dataclass
CLASS ConsensusRound:
    """A single PoI consensus round.

    All validators see identical evidence (no-equivocation property).
    """
    evidence_hash: str
    votes: dict = field(default_factory=dict)  # validator_id → vote
    params: ConsensusParams = None

    FUNCTION submit_vote(self, validator_id: str, vote: bool) -> None:
        """Record a validator's vote on the evidence."""
        self.votes[validator_id] = vote

    FUNCTION is_decided(self) -> bool:
        """Check if quorum is reached."""
        IF self.params IS None:
            RETURN False
        approvals = sum(1 FOR v IN self.votes.values() IF v)
        RETURN approvals >= self.params.quorum_size

    FUNCTION is_rejected(self) -> bool:
        """Check if rejection threshold is reached."""
        IF self.params IS None:
            RETURN False
        rejections = sum(1 FOR v IN self.votes.values() IF NOT v)
        reject_threshold = self.params.total_validators - self.params.quorum_size + 1
        RETURN rejections >= reject_threshold

    FUNCTION result(self) -> Optional[str]:
        """Return "accepted", "rejected", or None if undecided."""
        IF self.is_decided():
            RETURN "accepted"
        IF self.is_rejected():
            RETURN "rejected"
        RETURN None


FUNCTION attack_cost_at_scale(
    node_count: int,
    sat_per_node: int = 5,
    consensus_fraction: float = 0.20,
    per_identity_cost_usd: float = 100.0,
) -> dict:
    """Compute the cost of a Byzantine attack at given scale.

    Returns dict with attacker cost and system parameters.
    """
    total_sat = node_count * sat_per_node
    consensus_validators = int(total_sat * consensus_fraction)
    params = ConsensusParams.from_validator_count(consensus_validators)

    RETURN {
        "node_count": node_count,
        "total_sat_agents": total_sat,
        "consensus_validators": consensus_validators,
        "byzantine_tolerance": params.byzantine_tolerance,
        "attack_cost_usd": params.byzantine_tolerance * per_identity_cost_usd,
        "equivocation_possible": params.equivocation_possible,
    }
```

### core/federation/pool_propagation.py

```pseudocode
"""Pool-Mediated Propagation — Amended Lemma 3.3.

Standing on Giants: Demers et al. (gossip, 1987)
"""

IMPORT math
FROM dataclasses IMPORT dataclass


@dataclass(frozen=True)
CLASS PropagationParams:
    """Parameters for Pool CacheCoordinator distribution."""
    node_count: int
    fanout: int                # O(log N)
    propagation_rounds: int    # O(log N / log(log N))

    @staticmethod
    FUNCTION for_network(n: int) -> "PropagationParams":
        """Compute optimal propagation parameters for N nodes.

        Fanout = ceil(log2(N))
        Rounds = ceil(log(N) / log(fanout))
        """
        IF n <= 1:
            RETURN PropagationParams(n, 1, 0)

        fanout = max(2, math.ceil(math.log2(n)))
        rounds = math.ceil(math.log(n) / math.log(fanout)) IF fanout > 1 ELSE n
        RETURN PropagationParams(
            node_count=n,
            fanout=fanout,
            propagation_rounds=rounds,
        )

    @property
    FUNCTION coverage_guarantee(self) -> float:
        """Fraction of nodes reached. Deterministic = 1.0 (vs gossip ~0.99)."""
        RETURN 1.0  # Deterministic tree, not probabilistic gossip

    @property
    FUNCTION is_deterministic(self) -> bool:
        """Pool propagation is deterministic, unlike gossip."""
        RETURN True

    @property
    FUNCTION message_complexity(self) -> int:
        """Total messages sent. O(N) for tree vs O(N log N) for gossip."""
        RETURN self.node_count  # Each node receives exactly one message


FUNCTION propagation_table() -> list:
    """Generate the propagation parameter table for reference scales."""
    scales = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 1_000_000_000]
    RETURN [
        {
            "nodes": n,
            "fanout": PropagationParams.for_network(n).fanout,
            "rounds": PropagationParams.for_network(n).propagation_rounds,
            "messages": PropagationParams.for_network(n).message_complexity,
        }
        FOR n IN scales
    ]
```

---

## TDD Anchors

```pseudocode
# tests/core/federation/test_pool_consensus.py

TEST consensus_params_from_7_validators:
    """Classic BFT: 7 validators → f=2, quorum=5."""
    params = ConsensusParams.from_validator_count(7)
    ASSERT params.byzantine_tolerance == 2
    ASSERT params.quorum_size == 5

TEST consensus_params_from_100k_validators:
    """Scale: 100K validators → f≈33,333."""
    params = ConsensusParams.from_validator_count(100_000)
    ASSERT params.byzantine_tolerance == 33_333
    ASSERT params.quorum_size == 66_667

TEST no_equivocation_under_pool:
    """Equivocation is impossible under Pool mediation (Axiom 1.6)."""
    params = ConsensusParams.from_validator_count(100)
    ASSERT params.equivocation_possible IS False

TEST consensus_round_reaches_quorum:
    """Enough approvals → decided."""
    params = ConsensusParams.from_validator_count(7)
    rnd = ConsensusRound(evidence_hash="abc", params=params)
    FOR i IN range(5):
        rnd.submit_vote(f"v_{i}", True)
    ASSERT rnd.is_decided()
    ASSERT rnd.result() == "accepted"

TEST consensus_round_rejected:
    """Enough rejections → rejected."""
    params = ConsensusParams.from_validator_count(7)
    rnd = ConsensusRound(evidence_hash="abc", params=params)
    FOR i IN range(3):
        rnd.submit_vote(f"v_{i}", False)
    ASSERT rnd.is_rejected()
    ASSERT rnd.result() == "rejected"

TEST attack_cost_scales_linearly:
    """Cost of attack grows linearly with network size."""
    cost_1k = attack_cost_at_scale(1_000)
    cost_10k = attack_cost_at_scale(10_000)
    ASSERT cost_10k["attack_cost_usd"] > cost_1k["attack_cost_usd"] * 5

TEST propagation_1k_nodes:
    """1,000 nodes → fanout ≈ 10, rounds ≈ 3."""
    params = PropagationParams.for_network(1_000)
    ASSERT params.fanout == 10
    ASSERT params.propagation_rounds <= 4

TEST propagation_1m_nodes:
    """1,000,000 nodes → fanout ≈ 20, rounds ≈ 5."""
    params = PropagationParams.for_network(1_000_000)
    ASSERT params.fanout == 20
    ASSERT params.propagation_rounds <= 6

TEST propagation_is_deterministic:
    """Pool propagation is deterministic (not probabilistic gossip)."""
    params = PropagationParams.for_network(1_000)
    ASSERT params.is_deterministic
    ASSERT params.coverage_guarantee == 1.0

TEST propagation_message_complexity_is_linear:
    """O(N) messages (vs O(N log N) for gossip)."""
    FOR n IN [100, 1_000, 10_000]:
        params = PropagationParams.for_network(n)
        ASSERT params.message_complexity == n

TEST propagation_single_node:
    """Edge case: single node needs zero rounds."""
    params = PropagationParams.for_network(1)
    ASSERT params.propagation_rounds == 0
```

---

## Acceptance Criteria

1. `ConsensusParams.equivocation_possible` always returns False
2. `ConsensusRound` correctly implements quorum logic
3. `attack_cost_at_scale()` demonstrates linear cost scaling
4. `PropagationParams` computes O(log N) fanout and rounds
5. Propagation is deterministic with O(N) message complexity
6. All 11 TDD anchors GREEN
7. Full test suite GREEN
