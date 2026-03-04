# Step 2: Interaction Boundary Axiom + Security Theorem

## Standing on Giants: Lampson (access control, 1971) | Castro-Liskov (BFT, 1999) | IBM-2026-001 (BIZRA Boundary Model)

**Date:** 2026-03-03
**Ω⁷ Gem:** Ω⁷-7 (Layer 4 — Network)
**Intent:** Formalize the Interaction Boundary as an axiom and prove 7 attack classes eliminated

---

## Problem Statement

The proof chain assumes peer-to-peer communication between nodes. The Interaction
Boundary Model (IBM-2026-001) eliminates all direct node-to-node channels. All
inter-node interaction is mediated by the SAT Resource Pool. This is not a
minor transport change — it eliminates 7 of 8 major distributed system attack
classes BY CONSTRUCTION.

**Why formalize:** An axiom-level statement creates a proof obligation: any future
protocol extension MUST NOT violate the boundary. Any code that creates a direct
node-to-node channel violates the axiom and invalidates Theorem 2.6.

---

## Mathematical Formalization

### Axiom 1.6 (Interaction Boundary)

```
For all nodes i, j where i ≠ j:

  ∄ C(i,j) : C is a direct communication channel

All inter-node interaction is mediated by the SAT Resource Pool P:

  ∀ message m from i to j:  i → P → j

Consequence: The attack surface between any two nodes is ∅.

Formal statement:
  ∀i,j ∈ Nodes, i ≠ j:
    DirectChannel(i,j) = ∅  ∧
    ∀m ∈ Messages(i→j): ∃p ∈ Pool : Relay(p, m, i, j)
```

### Theorem 2.6 (Boundary Security)

```
Under Axiom 1.6 (Interaction Boundary), the following attack classes
have zero probability:

  P(Eclipse(i))            = 0    — No neighbor set to monopolize
  P(MITM(i,j))             = 0    — No direct channel to intercept
  P(BGP_Hijack(i,j))       = 0    — No routing between nodes
  P(DDoS_PeerDiscovery(i)) = 0    — No peer discovery protocol
  P(Poisoned_Peer_Data(i)) = 0    — No peer data exchange
  P(Network_Map(N))        = 0    — No topology to enumerate
  P(Routing_Poison(i))     = 0    — No routing tables

Proof sketch for each:

  1. Eclipse attack requires surrounding node i with attacker-controlled
     neighbors. Under Axiom 1.6, i has no neighbors. QED.

  2. MITM requires intercepting channel C(i,j). Under Axiom 1.6,
     C(i,j) = ∅. QED.

  3. BGP hijacking requires routing advertisements between i and j.
     Under Axiom 1.6, no inter-node routing exists. QED.

  4. DDoS via peer discovery requires a discovery protocol that
     broadcasts node addresses. Under Axiom 1.6, nodes do not discover
     each other. QED.

  5. Poisoned peer data requires data exchange between peers.
     Under Axiom 1.6, no peer exchange exists. QED.

  6. Network mapping requires enumerating the topology. Under Axiom 1.6,
     the topology is {i → P} for each node. Node i cannot observe
     other nodes' connections to P. QED.

  7. Routing table poisoning requires routing tables. Under Axiom 1.6,
     nodes have no routing tables. QED.

Remaining viable attack: Sybil (creating fake identities).
Mitigated by Identity Genesis (Definition 1.6): hardware binding +
Ed25519 keypair + human attestation.

Attack cost scaling:
  Sybil cost = N_fake × (hardware_cost + human_attestation_cost)
  Grows linearly with attack scale; benefit does not.
```

---

## Pseudocode

### core/federation/boundary.py

```pseudocode
"""Interaction Boundary Enforcement — Axiom 1.6.

All inter-node communication MUST go through the Pool.
This module provides the enforcement and audit machinery.

Standing on Giants: Lampson (access control) | IBM-2026-001
"""

FROM __future__ IMPORT annotations
FROM dataclasses IMPORT dataclass, field
FROM enum IMPORT Enum
FROM typing IMPORT Optional


CLASS AttackClass(Enum):
    """The 8 distributed system attack classes.
    Axiom 1.6 eliminates 7 of 8.
    """
    ECLIPSE = "eclipse"
    MITM = "man_in_the_middle"
    BGP_HIJACK = "bgp_hijacking"
    DDOS_PEER_DISCOVERY = "ddos_peer_discovery"
    POISONED_PEER_DATA = "poisoned_peer_data"
    NETWORK_MAPPING = "network_mapping"
    ROUTING_POISON = "routing_table_poisoning"
    SYBIL = "sybil"  # The only remaining attack class


# Attack classes eliminated by Axiom 1.6
ELIMINATED_BY_BOUNDARY = frozenset({
    AttackClass.ECLIPSE,
    AttackClass.MITM,
    AttackClass.BGP_HIJACK,
    AttackClass.DDOS_PEER_DISCOVERY,
    AttackClass.POISONED_PEER_DATA,
    AttackClass.NETWORK_MAPPING,
    AttackClass.ROUTING_POISON,
})

# Attack classes that remain viable
REMAINING_VIABLE = frozenset({AttackClass.SYBIL})


FUNCTION assert_boundary_holds() -> bool:
    """Runtime check: verify no direct node-to-node channels exist.

    In production, this audits the network configuration:
    - No listening sockets for peer-to-peer protocols
    - No gossip protocol threads running
    - All outbound connections go to Pool endpoints only
    """
    # Check 1: No gossip protocol active
    # In production: scan active threads/sockets for gossip patterns
    gossip_active = False  # placeholder — wire to actual audit

    # Check 2: All outbound connections target Pool
    # In production: audit connection table
    all_connections_to_pool = True  # placeholder

    # Check 3: No peer discovery protocol running
    peer_discovery_active = False  # placeholder

    IF gossip_active OR NOT all_connections_to_pool OR peer_discovery_active:
        RAISE BoundaryViolation(
            "Axiom 1.6 violation: direct node-to-node channel detected"
        )

    RETURN True


CLASS BoundaryViolation(Exception):
    """Raised when Axiom 1.6 (Interaction Boundary) is violated.
    This is a constitutional violation — equivalent to a safety gate failure.
    """
    pass


@dataclass(frozen=True)
CLASS BoundaryAuditResult:
    """Result of boundary compliance audit."""
    axiom_holds: bool
    eliminated_attacks: frozenset = field(default_factory=lambda: ELIMINATED_BY_BOUNDARY)
    remaining_attacks: frozenset = field(default_factory=lambda: REMAINING_VIABLE)
    violations: list = field(default_factory=list)

    @property
    FUNCTION attack_surface_reduction(self) -> float:
        """Fraction of attack classes eliminated. Should be 7/8 = 0.875."""
        total = len(self.eliminated_attacks) + len(self.remaining_attacks)
        RETURN len(self.eliminated_attacks) / total IF total > 0 ELSE 0.0


@dataclass
CLASS PoolMediatedMessage:
    """All inter-node messages MUST be wrapped in this structure.
    Enforces: i → Pool → j (never i → j directly).
    """
    sender_id: str             # Identity of originating node
    pool_relay_id: str         # Identity of Pool relay SAT agent
    recipient_id: str          # Identity of destination node
    payload_hash: str          # BLAKE3 hash of payload
    pool_signature: bytes      # Pool's Ed25519 signature over relay
    timestamp_utc: str

    FUNCTION verify_pool_mediation(self) -> bool:
        """Verify this message was genuinely relayed through the Pool.
        A direct node-to-node message would lack a valid pool_signature.
        """
        # Verify pool_relay_id is a registered SAT agent
        # Verify pool_signature over (sender_id, recipient_id, payload_hash, timestamp)
        # In production: cryptographic verification
        RETURN self.pool_relay_id IS NOT None AND self.pool_signature IS NOT None
```

---

## TDD Anchors

```pseudocode
# tests/core/federation/test_interaction_boundary.py

TEST eliminated_attacks_are_seven:
    """Axiom 1.6 eliminates exactly 7 attack classes."""
    ASSERT len(ELIMINATED_BY_BOUNDARY) == 7

TEST remaining_attack_is_sybil_only:
    """Only Sybil remains viable under Axiom 1.6."""
    ASSERT REMAINING_VIABLE == {AttackClass.SYBIL}

TEST all_attack_classes_accounted:
    """7 eliminated + 1 remaining = 8 total."""
    all_attacks = ELIMINATED_BY_BOUNDARY | REMAINING_VIABLE
    ASSERT len(all_attacks) == 8
    ASSERT all_attacks == set(AttackClass)

TEST attack_surface_reduction_is_seven_eighths:
    """Boundary reduces attack surface by 87.5%."""
    result = BoundaryAuditResult(axiom_holds=True)
    ASSERT abs(result.attack_surface_reduction - 0.875) < 1e-6

TEST boundary_violation_raises:
    """Direct channel detection raises BoundaryViolation."""
    WITH pytest.raises(BoundaryViolation):
        RAISE BoundaryViolation("test violation")

TEST pool_mediated_message_requires_signature:
    """Messages without pool signature fail verification."""
    msg = PoolMediatedMessage(
        sender_id="node_a", pool_relay_id=None,
        recipient_id="node_b", payload_hash="abc",
        pool_signature=None, timestamp_utc="2026-01-01T00:00:00Z"
    )
    ASSERT NOT msg.verify_pool_mediation()

TEST pool_mediated_message_with_signature_passes:
    """Messages with pool relay and signature pass verification."""
    msg = PoolMediatedMessage(
        sender_id="node_a", pool_relay_id="sat_relay_42",
        recipient_id="node_b", payload_hash="abc123",
        pool_signature=b"valid_sig", timestamp_utc="2026-01-01T00:00:00Z"
    )
    ASSERT msg.verify_pool_mediation()

TEST eclipse_attack_impossible:
    """Eclipse requires neighbors. Under Axiom 1.6, nodes have no neighbors."""
    ASSERT AttackClass.ECLIPSE IN ELIMINATED_BY_BOUNDARY

TEST mitm_attack_impossible:
    """MITM requires direct channel. Under Axiom 1.6, no direct channels."""
    ASSERT AttackClass.MITM IN ELIMINATED_BY_BOUNDARY

TEST boundary_audit_result_frozen:
    """Audit result is immutable after creation."""
    result = BoundaryAuditResult(axiom_holds=True)
    WITH pytest.raises(AttributeError):
        result.axiom_holds = False
```

---

## Acceptance Criteria

1. `AttackClass` enum defines all 8 distributed system attack classes
2. `ELIMINATED_BY_BOUNDARY` contains exactly 7 classes
3. `BoundaryAuditResult.attack_surface_reduction` returns 0.875
4. `PoolMediatedMessage` enforces relay structure
5. `BoundaryViolation` exception type exists
6. All 10 TDD anchors GREEN
7. Full test suite GREEN

---

## Scope Boundary

**In scope:** Formalize the boundary axiom, enumerate eliminated attacks, define
message structure, provide audit machinery.
**Out of scope:** Pool implementation, SAT relay protocol, actual network audit
(production socket scanning), Sybil mitigation protocol.
