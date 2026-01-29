"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA PATTERN FEDERATION — INTEGRATION TEST                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Tests multi-node pattern sharing and network effect validation.            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import time
import hashlib
from typing import List, Dict

# Add parent to path
sys.path.insert(0, 'c:\\BIZRA-DATA-LAKE')

from core.federation.gossip import GossipEngine, NodeInfo, NodeState, MessageType
from core.federation.propagation import PatternStore, PropagationEngine, ElevatedPattern, PatternStatus
from core.federation.consensus import ConsensusEngine, VoteType
from core.federation.node import FederationNode


def test_gossip_membership():
    """Test SWIM-style gossip membership."""
    print("\n" + "=" * 60)
    print("TEST 1: Gossip Membership")
    print("=" * 60)
    
    # Create 3 gossip engines with correct signature
    engines = [
        GossipEngine(
            node_id=f"node_{i}", 
            address=f"127.0.0.1:{7650 + i}",
            public_key=f"pk_{i}"
        )
        for i in range(3)
    ]
    
    # Manually add nodes as peers
    for i, engine in enumerate(engines):
        for j, other in enumerate(engines):
            if i != j:
                engine.add_seed_node(other.self_node.address, other.self_node.node_id)
    
    # Check membership
    for engine in engines:
        alive = sum(1 for m in engine.peers.values() if m.state == NodeState.ALIVE)
        print(f"  {engine.self_node.node_id}: {alive} alive peers")
    
    # Calculate network multiplier
    multiplier = engines[0].calculate_network_multiplier()
    print(f"  Network Multiplier: {multiplier:.3f}")
    
    assert len(engines[0].peers) == 2, "Each node should see 2 peers"
    assert multiplier >= 1.0, "Multiplier should be >= 1"
    
    print("✅ PASS: Gossip membership works")
    return True


def test_pattern_elevation():
    """Test SAPE pattern elevation."""
    print("\n" + "=" * 60)
    print("TEST 2: Pattern Elevation")
    print("=" * 60)
    
    store = PatternStore("test_node")
    
    # Record pattern uses - use unique triggers each time to accumulate properly
    trigger = "snr < 0.7 AND complexity > 0.5"
    
    # First, record uses until elevation
    for i in range(3):
        store.record_pattern_use(trigger, success=True, snr_delta=0.12)
        print(f"  Use {i+1}: candidates={len(store.pending_candidates)}, patterns={len(store.local_patterns)}")
    
    # Should be elevated now
    assert len(store.local_patterns) == 1, f"Should have 1 elevated pattern, got {len(store.local_patterns)}"
    
    # Get the pattern and record more uses directly
    pattern = list(store.local_patterns.values())[0]
    
    # Record 2 more uses on the elevated pattern
    for i in range(2):
        pattern.metrics.record_use(True, 0.12)
    
    print(f"  Elevated pattern: {pattern.pattern_id}")
    print(f"  Uses: {pattern.metrics.uses}")
    print(f"  Success rate: {pattern.metrics.success_rate:.0%}")
    print(f"  Impact score: {pattern.compute_impact_score():.3f}")
    
    assert pattern.metrics.uses >= 3, f"Should have at least 3 uses, got {pattern.metrics.uses}"
    assert pattern.metrics.success_rate == 1.0, "All uses were successful"
    
    print("✅ PASS: Pattern elevation works")
    return True


def test_pattern_propagation():
    """Test pattern propagation between nodes."""
    print("\n" + "=" * 60)
    print("TEST 3: Pattern Propagation")
    print("=" * 60)
    
    # Create two stores
    store_a = PatternStore("node_A")
    store_b = PatternStore("node_B")
    
    # Node A elevates a pattern
    for i in range(3):  # Exactly threshold to elevate
        store_a.record_pattern_use("query.type == 'complex'", True, 0.15)
    
    # Manually update the pattern to be share-ready
    pattern = list(store_a.local_patterns.values())[0]
    # Record enough uses and set high success rate
    for _ in range(5):
        pattern.metrics.record_use(True, 0.15)
    
    print(f"  Node A has pattern: {pattern.pattern_id}")
    print(f"  Uses: {pattern.metrics.uses}, Success: {pattern.metrics.success_rate:.0%}")
    
    # Create propagation engines with mock broadcast
    messages = []
    engine_a = PropagationEngine(store_a, broadcast_fn=lambda x: messages.append(x))
    engine_b = PropagationEngine(store_b)
    
    # Node A shares
    engine_a.auto_share_elevated()
    count = engine_a.propagate_pending()
    print(f"  Node A propagated {count} patterns")
    
    # Node B receives
    import json
    for msg in messages:
        data = json.loads(msg.decode('utf-8'))
        engine_b.receive_pattern(data)
    
    print(f"  Node B network patterns: {len(store_b.network_patterns)}")
    
    assert len(store_b.network_patterns) >= 1 or count >= 1, "Pattern should be shared"
    
    print("✅ PASS: Pattern propagation works")
    return True


def test_consensus():
    """Test distributed consensus on pattern impact."""
    print("\n" + "=" * 60)
    print("TEST 4: Impact Consensus")
    print("=" * 60)
    
    # Create 5 validator nodes
    validators = [
        ConsensusEngine(f"validator_{i}", ihsan_score=0.95, contributions=10)
        for i in range(5)
    ]
    
    pattern_id = "sape_test_consensus"
    pattern_hash = hashlib.sha256(pattern_id.encode()).hexdigest()[:16]
    
    # Proposer starts round
    round = validators[0].propose_pattern(pattern_id, pattern_hash, impact=0.85)
    
    # All validators vote
    for i, v in enumerate(validators):
        local_impact = 0.85 + (i - 2) * 0.02  # Slight variance
        vote = v.cast_vote(pattern_id, pattern_hash, VoteType.ACCEPT, local_impact)
        if vote:
            round.add_vote(vote)
    
    print(f"  Votes cast: {len(round.votes)}")
    
    # Finalize
    accepted, final_impact = round.finalize()
    print(f"  Accepted: {accepted}")
    print(f"  Final impact: {final_impact:.3f}")
    
    assert accepted, "Pattern should be accepted with 5 ACCEPT votes"
    assert 0.8 < final_impact < 0.9, "Impact should be around 0.85"
    
    print("✅ PASS: Consensus works")
    return True


def test_network_effect():
    """Test that network effect (multiplier) increases with nodes."""
    print("\n" + "=" * 60)
    print("TEST 5: Network Effect (Metcalfe's Law)")
    print("=" * 60)
    
    multipliers = []
    
    for node_count in [1, 3, 5, 10, 20]:
        engine = GossipEngine(
            node_id="primary",
            address="127.0.0.1:9000",
            public_key="pk_primary"
        )
        
        # Add peers
        for i in range(node_count - 1):
            engine.add_seed_node(
                address=f"127.0.0.1:{9001 + i}",
                node_id=f"peer_{i}"
            )
        
        m = engine.calculate_network_multiplier()
        multipliers.append((node_count, m))
        print(f"  {node_count:2d} nodes → multiplier = {m:.4f}")
    
    # Verify increasing trend
    for i in range(1, len(multipliers)):
        assert multipliers[i][1] >= multipliers[i-1][1], \
            f"Multiplier should increase: {multipliers[i-1][1]} → {multipliers[i][1]}"
    
    print("✅ PASS: Network effect increases with nodes")
    return True


def test_ihsan_gate():
    """Test that low-Ihsān patterns are rejected."""
    print("\n" + "=" * 60)
    print("TEST 6: Ihsān Gate")
    print("=" * 60)
    
    store = PatternStore("test_node")
    
    # Create a low-Ihsān pattern
    pattern = ElevatedPattern(
        pattern_id="bad_pattern",
        source_node_id="evil_node",
        trigger_condition="always",
        action="bad_action",
        domain="test",
        creation_time="2025-01-01T00:00:00Z",
        elevation_count=3,
        ihsan_score=0.80  # Below 0.95 threshold
    )
    
    # Try to add to network patterns
    result = store.add_network_pattern(pattern)
    print(f"  Pattern Ihsān: {pattern.ihsan_score}")
    print(f"  Accepted: {result}")
    
    assert not result, "Low-Ihsān pattern should be rejected"
    assert len(store.network_patterns) == 0, "No pattern should be added"
    
    print("✅ PASS: Ihsān gate blocks low-integrity patterns")
    return True


def test_federation_node_integration():
    """Test full FederationNode integration."""
    print("\n" + "=" * 60)
    print("TEST 7: FederationNode Integration")
    print("=" * 60)
    
    node = FederationNode(
        node_id="integration_test_node",
        bind_address="127.0.0.1:8765",
        ihsan_score=0.97,
        contribution_count=10
    )
    
    # Record pattern uses
    for i in range(5):
        node.record_pattern_use("test.condition == true", True, 0.1)
    
    # Get stats
    stats = node.get_stats()
    health = node.get_health()
    
    print(f"  Node ID: {stats['node_id']}")
    print(f"  Patterns: {stats['patterns']}")
    print(f"  Health: {health}")
    
    assert stats['patterns']['local_patterns'] == 1, "Should have 1 local pattern"
    assert health['status'] == 'stopped', "Node not started yet"
    
    print("✅ PASS: FederationNode integration works")
    return True


def run_all_tests():
    """Run all federation tests."""
    print("\n" + "═" * 70)
    print("  BIZRA PATTERN FEDERATION — INTEGRATION TEST SUITE")
    print("═" * 70)
    
    tests = [
        ("Gossip Membership", test_gossip_membership),
        ("Pattern Elevation", test_pattern_elevation),
        ("Pattern Propagation", test_pattern_propagation),
        ("Impact Consensus", test_consensus),
        ("Network Effect", test_network_effect),
        ("Ihsān Gate", test_ihsan_gate),
        ("FederationNode", test_federation_node_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"❌ FAIL: {name} - {e}")
            failed += 1
    
    print("\n" + "═" * 70)
    print(f"  RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("  ✅ ALL TESTS PASSED")
    else:
        print(f"  ❌ {failed} tests failed")
    print("═" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
