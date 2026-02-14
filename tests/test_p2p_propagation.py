import asyncio
import sys
import time
import json
from pathlib import Path

# Add project root to path (portable)
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from core.federation.node import FederationNode
from core.federation.propagation import ElevatedPattern


async def test_p2p_propagation():
    print("=" * 60)
    print("🧬 P2P PATTERN PROPAGATION TEST (DEBUG)")
    print("=" * 60)

    # Node A: Source
    node_a = FederationNode(
        node_id="node_a_opt", bind_address="127.0.0.1:7656", ihsan_score=0.99
    )

    # Node B: Receiver
    node_b = FederationNode(
        node_id="node_b_learn", bind_address="127.0.0.1:7657", ihsan_score=0.98
    )

    print("\n[1] Starting nodes...")
    await node_a.start()
    await node_b.start(seed_nodes=["127.0.0.1:7656"])

    print("\n[2] Waiting for gossip convergence (3s)...")
    await asyncio.sleep(3)

    # Create an elevated pattern on Node A manually
    print("\n[3] Elevating pattern on Node A...")
    node_a.record_pattern_use("test_trigger > 0.9", True, 0.8)
    node_a.record_pattern_use("test_trigger > 0.9", True, 0.8)  # 2
    node_a.record_pattern_use("test_trigger > 0.9", True, 0.8)  # 3 -> Elevate

    # Use it more to meet sharing threshold (uses >= ELEVATION_THRESHOLD)
    print("    Generating additional usage for sharing qualification...")
    for _ in range(3):
        node_a.record_pattern_use("test_trigger > 0.9", True, 0.8)

    # DEBUG: Check pattern state
    try:
        patterns = list(node_a.pattern_store.local_patterns.values())
        if patterns:
            p = patterns[0]
            print(
                f"    DEBUG: Pattern {p.pattern_id} status={p.status} uses={p.metrics.uses} success={p.metrics.success_rate} ihsan={p.ihsan_score}"
            )
        else:
            print("    DEBUG: No local patterns found!")
    except Exception as e:
        print(f"    DEBUG ERROR: {e}")

    # Force propagation (usually happens on timer)
    print("\n[4] Forcing propagation from Node A...")
    node_a.propagation.auto_share_elevated()
    count = node_a.propagation.propagate_pending()
    print(f"    Propagated {count} patterns")

    print("\n[5] Waiting for network delivery (3s)...")
    await asyncio.sleep(3)

    # Check Node B pattern store
    print("\n[6] Checking Node B pattern store...")
    network_patterns = node_b.pattern_store.network_patterns
    print(f"    Node B has {len(network_patterns)} network patterns")

    success = False
    for pid, p in network_patterns.items():
        print(f"    - Found pattern: {pid} ({p.trigger_condition})")
        if p.trigger_condition == "test_trigger > 0.9":
            success = True

    if success:
        print("\n✅ PATTERN PROPAGATION SUCCESSFUL!")
    else:
        print("\n❌ PATTERN PROPAGATION FAILED")

    print("\n[7] Shutting down...")
    await node_b.stop()
    await node_a.stop()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(test_p2p_propagation())
    except KeyboardInterrupt:
        pass
