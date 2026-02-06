import asyncio
import sys
import time
from pathlib import Path

# Add project root to path (portable)
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from core.federation.node import FederationNode


async def test_p2p_handshake():
    print("=" * 60)
    print("🧪 P2P REAL NETWORKING TEST")
    print("=" * 60)

    # Node A: Seed
    node_a = FederationNode(
        node_id="node_a_seed", bind_address="127.0.0.1:7654", ihsan_score=0.99
    )

    # Node B: Peer
    node_b = FederationNode(
        node_id="node_b_peer",
        bind_address="127.0.0.1:7655",  # Different port
        ihsan_score=0.98,
    )

    print("\n[1] Starting Node A (Seed)...")
    await node_a.start()

    print("\n[2] Starting Node B and joining Node A...")
    await node_b.start(seed_nodes=["127.0.0.1:7654"])

    # Wait for gossip to happen (ping/ack/announce)
    print("\n[3] Waiting for gossip convergence (5s)...")
    await asyncio.sleep(5)

    # Check Peers
    peers_a = len(node_a.gossip.get_alive_peers())
    peers_b = len(node_b.gossip.get_alive_peers())

    print(f"\n[4] Node A has {peers_a} peers")
    print(f"      - {', '.join([p.node_id for p in node_a.gossip.get_alive_peers()])}")

    print(f"    Node B has {peers_b} peers")
    print(f"      - {', '.join([p.node_id for p in node_b.gossip.get_alive_peers()])}")

    success = (peers_a >= 1) and (peers_b >= 1)

    if success:
        print("\n✅ P2P CONNECTION SUCCESSFUL!")
    else:
        print("\n❌ P2P CONNECTION FAILED")

    print("\n[5] Shutting down...")
    await node_b.stop()
    await node_a.stop()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(test_p2p_handshake())
    except KeyboardInterrupt:
        pass
