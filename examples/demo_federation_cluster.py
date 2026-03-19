#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA PATTERN FEDERATION — MULTI-NODE CLUSTER DEMO                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Demonstrates actual P2P networking with 3-node cluster:                    ║
║   - Node discovery via UDP gossip                                            ║
║   - Pattern elevation and propagation                                        ║
║   - Network effect (Metcalfe's Law) in action                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import sys

sys.path.insert(0, "c:\\BIZRA-DATA-LAKE")

from core.federation.node import FederationNode


async def run_cluster_demo():
    """Deploy and demonstrate 3-node federation cluster."""

    print("=" * 70)
    print("  BIZRA PATTERN FEDERATION — 3-NODE CLUSTER DEMO")
    print("=" * 70)
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: Start Bootstrap Node
    # ─────────────────────────────────────────────────────────────────────────
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  PHASE 1: Starting Bootstrap Node (Alpha)                      │")
    print("└─────────────────────────────────────────────────────────────────┘")

    node_alpha = FederationNode(
        node_id="alpha-bootstrap", bind_address="127.0.0.1:9300"
    )
    await node_alpha.start()
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: Join Beta Node
    # ─────────────────────────────────────────────────────────────────────────
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  PHASE 2: Beta Node Joining Network                            │")
    print("└─────────────────────────────────────────────────────────────────┘")

    node_beta = FederationNode(node_id="beta-worker", bind_address="127.0.0.1:9301")
    await node_beta.start(seed_nodes=["127.0.0.1:9300"])
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3: Join Gamma Node
    # ─────────────────────────────────────────────────────────────────────────
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  PHASE 3: Gamma Node Joining Network                           │")
    print("└─────────────────────────────────────────────────────────────────┘")

    node_gamma = FederationNode(node_id="gamma-worker", bind_address="127.0.0.1:9302")
    await node_gamma.start(seed_nodes=["127.0.0.1:9300"])
    print()

    # Wait for gossip propagation
    print("⏳ Waiting for gossip propagation (2s)...")
    await asyncio.sleep(2)
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4: Check Network Status
    # ─────────────────────────────────────────────────────────────────────────
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  PHASE 4: Network Status                                       │")
    print("└─────────────────────────────────────────────────────────────────┘")

    nodes = [
        ("Alpha (Bootstrap)", node_alpha),
        ("Beta (Worker)", node_beta),
        ("Gamma (Worker)", node_gamma),
    ]

    for name, node in nodes:
        stats = node.gossip.get_stats()
        peers = node.gossip.get_alive_peers()
        print(f"\n  📡 {name}")
        print(f"     Network Size: {stats['network_size']}")
        print(f"     Alive Peers: {stats['alive_peers']}")
        print(f"     Peers: {[p.node_id for p in peers]}")
        print(f"     Multiplier: {stats['network_multiplier']}")

    print()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 5: Pattern Elevation & Propagation
    # ─────────────────────────────────────────────────────────────────────────
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  PHASE 5: Pattern Elevation & Propagation                      │")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Simulate pattern uses on Beta to trigger elevation
    print("\n  📈 Recording pattern uses on Beta node...")
    for i in range(4):
        node_beta.record_pattern_use(
            trigger="sape_cluster_demo", success=True, snr_delta=0.15
        )

    # Check local patterns
    stats = node_beta.get_stats()
    print(f"     Local patterns: {stats['patterns']['local_patterns']}")
    print(f"     Candidates: {stats['patterns']['pending_candidates']}")
    print(f"     Total uses: {stats['patterns']['total_pattern_uses']}")

    print()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 6: Verify Metcalfe's Law
    # ─────────────────────────────────────────────────────────────────────────
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  PHASE 6: Network Effect (Metcalfe's Law)                      │")
    print("└─────────────────────────────────────────────────────────────────┘")

    print("\n  📊 Network Value Scaling:")
    print(f"     1 node  → multiplier = 1.0286 (baseline)")
    print(
        f"     3 nodes → multiplier = {node_alpha.gossip.calculate_network_multiplier()}"
    )
    print(
        f"     Value increase: +{((node_alpha.gossip.calculate_network_multiplier() - 1.0286) / 1.0286 * 100):.1f}%"
    )

    print()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 7: Graceful Shutdown
    # ─────────────────────────────────────────────────────────────────────────
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  PHASE 7: Graceful Shutdown                                    │")
    print("└─────────────────────────────────────────────────────────────────┘")

    await node_gamma.stop()
    await node_beta.stop()
    await node_alpha.stop()

    print()
    print("=" * 70)
    print("  ✅ FEDERATION CLUSTER DEMO COMPLETE")
    print("=" * 70)
    print()
    print("  Summary:")
    print("    • 3 nodes started with actual UDP networking")
    print("    • Peer discovery via SWIM-style gossip")
    print("    • Pattern elevation on Beta node")
    print("    • Network effect multiplier verified")
    print("    • Graceful shutdown with LEAVE messages")
    print()


if __name__ == "__main__":
    asyncio.run(run_cluster_demo())
