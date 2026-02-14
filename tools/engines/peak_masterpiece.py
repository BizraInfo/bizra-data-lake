"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA PEAK MASTERPIECE — SOVEREIGN ORGANISM v1.∞                           ║
║   "The Ultimate Implementation"                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Combines:                                                                  ║
║   1. Autonomous P2P Networking (FederationNode + Gossip)                     ║
║   2. Active Reasoning (ARTE Engine + Graph of Thoughts)                      ║
║   3. Epistemic Humility (Shoulders of Giants Protocol)                       ║
║   4. Pattern Elevation & Propagation (Swarm Intelligence)                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import asyncio
import time
import random
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict

# BIZRA Core Imports
sys.path.insert(0, "c:/BIZRA-DATA-LAKE")
from core.federation.node import FederationNode
from arte_engine import ARTEEngine, ShouldersOfGiantsProtocol

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("SOVEREIGN")


class SovereignOrganism:
    """
    The embodied BIZRA system.
    Unites the 'Brain' (ARTE) with the 'Body' (Node) to form a sovereign entity.
    """

    def __init__(self, node_id: str, port: int):
        self.node_id = node_id

        # 1. The Body (Networking)
        self.node = FederationNode(
            node_id=node_id, bind_address=f"127.0.0.1:{port}", ihsan_score=0.99
        )

        # 2. The Brain (Reasoning)
        self.brain = ARTEEngine()

        logger.info(f"✨ SOVEREIGN ORGANISM AWAKENED: {node_id}")

    async def live(self, max_cycles: int = 2):
        """Main lifecycle loop of the sovereign organism."""

        # Start Networking
        logger.info("🔌 Connecting to Federation...")
        await self.node.start()

        # Seed with initial patterns if empty (Self-Bootstrapping)
        if not self.node.pattern_store.local_patterns:
            logger.info("🌱 Seeding initial patterns...")
            # Simulate historical usage to elevate "Giants"
            # We must trigger it enough times to elevate it to a Giant (ElevatedPattern)
            for _ in range(4):  # Threshold is 3
                self.node.record_pattern_use(
                    trigger="query_type:latency", success=True, snr_delta=0.9
                )

        cycles = 0
        try:
            while cycles < max_cycles:
                cycles += 1
                logger.info(f"\n🧠 THINKING CYCLE {cycles} STARTED...")

                # 3. Sense: Retrieve patterns
                local_patterns = list(self.node.pattern_store.local_patterns.values())
                if not local_patterns:
                    logger.warning("Empty memory.")
                    break

                target = random.choice(local_patterns)
                stimulus = f"Verify pattern: {target.trigger_condition}"

                # 4. Reason: Construct Symbolic Knowledge from Pattern Store
                symbolic_knowledge = [
                    {
                        "doc_id": p.pattern_id,
                        "text": f"Pattern {p.trigger_condition} -> {p.action} (Ihsan: {p.ihsan_score})",
                    }
                    for p in local_patterns
                ]

                # Neural Intuition (Simulated for demo speed, usually via VectorDB)
                query_vec = np.random.rand(384).astype(np.float32)
                context_vecs = np.random.rand(len(local_patterns), 384).astype(
                    np.float32
                )
                neural_results = [
                    {
                        "chunk_id": f"chunk_{p.pattern_id}",
                        "doc_id": p.pattern_id,
                        "text": p.action,
                        "score": 0.8 + (random.random() * 0.1),
                    }
                    for p in local_patterns
                ]

                insight = self.brain.resolve_tension(
                    query=stimulus,
                    symbolic_facts=symbolic_knowledge,
                    neural_results=neural_results,
                    query_embedding=query_vec,
                    context_embeddings=context_vecs,
                )

                logger.info(f"💡 INSIGHT: {insight['conclusion'][:60]}...")

                # Check Protocol
                protocol = insight.get("giants_protocol", {})
                if protocol.get("valid"):
                    logger.info("✅ PROTOCOL VERIFIED")

                    # 5. Reinforce & Share
                    logger.info(f"📈 Reinforcing: '{target.trigger_condition}'")
                    self.node.record_pattern_use(target.trigger_condition, True, 0.1)

                    count = self.node.propagation.propagate_pending()
                    if count > 0:
                        logger.info(f"📡 Broadcast {count} reinforced patterns.")
                else:
                    logger.warning(f"❌ PROTOCOL VIOLATION: {protocol.get('failures')}")

                await asyncio.sleep(2)

            logger.info("🛑 Simulation complete.")

        except asyncio.CancelledError:
            logger.info("💤 Going to sleep...")
        finally:
            await self.node.stop()


if __name__ == "__main__":
    organism = SovereignOrganism("BIZRA_PRIME", 7777)
    try:
        asyncio.run(organism.live())
    except KeyboardInterrupt:
        pass
