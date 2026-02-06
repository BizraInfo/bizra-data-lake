"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA PATTERN FEDERATION — FEDERATION NODE                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Main integration class for P2P pattern federation.                         ║
║                                                                              ║
║   Components:                                                                ║
║   - GossipEngine: Node discovery and health monitoring                       ║
║   - PatternStore: Local pattern storage and elevation                        ║
║   - PropagationEngine: Pattern broadcast                                     ║
║   - ConsensusEngine: Distributed validation                                  ║
║                                                                              ║
║   Network Effect: Value ∝ n² (Metcalfe's Law)                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import logging
import uuid
from typing import Callable, Dict, List, Optional

from .consensus import ConsensusEngine, Proposal, Vote
from .gossip import GossipEngine
from .propagation import ElevatedPattern, PatternStatus, PatternStore, PropagationEngine

logger = logging.getLogger("FEDERATION")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SYNC_INTERVAL_SECONDS = 60  # Pattern sync every minute
CONSENSUS_CHECK_INTERVAL = 30  # Check consensus every 30s
NETWORK_MULTIPLIER_MIN = 1.0  # Base multiplier (single node)

# ═══════════════════════════════════════════════════════════════════════════════
# FEDERATION NODE
# ═══════════════════════════════════════════════════════════════════════════════


class FederationNode:
    """
    Main class for participating in the BIZRA pattern federation network.

    This is where Metcalfe's Law comes alive:
    - More nodes → more patterns discovered
    - More patterns → better collective intelligence
    - Better intelligence → more value per node
    - More value → attracts more nodes
    → Positive feedback loop (Network Effect)
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        bind_address: str = "0.0.0.0:7654",
        public_key: str = "",
        private_key: str = "",
        ihsan_score: float = 0.95,
        contribution_count: int = 0,
    ):
        self.node_id = node_id or f"bizra_{uuid.uuid4().hex[:8]}"
        self.bind_address = bind_address
        self.ihsan_score = ihsan_score
        self.contribution_count = contribution_count

        # Generate keypair if not provided (SEC-016/017 compliance)
        if not public_key or not private_key or len(public_key) < 64:
            from core.pci import generate_keypair

            private_key, public_key = generate_keypair()
            print(f"[FederationNode] Generated Ed25519 keypair for {self.node_id}")

        self.public_key = public_key
        self.private_key = private_key

        # Core components - pass private_key for message signing (SEC-016)
        self.gossip = GossipEngine(
            node_id=self.node_id,
            address=bind_address,
            public_key=self.public_key,
            private_key=self.private_key,
            on_pattern_received=self._on_pattern_received,
        )

        self.pattern_store = PatternStore(self.node_id)
        # SEC-016: Pass PCI credentials to PropagationEngine for envelope signing
        self.propagation = PropagationEngine(
            self.pattern_store,
            broadcast_fn=self._broadcast_pattern,
            node_id=self.node_id,
            private_key=self.private_key,
            public_key=self.public_key,
        )
        self.consensus = ConsensusEngine(
            self.node_id, private_key=self.private_key, public_key=self.public_key
        )

        # State
        self._running = False
        self._message_handlers: Dict[str, Callable] = {}
        self._pending_votes: List[Vote] = []

        # Register message handlers
        self._register_handlers()
        self._setup_consensus_callbacks()

    def _on_pattern_received(self, data: Dict):
        """Callback when gossip receives a pattern."""
        msg_type = data.get("type", "PATTERN_PROPAGATE")
        if msg_type in self._message_handlers:
            self._message_handlers[msg_type](data)

    def _register_handlers(self):
        """Register handlers for different message types."""
        self._message_handlers = {
            "PATTERN_PROPAGATE": self._handle_pattern_propagate,
            "PROPOSE": self._handle_propose,
            "VOTE": self._handle_vote,
            "COMMIT": self._handle_commit,
            "PATTERN_REQUEST": self._handle_pattern_request,
        }

    def _setup_consensus_callbacks(self):
        """Setup callbacks for consensus engine to broadcast commits."""

        def on_commit(payload: Dict):
            """Broadcast commit certificate to network."""
            self._broadcast_consensus_msg("COMMIT", payload)

        self.consensus.on_commit_broadcast = on_commit

    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════

    async def start(self, seed_nodes: Optional[List[str]] = None):
        """
        Start the federation node with P2P networking.
        Binds UDP socket and joins the gossip network.
        """
        print(f"🚀 Starting FederationNode {self.node_id}")
        print(f"   Address: {self.bind_address}")

        self._running = True

        # Start gossip engine with UDP networking
        await self.gossip.start()

        # Join network via seed nodes
        if seed_nodes:
            for seed in seed_nodes:
                host, port = seed.split(":")
                await self.gossip.join_network(host, int(port))
                print(f"   Joined via seed: {seed}")

        # Start background tasks
        asyncio.create_task(self._pattern_sync_loop())
        asyncio.create_task(self._consensus_check_loop())

        print(f"✅ FederationNode {self.node_id} started (P2P ENABLED)")

    async def stop(self):
        """Gracefully shutdown the node."""
        print(f"🛑 Stopping FederationNode {self.node_id}")
        self._running = False
        # Stop gossip engine and close network connections
        await self.gossip.stop()
        print(f"✅ FederationNode {self.node_id} stopped")

    # ═══════════════════════════════════════════════════════════════════════════
    # PATTERN OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def record_pattern_use(self, trigger: str, success: bool, snr_delta: float):
        """
        Record a pattern trigger (e.g., from SAPE).
        May result in automatic elevation.
        """
        self.pattern_store.record_pattern_use(trigger, success, snr_delta)

    def get_applicable_patterns(self, context: Dict) -> List[ElevatedPattern]:
        """Get patterns that apply to the current context."""
        return self.pattern_store.get_applicable_patterns(context)

    def share_pattern(self, pattern: ElevatedPattern):
        """
        Manually share a pattern with the network.
        Queues for propagation.
        """
        self.propagation.queue_for_propagation(pattern)

    # ═══════════════════════════════════════════════════════════════════════════
    # MESSAGE HANDLING
    # ═══════════════════════════════════════════════════════════════════════════

    def _broadcast_pattern(self, data: bytes):
        """Broadcast pattern data to peers via gossip."""
        # Parse and forward through gossip engine
        try:
            pattern_data = json.loads(
                data.decode("utf-8") if isinstance(data, bytes) else data
            )
            # Schedule async broadcast
            asyncio.create_task(self.gossip.broadcast_pattern_async(pattern_data))
        except Exception as e:
            print(f"⚠️ Broadcast failed: {e}")

    def _handle_pattern_propagate(self, payload: Dict):
        """Handle incoming pattern from the network."""
        # Use propagation engine to validate and store network pattern
        success = self.propagation.receive_pattern(payload)
        if success:
            logger.info(
                f"📥 Received pattern from network: {payload.get('pattern_id', 'unknown')}"
            )

    def _handle_propose(self, payload: Dict):
        """Handle incoming BFT proposal."""
        proposal = Proposal(**payload)
        self.consensus.active_proposals[proposal.proposal_id] = proposal

        # Auto-validate and vote
        pattern_data = proposal.pattern_data
        # Simulated Ihsan check (in real app, this would use arte_engine)
        ihsan = pattern_data.get("ihsan", 0.95)

        vote = self.consensus.cast_vote(proposal, ihsan)
        if vote:
            self._broadcast_consensus_msg("VOTE", vote.__dict__)

    def _handle_vote(self, payload: Dict):
        """Handle incoming BFT vote."""
        vote = Vote(**payload)
        alive_count = self.gossip.get_network_size()
        self.consensus.receive_vote(vote, alive_count)

    def _handle_commit(self, payload: Dict):
        """Handle incoming BFT commit certificate."""
        pattern_id = payload.get("proposal_id")
        if pattern_id in self.consensus.active_proposals:
            self.pattern_store.network_patterns[pattern_id].status = (
                PatternStatus.VALIDATED
            )
            logger.info(f"✅ Pattern {pattern_id} validated by Global Consensus")

    def _broadcast_consensus_msg(self, msg_type: str, data: Dict):
        """Helper to broadcast BFT messages."""
        msg = {"type": msg_type, **data}
        self._broadcast_pattern(json.dumps(msg).encode("utf-8"))

    def _handle_pattern_request(self, payload: Dict):
        """Handle request for patterns from a peer."""
        # Return our local patterns
        patterns = [p.to_dict() for p in self.pattern_store.local_patterns.values()]
        return {"patterns": patterns}

    # ═══════════════════════════════════════════════════════════════════════════
    # BACKGROUND TASKS
    # ═══════════════════════════════════════════════════════════════════════════

    async def _pattern_sync_loop(self):
        """Periodically share local patterns with the network."""
        while self._running:
            await asyncio.sleep(SYNC_INTERVAL_SECONDS)

            # Auto-share eligible patterns
            self.propagation.auto_share_elevated()
            count = self.propagation.propagate_pending()

            if count > 0:
                print(f"📡 Synced {count} patterns to network")

    async def _consensus_check_loop(self):
        """Periodically check and finalize consensus rounds."""
        while self._running:
            await asyncio.sleep(CONSENSUS_CHECK_INTERVAL)

            # Broadcast pending votes
            for vote in self._pending_votes:
                msg = json.dumps({"type": "PATTERN_VOTE", **vote.__dict__})
                self._broadcast_pattern(msg.encode("utf-8"))
            self._pending_votes.clear()

            # Check for completed rounds
            results = self.consensus.check_and_finalize()

            for pattern_id, accepted, impact in results:
                if accepted:
                    self.contribution_count += 1
                    # Broadcast acceptance
                    msg = json.dumps(
                        {
                            "type": "PATTERN_ACCEPTED",
                            "pattern_id": pattern_id,
                            "final_impact": impact,
                        }
                    )
                    self._broadcast_pattern(msg.encode("utf-8"))

    # ═══════════════════════════════════════════════════════════════════════════
    # NETWORK METRICS
    # ═══════════════════════════════════════════════════════════════════════════

    def get_network_multiplier(self) -> float:
        """
        Calculate network effect multiplier.
        M = 1 + (log₁₀(n + 1) / 10) × D × I

        Where:
        - n = number of alive nodes
        - D = pattern diversity (unique domains)
        - I = average Ihsān
        """
        return self.gossip.calculate_network_multiplier()

    def get_stats(self) -> Dict:
        """Get comprehensive node statistics."""
        return {
            "node_id": self.node_id,
            "ihsan_score": self.ihsan_score,
            "contributions": self.contribution_count,
            "network_multiplier": self.get_network_multiplier(),
            "gossip": self.gossip.get_stats(),
            "patterns": self.pattern_store.get_stats(),
            "consensus": self.consensus.get_stats(),
            "uptime_seconds": 0,  # Simplified for now
        }

    def get_health(self) -> Dict:
        """Quick health check."""
        alive_nodes = len(self.gossip.get_alive_peers())

        return {
            "status": "healthy" if self._running else "stopped",
            "alive_peers": alive_nodes,
            "local_patterns": len(self.pattern_store.local_patterns),
            "network_patterns": len(self.pattern_store.network_patterns),
            "network_multiplier": self.get_network_multiplier(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SYNC API (for non-async contexts)
# ═══════════════════════════════════════════════════════════════════════════════


class SyncFederationNode:
    """
    Synchronous wrapper for FederationNode.
    Use when not in an async context.
    """

    def __init__(self, *args, **kwargs):
        self._async_node = FederationNode(*args, **kwargs)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self, seed_nodes: Optional[List[str]] = None):
        """Start the node (blocking)."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._async_node.start(seed_nodes))

    def stop(self):
        """Stop the node."""
        if self._loop:
            self._loop.run_until_complete(self._async_node.stop())
            self._loop.close()

    def record_pattern_use(self, trigger: str, success: bool, snr_delta: float):
        return self._async_node.record_pattern_use(trigger, success, snr_delta)

    def get_stats(self) -> Dict:
        return self._async_node.get_stats()

    def get_health(self) -> Dict:
        return self._async_node.get_health()


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("BIZRA FEDERATION NODE — Demo")
    print("=" * 70)

    # Create a demo node (sync mode)
    node = FederationNode(
        node_id="demo_node_001",
        bind_address="127.0.0.1:7654",
        ihsan_score=0.97,
        contribution_count=5,
    )

    # Simulate pattern discovery
    print("\n[Demo] Simulating pattern discovery...")
    for i in range(5):
        node.record_pattern_use("query.complexity > 0.8", success=True, snr_delta=0.1)

    # Check stats
    print("\n[Stats]")
    stats = node.get_stats()
    print(f"  Node ID: {stats['node_id']}")
    print(f"  Ihsān: {stats['ihsan_score']}")
    print(f"  Contributions: {stats['contributions']}")
    print(f"  Network Multiplier: {stats['network_multiplier']:.3f}")
    print(f"  Local Patterns: {stats['patterns']['local_patterns']}")

    # Health check
    print("\n[Health]")
    health = node.get_health()
    for key, value in health.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("✅ Federation Node Demo Complete")
    print("=" * 70)
    print("\nTo run full async demo:")
    print("  asyncio.run(node.start(['seed_host:7654']))")
