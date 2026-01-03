"""
BIZRA Federation Manager
Phase 9: 3-Node Local Federation Orchestrator

Manages the complete federation lifecycle including:
- Node initialization and coordination
- Consensus protocol orchestration
- Knowledge graph sharding management
- Distributed reasoning coordination
- Health monitoring and failure recovery
"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
import json
import os

try:
    # Try relative imports first (for package usage)
    from .federation_node import FederationNode
    from .federation_consensus import FederationConsensusEngine
    from .knowledge_graph_sharding import KnowledgeGraphSharding
    from .graph_reasoning_federation import GraphReasoningFederation
    from .memory_system import CognitivePermanence
    from .state_ledger import StateLedger
    from .chaos_engine import ChaosEngine, default_chaos_alert_handler
except ImportError:
    # Fall back to absolute imports (for direct script execution)
    from federation_node import FederationNode
    from federation_consensus import FederationConsensusEngine
    from knowledge_graph_sharding import KnowledgeGraphSharding
    from graph_reasoning_federation import GraphReasoningFederation
    from memory_system import CognitivePermanence
    from state_ledger import StateLedger
    from chaos_engine import ChaosEngine, default_chaos_alert_handler


@dataclass
class FederationConfig:
    """Configuration for the federation."""
    node_id: str
    peer_nodes: List[str]
    port: int = 8888
    consensus_quorum: int = 2  # For 3 nodes: 2 out of 3
    heartbeat_interval: float = 1.0
    election_timeout: float = 3.0
    shard_replication: int = 2
    reasoning_timeout: float = 30.0


@dataclass
class FederationStatus:
    """Current status of the federation."""
    active_nodes: Set[str] = field(default_factory=set)
    leader_node: Optional[str] = None
    total_shards: int = 0
    total_entities: int = 0
    active_sessions: int = 0
    consensus_rounds: int = 0
    last_heartbeat: float = 0.0
    health_score: float = 0.0  # 0.0 to 1.0


class FederationManager:
    """
    Orchestrates the BIZRA 3-Node Local Federation.

    Manages the complete federation lifecycle and coordinates all components:
    - Federation nodes and networking
    - Consensus engine and leader election
    - Knowledge graph sharding
    - Distributed reasoning
    - Health monitoring and recovery
    """

    def __init__(self, config: FederationConfig):
        self.config = config

        # Core components
        self.state_ledger = StateLedger()
        self.memory_system = CognitivePermanence(agent_id=config.node_id)

        # Federation components
        self.sharding = KnowledgeGraphSharding(
            node_ids=[config.node_id] + config.peer_nodes,
            replication_factor=config.shard_replication
        )

        self.consensus_engine = FederationConsensusEngine(
            node_id=config.node_id,
            peer_nodes=config.peer_nodes,
            state_ledger=self.state_ledger
        )

        self.reasoning_federation = GraphReasoningFederation(
            node_id=config.node_id,
            sharding=self.sharding,
            memory_system=self.memory_system
        )

        # Networking
        self.federation_node = FederationNode(
            node_id=config.node_id,
            peer_nodes=config.peer_nodes,
            port=config.port,
            consensus_state=self.consensus_engine.consensus_state,
            consensus_engine=self.consensus_engine
        )

        # Chaos Engineering Engine
        self.chaos_engine = ChaosEngine(self)
        self.chaos_engine.register_alert_callback(default_chaos_alert_handler)

        # Status tracking
        self.status = FederationStatus()
        self.status.active_nodes.add(config.node_id)

        # Control
        self.running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.recovery_task: Optional[asyncio.Task] = None
        self.chaos_task: Optional[asyncio.Task] = None

        # Persistence
        self.state_file = f"federation_state_{config.node_id}.json"

    async def start_federation(self):
        """Start the federation and all its components."""
        print(f"[+] Starting BIZRA Federation for node {self.config.node_id}")

        self.running = True

        # Load previous state if exists
        self._load_federation_state()

        # Connect consensus engine to networking
        self.consensus_engine.set_broadcast_callback(self.federation_node._broadcast_message)

        # Start federation node (networking)
        node_task = asyncio.create_task(self.federation_node.start())

        # Start monitoring and recovery
        self.monitoring_task = asyncio.create_task(self._health_monitoring_loop())
        self.recovery_task = asyncio.create_task(self._failure_recovery_loop())

        # Start chaos monitoring
        self.chaos_task = asyncio.create_task(self.chaos_engine.start_chaos_monitoring())

        # Initialize knowledge graph sharding
        await self._initialize_sharding()

        # Wait for node to be ready
        await asyncio.sleep(2)

        print(f"[+] Federation started successfully. Leader: {self.consensus_engine.consensus_state.current_leader}")
        print(f"[+] Active nodes: {len(self.status.active_nodes)}")
        print(f"[+] Chaos monitoring active: MTTR target <=30s")

        # Keep federation running
        await node_task

    async def stop_federation(self):
        """Stop the federation gracefully."""
        print(f"[-] Stopping BIZRA Federation for node {self.config.node_id}")

        self.running = False

        # Stop monitoring tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.recovery_task:
            self.recovery_task.cancel()
        if self.chaos_task:
            self.chaos_task.cancel()

        # Stop chaos monitoring
        await self.chaos_engine.stop_chaos_monitoring()

        # Stop federation node
        await self.federation_node.stop()

        # Save state
        self._save_federation_state()

        print("[-] Federation stopped")

    async def submit_federated_request(self, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a request that requires federation consensus.

        This handles the complete flow from request submission through
        consensus to execution and response.
        """
        # Submit to consensus engine
        request_id = await self.consensus_engine.submit_request(operation, data, self.config.node_id)

        # Wait for completion (with timeout)
        timeout = 30.0  # 30 seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.consensus_engine.get_request_status(request_id)
            if status and status.get("status") == "executed":
                return {
                    "request_id": request_id,
                    "status": "completed",
                    "result": status.get("result"),
                    "execution_time": time.time() - start_time
                }
            await asyncio.sleep(0.1)

        return {
            "request_id": request_id,
            "status": "timeout",
            "error": "Request timed out waiting for consensus"
        }

    async def initiate_distributed_reasoning(self, query: str) -> str:
        """
        Initiate distributed Graph-of-Thoughts reasoning across the federation.

        Returns a session ID for tracking the reasoning process.
        """
        session_id = await self.reasoning_federation.initiate_reasoning_session(query)
        return session_id

    async def add_knowledge_entity(self, entity_id: str, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new entity to the distributed knowledge graph.

        This involves sharding assignment, consensus, and replication.
        """
        # Assign to shard
        assignment = self.sharding.assign_entity_to_shard(entity_id, entity_data)

        # Submit consensus request to add to ledger
        consensus_data = {
            "action_name": "add_entity",
            "action_data": {
                "entity_id": entity_id,
                "entity_data": entity_data,
                "shard_id": assignment.shard_id,
                "node_id": assignment.node_id
            },
            "metrics": {
                "im_score": 0.95,
                "status": "APPROVED",
                "timestamp": time.time(),
                "signature": "federation_generated"
            }
        }

        result = await self.submit_federated_request("validate_and_commit", consensus_data)

        if result.get("status") == "completed":
            # Update local memory if this node owns the primary shard
            if assignment.node_id == self.config.node_id:
                self.memory_system.add_semantic_fact(
                    entity_data.get("entity", entity_id),
                    entity_data.get("fact", ""),
                    entity_data.get("rels", {})
                )

        return {
            "entity_id": entity_id,
            "shard_assignment": f"{assignment.node_id}:{assignment.shard_id}",
            "consensus_result": result
        }

    async def _initialize_sharding(self):
        """Initialize the knowledge graph sharding system."""
        # Load existing entities from memory and assign to shards
        entities_assigned = 0

        for entity_id, entity_data in self.memory_system.layers["L4"].items():
            try:
                assignment = self.sharding.assign_entity_to_shard(entity_id, entity_data)
                entities_assigned += 1
            except Exception as e:
                print(f"[!] Failed to assign entity {entity_id}: {e}")

        print(f"[+] Initialized sharding with {entities_assigned} entities across {len(self.sharding.shards)} shards")

        # Update status
        self.status.total_shards = len(self.sharding.shards)
        self.status.total_entities = entities_assigned

    async def _health_monitoring_loop(self):
        """Continuously monitor federation health."""
        while self.running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                # Update status
                self.status.last_heartbeat = time.time()
                self.status.leader_node = self.consensus_engine.consensus_state.current_leader
                self.status.active_sessions = len(self.reasoning_federation.active_sessions)
                self.status.consensus_rounds = self.consensus_engine.sequence_number
                
                # Sync active nodes from network connections
                active_set = {self.config.node_id}
                async with self.federation_node.network_lock:
                    active_set.update(self.federation_node.peer_connections.keys())
                self.status.active_nodes = active_set

                # Calculate health score
                health_factors = []

                # Node connectivity
                expected_nodes = len(self.config.peer_nodes) + 1
                connectivity_score = len(self.status.active_nodes) / expected_nodes
                health_factors.append(connectivity_score)

                # Leader availability
                leader_score = 1.0 if self.status.leader_node else 0.0
                health_factors.append(leader_score)

                # Consensus activity (recent rounds indicate activity)
                consensus_score = min(1.0, self.status.consensus_rounds / 100.0)
                health_factors.append(consensus_score)

                # Average health factors
                self.status.health_score = sum(health_factors) / len(health_factors)

                # Log health issues
                if self.status.health_score < 0.7:
                    print(f"[!] Federation health degraded: {self.status.health_score:.2f}")

            except Exception as e:
                print(f"[!] Health monitoring error: {e}")
                await asyncio.sleep(5)

    async def _failure_recovery_loop(self):
        """Monitor for failures and initiate recovery."""
        while self.running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                # Check for node failures
                failed_nodes = []
                for peer in self.config.peer_nodes:
                    if peer not in self.federation_node.peer_connections:
                        failed_nodes.append(peer)

                if failed_nodes:
                    print(f"[!] Detected failed nodes: {failed_nodes}")
                    await self._handle_node_failures(failed_nodes)

                # Check shard balance
                sharding_stats = self.sharding.get_sharding_stats()
                load_distribution = sharding_stats.get("load_distribution", {})

                min_shards = load_distribution.get("min_shards_per_node", 0)
                max_shards = load_distribution.get("max_shards_per_node", 0)

                if max_shards > min_shards * 2:  # Significant imbalance
                    print("[!] Shard imbalance detected, triggering rebalancing")
                    migrations = self.sharding.rebalance_shards()
                    await self._execute_shard_migrations(migrations)

            except Exception as e:
                print(f"[!] Recovery loop error: {e}")
                await asyncio.sleep(5)

    async def _handle_node_failures(self, failed_nodes: List[str]):
        """Handle failure of one or more nodes."""
        for failed_node in failed_nodes:
            print(f"[*] Initiating recovery for failed node {failed_node}")

            # Rebalance shards
            migrations = self.sharding.rebalance_shards(failed_node=failed_node)

            # Execute migrations
            await self._execute_shard_migrations(migrations)

            # Update consensus state
            if self.consensus_engine.consensus_state.current_leader == failed_node:
                print(f"[!] Leader {failed_node} failed, triggering election")
                # Election will be handled by the consensus engine

    async def _execute_shard_migrations(self, migrations: Dict[str, List[str]]):
        """Execute shard migrations for load balancing or recovery."""
        for source_node, shard_ids in migrations.items():
            for shard_id in shard_ids:
                print(f"[*] Migrating shard {shard_id} from {source_node} to new node")

                # In a real implementation, this would:
                # 1. Notify source node to prepare shard data
                # 2. Transfer shard data to new node
                # 3. Update metadata and routing
                # 4. Verify data integrity

                # For now, just update local metadata
                if shard_id in self.sharding.shards:
                    # Find new node (simplified - use consistent hashing)
                    new_node = self.sharding._find_replacement_node(source_node)
                    if new_node:
                        self.sharding.shards[shard_id].node_id = new_node
                        self.sharding.node_shards[new_node].add(shard_id)
                        if source_node in self.sharding.node_shards:
                            self.sharding.node_shards[source_node].discard(shard_id)

    async def trigger_severed_link_scenario(self, affected_nodes: List[str],
                                         isolated_from: List[str] = None,
                                         duration_seconds: float = 30.0) -> str:
        """
        Trigger the "Severed Link" chaos scenario.

        Creates network partition between specified nodes and measures MTTR.
        """
        if isolated_from is None:
            # Default: isolate affected nodes from all other nodes
            isolated_from = [n for n in self.config.peer_nodes + [self.config.node_id]
                           if n not in affected_nodes]

        return await self.chaos_engine.trigger_severed_link_scenario(
            affected_nodes=affected_nodes,
            isolated_from=isolated_from,
            duration_seconds=duration_seconds
        )

    def get_chaos_status(self) -> Dict[str, Any]:
        """Get chaos engineering status."""
        return self.chaos_engine.get_chaos_status()

    def get_mttr_report(self) -> Dict[str, Any]:
        """Get MTTR performance report."""
        return self.chaos_engine.get_mttr_report()

    def get_federation_status(self) -> Dict[str, Any]:
        """Get comprehensive federation status."""
        sharding_stats = self.sharding.get_sharding_stats()
        reasoning_stats = self.reasoning_federation.get_reasoning_stats()
        consensus_status = self.consensus_engine.get_consensus_status()
        chaos_status = self.get_chaos_status()

        return {
            "node_id": self.config.node_id,
            "federation_status": {
                "active_nodes": list(self.status.active_nodes),
                "leader_node": self.status.leader_node,
                "health_score": self.status.health_score,
                "total_shards": self.status.total_shards,
                "total_entities": self.status.total_entities,
                "active_sessions": self.status.active_sessions,
                "consensus_rounds": self.status.consensus_rounds,
                "last_heartbeat": self.status.last_heartbeat
            },
            "sharding_stats": sharding_stats,
            "reasoning_stats": reasoning_stats,
            "consensus_status": consensus_status,
            "network_status": {
                "connected_peers": len(self.federation_node.peer_connections),
                "expected_peers": len(self.config.peer_nodes)
            },
            "chaos_status": chaos_status
        }

    def _save_federation_state(self):
        """Save federation state to disk."""
        state = {
            "node_id": self.config.node_id,
            "timestamp": time.time(),
            "sharding": {
                "shards": {sid: {
                    "node_id": s.node_id,
                    "entity_count": s.entity_count,
                    "semantic_clusters": list(s.semantic_clusters)
                } for sid, s in self.sharding.shards.items()},
                "entity_assignments": {eid: [{
                    "shard_id": a.shard_id,
                    "node_id": a.node_id,
                    "primary": a.primary
                } for a in assignments] for eid, assignments in self.sharding.entity_assignments.items()}
            },
            "consensus": {
                "view_number": self.consensus_engine.consensus_state.view_number,
                "sequence_number": self.consensus_engine.sequence_number,
                "current_leader": self.consensus_engine.consensus_state.current_leader
            }
        }

        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"[+] Federation state saved to {self.state_file}")
        except Exception as e:
            print(f"[!] Failed to save federation state: {e}")

    def _load_federation_state(self):
        """Load federation state from disk."""
        if not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Restore sharding state
            if "sharding" in state:
                sharding_data = state["sharding"]

                # Restore shards
                for sid, s_data in sharding_data.get("shards", {}).items():
                    from bizra_kernel.knowledge_graph_sharding import ShardInfo
                    self.sharding.shards[sid] = ShardInfo(
                        shard_id=sid,
                        node_id=s_data["node_id"],
                        entity_count=s_data["entity_count"],
                        semantic_clusters=set(s_data["semantic_clusters"])
                    )

                # Restore assignments
                for eid, assignments_data in sharding_data.get("entity_assignments", {}).items():
                    assignments = []
                    for a_data in assignments_data:
                        from bizra_kernel.knowledge_graph_sharding import ShardAssignment
                        assignments.append(ShardAssignment(
                            entity_id=eid,
                            shard_id=a_data["shard_id"],
                            node_id=a_data["node_id"],
                            primary=a_data["primary"]
                        ))
                    self.sharding.entity_assignments[eid] = assignments

            # Restore consensus state
            if "consensus" in state:
                consensus_data = state["consensus"]
                self.consensus_engine.consensus_state.view_number = consensus_data.get("view_number", 0)
                self.consensus_engine.sequence_number = consensus_data.get("sequence_number", 0)
                self.consensus_engine.consensus_state.current_leader = consensus_data.get("current_leader")

            print(f"[+] Federation state loaded from {self.state_file}")

        except Exception as e:
            print(f"[!] Failed to load federation state: {e}")


async def create_federation_node(node_id: str, peer_nodes: List[str], port: int = 8888) -> FederationManager:
    """Factory function to create a federation node."""
    config = FederationConfig(
        node_id=node_id,
        peer_nodes=peer_nodes,
        port=port
    )

    manager = FederationManager(config)
    return manager


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python federation_manager.py <node_id> [peer1] [peer2]")
        sys.exit(1)

    node_id = sys.argv[1]
    peers = sys.argv[2:] if len(sys.argv) > 2 else []

    async def main():
        manager = await create_federation_node(node_id, peers)

        try:
            await manager.start_federation()
        except KeyboardInterrupt:
            await manager.stop_federation()

    asyncio.run(main())