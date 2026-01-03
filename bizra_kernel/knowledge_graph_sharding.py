"""
BIZRA Knowledge Graph Sharding
Phase 9: Distributed HyperGraph Distribution Strategy

Implements intelligent sharding of the knowledge graph across federation nodes.
Uses consistent hashing and semantic clustering for optimal distribution.
"""

import hashlib
import json
import math
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from .memory_system import CognitivePermanence
except ImportError:
    from memory_system import CognitivePermanence


@dataclass
class ShardInfo:
    """Information about a knowledge graph shard."""
    shard_id: str
    node_id: str
    entity_count: int = 0
    relationship_count: int = 0
    semantic_clusters: Set[str] = field(default_factory=set)
    last_updated: float = 0.0


@dataclass
class ShardAssignment:
    """Assignment of an entity to a shard and node."""
    entity_id: str
    shard_id: str
    node_id: str
    primary: bool = True  # True for primary replica, False for backup
    timestamp: float = field(default_factory=lambda: __import__('time').time())


class KnowledgeGraphSharding:
    """
    Intelligent sharding system for BIZRA's knowledge graph.

    Distributes the HyperGraph across federation nodes using:
    - Consistent hashing for load balancing
    - Semantic clustering for query locality
    - Replication for fault tolerance
    """

    def __init__(self, node_ids: List[str], replication_factor: int = 2):
        self.node_ids = node_ids
        self.replication_factor = min(replication_factor, len(node_ids))

        # Sharding state
        self.shards: Dict[str, ShardInfo] = {}
        self.entity_assignments: Dict[str, List[ShardAssignment]] = {}
        self.node_shards: Dict[str, Set[str]] = {node: set() for node in node_ids}

        # Consistent hashing ring
        self.hash_ring: List[Tuple[int, str]] = []
        self._build_hash_ring()

        # Semantic clustering
        self.semantic_clusters: Dict[str, Set[str]] = {}  # cluster_id -> entity_ids
        self.entity_clusters: Dict[str, str] = {}  # entity_id -> cluster_id

        # Sharding parameters
        self.shard_count = len(node_ids) * 4  # 4 shards per node for good distribution
        self.max_shard_size = 1000  # Maximum entities per shard

    def _build_hash_ring(self):
        """Build the consistent hashing ring."""
        self.hash_ring = []

        for node_id in self.node_ids:
            # Add multiple virtual nodes per physical node for better distribution
            for i in range(10):  # 10 virtual nodes per physical node
                key = f"{node_id}:{i}"
                hash_value = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
                self.hash_ring.append((hash_value, node_id))

        # Sort the ring
        self.hash_ring.sort(key=lambda x: x[0])

    def get_node_for_entity(self, entity_id: str) -> str:
        """Get the primary node for an entity using consistent hashing."""
        entity_hash = int(hashlib.sha256(entity_id.encode()).hexdigest(), 16) % (2**32)

        # Find the first node with hash >= entity_hash
        for hash_value, node_id in self.hash_ring:
            if hash_value >= entity_hash:
                return node_id

        # Wrap around to first node
        return self.hash_ring[0][1]

    def assign_entity_to_shard(self, entity_id: str, entity_data: Dict[str, Any]) -> ShardAssignment:
        """
        Assign an entity to a shard and node.

        Uses semantic clustering and consistent hashing for optimal assignment.
        """
        # Determine semantic cluster
        cluster_id = self._determine_semantic_cluster(entity_id, entity_data)

        # Get primary node using consistent hashing
        primary_node = self.get_node_for_entity(entity_id)

        # Create or get appropriate shard
        shard_id = self._get_or_create_shard(primary_node, cluster_id)

        # Create assignment
        assignment = ShardAssignment(
            entity_id=entity_id,
            shard_id=shard_id,
            node_id=primary_node,
            primary=True
        )

        # Store assignment
        if entity_id not in self.entity_assignments:
            self.entity_assignments[entity_id] = []
        self.entity_assignments[entity_id].append(assignment)

        # Update shard info
        if shard_id in self.shards:
            self.shards[shard_id].entity_count += 1
            self.shards[shard_id].semantic_clusters.add(cluster_id)
            self.shards[shard_id].last_updated = assignment.timestamp

        # Add to node's shard list
        self.node_shards[primary_node].add(shard_id)

        # Create replica assignments for fault tolerance
        self._create_replicas(entity_id, shard_id, primary_node)

        return assignment

    def _determine_semantic_cluster(self, entity_id: str, entity_data: Dict[str, Any]) -> str:
        """Determine the semantic cluster for an entity based on its content."""
        # Extract semantic features
        features = []

        # Entity type
        if "entity" in entity_data:
            features.append(f"type:{entity_data['entity']}")

        # Relationships
        if "rels" in entity_data and isinstance(entity_data["rels"], dict):
            for rel_type in entity_data["rels"].keys():
                features.append(f"rel:{rel_type}")

        # Content keywords (simplified)
        if "fact" in entity_data:
            fact_text = str(entity_data["fact"]).lower()
            # Extract some keywords
            keywords = [word for word in fact_text.split() if len(word) > 3][:5]
            features.extend([f"kw:{kw}" for kw in keywords])

        # Create cluster ID from features
        if features:
            cluster_key = "|".join(sorted(features))
            cluster_id = hashlib.sha256(cluster_key.encode()).hexdigest()[:8]

            # Update cluster membership
            if cluster_id not in self.semantic_clusters:
                self.semantic_clusters[cluster_id] = set()
            self.semantic_clusters[cluster_id].add(entity_id)
            self.entity_clusters[entity_id] = cluster_id

            return cluster_id
        else:
            # Default cluster
            return "default"

    def _get_or_create_shard(self, primary_node: str, cluster_id: str) -> str:
        """Get an existing shard or create a new one for the given node and cluster."""
        # Look for existing shard on this node with the same cluster
        for shard_id in self.node_shards[primary_node]:
            if shard_id in self.shards:
                shard = self.shards[shard_id]
                if cluster_id in shard.semantic_clusters and shard.entity_count < self.max_shard_size:
                    return shard_id

        # Create new shard
        shard_id = f"shard_{primary_node}_{cluster_id}_{len(self.shards)}"
        self.shards[shard_id] = ShardInfo(
            shard_id=shard_id,
            node_id=primary_node,
            semantic_clusters={cluster_id}
        )

        return shard_id

    def _create_replicas(self, entity_id: str, shard_id: str, primary_node: str):
        """Create replica assignments for fault tolerance."""
        # Get replica nodes (next N nodes on the hash ring)
        primary_hash = int(hashlib.sha256(primary_node.encode()).hexdigest(), 16) % (2**32)
        replica_nodes = []

        # Find replica nodes on the hash ring
        ring_size = len(self.hash_ring)
        start_idx = 0
        for i, (hash_val, node) in enumerate(self.hash_ring):
            if node == primary_node:
                start_idx = (i + 1) % ring_size
                break

        # Collect replica nodes
        visited = set([primary_node])
        current_idx = start_idx
        while len(replica_nodes) < self.replication_factor - 1 and len(visited) < len(self.node_ids):
            _, node = self.hash_ring[current_idx]
            if node not in visited:
                replica_nodes.append(node)
                visited.add(node)
            current_idx = (current_idx + 1) % ring_size

        # Create replica assignments
        for replica_node in replica_nodes:
            replica_shard_id = f"{shard_id}_replica_{replica_node}"
            replica_assignment = ShardAssignment(
                entity_id=entity_id,
                shard_id=replica_shard_id,
                node_id=replica_node,
                primary=False
            )

            self.entity_assignments[entity_id].append(replica_assignment)

            # Create replica shard if it doesn't exist
            if replica_shard_id not in self.shards:
                self.shards[replica_shard_id] = ShardInfo(
                    shard_id=replica_shard_id,
                    node_id=replica_node,
                    semantic_clusters=self.shards[shard_id].semantic_clusters.copy()
                )

            self.node_shards[replica_node].add(replica_shard_id)

    def get_entity_location(self, entity_id: str) -> List[ShardAssignment]:
        """Get all locations (primary + replicas) for an entity."""
        return self.entity_assignments.get(entity_id, [])

    def get_shard_entities(self, shard_id: str) -> List[str]:
        """Get all entities in a shard."""
        entities = []
        for entity_id, assignments in self.entity_assignments.items():
            for assignment in assignments:
                if assignment.shard_id == shard_id:
                    entities.append(entity_id)
                    break
        return entities

    def get_node_shards(self, node_id: str) -> List[str]:
        """Get all shards assigned to a node."""
        return list(self.node_shards.get(node_id, set()))

    def rebalance_shards(self, failed_node: str = None) -> Dict[str, List[str]]:
        """
        Rebalance shards when a node fails or for optimization.

        Returns a dictionary of migration operations: {source_node: [shard_ids_to_move]}
        """
        migrations = defaultdict(list)

        if failed_node:
            # Handle node failure - redistribute its shards
            failed_shards = self.get_node_shards(failed_node)

            for shard_id in failed_shards:
                if shard_id in self.shards:
                    # Find new node for this shard
                    # Use the cluster information to place it optimally
                    shard = self.shards[shard_id]
                    if shard.semantic_clusters:
                        # Use first cluster as representative
                        cluster_id = next(iter(shard.semantic_clusters))
                        # Find best node for this cluster
                        best_node = self._find_best_node_for_cluster(cluster_id, exclude_nodes={failed_node})
                    else:
                        # Fallback to consistent hashing
                        best_node = self._find_replacement_node(failed_node)

                    if best_node:
                        migrations[failed_node].append(shard_id)
                        # Update shard assignment
                        self.shards[shard_id].node_id = best_node
                        self.node_shards[best_node].add(shard_id)

            # Remove failed node from our tracking
            if failed_node in self.node_shards:
                del self.node_shards[failed_node]

        # Check for load balancing
        avg_shards_per_node = len(self.shards) / len(self.node_ids) if self.node_ids else 0

        for node_id in self.node_ids:
            if node_id == failed_node:
                continue

            node_shard_count = len(self.node_shards.get(node_id, set()))
            if node_shard_count > avg_shards_per_node * 1.5:  # Overloaded
                # Move some shards to less loaded nodes
                shards_to_move = self._select_shards_to_move(node_id, int(node_shard_count - avg_shards_per_node))
                for shard_id in shards_to_move:
                    target_node = self._find_least_loaded_node(exclude_nodes={node_id})
                    if target_node:
                        migrations[node_id].append(shard_id)
                        # Update assignment
                        self.shards[shard_id].node_id = target_node
                        self.node_shards[target_node].add(shard_id)
                        self.node_shards[node_id].discard(shard_id)

        return dict(migrations)

    def _find_best_node_for_cluster(self, cluster_id: str, exclude_nodes: Set[str] = None) -> Optional[str]:
        """Find the best node for a semantic cluster."""
        if exclude_nodes is None:
            exclude_nodes = set()

        # Count entities per cluster per node
        node_cluster_counts = defaultdict(int)

        for entity_id in self.semantic_clusters.get(cluster_id, set()):
            assignments = self.get_entity_location(entity_id)
            for assignment in assignments:
                if assignment.primary and assignment.node_id not in exclude_nodes:
                    node_cluster_counts[assignment.node_id] += 1

        # Return node with most entities from this cluster
        if node_cluster_counts:
            return max(node_cluster_counts.items(), key=lambda x: x[1])[0]

        # Fallback: return any available node
        available_nodes = [n for n in self.node_ids if n not in exclude_nodes]
        return available_nodes[0] if available_nodes else None

    def _find_replacement_node(self, failed_node: str) -> Optional[str]:
        """Find a replacement node using consistent hashing."""
        available_nodes = [n for n in self.node_ids if n != failed_node]
        return available_nodes[0] if available_nodes else None

    def _find_least_loaded_node(self, exclude_nodes: Set[str] = None) -> Optional[str]:
        """Find the least loaded node."""
        if exclude_nodes is None:
            exclude_nodes = set()

        node_loads = {}
        for node_id in self.node_ids:
            if node_id not in exclude_nodes:
                node_loads[node_id] = len(self.node_shards.get(node_id, set()))

        if node_loads:
            return min(node_loads.items(), key=lambda x: x[1])[0]
        return None

    def _select_shards_to_move(self, node_id: str, count: int) -> List[str]:
        """Select shards to move from an overloaded node."""
        node_shards = list(self.node_shards.get(node_id, set()))
        # Prioritize moving shards with fewer semantic clusters (less coupled)
        shard_scores = []

        for shard_id in node_shards:
            if shard_id in self.shards:
                shard = self.shards[shard_id]
                # Lower score for shards with fewer clusters (prefer to move isolated shards)
                score = len(shard.semantic_clusters)
                shard_scores.append((shard_id, score))

        # Sort by score (ascending) and return top count
        shard_scores.sort(key=lambda x: x[1])
        return [shard_id for shard_id, _ in shard_scores[:count]]

    def get_sharding_stats(self) -> Dict[str, Any]:
        """Get comprehensive sharding statistics."""
        total_entities = len(self.entity_assignments)
        total_shards = len(self.shards)

        node_stats = {}
        for node_id in self.node_ids:
            shards = self.get_node_shards(node_id)
            entities = sum(len(self.get_shard_entities(sid)) for sid in shards)
            node_stats[node_id] = {
                "shards": len(shards),
                "entities": entities,
                "replication_factor": self.replication_factor
            }

        cluster_stats = {
            "total_clusters": len(self.semantic_clusters),
            "avg_entities_per_cluster": total_entities / len(self.semantic_clusters) if self.semantic_clusters else 0
        }

        return {
            "total_entities": total_entities,
            "total_shards": total_shards,
            "replication_factor": self.replication_factor,
            "node_stats": node_stats,
            "cluster_stats": cluster_stats,
            "load_distribution": {
                "min_shards_per_node": min(len(self.node_shards[n]) for n in self.node_ids) if self.node_ids else 0,
                "max_shards_per_node": max(len(self.node_shards[n]) for n in self.node_ids) if self.node_ids else 0,
                "avg_shards_per_node": total_shards / len(self.node_ids) if self.node_ids else 0
            }
        }


if __name__ == "__main__":
    # Test the sharding system
    nodes = ["node_0", "node_1", "node_2"]
    sharding = KnowledgeGraphSharding(nodes)

    # Test entity assignment
    test_entities = [
        {"entity": "BIZRA", "fact": "The primary sovereign node for post-labor economics", "rels": {"Ihsan": "high", "Consensus": "core"}},
        {"entity": "Consensus", "fact": "Distributed agreement protocol", "rels": {"BIZRA": "uses", "Security": "requires"}},
        {"entity": "Security", "fact": "Cryptographic protection mechanisms", "rels": {"Consensus": "enables", "Privacy": "provides"}},
        {"entity": "AI", "fact": "Artificial intelligence systems", "rels": {"BIZRA": "integrates", "Learning": "performs"}},
    ]

    for i, entity_data in enumerate(test_entities):
        entity_id = f"entity_{i}"
        assignment = sharding.assign_entity_to_shard(entity_id, entity_data)
        print(f"Assigned {entity_id} to {assignment.node_id}:{assignment.shard_id}")

    # Print stats
    stats = sharding.get_sharding_stats()
    print(f"\nSharding Stats: {stats}")