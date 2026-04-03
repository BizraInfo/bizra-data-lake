"""
LocalMerkleDAG - Tamper-Proof Evidence Chain

Implements a local Merkle Directed Acyclic Graph for tamper-proof evidence chains.
Provides cryptographic verification of data integrity and operation history.

Key Features:
- Merkle tree structure for efficient verification
- DAG topology for complex dependency relationships
- SHA-256 and BLAKE3 hashing for integrity
- Local storage with no external dependencies
- Tamper detection via hash chain verification
- Domain separation: "bizra-pci-v1:"

NO external dependencies beyond stdlib (blake3 optional).
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid
import os

# Optional blake3 for enhanced security
try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


@dataclass
class MerkleNode:
    """Represents a node in the Merkle DAG."""
    node_id: str
    timestamp: str
    data: Dict[str, Any]
    parents: List[str]  # Parent node IDs (empty for genesis)
    children: List[str] = field(default_factory=list)  # Child node IDs
    hash: str = ""  # Merkle hash of this node
    merkle_root: str = ""  # Root hash including all parents


@dataclass
class VerificationResult:
    """Result of DAG verification."""
    valid: bool
    total_nodes: int
    verified_nodes: int
    tampered_nodes: List[str]
    orphaned_nodes: List[str]
    message: str


class LocalMerkleDAG:
    """
    Tamper-proof evidence chain using Merkle DAG structure.

    Provides cryptographic verification of operation history and data integrity.
    """

    DOMAIN_PREFIX = "bizra-pci-v1:"
    GENESIS_ID = "genesis"

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize LocalMerkleDAG.

        Args:
            storage_path: Path to store DAG data (None for in-memory only)
        """
        self.nodes: Dict[str, MerkleNode] = {}
        self.storage_path = storage_path

        # Initialize genesis node if needed
        if not self.nodes:
            self._create_genesis_node()

        # Load from storage if path provided
        if storage_path and os.path.exists(storage_path):
            self.load_from_file(storage_path)

    def add_node(
        self,
        data: Dict[str, Any],
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MerkleNode:
        """
        Add new node to the DAG.

        Args:
            data: Node data to store
            parent_ids: List of parent node IDs (None = attach to genesis)
            metadata: Additional metadata

        Returns:
            Created MerkleNode
        """
        # Default to genesis as parent
        if parent_ids is None:
            parent_ids = [self.GENESIS_ID]

        # Verify all parents exist
        for parent_id in parent_ids:
            if parent_id not in self.nodes:
                raise ValueError(f"Parent node '{parent_id}' not found")

        # Create node
        node_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Combine data and metadata
        full_data = {
            'data': data,
            'metadata': metadata or {},
            'timestamp': timestamp,
        }

        node = MerkleNode(
            node_id=node_id,
            timestamp=timestamp,
            data=full_data,
            parents=parent_ids,
            children=[]
        )

        # Calculate hash
        node.hash = self._calculate_node_hash(node)

        # Calculate Merkle root (includes parent hashes)
        node.merkle_root = self._calculate_merkle_root(node)

        # Add to graph
        self.nodes[node_id] = node

        # Update parent nodes' children lists
        for parent_id in parent_ids:
            parent = self.nodes[parent_id]
            if node_id not in parent.children:
                parent.children.append(node_id)

        # Persist if storage configured
        if self.storage_path:
            self.save_to_file(self.storage_path)

        return node

    def get_node(self, node_id: str) -> Optional[MerkleNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> List[MerkleNode]:
        """Get all children of a node."""
        node = self.nodes.get(node_id)
        if not node:
            return []

        return [self.nodes[child_id] for child_id in node.children
                if child_id in self.nodes]

    def get_parents(self, node_id: str) -> List[MerkleNode]:
        """Get all parents of a node."""
        node = self.nodes.get(node_id)
        if not node:
            return []

        return [self.nodes[parent_id] for parent_id in node.parents
                if parent_id in self.nodes]

    def get_ancestors(self, node_id: str) -> Set[str]:
        """
        Get all ancestor node IDs (recursive).

        Args:
            node_id: Starting node ID

        Returns:
            Set of ancestor node IDs
        """
        ancestors = set()
        to_visit = [node_id]

        while to_visit:
            current_id = to_visit.pop()
            node = self.nodes.get(current_id)
            if not node:
                continue

            for parent_id in node.parents:
                if parent_id not in ancestors:
                    ancestors.add(parent_id)
                    to_visit.append(parent_id)

        return ancestors

    def get_descendants(self, node_id: str) -> Set[str]:
        """
        Get all descendant node IDs (recursive).

        Args:
            node_id: Starting node ID

        Returns:
            Set of descendant node IDs
        """
        descendants = set()
        to_visit = [node_id]

        while to_visit:
            current_id = to_visit.pop()
            node = self.nodes.get(current_id)
            if not node:
                continue

            for child_id in node.children:
                if child_id not in descendants:
                    descendants.add(child_id)
                    to_visit.append(child_id)

        return descendants

    def verify_node(self, node_id: str) -> bool:
        """
        Verify integrity of a single node.

        Args:
            node_id: Node ID to verify

        Returns:
            True if node is valid
        """
        node = self.nodes.get(node_id)
        if not node:
            return False

        # Recalculate hash
        calculated_hash = self._calculate_node_hash(node)
        if calculated_hash != node.hash:
            return False

        # Recalculate Merkle root
        calculated_root = self._calculate_merkle_root(node)
        if calculated_root != node.merkle_root:
            return False

        return True

    def verify_dag(self) -> VerificationResult:
        """
        Verify integrity of entire DAG.

        Returns:
            VerificationResult with detailed verification status
        """
        total_nodes = len(self.nodes)
        verified_nodes = 0
        tampered_nodes = []
        orphaned_nodes = []

        # Verify each node
        for node_id, node in self.nodes.items():
            # Skip genesis
            if node_id == self.GENESIS_ID:
                verified_nodes += 1
                continue

            # Verify node integrity
            if not self.verify_node(node_id):
                tampered_nodes.append(node_id)
                continue

            # Check if parents exist
            for parent_id in node.parents:
                if parent_id not in self.nodes:
                    orphaned_nodes.append(node_id)
                    break

            verified_nodes += 1

        valid = len(tampered_nodes) == 0 and len(orphaned_nodes) == 0

        message = "DAG verification successful"
        if tampered_nodes:
            message = f"Tampered nodes detected: {len(tampered_nodes)}"
        elif orphaned_nodes:
            message = f"Orphaned nodes detected: {len(orphaned_nodes)}"

        return VerificationResult(
            valid=valid,
            total_nodes=total_nodes,
            verified_nodes=verified_nodes,
            tampered_nodes=tampered_nodes,
            orphaned_nodes=orphaned_nodes,
            message=message
        )

    def get_proof_chain(self, node_id: str) -> List[MerkleNode]:
        """
        Get proof chain from genesis to specified node.

        Args:
            node_id: Target node ID

        Returns:
            List of nodes forming proof chain
        """
        chain = []
        visited = set()

        def build_chain(current_id: str) -> bool:
            if current_id in visited:
                return False

            visited.add(current_id)
            node = self.nodes.get(current_id)
            if not node:
                return False

            # Reached target
            if current_id == node_id:
                chain.append(node)
                return True

            # Try each child
            for child_id in node.children:
                if build_chain(child_id):
                    chain.append(node)
                    return True

            return False

        # Start from genesis
        build_chain(self.GENESIS_ID)

        # Reverse to get genesis -> target order
        chain.reverse()

        return chain

    def export_to_dict(self) -> Dict[str, Any]:
        """Export DAG to dictionary."""
        return {
            'domain': self.DOMAIN_PREFIX,
            'genesis_id': self.GENESIS_ID,
            'node_count': len(self.nodes),
            'nodes': {
                node_id: asdict(node)
                for node_id, node in self.nodes.items()
            }
        }

    def import_from_dict(self, data: Dict[str, Any]) -> None:
        """Import DAG from dictionary."""
        if data.get('domain') != self.DOMAIN_PREFIX:
            raise ValueError("Domain mismatch")

        # Clear existing nodes
        self.nodes.clear()

        # Import nodes
        for node_id, node_data in data['nodes'].items():
            node = MerkleNode(**node_data)
            self.nodes[node_id] = node

        # Verify after import
        result = self.verify_dag()
        if not result.valid:
            raise ValueError(f"Imported DAG verification failed: {result.message}")

    def save_to_file(self, path: str) -> None:
        """Save DAG to JSON file."""
        data = self.export_to_dict()

        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, path: str) -> None:
        """Load DAG from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.import_from_dict(data)

    def _create_genesis_node(self) -> None:
        """Create genesis (root) node."""
        genesis = MerkleNode(
            node_id=self.GENESIS_ID,
            timestamp=datetime.utcnow().isoformat(),
            data={
                'type': 'genesis',
                'domain': self.DOMAIN_PREFIX,
                'message': 'BIZRA Sovereignty - Genesis Block'
            },
            parents=[],
            children=[]
        )

        # Genesis hash is based only on its data
        genesis.hash = self._calculate_node_hash(genesis)
        genesis.merkle_root = genesis.hash

        self.nodes[self.GENESIS_ID] = genesis

    def _calculate_node_hash(self, node: MerkleNode) -> str:
        """Calculate hash of node data."""
        # Create canonical representation
        hash_data = {
            'node_id': node.node_id,
            'timestamp': node.timestamp,
            'data': node.data,
            'parents': sorted(node.parents),  # Sort for determinism
        }

        hash_json = json.dumps(hash_data, sort_keys=True)
        hash_bytes = (self.DOMAIN_PREFIX + hash_json).encode('utf-8')

        # Use BLAKE3 if available, otherwise SHA-256
        if HAS_BLAKE3:
            return blake3.blake3(hash_bytes).hexdigest()
        else:
            return hashlib.sha256(hash_bytes).hexdigest()

    def _calculate_merkle_root(self, node: MerkleNode) -> str:
        """Calculate Merkle root including parent hashes."""
        # Collect parent hashes
        parent_hashes = []
        for parent_id in sorted(node.parents):  # Sort for determinism
            parent = self.nodes.get(parent_id)
            if parent:
                parent_hashes.append(parent.merkle_root)

        # Combine node hash with parent roots
        combined = node.hash + ''.join(parent_hashes)
        combined_bytes = (self.DOMAIN_PREFIX + combined).encode('utf-8')

        # Hash the combination
        if HAS_BLAKE3:
            return blake3.blake3(combined_bytes).hexdigest()
        else:
            return hashlib.sha256(combined_bytes).hexdigest()


def main():
    """Demo LocalMerkleDAG functionality."""
    print("LocalMerkleDAG - Tamper-Proof Evidence Chain")
    print("=" * 60)

    # Create DAG
    dag = LocalMerkleDAG()

    print(f"Genesis node created: {dag.GENESIS_ID}")
    print(f"BLAKE3 available: {HAS_BLAKE3}")

    # Add nodes
    print("\n1. Adding nodes to DAG...")

    node1 = dag.add_node(
        data={'operation': 'embed', 'text': 'Hello BIZRA'},
        metadata={'source': 'test'}
    )
    print(f"Added node: {node1.node_id[:8]}...")

    node2 = dag.add_node(
        data={'operation': 'verify', 'result': 'passed'},
        parent_ids=[node1.node_id],
        metadata={'source': 'test'}
    )
    print(f"Added node: {node2.node_id[:8]}...")

    node3 = dag.add_node(
        data={'operation': 'store', 'location': 'local'},
        parent_ids=[node1.node_id, node2.node_id],  # Multiple parents
        metadata={'source': 'test'}
    )
    print(f"Added node: {node3.node_id[:8]}... (2 parents)")

    print(f"\nTotal nodes: {len(dag.nodes)}")

    # Verify DAG
    print("\n2. Verifying DAG integrity...")
    result = dag.verify_dag()
    print(f"Valid: {result.valid}")
    print(f"Verified nodes: {result.verified_nodes}/{result.total_nodes}")
    print(f"Message: {result.message}")

    # Get proof chain
    print("\n3. Getting proof chain...")
    chain = dag.get_proof_chain(node3.node_id)
    print(f"Proof chain length: {len(chain)}")
    for i, node in enumerate(chain):
        print(f"  {i}. {node.node_id[:8]}... - {node.data.get('data', {}).get('operation', 'genesis')}")

    # Test tampering detection
    print("\n4. Testing tamper detection...")
    print("Tampering with node data...")
    node2.data['operation'] = 'TAMPERED'

    result2 = dag.verify_dag()
    print(f"Valid: {result2.valid}")
    print(f"Tampered nodes: {len(result2.tampered_nodes)}")
    if result2.tampered_nodes:
        print(f"  Detected tampering in: {result2.tampered_nodes[0][:8]}...")

    # Restore
    node2.data['operation'] = 'verify'
    node2.hash = dag._calculate_node_hash(node2)
    node2.merkle_root = dag._calculate_merkle_root(node2)

    print("\n5. After restoration:")
    result3 = dag.verify_dag()
    print(f"Valid: {result3.valid}")


if __name__ == "__main__":
    main()
