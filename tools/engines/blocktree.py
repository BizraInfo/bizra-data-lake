# BIZRA BlockTree Implementation v1.0
# Implements a Merkle-DAG for Knowledge Base Integrity and Provenance
# Part of Phase 1: Foundation Deploy

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime

class BlockNode:
    def __init__(self, data, node_type="Data", parents=None):
        self.timestamp = datetime.now().isoformat()
        self.node_type = node_type
        self.data = data
        self.parents = parents or []
        self.hash = self._calculate_hash()

    def _calculate_hash(self):
        header = f"{self.node_type}|{self.timestamp}|{json.dumps(self.parents)}"
        payload = str(self.data).encode()
        return hashlib.sha256(header.encode() + payload).hexdigest()

    def to_dict(self):
        return {
            "hash": self.hash,
            "type": self.node_type,
            "timestamp": self.timestamp,
            "parents": self.parents,
            "data_summary": str(self.data)[:100] if isinstance(self.data, str) else "complex_object"
        }

class BlockTree:
    def __init__(self, storage_path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "blocktree_index.jsonl"
        self.nodes = {}
        self.load_index()

    def load_index(self):
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                for line in f:
                    node = json.loads(line)
                    self.nodes[node['hash']] = node

    def add_leaf(self, file_path):
        """Creates a leaf node from a file."""
        path = Path(file_path)
        if not path.exists():
            return None
        
        content = path.read_text(errors='ignore')
        node = BlockNode(data=content, node_type="FileLeaf")
        self._persist_node(node)
        return node.hash

    def add_concept(self, label, member_hashes):
        """Creates a branch node connecting multiple leaves/branches."""
        node = BlockNode(data=label, node_type="ConceptBranch", parents=member_hashes)
        self._persist_node(node)
        return node.hash

    def _persist_node(self, node):
        self.nodes[node.hash] = node.to_dict()
        with open(self.index_file, 'a') as f:
            f.write(json.dumps(node.to_dict()) + "\n")

if __name__ == "__main__":
    print("🌳 Initializing BIZRA BlockTree (DAG Architecture)")
    bt = BlockTree("C:/BIZRA-DATA-LAKE/03_INDEXED/blocktree")
    
    # Example: link some important files
    files = [
        "C:/BIZRA-DATA-LAKE/FINAL_OMNI_BLUEPRINT.md",
        "C:/BIZRA-DATA-LAKE/README.md"
    ]
    
    hashes = []
    for f in files:
        h = bt.add_leaf(f)
        if h:
            print(f"✅ Added {f} -> {h[:12]}")
            hashes.append(h)
            
    # Create a root concept
    root_h = bt.add_concept("Foundation_v1", hashes)
    print(f"🌐 Created Root Concept: {root_h[:12]}")
