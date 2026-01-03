import os
import uuid
import json
import threading
import hashlib
from datetime import datetime

class RecursiveNode:
    """
    Self-spawning node architecture for the BIZRA Network State.
    Hardened with Ω-Class Depth Guards and Thread-Safety.
    """
    MAX_DEPTH = 3
    _lock = threading.RLock()
    
    def __init__(self, node_id=None, parent_id=None, depth=0, max_depth=None):
        if not node_id:
            # Phase 4: Ancestry-Linked ID (Prevent Spoofing)
            import hashlib
            salt = uuid.uuid4().hex[:4]
            base = f"{parent_id}:{salt}" if parent_id else f"ROOT:{salt}"
            node_hash = hashlib.sha256(base.encode()).hexdigest()[:8].upper()
            self.node_id = f"BIZRA-{node_hash}"
        else:
            self.node_id = node_id
            
        self.parent_id = parent_id
        self.depth = depth
        self.max_depth = max_depth if max_depth is not None else self.MAX_DEPTH
        self.children = []
        self.status = "SEED"
        self.config_dir = f"bizra_network/{self.node_id}"
        os.makedirs(self.config_dir, exist_ok=True)
        
    def activate(self):
        self.status = "ORGANISM"
        self._save_state()
        print(f"[NODE ACTIVATED] {self.node_id} (Depth: {self.depth}, Parent: {self.parent_id})")
        
    def spawn_child(self):
        """Recursively scale the network by birthing a child node. Thread-safe."""
        with self._lock:
            if self.depth >= self.max_depth:
                print(f"[!] RECURSION BLOCKED: Node {self.node_id} reached Max Depth ({self.max_depth}).")
                return None
                
            child = RecursiveNode(parent_id=self.node_id, depth=self.depth + 1, max_depth=self.max_depth)
            self.children.append(child.node_id)
            child.activate()
            self._save_state()
            return child

    def budget_aware_spawn(self, budget_score: float, threshold: float = 0.35):
        """Spawn only if cognitive budget supports recursion."""
        if budget_score < threshold:
            print(f"[!] RECURSION PAUSED: Budget {budget_score:.3f} below threshold {threshold:.2f}.")
            return None
        return self.spawn_child()

    def set_dynamic_max_depth(self, max_depth: int):
        """Allow SovereignEngine to tune depth dynamically."""
        with self._lock:
            self.max_depth = max(0, max_depth)
        
    def _save_state(self):
        state = {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "children": self.children,
            "status": self.status,
            "max_depth": self.max_depth,
            "birth_time": datetime.utcnow().isoformat()
        }
        with open(f"{self.config_dir}/manifest.json", "w") as f:
            json.dump(state, f, indent=4)

if __name__ == "__main__":
    print("--- BIZRA RECURSIVE NETWORK GENESIS ---")
    root = RecursiveNode(node_id="BIZRA-MASTER-0")
    root.activate()
    
    # Spawn 2nd generation
    print("\nScaling to 2nd Generation...")
    child_1 = root.spawn_child()
    
    # Spawn 3rd generation from child
    print("\nScaling to 3rd Generation (Recursive)...")
    grand_child = child_1.spawn_child()
    
    print(f"\nNetwork State: Root -> {root.children[0]} -> {child_1.children[0]}")
    print("--- RECURSIVE SCALING SUCCESSFUL ---")
