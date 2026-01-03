import hashlib
import json
import datetime
import os
from typing import List, Dict, Any

class StateLedger:
    """
    BIZRA State Ledger — The Immutable Chain of Truth. 
    Uses hash-chaining to ensure state integrity.
    """
    
    def __init__(self, storage_path="bizra_memory/ledger.json"):
        self.chain: List[Dict[str, Any]] = []
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        if os.path.exists(self.storage_path):
            self._load_from_disk()
        else:
            self._create_genesis_block()
            
    def _load_from_disk(self):
        try:
            with open(self.storage_path, 'r') as f:
                self.chain = json.load(f)
            if not self.verify_integrity():
                print("[!] CRITICAL: Ledger Integrity Failure detected on boot! Resetting to Genesis.")
                self.chain = []
                self._create_genesis_block()
        except Exception as e:
            print(f"[!] Ledger Load Error: {e}. Starting fresh.")
            self._create_genesis_block()

    def _create_genesis_block(self):
        genesis = {
            "index": 0,
            "timestamp": "2026-01-01T00:00:00Z",
            "state": "GENESIS",
            "data": "Universal Genesis Activation — BIZRA Node0",
            "prev_hash": "0" * 64
        }
        genesis["hash"] = self._calculate_hash(genesis)
        self.chain.append(genesis)
        self._save_to_disk()
        
    def _save_to_disk(self):
        # Phase 2: Implementation of Atomic Write would go here, using temp file
        temp_path = self.storage_path + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(self.chain, f, indent=4)
        os.replace(temp_path, self.storage_path)
        
    def _calculate_hash(self, block: Dict[str, Any]) -> str:
        import hashlib
        block_str = json.dumps(block, sort_keys=True)
        return hashlib.sha256(block_str.encode()).hexdigest()
        
    def append_state(self, state_name: str, data: Any):
        prev_block = self.chain[-1]
        new_block = {
            "index": len(self.chain),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "state": state_name,
            "data": data,
            "prev_hash": prev_block["hash"]
        }
        new_block["hash"] = self._calculate_hash(new_block)
        self.chain.append(new_block)
        self._save_to_disk()
        return new_block["hash"]
        
    def verify_integrity(self) -> bool:
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            prev = self.chain[i-1]
            
            # Verify self hash
            temp_block = current.copy()
            temp_block.pop("hash")
            if current["hash"] != self._calculate_hash(temp_block):
                return False
                
            # Verify chain link
            if current["prev_hash"] != prev["hash"]:
                return False
        return True

    def get_latest_state(self) -> Dict[str, Any]:
        return self.chain[-1]
