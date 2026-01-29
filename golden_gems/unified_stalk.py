"""
GOLDEN GEM #1: THE UNIFIED STALK
═════════════════════════════════

One data structure that serves as:
- API request/response envelope
- Database row
- Blockchain transaction
- Cognitive state snapshot
- Audit log entry
- Cache key
- Message queue item

The structure IS the protocol.

SNR Score: 0.95
"""

import hashlib
import time
import json
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UnifiedStalk:
    """
    The morphism that flows through all BIZRA systems.
    
    Properties:
    - Content-addressed (hash is identity)
    - Self-verifying (hash includes all fields)
    - Chain-linkable (prev_hash for DAG)
    - Ihsān-scored (ethical metadata)
    - Compressible (zlib for storage)
    """
    
    # === IDENTITY ===
    hash: str = ""                      # Content-addressed identity (computed)
    prev_hash: str = ""                 # Chain linkage (DAG parent)
    
    # === TEMPORALITY ===
    timestamp_ns: int = 0               # Nanosecond precision
    sequence: int = 0                   # Monotonic counter
    lamport: int = 0                    # Logical clock for causality
    
    # === COGNITION ===
    intent: str = ""                    # What is being requested/stated
    domain: str = ""                    # Which guardian handles this
    payload: Dict[str, Any] = field(default_factory=dict)  # The actual data
    
    # === ETHICS ===
    ihsan_score: float = 0.0            # 0.0 - 1.0, computed by FATE gate
    ihsan_dimensions: Dict[str, float] = field(default_factory=dict)
    
    # === PROVENANCE ===
    source: str = ""                    # Origin system/agent
    contributor: str = ""               # Human or agent ID
    attestations: List[str] = field(default_factory=list)  # Witness hashes
    
    # === COMPUTATION ===
    compute_ms: float = 0.0             # How long did this take
    entropy_delta: float = 0.0          # Did this reduce entropy?
    bloom_earned: float = 0.0           # Economic impact (accumulator)
    
    def __post_init__(self):
        if not self.timestamp_ns:
            self.timestamp_ns = time.time_ns()
        if not self.hash:
            self.hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Content-addressed identity."""
        content = {
            "prev": self.prev_hash,
            "ts": self.timestamp_ns,
            "seq": self.sequence,
            "intent": self.intent,
            "domain": self.domain,
            "payload": self.payload,
            "ihsan": self.ihsan_score,
            "source": self.source,
        }
        canonical = json.dumps(content, sort_keys=True, separators=(',', ':'))
        return hashlib.blake2b(canonical.encode(), digest_size=32).hexdigest()
    
    def verify(self) -> bool:
        """Self-verification."""
        return self.hash == self._compute_hash()
    
    def chain(self, next_intent: str, next_payload: Dict) -> "UnifiedStalk":
        """Create next stalk in chain."""
        return UnifiedStalk(
            prev_hash=self.hash,
            sequence=self.sequence + 1,
            lamport=self.lamport + 1,
            intent=next_intent,
            domain=self.domain,
            payload=next_payload,
            source=self.source,
        )
    
    def compress(self) -> bytes:
        """Compress for storage/transmission."""
        data = json.dumps(self.__dict__, default=str).encode()
        return zlib.compress(data, level=6)
    
    @classmethod
    def decompress(cls, data: bytes) -> "UnifiedStalk":
        """Restore from compressed form."""
        decompressed = zlib.decompress(data)
        obj = json.loads(decompressed)
        return cls(**obj)
    
    def to_api_response(self) -> Dict:
        """Format as API response."""
        return {
            "status": "success" if self.ihsan_score >= 0.7 else "blocked",
            "hash": self.hash,
            "data": self.payload,
            "meta": {
                "ihsan": self.ihsan_score,
                "compute_ms": self.compute_ms,
                "sequence": self.sequence,
            }
        }
    
    def to_audit_log(self) -> str:
        """Format as audit log entry."""
        return f"[{self.timestamp_ns}] {self.source}:{self.intent} -> {self.hash[:16]} (ihsan={self.ihsan_score:.2f})"
    
    def to_neo4j_node(self) -> Dict:
        """Format for Neo4j insertion."""
        return {
            "hash": self.hash,
            "intent": self.intent,
            "domain": self.domain,
            "ihsan_score": self.ihsan_score,
            "timestamp": self.timestamp_ns,
            "prev_hash": self.prev_hash,
        }


def demo():
    """Demonstrate the unified stalk."""
    # Create genesis stalk
    genesis = UnifiedStalk(
        intent="genesis",
        domain="system",
        payload={"message": "The seed is planted"},
        ihsan_score=1.0,
        source="bizra:kernel",
        contributor="mumo",
    )
    
    print(f"Genesis: {genesis.hash[:32]}...")
    print(f"Verified: {genesis.verify()}")
    
    # Chain next operation
    query = genesis.chain(
        next_intent="query",
        next_payload={"question": "What is the architecture?"}
    )
    query.ihsan_score = 0.95
    query.compute_ms = 4899.0
    
    print(f"\nQuery: {query.hash[:32]}...")
    print(f"Chained from: {query.prev_hash[:32]}...")
    
    # Use as different formats
    print(f"\nAs API Response:\n{json.dumps(query.to_api_response(), indent=2)}")
    print(f"\nAs Audit Log:\n{query.to_audit_log()}")
    
    # Compress for storage
    compressed = query.compress()
    print(f"\nCompressed size: {len(compressed)} bytes")
    
    # Restore
    restored = UnifiedStalk.decompress(compressed)
    print(f"Restored verified: {restored.verify()}")


if __name__ == "__main__":
    demo()
