"""
Resource Pool — the 'Sea' that all nodes draw from through the URP membrane.

Contains: knowledge entries, SEED treasury, shared compiled reflexes.
Governed by constitutional thresholds (Gini, Ihsan, Zakat).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bizra.urp.resource_pool")


@dataclass
class KnowledgeEntry:
    """A single knowledge contribution to the House of Wisdom."""

    content_hash: str
    content_preview: str
    contributor_node: str
    ihsan_score: float
    receipt_id: str
    timestamp: float
    embedding: Optional[List[float]] = None


@dataclass
class SharedReflex:
    """A compiled reflex pattern shared with the network."""

    pattern_hash: str
    pattern_description: str
    creator_node: str
    confidence: float
    adopters: List[str] = field(default_factory=list)
    created_at: float = 0.0

    def adopt(self, node_id: str, confidence: float) -> None:
        """Another node independently discovered the same pattern."""
        if node_id not in self.adopters:
            self.adopters.append(node_id)
            # Confidence increases with independent discovery
            self.confidence = min(1.0, self.confidence + confidence * 0.1)


class ResourcePool:
    """The collective resource pool — knowledge, SEED, reflexes."""

    def __init__(self) -> None:
        self.knowledge: List[KnowledgeEntry] = []
        self.shared_reflexes: Dict[str, SharedReflex] = {}
        self.seed_treasury: float = 0.0
        self.zakat_pool: float = 0.0
        self.receipt_log: List[Dict[str, Any]] = []
        self._content_hashes: set = set()

    def mint_genesis_seed(
        self, founder: str, treasury: float, zakat: float
    ) -> Dict[str, Any]:
        """One-time genesis SEED allocation."""
        self.seed_treasury = treasury
        self.zakat_pool = zakat
        receipt = {
            "type": "genesis_mint",
            "founder": founder,
            "treasury": treasury,
            "zakat": zakat,
            "timestamp": time.time(),
        }
        self.receipt_log.append(receipt)
        logger.info("Genesis SEED: treasury=%.2f, zakat=%.2f", treasury, zakat)
        return receipt

    def contribute_knowledge(
        self,
        content: str,
        contributor: str,
        ihsan_score: float,
        receipt_id: str,
        ihsan_floor: float = 0.95,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Contribute knowledge to the pool. Returns True if accepted."""
        if ihsan_score < ihsan_floor:
            logger.debug(
                "Knowledge rejected: ihsan %.4f < floor %.4f", ihsan_score, ihsan_floor
            )
            return False

        content_hash = hashlib.blake2b(content.encode(), digest_size=32).hexdigest()
        if content_hash in self._content_hashes:
            logger.debug("Knowledge rejected: duplicate %s", content_hash[:16])
            return False

        entry = KnowledgeEntry(
            content_hash=content_hash,
            content_preview=content[:200],
            contributor_node=contributor,
            ihsan_score=ihsan_score,
            receipt_id=receipt_id,
            timestamp=time.time(),
            embedding=embedding,
        )
        self.knowledge.append(entry)
        self._content_hashes.add(content_hash)
        return True

    def contribute_reflex(
        self,
        pattern_hash: str,
        description: str,
        creator: str,
        confidence: float,
        threshold: float = 0.85,
    ) -> bool:
        """Share a compiled reflex with the pool."""
        if confidence < threshold:
            return False

        if pattern_hash in self.shared_reflexes:
            self.shared_reflexes[pattern_hash].adopt(creator, confidence)
        else:
            self.shared_reflexes[pattern_hash] = SharedReflex(
                pattern_hash=pattern_hash,
                pattern_description=description,
                creator_node=creator,
                confidence=confidence,
                adopters=[creator],
                created_at=time.time(),
            )
        return True

    def draw_knowledge(
        self, requesting_node: str, limit: int = 10
    ) -> List[KnowledgeEntry]:
        """Draw knowledge from the pool (excludes own contributions)."""
        return [
            e
            for e in sorted(self.knowledge, key=lambda x: x.ihsan_score, reverse=True)
            if e.contributor_node != requesting_node
        ][:limit]

    def record_receipt(self, receipt: Dict[str, Any]) -> None:
        """Record a validated receipt in the pool."""
        self.receipt_log.append(receipt)

    def stats(self) -> Dict[str, Any]:
        """Pool statistics."""
        return {
            "knowledge_entries": len(self.knowledge),
            "shared_reflexes": len(self.shared_reflexes),
            "seed_treasury": self.seed_treasury,
            "zakat_pool": self.zakat_pool,
            "receipts_recorded": len(self.receipt_log),
            "unique_contributors": len(set(e.contributor_node for e in self.knowledge)),
        }
