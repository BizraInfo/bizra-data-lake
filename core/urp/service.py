"""
URP Service — the orchestrator that ties membrane, pool, and SAT agents together.

This is the heart of the Universal Resource Protocol.
Node0 mints it at genesis. Every node connects through it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.urp.constitution import Constitution
from core.urp.membrane import ConstitutionalMembrane
from core.urp.resource_pool import ResourcePool

logger = logging.getLogger("bizra.urp.service")


@dataclass
class NodeRegistration:
    """Record of a connected node."""

    node_id: str
    public_key: str
    connected_at: float
    sat_agents_contributed: int = 5


@dataclass
class SATAgent:
    """A SAT agent deployed into the URP."""

    agent_id: str
    role: str
    frozen: bool = False
    missions_processed: int = 0


@dataclass
class URPGenesisReceipt:
    """Proof that the URP was minted."""

    urp_id: str
    constitution_hash: str
    sat_count: int
    founder_node: str
    timestamp: float


class URPService:
    """The Universal Resource Protocol — constitutional membrane for sovereign nodes."""

    def __init__(self) -> None:
        self.constitution: Optional[Constitution] = None
        self.membrane: Optional[ConstitutionalMembrane] = None
        self.resource_pool: Optional[ResourcePool] = None
        self.sat_agents: Dict[str, SATAgent] = {}
        self.connected_nodes: Dict[str, NodeRegistration] = {}
        self.genesis_complete: bool = False
        self._urp_id: str = ""
        self._genesis_receipt: Optional[URPGenesisReceipt] = None

    def mint_genesis(
        self,
        founder_node_id: str,
        founder_public_key: str,
        sat_agent_ids: Optional[List[str]] = None,
    ) -> URPGenesisReceipt:
        """Called ONCE by Node0 at system genesis. Creates the entire membrane."""
        if self.genesis_complete:
            raise RuntimeError("URP genesis already complete — one-time only")

        import hashlib

        # URP gets its own identity
        self._urp_id = hashlib.blake2b(
            f"urp-genesis:{founder_node_id}:{time.time()}".encode(),
            digest_size=32,
        ).hexdigest()

        # Initialize constitution
        self.constitution = Constitution()
        self.membrane = ConstitutionalMembrane(self.constitution)
        self.resource_pool = ResourcePool()

        # Deploy SAT agents
        default_sat = sat_agent_ids or [
            "S1-Validator",
            "S2-Oracle",
            "S3-Mediator",
            "S4-Archivist",
            "S5-Sentinel",
        ]
        for agent_id in default_sat:
            frozen = agent_id == "S2-Oracle"
            role = agent_id.split("-", 1)[1] if "-" in agent_id else agent_id
            self.sat_agents[agent_id] = SATAgent(
                agent_id=agent_id,
                role=role.lower(),
                frozen=frozen,
            )

        # Genesis SEED allocation
        self.resource_pool.mint_genesis_seed(
            founder=founder_node_id,
            treasury=100_000.0,
            zakat=2_500.0,
        )

        # Register founder as first connected node
        self.connected_nodes[founder_node_id] = NodeRegistration(
            node_id=founder_node_id,
            public_key=founder_public_key,
            connected_at=time.time(),
        )

        self.genesis_complete = True
        self._genesis_receipt = URPGenesisReceipt(
            urp_id=self._urp_id,
            constitution_hash=self.constitution.hash(),
            sat_count=len(self.sat_agents),
            founder_node=founder_node_id,
            timestamp=time.time(),
        )

        logger.info(
            "URP Genesis: id=%s, constitution=%s, SAT=%d, founder=%s",
            self._urp_id[:16],
            self.constitution.hash()[:16],
            len(self.sat_agents),
            founder_node_id,
        )
        return self._genesis_receipt

    def register_node(self, node_id: str, public_key: str) -> tuple[bool, str]:
        """Register a new node through the membrane."""
        if not self.genesis_complete:
            return False, "urp_not_initialized"

        if node_id in self.connected_nodes:
            return True, "already_connected"

        # Membrane admission check
        admitted, reason, record = self.membrane.filter_inbound(
            node_id=node_id,
            event_type="node_registration",
            payload={"public_key": public_key},
        )
        if not admitted:
            return False, reason

        self.connected_nodes[node_id] = NodeRegistration(
            node_id=node_id,
            public_key=public_key,
            connected_at=time.time(),
        )
        logger.info(
            "Node registered: %s (total: %d)", node_id, len(self.connected_nodes)
        )
        return True, "registered"

    def submit_receipt(self, node_id: str, receipt: Dict[str, Any]) -> tuple[bool, str]:
        """Node submits a mission receipt through the membrane."""
        if node_id not in self.connected_nodes:
            return False, "not_registered"

        # Constitutional membrane filtering
        admitted, reason, record = self.membrane.filter_inbound(
            node_id=node_id,
            event_type="receipt",
            payload=receipt,
        )
        if not admitted:
            return False, reason

        # SAT agents process the receipt
        for agent_id, agent in self.sat_agents.items():
            if agent.frozen:
                continue
            agent.missions_processed += 1

        # Record in resource pool
        self.resource_pool.record_receipt(receipt)
        return True, "accepted"

    def query_knowledge(
        self, node_id: str, query: str, top_k: int = 5
    ) -> tuple[bool, Any]:
        """Node requests knowledge from the House of Wisdom."""
        if node_id not in self.connected_nodes:
            return False, "not_registered"

        # Draw from pool (excludes own contributions)
        results = self.resource_pool.draw_knowledge(node_id, limit=top_k)
        return True, results

    def contribute_knowledge(
        self,
        node_id: str,
        content: str,
        ihsan_score: float,
        receipt_id: str,
    ) -> bool:
        """Node contributes knowledge through the membrane."""
        if node_id not in self.connected_nodes:
            return False

        return self.resource_pool.contribute_knowledge(
            content=content,
            contributor=node_id,
            ihsan_score=ihsan_score,
            receipt_id=receipt_id,
            ihsan_floor=self.constitution.ihsan_floor,
        )

    def status(self) -> Dict[str, Any]:
        """Full URP status."""
        return {
            "urp_id": self._urp_id[:16] if self._urp_id else None,
            "genesis_complete": self.genesis_complete,
            "constitution_hash": (
                self.constitution.hash()[:16] if self.constitution else None
            ),
            "connected_nodes": len(self.connected_nodes),
            "sat_agents": len(self.sat_agents),
            "sat_frozen": sum(1 for a in self.sat_agents.values() if a.frozen),
            "membrane": self.membrane.stats() if self.membrane else None,
            "resource_pool": self.resource_pool.stats() if self.resource_pool else None,
        }
