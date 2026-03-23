"""
URP Persistence — the sea must survive restarts.

Saves/restores URP state to disk so the flywheel doesn't reset
between Python process invocations. JSONL for append-only truth,
JSON for snapshot state.

Standing on: Lamport (state persistence), Deming (no data loss).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("bizra.urp.persistence")

DEFAULT_URP_DIR = Path.home() / ".bizra" / "urp"


def save_urp_state(urp_service: Any, state_dir: Optional[Path] = None) -> Path:
    """Persist URP state to disk."""
    d = state_dir or DEFAULT_URP_DIR
    d.mkdir(parents=True, exist_ok=True)

    state = {
        "urp_id": urp_service._urp_id,
        "genesis_complete": urp_service.genesis_complete,
        "constitution_hash": (
            urp_service.constitution.hash() if urp_service.constitution else None
        ),
        "connected_nodes": {
            nid: {
                "node_id": reg.node_id,
                "public_key": reg.public_key,
                "connected_at": reg.connected_at,
            }
            for nid, reg in urp_service.connected_nodes.items()
        },
        "sat_agents": {
            aid: {
                "agent_id": a.agent_id,
                "role": a.role,
                "frozen": a.frozen,
                "missions_processed": a.missions_processed,
            }
            for aid, a in urp_service.sat_agents.items()
        },
        "resource_pool": {
            "seed_treasury": (
                urp_service.resource_pool.seed_treasury
                if urp_service.resource_pool
                else 0
            ),
            "zakat_pool": (
                urp_service.resource_pool.zakat_pool if urp_service.resource_pool else 0
            ),
            "knowledge_count": (
                len(urp_service.resource_pool.knowledge)
                if urp_service.resource_pool
                else 0
            ),
            "reflex_count": (
                len(urp_service.resource_pool.shared_reflexes)
                if urp_service.resource_pool
                else 0
            ),
        },
        "membrane": urp_service.membrane.stats() if urp_service.membrane else {},
        "saved_at": time.time(),
    }

    state_path = d / "urp_state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    # Save knowledge entries as JSONL (append-only truth)
    if urp_service.resource_pool and urp_service.resource_pool.knowledge:
        knowledge_path = d / "knowledge.jsonl"
        with open(knowledge_path, "w") as f:
            for entry in urp_service.resource_pool.knowledge:
                record = {
                    "content_hash": entry.content_hash,
                    "content_preview": entry.content_preview,
                    "contributor_node": entry.contributor_node,
                    "ihsan_score": entry.ihsan_score,
                    "receipt_id": entry.receipt_id,
                    "timestamp": entry.timestamp,
                }
                f.write(json.dumps(record) + "\n")

    # Save receipt log as JSONL
    if urp_service.resource_pool and urp_service.resource_pool.receipt_log:
        receipt_path = d / "receipts.jsonl"
        with open(receipt_path, "w") as f:
            for receipt in urp_service.resource_pool.receipt_log:
                f.write(json.dumps(receipt, default=str) + "\n")

    logger.info("URP state saved to %s", d)
    return state_path


def load_urp_state(state_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load URP state from disk. Returns None if no state exists."""
    d = state_dir or DEFAULT_URP_DIR
    state_path = d / "urp_state.json"

    if not state_path.exists():
        return None

    with open(state_path) as f:
        state = json.load(f)

    # Load knowledge entries
    knowledge_path = d / "knowledge.jsonl"
    knowledge = []
    if knowledge_path.exists():
        with open(knowledge_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    knowledge.append(json.loads(line))
    state["knowledge_entries"] = knowledge

    # Load receipt log
    receipt_path = d / "receipts.jsonl"
    receipts = []
    if receipt_path.exists():
        with open(receipt_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    receipts.append(json.loads(line))
    state["receipt_log"] = receipts

    logger.info(
        "URP state loaded: %d knowledge, %d receipts",
        len(knowledge),
        len(receipts),
    )
    return state


def restore_urp_from_state(state: Dict[str, Any]) -> Any:
    """Restore a URPService from persisted state."""
    from core.urp.constitution import Constitution
    from core.urp.membrane import ConstitutionalMembrane
    from core.urp.resource_pool import KnowledgeEntry, ResourcePool
    from core.urp.service import (
        NodeRegistration,
        SATAgent,
        URPGenesisReceipt,
        URPService,
    )

    urp = URPService()
    urp._urp_id = state.get("urp_id", "")
    urp.genesis_complete = state.get("genesis_complete", False)
    urp.constitution = Constitution()
    urp.membrane = ConstitutionalMembrane(urp.constitution)
    urp.resource_pool = ResourcePool()

    # Restore resource pool
    pool_state = state.get("resource_pool", {})
    urp.resource_pool.seed_treasury = pool_state.get("seed_treasury", 0)
    urp.resource_pool.zakat_pool = pool_state.get("zakat_pool", 0)

    # Restore knowledge
    for entry in state.get("knowledge_entries", []):
        ke = KnowledgeEntry(
            content_hash=entry["content_hash"],
            content_preview=entry["content_preview"],
            contributor_node=entry["contributor_node"],
            ihsan_score=entry["ihsan_score"],
            receipt_id=entry["receipt_id"],
            timestamp=entry["timestamp"],
        )
        urp.resource_pool.knowledge.append(ke)
        urp.resource_pool._content_hashes.add(entry["content_hash"])

    # Restore receipt log
    urp.resource_pool.receipt_log = state.get("receipt_log", [])

    # Restore SAT agents
    for aid, astate in state.get("sat_agents", {}).items():
        urp.sat_agents[aid] = SATAgent(
            agent_id=astate["agent_id"],
            role=astate["role"],
            frozen=astate.get("frozen", False),
            missions_processed=astate.get("missions_processed", 0),
        )

    # Restore connected nodes
    for nid, nstate in state.get("connected_nodes", {}).items():
        urp.connected_nodes[nid] = NodeRegistration(
            node_id=nstate["node_id"],
            public_key=nstate["public_key"],
            connected_at=nstate["connected_at"],
        )

    # Build genesis receipt
    urp._genesis_receipt = URPGenesisReceipt(
        urp_id=urp._urp_id,
        constitution_hash=urp.constitution.hash(),
        sat_count=len(urp.sat_agents),
        founder_node=next(iter(urp.connected_nodes), "unknown"),
        timestamp=state.get("saved_at", 0),
    )

    logger.info(
        "URP restored: id=%s, nodes=%d, knowledge=%d, receipts=%d",
        urp._urp_id[:16],
        len(urp.connected_nodes),
        len(urp.resource_pool.knowledge),
        len(urp.resource_pool.receipt_log),
    )
    return urp
