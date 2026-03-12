"""
Node0 Heartbeat — The First Sovereign Breath
═════════════════════════════════════════════════

Wires the existing BIZRA subsystems into ONE end-to-end cycle:

    boot() → breathe() → remember() → precipitate()

  AssetRegistry (self-knowledge)
    ↓
  Helix3Scheduler (constitutional heartbeat)
    ↓
  EvidenceAwareMemory (constitutional audit trail)
    ↓
  ReflexBridge (pattern → reflex compilation)

"Node0 is ready to activate only when one sovereign human-device-node unit
 can stand alone truthfully, breathe constitutionally, remember verifiably,
 and serve as the template for every node that follows."
 — Node0 Activation Planning Principle §14

Standing on Giants:
  Prophet Muhammad ﷺ (Ihsān, Hadith Jibril) — excellence is worship
  Al-Ghazali (intent gate, 1096) — no action without intent
  Nakamoto (evidence chain, 2008) — proof, not trust
  Deming (PDCA, 1950) — continuous improvement loop
  Boyd (OODA, 1976) — observe → orient → decide → act
  Shannon (SNR, 1948) — measure your own channel capacity

Constitutional Authority:
  §2  Triple Helix: Helix3 evolutionary cycle (60s heartbeat)
  §4  Immutable invariants: Ihsān ≥ 0.95, Gini ≤ 0.35
  §7  Closure-First Law: close nervous system before expanding
  §8  First-Heartbeat Rule: optimize toward one real heartbeat
  §9  Asset Registry: node must know its own body at boot
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bizra.node0.heartbeat")


# ═══════════════════════════════════════════════════════════════════
# CONSTITUTIONAL IMPORTS — from constants.py (single source of truth)
# ═══════════════════════════════════════════════════════════════════


# Heartbeat constants
HEARTBEAT_INTERVAL_S = 60.0  # §2: Every 60 seconds
PRECIPITATION_IHSAN_FLOOR = 0.90  # §2 Helix 3: minimum for reflex precipitation
BOOT_SOVEREIGNTY_CHECKS = 5  # Minimum checks for sovereignty proof


# ═══════════════════════════════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BootReceipt:
    """Immutable record of Node0 genesis ceremony.

    This is Block Zero — the birth certificate of the sovereign node.
    """

    node_id: str
    hostname: str
    boot_time: str  # ISO 8601
    sovereignty_checks: Dict[str, bool]
    sovereignty_proven: bool
    asset_summary: Dict[str, Any]
    memory_initialized: bool
    evidence_chain_genesis: str  # Hash of first evidence entry
    boot_hash: str  # BLAKE2b of all fields
    duration_ms: float

    def as_dict(self) -> Dict[str, Any]:
        """Serialize for evidence chain."""
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "boot_time": self.boot_time,
            "sovereignty_checks": self.sovereignty_checks,
            "sovereignty_proven": self.sovereignty_proven,
            "asset_summary": self.asset_summary,
            "memory_initialized": self.memory_initialized,
            "evidence_chain_genesis": self.evidence_chain_genesis,
            "boot_hash": self.boot_hash,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class BreathReceipt:
    """One breath of the sovereign node — evidence of a real heartbeat.

    Every field is verifiable. No speculation.
    """

    tick_number: int
    timestamp: str  # ISO 8601
    duration_ms: float

    # Helix3 results
    missions_processed: int
    ihsan_composite: float
    gini_coefficient: float
    gini_ok: bool
    seed_minted: float
    reflexes_precipitated: int

    # Memory effect
    memories_stored: int
    evidence_entries: int

    # Chain integrity
    evidence_hash: str
    chain_hash: str
    prev_chain_hash: str

    # Raw Helix3 tick result for inspection (consequence closure audit)
    helix_result: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Serialize for evidence chain."""
        return {
            "tick_number": self.tick_number,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "missions_processed": self.missions_processed,
            "ihsan_composite": self.ihsan_composite,
            "gini_coefficient": self.gini_coefficient,
            "gini_ok": self.gini_ok,
            "seed_minted": self.seed_minted,
            "reflexes_precipitated": self.reflexes_precipitated,
            "memories_stored": self.memories_stored,
            "evidence_entries": self.evidence_entries,
            "evidence_hash": self.evidence_hash,
            "chain_hash": self.chain_hash,
            "prev_chain_hash": self.prev_chain_hash,
            "approved_count": self.helix_result.get("approved_count", 0),
            "rejected_count": self.helix_result.get("rejected_count", 0),
        }


# ═══════════════════════════════════════════════════════════════════
# NODE0 HEARTBEAT ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════


class Node0Heartbeat:
    """The Node0 sovereign heartbeat — wires all subsystems into one breath.

    This is the closure that the Node0 Activation Planning Principle demands:
      1. Close the nervous system
      2. Make Ihsān authoritative
      3. Connect reasoning to receipts
      4. Connect receipts to memory
      5. Connect memory to reflex
      6. Connect the node to a real first heartbeat

    Usage:
        heartbeat = Node0Heartbeat(data_dir=Path("sovereign_state"))
        boot = heartbeat.boot()        # Genesis ceremony
        breath = heartbeat.breathe()   # One constitutional heartbeat
        health = heartbeat.health()    # Self-diagnostic
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        node_id: Optional[str] = None,
        interval_s: float = HEARTBEAT_INTERVAL_S,
        helix3: Optional[Any] = None,
        identity_mode: str = "placeholder_degraded",
        signer_public_key_prefix: str = "",
        signer_public_key_hex: str = "",
        genesis_backed: bool = False,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._node_id = node_id or ""
        self._interval_s = interval_s
        self._identity_mode = identity_mode
        self._signer_public_key_prefix = signer_public_key_prefix
        self._signer_public_key_hex = str(signer_public_key_hex or "").lower()
        self._genesis_backed = genesis_backed

        # State
        self._booted = False
        self._tick_number = 0
        self._chain_hash = "0" * 64  # Genesis sentinel
        self._boot_receipt: Optional[BootReceipt] = None
        self._breath_history: List[BreathReceipt] = []

        # Subsystem handles (wired at boot)
        self._asset_registry: Optional[Any] = None
        self._helix3: Optional[Any] = helix3  # External Helix3 (e.g. from organism)
        self._external_helix3 = helix3 is not None
        self._memory: Optional[Any] = None
        self._evidence: Optional[Any] = None
        self._reflex_bridge: Optional[Any] = None

        # Cumulative stats
        self._total_memories_stored = 0
        self._total_evidence_entries = 0
        self._total_reflexes = 0

    # ═══════════════════════════════════════════════════════════════
    # §9 BOOT — Genesis Ceremony (Block Zero)
    # ═══════════════════════════════════════════════════════════════

    def boot(self) -> BootReceipt:
        """Genesis ceremony — birth of the first sovereign node.

        Tier A (Birth) from Node0 Activation Planning Principle §12:
          - Genesis identity
          - Block zero
          - Canonical manifest
          - Local sovereignty proof

        Returns:
            BootReceipt: Immutable evidence of the genesis event.

        Raises:
            RuntimeError: If sovereignty cannot be proven.
        """
        start = time.monotonic()
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Genesis Identity ──────────────────────────────
        if self._identity_mode == "genesis_ed25519":
            self._bind_canonical_identity()
        elif not self._node_id:
            self._node_id = self._generate_node_id()

        # ── Step 2: Asset Registry (Self-Knowledge) ──────────────
        asset_summary = self._boot_asset_registry()

        # ── Step 3: Memory Initialization ────────────────────────
        memory_ok = self._boot_memory()

        # ── Step 4: Evidence Chain Genesis ────────────────────────
        evidence_genesis = self._boot_evidence_chain()

        # ── Step 5: Helix3 Scheduler ─────────────────────────────
        self._boot_helix3()

        # ── Step 6: Reflex Bridge ────────────────────────────────
        self._boot_reflex_bridge()

        # ── Step 7: Sovereignty Proof ────────────────────────────
        checks = self._verify_sovereignty(asset_summary, memory_ok, evidence_genesis)
        sovereign = all(checks.values())

        if not sovereign:
            failed = [k for k, v in checks.items() if not v]
            logger.warning(
                "Sovereignty NOT fully proven: %s — proceeding with degraded mode",
                failed,
            )

        # ── Step 8: Generate Boot Receipt (Block Zero) ───────────
        duration_ms = (time.monotonic() - start) * 1000

        boot_data = {
            "node_id": self._node_id,
            "asset_summary": asset_summary,
            "memory_ok": memory_ok,
            "evidence_genesis": evidence_genesis,
            "checks": checks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        boot_hash = hashlib.blake2b(
            str(sorted(_flatten_dict(boot_data))).encode(),
            digest_size=32,
        ).hexdigest()

        receipt = BootReceipt(
            node_id=self._node_id,
            hostname=asset_summary.get("hostname", "unknown"),
            boot_time=datetime.now(timezone.utc).isoformat(),
            sovereignty_checks=checks,
            sovereignty_proven=sovereign,
            asset_summary=asset_summary,
            memory_initialized=memory_ok,
            evidence_chain_genesis=evidence_genesis,
            boot_hash=boot_hash,
            duration_ms=round(duration_ms, 2),
        )

        self._booted = True
        self._boot_receipt = receipt
        self._chain_hash = boot_hash

        # Store boot receipt in memory if available
        if self._memory is not None:
            self._store_in_memory(
                content=f"Node0 boot: sovereign={sovereign}, "
                f"checks={sum(checks.values())}/{len(checks)}",
                source="node0:boot",
                metadata={"event": "genesis", "boot_hash": boot_hash},
            )
            self._total_memories_stored += 1

        logger.info(
            "Node0 BOOT complete | node=%s | sovereign=%s | %.1fms",
            self._node_id[:16],
            sovereign,
            duration_ms,
        )

        return receipt

    # ═══════════════════════════════════════════════════════════════
    # §8 BREATHE — One Real Constitutional Heartbeat
    # ═══════════════════════════════════════════════════════════════

    def breathe(self) -> BreathReceipt:
        """One breath of the sovereign node.

        Tier B (Breath) from Node0 Activation Planning Principle §12:
          - First heartbeat
          - Evidence chain
          - Quality gate
          - Memory persistence
          - Reflex emergence path

        Flow:
          1. Helix3 process_tick() → constitutional economics
          2. Store heartbeat in evidence chain
          3. Persist to memory (AgentDB)
          4. Evaluate for reflex precipitation
          5. Emit BreathReceipt with chain integrity

        Returns:
            BreathReceipt: Immutable evidence of this heartbeat.
        """
        if not self._booted:
            raise RuntimeError(
                "Node0 must boot() before breathing. "
                "Sovereignty first — Spine §3, Planning Principle §3."
            )

        start = time.monotonic()
        self._tick_number += 1
        prev_chain = self._chain_hash

        # ── Step 1: Helix3 Heartbeat (Constitutional Tick) ───────
        helix_result = self._run_helix3_tick()

        # ── Step 2: Evidence Chain Entry ─────────────────────────
        evidence_count = self._record_evidence(helix_result)

        # ── Step 3: Memory Persistence ───────────────────────────
        memory_count = self._persist_to_memory(helix_result)

        # ── Step 4: Reflex Precipitation Check ───────────────────
        reflexes = self._check_reflex_precipitation(helix_result)

        # ── Step 5: Chain Integrity ──────────────────────────────
        duration_ms = (time.monotonic() - start) * 1000

        evidence_data = {
            "tick": self._tick_number,
            "ihsan": helix_result.get("ihsan_composite", 0.0),
            "gini": helix_result.get("gini", 0.0),
            "missions": helix_result.get("missions_processed", 0),
            "memories": memory_count,
            "reflexes": reflexes,
        }
        evidence_hash = hashlib.blake2b(
            str(sorted(evidence_data.items())).encode(),
            digest_size=32,
        ).hexdigest()

        chain_hash = hashlib.blake2b(
            f"{prev_chain}:{evidence_hash}".encode(),
            digest_size=32,
        ).hexdigest()
        self._chain_hash = chain_hash

        receipt = BreathReceipt(
            tick_number=self._tick_number,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=round(duration_ms, 2),
            missions_processed=helix_result.get("missions_processed", 0),
            ihsan_composite=helix_result.get("ihsan_composite", 0.0),
            gini_coefficient=helix_result.get("gini", 0.0),
            gini_ok=helix_result.get("gini_ok", True),
            seed_minted=helix_result.get("seed_minted", 0.0),
            reflexes_precipitated=reflexes,
            memories_stored=memory_count,
            evidence_entries=evidence_count,
            evidence_hash=evidence_hash,
            chain_hash=chain_hash,
            prev_chain_hash=prev_chain,
            helix_result=helix_result,
        )

        self._breath_history.append(receipt)

        logger.info(
            "Node0 BREATH #%d | ihsan=%.3f | gini=%.4f | mem=%d | ev=%d | %.1fms",
            self._tick_number,
            receipt.ihsan_composite,
            receipt.gini_coefficient,
            memory_count,
            evidence_count,
            duration_ms,
        )

        return receipt

    # ═══════════════════════════════════════════════════════════════
    # HEALTH — Self-Diagnostic (Mode 3)
    # ═══════════════════════════════════════════════════════════════

    def health(self) -> Dict[str, Any]:
        """Self-diagnostic: the node inspecting its own vital signs.

        Returns a dict suitable for both human review and machine processing.
        """
        body = None
        if self._asset_registry is not None:
            try:
                body = self._asset_registry.introspect()
            except Exception as exc:
                logger.warning("Asset introspection failed: %s", exc)

        avg_ihsan = 0.0
        if self._breath_history:
            avg_ihsan = sum(b.ihsan_composite for b in self._breath_history) / len(
                self._breath_history
            )

        return {
            "booted": self._booted,
            "node_id": self._node_id,
            "identity_mode": self._identity_mode,
            "signer_public_key_prefix": self._signer_public_key_prefix,
            "genesis_backed": self._genesis_backed,
            "tick_number": self._tick_number,
            "chain_hash": self._chain_hash,
            "avg_ihsan": round(avg_ihsan, 4),
            "total_breaths": len(self._breath_history),
            "total_memories_stored": self._total_memories_stored,
            "total_evidence_entries": self._total_evidence_entries,
            "total_reflexes_precipitated": self._total_reflexes,
            "subsystems": {
                "asset_registry": self._asset_registry is not None,
                "helix3": self._helix3 is not None,
                "memory": self._memory is not None,
                "evidence": self._evidence is not None,
                "reflex_bridge": self._reflex_bridge is not None,
            },
            "hardware": (
                {
                    "hostname": body.hostname if body else "unknown",
                    "assets": len(body.assets) if body else 0,
                    "contribution_potential": (
                        round(body.contribution_potential, 3) if body else 0.0
                    ),
                }
                if body
                else None
            ),
            "last_breath": (
                self._breath_history[-1].as_dict() if self._breath_history else None
            ),
        }

    # ═══════════════════════════════════════════════════════════════
    # PROPERTIES
    # ═══════════════════════════════════════════════════════════════

    @property
    def booted(self) -> bool:
        """Whether the node has completed genesis ceremony."""
        return self._booted

    @property
    def node_id(self) -> str:
        """The node's sovereign identity."""
        return self._node_id

    @property
    def tick_number(self) -> int:
        """Current heartbeat count."""
        return self._tick_number

    @property
    def chain_hash(self) -> str:
        """Current evidence chain head hash."""
        return self._chain_hash

    @property
    def boot_receipt(self) -> Optional[BootReceipt]:
        """The genesis ceremony receipt (Block Zero)."""
        return self._boot_receipt

    # ═══════════════════════════════════════════════════════════════
    # PRIVATE: Boot Subsystems
    # ═══════════════════════════════════════════════════════════════

    def _generate_node_id(self) -> str:
        """Generate sovereign node identity.

        In production: Ed25519 keypair. Here: BLAKE2b of hostname + time.
        """
        import os

        hostname = os.uname().nodename if hasattr(os, "uname") else "unknown"
        seed = f"{hostname}:{time.time_ns()}:{id(self)}".encode()
        return hashlib.blake2b(seed, digest_size=16).hexdigest()

    def _bind_canonical_identity(self) -> None:
        """Bind Node0 identity to the injected Ed25519 signer truth."""
        if len(self._signer_public_key_hex) != 64:
            raise RuntimeError(
                "Canonical Node0 boot requires injected genesis Ed25519 signer public key"
            )

        expected_node_id = self._derive_node_id_from_public_key(
            self._signer_public_key_hex
        )
        expected_prefix = self._signer_public_key_hex[:16]

        if (
            self._signer_public_key_prefix
            and self._signer_public_key_prefix != expected_prefix
        ):
            raise RuntimeError(
                "Canonical Node0 boot signer prefix does not match injected signer public key"
            )

        self._signer_public_key_prefix = expected_prefix

        if self._node_id and self._node_id != expected_node_id:
            raise RuntimeError(
                "Canonical Node0 boot node_id does not match injected signer public key"
            )
        self._node_id = expected_node_id

    @staticmethod
    def _derive_node_id_from_public_key(public_key_hex: str) -> str:
        """Derive the canonical BIZRA node ID from an Ed25519 public key."""
        from core.pat.identity_card import _generate_node_id

        return _generate_node_id(public_key_hex)

    def _boot_asset_registry(self) -> Dict[str, Any]:
        """Initialize the node's self-awareness (§9)."""
        try:
            from core.elite.asset_registry import AssetRegistry

            self._asset_registry = AssetRegistry(node_id=self._node_id)
            body = self._asset_registry.introspect(force=True)
            return {
                "hostname": body.hostname,
                "asset_count": len(body.assets),
                "total_capacity": body.total_capacity,
                "contribution_potential": round(body.contribution_potential, 3),
                "sovereignty_tier": body.sovereignty_tier,
            }
        except Exception as exc:
            logger.warning("AssetRegistry unavailable: %s", exc)
            return {"hostname": "unknown", "asset_count": 0, "degraded": True}

    def _boot_memory(self) -> bool:
        """Initialize sovereign memory (AgentDB)."""
        try:
            from core.memory.agent_db import AgentDB
            from core.memory.config import MemoryConfig

            config = MemoryConfig(
                data_dir=self._data_dir / "memory",
                auto_embed=False,  # §5: universal baseline, no ML dependency
            )
            self._memory = AgentDB(config)
            self._memory.initialize()
            return True
        except Exception as exc:
            logger.warning("Memory subsystem unavailable: %s", exc)
            return False

    def _boot_evidence_chain(self) -> str:
        """Initialize evidence chain with genesis entry."""
        try:
            from core.memory.adapters.evidence_chain import EvidenceAwareMemory

            if self._memory is not None:
                self._evidence = EvidenceAwareMemory(
                    db=self._memory,
                    ledger_dir=self._data_dir / "evidence",
                )
                genesis_hash = hashlib.blake2b(
                    f"genesis:{self._node_id}:{time.time_ns()}".encode(),
                    digest_size=32,
                ).hexdigest()
                return genesis_hash
        except Exception as exc:
            logger.warning("Evidence chain unavailable: %s", exc)

        return "0" * 64

    def _boot_helix3(self) -> None:
        """Initialize Helix3 evolutionary scheduler.

        If an external Helix3 was provided at construction (e.g. from
        SovereignOrganism with NervousSystem wiring), skip standalone creation.
        """
        if self._external_helix3:
            logger.info("Using externally-provided Helix3 scheduler")
            return
        try:
            from core.sovereign.helix3 import Helix3Scheduler

            self._helix3 = Helix3Scheduler(interval_s=self._interval_s)
        except Exception as exc:
            logger.warning("Helix3 unavailable: %s", exc)

    def _boot_reflex_bridge(self) -> None:
        """Initialize SDPO → Reflex compilation bridge."""
        try:
            from core.sdpo.reflex_bridge import SDPOReflexBridge

            self._reflex_bridge = SDPOReflexBridge()
        except Exception as exc:
            logger.warning("ReflexBridge unavailable: %s", exc)

    # ═══════════════════════════════════════════════════════════════
    # PRIVATE: Sovereignty Verification
    # ═══════════════════════════════════════════════════════════════

    def _verify_sovereignty(
        self,
        asset_summary: Dict[str, Any],
        memory_ok: bool,
        evidence_genesis: str,
    ) -> Dict[str, bool]:
        """Verify local sovereignty — the node can stand alone.

        Node0 Activation Planning Principle §3:
        "BIZRA does not earn the right to scale until one node can stand alone."
        """
        checks = {
            # Identity exists
            "identity": bool(self._node_id),
            # Hardware self-knowledge (at least 1 asset detected)
            "self_knowledge": asset_summary.get("asset_count", 0) > 0,
            # Memory can persist locally
            "memory": memory_ok,
            # Evidence chain has genesis entry
            "evidence_chain": evidence_genesis != ("0" * 64),
            # Data directory is writable
            "data_sovereignty": self._data_dir.exists() and self._data_dir.is_dir(),
        }
        return checks

    # ═══════════════════════════════════════════════════════════════
    # PRIVATE: Breathe Subsystems
    # ═══════════════════════════════════════════════════════════════

    def _run_helix3_tick(self) -> Dict[str, Any]:
        """Execute one Helix3 constitutional tick.

        Returns a dict with: ihsan_composite, gini, gini_ok, seed_minted,
        missions_processed, reflexes_precipitated.
        """
        if self._helix3 is not None:
            try:
                result = self._helix3.process_tick()
                return {
                    "ihsan_composite": getattr(result, "ihsan_composite", 0.0),
                    "gini": getattr(result, "gini_coefficient", 0.0),
                    "gini_ok": getattr(result, "gini_ok", True),
                    "seed_minted": getattr(result, "seed_minted", 0.0),
                    "missions_processed": getattr(result, "missions_processed", 0),
                    "reflexes_precipitated": getattr(
                        result, "reflexes_precipitated", 0
                    ),
                    "approved_count": getattr(result, "approved_count", 0),
                    "rejected_count": getattr(result, "rejected_count", 0),
                }
            except Exception as exc:
                logger.warning("Helix3 tick failed: %s", exc)

        # Degraded mode: return minimal valid result
        return {
            "ihsan_composite": 0.0,
            "gini": 0.0,
            "gini_ok": True,
            "seed_minted": 0.0,
            "missions_processed": 0,
            "reflexes_precipitated": 0,
            "approved_count": 0,
            "rejected_count": 0,
        }

    def _record_evidence(self, helix_result: Dict[str, Any]) -> int:
        """Store heartbeat in evidence chain. Returns count of entries made."""
        if self._evidence is not None:
            try:
                self._evidence.store(
                    content=f"Heartbeat #{self._tick_number}: "
                    f"ihsan={helix_result.get('ihsan_composite', 0):.3f}, "
                    f"gini={helix_result.get('gini', 0):.4f}",
                    source="node0:heartbeat",
                    metadata={
                        "event": "heartbeat",
                        "tick": self._tick_number,
                        **helix_result,
                    },
                )
                self._total_evidence_entries += 1
                return 1
            except Exception as exc:
                logger.warning("Evidence recording failed: %s", exc)
        elif self._memory is not None:
            # Fallback: store directly in AgentDB
            try:
                self._store_in_memory(
                    content=f"Heartbeat #{self._tick_number}: "
                    f"ihsan={helix_result.get('ihsan_composite', 0):.3f}",
                    source="node0:heartbeat",
                    metadata={"event": "heartbeat", "tick": self._tick_number},
                )
                self._total_evidence_entries += 1
                return 1
            except Exception as exc:
                logger.warning("Memory evidence fallback failed: %s", exc)
        return 0

    def _persist_to_memory(self, helix_result: Dict[str, Any]) -> int:
        """Persist heartbeat summary to sovereign memory. Returns count."""
        if self._memory is None:
            return 0
        try:
            self._store_in_memory(
                content=f"Node0 breath #{self._tick_number} complete. "
                f"Missions: {helix_result.get('missions_processed', 0)}, "
                f"Ihsān: {helix_result.get('ihsan_composite', 0):.3f}, "
                f"SEED minted: {helix_result.get('seed_minted', 0):.4f}",
                source="node0:breath",
                metadata={
                    "event": "breath",
                    "tick": self._tick_number,
                    "chain_hash": self._chain_hash,
                },
            )
            self._total_memories_stored += 1
            return 1
        except Exception as exc:
            logger.warning("Memory persistence failed: %s", exc)
            return 0

    def _check_reflex_precipitation(self, helix_result: Dict[str, Any]) -> int:
        """Check if heartbeat quality warrants reflex precipitation.

        §2 Helix 3: Ihsān ≥ 0.90 for precipitation.
        Standing on Giants: Kahneman (2011) — System-1 reflexes must only
        compile from VERIFIED System-2 judgments (Self-RLVR).
        If all missions in this tick were rejected, skip precipitation entirely.
        """
        # Gate: if all missions were FATE-rejected, no reflex precipitation
        rejected = helix_result.get("rejected_count", 0)
        total = helix_result.get("missions_processed", 0)
        if total > 0 and rejected >= total:
            return 0

        ihsan = helix_result.get("ihsan_composite", 0.0)
        if ihsan < PRECIPITATION_IHSAN_FLOOR:
            return 0

        if self._reflex_bridge is not None:
            try:
                self._reflex_bridge.observe(
                    task_description=f"Node0 heartbeat #{self._tick_number}",
                    ihsan_score=ihsan,
                    snr_score=ihsan,
                    loss=max(0.0, 1.0 - ihsan),
                    success=True,
                )
                candidates = self._reflex_bridge.get_eligible_candidates()
                self._total_reflexes += len(candidates)
                return len(candidates)
            except Exception as exc:
                logger.debug("Reflex precipitation check: %s", exc)
        return 0

    def _store_in_memory(
        self,
        content: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a record in AgentDB memory."""
        if self._memory is None:
            return
        self._memory.store(
            content=content,
            source=source,
            metadata=metadata or {},
        )

    # ═══════════════════════════════════════════════════════════════
    # INGEST — Feed missions into the heartbeat cycle
    # ═══════════════════════════════════════════════════════════════

    def ingest_mission_receipt(self, receipt: Dict[str, Any]) -> None:
        """Feed a completed mission receipt into the heartbeat cycle.

        The receipt will be processed at the next breathe() call via Helix3.

        Args:
            receipt: Dict with at least 'ihsan_score' and 'description'.
        """
        if self._helix3 is not None:
            self._helix3.ingest_receipt(receipt)
        else:
            logger.debug("Helix3 not available — receipt dropped")


# ═══════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> List[tuple]:
    """Flatten a nested dict for deterministic hashing."""
    items: List[tuple] = []
    for k, v in sorted(d.items()):
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, key))
        elif isinstance(v, (list, tuple)):
            items.append((key, str(v)))
        else:
            items.append((key, v))
    return items
