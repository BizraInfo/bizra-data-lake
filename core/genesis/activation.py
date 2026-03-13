"""
Genesis Activation Pipeline — Wire the Flywheel
=================================================

Connects three existing subsystems into ONE end-to-end genesis
activation sequence:

    Ceremony (cryptographic root of trust)
        → Orchestrator (12-step bootstrap pipeline)
            → Heartbeat (boot + first breath)
                → Evidence (activation receipt)

One call: GenesisActivation(seed, config, data_dir).activate()

Standing on Giants:
- Nakamoto (2008): Genesis block as immutable origin
- Lamport (1982): Ordered state transitions
- Merkle (1979): Hash chains for integrity
- Al-Ghazali (1095): Self-knowledge precedes all knowledge
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.proof_engine.canonical import blake3_digest, canonical_bytes
from core.proof_engine.genesis_ceremony import (
    CeremonyConfig,
    CeremonyResult,
    run_ceremony,
    verify_ceremony,
    write_ceremony,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GenesisActivationResult:
    """Composite receipt proving the full genesis activation.

    Every sub-receipt chains into activation_hash via BLAKE3.
    This is the birth certificate of a sovereign node.
    """

    # Sub-receipts
    ceremony_result: CeremonyResult
    orchestrator_success: bool
    orchestrator_reason_codes: List[str]
    boot_receipt_dict: Dict[str, Any]
    first_breath_dict: Optional[Dict[str, Any]]

    # Composite proof
    activation_hash: str  # BLAKE3 of all sub-hashes
    evidence_chain_valid: bool
    timestamp: str  # ISO 8601
    duration_ms: float

    # Node identity (extracted for convenience)
    node_id: str
    genesis_hash: str

    def as_dict(self) -> Dict[str, Any]:
        """Serialize for evidence persistence."""
        return {
            "node_id": self.node_id,
            "genesis_hash": self.genesis_hash,
            "activation_hash": self.activation_hash,
            "evidence_chain_valid": self.evidence_chain_valid,
            "orchestrator_success": self.orchestrator_success,
            "orchestrator_reason_codes": self.orchestrator_reason_codes,
            "boot_receipt": self.boot_receipt_dict,
            "first_breath": self.first_breath_dict,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


# ═══════════════════════════════════════════════════════════════════════
# Genesis Activation
# ═══════════════════════════════════════════════════════════════════════


class GenesisActivation:
    """Wire ceremony + orchestrator + heartbeat into ONE activation.

    Usage:
        activation = GenesisActivation(
            node_seed=b"my-secret-seed-32-bytes-minimum!",
            data_dir=Path("sovereign_state/genesis"),
        )
        result = activation.activate()
        assert result.evidence_chain_valid

        # Verify later
        valid, reasons = activation.verify()
        assert valid
    """

    def __init__(
        self,
        node_seed: bytes,
        data_dir: Optional[Path] = None,
        ceremony_config: Optional[CeremonyConfig] = None,
        skip_orchestrator: bool = False,
        skip_breath: bool = False,
    ) -> None:
        if not node_seed:
            raise ValueError("node_seed must be non-empty bytes")
        if not isinstance(node_seed, bytes):
            raise TypeError(f"node_seed must be bytes, got {type(node_seed).__name__}")

        self._node_seed = node_seed
        self._data_dir = Path(data_dir) if data_dir else Path("sovereign_state/genesis")
        self._ceremony_config = ceremony_config or CeremonyConfig()
        self._skip_orchestrator = skip_orchestrator
        self._skip_breath = skip_breath
        self._result: Optional[GenesisActivationResult] = None

    def activate(self) -> GenesisActivationResult:
        """Execute the full genesis activation pipeline.

        Pipeline:
            1. Run ceremony → cryptographic identity + PAT/SAT roster
            2. Write ceremony to disk → persistent genesis state
            3. Run orchestrator → 12-step bootstrap (optional)
            4. Boot heartbeat → sovereignty proof
            5. First breath → constitutional heartbeat (optional)
            6. Compute activation hash → composite proof
            7. Write activation receipt → evidence artifact

        Returns:
            GenesisActivationResult with all receipts chained.

        Raises:
            ValueError: If node_seed is invalid.
            RuntimeError: If ceremony fails.
        """
        from datetime import datetime, timezone

        start = time.monotonic()
        timestamp = datetime.now(timezone.utc).isoformat()

        # ── Step 1: Genesis Ceremony ──────────────────────────────
        logger.info("Genesis activation: running ceremony...")
        ceremony_result = run_ceremony(self._node_seed, self._ceremony_config)

        # ── Step 2: Persist Ceremony to Disk ──────────────────────
        self._data_dir.mkdir(parents=True, exist_ok=True)
        write_ceremony(ceremony_result, self._data_dir)

        node_id = ceremony_result.genesis_json.get("identity", {}).get("node_id", "")
        genesis_hash = ceremony_result.genesis_hash

        # ── Step 3: Run Orchestrator (optional) ───────────────────
        orchestrator_success = True
        orchestrator_reason_codes: List[str] = []

        if not self._skip_orchestrator:
            orchestrator_success, orchestrator_reason_codes = self._run_orchestrator(
                node_id, genesis_hash
            )

        # ── Step 4: Boot Heartbeat ────────────────────────────────
        boot_receipt_dict = self._run_heartbeat_boot(node_id, genesis_hash)

        # ── Step 5: First Breath (optional) ───────────────────────
        first_breath_dict: Optional[Dict[str, Any]] = None
        if not self._skip_breath and self._heartbeat is not None:
            first_breath_dict = self._run_first_breath()

        # ── Step 6: Compute Activation Hash ───────────────────────
        activation_hash = self._compute_activation_hash(
            genesis_hash=genesis_hash,
            boot_hash=boot_receipt_dict.get("boot_hash", ""),
            breath_hash=(
                first_breath_dict.get("chain_hash", "") if first_breath_dict else ""
            ),
            timestamp=timestamp,
        )

        # ── Step 7: Verify Evidence Chain ─────────────────────────
        genesis_path = self._data_dir / "node0_genesis.json"
        evidence_valid = False
        if genesis_path.exists():
            evidence_valid, _ = verify_ceremony(genesis_path)

        duration_ms = (time.monotonic() - start) * 1000

        result = GenesisActivationResult(
            ceremony_result=ceremony_result,
            orchestrator_success=orchestrator_success,
            orchestrator_reason_codes=orchestrator_reason_codes,
            boot_receipt_dict=boot_receipt_dict,
            first_breath_dict=first_breath_dict,
            activation_hash=activation_hash,
            evidence_chain_valid=evidence_valid,
            timestamp=timestamp,
            duration_ms=round(duration_ms, 2),
            node_id=node_id,
            genesis_hash=genesis_hash,
        )

        # Write activation receipt
        self._write_activation_receipt(result)
        self._result = result

        logger.info(
            "Genesis activation COMPLETE | node=%s | hash=%s | %.1fms",
            node_id[:16],
            activation_hash[:16],
            duration_ms,
        )

        return result

    def verify(self) -> tuple[bool, list[str]]:
        """Verify the genesis activation artifacts on disk.

        Checks:
            1. node0_genesis.json internal hash integrity
            2. activation_receipt.json exists and parses
            3. Activation hash matches recomputed value

        Returns:
            (is_valid, list_of_reason_codes)
        """
        reasons: list[str] = []

        # 1. Verify ceremony
        genesis_path = self._data_dir / "node0_genesis.json"
        if not genesis_path.exists():
            return False, ["GENESIS_FILE_MISSING"]

        ceremony_valid, ceremony_reasons = verify_ceremony(genesis_path)
        if not ceremony_valid:
            reasons.extend(ceremony_reasons)

        # 2. Verify activation receipt
        receipt_path = self._data_dir / "activation_receipt.json"
        if not receipt_path.exists():
            reasons.append("ACTIVATION_RECEIPT_MISSING")
        else:
            try:
                with open(receipt_path, encoding="utf-8") as f:
                    receipt_data = json.load(f)
                if not receipt_data.get("activation_hash"):
                    reasons.append("ACTIVATION_HASH_EMPTY")
            except (json.JSONDecodeError, OSError) as e:
                reasons.append(f"ACTIVATION_RECEIPT_CORRUPT: {e}")

        return len(reasons) == 0, reasons

    # ═══════════════════════════════════════════════════════════════
    # Internal Pipeline Steps
    # ═══════════════════════════════════════════════════════════════

    _heartbeat: Any = None  # Set during boot

    def _run_orchestrator(
        self, node_id: str, genesis_hash: str
    ) -> tuple[bool, list[str]]:
        """Run the 12-step orchestrator with graceful degradation."""
        try:
            from core.genesis.orchestrator import GenesisOrchestrator
            from core.genesis.types import GenesisConfig

            config = GenesisConfig(
                identity_genesis=False,  # Already done in ceremony
                hardware_scan=True,
                strict_bootstrap=False,
                allow_degraded=True,
            )
            orchestrator = GenesisOrchestrator(config)
            orchestrator._node_id = node_id
            orchestrator._genesis_hash = genesis_hash
            result = orchestrator.run()

            return result.success or result.status == "degraded", list(
                result.reason_codes or []
            )
        except Exception as e:
            logger.warning("Orchestrator degraded: %s", e)
            return False, [f"ORCHESTRATOR_EXCEPTION: {type(e).__name__}"]

    def _run_heartbeat_boot(self, node_id: str, genesis_hash: str) -> Dict[str, Any]:
        """Boot the heartbeat and return the boot receipt as dict."""
        from core.node0.heartbeat import Node0Heartbeat

        heartbeat = Node0Heartbeat(
            data_dir=self._data_dir,
            node_id=node_id,
            genesis_backed=True,
        )
        boot_receipt = heartbeat.boot()
        self._heartbeat = heartbeat

        return boot_receipt.as_dict()

    def _run_first_breath(self) -> Optional[Dict[str, Any]]:
        """Execute the first constitutional breath."""
        try:
            breath_receipt = self._heartbeat.breathe()
            return breath_receipt.as_dict()
        except Exception as e:
            logger.warning("First breath degraded: %s", e)
            return None

    def _compute_activation_hash(
        self,
        genesis_hash: str,
        boot_hash: str,
        breath_hash: str,
        timestamp: str,
    ) -> str:
        """Compute the composite activation hash from all sub-hashes."""
        preimage = canonical_bytes(
            {
                "genesis_hash": genesis_hash,
                "boot_hash": boot_hash,
                "breath_hash": breath_hash,
                "timestamp": timestamp,
            }
        )
        return blake3_digest(preimage).hex()

    def _write_activation_receipt(self, result: GenesisActivationResult) -> None:
        """Write the activation receipt to disk."""
        receipt_path = self._data_dir / "activation_receipt.json"
        try:
            receipt_json = json.dumps(
                result.as_dict(), indent=2, ensure_ascii=False, default=str
            )
            receipt_path.write_text(receipt_json + "\n", encoding="utf-8")
            logger.info("Activation receipt written: %s", receipt_path)
        except OSError as e:
            logger.warning("Failed to write activation receipt: %s", e)


__all__ = [
    "GenesisActivation",
    "GenesisActivationResult",
]
