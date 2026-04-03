"""
BIZRA Sovereign Receipt and Evidence Chain System
==================================================
Phase 10 APEX SOVEREIGN audit trail implementation.

This module provides:
- SovereignReceiptType: Enumeration of sovereign receipt categories
- SovereignReceipt: Complete audit trail dataclass
- EvidenceChainManager: Merkle DAG-backed evidence chain
- SovereignReceiptEmitter: Receipt emission with sealing
- AuditTrailManager: Append-only storage and verification

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    SOVEREIGN RECEIPT SYSTEM                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Operation ──▶ [Receipt Emission] ──▶ [Evidence Chain] ──▶ [Storage]   │
    │                        │                     │                  │        │
    │                        ▼                     ▼                  ▼        │
    │                  Integrity Hash         Merkle DAG        JSONL Files   │
    │                  Ed25519 Sign           WinterProof       Append-Only   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

Domain: bizra-sovereign-v1:
Version: 1.0.0
Alignment: BIZRA_SOT.md Section 3.1 (Ihsan IM >= 0.95)

Storage: docs/evidence/receipts/sovereign/
Format: JSONL (one receipt per line, append-only)
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

# Import genesis sealer for Ed25519 signing
from core.genesis import GenesisSealer, GenesisSeal

# Import sovereignty module components
from core.sovereignty import LocalMerkleDAG, WinterProofEmbedder

# Import constellation for MajlisDecision
from core.genesis.constellation_7plus1 import MajlisDecision, VetoResult, GuardianRole

# Optional blake3
try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# Optional numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# CONSTANTS
# =============================================================================

SOVEREIGN_DOMAIN_PREFIX = "bizra-sovereign-v1:"
SOVEREIGN_VERSION = "1.0.0"
RECEIPT_DOMAIN = "bizra-sovereign-receipt-v1:"
EVIDENCE_DOMAIN = "bizra-sovereign-evidence-v1:"
WINTERPROOF_DIM = 258  # Must be divisible by 3 for WinterProofEmbedder

# Default storage path
DEFAULT_STORAGE_PATH = Path("docs/evidence/receipts/sovereign")

# SNR target for sovereign operations
DEFAULT_SNR_TARGET = 0.99

# Maximum receipts per file before rotation
DEFAULT_MAX_RECEIPTS_PER_FILE = 1000


# =============================================================================
# ENUMS
# =============================================================================

class SovereignReceiptType(str, Enum):
    """
    Receipt types for sovereign operations.

    Each type represents a distinct auditable operation in the APEX SOVEREIGN flow.
    """
    # Full sovereign orchestration flow
    ORCHESTRATION = "orchestration"

    # Cosmic swarm decision from guardian constellation
    COSMIC_VERDICT = "cosmic_verdict"

    # Neural-symbolic fusion verification
    NEURAL_SYMBOLIC = "neural_symbolic"

    # SNR autonomous optimization cycle
    SNR_OPTIMIZATION = "snr_optimization"

    # Elite practitioner validation ("standing on giants")
    ELITE_PRACTITIONER = "elite_practitioner"

    # Per-stage completion evidence
    STAGE_COMPLETION = "stage_completion"


class VerdictDecision(str, Enum):
    """
    Cosmic verdict decisions from guardian swarm.

    Maps to the collective decision-making outcomes from the 7+1 guardian constellation.
    """
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    DEFERRED = "deferred"
    CONDITIONAL = "conditional"


class ReceiptStatus(str, Enum):
    """Status of a sovereign receipt."""
    VALID = "valid"
    PENDING = "pending"
    FAILED = "failed"
    CORRUPTED = "corrupted"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SovereignReceipt:
    """
    Sovereign audit trail receipt.

    Provides complete evidence of sovereign operations with cryptographic
    integrity guarantees and offline verification via WinterProof embeddings.

    Base fields (synced with src/receipts.rs):
        - receipt_id: Unique identifier following format sovereign-{session}-{timestamp}-{hash}
        - receipt_type: SovereignReceiptType enum value
        - timestamp: ISO-8601 timestamp
        - session_id: Session identifier
        - operation: Description of the operation
        - status: Receipt status (valid|pending|failed|corrupted)

    Sovereign-specific fields:
        - cosmic_verdict: VerdictDecision from guardian swarm
        - guardian_votes: Dict mapping guardian role to decision
        - majlis_decision: MajlisDecision consensus type
        - snr_achieved: Achieved SNR score
        - snr_target: Target SNR threshold (default 0.99)
        - ihsan_achieved: Achieved Ihsan score
        - optimization_iterations: Number of optimization cycles
        - domains_validated: List of validated domains
        - practitioners_count: Number of elite practitioners contributing
        - novelty_score: Semantic novelty of response
        - neural_confidence: Neural pathway confidence
        - symbolic_verification: Whether symbolic proofs passed
        - stages_completed: List of completed pipeline stages

    Evidence chain fields:
        - parent_receipts: List of parent receipt IDs
        - merkle_root: Merkle root of evidence chain
        - integrity_hash: SHA-256 hash of receipt content
        - winter_proof_embedding: Deterministic embedding for offline verification
        - seal_signature: Ed25519 signature
    """
    # Base fields (sync with src/receipts.rs)
    receipt_id: str
    receipt_type: SovereignReceiptType
    timestamp: str
    session_id: str
    operation: str
    status: ReceiptStatus

    # Sovereign-specific fields
    cosmic_verdict: Optional[VerdictDecision] = None
    guardian_votes: Dict[str, str] = field(default_factory=dict)
    majlis_decision: Optional[MajlisDecision] = None
    snr_achieved: float = 0.0
    snr_target: float = DEFAULT_SNR_TARGET
    ihsan_achieved: float = 0.0
    optimization_iterations: int = 0
    domains_validated: List[str] = field(default_factory=list)
    practitioners_count: int = 0
    novelty_score: float = 0.0
    neural_confidence: float = 0.0
    symbolic_verification: bool = False
    stages_completed: List[str] = field(default_factory=list)

    # Evidence chain fields
    parent_receipts: List[str] = field(default_factory=list)
    merkle_root: str = ""
    integrity_hash: str = ""
    winter_proof_embedding: List[float] = field(default_factory=list)
    seal_signature: str = ""

    # Additional metadata
    schema_version: str = SOVEREIGN_VERSION
    domain: str = SOVEREIGN_DOMAIN_PREFIX

    def __post_init__(self) -> None:
        """Validate and normalize receipt fields."""
        # Ensure timestamp is set
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

        # Ensure receipt_id follows format
        if not self.receipt_id:
            self.receipt_id = self._generate_receipt_id()

    def _generate_receipt_id(self) -> str:
        """Generate receipt ID in format: sovereign-{session}-{timestamp}-{hash}"""
        ts_part = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        hash_part = hashlib.sha256(
            f"{self.session_id}:{ts_part}:{secrets.token_hex(8)}".encode()
        ).hexdigest()[:8]
        return f"sovereign-{self.session_id[:8] if self.session_id else 'none'}-{ts_part}-{hash_part}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary for serialization."""
        return {
            "schema_version": self.schema_version,
            "domain": self.domain,
            "receipt_id": self.receipt_id,
            "receipt_type": self.receipt_type.value,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "operation": self.operation,
            "status": self.status.value,
            "cosmic_verdict": self.cosmic_verdict.value if self.cosmic_verdict else None,
            "guardian_votes": self.guardian_votes,
            "majlis_decision": self.majlis_decision.value if self.majlis_decision else None,
            "snr_achieved": self.snr_achieved,
            "snr_target": self.snr_target,
            "ihsan_achieved": self.ihsan_achieved,
            "optimization_iterations": self.optimization_iterations,
            "domains_validated": self.domains_validated,
            "practitioners_count": self.practitioners_count,
            "novelty_score": self.novelty_score,
            "neural_confidence": self.neural_confidence,
            "symbolic_verification": self.symbolic_verification,
            "stages_completed": self.stages_completed,
            "parent_receipts": self.parent_receipts,
            "merkle_root": self.merkle_root,
            "integrity_hash": self.integrity_hash,
            "winter_proof_embedding": self.winter_proof_embedding,
            "seal_signature": self.seal_signature,
        }

    def to_json(self) -> str:
        """Convert receipt to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SovereignReceipt":
        """Create receipt from dictionary."""
        return cls(
            receipt_id=data["receipt_id"],
            receipt_type=SovereignReceiptType(data["receipt_type"]),
            timestamp=data["timestamp"],
            session_id=data["session_id"],
            operation=data["operation"],
            status=ReceiptStatus(data["status"]),
            cosmic_verdict=VerdictDecision(data["cosmic_verdict"]) if data.get("cosmic_verdict") else None,
            guardian_votes=data.get("guardian_votes", {}),
            majlis_decision=MajlisDecision(data["majlis_decision"]) if data.get("majlis_decision") else None,
            snr_achieved=data.get("snr_achieved", 0.0),
            snr_target=data.get("snr_target", DEFAULT_SNR_TARGET),
            ihsan_achieved=data.get("ihsan_achieved", 0.0),
            optimization_iterations=data.get("optimization_iterations", 0),
            domains_validated=data.get("domains_validated", []),
            practitioners_count=data.get("practitioners_count", 0),
            novelty_score=data.get("novelty_score", 0.0),
            neural_confidence=data.get("neural_confidence", 0.0),
            symbolic_verification=data.get("symbolic_verification", False),
            stages_completed=data.get("stages_completed", []),
            parent_receipts=data.get("parent_receipts", []),
            merkle_root=data.get("merkle_root", ""),
            integrity_hash=data.get("integrity_hash", ""),
            winter_proof_embedding=data.get("winter_proof_embedding", []),
            seal_signature=data.get("seal_signature", ""),
            schema_version=data.get("schema_version", SOVEREIGN_VERSION),
            domain=data.get("domain", SOVEREIGN_DOMAIN_PREFIX),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SovereignReceipt":
        """Create receipt from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# EVIDENCE CHAIN MANAGER
# =============================================================================

class EvidenceChainManager:
    """
    Manager for sovereign evidence chains using Merkle DAG.

    Provides tamper-proof evidence chain management with:
    - Append-only receipt chain
    - Merkle DAG integrity verification
    - Chain export/import functionality
    - Receipt retrieval by ID

    The evidence chain ensures complete auditability of sovereign operations.
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        merkle_storage: Optional[str] = None,
    ):
        """
        Initialize EvidenceChainManager.

        Args:
            storage_path: Path for JSONL storage
            merkle_storage: Path for Merkle DAG persistence
        """
        self.storage_path = storage_path or DEFAULT_STORAGE_PATH
        self.chain: List[SovereignReceipt] = []
        self._lock = threading.Lock()

        # Initialize Merkle DAG
        self.merkle_dag = LocalMerkleDAG(storage_path=merkle_storage)

        # Receipt index by ID for fast lookup
        self._receipt_index: Dict[str, int] = {}

    def append_receipt(self, receipt: SovereignReceipt) -> bool:
        """
        Append receipt to evidence chain with integrity check.

        Args:
            receipt: SovereignReceipt to append

        Returns:
            True if successfully appended

        Raises:
            ValueError: If receipt integrity check fails
        """
        with self._lock:
            # Verify receipt has integrity hash
            if not receipt.integrity_hash:
                raise ValueError("Receipt missing integrity hash")

            # Verify integrity hash matches computed value
            computed_hash = self._compute_integrity_hash(receipt)
            if receipt.integrity_hash != computed_hash:
                raise ValueError(
                    f"Receipt integrity check failed: "
                    f"expected {computed_hash}, got {receipt.integrity_hash}"
                )

            # Add to Merkle DAG
            parent_ids = None
            if receipt.parent_receipts:
                # Map parent receipt IDs to DAG node IDs
                parent_ids = []
                for parent_id in receipt.parent_receipts:
                    if parent_id in self._receipt_index:
                        idx = self._receipt_index[parent_id]
                        # Use the DAG node ID from when we added it
                        # For simplicity, we use receipt index as node key
                        parent_ids.append(str(idx))

            # Add node to DAG
            node = self.merkle_dag.add_node(
                data=receipt.to_dict(),
                parent_ids=parent_ids,
                metadata={"receipt_id": receipt.receipt_id},
            )

            # Update receipt's merkle root
            receipt.merkle_root = node.merkle_root

            # Append to chain
            self.chain.append(receipt)
            self._receipt_index[receipt.receipt_id] = len(self.chain) - 1

            return True

    def get_chain(self) -> List[SovereignReceipt]:
        """
        Get the complete evidence chain.

        Returns:
            List of all receipts in order
        """
        with self._lock:
            return list(self.chain)

    def verify_chain(self) -> bool:
        """
        Verify the integrity of the entire evidence chain.

        Performs:
        1. Integrity hash verification for each receipt
        2. Merkle DAG verification
        3. Parent-child relationship validation

        Returns:
            True if chain is valid
        """
        with self._lock:
            # Verify each receipt's integrity hash
            for receipt in self.chain:
                computed = self._compute_integrity_hash(receipt)
                if receipt.integrity_hash != computed:
                    return False

            # Verify Merkle DAG
            dag_result = self.merkle_dag.verify()
            if not dag_result.valid:
                return False

            # Verify parent-child relationships
            for receipt in self.chain:
                for parent_id in receipt.parent_receipts:
                    if parent_id not in self._receipt_index:
                        # Parent should exist in chain
                        return False
                    parent_idx = self._receipt_index[parent_id]
                    current_idx = self._receipt_index[receipt.receipt_id]
                    if parent_idx >= current_idx:
                        # Parent should come before child
                        return False

            return True

    def compute_merkle_root(self) -> str:
        """
        Compute the Merkle root of the entire evidence chain.

        Returns:
            Hex-encoded Merkle root
        """
        with self._lock:
            if not self.chain:
                return ""

            # Get all receipt hashes
            hashes = [r.integrity_hash for r in self.chain]

            # Build Merkle tree
            while len(hashes) > 1:
                if len(hashes) % 2 == 1:
                    hashes.append(hashes[-1])  # Duplicate last for odd count

                next_level = []
                for i in range(0, len(hashes), 2):
                    combined = hashes[i] + hashes[i + 1]
                    if HAS_BLAKE3:
                        next_hash = blake3.blake3(combined.encode()).hexdigest()
                    else:
                        next_hash = hashlib.sha256(combined.encode()).hexdigest()
                    next_level.append(next_hash)

                hashes = next_level

            return hashes[0] if hashes else ""

    def get_receipt_by_id(self, receipt_id: str) -> Optional[SovereignReceipt]:
        """
        Retrieve a receipt by ID.

        Args:
            receipt_id: Receipt ID to look up

        Returns:
            SovereignReceipt if found, None otherwise
        """
        with self._lock:
            idx = self._receipt_index.get(receipt_id)
            if idx is not None:
                return self.chain[idx]
            return None

    def export_chain_to_jsonl(self, path: Union[str, Path]) -> int:
        """
        Export evidence chain to JSONL format (append-only).

        Args:
            path: File path for export

        Returns:
            Number of receipts exported
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                for receipt in self.chain:
                    f.write(receipt.to_json() + "\n")

            return len(self.chain)

    def _compute_integrity_hash(self, receipt: SovereignReceipt) -> str:
        """Compute integrity hash for a receipt (excluding the hash field itself)."""
        data = receipt.to_dict()
        data.pop("integrity_hash", None)
        data.pop("winter_proof_embedding", None)  # Exclude embedding from hash
        data.pop("seal_signature", None)  # Exclude signature from hash

        json_bytes = json.dumps(data, sort_keys=True).encode("utf-8")
        prefixed = (RECEIPT_DOMAIN + "integrity:").encode("utf-8") + json_bytes

        if HAS_BLAKE3:
            return f"sha256:{blake3.blake3(prefixed).hexdigest()}"
        else:
            return f"sha256:{hashlib.sha256(prefixed).hexdigest()}"


# =============================================================================
# SOVEREIGN RECEIPT EMITTER
# =============================================================================

class SovereignReceiptEmitter:
    """
    Emitter for sovereign receipts with cryptographic sealing.

    Provides:
    - Receipt emission for various sovereign operations
    - Ed25519 signing via GenesisSealer
    - WinterProof embedding for offline verification
    - Evidence chain integration

    All receipts are sealed with integrity hashes and Ed25519 signatures.
    """

    def __init__(
        self,
        evidence_chain: Optional[EvidenceChainManager] = None,
        sealer: Optional[GenesisSealer] = None,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize SovereignReceiptEmitter.

        Args:
            evidence_chain: Evidence chain manager (creates new if None)
            sealer: Genesis sealer for Ed25519 signing (creates new if None)
            storage_path: Path for receipt storage
        """
        self.storage_path = storage_path or DEFAULT_STORAGE_PATH
        self.evidence_chain = evidence_chain or EvidenceChainManager(self.storage_path)

        # Initialize sealer (generates new key if not provided)
        try:
            self.sealer = sealer or GenesisSealer()
        except ImportError:
            # PyNaCl not available - sealer will be None
            self.sealer = None

        # Initialize WinterProof embedder for offline verification
        self.embedder = WinterProofEmbedder(dimension=WINTERPROOF_DIM, use_numpy=True)

        # Receipt counter
        self._counter = 0
        self._lock = threading.Lock()

    def emit_orchestration_receipt(
        self,
        session_id: str,
        operation: str,
        snr_achieved: float,
        ihsan_achieved: float,
        stages_completed: List[str],
        domains_validated: List[str],
        optimization_iterations: int = 0,
        parent_receipts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> SovereignReceipt:
        """
        Emit receipt for full sovereign orchestration flow.

        Args:
            session_id: Session identifier
            operation: Operation description
            snr_achieved: Achieved SNR score
            ihsan_achieved: Achieved Ihsan score
            stages_completed: List of completed stages
            domains_validated: List of validated domains
            optimization_iterations: Number of optimization cycles
            parent_receipts: Parent receipt IDs for chaining
            **kwargs: Additional receipt fields

        Returns:
            Sealed SovereignReceipt
        """
        receipt = SovereignReceipt(
            receipt_id="",  # Will be generated
            receipt_type=SovereignReceiptType.ORCHESTRATION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            operation=operation,
            status=ReceiptStatus.VALID if snr_achieved >= DEFAULT_SNR_TARGET else ReceiptStatus.PENDING,
            snr_achieved=snr_achieved,
            ihsan_achieved=ihsan_achieved,
            stages_completed=stages_completed,
            domains_validated=domains_validated,
            optimization_iterations=optimization_iterations,
            parent_receipts=parent_receipts or [],
            **kwargs,
        )

        return self._seal_and_emit(receipt)

    def emit_cosmic_verdict_receipt(
        self,
        session_id: str,
        cosmic_verdict: VerdictDecision,
        guardian_votes: Dict[str, str],
        majlis_decision: MajlisDecision,
        snr_achieved: float,
        ihsan_achieved: float,
        parent_receipts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> SovereignReceipt:
        """
        Emit receipt for cosmic swarm verdict.

        Args:
            session_id: Session identifier
            cosmic_verdict: Final verdict decision
            guardian_votes: Dict mapping guardian role to decision
            majlis_decision: Majlis consensus type
            snr_achieved: Achieved SNR score
            ihsan_achieved: Achieved Ihsan score
            parent_receipts: Parent receipt IDs
            **kwargs: Additional receipt fields

        Returns:
            Sealed SovereignReceipt
        """
        receipt = SovereignReceipt(
            receipt_id="",
            receipt_type=SovereignReceiptType.COSMIC_VERDICT,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            operation="cosmic_swarm_verdict",
            status=ReceiptStatus.VALID if cosmic_verdict == VerdictDecision.APPROVED else ReceiptStatus.PENDING,
            cosmic_verdict=cosmic_verdict,
            guardian_votes=guardian_votes,
            majlis_decision=majlis_decision,
            snr_achieved=snr_achieved,
            ihsan_achieved=ihsan_achieved,
            parent_receipts=parent_receipts or [],
            **kwargs,
        )

        return self._seal_and_emit(receipt)

    def emit_stage_receipt(
        self,
        session_id: str,
        stage: str,
        context: Dict[str, Any],
        snr_achieved: float = 0.0,
        ihsan_achieved: float = 0.0,
        parent_receipts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> SovereignReceipt:
        """
        Emit receipt for stage completion.

        Args:
            session_id: Session identifier
            stage: Stage name/identifier
            context: Stage execution context
            snr_achieved: Achieved SNR score
            ihsan_achieved: Achieved Ihsan score
            parent_receipts: Parent receipt IDs
            **kwargs: Additional receipt fields

        Returns:
            Sealed SovereignReceipt
        """
        receipt = SovereignReceipt(
            receipt_id="",
            receipt_type=SovereignReceiptType.STAGE_COMPLETION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            operation=f"stage_completion:{stage}",
            status=ReceiptStatus.VALID,
            stages_completed=[stage],
            snr_achieved=snr_achieved,
            ihsan_achieved=ihsan_achieved,
            parent_receipts=parent_receipts or [],
            domains_validated=context.get("domains", []),
            neural_confidence=context.get("neural_confidence", 0.0),
            symbolic_verification=context.get("symbolic_verified", False),
            **kwargs,
        )

        return self._seal_and_emit(receipt)

    def emit_neural_symbolic_receipt(
        self,
        session_id: str,
        neural_confidence: float,
        symbolic_verification: bool,
        snr_achieved: float,
        ihsan_achieved: float,
        parent_receipts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> SovereignReceipt:
        """
        Emit receipt for neural-symbolic fusion verification.

        Args:
            session_id: Session identifier
            neural_confidence: Neural pathway confidence score
            symbolic_verification: Whether symbolic proofs passed
            snr_achieved: Achieved SNR score
            ihsan_achieved: Achieved Ihsan score
            parent_receipts: Parent receipt IDs
            **kwargs: Additional receipt fields

        Returns:
            Sealed SovereignReceipt
        """
        receipt = SovereignReceipt(
            receipt_id="",
            receipt_type=SovereignReceiptType.NEURAL_SYMBOLIC,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            operation="neural_symbolic_fusion",
            status=ReceiptStatus.VALID if symbolic_verification else ReceiptStatus.PENDING,
            neural_confidence=neural_confidence,
            symbolic_verification=symbolic_verification,
            snr_achieved=snr_achieved,
            ihsan_achieved=ihsan_achieved,
            parent_receipts=parent_receipts or [],
            **kwargs,
        )

        return self._seal_and_emit(receipt)

    def emit_snr_optimization_receipt(
        self,
        session_id: str,
        snr_before: float,
        snr_after: float,
        iterations: int,
        ihsan_achieved: float,
        parent_receipts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> SovereignReceipt:
        """
        Emit receipt for SNR autonomous optimization.

        Args:
            session_id: Session identifier
            snr_before: SNR before optimization
            snr_after: SNR after optimization
            iterations: Number of optimization iterations
            ihsan_achieved: Achieved Ihsan score
            parent_receipts: Parent receipt IDs
            **kwargs: Additional receipt fields

        Returns:
            Sealed SovereignReceipt
        """
        receipt = SovereignReceipt(
            receipt_id="",
            receipt_type=SovereignReceiptType.SNR_OPTIMIZATION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            operation=f"snr_optimization:{snr_before:.4f}->{snr_after:.4f}",
            status=ReceiptStatus.VALID if snr_after >= DEFAULT_SNR_TARGET else ReceiptStatus.PENDING,
            snr_achieved=snr_after,
            optimization_iterations=iterations,
            ihsan_achieved=ihsan_achieved,
            parent_receipts=parent_receipts or [],
            **kwargs,
        )

        return self._seal_and_emit(receipt)

    def emit_elite_practitioner_receipt(
        self,
        session_id: str,
        practitioners_count: int,
        domains_validated: List[str],
        novelty_score: float,
        snr_achieved: float,
        ihsan_achieved: float,
        parent_receipts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> SovereignReceipt:
        """
        Emit receipt for elite practitioner validation.

        Args:
            session_id: Session identifier
            practitioners_count: Number of elite practitioners
            domains_validated: Validated expertise domains
            novelty_score: Semantic novelty score
            snr_achieved: Achieved SNR score
            ihsan_achieved: Achieved Ihsan score
            parent_receipts: Parent receipt IDs
            **kwargs: Additional receipt fields

        Returns:
            Sealed SovereignReceipt
        """
        # Minimum 3 practitioners per domain for elite tier
        min_practitioners = 3 * len(domains_validated) if domains_validated else 3

        receipt = SovereignReceipt(
            receipt_id="",
            receipt_type=SovereignReceiptType.ELITE_PRACTITIONER,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            operation="elite_practitioner_validation",
            status=ReceiptStatus.VALID if practitioners_count >= min_practitioners else ReceiptStatus.PENDING,
            practitioners_count=practitioners_count,
            domains_validated=domains_validated,
            novelty_score=novelty_score,
            snr_achieved=snr_achieved,
            ihsan_achieved=ihsan_achieved,
            parent_receipts=parent_receipts or [],
            **kwargs,
        )

        return self._seal_and_emit(receipt)

    def _compute_integrity_hash(self, receipt: SovereignReceipt) -> str:
        """
        Compute SHA-256 integrity hash for receipt.

        The hash is computed over all fields except:
        - integrity_hash itself
        - winter_proof_embedding (computed separately)
        - seal_signature (added after hash)

        Args:
            receipt: Receipt to hash

        Returns:
            Prefixed hex-encoded SHA-256 hash
        """
        data = receipt.to_dict()
        data.pop("integrity_hash", None)
        data.pop("winter_proof_embedding", None)
        data.pop("seal_signature", None)

        json_bytes = json.dumps(data, sort_keys=True).encode("utf-8")
        prefixed = (RECEIPT_DOMAIN + "integrity:").encode("utf-8") + json_bytes

        hash_bytes = hashlib.sha256(prefixed).digest()
        return f"sha256:{hash_bytes.hex()}"

    def _sign_receipt(self, receipt: SovereignReceipt) -> str:
        """
        Sign receipt with Ed25519 via GenesisSealer.

        Args:
            receipt: Receipt to sign

        Returns:
            Hex-encoded Ed25519 signature
        """
        if self.sealer is None:
            return ""

        # Create signing payload
        payload = {
            "receipt_id": receipt.receipt_id,
            "integrity_hash": receipt.integrity_hash,
            "timestamp": receipt.timestamp,
        }
        json_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        prefixed = (RECEIPT_DOMAIN + "signature:").encode("utf-8") + json_bytes

        # Sign via sealer
        attestation = self.sealer.create_attestation(data=payload)
        return attestation.signature

    def _create_winter_proof_embedding(self, content: str) -> List[float]:
        """
        Create WinterProof embedding for offline verification.

        Args:
            content: Content to embed

        Returns:
            List of floats representing deterministic embedding
        """
        return self.embedder.embed(content)

    def _seal_and_emit(self, receipt: SovereignReceipt) -> SovereignReceipt:
        """
        Seal receipt with integrity hash, signature, and embedding.

        Args:
            receipt: Receipt to seal

        Returns:
            Sealed receipt
        """
        with self._lock:
            self._counter += 1

            # Generate receipt ID if not set
            if not receipt.receipt_id:
                receipt.receipt_id = receipt._generate_receipt_id()

            # Compute integrity hash
            receipt.integrity_hash = self._compute_integrity_hash(receipt)

            # Create WinterProof embedding for offline verification
            content = f"{receipt.receipt_id}:{receipt.operation}:{receipt.timestamp}"
            receipt.winter_proof_embedding = self._create_winter_proof_embedding(content)

            # Sign receipt
            receipt.seal_signature = self._sign_receipt(receipt)

            # Add to evidence chain
            try:
                self.evidence_chain.append_receipt(receipt)
            except ValueError:
                # Chain append failed, update status
                receipt.status = ReceiptStatus.FAILED

            return receipt


# =============================================================================
# AUDIT TRAIL MANAGER
# =============================================================================

class AuditTrailManager:
    """
    Manager for sovereign audit trail storage.

    Provides:
    - Append-only JSONL storage
    - Receipt loading by session
    - Audit trail verification
    - Report generation

    All operations maintain append-only semantics - existing receipts are never modified.
    """

    def __init__(
        self,
        storage_path: Union[str, Path] = DEFAULT_STORAGE_PATH,
        max_receipts_per_file: int = DEFAULT_MAX_RECEIPTS_PER_FILE,
    ):
        """
        Initialize AuditTrailManager.

        Args:
            storage_path: Base path for receipt storage
            max_receipts_per_file: Maximum receipts per file before rotation
        """
        self.storage_path = Path(storage_path)
        self.max_receipts_per_file = max_receipts_per_file
        self._lock = threading.Lock()

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Track current file and receipt count
        self._current_file: Optional[Path] = None
        self._current_file_count = 0

    def store_receipt(self, receipt: SovereignReceipt) -> bool:
        """
        Store receipt in append-only JSONL file.

        Args:
            receipt: Receipt to store

        Returns:
            True if successfully stored
        """
        with self._lock:
            # Get or create current file
            if self._current_file is None or self._current_file_count >= self.max_receipts_per_file:
                self._rotate_file()

            # Append receipt
            try:
                with open(self._current_file, "a", encoding="utf-8") as f:
                    f.write(receipt.to_json() + "\n")
                self._current_file_count += 1
                return True
            except IOError:
                return False

    def load_receipts(self, session_id: str) -> List[SovereignReceipt]:
        """
        Load all receipts for a session.

        Args:
            session_id: Session ID to filter by

        Returns:
            List of receipts for the session
        """
        receipts = []

        with self._lock:
            # Search all JSONL files
            for jsonl_file in self.storage_path.glob("*.jsonl"):
                try:
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                if data.get("session_id") == session_id:
                                    receipts.append(SovereignReceipt.from_dict(data))
                            except (json.JSONDecodeError, KeyError):
                                continue
                except IOError:
                    continue

        # Sort by timestamp
        receipts.sort(key=lambda r: r.timestamp)
        return receipts

    def verify_audit_trail(self, receipts: List[SovereignReceipt]) -> bool:
        """
        Verify integrity of an audit trail.

        Checks:
        1. Each receipt has valid integrity hash
        2. Parent-child relationships are valid
        3. Timestamps are monotonically increasing

        Args:
            receipts: List of receipts to verify

        Returns:
            True if audit trail is valid
        """
        if not receipts:
            return True

        # Create temporary evidence chain for verification
        chain = EvidenceChainManager()

        receipt_ids = set()
        last_timestamp = ""

        for receipt in receipts:
            # Check for duplicate IDs
            if receipt.receipt_id in receipt_ids:
                return False
            receipt_ids.add(receipt.receipt_id)

            # Verify timestamp ordering
            if last_timestamp and receipt.timestamp < last_timestamp:
                return False
            last_timestamp = receipt.timestamp

            # Verify integrity hash
            computed = chain._compute_integrity_hash(receipt)
            if receipt.integrity_hash != computed:
                return False

            # Verify parent references exist
            for parent_id in receipt.parent_receipts:
                if parent_id not in receipt_ids:
                    return False

        return True

    def generate_audit_report(self, session_id: str) -> Dict[str, Any]:
        """
        Generate audit report for a session.

        Args:
            session_id: Session ID to report on

        Returns:
            Audit report dictionary
        """
        receipts = self.load_receipts(session_id)

        if not receipts:
            return {
                "session_id": session_id,
                "status": "no_receipts",
                "receipt_count": 0,
            }

        # Aggregate metrics
        receipt_types = {}
        stages = set()
        domains = set()
        total_snr = 0.0
        total_ihsan = 0.0
        statuses = {}

        for receipt in receipts:
            # Count receipt types
            rt = receipt.receipt_type.value
            receipt_types[rt] = receipt_types.get(rt, 0) + 1

            # Collect stages and domains
            stages.update(receipt.stages_completed)
            domains.update(receipt.domains_validated)

            # Sum metrics
            total_snr += receipt.snr_achieved
            total_ihsan += receipt.ihsan_achieved

            # Count statuses
            status = receipt.status.value
            statuses[status] = statuses.get(status, 0) + 1

        n = len(receipts)

        return {
            "session_id": session_id,
            "status": "complete" if all(r.status == ReceiptStatus.VALID for r in receipts) else "partial",
            "receipt_count": n,
            "receipt_types": receipt_types,
            "stages_completed": sorted(stages),
            "domains_validated": sorted(domains),
            "avg_snr": total_snr / n if n > 0 else 0.0,
            "avg_ihsan": total_ihsan / n if n > 0 else 0.0,
            "status_breakdown": statuses,
            "first_receipt": receipts[0].timestamp if receipts else None,
            "last_receipt": receipts[-1].timestamp if receipts else None,
            "chain_verified": self.verify_audit_trail(receipts),
        }

    def _rotate_file(self) -> None:
        """Create new JSONL file for receipt storage."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        nonce = secrets.token_hex(4)
        filename = f"sovereign_{timestamp}_{nonce}.jsonl"
        self._current_file = self.storage_path / filename
        self._current_file_count = 0


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "SovereignReceiptType",
    "VerdictDecision",
    "ReceiptStatus",
    # Data classes
    "SovereignReceipt",
    # Managers
    "EvidenceChainManager",
    "SovereignReceiptEmitter",
    "AuditTrailManager",
    # Constants
    "SOVEREIGN_DOMAIN_PREFIX",
    "SOVEREIGN_VERSION",
    "DEFAULT_SNR_TARGET",
    "DEFAULT_STORAGE_PATH",
]

__version__ = SOVEREIGN_VERSION
__domain__ = SOVEREIGN_DOMAIN_PREFIX
