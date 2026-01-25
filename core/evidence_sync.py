"""
BIZRA Evidence Sync Module
Bidirectional synchronization between PoI Ledger and Receipts.

Connects:
- Data Lake: /mnt/c/BIZRA-DATA-LAKE/04_GOLD/poi_ledger.jsonl
- Dual-Agentic: docs/evidence/receipts/

Evidence Types:
- PoI Attestations (Data Lake) ←→ Receipts (Dual-Agentic)
- Cryptographic integrity verification
- Append-only audit trail
"""

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Configuration
DATA_LAKE_PATH = os.getenv("DATA_LAKE_PATH", "/mnt/c/BIZRA-DATA-LAKE")
POI_LEDGER_PATH = Path(DATA_LAKE_PATH) / "04_GOLD" / "poi_ledger.jsonl"
RECEIPTS_PATH = Path(os.getenv("BIZRA_RECEIPT_PATH", "docs/evidence/receipts"))
SYNC_INTERVAL = int(os.getenv("EVIDENCE_SYNC_INTERVAL", "60"))


@dataclass
class PoIAttestation:
    """Proof-of-Impact attestation from Data Lake."""

    version: str
    chain_id: str
    contributor: str
    action: str
    resources: Dict[str, Any]
    benchmarks: Dict[str, Any]
    timestamp: str
    attestation_hash: str
    genesis_merkle_root: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoIAttestation":
        return cls(
            version=data.get("version", "poi-0.2"),
            chain_id=data.get("chain_id", "bizra-main-alpha"),
            contributor=data.get("contributor", "unknown"),
            action=data.get("action", ""),
            resources=data.get("resources", {}),
            benchmarks=data.get("benchmarks", {}),
            timestamp=data.get("timestamp", ""),
            attestation_hash=data.get("attestation_hash", ""),
            genesis_merkle_root=data.get("genesis_merkle_root"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "version": self.version,
            "chain_id": self.chain_id,
            "contributor": self.contributor,
            "action": self.action,
            "resources": self.resources,
            "benchmarks": self.benchmarks,
            "timestamp": self.timestamp,
            "attestation_hash": self.attestation_hash,
        }
        if self.genesis_merkle_root:
            result["genesis_merkle_root"] = self.genesis_merkle_root
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def verify_integrity(self) -> bool:
        """Verify the attestation hash matches content."""
        content = {
            "contributor": self.contributor,
            "action": self.action,
            "resources": self.resources,
            "benchmarks": self.benchmarks,
            "timestamp": self.timestamp,
        }
        expected_hash = hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()
        return self.attestation_hash == expected_hash


@dataclass
class Receipt:
    """Receipt from Dual-Agentic system."""

    receipt_id: str
    timestamp: str
    task_summary: str
    rejection_codes: List[str]
    escalation_level: str
    integrity_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Receipt":
        return cls(
            receipt_id=data.get("receipt_id", ""),
            timestamp=data.get("timestamp", ""),
            task_summary=data.get("task_summary", ""),
            rejection_codes=data.get("rejection_codes", []),
            escalation_level=data.get("escalation_level", "None"),
            integrity_hash=data.get("integrity_hash", ""),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "task_summary": self.task_summary,
            "rejection_codes": self.rejection_codes,
            "escalation_level": self.escalation_level,
            "integrity_hash": self.integrity_hash,
            "metadata": self.metadata,
        }

    def to_poi_attestation(self) -> PoIAttestation:
        """Convert receipt to PoI attestation format."""
        return PoIAttestation(
            version="poi-0.2",
            chain_id="bizra-main-alpha",
            contributor="dual-agentic-system",
            action=self.task_summary,
            resources={
                "source": "receipt",
                "receipt_id": self.receipt_id,
            },
            benchmarks={
                "success": len(self.rejection_codes) == 0,
                "escalation_level": self.escalation_level,
                "rejection_count": len(self.rejection_codes),
            },
            timestamp=self.timestamp,
            attestation_hash=self.integrity_hash,
            metadata={
                "rejection_codes": self.rejection_codes,
                "original_type": "receipt",
            },
        )

    def verify_integrity(self) -> bool:
        """Verify the integrity hash matches content."""
        content = {
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "task_summary": self.task_summary,
            "rejection_codes": self.rejection_codes,
            "escalation_level": self.escalation_level,
        }
        expected_hash = hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()
        return self.integrity_hash == expected_hash


@dataclass
class SyncResult:
    """Result of an evidence sync operation."""

    direction: str  # "lake_to_receipts" or "receipts_to_lake"
    items_synced: int
    items_skipped: int
    errors: List[str]
    timestamp: str
    duration_ms: float


class EvidenceSync:
    """
    Bidirectional evidence synchronization.

    Syncs:
    - PoI Ledger (Data Lake) ←→ Receipts (Dual-Agentic)

    Maintains:
    - Append-only semantics
    - Cryptographic integrity
    - Deduplication via fingerprints
    """

    def __init__(
        self,
        poi_ledger_path: Path = POI_LEDGER_PATH,
        receipts_path: Path = RECEIPTS_PATH,
    ):
        self.poi_ledger_path = poi_ledger_path
        self.receipts_path = receipts_path
        self._synced_hashes: Set[str] = set()

    def _load_synced_hashes(self) -> None:
        """Load previously synced hashes to avoid duplicates."""
        sync_state_path = self.receipts_path / ".sync_state.json"
        if sync_state_path.exists():
            with open(sync_state_path) as f:
                state = json.load(f)
                self._synced_hashes = set(state.get("synced_hashes", []))

    def _save_synced_hashes(self) -> None:
        """Save synced hashes state."""
        self.receipts_path.mkdir(parents=True, exist_ok=True)
        sync_state_path = self.receipts_path / ".sync_state.json"
        with open(sync_state_path, "w") as f:
            json.dump({
                "synced_hashes": list(self._synced_hashes),
                "last_sync": datetime.utcnow().isoformat(),
            }, f, indent=2)

    def read_poi_ledger(self) -> List[PoIAttestation]:
        """Read all attestations from PoI ledger."""
        attestations = []
        if not self.poi_ledger_path.exists():
            logger.warning("PoI ledger not found: %s", self.poi_ledger_path)
            return attestations

        with open(self.poi_ledger_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        attestations.append(PoIAttestation.from_dict(data))
                    except json.JSONDecodeError as e:
                        logger.warning("Invalid JSON in PoI ledger: %s", e)

        return attestations

    def read_receipts(self) -> List[Receipt]:
        """Read all receipts from receipts directory."""
        receipts = []
        if not self.receipts_path.exists():
            logger.warning("Receipts path not found: %s", self.receipts_path)
            return receipts

        for receipt_file in self.receipts_path.glob("*.jsonl"):
            with open(receipt_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            receipts.append(Receipt.from_dict(data))
                        except json.JSONDecodeError as e:
                            logger.warning("Invalid JSON in receipt file %s: %s", receipt_file, e)

        return receipts

    def write_to_poi_ledger(self, attestation: PoIAttestation) -> bool:
        """Append attestation to PoI ledger (append-only)."""
        try:
            self.poi_ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.poi_ledger_path, "a") as f:
                f.write(json.dumps(attestation.to_dict()) + "\n")
            return True
        except Exception as e:
            logger.error("Failed to write to PoI ledger: %s", e)
            return False

    def write_receipt(self, receipt: Receipt) -> bool:
        """Write receipt to receipts directory."""
        try:
            self.receipts_path.mkdir(parents=True, exist_ok=True)
            # Use date-based file naming
            date_str = receipt.timestamp[:10]  # YYYY-MM-DD
            receipt_file = self.receipts_path / f"receipts_{date_str}.jsonl"
            with open(receipt_file, "a") as f:
                f.write(json.dumps(receipt.to_dict()) + "\n")
            return True
        except Exception as e:
            logger.error("Failed to write receipt: %s", e)
            return False

    async def sync_lake_to_receipts(self) -> SyncResult:
        """
        Sync PoI attestations from Data Lake to local receipts.

        Converts PoI attestations to receipt format and writes to receipts directory.
        """
        start_time = datetime.utcnow()
        self._load_synced_hashes()

        items_synced = 0
        items_skipped = 0
        errors = []

        attestations = self.read_poi_ledger()
        for attestation in attestations:
            # Skip if already synced
            if attestation.attestation_hash in self._synced_hashes:
                items_skipped += 1
                continue

            # Convert to receipt
            receipt = Receipt(
                receipt_id=f"POI-{attestation.attestation_hash[:12]}",
                timestamp=attestation.timestamp,
                task_summary=f"[Data Lake] {attestation.action}",
                rejection_codes=[],
                escalation_level="None",
                integrity_hash=attestation.attestation_hash,
                metadata={
                    "source": "poi_ledger",
                    "contributor": attestation.contributor,
                    "benchmarks": attestation.benchmarks,
                },
            )

            # Write receipt
            if self.write_receipt(receipt):
                self._synced_hashes.add(attestation.attestation_hash)
                items_synced += 1
            else:
                errors.append(f"Failed to write receipt for {attestation.attestation_hash}")

        self._save_synced_hashes()

        duration = (datetime.utcnow() - start_time).total_seconds() * 1000

        return SyncResult(
            direction="lake_to_receipts",
            items_synced=items_synced,
            items_skipped=items_skipped,
            errors=errors,
            timestamp=datetime.utcnow().isoformat(),
            duration_ms=duration,
        )

    async def sync_receipts_to_lake(self) -> SyncResult:
        """
        Sync receipts to Data Lake PoI ledger.

        Converts receipts to PoI attestation format and appends to ledger.
        """
        start_time = datetime.utcnow()
        self._load_synced_hashes()

        items_synced = 0
        items_skipped = 0
        errors = []

        receipts = self.read_receipts()
        for receipt in receipts:
            # Skip if already synced
            if receipt.integrity_hash in self._synced_hashes:
                items_skipped += 1
                continue

            # Convert to PoI attestation
            attestation = receipt.to_poi_attestation()

            # Write to ledger
            if self.write_to_poi_ledger(attestation):
                self._synced_hashes.add(receipt.integrity_hash)
                items_synced += 1
            else:
                errors.append(f"Failed to write attestation for {receipt.receipt_id}")

        self._save_synced_hashes()

        duration = (datetime.utcnow() - start_time).total_seconds() * 1000

        return SyncResult(
            direction="receipts_to_lake",
            items_synced=items_synced,
            items_skipped=items_skipped,
            errors=errors,
            timestamp=datetime.utcnow().isoformat(),
            duration_ms=duration,
        )

    async def sync_bidirectional(self) -> Dict[str, SyncResult]:
        """
        Perform bidirectional sync.

        Returns:
            Dict with "lake_to_receipts" and "receipts_to_lake" results
        """
        lake_result = await self.sync_lake_to_receipts()
        receipts_result = await self.sync_receipts_to_lake()

        return {
            "lake_to_receipts": lake_result,
            "receipts_to_lake": receipts_result,
        }

    async def verify_integrity(self) -> Dict[str, Any]:
        """Verify integrity of all evidence."""
        poi_valid = 0
        poi_invalid = 0
        receipts_valid = 0
        receipts_invalid = 0

        # Verify PoI attestations
        for attestation in self.read_poi_ledger():
            if attestation.verify_integrity():
                poi_valid += 1
            else:
                poi_invalid += 1

        # Verify receipts
        for receipt in self.read_receipts():
            if receipt.verify_integrity():
                receipts_valid += 1
            else:
                receipts_invalid += 1

        return {
            "poi_ledger": {
                "valid": poi_valid,
                "invalid": poi_invalid,
                "total": poi_valid + poi_invalid,
            },
            "receipts": {
                "valid": receipts_valid,
                "invalid": receipts_invalid,
                "total": receipts_valid + receipts_invalid,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }


# Singleton instance
_evidence_sync: Optional[EvidenceSync] = None


def get_evidence_sync() -> EvidenceSync:
    """Get the singleton EvidenceSync instance."""
    global _evidence_sync
    if _evidence_sync is None:
        _evidence_sync = EvidenceSync()
    return _evidence_sync


async def sync_evidence() -> Dict[str, SyncResult]:
    """Convenience function for bidirectional sync."""
    sync = get_evidence_sync()
    return await sync.sync_bidirectional()


async def verify_evidence() -> Dict[str, Any]:
    """Convenience function for integrity verification."""
    sync = get_evidence_sync()
    return await sync.verify_integrity()


if __name__ == "__main__":
    # Test evidence sync
    async def test():
        sync = EvidenceSync()

        # Verify integrity
        print("Verifying evidence integrity...")
        integrity = await sync.verify_integrity()
        print(f"PoI Ledger: {integrity['poi_ledger']['valid']}/{integrity['poi_ledger']['total']} valid")
        print(f"Receipts: {integrity['receipts']['valid']}/{integrity['receipts']['total']} valid")

        # Perform bidirectional sync
        print("\nPerforming bidirectional sync...")
        results = await sync.sync_bidirectional()

        for direction, result in results.items():
            print(f"\n{direction}:")
            print(f"  Synced: {result.items_synced}")
            print(f"  Skipped: {result.items_skipped}")
            print(f"  Errors: {len(result.errors)}")
            print(f"  Duration: {result.duration_ms:.2f}ms")

    asyncio.run(test())
