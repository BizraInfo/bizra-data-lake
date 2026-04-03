"""
bizra_kernel/genesis_sync.py - The One-Way Genesis Mirror
======================================================
Enforces the 'Scripture' rule: Genesis is written once, never overwritten.
Syncs the TaskMaster Source-of-Truth to the read-only Data Lake.
"""

import os
import shutil
import json
import logging
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("BIZRA_GENESIS_SYNC")

class GenesisTamperError(Exception):
    """Raised when Genesis Script attempts to overwrite existing scripture."""
    pass


@dataclass
class GenesisLocations:
    taskmaster: Path
    data_lake: Path
    local: Path
    backup_dir: Path
    local_archive: Path


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _archive_local(local_path: Path, archive_dir: Path, reason: str, source_hash: str, local_hash: str) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = archive_dir / f"genesis_block_{stamp}.json"
    receipt_path = archive_dir / f"tamper_receipt_{stamp}.json"

    shutil.copy2(local_path, archive_path)
    receipt = {
        "schema": "bizra_genesis_tamper_receipt_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "local_hash": local_hash,
        "source_hash": source_hash,
        "archived_to": str(archive_path),
    }
    _write_json(receipt_path, receipt)


def sync_genesis(*, locations: GenesisLocations) -> Dict[str, Any]:
    """
    Sync genesis from TaskMaster -> Data Lake + Local.

    Rules:
    - Data Lake is immutable: mismatch raises GenesisTamperError.
    - Local mismatch is archived (tamper receipt) before overwrite.
    """
    taskmaster = locations.taskmaster
    data_lake = locations.data_lake
    local = locations.local

    if not taskmaster.exists():
        raise FileNotFoundError(f"Source Genesis not found at {taskmaster}")

    source = _read_json(taskmaster)
    source_bytes = _canonical_json_bytes(source)
    source_hash = _hash_bytes(source_bytes)

    locations_synced: List[Dict[str, Any]] = []

    # Data Lake
    if data_lake.exists():
        target = _read_json(data_lake)
        target_hash = _hash_bytes(_canonical_json_bytes(target))
        if target_hash != source_hash:
            raise GenesisTamperError(
                "TAMPER ALERT: Genesis in Data Lake differs from TaskMaster."
            )
        locations_synced.append({"location": "data_lake", "status": "already_synced"})
    else:
        data_lake.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(taskmaster, data_lake)
        os.chmod(data_lake, 0o444)
        locations_synced.append({"location": "data_lake", "status": "synced"})

    # Local
    if local.exists():
        local_hash = _hash_bytes(_canonical_json_bytes(_read_json(local)))
        if local_hash != source_hash:
            _archive_local(
                local,
                locations.local_archive,
                reason="local_genesis_mismatch",
                source_hash=source_hash,
                local_hash=local_hash,
            )
            shutil.copy2(taskmaster, local)
            locations_synced.append({"location": "local", "status": "archived_then_synced"})
        else:
            locations_synced.append({"location": "local", "status": "already_synced"})
    else:
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(taskmaster, local)
        locations_synced.append({"location": "local", "status": "synced"})

    # Backup
    backup_dir = locations.backup_dir
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"genesis_backup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    _write_json(backup_path, source)
    locations_synced.append({"location": "backup", "status": "written", "path": str(backup_path)})

    return {
        "success": True,
        "source_hash": source_hash,
        "locations_synced": locations_synced,
    }


class GenesisMirror:
    """
    Manages the immutable replication of Genesis artifacts.
    """
    
    def __init__(self, source_path: str, mirror_root: str):
        self.source = Path(source_path)
        self.mirror_dir = Path(mirror_root) / "04_GOLD"
        self.mirror_file = self.mirror_dir / "genesis.json"

    def consecrate(self):
        """
        Performs the One-Way Sync.
        Raises GenesisTamperError if target exists and differs.
        """
        if not self.source.exists():
            raise FileNotFoundError(f"Source Genesis not found at {self.source}")

        # Ensure sanctuary exists
        self.mirror_dir.mkdir(parents=True, exist_ok=True)

        # CHECK 1: Preservation of Scripture
        if self.mirror_file.exists():
            # Calculate hashes to see if identical (idempotent is ok, overwrite is not)
            src_hash = self._hash_file(self.source)
            dst_hash = self._hash_file(self.mirror_file)

            if src_hash == dst_hash:
                print("[*] Genesis Mirror: Scripture verified intact (Matches Source).")
                return
            else:
                # CRITICAL: Attempt to change immutable history
                raise GenesisTamperError(
                    f"TAMPER ALERT: Genesis already exists in Data Lake and differs from Source.\n"
                    f"Existing Hash: {dst_hash}\n"
                    f"Attempted Hash: {src_hash}\n"
                    f"Refusing to overwrite immutable origin."
                )

        # ACTION: First Scribing
        print(f"[*] Genesis Mirror: Scribing One-Way Copy to {self.mirror_file}...")
        try:
            shutil.copy2(self.source, self.mirror_file)
            # Set read-only (chattr +i if privileged, otherwise chmod)
            os.chmod(self.mirror_file, 0o444) 
            print("[+] Genesis Scribed and Sealed (Read-Only).")
        except Exception as e:
            logger.error(f"Failed to scribe genesis: {e}")
            raise

    def _hash_file(self, path: Path) -> str:
        """SHA-256 of file content."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

if __name__ == "__main__":
    import hashlib # re-import for module run
    # Default Paths assuming execution from root
    # Note: Using absolute paths derived from BIZRA standard structure
    SOURCE = "/root/bizra-genesis/genesis/blocks/genesis_block_0.json"
    MIRROR = "/mnt/c/BIZRA-DATA-LAKE" # Windows mount path assumption for Data Lake
    
    # Fallback for Linux-only env
    if not os.path.exists(MIRROR):
        MIRROR = "/root/bizra-genesis/bizra_data_vault/roots/sovereign_data"

    try:
        mirror = GenesisMirror(SOURCE, MIRROR)
        mirror.consecrate()
    except Exception as e:
        print(f"\n[FATAL] GENESIS SYNC FAILED: {e}")
        exit(1)
