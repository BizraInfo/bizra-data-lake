"""
BIZRA GENESIS SYNC (v2.0)
"Genesis is Scripture — Written Once, Never Overwritten"

This module handles the ONE-WAY, APPEND-ONLY propagation of Genesis
from the source of truth (TaskMaster) to the read-only mirror (Data Lake).

⚠️ RULES:
1. Genesis is WRITTEN ONCE
2. Genesis is NEVER OVERWRITTEN
3. Genesis is NEVER MERGED
4. Genesis is NEVER RECONCILED
5. Any mismatch = TAMPER ALERT, not overwrite

This is not configuration. This is scripture.
"""

import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS — Source of Truth and Mirrors
# ═══════════════════════════════════════════════════════════════════════════════

# The ONE TRUE SOURCE
SOURCE_GENESIS = Path(r"C:\BIZRA-Dual-Agentic-system--main\genesis.json")

# Read-only mirrors
DATA_LAKE_GENESIS = Path(r"C:\BIZRA-DATA-LAKE\04_GOLD\genesis.json")
WSL_GENESIS = Path(r"/mnt/c/BIZRA-DATA-LAKE/04_GOLD/genesis.json")  # Same file, WSL path

# Tamper log
TAMPER_LOG = Path(r"C:\BIZRA-DATA-LAKE\04_GOLD\genesis_tamper_log.jsonl")


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class GenesisTamperError(Exception):
    """
    Raised when Genesis integrity is compromised.
    This is a CRITICAL SECURITY EVENT.
    """
    pass


class GenesisAlreadyExistsError(Exception):
    """
    Raised when attempting to write Genesis to a location where it already exists.
    Genesis can only be written ONCE.
    """
    pass


class GenesisNotFoundError(Exception):
    """
    Raised when source Genesis does not exist.
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_genesis_hash(genesis_path: Path) -> str:
    """Compute SHA256 hash of Genesis file."""
    if not genesis_path.exists():
        return "FILE_NOT_FOUND"
    
    with open(genesis_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_genesis_integrity(source: Path, mirror: Path) -> bool:
    """
    Verify that mirror is an exact copy of source.
    Returns True if identical, False otherwise.
    """
    source_hash = compute_genesis_hash(source)
    mirror_hash = compute_genesis_hash(mirror)
    return source_hash == mirror_hash


def log_tamper_event(event_type: str, details: dict):
    """Log a tamper event to the tamper log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "details": details
    }
    
    TAMPER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TAMPER_LOG, 'a') as f:
        f.write(json.dumps(entry) + "\n")
    
    print(f"🚨 TAMPER EVENT LOGGED: {event_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# SYNC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def propagate_genesis_to_data_lake(force_overwrite: bool = False) -> bool:
    """
    Propagate Genesis from source to Data Lake (one-way, append-only).
    
    Rules:
    1. If Data Lake Genesis does NOT exist → COPY (first write)
    2. If Data Lake Genesis EXISTS and matches source → NO ACTION (already synced)
    3. If Data Lake Genesis EXISTS and DIFFERS from source → TAMPER ALERT
    
    Args:
        force_overwrite: If True, allows overwriting (DANGEROUS, for recovery only)
    
    Returns:
        True if sync successful, False otherwise
    
    Raises:
        GenesisAlreadyExistsError: If mirror exists and differs (tamper detected)
        GenesisNotFoundError: If source doesn't exist
    """
    
    # Check source exists
    if not SOURCE_GENESIS.exists():
        # Try Data Lake as source (in case TaskMaster doesn't have it yet)
        if DATA_LAKE_GENESIS.exists():
            print("ℹ️ Source not found, but Data Lake has Genesis. Using Data Lake as source.")
            return True
        raise GenesisNotFoundError(f"Source Genesis not found at {SOURCE_GENESIS}")
    
    # Case 1: Mirror doesn't exist → First write (ALLOWED)
    if not DATA_LAKE_GENESIS.exists():
        DATA_LAKE_GENESIS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_GENESIS, DATA_LAKE_GENESIS)
        print(f"✅ GENESIS PROPAGATED (First Write)")
        print(f"   Source: {SOURCE_GENESIS}")
        print(f"   Mirror: {DATA_LAKE_GENESIS}")
        print(f"   Hash: {compute_genesis_hash(DATA_LAKE_GENESIS)[:16]}...")
        return True
    
    # Case 2: Mirror exists and matches → Already synced
    if verify_genesis_integrity(SOURCE_GENESIS, DATA_LAKE_GENESIS):
        print(f"✅ GENESIS ALREADY SYNCED (No action needed)")
        print(f"   Hash: {compute_genesis_hash(DATA_LAKE_GENESIS)[:16]}...")
        return True
    
    # Case 3: Mirror exists and DIFFERS → TAMPER ALERT
    source_hash = compute_genesis_hash(SOURCE_GENESIS)
    mirror_hash = compute_genesis_hash(DATA_LAKE_GENESIS)
    
    log_tamper_event("GENESIS_MISMATCH", {
        "source_path": str(SOURCE_GENESIS),
        "source_hash": source_hash,
        "mirror_path": str(DATA_LAKE_GENESIS),
        "mirror_hash": mirror_hash,
        "action": "OVERWRITE" if force_overwrite else "BLOCKED"
    })
    
    if force_overwrite:
        # DANGEROUS: Recovery mode
        backup_path = DATA_LAKE_GENESIS.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(DATA_LAKE_GENESIS, backup_path)
        shutil.copy2(SOURCE_GENESIS, DATA_LAKE_GENESIS)
        print(f"⚠️ GENESIS OVERWRITTEN (Recovery Mode)")
        print(f"   Backup: {backup_path}")
        return True
    
    raise GenesisAlreadyExistsError(
        f"TAMPER DETECTED: Genesis already exists in Data Lake with different hash.\n"
        f"  Source Hash: {source_hash[:16]}...\n"
        f"  Mirror Hash: {mirror_hash[:16]}...\n"
        f"This is a SECURITY EVENT. Genesis is scripture — it cannot be overwritten.\n"
        f"If this is intentional (recovery), use force_overwrite=True."
    )


def verify_all_genesis_copies() -> dict:
    """
    Verify integrity of all Genesis copies across the system.
    
    Returns:
        Dict with verification status for each location
    """
    results = {
        "source": {
            "path": str(SOURCE_GENESIS),
            "exists": SOURCE_GENESIS.exists(),
            "hash": compute_genesis_hash(SOURCE_GENESIS) if SOURCE_GENESIS.exists() else None
        },
        "data_lake": {
            "path": str(DATA_LAKE_GENESIS),
            "exists": DATA_LAKE_GENESIS.exists(),
            "hash": compute_genesis_hash(DATA_LAKE_GENESIS) if DATA_LAKE_GENESIS.exists() else None
        }
    }
    
    # Check if all hashes match
    hashes = [r["hash"] for r in results.values() if r["hash"] and r["hash"] != "FILE_NOT_FOUND"]
    results["all_match"] = len(set(hashes)) <= 1 if hashes else False
    results["tamper_detected"] = len(set(hashes)) > 1
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python genesis_sync.py [propagate|verify|status]")
        print("")
        print("Commands:")
        print("  propagate  - Copy Genesis from source to Data Lake (one-way)")
        print("  verify     - Check integrity of all Genesis copies")
        print("  status     - Show current Genesis status")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "propagate":
        try:
            propagate_genesis_to_data_lake()
        except GenesisAlreadyExistsError as e:
            print(f"\n🚨 {e}")
            sys.exit(1)
        except GenesisNotFoundError as e:
            print(f"\n❌ {e}")
            sys.exit(1)
    
    elif command == "verify":
        results = verify_all_genesis_copies()
        print("\n📋 GENESIS INTEGRITY REPORT")
        print("=" * 50)
        for location, info in results.items():
            if isinstance(info, dict):
                status = "✅" if info.get("exists") else "❌"
                print(f"  {location}: {status}")
                if info.get("hash"):
                    print(f"    Hash: {info['hash'][:16]}...")
        print("=" * 50)
        if results.get("tamper_detected"):
            print("🚨 TAMPER DETECTED: Hashes do not match!")
            sys.exit(1)
        elif results.get("all_match"):
            print("✅ All Genesis copies are identical")
        else:
            print("⚠️ Some Genesis files are missing")
    
    elif command == "status":
        print("\n🌱 GENESIS STATUS")
        print("=" * 50)
        if DATA_LAKE_GENESIS.exists():
            with open(DATA_LAKE_GENESIS, 'r') as f:
                genesis = json.load(f)
            print(f"  Node ID: {genesis.get('node_id', 'N/A')}")
            print(f"  Author: {genesis.get('author', 'N/A')}")
            print(f"  Created: {genesis.get('created_at', 'N/A')}")
            hw = genesis.get('hardware_covenant', {}).get('tier_1_root', {})
            print(f"  Root Hash: {hw.get('combined_root_hash', 'N/A')[:16]}...")
        else:
            print("  Genesis not initialized in Data Lake")
        print("=" * 50)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
