"""
Receipt Ledger — Durable chain persistence for canonical receipts.

Appends canonical receipts to a JSONL ledger and maintains chain
continuity across process restarts. The previous_receipt hash is
read from the last ledger entry on startup, not stored in memory.

RUNTIME_CUTOVER_03: chain survives winter.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

LEDGER_PATH = Path(
    os.environ.get(
        "BIZRA_RECEIPT_LEDGER",
        "/mnt/c/Users/BIZRA-OS/.bizra-kernel/ledger/canonical_receipts.jsonl",
    )
)

# Genesis seed — must match Rust and Python canonical adapter
GENESIS_SEED_HEX = "b12af37ed491c8562f0e8bd7439a5c11e72d60f81b37a4ce954f0d82763cb90a"


def _ensure_ledger() -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER_PATH.exists():
        LEDGER_PATH.touch()


def read_last_receipt_hash() -> bytes:
    """Read the last receipt_id from the ledger for chain linkage.

    Returns genesis seed if ledger is empty or unreadable.
    """
    _ensure_ledger()
    try:
        last_line = ""
        with open(LEDGER_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
        if last_line:
            entry = json.loads(last_line)
            rid = entry.get("receipt_id", "")
            if len(rid) == 64:
                return bytes.fromhex(rid)
    except (json.JSONDecodeError, ValueError, OSError):
        pass
    return bytes.fromhex(GENESIS_SEED_HEX)


def append_receipt(receipt_dict: dict) -> None:
    """Append a canonical receipt to the durable ledger.

    Uses file locking for concurrent access safety.
    """
    _ensure_ledger()
    line = json.dumps(receipt_dict, separators=(",", ":")) + "\n"
    with open(LEDGER_PATH, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def chain_length() -> int:
    """Count receipts in the ledger."""
    _ensure_ledger()
    try:
        with open(LEDGER_PATH) as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0
