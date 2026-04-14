"""MVDA Development Ledger — append-only, hash-chained JSONL."""

import fcntl
import json
import os
from pathlib import Path
from typing import Optional

from mvda.config import LEDGER_PATH
from mvda.receipt import MvdaReceipt


class MvdaLedger:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or LEDGER_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = self._read_last_hash()

    def _read_last_hash(self) -> str:
        if not self.path.exists():
            return "GENESIS"
        try:
            lines = self.path.read_text().strip().split("\n")
            if lines and lines[-1]:
                entry = json.loads(lines[-1])
                return entry.get("receipt_hash", "GENESIS")
        except (json.JSONDecodeError, IndexError):
            pass
        return "GENESIS"

    def append(self, receipt: MvdaReceipt) -> str:
        receipt.prev_hash = self._last_hash
        receipt.compute_hash()
        self._last_hash = receipt.receipt_hash
        with open(self.path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(receipt.to_dict(), sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return receipt.receipt_hash

    def verify_chain(self) -> tuple[bool, int]:
        """Verify hash chain integrity. Returns (valid, entry_count)."""
        if not self.path.exists():
            return True, 0
        lines = self.path.read_text().strip().split("\n")
        prev = "GENESIS"
        for i, line in enumerate(lines):
            entry = json.loads(line)
            if entry.get("prev_hash") != prev:
                return False, i
            prev = entry.get("receipt_hash", "")
        return True, len(lines)
