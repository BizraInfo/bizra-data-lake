"""
SEED Ledger — Persistent token balance across sessions.

Append-only JSONL ledger. Every SEED earned/spent is a line.
Balance = sum of all entries.

Standing on: Satoshi (append-only ledger), Adl (ExactAmount precision).
"""

import json
from pathlib import Path
from typing import Dict, List


def _ledger_path() -> Path:
    d = Path.home() / ".bizra"
    d.mkdir(parents=True, exist_ok=True)
    return d / "seed_ledger.jsonl"


def append(amount: int, reason: str, receipt_id: str = "") -> int:
    """Append a SEED transaction. Returns new balance."""
    import time

    entry = {
        "amount": amount,
        "reason": reason,
        "receipt_id": receipt_id,
        "ts": int(time.time()),
    }
    path = _ledger_path()
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return balance()


def balance() -> int:
    """Current SEED balance (sum of all entries)."""
    path = _ledger_path()
    if not path.exists():
        return 0
    total = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    total += json.loads(line)["amount"]
                except (json.JSONDecodeError, KeyError):
                    pass
    return total


def history(limit: int = 20) -> List[Dict]:
    """Recent transactions."""
    path = _ledger_path()
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries[-limit:]


def format_balance() -> str:
    """Format for display."""
    b = balance()
    h = history(5)
    lines = [f"🌱 {b} SEED"]
    if h:
        lines.append("Recent:")
        for e in h:
            sign = "+" if e["amount"] >= 0 else ""
            lines.append(f"  {sign}{e['amount']} — {e['reason']}")
    return "\n".join(lines)
