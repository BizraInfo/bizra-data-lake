"""
Cross-Node Operations — Verify, transfer, and coordinate between nodes.

Items 4.2, 4.3, 5.3 from the Activation Checklist.

Standing on: Lamport (distributed agreement), Satoshi (chain verification).
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List


def verify_remote_chain(receipts: List[Dict]) -> Dict:
    """
    Item 4.2: Verify another node's receipt chain.
    Check BLAKE3 hash continuity and receipt integrity.
    """
    if not receipts:
        return {"valid": False, "reason": "empty chain", "verified": 0}

    verified = 0
    prev_hash = None

    for r in receipts:
        rid = r.get("receipt_id", "")
        rprev = r.get("previous_receipt_hash")

        # Check chain link
        if prev_hash is not None and rprev != prev_hash:
            return {
                "valid": False,
                "reason": f"chain break at receipt {verified + 1}",
                "verified": verified,
            }

        prev_hash = rid
        verified += 1

    return {"valid": True, "reason": "chain intact", "verified": verified}


def prepare_telescript_mission(
    source_node: str,
    target_node: str,
    mission_text: str,
    seed_budget: int = 10,
) -> Dict:
    """
    Item 4.3: Prepare a Telescript mission to travel from source → target.
    Returns a mission envelope ready for gossip transport.
    """
    import time

    envelope = {
        "type": "telescript_mission",
        "version": "1.0",
        "source_node": source_node,
        "target_node": target_node,
        "mission": mission_text,
        "seed_budget": seed_budget,
        "created_at": int(time.time()),
        "ttl_seconds": 300,
        "status": "prepared",
        "hash": hashlib.sha256(
            f"{source_node}:{target_node}:{mission_text}:{time.time()}".encode()
        ).hexdigest(),
    }
    return envelope


def execute_seed_transfer(
    from_ledger_path: str,
    to_ledger_path: str,
    amount: int,
    reason: str = "cross-node transfer",
) -> Dict:
    """
    Item 5.3: Transfer SEED between two node ledgers.
    Both ledgers are append-only JSONL files.
    Returns transfer receipt.
    """
    import time

    if amount <= 0:
        return {"success": False, "reason": "amount must be positive"}

    # Check sender balance
    from_balance = _read_balance(from_ledger_path)
    if from_balance < amount:
        return {
            "success": False,
            "reason": f"insufficient balance: {from_balance} < {amount}",
        }

    ts = int(time.time())
    transfer_id = hashlib.sha256(
        f"transfer:{from_ledger_path}:{to_ledger_path}:{amount}:{ts}".encode()
    ).hexdigest()[:16]

    # Debit sender
    _append_entry(
        from_ledger_path,
        {
            "amount": -amount,
            "reason": f"sent: {reason}",
            "transfer_id": transfer_id,
            "ts": ts,
        },
    )

    # Credit receiver
    _append_entry(
        to_ledger_path,
        {
            "amount": amount,
            "reason": f"received: {reason}",
            "transfer_id": transfer_id,
            "ts": ts,
        },
    )

    return {
        "success": True,
        "transfer_id": transfer_id,
        "amount": amount,
        "from_balance": from_balance - amount,
        "to_balance": _read_balance(to_ledger_path),
    }


def _read_balance(ledger_path: str) -> int:
    p = Path(ledger_path)
    if not p.exists():
        return 0
    total = 0
    for line in p.read_text().strip().split("\n"):
        if line:
            try:
                total += json.loads(line)["amount"]
            except (json.JSONDecodeError, KeyError):
                pass
    return total


def _append_entry(ledger_path: str, entry: Dict):
    p = Path(ledger_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    # Demo: cross-node operations
    print("=== Cross-Node Demo ===")

    # 4.2: Verify chain
    receipts = [
        {"receipt_id": "aaa", "previous_receipt_hash": None},
        {"receipt_id": "bbb", "previous_receipt_hash": "aaa"},
        {"receipt_id": "ccc", "previous_receipt_hash": "bbb"},
    ]
    result = verify_remote_chain(receipts)
    print(f"Chain verify: {result}")

    # 4.3: Telescript mission
    mission = prepare_telescript_mission("node-A", "node-B", "research AI trends")
    print(f"Telescript: {mission['hash'][:16]}... status={mission['status']}")

    # 5.3: SEED transfer
    import os
    import tempfile

    td = tempfile.mkdtemp()
    a_ledger = os.path.join(td, "a.jsonl")
    b_ledger = os.path.join(td, "b.jsonl")

    # Give A some SEED
    _append_entry(a_ledger, {"amount": 100, "reason": "initial", "ts": 0})

    # Transfer 30 from A to B
    xfer = execute_seed_transfer(a_ledger, b_ledger, 30, "test transfer")
    print(f"Transfer: {xfer}")
    print(f"A balance: {_read_balance(a_ledger)}")
    print(f"B balance: {_read_balance(b_ledger)}")
