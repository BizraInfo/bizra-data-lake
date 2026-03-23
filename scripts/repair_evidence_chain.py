"""
Evidence Chain Repair - G6 Fork Remediation
============================================

Repairs the seq=14 fork in sovereign_state/evidence.jsonl caused by
a concurrent-write race (G6). The G6 fix (fcntl.LOCK_EX) prevents future
forks; this script heals the historical chain.

Strategy: Chain Rehash (not archive+genesis)
- All 26 receipts are preserved untouched (evidence is real)
- The duplicate seq=14 fork entry (second occurrence, line 15) is dropped
- Entries seq=15–26 retain their receipts but have prev_hash and entry_hash
  recomputed against the corrected chain tip
- Chain is then fully valid: verify_chain() returns True

Standing on Giants:
- Merkle (1979): Hash chains - entry_hash = blake3(seq + receipt + prev_hash)
- Lamport (1978): Linearizability - one canonical ordering of events
- BIZRA G6 fix: fcntl.LOCK_EX in evidence_ledger.py prevents recurrence

Usage:
    cd /mnt/c/BIZRA-DATA-LAKE  # or WSL path
    source .venv-linux/bin/activate
    python scripts/repair_evidence_chain.py [--dry-run]
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

EVIDENCE_PATH = REPO_ROOT / "sovereign_state" / "evidence.jsonl"
ARCHIVE_SUFFIX = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
ARCHIVE_PATH = (
    REPO_ROOT / "sovereign_state" / f"evidence_archive_{ARCHIVE_SUFFIX}.jsonl"
)

GENESIS_HASH = "0" * 64


def load_entries(path: Path) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARN: line {lineno} parse error: {e}")
    return entries


def compute_entry_hash_blake3(seq: int, receipt: dict, prev_hash: str) -> str:
    """BLAKE3 entry hash - current algorithm post SEC-001."""
    from core.proof_engine.canonical import hex_digest

    canonical = json.dumps(
        {"seq": seq, "receipt": receipt, "prev_hash": prev_hash},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hex_digest(canonical)


def compute_entry_hash_sha256(seq: int, receipt: dict, prev_hash: str) -> str:
    """SHA-256 entry hash - legacy algorithm pre SEC-001."""
    import hashlib

    canonical = json.dumps(
        {"seq": seq, "receipt": receipt, "prev_hash": prev_hash},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def hash_valid(seq: int, receipt: dict, prev_hash: str, stored_hash: str) -> bool:
    """Return True if stored_hash matches BLAKE3 or legacy SHA-256 for these inputs."""
    if stored_hash == compute_entry_hash_blake3(seq, receipt, prev_hash):
        return True
    if stored_hash == compute_entry_hash_sha256(seq, receipt, prev_hash):
        return True
    return False


def repair_chain(entries: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Remove duplicate seq entries (keep first occurrence) and recompute
    the Merkle linkage for all entries after the first fork point.

    verify_chain accepts both BLAKE3 and legacy SHA-256 hashes.  Entries
    whose stored entry_hash is valid under either algorithm and whose
    prev_hash matches the chain tip are kept unchanged.  Only entries
    with a broken prev_hash link (caused by the fork) are rechained.

    Returns (repaired_entries, log_messages).
    """
    log = []
    seen_seqs: dict[int, int] = {}  # seq -> index of first occurrence
    deduped: list[dict] = []

    for i, entry in enumerate(entries):
        seq = entry["seq"]
        if seq in seen_seqs:
            log.append(
                f"  DROP  line {i+1}: seq={seq} fork "
                f"(first seen at line {seen_seqs[seq]+1}, "
                f"entry_hash={entry['entry_hash'][:16]}...)"
            )
        else:
            seen_seqs[seq] = i
            deduped.append(entry)

    repaired: list[dict] = []
    prev_hash = GENESIS_HASH

    for entry in deduped:
        seq = entry["seq"]
        receipt = entry["receipt"]
        declared_prev = entry["prev_hash"]
        declared_hash = entry["entry_hash"]

        # An entry is correct when:
        #   1. Its prev_hash points to the current chain tip, AND
        #   2. Its entry_hash is valid under BLAKE3 or legacy SHA-256
        #      (verify_chain accepts both - see evidence_ledger.py:318-338)
        prev_ok = declared_prev == prev_hash
        hash_ok = hash_valid(seq, receipt, declared_prev, declared_hash)

        if prev_ok and hash_ok:
            # Entry is already correct - keep as-is (including SHA-256 hashes)
            repaired.append(entry)
            prev_hash = declared_hash
        else:
            # prev_hash is stale (points to the dropped fork entry) -
            # fix linkage and recompute entry_hash with BLAKE3.
            new_hash = compute_entry_hash_blake3(seq, receipt, prev_hash)
            new_entry = dict(entry)
            new_entry["prev_hash"] = prev_hash
            new_entry["entry_hash"] = new_hash
            log.append(
                f"  RECHAIN seq={seq}: "
                f"prev {declared_prev[:16]}...->{prev_hash[:16]}..., "
                f"hash {declared_hash[:16]}...->{new_hash[:16]}..."
            )
            repaired.append(new_entry)
            prev_hash = new_hash

    return repaired, log


def verify_repaired(entries: list[dict]) -> list[str]:
    """Walk the repaired chain and return any remaining errors.

    Mirrors verify_chain() in evidence_ledger.py: accepts BLAKE3 or SHA-256
    so pre-migration SHA-256 entries pass without needing rehashing.
    """
    errors = []
    prev_hash = GENESIS_HASH
    expected_seq = 1

    for i, entry in enumerate(entries):
        seq = entry["seq"]
        declared_prev = entry["prev_hash"]
        declared_hash = entry["entry_hash"]

        if seq != expected_seq:
            errors.append(f"  line {i+1}: seq={seq} expected={expected_seq}")
        if declared_prev != prev_hash:
            errors.append(
                f"  line {i+1} seq={seq}: prev_hash mismatch "
                f"got={declared_prev[:16]} expected={prev_hash[:16]}"
            )

        if not hash_valid(seq, entry["receipt"], declared_prev, declared_hash):
            b3 = compute_entry_hash_blake3(seq, entry["receipt"], declared_prev)
            errors.append(
                f"  line {i+1} seq={seq}: entry_hash invalid "
                f"stored={declared_hash[:16]} blake3={b3[:16]}"
            )

        prev_hash = declared_hash
        expected_seq += 1

    return errors


def write_chain(entries: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            line = json.dumps(entry, separators=(",", ":"), sort_keys=True)
            f.write(line + "\n")


def main() -> int:
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("BIZRA Evidence Chain Repair - G6 Fork Remediation")
    print("=" * 60)

    if not EVIDENCE_PATH.exists():
        print(f"ERROR: {EVIDENCE_PATH} not found")
        return 1

    # 1. Load
    print(f"\n[1] Loading {EVIDENCE_PATH}")
    entries = load_entries(EVIDENCE_PATH)
    print(f"    {len(entries)} entries loaded")

    # 2. Repair
    print("\n[2] Detecting forks and recomputing chain linkage...")
    repaired, repair_log = repair_chain(entries)
    for msg in repair_log:
        print(msg)
    print(f"    {len(entries)} -> {len(repaired)} entries after dedup")

    # 3. Verify
    print("\n[3] Verifying repaired chain...")
    errors = verify_repaired(repaired)
    if errors:
        print("  CHAIN STILL INVALID - errors:")
        for e in errors:
            print(e)
        return 1
    else:
        print(f"  CHAIN VALID - {len(repaired)} entries, no errors")

    if dry_run:
        print("\n[DRY RUN] No files written. Re-run without --dry-run to apply.")
        return 0

    # 4. Archive original
    print(f"\n[4] Archiving original -> {ARCHIVE_PATH.name}")
    shutil.copy2(EVIDENCE_PATH, ARCHIVE_PATH)
    print(f"    Archived {EVIDENCE_PATH.stat().st_size} bytes")

    # 5. Write repaired chain
    print(f"\n[5] Writing repaired chain -> {EVIDENCE_PATH.name}")
    write_chain(repaired, EVIDENCE_PATH)
    print(f"    Written {EVIDENCE_PATH.stat().st_size} bytes")

    # 6. Final summary
    print("\n" + "=" * 60)
    print("REPAIR COMPLETE")
    print(f"  Archive: {ARCHIVE_PATH}")
    print(f"  Repaired: {EVIDENCE_PATH}")
    print(f"  Entries: {len(repaired)}")
    print("  Chain: VALID")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
