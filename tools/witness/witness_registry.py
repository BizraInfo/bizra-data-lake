#!/usr/bin/env python3
"""
BIZRA Witness Registry — the inbox of the Economic axis (Z1).

Gap it closes: the reproducer can ASK the world a question, but nothing exists
to FILE the answer. When an external witness returns an ATTESTATION.json, it
must land as a numbered, hash-chained, tamper-evident receipt.

W-R binding seal (integration, 2026-06-11): reproduce.sh now always emits
`witness_sha256` (sha256 of the witness JSON) alongside the possibly-BLAKE3
`witness_hash`. The binding check PREFERS witness_sha256 — always stdlib-
verifiable — so the first attestation lands `verified`, not `deferred_blake3`.

Mechanism:
  ingest <attestation.json> [--witness-json <file>]
      -> validate schema (fail-closed)
      -> binding check: if witness_sha256 present, recompute sha256(witness_json)
         and require MATCH (stdlib, always); else fall back to hash_algo
      -> reject duplicates (same witness_hash)
      -> append to .canon/WITNESS_LEDGER.jsonl, entry_hash chained to previous
  verify  -> recompute the full chain; ANY break => exit 1
  status  -> attestation count + table (the number DoD-3 graduates on)

Law honored: status is GENERATED from state. Stdlib only. Append-only.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

LEDGER_PATH = ".canon/WITNESS_LEDGER.jsonl"
LEDGER_SCHEMA = "bizra.witness_ledger.v0.1"
GENESIS_PREV = "0" * 64

REQUIRED_FIELDS = [
    "attestation_version", "claim", "repo", "commit", "witness_json",
    "witness_hash", "hash_algo", "result", "environment",
    "witness_identity", "witness_statement",
]
VALID_RESULTS = {"WITNESSED", "FAILED", "REFUSED"}


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_file_legacy(path: str, algo: str):
    # Legacy fallback when no witness_sha256 is present. stdlib cannot recompute
    # BLAKE3 (b3sum) -> return None and defer honestly rather than fake equality.
    if algo == "b3sum":
        return None
    return sha256_of_file(path)


def load_ledger(root: str):
    p = os.path.join(root, LEDGER_PATH)
    if not os.path.exists(p):
        return []
    entries = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def validate_attestation(att: dict):
    errs = []
    for field in REQUIRED_FIELDS:
        if field not in att:
            errs.append(f"missing field: {field}")
    if att.get("result") not in VALID_RESULTS:
        errs.append(f"result must be one of {sorted(VALID_RESULTS)}, "
                    f"got {att.get('result')!r}")
    commit = att.get("commit", "")
    if not (isinstance(commit, str) and len(commit) == 40
            and all(c in "0123456789abcdef" for c in commit)):
        errs.append("commit must be a 40-char lowercase hex SHA")
    wid = att.get("witness_identity", "")
    if not wid or wid.startswith("FILL_IN"):
        errs.append("witness_identity is unfilled — an attestation without "
                    "an identity (or explicit 'anonymous') is not filed")
    return errs


def cmd_ingest(root: str, att_path: str, witness_json_path):
    try:
        att = json.load(open(att_path, encoding="utf-8"))
    except Exception as e:
        print(f"[registry] FAIL: cannot parse attestation: {e}", file=sys.stderr)
        return 1

    errs = validate_attestation(att)
    if errs:
        print("[registry] FAIL: attestation rejected (fail-closed):", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 1

    # Binding check: PREFER the stdlib-verifiable witness_sha256 (W-R seal).
    binding = "not_supplied"
    if witness_json_path:
        if att.get("witness_sha256"):
            recomputed = sha256_of_file(witness_json_path)
            if recomputed == att["witness_sha256"]:
                binding = "verified"  # stdlib sha256, always checkable
            else:
                print("[registry] FAIL: witness_sha256 MISMATCH — attestation "
                      "binds a different witness JSON than supplied:", file=sys.stderr)
                print(f"  claimed (sha256) : {att['witness_sha256']}", file=sys.stderr)
                print(f"  computed (sha256): {recomputed}", file=sys.stderr)
                return 1
        else:
            recomputed = hash_file_legacy(witness_json_path, att["hash_algo"])
            if recomputed is None:
                binding = "deferred_blake3"  # honest: stdlib can't do BLAKE3
            elif recomputed == att["witness_hash"]:
                binding = "verified"
            else:
                print("[registry] FAIL: witness_hash MISMATCH:", file=sys.stderr)
                print(f"  claimed : {att['witness_hash']}", file=sys.stderr)
                print(f"  computed: {recomputed}", file=sys.stderr)
                return 1

    entries = load_ledger(root)
    if any(e["attestation"]["witness_hash"] == att["witness_hash"] for e in entries):
        print("[registry] FAIL: duplicate — this witness_hash is already filed.",
              file=sys.stderr)
        return 1

    prev_hash = entries[-1]["entry_hash"] if entries else GENESIS_PREV
    entry = {
        "schema": LEDGER_SCHEMA,
        "witness_number": len(entries) + 1,
        "filed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "binding_check": binding,
        "attestation": att,
        "prev_entry_hash": prev_hash,
    }
    entry["entry_hash"] = sha256_hex(
        (prev_hash + canonical({k: v for k, v in entry.items()
                                if k != "entry_hash"})).encode())

    out = os.path.join(root, LEDGER_PATH)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(canonical(entry) + "\n")

    print(f"[registry] FILED — External Witness #{entry['witness_number']}")
    print(f"  identity : {att['witness_identity']}")
    print(f"  result   : {att['result']}")
    print(f"  commit   : {att['commit'][:16]}…")
    print(f"  binding  : {binding}")
    print(f"  entry    : {entry['entry_hash'][:16]}… (chained)")
    return 0


def cmd_verify(root: str):
    entries = load_ledger(root)
    if not entries:
        print("[registry] ledger empty — nothing to verify (0 witnesses).")
        return 0
    prev = GENESIS_PREV
    for i, e in enumerate(entries, 1):
        if e.get("prev_entry_hash") != prev:
            print(f"[registry] FAIL: chain break at entry {i}", file=sys.stderr)
            return 1
        recomputed = sha256_hex(
            (prev + canonical({k: v for k, v in e.items()
                               if k != "entry_hash"})).encode())
        if recomputed != e.get("entry_hash"):
            print(f"[registry] FAIL: entry {i} TAMPERED", file=sys.stderr)
            print(f"  stored    : {e.get('entry_hash')}", file=sys.stderr)
            print(f"  recomputed: {recomputed}", file=sys.stderr)
            return 1
        prev = e["entry_hash"]
    print(f"[registry] PASS — chain intact, {len(entries)} attestation(s).")
    return 0


def cmd_status(root: str):
    entries = load_ledger(root)
    witnessed = [e for e in entries if e["attestation"]["result"] == "WITNESSED"]
    print(f"EXTERNAL ATTESTATIONS: {len(witnessed)} witnessed / {len(entries)} filed")
    for e in entries:
        a = e["attestation"]
        print(f"  #{e['witness_number']}  {a['result']:9s}  "
              f"{a['witness_identity'][:32]:32s}  commit {a['commit'][:12]}  "
              f"binding={e['binding_check']}")
    if not witnessed:
        print("  Economic axis: mechanics complete — awaiting first WITNESSED entry.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="BIZRA external witness registry")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_in = sub.add_parser("ingest")
    p_in.add_argument("attestation")
    p_in.add_argument("--witness-json", default=None)
    p_in.add_argument("--root", default=".")
    p_v = sub.add_parser("verify"); p_v.add_argument("--root", default=".")
    p_s = sub.add_parser("status"); p_s.add_argument("--root", default=".")
    a = ap.parse_args()
    if a.cmd == "ingest":
        sys.exit(cmd_ingest(a.root, a.attestation, a.witness_json))
    sys.exit({"verify": cmd_verify, "status": cmd_status}[a.cmd](a.root))


if __name__ == "__main__":
    main()
