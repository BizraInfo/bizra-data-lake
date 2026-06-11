#!/usr/bin/env python3
"""
BIZRA Canon Seal — integrity gate for canonical documents.

Problem it closes (finding N3, 2026-06-11): TOPOLOGY_CANON.md — the document that
wins all contradictions — accumulated 114 lines of accidental paste and no
mechanism noticed for 26 days. A canonical file must fail CI within one run of
any unsealed modification.

Mechanism (mirrors Dema's root-canon pattern):
  seal   -> compute sha256 of each canonical file, write .canon/CANON_SEAL.json
  verify -> recompute and compare; ANY drift => exit 1 with a precise report

Legitimate canon updates re-run `seal` IN THE SAME COMMIT as the edit, so the
manifest and the file move together. An edit without a re-seal — accidental
paste, hostile change, tool mishap — turns CI red immediately.

Stdlib only. No third-party dependencies, by constitutional preference.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

MANIFEST_PATH = ".canon/CANON_SEAL.json"
SEAL_SCHEMA = "bizra.canon_seal.v0.1"

# The canonical set. Extend ONLY via a consented commit that also re-seals.
DEFAULT_CANON_FILES = [
    "TOPOLOGY_CANON.md",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(root: str):
    p = os.path.join(root, MANIFEST_PATH)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_seal(root: str, files):
    entries = {}
    for rel in files:
        full = os.path.join(root, rel)
        if not os.path.exists(full):
            print(f"[canon-seal] FATAL: canonical file missing: {rel}", file=sys.stderr)
            return 2
        entries[rel] = {
            "sha256": sha256_file(full),
            "bytes": os.path.getsize(full),
        }
    manifest = {
        "schema": SEAL_SCHEMA,
        "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": (
            "Canonical files listed here must match these hashes byte-for-byte. "
            "Legitimate updates re-run `canon_seal.py seal` in the SAME commit as "
            "the edit. verify failing means unsealed canon drift: investigate "
            "before anything else."
        ),
        "files": entries,
    }
    out = os.path.join(root, MANIFEST_PATH)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[canon-seal] SEALED {len(entries)} file(s) -> {MANIFEST_PATH}")
    for rel, e in entries.items():
        print(f"  {e['sha256'][:16]}…  {rel} ({e['bytes']} bytes)")
    return 0


def cmd_verify(root: str):
    manifest = load_manifest(root)
    if manifest is None:
        print(f"[canon-seal] FAIL: manifest absent at {MANIFEST_PATH}. "
              f"Run `seal` (consented) to establish the canon baseline.",
              file=sys.stderr)
        return 1
    if manifest.get("schema") != SEAL_SCHEMA:
        print(f"[canon-seal] FAIL: unknown manifest schema "
              f"{manifest.get('schema')!r}", file=sys.stderr)
        return 1
    failures = []
    for rel, expected in sorted(manifest.get("files", {}).items()):
        full = os.path.join(root, rel)
        if not os.path.exists(full):
            failures.append((rel, "MISSING", expected["sha256"], None))
            continue
        actual = sha256_file(full)
        if actual != expected["sha256"]:
            failures.append((rel, "DRIFT", expected["sha256"], actual))
    if failures:
        print("[canon-seal] FAIL — unsealed canon drift detected:", file=sys.stderr)
        for rel, kind, exp, act in failures:
            print(f"  {kind}: {rel}", file=sys.stderr)
            print(f"    sealed : {exp}", file=sys.stderr)
            print(f"    actual : {act or '(file absent)'}", file=sys.stderr)
        print("[canon-seal] If this change is legitimate: re-run `seal` in the "
              "same commit. If not: the canon has been modified without consent.",
              file=sys.stderr)
        return 1
    n = len(manifest.get("files", {}))
    print(f"[canon-seal] PASS — {n} canonical file(s) match the seal "
          f"(sealed {manifest.get('sealed_at_utc')}).")
    return 0


def main():
    ap = argparse.ArgumentParser(description="BIZRA canon integrity seal")
    ap.add_argument("command", choices=["seal", "verify"])
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    ap.add_argument("--file", action="append", dest="files",
                    help="canonical file (repeatable; default: built-in set). "
                         "Used by `seal` only.")
    args = ap.parse_args()
    files = args.files or DEFAULT_CANON_FILES
    if args.command == "seal":
        sys.exit(cmd_seal(args.root, files))
    sys.exit(cmd_verify(args.root))


if __name__ == "__main__":
    main()
