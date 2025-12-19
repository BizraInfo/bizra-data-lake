#!/usr/bin/env python3
"""
BIZRA Ops Bootstrap (Local-First)
- Generates file inventory (json + csv)
- Detects duplicates (hash-based)
- Scans for secret-like patterns (baseline)
- Writes a timestamped evidence bundle

This is NOT a replacement for gitleaks/trufflehog.
It's a fast first line of defense and an evidence generator.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_EXCLUDES = {
    ".git", "node_modules", "target", "dist", "build", ".next", ".cache", "__pycache__",
    ".venv", "venv", ".mypy_cache", ".pytest_cache"
}

SECRET_PATTERNS = [
    ("AWS Access Key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Private Key Header", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("Generic API Key", re.compile(r"\b(api[_-]?key|secret|password)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]", re.IGNORECASE)),
    ("Bearer Token", re.compile(r"\bBearer\s+[A-Za-z0-9\-\._~\+\/]+=*\b")),
]

@dataclasses.dataclass
class FileRecord:
    path: str
    size: int
    mtime: float
    sha256: str

def sha256_file(p: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def should_exclude(path_parts: Tuple[str, ...], excludes: set[str]) -> bool:
    return any(part in excludes for part in path_parts)

def walk_files(root: Path, excludes: set[str]) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        # prune excluded dirs in-place for speed
        dirnames[:] = [d for d in dirnames if d not in excludes]
        for fn in filenames:
            p = dp / fn
            rel_parts = p.relative_to(root).parts
            if should_exclude(rel_parts, excludes):
                continue
            yield p

def scan_secrets(text: str) -> List[Dict[str, str]]:
    hits: List[Dict[str, str]] = []
    for name, pattern in SECRET_PATTERNS:
        for m in pattern.finditer(text):
            snippet = m.group(0)
            hits.append({"type": name, "match": snippet[:80]})
    return hits

def safe_read_text(p: Path, max_bytes: int = 2_000_000) -> Optional[str]:
    try:
        if p.stat().st_size > max_bytes:
            return None
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repo/project root to scan")
    ap.add_argument("--ops-dir", default="momo_ops", help="Output ops workspace folder (created under root)")
    ap.add_argument("--fail-on-secrets", action="store_true", help="Exit non-zero if secret patterns found")
    ap.add_argument("--exclude", action="append", default=[], help="Additional folder names to exclude")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        return 2

    excludes = set(DEFAULT_EXCLUDES) | set(args.exclude)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ops_dir = root / args.ops_dir
    run_dir = ops_dir / "evidence" / "runs" / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    command_record = {
        "timestamp": ts,
        "root": str(root),
        "excludes": sorted(excludes),
        "argv": sys.argv,
    }
    (run_dir / "command.json").write_text(json.dumps(command_record, indent=2), encoding="utf-8")

    records: List[FileRecord] = []
    duplicates: Dict[str, List[str]] = {}
    secret_report: List[Dict[str, object]] = []

    for p in walk_files(root, excludes):
        rel = p.relative_to(root).as_posix()
        try:
            st = p.stat()
        except FileNotFoundError:
            continue

        # skip ops-dir itself to avoid recursion
        if rel.startswith(f"{args.ops_dir}/"):
            continue

        digest = sha256_file(p)
        rec = FileRecord(path=rel, size=st.st_size, mtime=st.st_mtime, sha256=digest)
        records.append(rec)

        duplicates.setdefault(digest, []).append(rel)

        txt = safe_read_text(p)
        if txt is not None:
            hits = scan_secrets(txt)
            if hits:
                secret_report.append({"path": rel, "hits": hits})

    # write inventory
    inv_json = [dataclasses.asdict(r) for r in records]
    (run_dir / "inventory.json").write_text(json.dumps(inv_json, indent=2), encoding="utf-8")

    with (run_dir / "inventory.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "size", "mtime", "sha256"])
        w.writeheader()
        for r in records:
            w.writerow(dataclasses.asdict(r))

    # write duplicates (only hashes with >1 file)
    dup_out = {h: paths for h, paths in duplicates.items() if len(paths) > 1}
    (run_dir / "duplicates.json").write_text(json.dumps(dup_out, indent=2), encoding="utf-8")

    # write secrets report
    (run_dir / "secrets_report.json").write_text(json.dumps(secret_report, indent=2), encoding="utf-8")

    # summary
    total_files = len(records)
    total_bytes = sum(r.size for r in records)
    dup_groups = len(dup_out)
    secret_hits = len(secret_report)

    largest = sorted(records, key=lambda r: r.size, reverse=True)[:15]

    summary_lines = [
        f"# momo_ops run summary ({ts})",
        "",
        f"- Root: `{root}`",
        f"- Total files scanned: **{total_files}**",
        f"- Total size scanned: **{total_bytes/1_048_576:.2f} MiB**",
        f"- Duplicate groups (by sha256): **{dup_groups}**",
        f"- Files with secret-pattern hits: **{secret_hits}**",
        "",
        "## Largest files (top 15)",
    ]
    for r in largest:
        summary_lines.append(f"- {r.path} — {r.size/1_048_576:.2f} MiB")

    if secret_hits:
        summary_lines += [
            "",
            "## Secret-pattern hits (baseline scan)",
            "Review `secrets_report.json` and remove/rotate anything real.",
        ]

    (run_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[OK] Evidence written to: {run_dir}")
    if secret_hits:
        print(f"[WARN] Secret-like patterns detected in {secret_hits} file(s). See secrets_report.json")
        return 3 if args.fail_on_secrets else 0
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
