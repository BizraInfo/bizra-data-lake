#!/usr/bin/env python3
"""BIZRA Migration Dedup Scanner.

Performs SHA-256 hash scanning across all BIZRA directories on C:\\ to identify
duplicate files before migration to B: drive.

Usage:
    python scripts/migration/dedup_scanner.py
    python scripts/migration/dedup_scanner.py --resume
    python scripts/migration/dedup_scanner.py --output-dir /mnt/b/BIZRA/06_INDEX
    python scripts/migration/dedup_scanner.py --workers 8 --min-size 4096
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SCAN_DIRS: list[str] = [
    "/mnt/c/BIZRA-DATA-LAKE",
    "/mnt/c/BIZRA-Dual-Agentic-system--main",
    "/mnt/c/BIZRA-NODE0",
    "/mnt/c/BIZRA-PROJECTS",
    "/mnt/c/BIZRA-TaskMaster",
    "/mnt/c/bizra-genesis-node",
    "/mnt/c/bizra-genesis-node-repaired",
    "/mnt/c/bizra-genesis-node-fresh",
    "/mnt/c/BIZRA-GENESIS-CLEAN",
    "/mnt/c/bizra-voice",
    "/mnt/c/award-winner-design",
    "/mnt/c/HERMES project",
    "/mnt/c/momo phone",
    "/mnt/c/BIZRA-NODE0-BACKUP",
]

SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        "target",
        "node_modules",
        ".git",
        "__pycache__",
    }
)

SKIP_DIR_PREFIXES: tuple[str, ...] = (".venv",)

HASH_CHUNK_SIZE: int = 65536  # 64 KB

ONE_KB: int = 1024
TEN_GB: int = 10 * 1024 * 1024 * 1024

LOG_DIR = Path("/mnt/c/BIZRA-DATA-LAKE/logs/migration")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging() -> logging.Logger:
    """Configure file + stderr logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = (
        LOG_DIR
        / f"dedup_scan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger("dedup_scanner")
    logger.setLevel(logging.DEBUG)

    # File handler -- verbose
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    logger.addHandler(fh)

    # Stderr handler -- warnings only (keep stdout clean for progress)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(sh)

    logger.info("Logging to %s", log_file)
    return logger


logger = _setup_logging()

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _should_skip_dir(dir_name: str) -> bool:
    """Return True if this directory name should be pruned."""
    if dir_name in SKIP_DIR_NAMES:
        return True
    for prefix in SKIP_DIR_PREFIXES:
        if dir_name.startswith(prefix):
            return True
    return False


def discover_files(
    scan_dirs: list[str],
    min_size: int,
    max_size: int,
) -> list[str]:
    """Walk scan directories and return list of file paths within size bounds.

    Uses os.scandir for speed. Prunes skip directories in-place.
    """
    files: list[str] = []
    visited_inodes: set[tuple[int, int]] = set()  # (dev, inode) to skip hardlinks

    for root_dir in scan_dirs:
        root_path = Path(root_dir)
        if not root_path.is_dir():
            logger.warning("Scan directory does not exist, skipping: %s", root_dir)
            continue

        logger.info("Discovering files in: %s", root_dir)

        for dirpath, dirnames, filenames in os.walk(
            root_dir, topdown=True, followlinks=False
        ):
            # Prune skip directories in-place (modifying dirnames)
            dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]

            for fname in filenames:
                try:
                    fpath = os.path.join(dirpath, fname)
                    stat = os.lstat(fpath)

                    # Skip symlinks
                    if os.path.islink(fpath):
                        continue

                    # Skip by size
                    if stat.st_size < min_size or stat.st_size > max_size:
                        continue

                    # Skip already-visited inodes (hardlinks)
                    inode_key = (stat.st_dev, stat.st_ino)
                    if inode_key in visited_inodes:
                        continue
                    visited_inodes.add(inode_key)

                    files.append(fpath)

                except PermissionError:
                    logger.warning(
                        "Permission denied during discovery: %s",
                        os.path.join(dirpath, fname),
                    )
                except OSError as exc:
                    logger.warning(
                        "OS error during discovery: %s — %s",
                        os.path.join(dirpath, fname),
                        exc,
                    )

    logger.info("Discovered %d files for hashing", len(files))
    return files


# ---------------------------------------------------------------------------
# Hashing (runs in worker processes)
# ---------------------------------------------------------------------------


def _hash_file(file_path: str) -> Optional[tuple[str, str, int, float]]:
    """Hash a single file. Returns (path, sha256, size, mtime) or None on error.

    This function runs in worker processes via multiprocessing.Pool.
    Errors are caught and logged to stderr (logger is per-process).
    """
    try:
        h = hashlib.sha256()
        size = 0
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(HASH_CHUNK_SIZE)
                if not chunk:
                    break
                h.update(chunk)
                size += len(chunk)

        stat = os.stat(file_path)
        mtime = stat.st_mtime
        return (file_path, h.hexdigest(), size, mtime)

    except PermissionError:
        print(f"WARNING: Permission denied: {file_path}", file=sys.stderr)
        return None
    except OSError as exc:
        print(f"WARNING: OS error hashing {file_path}: {exc}", file=sys.stderr)
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: Unexpected error hashing {file_path}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Source directory resolution
# ---------------------------------------------------------------------------


def _resolve_source_dir(file_path: str, scan_dirs: list[str]) -> str:
    """Determine which scan directory a file belongs to."""
    for d in sorted(scan_dirs, key=len, reverse=True):
        if file_path.startswith(d + "/") or file_path.startswith(d + os.sep):
            return d
    return os.path.dirname(file_path)


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------


def _human_size(nbytes: int) -> str:
    """Convert bytes to human-readable string."""
    if nbytes < 1024:
        return f"{nbytes} B"
    for unit in ("KB", "MB", "GB", "TB"):
        nbytes /= 1024.0
        if nbytes < 1024.0:
            return f"{nbytes:.1f} {unit}"
    return f"{nbytes:.1f} PB"


def _iso_mtime(mtime: float) -> str:
    """Convert mtime float to ISO 8601 string."""
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def _load_resume_set(manifest_path: Path) -> set[str]:
    """Load already-scanned paths from an existing manifest.jsonl."""
    scanned: set[str] = set()
    if not manifest_path.exists():
        return scanned

    logger.info("Loading resume data from %s", manifest_path)
    count = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                scanned.add(record["path"])
                count += 1
            except (json.JSONDecodeError, KeyError):
                continue

    logger.info("Loaded %d previously scanned files for resume", count)
    return scanned


def _load_resume_records(manifest_path: Path) -> list[dict]:
    """Load full records from existing manifest for duplicate analysis."""
    records: list[dict] = []
    if not manifest_path.exists():
        return records

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Simple stdout progress tracker. No tqdm dependency."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.scanned = 0
        self.duplicates = 0
        self.wasted_bytes = 0
        self._last_print = 0
        self._start_time = time.monotonic()

    def update(self, scanned: int, duplicates: int, wasted: int) -> None:
        self.scanned = scanned
        self.duplicates = duplicates
        self.wasted_bytes = wasted
        if self.scanned - self._last_print >= 1000 or self.scanned == self.total:
            self._print()
            self._last_print = self.scanned

    def _print(self) -> None:
        pct = (self.scanned / self.total * 100) if self.total > 0 else 0.0
        elapsed = time.monotonic() - self._start_time
        rate = self.scanned / elapsed if elapsed > 0 else 0
        sys.stdout.write(
            f"\rScanned {self.scanned}/{self.total} files ({pct:.1f}%) "
            f"| {self.duplicates} duplicates found "
            f"| {_human_size(self.wasted_bytes)} wasted "
            f"| {rate:.0f} files/sec"
        )
        sys.stdout.flush()

    def finish(self) -> None:
        if self._last_print != self.scanned:
            self._print()
        sys.stdout.write("\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Core scan orchestration
# ---------------------------------------------------------------------------


def run_scan(args: argparse.Namespace) -> None:
    """Execute the full dedup scan pipeline."""
    scan_dirs = args.dirs if args.dirs else DEFAULT_SCAN_DIRS
    workers = args.workers
    min_size = args.min_size
    max_size = args.max_size
    output_dir = Path(args.output_dir)
    resume = args.resume

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    duplicates_path = output_dir / "duplicates.jsonl"
    summary_path = output_dir / "scan_summary.json"

    print("BIZRA Dedup Scanner")
    print(f"{'=' * 60}")
    print(f"Scan directories:  {len(scan_dirs)}")
    print(f"Workers:           {workers}")
    print(f"Min file size:     {_human_size(min_size)}")
    print(f"Max file size:     {_human_size(max_size)}")
    print(f"Output directory:  {output_dir}")
    print(f"Resume mode:       {'ON' if resume else 'OFF'}")
    print(f"{'=' * 60}")
    print()

    # Phase 1: Discover files
    print("[1/4] Discovering files...")
    t0 = time.monotonic()
    all_files = discover_files(scan_dirs, min_size, max_size)
    t_discover = time.monotonic() - t0
    print(f"      Found {len(all_files):,} files in {t_discover:.1f}s")

    # Phase 2: Resume filtering
    resume_records: list[dict] = []
    if resume:
        already_scanned = _load_resume_set(manifest_path)
        if already_scanned:
            resume_records = _load_resume_records(manifest_path)
            original_count = len(all_files)
            all_files = [f for f in all_files if f not in already_scanned]
            print(
                f"      Resume: skipping {original_count - len(all_files):,} already-scanned files"
            )
            print(f"      Remaining: {len(all_files):,} files to hash")

    total_to_hash = len(all_files)

    if total_to_hash == 0 and not resume_records:
        print("\nNo files to scan. Exiting.")
        return

    # Phase 3: Hash files in parallel
    print(f"\n[2/4] Hashing {total_to_hash:,} files with {workers} workers...")
    progress = ProgressTracker(total_to_hash)

    # Track duplicates incrementally
    hash_to_files: dict[str, list[dict]] = defaultdict(list)

    # Seed from resume records
    for rec in resume_records:
        hash_to_files[rec["sha256"]].append(rec)

    new_records: list[dict] = []
    dup_count = 0
    wasted_bytes = 0

    # Compute initial duplicate state from resumed data
    for sha, group in hash_to_files.items():
        if len(group) > 1:
            dup_count += len(group)
            wasted_bytes += (len(group) - 1) * group[0]["size"]

    t0 = time.monotonic()

    if total_to_hash > 0:
        with Pool(processes=workers) as pool:
            for result in pool.imap_unordered(_hash_file, all_files, chunksize=64):
                if result is None:
                    progress.update(progress.scanned + 1, dup_count, wasted_bytes)
                    continue

                fpath, sha256, size, mtime = result
                source_dir = _resolve_source_dir(fpath, scan_dirs)

                record = {
                    "path": fpath,
                    "sha256": sha256,
                    "size": size,
                    "mtime": _iso_mtime(mtime),
                    "source_dir": source_dir,
                }
                new_records.append(record)

                # Update duplicate tracking
                group = hash_to_files[sha256]
                group.append(record)
                if len(group) == 2:
                    # First duplicate pair found -- both files count
                    dup_count += 2
                    wasted_bytes += size
                elif len(group) > 2:
                    # Additional duplicate
                    dup_count += 1
                    wasted_bytes += size

                progress.update(progress.scanned + 1, dup_count, wasted_bytes)

    progress.finish()
    t_hash = time.monotonic() - t0
    print(f"      Hashing completed in {t_hash:.1f}s")

    # Phase 4: Write outputs
    print(f"\n[3/4] Writing results to {output_dir}...")

    # Write manifest (append if resume, overwrite otherwise)
    mode = "a" if resume and resume_records else "w"
    with open(manifest_path, mode, encoding="utf-8") as f:
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(
        f"      manifest.jsonl: {len(resume_records) + len(new_records):,} entries ({mode}ppend)"
        if mode == "a"
        else f"      manifest.jsonl: {len(new_records):,} entries"
    )

    # Build duplicate groups
    duplicate_groups: list[dict] = []
    for sha, group in hash_to_files.items():
        if len(group) >= 2:
            duplicate_groups.append(
                {
                    "sha256": sha,
                    "size": group[0]["size"],
                    "count": len(group),
                    "files": [r["path"] for r in group],
                }
            )

    duplicate_groups.sort(key=lambda g: g["size"] * (g["count"] - 1), reverse=True)

    with open(duplicates_path, "w", encoding="utf-8") as f:
        for grp in duplicate_groups:
            f.write(json.dumps(grp, ensure_ascii=False) + "\n")
    print(f"      duplicates.jsonl: {len(duplicate_groups):,} groups")

    # Build per-directory breakdown
    all_records = resume_records + new_records
    dir_stats: dict[str, dict] = defaultdict(lambda: {"file_count": 0, "total_size": 0})
    unique_hashes: set[str] = set()
    total_size = 0

    for rec in all_records:
        sd = rec.get("source_dir", "unknown")
        dir_stats[sd]["file_count"] += 1
        dir_stats[sd]["total_size"] += rec["size"]
        unique_hashes.add(rec["sha256"])
        total_size += rec["size"]

    total_files = len(all_records)
    total_unique = len(unique_hashes)
    total_duplicate_files = sum(g["count"] - 1 for g in duplicate_groups)
    total_wasted = sum(g["size"] * (g["count"] - 1) for g in duplicate_groups)

    # Human-readable dir stats
    dir_breakdown = {}
    for d in sorted(dir_stats.keys()):
        s = dir_stats[d]
        dir_breakdown[d] = {
            "file_count": s["file_count"],
            "total_size": s["total_size"],
            "total_size_human": _human_size(s["total_size"]),
        }

    summary = {
        "scan_timestamp": datetime.now(timezone.utc).isoformat(),
        "scan_dirs": scan_dirs,
        "total_files": total_files,
        "total_size": total_size,
        "total_size_human": _human_size(total_size),
        "unique_files": total_unique,
        "duplicate_groups": len(duplicate_groups),
        "duplicate_files": total_duplicate_files,
        "space_wasted": total_wasted,
        "space_wasted_human": _human_size(total_wasted),
        "per_directory": dir_breakdown,
        "workers": workers,
        "min_size": min_size,
        "max_size": max_size,
        "hash_time_seconds": round(t_hash, 2),
        "discover_time_seconds": round(t_discover, 2),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("      scan_summary.json: written")

    # Phase 5: Print summary
    print("\n[4/4] Scan Summary")
    print(f"{'=' * 60}")
    print(f"Total files scanned:     {total_files:>12,}")
    print(f"Total size:              {_human_size(total_size):>12}")
    print(f"Unique files:            {total_unique:>12,}")
    print(f"Duplicate groups:        {len(duplicate_groups):>12,}")
    print(f"Duplicate file copies:   {total_duplicate_files:>12,}")
    print(f"Space wasted:            {_human_size(total_wasted):>12}")
    print(f"{'=' * 60}")

    # Top 10 largest duplicate groups
    if duplicate_groups:
        print(f"\nTop {min(10, len(duplicate_groups))} Largest Duplicate Groups:")
        print(f"{'─' * 60}")
        for i, grp in enumerate(duplicate_groups[:10], 1):
            wasted = grp["size"] * (grp["count"] - 1)
            print(
                f"  {i:>2}. {_human_size(wasted)} wasted | {grp['count']} copies | size={_human_size(grp['size'])}"
            )
            for fp in grp["files"][:4]:
                print(f"      - {fp}")
            if len(grp["files"]) > 4:
                print(f"      ... and {len(grp['files']) - 4} more")
        print()

    # Per-directory breakdown
    print("Per-Directory Breakdown:")
    print(f"{'─' * 60}")
    for d in sorted(dir_breakdown.keys()):
        s = dir_breakdown[d]
        print(f"  {d}")
        print(f"    Files: {s['file_count']:,}  |  Size: {s['total_size_human']}")
    print()

    logger.info(
        "Scan complete: %d files, %d unique, %d dup groups, %s wasted",
        total_files,
        total_unique,
        len(duplicate_groups),
        _human_size(total_wasted),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _detect_output_dir() -> str:
    """Auto-detect output directory: prefer B: drive, fall back to local."""
    b_index = Path("/mnt/b/BIZRA/06_INDEX")
    if b_index.is_dir():
        return str(b_index)
    return "/mnt/c/BIZRA-DATA-LAKE/06_INDEX"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="BIZRA Migration Dedup Scanner -- SHA-256 duplicate detection across BIZRA directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python dedup_scanner.py                      # Full scan with defaults\n"
            "  python dedup_scanner.py --resume              # Continue interrupted scan\n"
            "  python dedup_scanner.py --workers 16          # Use 16 hashing workers\n"
            "  python dedup_scanner.py --dirs /mnt/c/BIZRA-DATA-LAKE /mnt/c/BIZRA-OS\n"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_detect_output_dir(),
        help="Output directory for manifest, duplicates, and summary (default: auto-detect B: or local 06_INDEX/)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() // 2),
        help=f"Number of parallel hashing workers (default: {max(1, cpu_count() // 2)})",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=ONE_KB,
        help=f"Minimum file size in bytes to include (default: {ONE_KB})",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=TEN_GB,
        help=f"Maximum file size in bytes to include (default: {TEN_GB})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing manifest.jsonl, skipping already-scanned files",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        type=str,
        default=None,
        help="Override scan directory list (space-separated paths)",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point."""
    args = parse_args()
    try:
        run_scan(args)
    except KeyboardInterrupt:
        print("\n\nScan interrupted by user. Partial results may be available.")
        sys.exit(130)
    except Exception:
        logger.exception("Fatal error during scan")
        raise


if __name__ == "__main__":
    main()
