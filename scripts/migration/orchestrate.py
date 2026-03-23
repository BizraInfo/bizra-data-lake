#!/usr/bin/env python3
"""BIZRA Sovereign Migration Orchestrator.

Master pipeline that ties together all 6 migration phases into a single
cohesive workflow with phase tracking, receipts, resume support, and
safety gates.

Phases:
    1. dedup_scanner     -- Build manifest of all files (non-destructive)
    2. artifact_cleaner  -- Clean build artifacts (~103 GB)
    3. repo_migrator     -- Migrate repos to B:\\01_CORE
    4. asset_mover       -- Move non-repo assets to B:\\03_ASSETS
    5. import_staging    -- Stage Downloads/Desktop/OneDrive (copy, not move)
    6. master_index      -- Generate final manifest.parquet + stats

Usage:
    python scripts/migration/orchestrate.py                  # dry-run all phases
    python scripts/migration/orchestrate.py --resume          # skip completed phases
    python scripts/migration/orchestrate.py --phase 4         # run only phase 4
    python scripts/migration/orchestrate.py --execute         # destructive mode
    python scripts/migration/orchestrate.py --dry-run         # explicit dry-run (default)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # /mnt/c/BIZRA-DATA-LAKE
LOG_DIR = REPO_ROOT / "logs" / "migration"

DEFAULT_SOVEREIGN_ROOT = "/mnt/b/BIZRA"
GENESIS_MANIFEST_NAME = "GENESIS_MANIFEST.yaml"

STATE_FILE_RELATIVE = "06_INDEX/migration_state.json"

HASH_CHUNK_SIZE = 65536  # 64 KB

TOTAL_PHASES = 6

PHASE_NAMES: dict[int, str] = {
    1: "dedup_scanner",
    2: "artifact_cleaner",
    3: "repo_migrator",
    4: "asset_mover",
    5: "import_staging",
    6: "master_index",
}

PHASE_DESCRIPTIONS: dict[int, str] = {
    1: "Build manifest of all files (non-destructive scan)",
    2: "Clean build artifacts (~103 GB recoverable)",
    3: "Migrate git repos to B:\\01_CORE",
    4: "Move non-repo assets to B:\\03_ASSETS",
    5: "Stage Downloads/Desktop/OneDrive to B:\\05_IMPORTS (copy)",
    6: "Generate final manifest + stats report",
}

# Phase 4: asset sources -> destinations (rsync moves)
ASSET_MOVES: list[tuple[str, str]] = [
    ("/mnt/c/bizra-voice/", "/mnt/b/BIZRA/03_ASSETS/voice/"),
    ("/mnt/c/award-winner-design/", "/mnt/b/BIZRA/03_ASSETS/design/"),
    ("/mnt/c/momo phone/", "/mnt/b/BIZRA/03_ASSETS/mobile/momo_phone/"),
    ("/mnt/c/BIZRA-NODE0-BACKUP/", "/mnt/b/BIZRA/04_ARCHIVE/backups/node0/"),
    ("/mnt/c/BIZRA_QUARANTINE_2025-12-30/", "/mnt/b/BIZRA/04_ARCHIVE/quarantine_2025/"),
]

# Phase 5: import sources -> destinations (rsync copies, source preserved)
IMPORT_COPIES: list[tuple[str, str]] = [
    ("/mnt/c/Users/BIZRA-OS/Downloads/", "/mnt/b/BIZRA/05_IMPORTS/downloads/"),
    ("/mnt/c/Users/BIZRA-OS/Desktop/", "/mnt/b/BIZRA/05_IMPORTS/desktop/"),
    ("/mnt/c/Users/BIZRA-OS/OneDrive/", "/mnt/b/BIZRA/05_IMPORTS/cloud/onedrive/"),
]

# Minimum free space thresholds
MIN_B_FREE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
WARN_C_FREE_BYTES = 50 * 1024 * 1024 * 1024  # 50 GB

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(timestamp: str) -> logging.Logger:
    """Configure file + stderr logging for the orchestrator."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"orchestrate_{timestamp}.log"

    logger = logging.getLogger("orchestrate")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler -- verbose
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    logger.addHandler(fh)

    # Stderr handler -- info level
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _human_size(nbytes: int) -> str:
    """Convert bytes to human-readable string."""
    if nbytes < 1024:
        return f"{nbytes} B"
    value = float(nbytes)
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024.0
        if value < 1024.0:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} PB"


def _human_duration(seconds: float) -> str:
    """Convert seconds to human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def _iso_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _iso_mtime(mtime: float) -> str:
    """Convert mtime float to ISO 8601 string."""
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def _disk_free(path: str) -> int:
    """Return free bytes on the filesystem containing *path*."""
    try:
        stat = os.statvfs(path)
        return stat.f_bavail * stat.f_frsize
    except OSError:
        return 0


def _disk_total(path: str) -> int:
    """Return total bytes on the filesystem containing *path*."""
    try:
        stat = os.statvfs(path)
        return stat.f_blocks * stat.f_frsize
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class MigrationState:
    """Tracks which phases have completed, with timestamps and receipts."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load state from disk, or initialize empty state."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "version": 1,
            "created": _iso_now(),
            "phases": {},
            "errors": [],
        }

    def save(self) -> None:
        """Persist state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.data["last_updated"] = _iso_now()
        tmp_path = self.state_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        os.replace(str(tmp_path), str(self.state_path))

    def is_phase_complete(self, phase: int) -> bool:
        """Return True if the given phase has completed successfully."""
        phase_key = str(phase)
        phase_data = self.data.get("phases", {}).get(phase_key, {})
        return phase_data.get("status") == "completed"

    def mark_started(self, phase: int) -> None:
        """Record that a phase has started."""
        phase_key = str(phase)
        if "phases" not in self.data:
            self.data["phases"] = {}
        self.data["phases"][phase_key] = {
            "name": PHASE_NAMES.get(phase, f"phase_{phase}"),
            "status": "running",
            "started": _iso_now(),
            "completed": None,
            "duration_s": None,
            "receipt": None,
            "error": None,
        }
        self.save()

    def mark_completed(
        self, phase: int, duration_s: float, receipt: Optional[dict] = None
    ) -> None:
        """Record that a phase completed successfully."""
        phase_key = str(phase)
        phase_data = self.data.get("phases", {}).get(phase_key, {})
        phase_data["status"] = "completed"
        phase_data["completed"] = _iso_now()
        phase_data["duration_s"] = round(duration_s, 2)
        phase_data["receipt"] = receipt
        phase_data["error"] = None
        self.data["phases"][phase_key] = phase_data
        self.save()

    def mark_failed(self, phase: int, duration_s: float, error: str) -> None:
        """Record that a phase failed."""
        phase_key = str(phase)
        phase_data = self.data.get("phases", {}).get(phase_key, {})
        phase_data["status"] = "failed"
        phase_data["completed"] = _iso_now()
        phase_data["duration_s"] = round(duration_s, 2)
        phase_data["error"] = error
        self.data["phases"][phase_key] = phase_data
        self.data["errors"].append(
            {
                "phase": phase,
                "timestamp": _iso_now(),
                "error": error,
            }
        )
        self.save()

    def mark_skipped(self, phase: int, reason: str) -> None:
        """Record that a phase was skipped."""
        phase_key = str(phase)
        self.data.setdefault("phases", {})[phase_key] = {
            "name": PHASE_NAMES.get(phase, f"phase_{phase}"),
            "status": "skipped",
            "started": _iso_now(),
            "completed": _iso_now(),
            "duration_s": 0,
            "receipt": None,
            "error": None,
            "skip_reason": reason,
        }
        self.save()


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def _verify_sovereign_mount(sovereign_root: str, logger: logging.Logger) -> bool:
    """Verify B: drive is mounted and GENESIS_MANIFEST.yaml exists."""
    manifest = Path(sovereign_root) / GENESIS_MANIFEST_NAME
    if not manifest.exists():
        logger.error(
            "ABORT: Sovereign drive not verified. " "Expected %s but file not found.",
            manifest,
        )
        logger.error(
            "       Ensure B: drive is mounted and %s exists.",
            GENESIS_MANIFEST_NAME,
        )
        return False
    logger.info("Sovereign drive verified: %s", manifest)
    return True


def _check_disk_space(sovereign_root: str, logger: logging.Logger) -> bool:
    """Check free space on B: and C: drives. Returns False only on critical failure."""
    # Check B: free space
    b_free = _disk_free(sovereign_root)
    b_total = _disk_total(sovereign_root)
    if b_free > 0:
        logger.info(
            "B: drive space: %s free / %s total",
            _human_size(b_free),
            _human_size(b_total),
        )
        if b_free < MIN_B_FREE_BYTES:
            logger.error(
                "ABORT: B: drive has only %s free (minimum: %s).",
                _human_size(b_free),
                _human_size(MIN_B_FREE_BYTES),
            )
            return False
    else:
        logger.warning("Could not determine B: drive free space.")

    # Check C: free space (warning only)
    c_free = _disk_free("/mnt/c")
    if c_free > 0:
        logger.info("C: drive space: %s free", _human_size(c_free))
        if c_free < WARN_C_FREE_BYTES:
            logger.warning(
                "WARNING: C: drive has only %s free (recommend >= %s).",
                _human_size(c_free),
                _human_size(WARN_C_FREE_BYTES),
            )
    return True


# ---------------------------------------------------------------------------
# Phase 1: Dedup Scanner (subprocess)
# ---------------------------------------------------------------------------


def _run_phase_1(
    sovereign_root: str, dry_run: bool, logger: logging.Logger, *, resume: bool = False
) -> dict:
    """Phase 1: Run dedup_scanner.py as subprocess."""
    script = SCRIPT_DIR / "dedup_scanner.py"
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    output_dir = f"{sovereign_root}/06_INDEX"

    cmd = [
        sys.executable,
        str(script),
        "--output-dir",
        output_dir,
    ]

    # Forward --resume to dedup_scanner so it skips already-hashed files
    if resume:
        cmd.append("--resume")

    logger.info("  Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    # Read summary if available
    summary_path = Path(output_dir) / "scan_summary.json"
    receipt: dict[str, Any] = {"output_dir": output_dir}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            receipt["total_files"] = summary.get("total_files", 0)
            receipt["total_size_human"] = summary.get("total_size_human", "unknown")
            receipt["duplicate_groups"] = summary.get("duplicate_groups", 0)
            receipt["space_wasted_human"] = summary.get("space_wasted_human", "unknown")
        except (json.JSONDecodeError, OSError):
            pass

    return receipt


# ---------------------------------------------------------------------------
# Phase 2: Artifact Cleaner (subprocess)
# ---------------------------------------------------------------------------


def _run_phase_2(sovereign_root: str, dry_run: bool, logger: logging.Logger) -> dict:
    """Phase 2: Run artifact_cleaner.py as subprocess."""
    script = SCRIPT_DIR / "artifact_cleaner.py"
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    cmd = [sys.executable, str(script)]

    if not dry_run:
        cmd.extend(["--execute", "--confirm", "I understand this deletes files"])

    logger.info("  Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    return {"mode": "execute" if not dry_run else "dry-run"}


# ---------------------------------------------------------------------------
# Phase 3: Repo Migrator (subprocess)
# ---------------------------------------------------------------------------


def _run_phase_3(sovereign_root: str, dry_run: bool, logger: logging.Logger) -> dict:
    """Phase 3: Run repo_migrator.py as subprocess."""
    script = SCRIPT_DIR / "repo_migrator.py"
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    cmd = [sys.executable, str(script)]

    if not dry_run:
        cmd.extend(["--execute", "--confirm", "I understand this moves repos"])

    # Forward BIZRA_SOVEREIGN_ROOT to the subprocess so it targets the correct drive
    env = os.environ.copy()
    env["BIZRA_SOVEREIGN_ROOT"] = sovereign_root

    logger.info("  Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)

    # Aggregate receipts from repo_migrator
    receipt: dict[str, Any] = {"mode": "execute" if not dry_run else "dry-run"}
    receipt_dir = Path(sovereign_root) / "06_INDEX" / "receipts"
    if receipt_dir.exists():
        receipts = sorted(receipt_dir.glob("*.json"))
        receipt["receipt_count"] = len(receipts)
        if receipts:
            try:
                with open(receipts[-1], "r", encoding="utf-8") as f:
                    receipt["latest_receipt"] = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

    return receipt


# ---------------------------------------------------------------------------
# Phase 4: Asset Mover (inline, rsync)
# ---------------------------------------------------------------------------


def _run_rsync(
    src: str,
    dst: str,
    move: bool,
    dry_run: bool,
    logger: logging.Logger,
) -> dict:
    """Run a single rsync transfer. Returns a receipt dict.

    Args:
        src: Source path (must end with / for directory contents).
        dst: Destination path.
        move: If True, delete source files after transfer (rsync --remove-source-files).
        dry_run: If True, pass --dry-run to rsync.
        logger: Logger instance.

    Returns:
        Dict with transfer metadata.
    """
    # Ensure destination parent exists
    dst_parent = Path(dst).parent if not dst.endswith("/") else Path(dst)
    dst_parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rsync",
        "-a",
        "--info=progress2",
        "--human-readable",
    ]
    if dry_run:
        cmd.append("--dry-run")
    if move:
        cmd.append("--remove-source-files")

    cmd.extend([src, dst])

    logger.info("  rsync: %s -> %s %s", src, dst, "[DRY RUN]" if dry_run else "")
    logger.debug("  Command: %s", " ".join(cmd))

    t0 = time.monotonic()
    subprocess.run(
        cmd,
        capture_output=False,
        check=True,
    )
    duration = time.monotonic() - t0

    return {
        "source": src,
        "destination": dst,
        "mode": "move" if move else "copy",
        "dry_run": dry_run,
        "duration_s": round(duration, 2),
    }


def _run_phase_4(sovereign_root: str, dry_run: bool, logger: logging.Logger) -> dict:
    """Phase 4: Move non-repo BIZRA assets to B:\\03_ASSETS and B:\\04_ARCHIVE."""
    # Check B: free space before data-moving phase
    b_free = _disk_free(sovereign_root)
    if b_free > 0 and b_free < MIN_B_FREE_BYTES:
        raise RuntimeError(
            f"Insufficient B: space: {_human_size(b_free)} free "
            f"(need >= {_human_size(MIN_B_FREE_BYTES)})"
        )

    transfers: list[dict] = []
    skipped: list[str] = []

    for src, dst in ASSET_MOVES:
        if not Path(src.rstrip("/")).exists():
            logger.warning("  Source not found, skipping: %s", src)
            skipped.append(src)
            continue

        try:
            receipt = _run_rsync(src, dst, move=True, dry_run=dry_run, logger=logger)
            transfers.append(receipt)
        except subprocess.CalledProcessError as exc:
            logger.error(
                "  rsync failed for %s -> %s: exit code %d", src, dst, exc.returncode
            )
            transfers.append(
                {
                    "source": src,
                    "destination": dst,
                    "error": f"rsync exit code {exc.returncode}",
                }
            )

    return {
        "transfers": transfers,
        "skipped": skipped,
        "mode": "dry-run" if dry_run else "execute",
    }


# ---------------------------------------------------------------------------
# Phase 5: Import Staging (inline, rsync copy)
# ---------------------------------------------------------------------------


def _run_phase_5(sovereign_root: str, dry_run: bool, logger: logging.Logger) -> dict:
    """Phase 5: Copy Downloads/Desktop/OneDrive to B:\\05_IMPORTS."""
    # Check B: free space
    b_free = _disk_free(sovereign_root)
    if b_free > 0 and b_free < MIN_B_FREE_BYTES:
        raise RuntimeError(
            f"Insufficient B: space: {_human_size(b_free)} free "
            f"(need >= {_human_size(MIN_B_FREE_BYTES)})"
        )

    transfers: list[dict] = []
    skipped: list[str] = []

    for src, dst in IMPORT_COPIES:
        if not Path(src.rstrip("/")).exists():
            logger.warning("  Source not found, skipping: %s", src)
            skipped.append(src)
            continue

        try:
            # Copy, NOT move — source is preserved
            receipt = _run_rsync(src, dst, move=False, dry_run=dry_run, logger=logger)
            transfers.append(receipt)
        except subprocess.CalledProcessError as exc:
            logger.error(
                "  rsync failed for %s -> %s: exit code %d", src, dst, exc.returncode
            )
            transfers.append(
                {
                    "source": src,
                    "destination": dst,
                    "error": f"rsync exit code {exc.returncode}",
                }
            )

    return {
        "transfers": transfers,
        "skipped": skipped,
        "mode": "dry-run" if dry_run else "execute",
    }


# ---------------------------------------------------------------------------
# Phase 6: Master Index (inline, multiprocessing hash)
# ---------------------------------------------------------------------------


def _hash_file_for_index(file_path: str) -> Optional[tuple[str, str, int, float]]:
    """Hash a single file for the master index. Worker-safe.

    Returns (path, sha256, size, mtime) or None on error.
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
        return (file_path, h.hexdigest(), size, stat.st_mtime)
    except (PermissionError, OSError):
        return None


def _categorize_path(file_path: str, sovereign_root: str) -> str:
    """Determine category based on path within sovereign directory structure."""
    rel = file_path[len(sovereign_root) :].lstrip("/")
    parts = rel.split("/", 1)
    if parts:
        top_dir = parts[0]
        category_map = {
            "00_CONSTITUTION": "constitution",
            "01_CORE": "core_repos",
            "02_DATA_PIPELINE": "data_pipeline",
            "03_ASSETS": "assets",
            "04_ARCHIVE": "archive",
            "05_IMPORTS": "imports",
            "06_INDEX": "index",
        }
        return category_map.get(top_dir, "other")
    return "root"


def _run_phase_6(sovereign_root: str, dry_run: bool, logger: logging.Logger) -> dict:
    """Phase 6: Walk B:\\BIZRA recursively, generate manifest + timeline + stats."""
    if dry_run:
        logger.info("  [DRY RUN] Would generate master index for %s", sovereign_root)
        return {"mode": "dry-run", "note": "Skipped in dry-run mode"}

    index_dir = Path(sovereign_root) / "06_INDEX"
    index_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = index_dir / "master_manifest.jsonl"
    timeline_path = index_dir / "timeline.jsonl"
    stats_path = index_dir / "stats_report.json"

    # Phase 6a: Discover all files
    logger.info("  Discovering files in %s ...", sovereign_root)
    all_files: list[str] = []
    skip_dirs = {".git", "__pycache__", "node_modules", "target", ".venv"}

    for dirpath, dirnames, filenames in os.walk(
        sovereign_root, topdown=True, followlinks=False
    ):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if not os.path.islink(fpath):
                all_files.append(fpath)

    total_files = len(all_files)
    logger.info("  Discovered %d files to index", total_files)

    if total_files == 0:
        logger.warning("  No files found in %s — writing empty index.", sovereign_root)
        for p in (manifest_path, timeline_path):
            p.write_text("")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump({"total_files": 0, "total_size": 0}, f, indent=2)
        return {"total_files": 0, "total_size": 0}

    # Phase 6b: Hash all files in parallel
    workers = max(1, cpu_count() // 2)
    logger.info("  Hashing %d files with %d workers ...", total_files, workers)

    t0 = time.monotonic()
    records: list[dict] = []
    errors = 0

    with Pool(processes=workers) as pool:
        processed = 0
        for result in pool.imap_unordered(
            _hash_file_for_index, all_files, chunksize=64
        ):
            processed += 1
            if processed % 5000 == 0 or processed == total_files:
                pct = processed / total_files * 100
                sys.stdout.write(
                    f"\r  Indexing: {processed}/{total_files} ({pct:.1f}%)"
                )
                sys.stdout.flush()

            if result is None:
                errors += 1
                continue

            fpath, sha256, size, mtime = result
            category = _categorize_path(fpath, sovereign_root)
            records.append(
                {
                    "path": fpath,
                    "sha256": sha256,
                    "size": size,
                    "mtime": _iso_mtime(mtime),
                    "category": category,
                }
            )

    sys.stdout.write("\n")
    sys.stdout.flush()
    hash_duration = time.monotonic() - t0
    logger.info(
        "  Hashing completed in %s (%d errors)", _human_duration(hash_duration), errors
    )

    # Phase 6c: Write manifest.jsonl
    logger.info("  Writing %s ...", manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Phase 6d: Build timeline (files per day)
    logger.info("  Building timeline ...")
    day_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files_count": 0, "total_size": 0}
    )
    for rec in records:
        # Extract date from mtime ISO string
        day = rec["mtime"][:10]  # YYYY-MM-DD
        day_stats[day]["files_count"] += 1
        day_stats[day]["total_size"] += rec["size"]

    with open(timeline_path, "w", encoding="utf-8") as f:
        for day in sorted(day_stats.keys()):
            entry = {
                "date": day,
                "files_count": day_stats[day]["files_count"],
                "total_size": day_stats[day]["total_size"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Phase 6e: Build stats report
    logger.info("  Building stats report ...")
    total_size = sum(r["size"] for r in records)

    # Per-directory breakdown (top-level sovereign dirs)
    dir_breakdown: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"file_count": 0, "total_size": 0}
    )
    for rec in records:
        rel = rec["path"][len(sovereign_root) :].lstrip("/")
        top = rel.split("/", 1)[0] if "/" in rel else "(root)"
        dir_breakdown[top]["file_count"] += 1
        dir_breakdown[top]["total_size"] += rec["size"]

    # Add human-readable sizes
    for key in dir_breakdown:
        dir_breakdown[key]["total_size_human"] = _human_size(
            dir_breakdown[key]["total_size"]
        )

    # Per-category breakdown
    cat_breakdown: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"file_count": 0, "total_size": 0}
    )
    for rec in records:
        cat_breakdown[rec["category"]]["file_count"] += 1
        cat_breakdown[rec["category"]]["total_size"] += rec["size"]
    for key in cat_breakdown:
        cat_breakdown[key]["total_size_human"] = _human_size(
            cat_breakdown[key]["total_size"]
        )

    stats = {
        "generated": _iso_now(),
        "sovereign_root": sovereign_root,
        "total_files": len(records),
        "total_size": total_size,
        "total_size_human": _human_size(total_size),
        "hash_errors": errors,
        "hash_duration_s": round(hash_duration, 2),
        "per_directory": dict(sorted(dir_breakdown.items())),
        "per_category": dict(sorted(cat_breakdown.items())),
        "b_drive_free": _disk_free(sovereign_root),
        "b_drive_free_human": _human_size(_disk_free(sovereign_root)),
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("  Wrote: %s", manifest_path)
    logger.info("  Wrote: %s", timeline_path)
    logger.info("  Wrote: %s", stats_path)

    return {
        "total_files": len(records),
        "total_size_human": _human_size(total_size),
        "hash_errors": errors,
        "manifest": str(manifest_path),
        "timeline": str(timeline_path),
        "stats": str(stats_path),
    }


# ---------------------------------------------------------------------------
# Phase dispatcher
# ---------------------------------------------------------------------------

PHASE_RUNNERS: dict[int, Any] = {
    1: _run_phase_1,
    2: _run_phase_2,
    3: _run_phase_3,
    4: _run_phase_4,
    5: _run_phase_5,
    6: _run_phase_6,
}


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------


def _print_header(dry_run: bool, resume: bool, sovereign_root: str) -> None:
    """Print the pipeline header."""
    mode = "DRY RUN" if dry_run else "EXECUTE"
    print()
    print("=" * 72)
    print("  BIZRA SOVEREIGN MIGRATION ORCHESTRATOR")
    print("=" * 72)
    print(f"  Mode:            {mode}")
    print(f"  Resume:          {'ON' if resume else 'OFF'}")
    print(f"  Sovereign root:  {sovereign_root}")
    print(f"  Timestamp:       {_iso_now()}")
    print("=" * 72)
    print()


def _print_phase_header(phase: int) -> None:
    """Print a phase header."""
    name = PHASE_NAMES.get(phase, f"phase_{phase}")
    desc = PHASE_DESCRIPTIONS.get(phase, "")
    print(f"{'─' * 72}")
    print(f"  Phase {phase}/{TOTAL_PHASES}: {name}")
    print(f"  {desc}")
    print(f"{'─' * 72}")


def _print_summary(state: MigrationState, total_duration: float) -> None:
    """Print the final summary table."""
    print()
    print("=" * 72)
    print("  MIGRATION SUMMARY")
    print("=" * 72)
    print()
    print(f"  {'Phase':<5} {'Name':<20} {'Status':<12} {'Duration':<12} {'Details'}")
    print(f"  {'─'*5} {'─'*20} {'─'*12} {'─'*12} {'─'*30}")

    for phase_num in range(1, TOTAL_PHASES + 1):
        phase_key = str(phase_num)
        phase_data = state.data.get("phases", {}).get(phase_key, {})
        name = PHASE_NAMES.get(phase_num, f"phase_{phase_num}")
        status = phase_data.get("status", "not_run")
        duration = phase_data.get("duration_s", 0)
        duration_str = _human_duration(duration) if duration else "--"

        # Status indicator
        if status == "completed":
            indicator = "[OK]"
        elif status == "failed":
            indicator = "[FAIL]"
        elif status == "skipped":
            indicator = "[SKIP]"
        elif status == "running":
            indicator = "[...]"
        else:
            indicator = "[--]"

        # Details column
        details = ""
        receipt = phase_data.get("receipt")
        if receipt:
            if phase_num == 1:
                details = f"{receipt.get('total_files', '?')} files, {receipt.get('space_wasted_human', '?')} wasted"
            elif phase_num == 6:
                details = f"{receipt.get('total_files', '?')} indexed, {receipt.get('total_size_human', '?')}"
            elif "mode" in receipt:
                details = receipt["mode"]
        elif status == "failed":
            details = (phase_data.get("error", "") or "")[:40]
        elif status == "skipped":
            details = (phase_data.get("skip_reason", "") or "")[:40]

        print(
            f"  {phase_num:<5} {name:<20} {indicator:<12} {duration_str:<12} {details}"
        )

    print()
    print(f"  Total duration: {_human_duration(total_duration)}")
    print(f"  State file:     {state.state_path}")
    print()
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> int:
    """Execute the migration pipeline."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger = _setup_logging(timestamp)

    sovereign_root = args.sovereign_root
    dry_run = not args.execute
    resume = args.resume
    target_phase = args.phase

    _print_header(dry_run, resume, sovereign_root)

    # Pre-flight: verify sovereign mount
    if not _verify_sovereign_mount(sovereign_root, logger):
        return 1

    # Pre-flight: check disk space
    if not _check_disk_space(sovereign_root, logger):
        return 1

    # Load state
    state_path = Path(sovereign_root) / STATE_FILE_RELATIVE
    state = MigrationState(state_path)
    logger.info("State file: %s", state_path)

    # Determine which phases to run
    if target_phase is not None:
        if target_phase < 1 or target_phase > TOTAL_PHASES:
            logger.error(
                "Invalid phase number: %d (must be 1-%d)", target_phase, TOTAL_PHASES
            )
            return 1
        phases_to_run = [target_phase]
    else:
        phases_to_run = list(range(1, TOTAL_PHASES + 1))

    pipeline_start = time.monotonic()
    exit_code = 0

    for phase_num in phases_to_run:
        # Resume: skip completed phases
        if resume and state.is_phase_complete(phase_num):
            logger.info(
                "Phase %d: already completed (resume mode), skipping.", phase_num
            )
            state.mark_skipped(phase_num, "already completed (resume)")
            continue

        _print_phase_header(phase_num)

        # Check disk space before data-moving phases
        # Phase 2 (artifact_cleaner) frees space on C:, NOT on B: — skip B: check
        # Phases 3-5 write to B: — check B: free space
        if phase_num in (3, 4, 5) and not dry_run:
            b_free = _disk_free(sovereign_root)
            if b_free > 0 and b_free < MIN_B_FREE_BYTES:
                msg = (
                    f"Insufficient B: space before phase {phase_num}: "
                    f"{_human_size(b_free)} free (need >= {_human_size(MIN_B_FREE_BYTES)})"
                )
                logger.error("  %s", msg)
                state.mark_failed(phase_num, 0, msg)
                exit_code = 1
                break

        runner = PHASE_RUNNERS.get(phase_num)
        if runner is None:
            logger.error("  No runner defined for phase %d", phase_num)
            state.mark_failed(phase_num, 0, "no runner defined")
            exit_code = 1
            break

        state.mark_started(phase_num)
        phase_start = time.monotonic()

        try:
            # Phase 1 gets the resume flag so it can forward --resume to dedup_scanner
            if phase_num == 1 and resume:
                receipt = runner(sovereign_root, dry_run, logger, resume=True)
            else:
                receipt = runner(sovereign_root, dry_run, logger)
            phase_duration = time.monotonic() - phase_start
            state.mark_completed(phase_num, phase_duration, receipt)
            logger.info(
                "  Phase %d completed in %s",
                phase_num,
                _human_duration(phase_duration),
            )
            print()

        except FileNotFoundError as exc:
            phase_duration = time.monotonic() - phase_start
            error_msg = str(exc)
            state.mark_failed(phase_num, phase_duration, error_msg)
            logger.error("  Phase %d FAILED: %s", phase_num, error_msg)
            exit_code = 1
            break

        except subprocess.CalledProcessError as exc:
            phase_duration = time.monotonic() - phase_start
            error_msg = f"Subprocess exit code {exc.returncode}"
            state.mark_failed(phase_num, phase_duration, error_msg)
            logger.error("  Phase %d FAILED: %s", phase_num, error_msg)
            exit_code = 1
            break

        except KeyboardInterrupt:
            phase_duration = time.monotonic() - phase_start
            state.mark_failed(phase_num, phase_duration, "interrupted by user")
            logger.warning("  Phase %d interrupted by user.", phase_num)
            exit_code = 130
            break

        except Exception as exc:
            phase_duration = time.monotonic() - phase_start
            error_msg = f"{type(exc).__name__}: {exc}"
            state.mark_failed(phase_num, phase_duration, error_msg)
            logger.exception("  Phase %d FAILED with unexpected error", phase_num)
            exit_code = 1
            break

    total_duration = time.monotonic() - pipeline_start

    # Record total duration in state
    state.data["total_duration_s"] = round(total_duration, 2)
    state.data["last_run_mode"] = "dry-run" if dry_run else "execute"
    state.save()

    _print_summary(state, total_duration)

    if exit_code == 0:
        logger.info("Pipeline completed successfully.")
    else:
        logger.error("Pipeline stopped with errors (exit code %d).", exit_code)

    return exit_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "BIZRA Sovereign Migration Orchestrator -- "
            "master pipeline for migrating C: drive data to B: sovereign drive."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python orchestrate.py                    # dry-run all 6 phases\n"
            "  python orchestrate.py --resume           # skip completed phases\n"
            "  python orchestrate.py --phase 4          # run only phase 4\n"
            "  python orchestrate.py --execute          # destructive mode\n"
            "  python orchestrate.py --phase 1          # scan only (always safe)\n"
            "\n"
            "Phase overview:\n"
            "  1  dedup_scanner     Build SHA-256 manifest (non-destructive)\n"
            "  2  artifact_cleaner  Remove build caches (~103 GB)\n"
            "  3  repo_migrator     Move git repos to B:\\01_CORE\n"
            "  4  asset_mover       Move non-repo assets to B:\\03_ASSETS\n"
            "  5  import_staging    Copy Downloads/Desktop/OneDrive to B:\\05_IMPORTS\n"
            "  6  master_index      Generate final manifest + stats\n"
        ),
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        metavar="N",
        help="Run only phase N (1-6). Default: run all phases in order.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip phases that already completed successfully.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Enable destructive operations (default: dry-run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Explicit dry-run mode (this is the default).",
    )
    parser.add_argument(
        "--sovereign-root",
        type=str,
        default=os.environ.get("BIZRA_SOVEREIGN_ROOT", DEFAULT_SOVEREIGN_ROOT),
        metavar="PATH",
        help=(
            f"Path to sovereign drive root "
            f"(default: $BIZRA_SOVEREIGN_ROOT or {DEFAULT_SOVEREIGN_ROOT})."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point."""
    args = parse_args()

    # --dry-run is the default; --execute overrides it
    # If both are passed, --dry-run wins (safety first)
    if args.dry_run and args.execute:
        print(
            "WARNING: Both --execute and --dry-run specified. "
            "Dry-run takes precedence (safety first).",
            file=sys.stderr,
        )
        args.execute = False

    try:
        sys.exit(run_pipeline(args))
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
