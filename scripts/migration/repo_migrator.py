#!/usr/bin/env python3
"""BIZRA Repository Migrator — Phase 53, Step 3.

Copies git repositories from C:\\ to B:\\BIZRA\\01_CORE\\ with full integrity
verification (HEAD SHA match + git fsck).  Genesis repos are consolidated:
the candidate with the most recent commit wins.

Default mode is DRY RUN.  Pass --execute --confirm "I understand this moves repos"
to perform the actual migration.

Usage:
    python scripts/migration/repo_migrator.py
    python scripts/migration/repo_migrator.py --repos data-lake genesis
    python scripts/migration/repo_migrator.py --execute --confirm "I understand this moves repos"
    python scripts/migration/repo_migrator.py --skip-large 10  # skip repos > 10 GB
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOVEREIGN_ROOT = Path(os.environ.get("BIZRA_SOVEREIGN_ROOT", "/mnt/b/BIZRA"))
CORE_DIR = SOVEREIGN_ROOT / "01_CORE"
RECEIPT_DIR = SOVEREIGN_ROOT / "06_INDEX" / "receipts"
LOG_DIR = Path("/mnt/c/BIZRA-DATA-LAKE/logs/migration")

SAFETY_CONFIRM_STRING = "I understand this moves repos"

REPO_MAP: dict[str, tuple[str, str]] = {
    "data-lake": ("/mnt/c/BIZRA-DATA-LAKE", str(CORE_DIR / "data-lake")),
    "node0": ("/mnt/c/BIZRA-NODE0", str(CORE_DIR / "node0")),
    "dual-agentic": (
        "/mnt/c/BIZRA-Dual-Agentic-system--main",
        str(CORE_DIR / "dual-agentic"),
    ),
    "projects": ("/mnt/c/BIZRA-PROJECTS", str(CORE_DIR / "projects")),
    "taskmaster": ("/mnt/c/BIZRA-TaskMaster", str(CORE_DIR / "taskmaster")),
}

GENESIS_CANDIDATES: list[str] = [
    "/mnt/c/bizra-genesis-node",
    "/mnt/c/bizra-genesis-node-repaired",
    "/mnt/c/bizra-genesis-node-fresh",
    "/mnt/c/BIZRA-GENESIS-CLEAN",
]
GENESIS_DEST = str(CORE_DIR / "genesis")

RSYNC_EXCLUDES: list[str] = [
    "target/",
    "node_modules/",
    ".venv*",
    "__pycache__/",
    ".mypy_cache/",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging() -> logging.Logger:
    """Configure file + stderr logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"repo_migrate_{ts}.log"

    logger = logging.getLogger("repo_migrator")
    logger.setLevel(logging.DEBUG)

    # File handler -- verbose
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger.addHandler(fh)

    # Stderr handler -- info+
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(levelname)-8s  %(message)s"))
    logger.addHandler(sh)

    logger.info("Log file: %s", log_file)
    return logger


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RepoResult:
    """Migration result for a single repository."""

    key: str
    source: str
    dest: str
    source_size_bytes: int = 0
    dest_size_bytes: int = 0
    source_head_sha: str = ""
    dest_head_sha: str = ""
    sha_match: bool = False
    fsck_ok: bool = False
    fsck_output: str = ""
    status: str = "PENDING"  # PENDING | SKIPPED | DRY_RUN | OK | FAILED
    error: str = ""
    duration_seconds: float = 0.0
    genesis_candidates_skipped: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _run_cmd(
    cmd: list[str],
    *,
    timeout: int = 600,
    logger: logging.Logger,
) -> tuple[int, str, str]:
    """Run a subprocess command and return (returncode, stdout, stderr)."""
    logger.debug("CMD: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"


def _git_head_sha(repo_path: str, *, logger: logging.Logger) -> Optional[str]:
    """Return HEAD SHA for a git repo, or None on failure."""
    rc, stdout, stderr = _run_cmd(
        ["git", "-C", repo_path, "rev-parse", "HEAD"],
        logger=logger,
    )
    if rc != 0:
        logger.warning("git rev-parse failed for %s: %s", repo_path, stderr)
        return None
    return stdout


def _git_commit_timestamp(repo_path: str, *, logger: logging.Logger) -> Optional[int]:
    """Return HEAD commit unix timestamp, or None on failure."""
    rc, stdout, stderr = _run_cmd(
        ["git", "-C", repo_path, "log", "-1", "--format=%ct"],
        logger=logger,
    )
    if rc != 0 or not stdout.strip().isdigit():
        return None
    return int(stdout.strip())


def _git_fsck(repo_path: str, *, logger: logging.Logger) -> tuple[bool, str]:
    """Run git fsck --no-dangling.  Return (ok, output)."""
    rc, stdout, stderr = _run_cmd(
        ["git", "-C", repo_path, "fsck", "--no-dangling"],
        timeout=300,
        logger=logger,
    )
    combined = (stdout + "\n" + stderr).strip()
    return rc == 0, combined


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _dir_size(
    path: str,
    *,
    exclude_names: frozenset[str],
    exclude_prefixes: tuple[str, ...],
) -> int:
    """Walk a directory and sum file sizes, skipping excluded dirs."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        # Prune excluded directories in-place
        dirnames[:] = [
            d
            for d in dirnames
            if d not in exclude_names
            and not any(d.startswith(p) for p in exclude_prefixes)
        ]
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _free_space_bytes(path: str) -> int:
    """Return free bytes on the filesystem containing *path*."""
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def _human_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# Core migration logic
# ---------------------------------------------------------------------------


def _rsync_copy(
    source: str,
    dest: str,
    *,
    excludes: list[str],
    logger: logging.Logger,
) -> tuple[bool, str]:
    """Copy source to dest using rsync.  Return (ok, stderr)."""
    Path(dest).mkdir(parents=True, exist_ok=True)

    cmd = ["rsync", "-a", "--info=progress2"]
    for ex in excludes:
        cmd.extend(["--exclude", ex])
    # Trailing slash on source means "contents of"
    cmd.append(source.rstrip("/") + "/")
    cmd.append(dest.rstrip("/") + "/")

    rc, stdout, stderr = _run_cmd(cmd, timeout=3600, logger=logger)
    if rc != 0:
        return False, stderr
    return True, ""


def _write_marker(
    source: str,
    dest: str,
    head_sha: str,
) -> None:
    """Write .MIGRATED_TO_B marker in the source directory."""
    marker = Path(source) / ".MIGRATED_TO_B"
    payload = {
        "dest": dest,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "head_sha": head_sha,
        "verified": True,
    }
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_receipt(result: RepoResult) -> None:
    """Write a JSON receipt to the index directory."""
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    receipt_path = RECEIPT_DIR / f"{result.key}_{ts}.json"
    receipt_path.write_text(
        json.dumps(asdict(result), indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _resolve_genesis(*, logger: logging.Logger) -> tuple[Optional[str], list[str]]:
    """Pick the genesis candidate with the most recent commit.

    Returns (winner_path, skipped_paths).  winner_path is None if no
    candidate has a valid .git/ directory.
    """
    best_path: Optional[str] = None
    best_ts: int = -1
    skipped: list[str] = []

    for candidate in GENESIS_CANDIDATES:
        git_dir = Path(candidate) / ".git"
        if not git_dir.exists():
            logger.info("  SKIP (no .git): %s", candidate)
            skipped.append(candidate)
            continue

        ts = _git_commit_timestamp(candidate, logger=logger)
        if ts is None:
            logger.warning("  SKIP (no commit timestamp): %s", candidate)
            skipped.append(candidate)
            continue

        sha = _git_head_sha(candidate, logger=logger) or "unknown"
        logger.info(
            "  Candidate: %s  HEAD=%s  ts=%s",
            candidate,
            sha[:12],
            datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
        )

        if ts > best_ts:
            if best_path is not None:
                skipped.append(best_path)
            best_path = candidate
            best_ts = ts
        else:
            skipped.append(candidate)

    if best_path:
        logger.info("  WINNER: %s (ts=%d)", best_path, best_ts)
    else:
        logger.warning("  No valid genesis candidate found")

    return best_path, skipped


def migrate_repo(
    key: str,
    source: str,
    dest: str,
    *,
    execute: bool,
    excludes: list[str],
    skip_large_gb: Optional[float],
    logger: logging.Logger,
    genesis_skipped: Optional[list[str]] = None,
) -> RepoResult:
    """Migrate a single repository.  Returns the result record."""
    result = RepoResult(key=key, source=source, dest=dest)
    if genesis_skipped:
        result.genesis_candidates_skipped = genesis_skipped

    t0 = time.monotonic()

    # --- Pre-checks ---

    git_dir = Path(source) / ".git"
    if not git_dir.exists():
        result.status = "FAILED"
        result.error = f"No .git directory in {source}"
        logger.error("[%s] %s", key, result.error)
        result.duration_seconds = time.monotonic() - t0
        return result

    # Source HEAD SHA
    head_sha = _git_head_sha(source, logger=logger)
    if head_sha is None:
        result.status = "FAILED"
        result.error = f"Cannot read HEAD SHA for {source}"
        logger.error("[%s] %s", key, result.error)
        result.duration_seconds = time.monotonic() - t0
        return result
    result.source_head_sha = head_sha
    logger.info("[%s] Source HEAD: %s", key, head_sha[:12])

    # Source size (with exclusions)
    exclude_names = frozenset({"target", "node_modules", "__pycache__", ".mypy_cache"})
    exclude_prefixes = (".venv",)
    if excludes == []:
        # --no-exclude mode: count everything
        exclude_names = frozenset()
        exclude_prefixes = ()

    src_size = _dir_size(
        source,
        exclude_names=exclude_names,
        exclude_prefixes=exclude_prefixes,
    )
    result.source_size_bytes = src_size
    logger.info("[%s] Source size: %s", key, _human_bytes(src_size))

    # Skip if too large
    if skip_large_gb is not None:
        limit_bytes = int(skip_large_gb * 1024 * 1024 * 1024)
        if src_size > limit_bytes:
            result.status = "SKIPPED"
            result.error = (
                f"Source size {_human_bytes(src_size)} exceeds "
                f"limit {skip_large_gb} GB"
            )
            logger.warning("[%s] %s", key, result.error)
            result.duration_seconds = time.monotonic() - t0
            return result

    # Check destination free space
    dest_parent = str(Path(dest).parent)
    if Path(dest_parent).exists():
        free = _free_space_bytes(dest_parent)
        # Require at least source_size + 1 GB headroom
        headroom = 1024 * 1024 * 1024
        if free < src_size + headroom:
            result.status = "FAILED"
            result.error = (
                f"Insufficient space on destination: "
                f"{_human_bytes(free)} free, need {_human_bytes(src_size + headroom)}"
            )
            logger.error("[%s] %s", key, result.error)
            result.duration_seconds = time.monotonic() - t0
            return result
    else:
        logger.warning(
            "[%s] Destination parent %s does not exist yet", key, dest_parent
        )

    # --- Dry run stops here ---

    if not execute:
        result.status = "DRY_RUN"
        logger.info(
            "[%s] DRY RUN: would copy %s -> %s (%s)",
            key,
            source,
            dest,
            _human_bytes(src_size),
        )
        result.duration_seconds = time.monotonic() - t0
        return result

    # --- Execute migration ---

    logger.info("[%s] Starting rsync: %s -> %s", key, source, dest)
    ok, err = _rsync_copy(source, dest, excludes=excludes, logger=logger)
    if not ok:
        result.status = "FAILED"
        result.error = f"rsync failed: {err}"
        logger.error("[%s] %s", key, result.error)
        result.duration_seconds = time.monotonic() - t0
        return result

    # Post-check: HEAD SHA
    dest_sha = _git_head_sha(dest, logger=logger)
    result.dest_head_sha = dest_sha or ""
    result.sha_match = dest_sha == head_sha
    if not result.sha_match:
        result.status = "FAILED"
        result.error = (
            f"HEAD SHA mismatch: source={head_sha[:12]} dest={str(dest_sha)[:12]}"
        )
        logger.error("[%s] %s", key, result.error)
        result.duration_seconds = time.monotonic() - t0
        return result
    logger.info("[%s] HEAD SHA verified: %s", key, head_sha[:12])

    # Post-check: git fsck
    fsck_ok, fsck_output = _git_fsck(dest, logger=logger)
    result.fsck_ok = fsck_ok
    result.fsck_output = fsck_output
    if not fsck_ok:
        result.status = "FAILED"
        result.error = f"git fsck failed: {fsck_output[:200]}"
        logger.error("[%s] %s", key, result.error)
        result.duration_seconds = time.monotonic() - t0
        return result
    logger.info("[%s] git fsck passed", key)

    # Dest size
    result.dest_size_bytes = _dir_size(
        dest,
        exclude_names=exclude_names,
        exclude_prefixes=exclude_prefixes,
    )

    # Write marker and receipt
    _write_marker(source, dest, head_sha)
    logger.info("[%s] Marker written: %s/.MIGRATED_TO_B", key, source)

    result.status = "OK"
    result.duration_seconds = time.monotonic() - t0
    _write_receipt(result)
    logger.info("[%s] Migration complete in %.1fs", key, result.duration_seconds)

    return result


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def _print_summary(results: list[RepoResult], *, logger: logging.Logger) -> None:
    """Print a formatted summary table."""
    sep = "-" * 100
    logger.info("")
    logger.info(sep)
    logger.info("MIGRATION SUMMARY")
    logger.info(sep)
    logger.info(
        "%-14s %-10s %12s %12s %-6s %-6s %8s",
        "REPO",
        "STATUS",
        "SRC SIZE",
        "DST SIZE",
        "SHA",
        "FSCK",
        "TIME(s)",
    )
    logger.info(sep)

    total_src = 0
    total_dst = 0
    failed: list[str] = []

    for r in results:
        sha_col = (
            "yes"
            if r.sha_match
            else ("n/a" if r.status in ("DRY_RUN", "SKIPPED", "PENDING") else "NO")
        )
        fsck_col = (
            "yes"
            if r.fsck_ok
            else ("n/a" if r.status in ("DRY_RUN", "SKIPPED", "PENDING") else "NO")
        )
        logger.info(
            "%-14s %-10s %12s %12s %-6s %-6s %8.1f",
            r.key,
            r.status,
            _human_bytes(r.source_size_bytes),
            _human_bytes(r.dest_size_bytes) if r.dest_size_bytes else "-",
            sha_col,
            fsck_col,
            r.duration_seconds,
        )
        if r.error:
            logger.info("  ERROR: %s", r.error)
        if r.genesis_candidates_skipped:
            logger.info(
                "  Genesis skipped: %s",
                ", ".join(r.genesis_candidates_skipped),
            )

        total_src += r.source_size_bytes
        total_dst += r.dest_size_bytes
        if r.status == "FAILED":
            failed.append(r.key)

    logger.info(sep)
    logger.info(
        "Total source: %s  |  Total dest: %s",
        _human_bytes(total_src),
        _human_bytes(total_dst),
    )
    if failed:
        logger.error("FAILED repos: %s", ", ".join(failed))
    else:
        logger.info(
            "All repos: %s",
            "OK" if any(r.status == "OK" for r in results) else "DRY RUN",
        )
    logger.info(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate BIZRA git repositories from C:\\ to B:\\BIZRA\\01_CORE\\",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                                     # dry-run all repos\n"
            "  %(prog)s --repos genesis data-lake            # dry-run specific repos\n"
            '  %(prog)s --execute --confirm "I understand this moves repos"\n'
            "  %(prog)s --skip-large 10                      # skip repos > 10 GB\n"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the migration (default: dry run)",
    )
    parser.add_argument(
        "--confirm",
        type=str,
        default="",
        help=f'Safety confirmation: "{SAFETY_CONFIRM_STRING}"',
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        choices=list(REPO_MAP.keys()) + ["genesis"],
        help="Migrate only these repos (default: all)",
    )
    parser.add_argument(
        "--skip-large",
        type=float,
        default=None,
        metavar="GB",
        help="Skip repos larger than N GB",
    )
    parser.add_argument(
        "--no-exclude",
        action="store_true",
        help="Don't exclude target/node_modules/etc (full copy)",
    )
    return parser


def main() -> int:
    """Entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    logger = _setup_logging()
    logger.info("BIZRA Repository Migrator — Phase 53, Step 3")
    logger.info("Sovereign root: %s", SOVEREIGN_ROOT)
    logger.info("Mode: %s", "EXECUTE" if args.execute else "DRY RUN")

    # Safety gate
    if args.execute and args.confirm != SAFETY_CONFIRM_STRING:
        logger.error('Execution requires --confirm "%s"', SAFETY_CONFIRM_STRING)
        return 1

    # Determine excludes
    excludes = [] if args.no_exclude else list(RSYNC_EXCLUDES)

    # Build repo list
    repo_keys = args.repos if args.repos else list(REPO_MAP.keys()) + ["genesis"]

    results: list[RepoResult] = []

    for key in repo_keys:
        if key == "genesis":
            logger.info("[genesis] Resolving best candidate...")
            winner, skipped = _resolve_genesis(logger=logger)
            if winner is None:
                result = RepoResult(
                    key="genesis",
                    source="(none)",
                    dest=GENESIS_DEST,
                    status="FAILED",
                    error="No valid genesis candidate found",
                    genesis_candidates_skipped=skipped,
                )
                results.append(result)
                continue

            result = migrate_repo(
                key="genesis",
                source=winner,
                dest=GENESIS_DEST,
                execute=args.execute,
                excludes=excludes,
                skip_large_gb=args.skip_large,
                logger=logger,
                genesis_skipped=skipped,
            )
            results.append(result)
        else:
            source, dest = REPO_MAP[key]
            result = migrate_repo(
                key=key,
                source=source,
                dest=dest,
                execute=args.execute,
                excludes=excludes,
                skip_large_gb=args.skip_large,
                logger=logger,
            )
            results.append(result)

    _print_summary(results, logger=logger)

    # Exit code: non-zero if any FAILED
    if any(r.status == "FAILED" for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
