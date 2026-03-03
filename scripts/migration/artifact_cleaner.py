#!/usr/bin/env python3
"""BIZRA Artifact Cleaner — Pre-Migration Disk Recovery Tool.

Scans BIZRA project directories on C:\\ and reports (or deletes) build
artifacts and caches that are safe to remove before migrating to B:\\ drive.

Default mode is DRY RUN.  Pass --execute --confirm "I understand this deletes files"
to perform actual deletion.

Usage:
    python artifact_cleaner.py                          # dry-run report
    python artifact_cleaner.py --execute --confirm "I understand this deletes files"
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WSL_PREFIX = Path("/mnt/c")

SCAN_ROOTS: list[Path] = [
    WSL_PREFIX / "BIZRA-DATA-LAKE",
    WSL_PREFIX / "BIZRA-Dual-Agentic-system--main",
    WSL_PREFIX / "BIZRA-NODE0",
    WSL_PREFIX / "BIZRA-PROJECTS",
    WSL_PREFIX / "BIZRA-TaskMaster",
    WSL_PREFIX / "bizra-genesis-node",
    WSL_PREFIX / "bizra-genesis-node-repaired",
    WSL_PREFIX / "bizra-genesis-node-fresh",
    WSL_PREFIX / "BIZRA-GENESIS-CLEAN",
    WSL_PREFIX / "award-winner-design",
    WSL_PREFIX / "bizra-voice",
    WSL_PREFIX / "HERMES project",
    WSL_PREFIX / "BIZRA-NODE0-BACKUP",
]

LOG_DIR = WSL_PREFIX / "BIZRA-DATA-LAKE" / "logs" / "migration"

# The one venv we explicitly preserve.
PROTECTED_VENV = (WSL_PREFIX / "BIZRA-DATA-LAKE" / ".venv-linux").resolve()

# Artifact categories: (display_name, pattern_kind, pattern_value)
#   pattern_kind = "dir_name"  -> match directory basename
#   pattern_kind = "dir_glob"  -> match directory basename via glob
#   pattern_kind = "file_glob" -> match file basename via glob
ARTIFACT_RULES: list[tuple[str, str, str]] = [
    ("Rust target/", "dir_name", "target"),
    ("node_modules/", "dir_name", "node_modules"),
    ("__pycache__/", "dir_name", "__pycache__"),
    (".mypy_cache/", "dir_name", ".mypy_cache"),
    (".pytest_cache/", "dir_name", ".pytest_cache"),
    (".ruff_cache/", "dir_name", ".ruff_cache"),
    (".tox/", "dir_name", ".tox"),
    ("dist/ (build output)", "dir_name", "dist"),
    ("build/ (build output)", "dir_name", "build"),
    (".next/ (Next.js)", "dir_name", ".next"),
    (".turbo/", "dir_name", ".turbo"),
    (".venv*/", "dir_glob", ".venv*"),
    ("*.egg-info/", "dir_glob", "*.egg-info"),
    ("*.pyc (stray)", "file_glob", "*.pyc"),
]

# Maximum parallel workers for scanning root directories.
MAX_WORKERS = 6

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Artifact:
    """A single discovered artifact (directory or file)."""

    path: Path
    category: str
    size_bytes: int
    is_dir: bool


@dataclass
class ScanResult:
    """Aggregated result of scanning one root directory."""

    root: Path
    artifacts: list[Artifact] = field(default_factory=list)
    scan_time_s: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def human_size(nbytes: int) -> str:
    """Return a human-readable size string."""
    value = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value):,} B"
            return f"{value:,.1f} {unit}"
        value /= 1024.0
    return f"{value:,.1f} PB"


def dir_size(path: Path) -> int:
    """Calculate total size of a directory tree.  Follows no symlinks."""
    total = 0
    try:
        for entry in os.scandir(str(path)):
            try:
                if entry.is_symlink():
                    continue
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += dir_size(Path(entry.path))
            except (PermissionError, OSError):
                continue
    except (PermissionError, OSError):
        pass
    return total


def is_protected(path: Path) -> bool:
    """Return True if *path* is the protected venv we must keep."""
    try:
        return path.resolve() == PROTECTED_VENV
    except (OSError, ValueError):
        return False


def _matches_glob(name: str, pattern: str) -> bool:
    """Simple glob match for basenames (supports leading dot + trailing *)."""
    from fnmatch import fnmatch

    return fnmatch(name, pattern)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def scan_root(root: Path) -> ScanResult:
    """Walk *root* and collect all matching artifacts."""
    result = ScanResult(root=root)
    t0 = time.monotonic()

    if not root.exists():
        result.error = "directory does not exist"
        result.scan_time_s = time.monotonic() - t0
        return result

    # We walk manually so we can prune entire subtrees we already matched.
    # os.walk with topdown=True lets us modify dirs-in-place to skip subtrees.
    protected_str = str(PROTECTED_VENV)
    for dirpath_str, dirnames, filenames in os.walk(str(root), topdown=True):
        dirpath = Path(dirpath_str)

        # Skip everything inside the protected venv — never enter it at all.
        if dirpath_str.startswith(protected_str):
            dirnames.clear()
            continue

        # Check each subdirectory against artifact rules.
        prune: set[str] = set()
        for dname in dirnames:
            full = dirpath / dname

            # Skip symlinks entirely.
            if full.is_symlink():
                prune.add(dname)
                continue

            # Never descend into the protected venv.
            if str(full) == protected_str:
                prune.add(dname)
                continue

            for category, kind, pattern in ARTIFACT_RULES:
                if kind == "dir_name" and dname == pattern:
                    matched = True
                elif kind == "dir_glob" and _matches_glob(dname, pattern):
                    matched = True
                else:
                    matched = False

                if not matched:
                    continue

                # Protect the sacred venv itself from .venv* glob.
                if "venv" in pattern.lower() and is_protected(full):
                    continue

                # Rust target/ — only match if it looks like a Cargo output dir.
                # Heuristic: contains 'debug/' or 'release/' or '.rustc_info.json'.
                if pattern == "target" and not _looks_like_cargo_target(full):
                    continue

                # build/ — only match if parent has setup.py, setup.cfg, pyproject.toml,
                # package.json, or Cargo.toml (avoid false positives on app dirs named build/).
                if pattern == "build" and not _looks_like_build_output(full):
                    continue

                # dist/ — same heuristic as build/.
                if pattern == "dist" and not _looks_like_dist_output(full):
                    continue

                size = dir_size(full)
                result.artifacts.append(
                    Artifact(path=full, category=category, size_bytes=size, is_dir=True)
                )
                prune.add(dname)
                break  # One category per directory.

        # Prune matched subtrees so we don't recurse into them.
        dirnames[:] = [d for d in dirnames if d not in prune]

        # Check files against file_glob rules.
        for fname in filenames:
            for category, kind, pattern in ARTIFACT_RULES:
                if kind != "file_glob":
                    continue
                if not _matches_glob(fname, pattern):
                    continue
                full = dirpath / fname
                if full.is_symlink():
                    continue
                try:
                    size = full.stat().st_size
                except OSError:
                    size = 0
                result.artifacts.append(
                    Artifact(
                        path=full, category=category, size_bytes=size, is_dir=False
                    )
                )
                break

    result.scan_time_s = time.monotonic() - t0
    return result


def _looks_like_cargo_target(path: Path) -> bool:
    """Heuristic: is this a Rust build target/ directory?"""
    parent = path.parent
    if (parent / "Cargo.toml").exists():
        return True
    # Check for telltale subdirectories inside target/.
    try:
        children = {
            e.name for e in os.scandir(str(path)) if e.is_dir(follow_symlinks=False)
        }
    except OSError:
        return False
    return bool(children & {"debug", "release", "doc"})


def _looks_like_build_output(path: Path) -> bool:
    """Heuristic: is this a Python/JS/Rust build output directory?"""
    parent = path.parent
    indicators = (
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
    )
    return any((parent / f).exists() for f in indicators)


def _looks_like_dist_output(path: Path) -> bool:
    """Heuristic: is this a Python/JS dist output directory?"""
    parent = path.parent
    indicators = (
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
    )
    return any((parent / f).exists() for f in indicators)


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


def delete_artifact(artifact: Artifact, logger: logging.Logger) -> bool:
    """Delete a single artifact.  Returns True on success."""
    try:
        if artifact.is_dir:
            shutil.rmtree(str(artifact.path))
        else:
            os.remove(str(artifact.path))
        logger.info("DELETED: %s (%s)", artifact.path, human_size(artifact.size_bytes))
        return True
    except Exception as exc:
        logger.error("FAILED to delete %s: %s", artifact.path, exc)
        return False


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(
    results: list[ScanResult],
    execute: bool,
    deleted_count: int,
    deleted_bytes: int,
    logger: logging.Logger,
) -> None:
    """Print the summary report to stdout and the log."""
    separator = "=" * 80

    all_artifacts: list[Artifact] = []
    for r in results:
        all_artifacts.extend(r.artifacts)

    # Sort by size descending for readability.
    all_artifacts.sort(key=lambda a: a.size_bytes, reverse=True)

    total_bytes = sum(a.size_bytes for a in all_artifacts)
    total_count = len(all_artifacts)

    # Per-category breakdown.
    cat_stats: dict[str, tuple[int, int]] = {}  # category -> (count, bytes)
    for a in all_artifacts:
        count, nbytes = cat_stats.get(a.category, (0, 0))
        cat_stats[a.category] = (count + 1, nbytes + a.size_bytes)

    lines: list[str] = []
    lines.append("")
    lines.append(separator)
    lines.append("  BIZRA ARTIFACT CLEANER — SUMMARY")
    lines.append(separator)
    lines.append("")

    mode = "EXECUTE MODE" if execute else "DRY RUN (pass --execute to delete)"
    lines.append(f"  Mode:              {mode}")
    lines.append(f"  Artifacts found:   {total_count}")
    lines.append(f"  Recoverable space: {human_size(total_bytes)}")
    if execute:
        lines.append(f"  Deleted:           {deleted_count} / {total_count}")
        lines.append(f"  Space recovered:   {human_size(deleted_bytes)}")
    lines.append("")

    # Per-root summary.
    lines.append("  Per-root breakdown:")
    lines.append(f"  {'Root':<55} {'Items':>6} {'Size':>12} {'Time':>8}")
    lines.append(f"  {'-'*55} {'-'*6} {'-'*12} {'-'*8}")
    for r in results:
        root_bytes = sum(a.size_bytes for a in r.artifacts)
        status = r.error or ""
        time_str = f"{r.scan_time_s:.1f}s"
        if r.error:
            lines.append(f"  {str(r.root):<55} {'--':>6} {'--':>12} {status}")
        else:
            lines.append(
                f"  {str(r.root):<55} {len(r.artifacts):>6} {human_size(root_bytes):>12} {time_str:>8}"
            )
    lines.append("")

    # Per-category breakdown.
    lines.append("  Per-category breakdown:")
    lines.append(f"  {'Category':<30} {'Items':>6} {'Size':>14}")
    lines.append(f"  {'-'*30} {'-'*6} {'-'*14}")
    for cat, (count, nbytes) in sorted(
        cat_stats.items(), key=lambda x: x[1][1], reverse=True
    ):
        lines.append(f"  {cat:<30} {count:>6} {human_size(nbytes):>14}")
    lines.append("")

    # Top 20 largest artifacts.
    top_n = min(20, len(all_artifacts))
    if top_n > 0:
        lines.append(f"  Top {top_n} largest artifacts:")
        lines.append(f"  {'Size':>14}  {'Category':<28} Path")
        lines.append(f"  {'-'*14}  {'-'*28} {'-'*40}")
        for a in all_artifacts[:top_n]:
            lines.append(f"  {human_size(a.size_bytes):>14}  {a.category:<28} {a.path}")
        lines.append("")

    lines.append(separator)

    report = "\n".join(lines)
    # Print to console directly; log to file only (suppress console handler
    # to avoid duplicated output).
    print(report)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
            prev_level = handler.level
            handler.setLevel(logging.CRITICAL + 1)
            break
    else:
        handler = None  # type: ignore[assignment]
        prev_level = logging.INFO
    for line in lines:
        logger.info(line)
    if handler is not None:
        handler.setLevel(prev_level)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    """Configure file + console logging.  Returns the root logger."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"artifact_clean_{timestamp}.log"

    logger = logging.getLogger("artifact_cleaner")
    logger.setLevel(logging.DEBUG)
    # Prevent duplicate handlers if setup_logging is called more than once
    # (e.g. in tests or repeated invocations within the same process).
    logger.handlers.clear()

    # File handler — verbose.
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # Console handler — info only.
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info("Log file: %s", log_file)
    return logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BIZRA Artifact Cleaner — scan and remove build caches before migration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python artifact_cleaner.py                          # dry-run\n"
            '  python artifact_cleaner.py --execute --confirm "I understand this deletes files"\n'
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually delete artifacts (default: dry-run only).",
    )
    parser.add_argument(
        "--confirm",
        type=str,
        default="",
        help='Safety confirmation string. Must be exactly: "I understand this deletes files"',
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Parallel scan workers (default: {MAX_WORKERS}).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    logger = setup_logging()

    # Safety gate.
    if args.execute and args.confirm != "I understand this deletes files":
        print(
            'ERROR: --execute requires --confirm "I understand this deletes files"\n'
            "       This is a safety measure.  In dry-run mode, nothing is deleted.",
            file=sys.stderr,
        )
        return 1

    execute = args.execute and args.confirm == "I understand this deletes files"

    if execute:
        logger.info("MODE: EXECUTE — artifacts WILL be deleted.")
    else:
        logger.info("MODE: DRY RUN — nothing will be deleted.")

    # Filter to roots that actually exist (log missing ones).
    existing_roots: list[Path] = []
    for root in SCAN_ROOTS:
        if root.exists():
            existing_roots.append(root)
            logger.debug("Scan root found: %s", root)
        else:
            logger.info("Scan root missing (skipped): %s", root)

    if not existing_roots:
        logger.error("No scan roots found.  Nothing to do.")
        return 0

    # Parallel scan.
    logger.info(
        "Scanning %d root directories with %d workers ...",
        len(existing_roots),
        args.workers,
    )
    results: list[ScanResult] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(scan_root, root): root for root in existing_roots}
        for future in as_completed(future_map):
            root = future_map[future]
            try:
                result = future.result()
                results.append(result)
                n = len(result.artifacts)
                logger.info(
                    "  Scanned %s — %d artifacts found (%.1fs)",
                    root.name,
                    n,
                    result.scan_time_s,
                )
            except Exception as exc:
                logger.error("  FAILED scanning %s: %s", root, exc)
                results.append(ScanResult(root=root, error=str(exc)))

    # Collect all artifacts.
    all_artifacts: list[Artifact] = []
    for r in results:
        all_artifacts.extend(r.artifacts)

    # Detailed listing (always logged, console only in dry-run).
    if all_artifacts:
        all_artifacts_sorted = sorted(
            all_artifacts, key=lambda a: a.size_bytes, reverse=True
        )
        logger.info("")
        logger.info("Artifacts discovered:")
        for a in all_artifacts_sorted:
            kind = "DIR " if a.is_dir else "FILE"
            logger.info(
                "  [%s] %14s  %-28s  %s",
                kind,
                human_size(a.size_bytes),
                a.category,
                a.path,
            )

    # Execute deletions if requested.
    deleted_count = 0
    deleted_bytes = 0
    if execute and all_artifacts:
        logger.info("")
        logger.info("Deleting %d artifacts ...", len(all_artifacts))
        # Delete largest first so we free maximum space quickly.
        for a in sorted(all_artifacts, key=lambda a: a.size_bytes, reverse=True):
            if delete_artifact(a, logger):
                deleted_count += 1
                deleted_bytes += a.size_bytes

    # Summary report.
    print_report(results, execute, deleted_count, deleted_bytes, logger)

    # Return non-zero if execute was requested but nothing was deleted
    # despite artifacts being found (indicates all deletions failed).
    if execute and all_artifacts and deleted_count == 0:
        logger.error(
            "Execute mode: %d artifacts found but 0 deleted — all deletions failed.",
            len(all_artifacts),
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
