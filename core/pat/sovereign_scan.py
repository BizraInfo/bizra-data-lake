#!/usr/bin/env python3
"""PAT Collection Pipeline — Step 1: Sovereign Discovery Scan

Scans all known BIZRA locations, fingerprints every file,
classifies by type, and produces a discovery manifest.

NO files are moved or modified. Read-only scan.
"""

import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ═══════════════════════════════════════════════════════════
# SCAN TARGETS — all known BIZRA locations
# ═══════════════════════════════════════════════════════════
SCAN_ROOTS = [
    Path("/mnt/c/BIZRA-DATA-LAKE"),
    Path("/mnt/c/BIZRA-Dual-Agentic-system--main"),
    Path("/mnt/c/BIZRA-NODE0"),
    Path("/mnt/c/BIZRA-NODE0/BIZRA-SC-UNIFICATION"),
    Path("/mnt/c/BIZRA-PROJECTS"),
    Path("/mnt/c/BIZRA-TaskMaster"),
    Path("/mnt/c/Users/BIZRA-OS/Downloads"),
    Path("/mnt/b/BIZRA-SOVEREIGN"),
]

# Skip directories that are not work product
SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    ".venv-linux",
    ".venv-apex",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    ".hypothesis",
    ".benchmarks",
    "dist",
    ".next",
    ".swarm",
    ".claude-flow",
    ".fastembed_cache",
    "coverage",
}

# Skip binary extensions that are not work product
SKIP_EXTENSIONS = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".whl",
    ".pyc",
    ".pyo",
    ".o",
    ".a",
    ".lib",
    ".lock",
    ".idx",
    ".pack",
}


@dataclass(frozen=True)
class ScanLimits:
    """Explicit bounds for sovereign discovery scans."""

    max_depth: int = 12
    max_files: int = 250_000
    max_total_bytes: int = 200 * 1024 * 1024 * 1024
    timeout_seconds: float = 120.0
    follow_symlinks: bool = False


@dataclass
class ScanAudit:
    """Structured accounting for skipped files/directories."""

    directories_checked: int = 0
    files_checked: int = 0
    total_bytes: int = 0
    skipped: list[dict[str, str]] = field(default_factory=list)

    def skip(self, path: Path, reason: str) -> None:
        self.skipped.append({"path": str(path), "reason": reason})

    def as_dict(self) -> dict:
        return {
            "directories_checked": self.directories_checked,
            "files_checked": self.files_checked,
            "total_bytes": self.total_bytes,
            "skipped": list(self.skipped),
            "limit_hit": bool(self.skipped),
        }


def _depth_from_root(root: Path, current: Path) -> int:
    try:
        return len(current.relative_to(root).parts)
    except ValueError:
        return 0


def _deadline_expired(started_at: float, timeout_seconds: float) -> bool:
    return (time.perf_counter() - started_at) > timeout_seconds


# File classification by extension
CLASSIFY = {
    # Code
    ".py": "code/python",
    ".rs": "code/rust",
    ".ts": "code/typescript",
    ".tsx": "code/typescript",
    ".js": "code/javascript",
    ".jsx": "code/react",
    ".sol": "code/solidity",
    ".sh": "code/shell",
    ".ps1": "code/powershell",
    ".bat": "code/batch",
    ".sql": "code/sql",
    ".r": "code/r",
    # Documents
    ".md": "doc/markdown",
    ".txt": "doc/text",
    ".pdf": "doc/pdf",
    ".docx": "doc/word",
    ".pptx": "doc/presentation",
    ".xlsx": "doc/spreadsheet",
    # Research / Data
    ".json": "data/json",
    ".jsonl": "data/jsonl",
    ".yaml": "data/yaml",
    ".yml": "data/yaml",
    ".toml": "data/toml",
    ".csv": "data/csv",
    ".parquet": "data/parquet",
    ".db": "data/database",
    ".sqlite": "data/database",
    # Web
    ".html": "web/html",
    ".css": "web/css",
    ".svg": "web/svg",
    # Config
    ".env": "config/env",
    ".ini": "config/ini",
    ".cfg": "config/cfg",
    ".gitignore": "config/git",
    ".dockerignore": "config/docker",
    # Media
    ".png": "media/image",
    ".jpg": "media/image",
    ".jpeg": "media/image",
    ".gif": "media/image",
    ".webp": "media/image",
    ".mp4": "media/video",
    ".m4a": "media/audio",
    ".wav": "media/audio",
    # Archives
    ".zip": "archive/zip",
    ".tar": "archive/tar",
    ".gz": "archive/gzip",
    ".tgz": "archive/gzip",
}


def classify_file(path: Path) -> str:
    """Classify a file by its extension."""
    ext = path.suffix.lower()
    if ext in CLASSIFY:
        return CLASSIFY[ext]
    name = path.name.lower()
    if name in ("dockerfile", "makefile", "license", "readme"):
        return "config/build"
    if name.startswith("claude") or name == ".mcp.json":
        return "config/ai"
    return f"other/{ext.lstrip('.') or 'noext'}"


def fingerprint(path: Path, quick: bool = True) -> str:
    """BLAKE2b fingerprint. Quick mode reads first 64KB + last 64KB + size."""
    try:
        size = path.stat().st_size
        if quick and size > 131072:
            h = hashlib.blake2b(digest_size=16)
            h.update(str(size).encode())
            with open(path, "rb") as f:
                h.update(f.read(65536))
                f.seek(max(0, size - 65536))
                h.update(f.read(65536))
            return h.hexdigest()
        else:
            h = hashlib.blake2b(digest_size=16)
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
    except (PermissionError, OSError):
        return "error"


def scan_directory(
    root: Path,
    limits: ScanLimits | None = None,
    audit: ScanAudit | None = None,
    started_at: float | None = None,
) -> list[dict]:
    """Walk a directory tree and catalog every file."""
    limits = limits or ScanLimits()
    audit = audit or ScanAudit()
    started_at = started_at if started_at is not None else time.perf_counter()
    results = []
    if not root.exists():
        print(f"  [SKIP] {root} — not found")
        audit.skip(root, "missing_root")
        return results
    for dirpath, dirnames, filenames in os.walk(
        root, followlinks=limits.follow_symlinks
    ):
        if _deadline_expired(started_at, limits.timeout_seconds):
            audit.skip(Path(dirpath), "timeout")
            dirnames[:] = []
            break

        # Prune skipped directories
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        dp = Path(dirpath)
        audit.directories_checked += 1

        depth = _depth_from_root(root, dp)
        if depth >= limits.max_depth:
            for dirname in dirnames:
                audit.skip(dp / dirname, "max_depth")
            dirnames[:] = []

        if not limits.follow_symlinks:
            kept_dirnames = []
            for dirname in dirnames:
                candidate = dp / dirname
                if candidate.is_symlink():
                    audit.skip(candidate, "symlink_directory")
                else:
                    kept_dirnames.append(dirname)
            dirnames[:] = kept_dirnames

        for fname in filenames:
            fp = dp / fname
            if audit.files_checked >= limits.max_files:
                audit.skip(fp, "max_files")
                dirnames[:] = []
                break
            if fp.is_symlink() and not limits.follow_symlinks:
                audit.skip(fp, "symlink_file")
                continue
            ext = fp.suffix.lower()
            if ext in SKIP_EXTENSIONS:
                continue
            try:
                stat = fp.stat()
                if audit.total_bytes + stat.st_size > limits.max_total_bytes:
                    audit.skip(fp, "max_total_bytes")
                    dirnames[:] = []
                    break
                audit.files_checked += 1
                audit.total_bytes += stat.st_size
                results.append(
                    {
                        "path": str(fp),
                        "name": fname,
                        "ext": ext,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(
                            stat.st_mtime, tz=timezone.utc
                        ).isoformat(),
                        "kind": classify_file(fp),
                        "source_root": str(root),
                        "hash": "",  # Filled in dedup pass
                    }
                )
            except (PermissionError, OSError) as exc:
                audit.skip(fp, f"stat_error:{type(exc).__name__}")
                continue
    return results


def run_discovery_scan(
    output_dir: Path = Path("04_GOLD"),
    limits: ScanLimits | None = None,
) -> dict:
    """Run full discovery scan across all BIZRA locations."""
    limits = limits or ScanLimits()
    audit = ScanAudit()
    print("=" * 60)
    print("  PAT SOVEREIGN DISCOVERY SCAN")
    print("  Read-only. No files moved or modified.")
    print("=" * 60)
    print()

    all_files = []
    t0 = time.perf_counter()

    scan_started_at = time.perf_counter()
    for root in SCAN_ROOTS:
        if _deadline_expired(scan_started_at, limits.timeout_seconds):
            audit.skip(root, "timeout")
            break
        print(f"  Scanning: {root}")
        found = scan_directory(
            root,
            limits=limits,
            audit=audit,
            started_at=scan_started_at,
        )
        print(f"    → {len(found):,} files found")
        all_files.extend(found)
        if audit.files_checked >= limits.max_files:
            audit.skip(root, "max_files")
            break

    scan_time = time.perf_counter() - t0
    print(f"\n  Total files discovered: {len(all_files):,} in {scan_time:.1f}s")

    # ── Classification summary ──
    by_kind = defaultdict(lambda: {"count": 0, "bytes": 0})
    for f in all_files:
        cat = f["kind"].split("/")[0]
        by_kind[cat]["count"] += 1
        by_kind[cat]["bytes"] += f["size"]

    print("\n  === CLASSIFICATION ===")
    for cat in sorted(by_kind, key=lambda k: -by_kind[k]["bytes"]):
        info = by_kind[cat]
        gb = info["bytes"] / 1e9
        print(f"    {cat:12s}: {info['count']:>8,} files  {gb:>8.2f} GB")

    # ── Source distribution ──
    by_source = defaultdict(int)
    for f in all_files:
        by_source[f["source_root"]] += 1
    print("\n  === BY SOURCE ===")
    for src in sorted(by_source, key=lambda k: -by_source[k]):
        print(f"    {src}: {by_source[src]:,}")

    # ── Fingerprint pass (dedup candidates: same name+size) ──
    print("\n  === DEDUP SCAN ===")
    t1 = time.perf_counter()
    # Group by (name, size) for quick dedup candidates
    candidates = defaultdict(list)
    for i, f in enumerate(all_files):
        key = (f["name"], f["size"])
        candidates[key].append(i)

    # Only fingerprint files that share (name, size) with another
    dup_groups = {k: v for k, v in candidates.items() if len(v) > 1}
    fingerprint_count = sum(len(v) for v in dup_groups.values())
    print(
        f"    Candidate duplicates: {len(dup_groups):,} groups ({fingerprint_count:,} files)"
    )

    hashed = 0
    for key, indices in dup_groups.items():
        for idx in indices:
            fp = Path(all_files[idx]["path"])
            all_files[idx]["hash"] = fingerprint(fp)
            hashed += 1

    # Count actual duplicates (same hash)
    hash_groups = defaultdict(list)
    for f in all_files:
        if f["hash"] and f["hash"] != "error":
            hash_groups[f["hash"]].append(f["path"])
    actual_dups = {h: paths for h, paths in hash_groups.items() if len(paths) > 1}
    dup_file_count = sum(len(v) - 1 for v in actual_dups.values())
    dup_bytes = 0
    for h, paths in actual_dups.items():
        for p in paths[1:]:  # Count all but one copy
            try:
                dup_bytes += Path(p).stat().st_size
            except OSError:
                pass

    dedup_time = time.perf_counter() - t1
    print(f"    Fingerprinted: {hashed:,} files in {dedup_time:.1f}s")
    print(
        f"    Confirmed duplicates: {len(actual_dups):,} groups ({dup_file_count:,} redundant files)"
    )
    print(f"    Recoverable space: {dup_bytes / 1e6:.1f} MB")

    # ── Write discovery manifest ──
    manifest_path = output_dir / "discovery_manifest.json"
    total_bytes = sum(f["size"] for f in all_files)

    manifest = {
        "scan_id": f"SCAN_{int(time.time())}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scan_roots": [str(r) for r in SCAN_ROOTS],
        "total_files": len(all_files),
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / 1e9, 2),
        "classification": {
            k: v for k, v in sorted(by_kind.items(), key=lambda x: -x[1]["bytes"])
        },
        "by_source": dict(sorted(by_source.items(), key=lambda x: -x[1])),
        "duplicates": {
            "groups": len(actual_dups),
            "redundant_files": dup_file_count,
            "recoverable_bytes": dup_bytes,
            "top_duplicates": [
                {"hash": h, "count": len(paths), "paths": paths[:5]}
                for h, paths in sorted(actual_dups.items(), key=lambda x: -len(x[1]))[
                    :20
                ]
            ],
        },
        "scan_duration_s": round(time.perf_counter() - t0, 1),
        "scan_limits": asdict(limits),
        "scan_audit": audit.as_dict(),
    }

    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2, default=str)
    print(f"\n  Manifest written: {manifest_path}")

    # ── Write full file list (for PAT to process) ──
    filelist_path = output_dir / "discovery_filelist.jsonl"
    with open(filelist_path, "w") as fp:
        for f in all_files:
            fp.write(json.dumps(f, default=str) + "\n")
    print(f"  File list written: {filelist_path} ({len(all_files):,} entries)")

    # ── Summary ──
    total_elapsed = time.perf_counter() - t0
    print(f"\n  {'='*50}")
    print("  DISCOVERY COMPLETE")
    print(f"  {'='*50}")
    print(f"  Files:     {len(all_files):,}")
    print(f"  Size:      {total_bytes / 1e9:.2f} GB")
    print(f"  Duplicates:{dup_file_count:,} redundant ({dup_bytes / 1e6:.1f} MB)")
    print(f"  Sources:   {len(by_source)}")
    print(f"  Time:      {total_elapsed:.1f}s")
    print("  Next step: PAT reviews manifest → clean → index → consolidate to B:\\")

    return manifest


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("04_GOLD")
    run_discovery_scan(out)
