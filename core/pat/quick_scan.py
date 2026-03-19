#!/usr/bin/env python3
"""PAT Quick Scan — fast file census across all BIZRA locations."""

import os, sys, time, json, hashlib
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

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
    ".tmp_prod_artifacts_v2",
    ".codex",
}
SKIP_EXT = {
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

ROOTS = [
    "/mnt/c/BIZRA-DATA-LAKE",
    "/mnt/c/BIZRA-Dual-Agentic-system--main",
    "/mnt/c/BIZRA-NODE0",
    "/mnt/c/Users/BIZRA-OS/Downloads",
    "/mnt/b/BIZRA-SOVEREIGN",
]

CLASSIFY = {
    ".py": "code",
    ".rs": "code",
    ".ts": "code",
    ".tsx": "code",
    ".js": "code",
    ".jsx": "code",
    ".sh": "code",
    ".ps1": "code",
    ".bat": "code",
    ".sql": "code",
    ".md": "doc",
    ".txt": "doc",
    ".pdf": "doc",
    ".docx": "doc",
    ".pptx": "doc",
    ".xlsx": "doc",
    ".json": "data",
    ".jsonl": "data",
    ".yaml": "data",
    ".yml": "data",
    ".toml": "data",
    ".csv": "data",
    ".parquet": "data",
    ".db": "data",
    ".html": "web",
    ".css": "web",
    ".svg": "web",
    ".png": "media",
    ".jpg": "media",
    ".jpeg": "media",
    ".gif": "media",
    ".webp": "media",
    ".mp4": "media",
    ".m4a": "media",
    ".zip": "archive",
    ".tar": "archive",
    ".gz": "archive",
    ".tgz": "archive",
}


def scan_root(root):
    rp = Path(root)
    if not rp.exists():
        return None
    t0 = time.time()
    files = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            fp = Path(dp) / fn
            ext = fp.suffix.lower()
            if ext in SKIP_EXT:
                continue
            try:
                st = fp.stat()
                files.append(
                    {
                        "path": str(fp),
                        "name": fn,
                        "ext": ext,
                        "size": st.st_size,
                        "mtime": st.st_mtime,
                        "kind": CLASSIFY.get(ext, "other"),
                        "root": root,
                    }
                )
            except (PermissionError, OSError):
                pass
    return {"root": root, "count": len(files), "files": files, "time": time.time() - t0}


def main():
    print("=" * 60)
    print("  PAT SOVEREIGN DISCOVERY — QUICK SCAN")
    print("  Read-only census across all BIZRA locations")
    print("=" * 60)
    print(flush=True)

    all_files = []
    by_source = {}
    t0 = time.time()

    for root in ROOTS:
        print(f"  Scanning: {root} ...", end=" ", flush=True)
        result = scan_root(root)
        if result is None:
            print("NOT FOUND")
            continue
        print(f"{result['count']:,} files ({result['time']:.1f}s)")
        all_files.extend(result["files"])
        by_source[root] = result["count"]

    total_time = time.time() - t0
    total_bytes = sum(f["size"] for f in all_files)
    print(
        f"\n  TOTAL: {len(all_files):,} files, {total_bytes/1e9:.2f} GB in {total_time:.1f}s"
    )

    # Classification
    by_kind = defaultdict(lambda: {"count": 0, "bytes": 0})
    by_ext = defaultdict(lambda: {"count": 0, "bytes": 0})
    for f in all_files:
        by_kind[f["kind"]]["count"] += 1
        by_kind[f["kind"]]["bytes"] += f["size"]
        by_ext[f["ext"]]["count"] += 1
        by_ext[f["ext"]]["bytes"] += f["size"]

    print("\n  === BY KIND ===")
    for k in sorted(by_kind, key=lambda x: -by_kind[x]["bytes"]):
        i = by_kind[k]
        print(f"    {k:10s}: {i['count']:>8,} files  {i['bytes']/1e9:>8.2f} GB")

    print("\n  === TOP EXTENSIONS BY COUNT ===")
    for ext, i in sorted(by_ext.items(), key=lambda x: -x[1]["count"])[:15]:
        print(
            f"    {ext or 'noext':8s}: {i['count']:>8,} files  {i['bytes']/1e6:>10.1f} MB"
        )

    # Quick dedup: group by (name, size)
    print("\n  === DEDUP CANDIDATES ===")
    by_key = defaultdict(list)
    for f in all_files:
        by_key[(f["name"], f["size"])].append(f["path"])
    dup_groups = {k: v for k, v in by_key.items() if len(v) > 1}
    dup_files = sum(len(v) - 1 for v in dup_groups.values())
    dup_bytes = sum(k[1] * (len(v) - 1) for k, v in dup_groups.items())
    print(f"    Groups with same name+size: {len(dup_groups):,}")
    print(f"    Redundant copies:           {dup_files:,}")
    print(f"    Recoverable space:          {dup_bytes/1e6:.1f} MB")

    # Top dups by wasted space
    print("\n  === TOP 10 DUPLICATES BY WASTED SPACE ===")
    top_dups = sorted(dup_groups.items(), key=lambda x: -x[0][1] * (len(x[1]) - 1))[:10]
    for (name, size), paths in top_dups:
        wasted = size * (len(paths) - 1)
        print(
            f"    {name[:40]:40s} {size/1e6:>8.1f} MB x{len(paths)} = {wasted/1e6:.1f} MB wasted"
        )
        for p in paths[:3]:
            print(f"      {p}")
        if len(paths) > 3:
            print(f"      ... +{len(paths)-3} more")

    # Write discovery manifest
    manifest = {
        "scan_id": f"QSCAN_{int(time.time())}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "roots_scanned": list(by_source.keys()),
        "roots_missing": [r for r in ROOTS if r not in by_source],
        "total_files": len(all_files),
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / 1e9, 2),
        "by_kind": {
            k: dict(v) for k, v in sorted(by_kind.items(), key=lambda x: -x[1]["bytes"])
        },
        "by_source": by_source,
        "duplicates": {
            "groups": len(dup_groups),
            "redundant_files": dup_files,
            "recoverable_bytes": dup_bytes,
        },
        "scan_duration_s": round(total_time, 1),
    }
    mpath = Path("04_GOLD/discovery_manifest.json")
    with open(mpath, "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"\n  Manifest: {mpath}")

    # Write full file list
    fpath = Path("04_GOLD/discovery_filelist.jsonl")
    with open(fpath, "w") as fp:
        for f in all_files:
            fp.write(json.dumps(f, default=str) + "\n")
    print(f"  File list: {fpath} ({len(all_files):,} entries)")

    print(f"\n  {'='*50}")
    print(f"  SCAN COMPLETE")
    print(f"  {'='*50}")
    print(f"  Files:       {len(all_files):,}")
    print(f"  Size:        {total_bytes/1e9:.2f} GB")
    print(f"  Duplicates:  {dup_files:,} redundant ({dup_bytes/1e6:.1f} MB)")
    print(f"  Sources:     {len(by_source)}")
    print(f"  Time:        {total_time:.1f}s")
    print(f"  Next: PAT clean → index → dedup → consolidate to B:\\")


if __name__ == "__main__":
    main()
