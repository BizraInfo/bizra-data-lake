# Phase 53.1: SHA-256 Deduplication Scanner

**Status:** SPEC DRAFT | **Script:** `scripts/migration/dedup_scanner.py`
**Giants:** Shannon (information entropy -- identical hashes = zero information gain), Merkle (hash trees for structural integrity)

---

## Purpose

Walk all BIZRA directories on C:\, compute SHA-256 for every file using streaming
chunked reads, build a manifest DataFrame, and detect duplicates. Output a Parquet
manifest and duplicate log to `06_INDEX/`.

## Data Flow

```
  C:\BIZRA-DATA-LAKE\  --+
  C:\BIZRA-NODE0\     --+
  C:\BIZRA-PROJECTS\  --+---> walk_source() ---> hash_file() ---> manifest.parquet
  C:\BIZRA-Dual-*\    --+     (skip patterns)    (8MB chunks)     duplicates.log
  C:\bizra-genesis-*\ --+                        (multiprocessing) scan_receipt.json
  C:\bizra-voice\     --+
```

## Skip Patterns

Regenerable content excluded from hashing:
`.git/objects`, `.git/lfs`, `target/debug`, `target/release`, `target/doc`,
`node_modules`, `__pycache__`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`,
`.venv`, `.venv-linux`, `.venv-wsl`, `htmlcov`, `.tox`, `.nox`, `dist/`, `*.egg-info`

## Pseudocode

```python
"""scripts/migration/dedup_scanner.py -- SHA-256 deduplication scanner."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm

BIZRA_SOVEREIGN_ROOT: str = os.environ.get("BIZRA_SOVEREIGN_ROOT", "/mnt/b/BIZRA")
CHUNK_SIZE: int = 8 * 1024 * 1024  # 8 MB streaming chunks
DEFAULT_WORKERS: int = min(16, os.cpu_count() or 4)

DEFAULT_SOURCES: list[str] = [
    "/mnt/c/BIZRA-DATA-LAKE", "/mnt/c/BIZRA-NODE0", "/mnt/c/BIZRA-PROJECTS",
    "/mnt/c/BIZRA-Dual-Agentic-system--main", "/mnt/c/bizra-genesis-node",
    "/mnt/c/bizra-genesis-node-backup", "/mnt/c/bizra-genesis-node-fresh",
    "/mnt/c/bizra-genesis-node-repaired", "/mnt/c/bizra-voice",
]

SKIP_PATTERNS: list[str] = [
    ".git/objects", ".git/lfs", "target/debug", "target/release", "target/doc",
    "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".venv", ".venv-linux", ".venv-wsl", "htmlcov", ".tox", ".nox", "dist/", "*.egg-info",
]

@dataclass
class FileRecord:
    path: str; sha256: str; size_bytes: int; mtime: float
    extension: str; category: str; source_root: str

@dataclass
class ScanResult:
    records: list[FileRecord] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    total_bytes_scanned: int = 0
    scan_start: str = ""; scan_end: str = ""

def should_skip(path: Path) -> bool:
    """Check if path matches any skip pattern (containment or fnmatch glob)."""
    path_str = str(path)
    for pattern in SKIP_PATTERNS:
        if "*" in pattern:
            import fnmatch
            if any(fnmatch.fnmatch(part, pattern) for part in path.parts):
                return True
        elif pattern.replace("/", os.sep) in path_str:
            return True
    return False

def hash_file(file_path: str) -> tuple[str, str, int, float, Optional[str]]:
    """Stream-hash a file. Returns (path, sha256, size, mtime, error_or_none)."""
    h = hashlib.sha256()
    try:
        stat = os.stat(file_path)
        with open(file_path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                h.update(chunk)
        return (file_path, h.hexdigest(), stat.st_size, stat.st_mtime, None)
    except (OSError, PermissionError) as e:
        return (file_path, "", 0, 0.0, str(e))

def classify_extension(ext: str) -> str:
    """Classify file extension into: code, data, media, document, config, binary, unknown."""
    ext = ext.lower()
    MAP = {
        "code": {".py",".rs",".ts",".tsx",".js",".jsx",".java",".go",".c",".cpp",
                 ".h",".sh",".bat",".ps1",".sql",".toml"},
        "data": {".parquet",".csv",".json",".jsonl",".yaml",".yml",".xml",".arrow",
                 ".sqlite",".db",".pkl"},
        "media": {".png",".jpg",".jpeg",".gif",".svg",".mp4",".webm",".mp3",".wav",
                  ".ogg",".flac",".webp",".ico",".bmp",".avi",".mkv",".mov"},
        "document": {".md",".txt",".pdf",".docx",".xlsx",".pptx",".html",".htm",".rst"},
        "config": {".env",".ini",".cfg",".conf",".lock",".gitignore",".dockerignore"},
        "binary": {".exe",".dll",".so",".dylib",".wasm",".bin",".o",".a",".lib"},
    }
    for cat, exts in MAP.items():
        if ext in exts:
            return cat
    return "unknown"

def walk_source(source_root: str) -> list[str]:
    """Walk directory, returning file paths that pass skip filters."""
    files: list[str] = []
    if not Path(source_root).exists():
        return files
    for dirpath, dirnames, filenames in os.walk(source_root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [d for d in dirnames if not should_skip(current / d)]
        for fname in filenames:
            full_path = current / fname
            if not should_skip(full_path):
                files.append(str(full_path))
    return files

def scan_sources(sources: list[str], workers: int = DEFAULT_WORKERS) -> ScanResult:
    """Walk all sources and hash files in parallel with progress bar."""
    result = ScanResult(scan_start=datetime.now(timezone.utc).isoformat())
    all_files: list[tuple[str, str]] = []
    for source in sources:
        paths = walk_source(source)
        all_files.extend((p, source) for p in paths)
    source_lookup = {fp: sr for fp, sr in all_files}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(hash_file, fp): fp for fp, _ in all_files}
        with tqdm(total=len(futures), desc="Hashing", unit="files") as pbar:
            for future in as_completed(futures):
                fp, sha256, size, mtime, error = future.result()
                if error:
                    result.errors.append({"path": fp, "error": error})
                else:
                    ext = Path(fp).suffix
                    result.records.append(FileRecord(
                        fp, sha256, size, mtime, ext,
                        classify_extension(ext), source_lookup[fp]))
                    result.total_bytes_scanned += size
                pbar.update(1)
    result.scan_end = datetime.now(timezone.utc).isoformat()
    return result

def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of files sharing a SHA-256 hash (groups with >1 member)."""
    counts = df.groupby("sha256").size()
    dup_hashes = counts[counts > 1].index
    return df[df["sha256"].isin(dup_hashes)].sort_values(["sha256", "size_bytes"])

def build_manifest(result: ScanResult) -> pd.DataFrame:
    """Convert ScanResult to DataFrame."""
    cols = ["path", "sha256", "size_bytes", "mtime", "extension", "category", "source_root"]
    if not result.records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([{c: getattr(r, c) for c in cols} for r in result.records])

def write_outputs(df, duplicates, result, output_dir: str) -> None:
    """Write manifest.parquet, duplicates.log, scan_receipt.json."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "manifest.parquet", index=False, engine="pyarrow")
    with open(out / "duplicates.log", "w") as f:
        for sha, group in duplicates.groupby("sha256"):
            waste = group["size_bytes"].sum() - group["size_bytes"].iloc[0]
            f.write(f"\n--- {sha} ({len(group)} copies, waste: {waste/1024**2:.1f} MB) ---\n")
            for _, row in group.iterrows():
                f.write(f"  {row['size_bytes']:>12,} B  {row['path']}\n")
    receipt = {"operation": "dedup_scan", "total_files": len(df),
               "total_bytes": int(result.total_bytes_scanned),
               "duplicate_groups": int(duplicates["sha256"].nunique()),
               "errors": len(result.errors)}
    with open(out / "scan_receipt.json", "w") as f:
        json.dump(receipt, f, indent=2)

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 53.1: Dedup Scanner")
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--output-dir", default=os.path.join(BIZRA_SOVEREIGN_ROOT, "06_INDEX"))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    result = scan_sources(args.sources, workers=args.workers)
    df = build_manifest(result)
    duplicates = detect_duplicates(df)
    write_outputs(df, duplicates, result, args.output_dir)
    print(f"Scanned {len(df):,} files ({result.total_bytes_scanned/1024**3:.1f} GB), "
          f"{duplicates['sha256'].nunique():,} dup groups")

if __name__ == "__main__":
    main()
```

## Performance Notes

- **I/O bound:** Cap workers at 16 to avoid saturating WSL/NTFS I/O bus.
- **Streaming:** 8 MB chunks = constant memory regardless of file size.
- **Skip pruning:** `dirnames[:] = ...` prevents descent into skipped dirs (avoids 85 GB target/).
- **ProcessPoolExecutor:** True parallelism for hashing (hashlib releases GIL).

## TDD Anchors

```python
"""tests/migration/test_dedup_scanner.py"""
import hashlib, os
from pathlib import Path
import pandas as pd
import pytest

class TestHashFile:
    def test_hash_single_file(self, tmp_path: Path) -> None:
        content = b"bismillah ar-rahman ar-raheem"
        f = tmp_path / "test.txt"; f.write_bytes(content)
        _, sha, size, mtime, error = hash_file(str(f))
        assert error is None and sha == hashlib.sha256(content).hexdigest()
        assert size == len(content) and mtime > 0

    def test_hash_large_file(self, tmp_path: Path) -> None:
        content = os.urandom(10 * 1024 * 1024)  # 10 MB > CHUNK_SIZE
        f = tmp_path / "large.bin"; f.write_bytes(content)
        _, sha, size, _, error = hash_file(str(f))
        assert error is None and sha == hashlib.sha256(content).hexdigest()

    def test_hash_nonexistent(self) -> None:
        _, sha, _, _, error = hash_file("/nonexistent/file.txt")
        assert error is not None and sha == ""

    def test_hash_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"; f.write_bytes(b"")
        _, sha, size, _, error = hash_file(str(f))
        assert error is None and sha == hashlib.sha256(b"").hexdigest() and size == 0

class TestSkipPatterns:
    def test_skip_git_objects(self) -> None:
        assert should_skip(Path("/repo/.git/objects/ab/cdef")) is True
    def test_skip_target_debug(self) -> None:
        assert should_skip(Path("/repo/target/debug/build")) is True
    def test_skip_node_modules(self) -> None:
        assert should_skip(Path("/repo/node_modules/lodash/index.js")) is True
    def test_no_skip_normal_file(self) -> None:
        assert should_skip(Path("/repo/core/main.py")) is False
    def test_no_skip_git_head(self) -> None:
        assert should_skip(Path("/repo/.git/HEAD")) is False

class TestDuplicateDetection:
    def test_duplicate_detection(self) -> None:
        df = pd.DataFrame({"path": ["/a.txt","/b.txt","/c.txt"],
                           "sha256": ["aaa","aaa","bbb"], "size_bytes": [100,100,200]})
        assert len(detect_duplicates(df)) == 2

    def test_no_duplicates(self) -> None:
        df = pd.DataFrame({"path": ["/a.txt","/b.txt"],
                           "sha256": ["aaa","bbb"], "size_bytes": [100,200]})
        assert len(detect_duplicates(df)) == 0

class TestManifestSchema:
    def test_manifest_has_columns(self) -> None:
        result = ScanResult(records=[FileRecord("/a.py","abc",100,1.0,".py","code","/src")])
        df = build_manifest(result)
        assert {"path","sha256","size_bytes","category"}.issubset(set(df.columns))

    def test_manifest_empty(self) -> None:
        assert len(build_manifest(ScanResult())) == 0

class TestMultiprocessing:
    def test_parallel_scan(self, tmp_path: Path) -> None:
        for i in range(5): (tmp_path / f"f{i}.txt").write_text(f"c{i}")
        result = scan_sources([str(tmp_path)], workers=2)
        assert len(result.records) == 5

class TestClassifyExtension:
    def test_python_is_code(self) -> None: assert classify_extension(".py") == "code"
    def test_parquet_is_data(self) -> None: assert classify_extension(".parquet") == "data"
    def test_png_is_media(self) -> None: assert classify_extension(".png") == "media"
    def test_unknown(self) -> None: assert classify_extension(".xyz") == "unknown"
```
