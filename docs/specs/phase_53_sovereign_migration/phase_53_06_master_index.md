# Phase 53.6: Master Index Generation

**Status:** SPEC DRAFT | **Script:** `scripts/migration/master_index.py`
**Giants:** Shannon (index = channel -- without it, data lake has max entropy, zero retrievability), Codd (relational normal forms applied to file metadata)

---

## Purpose

Generate the authoritative master index for B:\BIZRA after migration. Enables
search ("where is my Rust API?"), timeline ("what did I work on in March 2024?"),
integrity verification, and statistics. The final seal on Phase 53.

## Outputs

| File | Format | Purpose |
|------|--------|---------|
| `06_INDEX/manifest.parquet` | Parquet | Complete file inventory with SHA-256 |
| `06_INDEX/timeline.jsonl` | JSONL | Files mapped to dates for temporal queries |
| `06_INDEX/stats_report.json` | JSON | Aggregate statistics |
| `06_INDEX/integrity_report.json` | JSON | Disk vs manifest comparison |

## Manifest Schema

```
sha256, path (relative), size_bytes, mtime, ctime, extension, category,
bizra_project, source_origin, migration_date, top_level_dir
```

## Data Flow

```
  B:\BIZRA\ ---> walk_tree() ---> hash_files() ---> manifest.parquet
  (all dirs)     (skip 06_INDEX)   (parallel 16w)      |
                                                   +---+---+---+
                                                   |       |       |
                                              timeline  stats  integrity
                                              .jsonl    .json  _report.json
```

## Pseudocode

```python
"""scripts/migration/master_index.py -- Master index generator."""
from __future__ import annotations
import argparse, hashlib, json, os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm

BIZRA_SOVEREIGN_ROOT = os.environ.get("BIZRA_SOVEREIGN_ROOT", "/mnt/b/BIZRA")
INDEX_DIR = os.path.join(BIZRA_SOVEREIGN_ROOT, "06_INDEX")
CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_WORKERS = min(16, os.cpu_count() or 4)
INDEX_SKIP = {"06_INDEX", ".git"}

def walk_bizra_tree(root: str = BIZRA_SOVEREIGN_ROOT) -> list[str]:
    files = []
    for dp, dns, fns in os.walk(root, followlinks=False):
        dns[:] = [d for d in dns if d not in INDEX_SKIP]
        files.extend(os.path.join(dp, f) for f in fns)
    return files

def _classify_ext(ext: str) -> str:
    CODE = {".py",".rs",".ts",".tsx",".js",".jsx",".go",".java",".c",".cpp",".sh",".bat",".ps1",".toml"}
    DATA = {".parquet",".csv",".json",".jsonl",".yaml",".yml",".xml",".arrow",".sqlite"}
    MEDIA = {".png",".jpg",".jpeg",".gif",".svg",".mp4",".webm",".mp3",".wav",".ogg",".webp",".mov"}
    DOC = {".md",".txt",".pdf",".docx",".xlsx",".pptx",".html",".htm"}
    CONFIG = {".env",".ini",".cfg",".conf",".lock",".gitignore"}
    if ext in CODE: return "code"
    if ext in DATA: return "data"
    if ext in MEDIA: return "media"
    if ext in DOC: return "document"
    if ext in CONFIG: return "config"
    return "other"

def _detect_project(rel: str) -> str:
    parts = Path(rel).parts
    if len(parts) >= 2 and parts[0] in ("01_CORE", "03_ASSETS"): return parts[1]
    return "general"

def _find_marker_field(file_path: str, root: str, field: str) -> str:
    """Walk up from file looking for .MIGRATED_TO_B marker."""
    cur = Path(file_path).parent
    while str(cur).startswith(root) and cur != Path(root):
        m = cur / ".MIGRATED_TO_B"
        if m.exists():
            try: return json.loads(m.read_text()).get(field, "unknown")
            except Exception: pass
        cur = cur.parent
    return "native" if field == "migrated_to" else "pre-existing"

def hash_file_with_meta(file_path: str, root: str) -> Optional[dict]:
    """Hash file and extract all manifest fields."""
    try:
        stat = os.stat(file_path)
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE): h.update(chunk)
        rel = os.path.relpath(file_path, root)
        ext = Path(file_path).suffix.lower()
        parts = Path(rel).parts
        return {"sha256": h.hexdigest(), "path": rel, "size_bytes": stat.st_size,
                "mtime": stat.st_mtime, "ctime": stat.st_ctime, "extension": ext,
                "category": _classify_ext(ext), "bizra_project": _detect_project(rel),
                "source_origin": _find_marker_field(file_path, root, "migrated_to"),
                "migration_date": _find_marker_field(file_path, root, "timestamp"),
                "top_level_dir": parts[0] if parts else ""}
    except (OSError, PermissionError): return None

def generate_manifest(root: str = BIZRA_SOVEREIGN_ROOT, workers: int = DEFAULT_WORKERS,
                      existing: Optional[pd.DataFrame] = None,
                      incremental: bool = False) -> pd.DataFrame:
    files = walk_bizra_tree(root)
    # Incremental: skip unchanged files
    unchanged = []
    if incremental and existing is not None:
        mtimes = {os.path.join(root, r["path"]): r.get("mtime", 0) for _, r in existing.iterrows()}
        modified = []
        for fp in files:
            try:
                if fp in mtimes and os.path.getmtime(fp) <= mtimes[fp]:
                    rel = os.path.relpath(fp, root)
                    match = existing[existing["path"] == rel]
                    if not match.empty: unchanged.append(match.iloc[0].to_dict()); continue
            except OSError: pass
            modified.append(fp)
        files = modified
    entries = list(unchanged)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(hash_file_with_meta, fp, root): fp for fp in files}
        with tqdm(total=len(futs), desc="Indexing", unit="files") as pbar:
            for fut in as_completed(futs):
                r = fut.result()
                if r: entries.append(r)
                pbar.update(1)
    return pd.DataFrame(entries)

def generate_timeline(df: pd.DataFrame, output: str) -> None:
    with open(output, "w") as f:
        for _, row in df.iterrows():
            if not row.get("ctime"): continue
            cdate = datetime.fromtimestamp(row["ctime"], tz=timezone.utc).strftime("%Y-%m-%d")
            f.write(json.dumps({"date": cdate, "path": row["path"],
                                "category": row.get("category",""), "project": row.get("bizra_project",""),
                                "action": "created", "size_bytes": int(row.get("size_bytes",0))}) + "\n")
            if row.get("mtime"):
                mdate = datetime.fromtimestamp(row["mtime"], tz=timezone.utc).strftime("%Y-%m-%d")
                if mdate != cdate:
                    f.write(json.dumps({"date": mdate, "path": row["path"],
                                        "category": row.get("category",""),
                                        "project": row.get("bizra_project",""),
                                        "action": "modified", "size_bytes": int(row.get("size_bytes",0))}) + "\n")

def generate_stats(df: pd.DataFrame, output: str) -> dict:
    stats = {"generated_at": datetime.now(timezone.utc).isoformat(),
             "total_files": len(df), "total_size_bytes": int(df["size_bytes"].sum()),
             "total_size_human": f"{df['size_bytes'].sum()/1024**3:.2f} GB"}
    for group_col, key in [("category","by_category"),("bizra_project","by_project"),
                            ("top_level_dir","by_directory")]:
        if group_col in df.columns:
            g = df.groupby(group_col).agg(count=("path","count"),total=("size_bytes","sum")).to_dict("index")
            stats[key] = {k: {"count": v["count"], "size_gb": round(v["total"]/1024**3,2)} for k,v in g.items()}
    if "mtime" in df.columns:
        dated = df[df["mtime"] > 0].copy()
        dated["ym"] = dated["mtime"].apply(lambda t: datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m"))
        stats["by_month"] = dict(sorted(dated.groupby("ym").size().to_dict().items()))
    if "extension" in df.columns:
        stats["top_extensions"] = dict(df["extension"].value_counts().head(20))
    with open(output, "w") as f: json.dump(stats, f, indent=2)
    return stats

def check_integrity(df: pd.DataFrame, root: str = BIZRA_SOVEREIGN_ROOT) -> dict:
    disk = {os.path.relpath(fp, root) for fp in walk_bizra_tree(root)}
    manifest = set(df["path"].tolist()) if "path" in df.columns else set()
    matched = len(disk & manifest)
    total = len(disk | manifest)
    score = matched / total if total else 1.0
    return {"files_on_disk": len(disk), "files_in_manifest": len(manifest),
            "matched": matched, "on_disk_not_in_manifest": len(disk - manifest),
            "in_manifest_not_on_disk": len(manifest - disk),
            "integrity_score": round(score, 4), "ihsan_compliant": score >= 0.95,
            "missing_from_manifest": sorted(list(disk - manifest))[:50],
            "missing_from_disk": sorted(list(manifest - disk))[:50]}

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 53.6: Master Index")
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--integrity-only", action="store_true")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    os.makedirs(INDEX_DIR, exist_ok=True)
    mp = os.path.join(INDEX_DIR, "manifest.parquet")
    existing = pd.read_parquet(mp) if os.path.exists(mp) else None
    if args.integrity_only:
        if existing is not None:
            r = check_integrity(existing)
            with open(os.path.join(INDEX_DIR, "integrity_report.json"), "w") as f: json.dump(r, f, indent=2)
            print(f"Integrity: {r['integrity_score']:.4f} ({'PASS' if r['ihsan_compliant'] else 'FAIL'})")
        return
    if args.stats_only:
        if existing is not None: generate_stats(existing, os.path.join(INDEX_DIR, "stats_report.json"))
        return
    df = generate_manifest(workers=args.workers, existing=existing, incremental=args.incremental)
    df.to_parquet(mp, index=False, engine="pyarrow")
    generate_timeline(df, os.path.join(INDEX_DIR, "timeline.jsonl"))
    stats = generate_stats(df, os.path.join(INDEX_DIR, "stats_report.json"))
    report = check_integrity(df)
    with open(os.path.join(INDEX_DIR, "integrity_report.json"), "w") as f: json.dump(report, f, indent=2)
    print(f"Indexed {len(df):,} files ({stats['total_size_human']}), "
          f"integrity={report['integrity_score']:.4f}")

if __name__ == "__main__":
    main()
```

## TDD Anchors

```python
"""tests/migration/test_master_index.py"""
import json, os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import pytest

class TestManifestSchema:
    def test_has_required_columns(self, tmp_path: Path) -> None:
        (tmp_path / "01_CORE" / "data-lake").mkdir(parents=True)
        (tmp_path / "01_CORE" / "data-lake" / "main.py").write_text("print('hello')")
        df = generate_manifest(root=str(tmp_path), workers=1)
        required = {"sha256","path","size_bytes","mtime","ctime","extension",
                     "category","bizra_project","source_origin","migration_date","top_level_dir"}
        assert required.issubset(set(df.columns)) and len(df) == 1

    def test_empty_tree(self, tmp_path: Path) -> None:
        assert len(generate_manifest(root=str(tmp_path), workers=1)) == 0

class TestTimelineGeneration:
    def test_produces_events(self, tmp_path: Path) -> None:
        now = datetime.now(timezone.utc).timestamp()
        df = pd.DataFrame([{"path": "a.py", "category": "code", "bizra_project": "data-lake",
                            "size_bytes": 100, "mtime": now, "ctime": now - 86400}])
        out = str(tmp_path / "timeline.jsonl")
        generate_timeline(df, out)
        lines = open(out).readlines()
        assert len(lines) >= 1
        e = json.loads(lines[0])
        assert "date" in e and "action" in e

class TestIncrementalUpdate:
    def test_reuses_unchanged(self, tmp_path: Path) -> None:
        (tmp_path / "stable.txt").write_text("unchanged")
        df1 = generate_manifest(root=str(tmp_path), workers=1)
        df2 = generate_manifest(root=str(tmp_path), workers=1, existing=df1, incremental=True)
        assert len(df2) == 1 and df2.iloc[0]["sha256"] == df1.iloc[0]["sha256"]

class TestIntegrityCheck:
    def test_passes_when_matched(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("x")
        df = generate_manifest(root=str(tmp_path), workers=1)
        r = check_integrity(df, root=str(tmp_path))
        assert r["integrity_score"] == 1.0 and r["ihsan_compliant"] is True

    def test_detects_missing(self, tmp_path: Path) -> None:
        df = pd.DataFrame([{"path": "ghost.txt", "sha256": "x", "size_bytes": 0,
                            "mtime": 0, "ctime": 0, "extension": ".txt", "category": "doc",
                            "bizra_project": "gen", "source_origin": "n", "migration_date": "x",
                            "top_level_dir": ""}])
        r = check_integrity(df, root=str(tmp_path))
        assert r["in_manifest_not_on_disk"] > 0 and r["integrity_score"] < 1.0

class TestStatsReport:
    def test_has_sections(self, tmp_path: Path) -> None:
        df = pd.DataFrame([
            {"path": "a.py", "size_bytes": 100, "category": "code", "bizra_project": "data-lake",
             "top_level_dir": "01_CORE", "extension": ".py", "mtime": 1700000000.0},
            {"path": "b.md", "size_bytes": 200, "category": "document", "bizra_project": "general",
             "top_level_dir": "03_ASSETS", "extension": ".md", "mtime": 1700000000.0},
        ])
        stats = generate_stats(df, str(tmp_path / "stats.json"))
        assert stats["total_files"] == 2 and "by_category" in stats and "by_project" in stats

class TestProjectDetection:
    def test_core(self) -> None: assert _detect_project("01_CORE/data-lake/x.py") == "data-lake"
    def test_assets(self) -> None: assert _detect_project("03_ASSETS/voice/x.wav") == "voice"
    def test_general(self) -> None: assert _detect_project("00_CONSTITUTION/readme.md") == "general"
```
