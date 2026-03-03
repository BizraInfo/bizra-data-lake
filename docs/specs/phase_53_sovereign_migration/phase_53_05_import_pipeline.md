# Phase 53.5: Import Pipeline for Downloads/Desktop/Cloud/Mobile

**Status:** SPEC DRAFT | **Script:** `scripts/migration/import_pipeline.py`
**Giants:** Deming (PDCA: copy, scan, classify, review, place), Shannon (eliminate duplicate transfers)

---

## Purpose

Orchestrate import of ~870 GB from Downloads (400 GB), Desktop (128 GB),
OneDrive (345 GB) into `05_IMPORTS/` staging, then drive through dedup,
classification, operator review, and final placement. Most interactive phase --
surfaces ambiguous files for human decision.

## Source Mapping

| Source | WSL Path | Staging | Est. Size |
|--------|----------|---------|-----------|
| Downloads | `/mnt/c/Users/BIZRA-OS/Downloads` | `05_IMPORTS/downloads` | 400 GB |
| Desktop | `/mnt/c/Users/BIZRA-OS/Desktop` | `05_IMPORTS/desktop` | 128 GB |
| OneDrive | `/mnt/c/Users/BIZRA-OS/OneDrive` | `05_IMPORTS/cloud` | 345 GB |
| Mobile | User-specified | `05_IMPORTS/mobile` | Variable |

## Pipeline Phases

```
  Phase 1: COPY (rsync --archive, never delete source)
  Phase 2: DEDUP (hash staging vs existing B: manifest)
  Phase 3: CLASSIFY (file_classifier.py on non-duplicates)
  Phase 4: REVIEW (operator approves/skips/redirects)
  Phase 5: PLACE (move approved files to final destinations)
  Checkpoints saved after each phase for resumability.
```

## Pseudocode

```python
"""scripts/migration/import_pipeline.py -- Import pipeline."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BIZRA_SOVEREIGN_ROOT = os.environ.get("BIZRA_SOVEREIGN_ROOT", "/mnt/b/BIZRA")
SOURCE_MAP: dict[str, str] = {
    "downloads": "/mnt/c/Users/BIZRA-OS/Downloads",
    "desktop": "/mnt/c/Users/BIZRA-OS/Desktop",
    "cloud": "/mnt/c/Users/BIZRA-OS/OneDrive",
    "mobile": "",
}
CONFIDENCE_THRESHOLD = 0.75
AUTO_APPROVE_THRESHOLD = 0.90

@dataclass
class PipelineState:
    phase: int; source_key: str
    started_at: str = ""; updated_at: str = ""
    total_copied: int = 0; total_deduplicated: int = 0
    total_classified: int = 0; total_placed: int = 0
    errors: list[dict] = field(default_factory=list)

# --- Phase 1: Copy to Staging ---
def copy_to_staging(source_key: str, execute: bool = False) -> tuple[int, int]:
    """rsync source to 05_IMPORTS/ staging. COPIES only, never moves."""
    source = SOURCE_MAP.get(source_key, "")
    if not source or not os.path.isdir(source): return 0, 0
    staging = os.path.join(BIZRA_SOVEREIGN_ROOT, "05_IMPORTS", source_key)
    if not execute:
        count, size = 0, 0
        for dp, _, fns in os.walk(source, followlinks=False):
            for f in fns:
                try: size += os.path.getsize(os.path.join(dp, f)); count += 1
                except OSError: pass
        print(f"[DRY] Would copy {count:,} files ({size/1024**3:.1f} GB)")
        return count, size
    os.makedirs(staging, exist_ok=True)
    rsync = ["rsync", "-a", "--info=progress2", "--no-delete",
             "--exclude=.git/", "--exclude=node_modules/",
             f"{source}/", f"{staging}/"]
    subprocess.run(rsync, timeout=7200)
    count, size = 0, 0
    for dp, _, fns in os.walk(staging, followlinks=False):
        for f in fns:
            try: size += os.path.getsize(os.path.join(dp, f)); count += 1
            except OSError: pass
    return count, size

# --- Phase 2: Dedup Against Existing ---
def dedup_against_existing(staging_dir: str,
                           manifest_path: Optional[str] = None) -> tuple[list[str], list[str]]:
    """Compare staging SHA-256 hashes against existing B: manifest."""
    existing: set[str] = set()
    if manifest_path and os.path.exists(manifest_path):
        import pandas as pd
        existing = set(pd.read_parquet(manifest_path)["sha256"].dropna())
    unique, dupes = [], []
    for dp, _, fns in os.walk(staging_dir, followlinks=False):
        for fname in fns:
            full = os.path.join(dp, fname)
            try:
                h = hashlib.sha256()
                with open(full, "rb") as f:
                    while chunk := f.read(8 * 1024 * 1024): h.update(chunk)
                fhash = h.hexdigest()
                if fhash in existing: dupes.append(full)
                else: unique.append(full); existing.add(fhash)
            except (OSError, PermissionError): unique.append(full)
    return unique, dupes

# --- Phase 3: Classification (delegates to file_classifier) ---
def run_classification(staging_dir: str, unique_files: list[str],
                       use_llm: bool = True) -> list[dict]:
    """Run file_classifier.classify_file() on each unique file."""
    # In production: from scripts.migration.file_classifier import classify_file
    return [{"path": f, "category": "unknown", "subcategory": "pending",
             "bizra_project": "unaffiliated", "relevance_score": 0.0,
             "suggested_destination": "05_IMPORTS", "confidence": 0.0} for f in unique_files]

# --- Phase 4: Operator Review ---
def present_for_review(classifications: list[dict],
                       skip_review: bool = False) -> list[dict]:
    """Auto-approve high-confidence, present low-confidence for human review."""
    for cls in classifications:
        conf = cls.get("confidence", 0.0)
        if conf >= AUTO_APPROVE_THRESHOLD:
            cls["review_decision"] = "approved"
        elif skip_review:
            cls["review_decision"] = "approved" if conf >= CONFIDENCE_THRESHOLD else "skipped"
        elif conf < CONFIDENCE_THRESHOLD:
            cls["review_decision"] = "pending"
            print(f"  REVIEW: {cls['path']} (conf={conf:.2f}, cat={cls['category']})")
            choice = input("    (a)pprove/(s)kip/(q)uit: ").strip().lower()
            cls["review_decision"] = {"a": "approved", "s": "skipped"}.get(choice, "pending")
            if choice == "q": break
        else:
            cls["review_decision"] = "approved"
    return classifications

# --- Phase 5: Place Files ---
def resolve_conflict(source: str, dest: str) -> str:
    """Compare SHA-256: identical -> skip, different -> rename."""
    def qhash(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while c := f.read(8*1024*1024): h.update(c)
        return h.hexdigest()
    try: return "skip" if qhash(source) == qhash(dest) else "rename"
    except OSError: return "rename"

def place_files(classifications: list[dict], execute: bool = False) -> tuple[int, int, list]:
    placed, skipped, errors = 0, 0, []
    for cls in classifications:
        if cls.get("review_decision") != "approved": skipped += 1; continue
        src = cls["path"]
        dest_dir = os.path.join(BIZRA_SOVEREIGN_ROOT, cls["suggested_destination"])
        dest_file = os.path.join(dest_dir, Path(src).name)
        if not execute: placed += 1; continue
        try:
            os.makedirs(dest_dir, exist_ok=True)
            if os.path.exists(dest_file):
                action = resolve_conflict(src, dest_file)
                if action == "skip": skipped += 1; continue
                stem, suf = Path(dest_file).stem, Path(dest_file).suffix
                dest_file = os.path.join(dest_dir, f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suf}")
            shutil.move(src, dest_file); placed += 1
        except (OSError, shutil.Error) as e: errors.append({"path": src, "error": str(e)})
    return placed, skipped, errors

# --- Large File Handling ---
def copy_large_file(source: str, dest: str, progress: bool = True) -> bool:
    """Chunked copy for files >1 GB with progress bar."""
    try:
        total = os.path.getsize(source); copied = 0
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(source, "rb") as s, open(dest, "wb") as d:
            while chunk := s.read(64 * 1024 * 1024):
                d.write(chunk); copied += len(chunk)
                if progress and total:
                    pct = copied / total * 100
                    print(f"\r  [{copied/1024**3:.1f}/{total/1024**3:.1f} GB] {pct:.0f}%", end="")
        if progress: print()
        return True
    except OSError as e: print(f"\n  Error: {e}"); return False

# --- Checkpoints ---
def save_checkpoint(state: PipelineState, cp_dir: str) -> None:
    os.makedirs(cp_dir, exist_ok=True)
    state.updated_at = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(cp_dir, f"checkpoint_{state.source_key}.json"), "w") as f:
        json.dump({"phase": state.phase, "source_key": state.source_key,
                   "started_at": state.started_at, "updated_at": state.updated_at,
                   "total_copied": state.total_copied, "total_deduplicated": state.total_deduplicated,
                   "total_classified": state.total_classified, "total_placed": state.total_placed}, f, indent=2)

def load_checkpoint(source_key: str, cp_dir: str) -> Optional[PipelineState]:
    cp = os.path.join(cp_dir, f"checkpoint_{source_key}.json")
    if not os.path.exists(cp): return None
    with open(cp) as f: d = json.load(f)
    return PipelineState(d["phase"], d["source_key"], d.get("started_at",""),
                         d.get("updated_at",""), d.get("total_copied",0),
                         d.get("total_deduplicated",0), d.get("total_classified",0),
                         d.get("total_placed",0))

# --- Entry Point ---
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 53.5: Import Pipeline")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source", choices=list(SOURCE_MAP.keys()))
    parser.add_argument("--skip-review", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mobile-path")
    args = parser.parse_args()
    if args.mobile_path: SOURCE_MAP["mobile"] = args.mobile_path
    sources = [args.source] if args.source else [k for k in SOURCE_MAP if SOURCE_MAP[k]]
    cp_dir = os.path.join(BIZRA_SOVEREIGN_ROOT, "06_INDEX", "checkpoints")
    manifest = os.path.join(BIZRA_SOVEREIGN_ROOT, "06_INDEX", "manifest.parquet")
    for sk in sources:
        state = load_checkpoint(sk, cp_dir) if args.resume else None
        if not state: state = PipelineState(1, sk, datetime.now(timezone.utc).isoformat())
        staging = os.path.join(BIZRA_SOVEREIGN_ROOT, "05_IMPORTS", sk)
        if state.phase <= 1:
            n, _ = copy_to_staging(sk, execute=args.execute); state.total_copied = n
            state.phase = 2
        if state.phase <= 2 and args.execute:
            unique, dupes = dedup_against_existing(staging, manifest)
            state.total_deduplicated = len(dupes); state.phase = 3
        else: unique = []
        if state.phase <= 3 and args.execute:
            cls = run_classification(staging, unique); state.total_classified = len(cls); state.phase = 4
        else: cls = []
        if state.phase <= 4 and args.execute:
            cls = present_for_review(cls, args.skip_review); state.phase = 5
        if state.phase <= 5 and args.execute:
            p, s, e = place_files(cls, execute=True); state.total_placed = p
            state.errors.extend(e); print(f"  Placed:{p} Skipped:{s} Errors:{len(e)}")
        if args.execute: save_checkpoint(state, cp_dir)

if __name__ == "__main__":
    main()
```

## TDD Anchors

```python
"""tests/migration/test_import_pipeline.py"""
import hashlib, json, os
from pathlib import Path
import pytest

class TestCopyToStaging:
    def test_dry_run(self, tmp_path: Path) -> None:
        src = tmp_path / "source"; src.mkdir()
        for i in range(5): (src / f"f{i}.txt").write_text(f"c{i}")
        with unittest.mock.patch.dict("os.environ", {"BIZRA_SOVEREIGN_ROOT": str(tmp_path)}):
            with unittest.mock.patch("scripts.migration.import_pipeline.SOURCE_MAP", {"test": str(src)}):
                count, size = copy_to_staging("test", execute=False)
        assert count == 5 and size > 0

class TestDedupVsExisting:
    def test_flags_duplicates(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"; staging.mkdir()
        content = b"duplicate content"
        (staging / "dup.txt").write_bytes(content)
        import pandas as pd
        sha = hashlib.sha256(content).hexdigest()
        m = pd.DataFrame({"sha256": [sha], "path": ["/existing"]})
        mp = tmp_path / "manifest.parquet"; m.to_parquet(mp)
        unique, dupes = dedup_against_existing(str(staging), str(mp))
        assert len(dupes) == 1 and len(unique) == 0

class TestConflictResolution:
    def test_identical_skips(self, tmp_path: Path) -> None:
        c = b"same content"
        (tmp_path / "a.txt").write_bytes(c); (tmp_path / "b.txt").write_bytes(c)
        assert resolve_conflict(str(tmp_path/"a.txt"), str(tmp_path/"b.txt")) == "skip"

    def test_different_renames(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_bytes(b"A"); (tmp_path / "b.txt").write_bytes(b"B")
        assert resolve_conflict(str(tmp_path/"a.txt"), str(tmp_path/"b.txt")) == "rename"

class TestLargeFileHandling:
    def test_chunked_copy(self, tmp_path: Path) -> None:
        src = tmp_path / "large.bin"; src.write_bytes(os.urandom(1024*1024))
        dst = tmp_path / "dest" / "large.bin"
        assert copy_large_file(str(src), str(dst), progress=False) is True
        assert dst.exists() and dst.stat().st_size == src.stat().st_size

class TestCheckpoints:
    def test_roundtrip(self, tmp_path: Path) -> None:
        state = PipelineState(3, "downloads", "2026-02-26T00:00:00Z", total_copied=1000)
        save_checkpoint(state, str(tmp_path))
        loaded = load_checkpoint("downloads", str(tmp_path))
        assert loaded and loaded.phase == 3 and loaded.total_copied == 1000
```
