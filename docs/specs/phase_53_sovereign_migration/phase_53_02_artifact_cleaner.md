# Phase 53.2: Build Artifact Cleaner

**Status:** SPEC DRAFT | **Script:** `scripts/migration/artifact_cleaner.py`
**Giants:** Deming (eliminate waste in the process), Dijkstra (separate essential from accidental complexity)

---

## Purpose

Remove regenerable build artifacts from C:\ before migration, freeing ~103 GB.
**DRY RUN by default.** Nothing is deleted without the explicit `--execute` flag.

## Targets

| Target | Location | Est. Size | Rationale |
|--------|----------|-----------|-----------|
| Rust target/ | `bizra-omega/target/` | ~85 GB | `cargo build` regenerates |
| Stale venvs | `.venv`, `.venv-wsl` | ~16 GB | Keep `.venv-linux` only |
| Python caches | `__pycache__/`, `.mypy_cache/`, `.pytest_cache/` | ~350 MB | Bytecode/cache |
| Ruff + coverage | `.ruff_cache/`, `.coverage`, `htmlcov/` | ~80 MB | Linter/coverage |
| MagicMock junk | `<MagicMock*` at repo root | ~10 MB | Test artifact pollution |
| Nested junk | `BIZRA-DATA-LAKE/C:/`, `BIZRA-DATA-LAKE/BIZRA-DATA-LAKE/` | ~4 GB | Self-referential |

## Data Flow

```
  scan_all_targets() --> List[CleanTarget]
        |
        v (dry-run default)
  report_findings()  --> Print table + sizes
        |
        v (--execute only)
  execute_cleanup()  --> Delete + generate receipt
        |
        v
  06_INDEX/cleanup_receipt.json
```

## Pseudocode

```python
"""scripts/migration/artifact_cleaner.py -- Build artifact cleaner."""
from __future__ import annotations
import argparse, json, os, shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BIZRA_SOVEREIGN_ROOT: str = os.environ.get("BIZRA_SOVEREIGN_ROOT", "/mnt/b/BIZRA")
SOURCE_ROOTS: list[str] = [
    "/mnt/c/BIZRA-DATA-LAKE", "/mnt/c/BIZRA-NODE0", "/mnt/c/BIZRA-PROJECTS",
    "/mnt/c/BIZRA-Dual-Agentic-system--main", "/mnt/c/bizra-genesis-node",
    "/mnt/c/bizra-genesis-node-backup", "/mnt/c/bizra-genesis-node-fresh",
    "/mnt/c/bizra-genesis-node-repaired",
]

@dataclass
class CleanTarget:
    path: str; size_bytes: int; category: str
    is_directory: bool; reason: str

@dataclass
class CleanupReceipt:
    timestamp: str; mode: str; targets_found: int; targets_cleaned: int
    bytes_freed: int; bytes_freed_human: str
    errors: list[dict]; targets: list[dict]

def dir_size(path: str) -> int:
    total = 0
    try:
        for dp, _, fns in os.walk(path, followlinks=False):
            for f in fns:
                try: total += os.path.getsize(os.path.join(dp, f))
                except OSError: pass
    except OSError: pass
    return total

def find_rust_targets(roots: list[str]) -> list[CleanTarget]:
    targets = []
    for root in roots:
        td = os.path.join(root, "bizra-omega", "target")
        if os.path.isdir(td):
            targets.append(CleanTarget(td, dir_size(td), "rust_target", True,
                                       "Rust build artifacts (cargo regenerates)"))
    return targets

def find_stale_venvs(roots: list[str]) -> list[CleanTarget]:
    targets = []
    for root in roots:
        for name in (".venv", ".venv-wsl"):
            vp = os.path.join(root, name)
            if os.path.isdir(vp):
                targets.append(CleanTarget(vp, dir_size(vp), "venv", True,
                                           f"Stale venv ({name}); keep .venv-linux"))
    return targets

def find_cache_dirs(roots: list[str]) -> list[CleanTarget]:
    targets = []
    cache_names = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "htmlcov"}
    for root in roots:
        if not os.path.isdir(root): continue
        for dp, dns, _ in os.walk(root, followlinks=False):
            if ".git" in dp.split(os.sep): continue
            for d in dns:
                if d in cache_names:
                    full = os.path.join(dp, d)
                    targets.append(CleanTarget(full, dir_size(full), "cache", True,
                                               f"Regenerable cache ({d})"))
            dns[:] = [d for d in dns if d not in {"target","node_modules",".git"} | cache_names]
    return targets

def find_coverage_files(roots: list[str]) -> list[CleanTarget]:
    targets = []
    for root in roots:
        for dp, _, fns in os.walk(root, followlinks=False):
            if ".git" in dp.split(os.sep): continue
            for f in fns:
                if f == ".coverage" or f.startswith(".coverage."):
                    full = os.path.join(dp, f)
                    try: size = os.path.getsize(full)
                    except OSError: size = 0
                    targets.append(CleanTarget(full, size, "coverage", False, "Coverage data"))
    return targets

def find_magicmock_junk(roots: list[str]) -> list[CleanTarget]:
    targets = []
    for root in roots:
        if not os.path.isdir(root): continue
        for entry in os.scandir(root):
            if entry.is_file() and entry.name.startswith("<MagicMock"):
                try: size = entry.stat().st_size
                except OSError: size = 0
                targets.append(CleanTarget(entry.path, size, "magicmock", False,
                                           "MagicMock junk from test framework"))
    return targets

def find_nested_junk(_roots: list[str]) -> list[CleanTarget]:
    targets = []
    dl = "/mnt/c/BIZRA-DATA-LAKE"
    for subdir, reason in [("C:", "Accidental Windows path copy"),
                           ("BIZRA-DATA-LAKE", "Recursive self-copy")]:
        path = os.path.join(dl, subdir)
        if os.path.isdir(path):
            targets.append(CleanTarget(path, dir_size(path), "nested_junk", True, reason))
    return targets

def scan_all_targets(roots: list[str], categories: Optional[set[str]] = None) -> list[CleanTarget]:
    finders = {"rust_target": find_rust_targets, "venv": find_stale_venvs,
               "cache": find_cache_dirs, "coverage": find_coverage_files,
               "magicmock": find_magicmock_junk, "nested_junk": find_nested_junk}
    all_t: list[CleanTarget] = []
    for cat, fn in finders.items():
        if categories and cat not in categories: continue
        all_t.extend(fn(roots))
    return all_t

def report_findings(targets: list[CleanTarget]) -> None:
    if not targets: print("No artifacts found."); return
    by_cat: dict[str, list[CleanTarget]] = {}
    for t in targets: by_cat.setdefault(t.category, []).append(t)
    total = 0
    print(f"\n{'='*60}\n  Artifact Cleaner Report (DRY RUN)\n{'='*60}")
    for cat, ts in sorted(by_cat.items()):
        cat_size = sum(t.size_bytes for t in ts); total += cat_size
        print(f"\n  [{cat.upper()}] {len(ts)} targets, {cat_size/1024**3:.2f} GB")
        for t in ts:
            print(f"    {'DIR' if t.is_directory else 'FILE'} {t.size_bytes/1024**2:>8.1f} MB  {t.path}")
    print(f"\n{'='*60}\n  TOTAL: {len(targets)} targets, {total/1024**3:.2f} GB\n{'='*60}")

def execute_cleanup(targets: list[CleanTarget]) -> CleanupReceipt:
    receipt = CleanupReceipt(datetime.now(timezone.utc).isoformat(), "execute",
                             len(targets), 0, 0, "", [], [])
    for t in targets:
        try:
            shutil.rmtree(t.path) if t.is_directory else os.remove(t.path)
            receipt.targets_cleaned += 1; receipt.bytes_freed += t.size_bytes
            receipt.targets.append({"path": t.path, "status": "deleted", "size": t.size_bytes})
        except (OSError, PermissionError) as e:
            receipt.errors.append({"path": t.path, "error": str(e)})
    receipt.bytes_freed_human = f"{receipt.bytes_freed/1024**3:.2f} GB"
    return receipt

def write_receipt(receipt: CleanupReceipt, output_dir: str) -> None:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "cleanup_receipt.json", "w") as f:
        json.dump({"operation": "artifact_cleanup", "timestamp": receipt.timestamp,
                   "mode": receipt.mode, "targets_found": receipt.targets_found,
                   "targets_cleaned": receipt.targets_cleaned, "bytes_freed": receipt.bytes_freed,
                   "bytes_freed_human": receipt.bytes_freed_human,
                   "errors": receipt.errors, "targets": receipt.targets}, f, indent=2)

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 53.2: Artifact Cleaner")
    parser.add_argument("--execute", action="store_true", help="Actually delete (default: dry run)")
    parser.add_argument("--targets", nargs="+",
                        choices=["rust_target","venv","cache","coverage","magicmock","nested_junk"])
    parser.add_argument("--output-dir", default=os.path.join(BIZRA_SOVEREIGN_ROOT, "06_INDEX"))
    args = parser.parse_args()
    categories = set(args.targets) if args.targets else None
    targets = scan_all_targets(SOURCE_ROOTS, categories)
    if not args.execute: report_findings(targets); return
    total_gb = sum(t.size_bytes for t in targets) / 1024**3
    confirm = input(f"DELETE {len(targets)} targets ({total_gb:.2f} GB)? Type 'yes': ").strip()
    if confirm != "yes": print("Aborted."); return
    receipt = execute_cleanup(targets)
    write_receipt(receipt, args.output_dir)
    print(f"Freed: {receipt.bytes_freed_human}, Errors: {len(receipt.errors)}")

if __name__ == "__main__":
    main()
```

## TDD Anchors

```python
"""tests/migration/test_artifact_cleaner.py"""
import json, os
from pathlib import Path
import pytest

class TestDryRunNoDelete:
    def test_dry_run_preserves_files(self, tmp_path: Path) -> None:
        cache = tmp_path / "project" / "__pycache__"; cache.mkdir(parents=True)
        (cache / "mod.pyc").write_bytes(b"fake")
        targets = scan_all_targets([str(tmp_path)])
        assert len(targets) > 0
        report_findings(targets)
        assert cache.exists()  # Still exists after dry run

class TestTargetDetection:
    def test_rust_target(self, tmp_path: Path) -> None:
        t = tmp_path / "bizra-omega" / "target" / "debug" / "build"; t.mkdir(parents=True)
        (t / "fake.o").write_bytes(b"x" * 1000)
        assert len(find_rust_targets([str(tmp_path)])) == 1

    def test_cache_dirs(self, tmp_path: Path) -> None:
        pc = tmp_path / "src" / "__pycache__"; pc.mkdir(parents=True)
        (pc / "mod.pyc").write_bytes(b"x")
        assert any(t.category == "cache" for t in find_cache_dirs([str(tmp_path)]))

class TestVenvSelection:
    def test_keeps_venv_linux(self, tmp_path: Path) -> None:
        for n in [".venv", ".venv-linux", ".venv-wsl"]:
            d = tmp_path / n; d.mkdir(); (d / "pyvenv.cfg").write_text("home=/usr/bin")
        names = {Path(t.path).name for t in find_stale_venvs([str(tmp_path)])}
        assert ".venv" in names and ".venv-wsl" in names and ".venv-linux" not in names

class TestMagicMockDetection:
    def test_detects_magicmock(self, tmp_path: Path) -> None:
        (tmp_path / "<MagicMock name='foo'>").write_text("junk")
        assert len(find_magicmock_junk([str(tmp_path)])) == 1

    def test_ignores_normal(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("code")
        assert len(find_magicmock_junk([str(tmp_path)])) == 0

class TestReceiptGenerated:
    def test_receipt_valid_json(self, tmp_path: Path) -> None:
        receipt = CleanupReceipt("2026-02-26T00:00:00Z", "execute", 3, 2,
                                 1024, "0.00 GB", [{"path":"/x","error":"perm"}], [])
        write_receipt(receipt, str(tmp_path))
        data = json.loads((tmp_path / "cleanup_receipt.json").read_text())
        assert data["operation"] == "artifact_cleanup" and data["targets_found"] == 3
```
