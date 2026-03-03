# Phase 53.3: Repository Migrator

**Status:** SPEC DRAFT | **Script:** `scripts/migration/repo_migrator.py`
**Giants:** Lamport (hash chains -- verify .git/HEAD across transfer), Torvalds (content-addressable storage as verification primitive)

---

## Purpose

Migrate git repositories from scattered C:\ to canonical homes in `B:\BIZRA\01_CORE\`.
Includes genesis consolidation (4 copies to 1), stale worktree cleanup, git integrity
verification, and backwards-compatible symlink creation.

## Migration Map

| Source (C:\) | Destination (B:\BIZRA\) |
|-------------|------------------------|
| `BIZRA-DATA-LAKE` | `01_CORE/data-lake` |
| `BIZRA-NODE0` | `01_CORE/node0` |
| `BIZRA-Dual-Agentic-system--main` | `01_CORE/dual-agentic` |
| `BIZRA-PROJECTS` | `01_CORE/projects` |
| `bizra-genesis-node` (canonical) | `01_CORE/genesis` |
| `bizra-voice` | `03_ASSETS/voice` |

## Data Flow

```
  C:\repos ----> verify_git_clean() ----> migrate_repo() ----> verify_integrity()
                                          (rsync -a)          (.git/HEAD SHA match)
  4x genesis --> consolidate_genesis() -> select canonical -> migrate to 01_CORE/genesis
  *-hexhash ---> find_stale_worktrees() -> offer removal
  post-migrate -> create_migrated_marker() + create_symlink() (optional)
```

## Pseudocode

```python
"""scripts/migration/repo_migrator.py -- Git repository migration."""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BIZRA_SOVEREIGN_ROOT: str = os.environ.get("BIZRA_SOVEREIGN_ROOT", "/mnt/b/BIZRA")

MIGRATION_MAP: dict[str, str] = {
    "/mnt/c/BIZRA-DATA-LAKE": "01_CORE/data-lake",
    "/mnt/c/BIZRA-NODE0": "01_CORE/node0",
    "/mnt/c/BIZRA-Dual-Agentic-system--main": "01_CORE/dual-agentic",
    "/mnt/c/BIZRA-PROJECTS": "01_CORE/projects",
    "/mnt/c/bizra-voice": "03_ASSETS/voice",
}
GENESIS_SOURCES = ["/mnt/c/bizra-genesis-node", "/mnt/c/bizra-genesis-node-backup",
                   "/mnt/c/bizra-genesis-node-fresh", "/mnt/c/bizra-genesis-node-repaired"]
GENESIS_DEST = "01_CORE/genesis"
WORKTREE_PATTERN = re.compile(r"-[0-9a-f]{8,}$")

@dataclass
class RepoInfo:
    path: str; has_git: bool; is_clean: bool; head_sha: str
    branch: str; last_commit_date: str; remote_url: str
    size_bytes: int; uncommitted_count: int

@dataclass
class GenesisCandidate:
    path: str; info: RepoInfo; commit_count: int
    has_clean_history: bool; score: float

def run_git(repo: str, *args: str) -> tuple[str, int]:
    try:
        r = subprocess.run(["git","-C",repo,*args], capture_output=True, text=True, timeout=30)
        return r.stdout.strip(), r.returncode
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "", 1

def inspect_repo(path: str) -> RepoInfo:
    """Gather git repo information for migration validation."""
    git_dir = os.path.join(path, ".git")
    has_git = os.path.isdir(git_dir) or os.path.isfile(git_dir)
    if not has_git:
        return RepoInfo(path, False, True, "", "", "", "", 0, 0)
    head_path = os.path.join(git_dir, "HEAD") if os.path.isdir(git_dir) else ""
    head_sha = ""
    if head_path and os.path.isfile(head_path):
        with open(head_path, "rb") as f: head_sha = hashlib.sha256(f.read()).hexdigest()
    status, _ = run_git(path, "status", "--porcelain")
    uncommitted = len(status.splitlines()) if status else 0
    branch, _ = run_git(path, "rev-parse", "--abbrev-ref", "HEAD")
    date, _ = run_git(path, "log", "-1", "--format=%ci")
    remote, _ = run_git(path, "remote", "get-url", "origin")
    try:
        du = subprocess.run(["du","-sb",path], capture_output=True, text=True, timeout=60)
        size = int(du.stdout.split()[0]) if du.returncode == 0 else 0
    except (subprocess.TimeoutExpired, ValueError, IndexError): size = 0
    return RepoInfo(path, has_git, uncommitted == 0, head_sha, branch, date, remote, size, uncommitted)

def evaluate_genesis_candidate(path: str) -> GenesisCandidate:
    """Score a genesis copy: git presence +1.0, clean tree +0.5, clean history +0.3."""
    info = inspect_repo(path)
    count_str, rc = run_git(path, "rev-list", "--count", "HEAD") if info.has_git else ("0", 1)
    try: commit_count = int(count_str) if rc == 0 else 0
    except ValueError: commit_count = 0
    reflog, _ = run_git(path, "reflog", "--format=%gs") if info.has_git else ("", 1)
    clean_hist = "reset:" not in reflog and "rebase" not in reflog
    score = (1.0 if info.has_git else 0) + (0.5 if info.is_clean else 0) + \
            (0.3 if clean_hist else 0) + commit_count * 0.001
    return GenesisCandidate(path, info, commit_count, clean_hist, score)

def consolidate_genesis(sources: list[str]) -> tuple[Optional[str], list[str]]:
    """Analyze 4 genesis copies, return (canonical_path, rejected_paths)."""
    candidates = [evaluate_genesis_candidate(s) for s in sources if os.path.isdir(s)]
    if not candidates: return None, []
    candidates.sort(key=lambda c: c.score, reverse=True)
    for c in candidates:
        tag = " <-- CANONICAL" if c == candidates[0] else ""
        print(f"  {c.path}: score={c.score:.3f} commits={c.commit_count}{tag}")
    return candidates[0].path, [c.path for c in candidates[1:]]

def find_stale_worktrees(search_dir: str = "/mnt/c") -> list[str]:
    """Find *-hexhash dirs with .git file (worktree marker) and *.worktrees dirs."""
    stale = []
    try:
        for entry in os.scandir(search_dir):
            if not entry.is_dir(): continue
            if WORKTREE_PATTERN.search(entry.name):
                if os.path.isfile(os.path.join(entry.path, ".git")):
                    stale.append(entry.path)
            if entry.name.endswith(".worktrees"):
                stale.append(entry.path)
    except PermissionError: pass
    return stale

def migrate_repo(source: str, dest_rel: str, execute: bool = False) -> dict:
    """Migrate repo via rsync, verify .git/HEAD SHA match post-transfer."""
    dest = os.path.join(BIZRA_SOVEREIGN_ROOT, dest_rel)
    src_info = inspect_repo(source)
    result = {"source": source, "destination": dest, "source_head_sha": src_info.head_sha,
              "dest_head_sha": "", "integrity_verified": False, "status": "dry_run"}
    if not execute: return result
    os.makedirs(dest, exist_ok=True)
    rsync = ["rsync", "-a", "--delete", "--exclude=target/debug/", "--exclude=target/release/",
             "--exclude=node_modules/", "--exclude=__pycache__/", "--info=progress2",
             f"{source}/", f"{dest}/"]
    proc = subprocess.run(rsync, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        result["status"] = "error"; result["error"] = proc.stderr; return result
    dst_info = inspect_repo(dest)
    result["dest_head_sha"] = dst_info.head_sha
    match = src_info.head_sha == dst_info.head_sha if src_info.head_sha else True
    result["integrity_verified"] = match
    result["status"] = "migrated" if match else "integrity_fail"
    return result

def create_migrated_marker(source: str, dest: str) -> None:
    """Write .MIGRATED_TO_B marker JSON in source directory."""
    with open(os.path.join(source, ".MIGRATED_TO_B"), "w") as f:
        json.dump({"migrated_to": dest, "timestamp": datetime.now(timezone.utc).isoformat(),
                   "migration_phase": "53.3"}, f, indent=2)

def create_symlink(source: str, dest: str) -> bool:
    """Create symlink from source to dest (only if source is already a symlink)."""
    try:
        if os.path.islink(source):
            os.unlink(source); os.symlink(dest, source); return True
        print(f"  Source still exists as dir. Manual: rm -rf '{source}' && ln -s '{dest}' '{source}'")
        return False
    except OSError as e:
        print(f"  Symlink failed: {e}"); return False

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 53.3: Repo Migrator")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument("--genesis-only", action="store_true")
    args = parser.parse_args()
    if args.genesis_only:
        consolidate_genesis(GENESIS_SOURCES); return
    # Build plan
    for src, dest_rel in MIGRATION_MAP.items():
        if not os.path.isdir(src): continue
        info = inspect_repo(src)
        clean = "CLEAN" if info.is_clean else f"DIRTY({info.uncommitted_count})"
        print(f"  {src} -> {dest_rel} [{clean}]")
        if args.execute:
            result = migrate_repo(src, dest_rel, execute=True)
            print(f"    Status: {result['status']}")
            if result["status"] == "migrated":
                create_migrated_marker(src, result["destination"])
                if args.symlink: create_symlink(src, result["destination"])
    # Genesis
    canonical, _ = consolidate_genesis(GENESIS_SOURCES)
    if canonical and args.execute:
        result = migrate_repo(canonical, GENESIS_DEST, execute=True)
        if result["status"] == "migrated":
            create_migrated_marker(canonical, result["destination"])

if __name__ == "__main__":
    main()
```

## TDD Anchors

```python
"""tests/migration/test_repo_migrator.py"""
import json, os, subprocess
from pathlib import Path
import pytest

class TestMigrationMap:
    def test_has_all_sources(self) -> None:
        required = {"BIZRA-DATA-LAKE","BIZRA-NODE0","BIZRA-PROJECTS","BIZRA-Dual-Agentic-system--main"}
        names = {Path(s).name for s in MIGRATION_MAP.keys()}
        assert required.issubset(names)

    def test_destinations_valid(self) -> None:
        for d in MIGRATION_MAP.values():
            assert d.startswith("01_CORE/") or d.startswith("03_ASSETS/")

class TestGenesisConsolidation:
    def test_selects_highest_score(self, tmp_path: Path) -> None:
        for name in ["genesis-a", "genesis-b"]:
            repo = tmp_path / name; repo.mkdir()
            subprocess.run(["git","init",str(repo)], capture_output=True, check=True)
            (repo / "README.md").write_text(f"readme {name}")
            subprocess.run(["git","-C",str(repo),"add","."], capture_output=True)
            subprocess.run(["git","-C",str(repo),"commit","-m","init"], capture_output=True)
        canonical, rejected = consolidate_genesis([str(tmp_path/"genesis-a"),str(tmp_path/"genesis-b")])
        assert canonical is not None and len(rejected) == 1

    def test_empty_sources(self) -> None:
        canonical, rejected = consolidate_genesis(["/nonexistent"])
        assert canonical is None and rejected == []

class TestWorktreeDetection:
    def test_hex_suffix_matches(self) -> None:
        assert WORKTREE_PATTERN.search("BIZRA-DATA-LAKE-a94c8f959cafa35e") is not None

    def test_normal_dir_no_match(self) -> None:
        assert WORKTREE_PATTERN.search("BIZRA-DATA-LAKE") is None

class TestGitIntegrityCheck:
    def test_head_sha_captured(self, tmp_path: Path) -> None:
        repo = tmp_path / "src"; repo.mkdir()
        subprocess.run(["git","init",str(repo)], capture_output=True, check=True)
        (repo / "f.txt").write_text("content")
        subprocess.run(["git","-C",str(repo),"add","."], capture_output=True)
        subprocess.run(["git","-C",str(repo),"commit","-m","init"], capture_output=True)
        info = inspect_repo(str(repo))
        assert info.head_sha != "" and info.has_git is True

class TestSymlinkCreation:
    def test_replaces_existing_symlink(self, tmp_path: Path) -> None:
        dest = tmp_path / "dest"; dest.mkdir()
        link = tmp_path / "link"; os.symlink(str(dest), str(link))
        assert create_symlink(str(link), str(dest)) is True

    def test_refuses_directory(self, tmp_path: Path) -> None:
        src = tmp_path / "src"; src.mkdir(); (src / "data.txt").write_text("x")
        dst = tmp_path / "dst"; dst.mkdir()
        assert create_symlink(str(src), str(dst)) is False
        assert src.is_dir()  # Preserved

class TestMigratedMarker:
    def test_marker_valid_json(self, tmp_path: Path) -> None:
        create_migrated_marker(str(tmp_path), "/mnt/b/BIZRA/01_CORE/data-lake")
        data = json.loads((tmp_path / ".MIGRATED_TO_B").read_text())
        assert data["migrated_to"] == "/mnt/b/BIZRA/01_CORE/data-lake"
        assert data["migration_phase"] == "53.3"
```
