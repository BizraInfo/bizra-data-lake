from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools.ecosystem.config import EcosystemConfig
from tools.ecosystem.hashing import sha256_canonical_json, sha256_file
from tools.ecosystem.vault import Vault, load_vault, resolve_root_keys


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectStatus:
    ACTIVE = "ACTIVE"
    ARCHIVED_TAGGED = "ARCHIVED_TAGGED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class GitInfo:
    head_sha: str
    dirty: bool


@dataclass(frozen=True)
class Project:
    id: str
    path: str
    status: str
    kind: str
    identity_sha256: str
    git: Optional[GitInfo]
    tags: Tuple[str, ...]


@dataclass(frozen=True)
class Root:
    key: str
    path: str


def _run_git(args: Sequence[str], *, cwd: Path, timeout_s: float = 5.0) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
    return (completed.stdout or "").strip()


def _try_git_info(path: Path) -> Optional[GitInfo]:
    try:
        head = _run_git(["rev-parse", "HEAD"], cwd=path)
        if not head or len(head) != 40:
            return None
        dirty = bool(_run_git(["status", "--porcelain"], cwd=path))
        return GitInfo(head_sha=head, dirty=dirty)
    except Exception:
        return None


def _is_archived(name: str, cfg: EcosystemConfig) -> bool:
    n = (name or "").strip().lower()
    if not n:
        return False

    for p in cfg.archive_tagging.archived_name_prefixes:
        if (name or "").startswith(p):
            return True

    for frag in cfg.archive_tagging.archived_name_fragments:
        f = (frag or "").strip().lower()
        if f and f in n:
            return True

    return False


def _safe_list_children(root: Path, *, skip_names: Sequence[str]) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []

    skip = {s.lower() for s in skip_names if str(s).strip()}
    children: List[Path] = []
    try:
        for entry in root.iterdir():
            name = entry.name
            if name.lower() in skip:
                continue
            # Only index directories as projects (fast + stable)
            if entry.is_dir():
                children.append(entry)
    except Exception:
        return []

    return sorted(children, key=lambda p: p.name.lower())


def _folder_identity(path: Path, *, skip_names: Sequence[str]) -> str:
    # Deterministic identity based on the *names* of immediate children.
    # This avoids touching file content and stays fast.
    try:
        names = []
        skip = {s.lower() for s in skip_names if str(s).strip()}
        for entry in path.iterdir():
            if entry.name.lower() in skip:
                continue
            names.append(entry.name)
        names = sorted(set(names), key=lambda s: s.lower())
    except Exception:
        names = []

    payload = {
        "path": str(path.resolve()),
        "children": names,
    }
    return sha256_canonical_json(payload)


def _project_identity(path: Path, git: Optional[GitInfo], *, skip_names: Sequence[str]) -> str:
    if git is not None:
        payload = {
            "path": str(path.resolve()),
            "kind": "git",
            "head_sha": git.head_sha,
            "dirty": git.dirty,
        }
        return sha256_canonical_json(payload)
    return _folder_identity(path, skip_names=skip_names)


def build_manifest(
    *,
    repo_root: Path,
    cfg: EcosystemConfig,
    max_projects: int = 500,
) -> Dict[str, Any]:
    vault_path = (repo_root / cfg.scan.vault_yaml_path).resolve()
    vault: Optional[Vault] = None
    resolved_roots: Dict[str, Path] = {}

    if vault_path.exists():
        vault = load_vault(vault_path)
        resolved_roots.update(resolve_root_keys(vault, cfg.scan.vault_root_keys))

    # Explicit roots (if provided)
    for idx, raw in enumerate(cfg.scan.explicit_roots):
        p = Path(raw).expanduser().resolve()
        resolved_roots[f"explicit_{idx}" ] = p

    roots_out: List[Root] = [Root(key=k, path=str(v)) for k, v in sorted(resolved_roots.items(), key=lambda kv: kv[0])]

    projects: List[Project] = []

    for root_key, root_path in sorted(resolved_roots.items(), key=lambda kv: kv[0]):
        for child in _safe_list_children(root_path, skip_names=cfg.scan.skip_directories):
            if len(projects) >= max_projects:
                break

            name = child.name
            status = ProjectStatus.ACTIVE
            tags: List[str] = []

            if _is_archived(name, cfg):
                status = ProjectStatus.ARCHIVED_TAGGED
                tags.append("archived_tagged")

            git = _try_git_info(child)
            kind = "git" if git is not None else "folder"
            identity = _project_identity(child, git, skip_names=cfg.scan.skip_directories)
            pid = sha256_canonical_json({"root": root_key, "path": str(child.resolve())})[:16]

            projects.append(
                Project(
                    id=pid,
                    path=str(child.resolve()),
                    status=status,
                    kind=kind,
                    identity_sha256=identity,
                    git=git,
                    tags=tuple(sorted(set(tags))),
                )
            )

        if len(projects) >= max_projects:
            break

    # Deterministic ordering
    projects_sorted = sorted(projects, key=lambda p: (p.status, p.path.lower()))

    config_sha = sha256_file(repo_root / "tools" / "ecosystem" / "ecosystem_config.yaml")

    architecture_hashes = {
        "ihsan_constitution_sha256": None,
        "model_family_manifest_sha256": None,
    }
    ihsan_path = repo_root / "constitution" / "ihsan_v1.yaml"
    if ihsan_path.exists():
        architecture_hashes["ihsan_constitution_sha256"] = sha256_file(ihsan_path)
    model_family = repo_root / "model-family-genesis-v1-SEALED.yaml"
    if model_family.exists():
        architecture_hashes["model_family_manifest_sha256"] = sha256_file(model_family)

    manifest: Dict[str, Any] = {
        "schema": "bizra_ecosystem_manifest_v1",
        "version": 1,
        "generated_at_utc": utc_now_iso(),
        "metadata": {
            "bismillah": "Bismillah-ir-Rahman-ir-Rahim",
            "authority": (vault.node_id if vault else "node_unknown"),
            "tool": "tools/ecosystem/indexer.py",
            "config_sha256": config_sha,
        },
        "roots": [asdict(r) for r in roots_out],
        "projects": [
            {
                **asdict(p),
                "git": asdict(p.git) if p.git is not None else None,
                "tags": list(p.tags),
            }
            for p in projects_sorted
        ],
        "canonical_invariants": [
            "deterministic_ordering",
            "sha256_identity",
            "no_delete_side_effects",
            "fail_closed_on_invalid_config",
        ],
        "architecture_hashes": architecture_hashes,
    }

    # Compute manifest_sha256 over the manifest excluding the field itself.
    manifest_sha = sha256_canonical_json(manifest)
    manifest["manifest_sha256"] = manifest_sha
    return manifest
