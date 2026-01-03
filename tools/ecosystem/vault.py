from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass(frozen=True)
class Vault:
    vault_version: int
    node_id: str
    roots: Dict[str, str]


def load_vault(path: Path) -> Vault:
    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"vault must be a mapping: {path}")

    vault_version = int(data.get("vault_version") or 1)
    node_id = str(data.get("node_id") or "")
    roots = data.get("roots") or {}
    if not isinstance(roots, dict):
        roots = {}

    normalized: Dict[str, str] = {}
    for k, v in roots.items():
        key = str(k).strip()
        val = str(v).strip()
        if key and val:
            normalized[key] = val

    return Vault(vault_version=vault_version, node_id=node_id, roots=normalized)


def resolve_root_keys(vault: Vault, keys) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for key in keys:
        if key in vault.roots:
            out[key] = Path(vault.roots[key]).expanduser().resolve()
    return out


def find_repo_root(start: Path) -> Optional[Path]:
    # Best-effort: walk up to find a .git folder.
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists():
            return parent
    return None
