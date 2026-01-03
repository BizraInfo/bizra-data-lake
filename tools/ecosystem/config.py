from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class ArchiveTagging:
    archived_name_fragments: Tuple[str, ...]
    archived_name_prefixes: Tuple[str, ...]


@dataclass(frozen=True)
class ScanConfig:
    vault_yaml_path: Path
    vault_root_keys: Tuple[str, ...]
    explicit_roots: Tuple[str, ...]
    skip_directories: Tuple[str, ...]


@dataclass(frozen=True)
class GitHubConfig:
    owner: str
    token_env: str


@dataclass(frozen=True)
class EcosystemConfig:
    version: int
    config_id: str
    scan: ScanConfig
    archive_tagging: ArchiveTagging
    known_folders_doc: Optional[str]
    github: GitHubConfig


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def _as_list_str(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def load_ecosystem_config(path: Path) -> EcosystemConfig:
    raw = load_yaml(path)

    version = int(raw.get("version") or 1)
    config_id = str(raw.get("id") or "bizra_ecosystem_config")

    scan_raw = raw.get("scan") or {}
    if not isinstance(scan_raw, dict):
        scan_raw = {}

    vault_yaml_path = Path(str(scan_raw.get("vault_yaml_path") or ".bizra/vault.yaml"))
    vault_root_keys = tuple(_as_list_str(scan_raw.get("vault_root_keys")))
    explicit_roots = tuple(_as_list_str(scan_raw.get("explicit_roots")))
    skip_directories = tuple(_as_list_str(scan_raw.get("skip_directories")))

    arch_raw = raw.get("archive_tagging") or {}
    if not isinstance(arch_raw, dict):
        arch_raw = {}

    archived_name_fragments = tuple(_as_list_str(arch_raw.get("archived_name_fragments")))
    archived_name_prefixes = tuple(_as_list_str(arch_raw.get("archived_name_prefixes")))

    github_raw = raw.get("github") or {}
    if not isinstance(github_raw, dict):
        github_raw = {}

    github_owner = str(github_raw.get("owner") or "")
    token_env = str(github_raw.get("token_env") or "GITHUB_TOKEN")

    known_folders_doc = raw.get("known_folders_doc")
    if known_folders_doc is not None:
        known_folders_doc = str(known_folders_doc)

    return EcosystemConfig(
        version=version,
        config_id=config_id,
        scan=ScanConfig(
            vault_yaml_path=vault_yaml_path,
            vault_root_keys=vault_root_keys,
            explicit_roots=explicit_roots,
            skip_directories=skip_directories,
        ),
        archive_tagging=ArchiveTagging(
            archived_name_fragments=archived_name_fragments,
            archived_name_prefixes=archived_name_prefixes,
        ),
        known_folders_doc=known_folders_doc,
        github=GitHubConfig(owner=github_owner, token_env=token_env),
    )
