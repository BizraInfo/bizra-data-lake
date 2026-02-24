#!/usr/bin/env python3
"""Shared helpers for evidence package tooling."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from blake3 import blake3

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "scripts" / "evidence" / "config" / "evidence_package.yaml"
)
DEFAULT_GATE_CONFIG_PATH = (
    REPO_ROOT / "scripts" / "evidence" / "config" / "final_gate.yaml"
)
DEFAULT_PACKAGE_ROOT = (
    REPO_ROOT / "artifacts" / "evidence" / "BIZRA-EVIDENCE-PACKAGE-v1.0-GENESIS"
)
DOC_EXTENSIONS = {".pdf", ".md", ".txt"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_json_bytes(obj: Any) -> bytes:
    return canonical_json_dumps(obj).encode("utf-8")


def hash_bytes_blake3(data: bytes) -> str:
    return blake3(data).hexdigest()


def hash_bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file_blake3(path: Path) -> str:
    h = blake3()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML map in {path}")
    return raw


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def resolve_path(repo_root: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return repo_root / p


def rel_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def path_matches_patterns(path_str: str, patterns: Iterable[str]) -> bool:
    normalized = path_str.replace("\\", "/")
    for pattern in patterns:
        p = pattern.replace("\\", "/")
        if fnmatch.fnmatch(normalized, p):
            return True
        if "/" not in p and p in normalized:
            return True
    return False


def iter_documents(root: Path, extensions: set[str] | None = None) -> list[Path]:
    exts = extensions or DOC_EXTENSIONS
    if not root.exists():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    out.sort(key=lambda x: x.as_posix())
    return out


def ensure_package_layout(package_root: Path) -> None:
    (package_root / "private_full").mkdir(parents=True, exist_ok=True)
    (package_root / "public_redacted").mkdir(parents=True, exist_ok=True)
    (package_root / "integrity").mkdir(parents=True, exist_ok=True)
    (package_root / "manifest").mkdir(parents=True, exist_ok=True)


def manifest_content_hash(
    stage: str, tier: str, policy_version: str, entries: list[dict[str, Any]]
) -> str:
    payload = {
        "stage": stage,
        "tier": tier,
        "policy_version": policy_version,
        "entries": sorted(entries, key=lambda e: e["logical_path"]),
    }
    return hash_bytes_blake3(canonical_json_bytes(payload))
