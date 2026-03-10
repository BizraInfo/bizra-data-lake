"""
Atomic file I/O — crash-safe JSON and text persistence.

Extracted from core/proof_engine/genesis_ceremony.py so that all
sovereign-state writes (lifecycle, authority, proof, assets, awareness,
URP) use the same crash-safe pattern: write → fsync → rename.

Standing on Giants:
- Lampson (1983): Hints for Computer System Design — crash recovery
- Nakamoto (2008): Immutable append-only state transitions
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(target: Path, content: str) -> None:
    """Write *content* to *target* atomically via write→fsync→rename.

    If the process crashes between steps, the old file remains intact.
    The temporary file is created in the same directory as *target* to
    guarantee ``os.replace`` is same-filesystem (no cross-device copy).
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    closed = False
    try:
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        closed = True
        os.replace(tmp, str(target))
    except BaseException:
        if not closed:
            os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(target: Path, payload: Any, *, indent: int = 2) -> None:
    """Serialize *payload* as JSON and write atomically to *target*."""
    content = json.dumps(payload, indent=indent, ensure_ascii=True) + "\n"
    atomic_write_text(target, content)


def read_json(path: Path, default: Any = None) -> Any:
    """Read and parse JSON from *path*, returning *default* on any failure."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


__all__ = [
    "atomic_write_text",
    "atomic_write_json",
    "read_json",
]
