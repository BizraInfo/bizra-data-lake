from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json_bytes(obj: Any) -> bytes:
    # Deterministic JSON encoding: stable keys, no whitespace.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_canonical_json(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def with_integrity_hash(payload: Dict[str, Any], *, field_name: str = "integrity_hash") -> Dict[str, Any]:
    cloned = dict(payload)
    cloned.pop(field_name, None)
    digest = sha256_canonical_json(cloned)
    out = dict(payload)
    out[field_name] = f"sha256:{digest}"
    return out
