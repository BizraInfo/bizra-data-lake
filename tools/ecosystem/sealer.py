from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from tools.ecosystem.hashing import sha256_canonical_json, with_integrity_hash


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(manifest: Dict[str, Any], *, out_path: Path) -> Path:
    out_path.write_text(
        __import__("json").dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path


def seal_manifest(
    *,
    manifest_path: Path,
    seal_note: Optional[str] = None,
) -> Dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(str(manifest_path))

    data = __import__("json").loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError("manifest is not a JSON object")

    manifest_sha = data.get("manifest_sha256")
    if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
        tmp = dict(data)
        tmp.pop("manifest_sha256", None)
        manifest_sha = sha256_canonical_json(tmp)

    payload: Dict[str, Any] = {
        "schema": "bizra_ecosystem_receipt_v1",
        "version": 1,
        "truth_label": "MEASURED",
        "timestamp_utc": utc_now_iso(),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "seal_note": seal_note,
    }

    return with_integrity_hash(payload)


def write_receipt(receipt: Dict[str, Any], *, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        __import__("json").dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path
