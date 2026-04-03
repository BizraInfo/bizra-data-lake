#!/usr/bin/env python3
"""
Generate a minimal certification bundle for Truth Kernel claims.
Outputs a JSON bundle and a SHA256 digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_dir():
        return None
    entries: List[str] = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            digest = sha256_file(file_path)
            rel = file_path.relative_to(path).as_posix()
            if digest:
                entries.append(f"{rel}:{digest}")
    if not entries:
        return None
    return sha256_bytes("\n".join(entries).encode("utf-8"))


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_bundle(repo_root: Path, evidence_dir: Path) -> Dict[str, object]:
    commit_sha = os.environ.get("GITHUB_SHA")
    if not commit_sha:
        git_head = repo_root / ".git" / "HEAD"
        if git_head.exists():
            commit_sha = git_head.read_text(encoding="utf-8", errors="ignore").strip()
        else:
            commit_sha = "unknown"

    artifacts = [
        ("ci_evidence_receipt", repo_root / "ci-evidence-receipt.json"),
        ("quality_radar", repo_root / "evidence" / "quality_radar_ci.json"),
        ("ihsan_constitution", repo_root / "constitution" / "ihsan_v1.yaml"),
        ("model_family", repo_root / "model-family-genesis-v1-SEALED.yaml"),
    ]

    artifacts_payload = []
    for name, path in artifacts:
        digest = sha256_file(path)
        artifacts_payload.append(
            {
                "name": name,
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": digest,
                "missing": digest is None,
            }
        )

    receipts_dir = repo_root / "docs" / "evidence" / "receipts"
    receipts_hash = sha256_dir(receipts_dir)

    bundle = {
        "schema": "bizra-cert-bundle-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": commit_sha,
        "artifacts": artifacts_payload,
        "receipts": {
            "path": receipts_dir.relative_to(repo_root).as_posix(),
            "sha256": receipts_hash,
            "missing": receipts_hash is None,
        },
    }

    quality_radar_path = repo_root / "evidence" / "quality_radar_ci.json"
    if quality_radar_path.exists():
        try:
            quality_data = json.loads(quality_radar_path.read_text(encoding="utf-8"))
            ihsan_data = quality_data.get("ihsan", {})
            bundle["snr"] = {
                "raw": ihsan_data.get("snr"),
                "tier": ihsan_data.get("tier"),
                "normalized": ihsan_data.get("snr_normalized"),
            }
        except Exception:
            bundle["snr"] = {"error": "quality_radar_parse_failed"}

    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate certification bundle for Truth Kernel claims.")
    parser.add_argument("--output", default="evidence/cert_bundle.json", help="Output bundle path")
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = build_bundle(repo_root, output_path.parent)
    bundle_json = json.dumps(bundle, indent=2, sort_keys=True)
    bundle_hash = sha256_bytes(bundle_json.encode("utf-8"))
    bundle["bundle_sha256"] = f"sha256:{bundle_hash}"

    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    (output_path.parent / "cert_bundle.sha256").write_text(bundle_hash, encoding="utf-8")

    print(bundle["bundle_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
