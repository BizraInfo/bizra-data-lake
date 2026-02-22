#!/usr/bin/env python3
"""Step 0: source-lock founding artifacts into canonical repo paths."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

try:
    from common import (
        DEFAULT_CONFIG_PATH,
        REPO_ROOT,
        hash_file_blake3,
        hash_file_sha256,
        load_yaml,
        resolve_path,
        utc_now_iso,
        write_json,
    )
except ModuleNotFoundError:
    from scripts.evidence.common import (
        DEFAULT_CONFIG_PATH,
        REPO_ROOT,
        hash_file_blake3,
        hash_file_sha256,
        load_yaml,
        resolve_path,
        utc_now_iso,
        write_json,
    )


def run(config_path: Path, repo_root: Path) -> int:
    cfg = load_yaml(config_path)
    policy_version = str(cfg.get("policy_version", "evidence-v1.0"))
    section = cfg.get("founding_import", {})
    assets = section.get("assets", [])
    receipt_rel = section.get("receipt_path", "artifacts/evidence/import_receipt.json")
    receipt_path = resolve_path(repo_root, str(receipt_rel))

    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for item in assets:
        logical_path = str(item["logical_path"])
        source = resolve_path(repo_root, str(item["source"]))
        dest = repo_root / logical_path
        expected_sha256 = item.get("expected_sha256")

        rec: dict[str, Any] = {
            "logical_path": logical_path,
            "source": source.as_posix(),
            "destination": dest.as_posix(),
            "expected_sha256": expected_sha256,
            "status": "pending",
        }

        if not source.exists():
            rec["status"] = "missing_source"
            errors.append(f"missing source: {source}")
            results.append(rec)
            continue

        actual_sha256 = hash_file_sha256(source)
        actual_blake3 = hash_file_blake3(source)
        rec["source_sha256"] = actual_sha256
        rec["source_blake3"] = actual_blake3

        if expected_sha256 and actual_sha256.lower() != str(expected_sha256).lower():
            rec["status"] = "sha256_mismatch"
            errors.append(
                f"sha256 mismatch for {logical_path}: expected {expected_sha256} got {actual_sha256}"
            )
            results.append(rec)
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)

        rec["destination_sha256"] = hash_file_sha256(dest)
        rec["destination_blake3"] = hash_file_blake3(dest)
        rec["status"] = "imported"
        results.append(rec)

    receipt = {
        "policy_version": policy_version,
        "generated_at": utc_now_iso(),
        "config_path": config_path.as_posix(),
        "imported_count": sum(1 for r in results if r.get("status") == "imported"),
        "failed_count": len(errors),
        "errors": errors,
        "assets": results,
    }
    write_json(receipt_path, receipt)

    if errors:
        print(f"ERROR: import failed for {len(errors)} asset(s)")
        print(f"Receipt: {receipt_path}")
        return 1

    print(f"Imported {receipt['imported_count']} assets")
    print(f"Receipt: {receipt_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Import and source-lock external founding assets")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    raise SystemExit(run(config_path=args.config, repo_root=args.repo_root))


if __name__ == "__main__":
    main()
