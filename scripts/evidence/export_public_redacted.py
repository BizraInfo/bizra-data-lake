#!/usr/bin/env python3
"""Export public redacted tier from an existing private tier manifest."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

try:
    from common import DEFAULT_PACKAGE_ROOT, manifest_content_hash, utc_now_iso, write_json
except ModuleNotFoundError:
    from scripts.evidence.common import (
        DEFAULT_PACKAGE_ROOT,
        manifest_content_hash,
        utc_now_iso,
        write_json,
    )


def run(package_root: Path, from_tier: str, to_tier: str) -> int:
    from_root = package_root / from_tier
    to_root = package_root / to_tier

    manifest_path = from_root / "manifest" / "evidence_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: missing source manifest {manifest_path}")
        return 1

    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])

    to_root.mkdir(parents=True, exist_ok=True)
    exported_entries: list[dict[str, Any]] = []

    for entry in sorted(entries, key=lambda e: e["logical_path"]):
        e = dict(entry)
        logical = str(e["logical_path"])
        visibility = str(e.get("visibility", "both"))
        public_mode = str(e.get("public_mode", "full"))
        source_from_private = from_root / logical

        should_copy = public_mode == "full" and visibility in {"both", "public_only"}

        if should_copy and source_from_private.exists() and source_from_private.is_file():
            dest = to_root / logical
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_from_private, dest)
            e["copied_path"] = logical
        else:
            e["copied_path"] = None

        exported_entries.append(e)

    stage = str(manifest.get("stage", "scaffold"))
    policy_version = str(manifest.get("policy_version", "evidence-v1.0"))

    exported_manifest = {
        "generated_at": utc_now_iso(),
        "exported_from_tier": from_tier,
        "stage": stage,
        "tier": to_tier,
        "policy_version": policy_version,
        "manifest_content_hash": manifest_content_hash(stage, to_tier, policy_version, exported_entries),
        "entries": exported_entries,
    }

    write_json(to_root / "manifest" / "evidence_manifest.json", exported_manifest)

    # Preserve research index and latest gate report as metadata artifacts.
    for rel in [
        "manifest/research_index.json",
        "gate_reports/latest_gate_report.json",
    ]:
        src = from_root / rel
        if src.exists() and src.is_file():
            dst = to_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    print(f"Exported {len(exported_entries)} manifest entries to {to_root}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Export public redacted evidence tier")
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--from", dest="from_tier", default="private_full")
    parser.add_argument("--to", dest="to_tier", default="public_redacted")
    args = parser.parse_args()
    raise SystemExit(run(package_root=args.package_root, from_tier=args.from_tier, to_tier=args.to_tier))


if __name__ == "__main__":
    main()
