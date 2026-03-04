#!/usr/bin/env python3
"""Emit a BIZRA resource fabric snapshot for proactive orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.skills.resource_fabric import ResourceFabric


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BIZRA resource fabric snapshot")
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--profile", type=Path, default=None)
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--include-assets", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("04_GOLD") / "resource_fabric_snapshot.json",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    fabric = ResourceFabric(project_root=args.project_root, profile_path=args.profile)
    snapshot = fabric.snapshot(
        limit=max(1, args.limit),
        include_assets=bool(args.include_assets),
        force=bool(args.force),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, ensure_ascii=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "fabric_score": snapshot.get("fabric_score"),
                "coverage_score": snapshot.get("coverage_score"),
                "total_assets": snapshot.get("total_assets"),
                "active_sources": snapshot.get("active_sources"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
