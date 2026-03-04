#!/usr/bin/env python3
"""Run BIZRA agentic self harness and emit a JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.elite.self_harness_engine import SelfHarnessEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BIZRA agentic self harness run")
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--profile", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--include-findings", action="store_true")
    p.add_argument("--findings-limit", type=int, default=200)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("04_GOLD") / "self_harness_report.json",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    engine = SelfHarnessEngine(project_root=args.project_root, profile_path=args.profile)
    report = engine.run(
        include_findings=bool(args.include_findings),
        findings_limit=max(1, args.findings_limit),
        force=bool(args.force),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "harness_score": report.get("harness_score"),
                "total_findings": report.get("total_findings"),
                "critical": (report.get("by_severity") or {}).get("critical"),
                "high": (report.get("by_severity") or {}).get("high"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
