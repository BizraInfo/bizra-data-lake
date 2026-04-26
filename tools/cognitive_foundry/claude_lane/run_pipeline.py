"""Entrypoint for the Claude Cognitive Archive Pilot.

Usage (from repo root):

    python tools/cognitive_foundry/claude_lane/run_pipeline.py \\
        --archive /absolute/path/to/claude-export.zip \\
        [--output-dir tools/cognitive_foundry/claude_lane/output] \\
        [--stages 1,2,3,4] \\
        [--run-id custom-id]

Runs four stages in order. Each stage writes its own manifest.
Final outputs land under <output-dir>/<run_id>/.

Stdlib-only. Runs in any Python 3.10+.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Set

# Support running as a module AND as a direct script. When run as a script,
# sys.path must include the parent of tools/ so relative imports work.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.cognitive_foundry.claude_lane.adjudicate import run_adjudication
    from tools.cognitive_foundry.claude_lane.config import default_config
    from tools.cognitive_foundry.claude_lane.distill import run_distillation
    from tools.cognitive_foundry.claude_lane.inventory import run_inventory
    from tools.cognitive_foundry.claude_lane.review_pack import run_review_pack
    from tools.cognitive_foundry.claude_lane.util import load_archive, make_run_id, write_manifest
else:
    from .adjudicate import run_adjudication
    from .config import default_config
    from .distill import run_distillation
    from .inventory import run_inventory
    from .review_pack import run_review_pack
    from .util import load_archive, make_run_id, write_manifest


DEFAULT_OUTPUT_DIR = Path("tools/cognitive_foundry/claude_lane/output")


def _parse_stages(value: str) -> Set[int]:
    out: Set[int] = set()
    for part in (value or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            n = int(part)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid stage: {part!r}")
        if n not in (1, 2, 3, 4):
            raise argparse.ArgumentTypeError(f"Stage must be 1, 2, 3, or 4 (got {n}).")
        out.add(n)
    if not out:
        out = {1, 2, 3, 4}
    return out


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="claude_lane.run_pipeline",
        description=(
            "Claude Cognitive Archive Pilot — 4-stage deterministic pipeline. "
            "Heuristic-only (no LLM). Never auto-promotes to canon."
        ),
    )
    parser.add_argument(
        "--archive",
        required=True,
        type=Path,
        help="Path to a Claude export zip (users.json + projects.json + memories.json + conversations.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--stages",
        type=_parse_stages,
        default="1,2,3,4",
        help="Comma-separated list of stages to run (default: 1,2,3,4).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override the auto-derived run id. Use carefully — breaks determinism guarantee.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    archive_path: Path = args.archive.resolve()
    if not archive_path.exists():
        print(f"[error] Archive not found: {archive_path}", file=sys.stderr)
        return 2

    run_id = args.run_id or make_run_id(archive_path)
    run_root = args.output_dir.resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    config = default_config()

    # Top-level manifest
    top_manifest: Dict[str, object] = {
        "run_id": run_id,
        "archive_path": str(archive_path),
        "started_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "stages_requested": sorted(args.stages),
        "output_root": str(run_root),
        "pipeline": "cognitive_foundry.claude_lane",
        "pipeline_version": "0.1.0-pilot",
    }
    write_manifest(run_root / "run_manifest.json", top_manifest)

    print(f"[info] run_id={run_id}")
    print(f"[info] output_root={run_root}")
    print(f"[info] stages={sorted(args.stages)}")

    # Load the archive once; all stages share the parsed data.
    try:
        archive = load_archive(archive_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 3

    stage_summaries: Dict[str, object] = {}

    if 1 in args.stages:
        print("[stage 1] inventory ...")
        s1 = run_inventory(
            archive=archive,
            run_root=run_root,
            config=config,
            run_id=run_id,
            archive_path=str(archive_path),
        )
        stage_summaries["stage_1_inventory"] = s1
        print(f"[stage 1] done — {json.dumps(s1.get('counts', {}))}")

    if 2 in args.stages:
        print("[stage 2] distillation ...")
        s2 = run_distillation(
            archive=archive,
            run_root=run_root,
            config=config,
            run_id=run_id,
            archive_path=str(archive_path),
        )
        stage_summaries["stage_2_distillation"] = s2
        print(f"[stage 2] done — {json.dumps(s2.get('counts', {}))}")

    if 3 in args.stages:
        print("[stage 3] adjudication ...")
        s3 = run_adjudication(
            run_root=run_root,
            config=config,
            run_id=run_id,
            archive_path=str(archive_path),
        )
        stage_summaries["stage_3_adjudication"] = s3
        print(f"[stage 3] done — {json.dumps(s3.get('counts', {}))}")

    if 4 in args.stages:
        print("[stage 4] review pack ...")
        s4 = run_review_pack(
            run_root=run_root,
            config=config,
            run_id=run_id,
            archive_path=str(archive_path),
        )
        stage_summaries["stage_4_review_pack"] = s4
        print(f"[stage 4] done — {json.dumps(s4.get('counts', {}))}")

    top_manifest["completed_at_utc"] = datetime.now(tz=timezone.utc).isoformat()
    top_manifest["stage_summaries"] = stage_summaries
    write_manifest(run_root / "run_manifest.json", top_manifest)

    print(f"[info] complete — see {run_root}")
    print("[info] nothing was promoted to canon. Open 04_review_pack/review_workbook.csv to review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
