"""CLI entry point for unified memory migration and stats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m core.memory",
        description="BIZRA AgentDB memory management",
    )
    sub = parser.add_subparsers(dest="command")

    mig = sub.add_parser("migrate", help="Migrate legacy memory into AgentDB")
    mig.add_argument("--dry-run", action="store_true", help="Count without writing")
    mig.add_argument("--v1-path", type=Path, help="Path to SQLite v1 database")
    mig.add_argument(
        "--claude-flow-db",
        type=Path,
        default=Path(".swarm") / "memory.db",
        help="Path to Claude-flow memory.db",
    )
    mig.add_argument(
        "--claude-flow-dir",
        type=Path,
        default=Path(".claude-flow") / "memory",
        help="Path to Claude-flow JSON artifact directory",
    )
    mig.add_argument("--report", type=Path, help="Write migration report JSON to path")
    mig.add_argument(
        "--strict-json",
        action="store_true",
        help="Treat malformed Claude-flow JSON artifacts as migration errors",
    )
    mig.add_argument(
        "--rebuild-indexes",
        action="store_true",
        help="Rebuild FTS and vector indexes after a successful migration",
    )

    stats_cmd = sub.add_parser("stats", help="Show AgentDB statistics")  # noqa: F841

    args = parser.parse_args()

    if args.command == "migrate":
        from .agent_db import AgentDB
        from .config import MemoryConfig
        from .orchestrator import MigrationOrchestrator

        config = MemoryConfig()
        db = AgentDB(config)
        db.initialize()

        orch = MigrationOrchestrator(
            db,
            on_progress=lambda src, done, total: print(f"  {src}: {done}/{total}"),
        )

        v1_path = args.v1_path or config.living_memory_db
        if v1_path:
            orch.set_v1_database(v1_path)
        orch.set_strict_json(args.strict_json)
        if args.claude_flow_db:
            orch.set_claude_flow_db(args.claude_flow_db)
        if args.claude_flow_dir:
            orch.set_claude_flow_artifact_dir(args.claude_flow_dir)

        result = orch.run(dry_run=args.dry_run)
        rebuild = None
        if args.rebuild_indexes and not args.dry_run:
            rebuild = db.rebuild_indexes()
        if args.report:
            import json

            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(
                    {
                        "migration": result.to_dict(),
                        "rebuild": rebuild,
                        "stats": db.stats(),
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        print(result.summary())
        sys.exit(1 if result.total_errors > 0 else 0)

    elif args.command == "stats":
        from .agent_db import AgentDB
        from .config import MemoryConfig

        config = MemoryConfig()
        db = AgentDB(config)
        db.initialize()

        import json

        print(json.dumps(db.stats(), indent=2, default=str))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
