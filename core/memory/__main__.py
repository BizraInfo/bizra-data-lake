"""CLI entry point: python -m core.memory migrate [--dry-run] [--v1-path PATH]"""

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

        result = orch.run(dry_run=args.dry_run)
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
