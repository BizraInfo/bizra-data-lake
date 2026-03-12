"""CLI entry point for unified memory migration and stats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="Count without writing")
    parser.add_argument("--v1-path", type=Path, help="Path to SQLite v1 database")
    parser.add_argument(
        "--claude-flow-db",
        type=Path,
        default=Path(".swarm") / "memory.db",
        help="Path to Claude-flow memory.db",
    )
    parser.add_argument(
        "--claude-flow-dir",
        type=Path,
        default=Path(".claude-flow") / "memory",
        help="Path to Claude-flow JSON artifact directory",
    )
    parser.add_argument("--report", type=Path, help="Write report JSON to path")


def _build_config():
    from .config import MemoryConfig

    return MemoryConfig()


def _build_db(config=None):
    from .agent_db import AgentDB

    config = config or _build_config()
    db = AgentDB(config)
    db.initialize()
    return config, db


def _run_migrate(args: argparse.Namespace) -> int:
    from .orchestrator import MigrationOrchestrator

    config, db = _build_db()
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
    return 1 if result.total_errors > 0 else 0


def _run_converge(args: argparse.Namespace) -> int:
    from .convergence import (
        ConvergencePolicy,
        format_convergence_summary,
        run_convergence,
    )

    config = _build_config()
    policy = ConvergencePolicy(
        strict_json=not args.allow_invalid_artifacts,
        require_artifact_clean=not args.allow_invalid_artifacts,
        require_healthy_indexes=not args.allow_stale_indexes,
        rebuild_indexes=not args.skip_rebuild_indexes,
    )
    exit_code, report = run_convergence(
        config=config,
        v1_path=args.v1_path or config.living_memory_db,
        claude_flow_db=args.claude_flow_db,
        claude_flow_dir=args.claude_flow_dir,
        dry_run=args.dry_run,
        policy=policy,
        on_progress=lambda src, done, total: print(f"  {src}: {done}/{total}"),
        report_path=args.report,
    )
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_convergence_summary(report))
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m core.memory",
        description="BIZRA AgentDB memory management",
    )
    sub = parser.add_subparsers(dest="command")

    mig = sub.add_parser("migrate", help="Migrate legacy memory into AgentDB")
    _add_source_args(mig)
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

    converge = sub.add_parser(
        "converge",
        help="Inspect, migrate, rebuild, and gate live memory convergence",
    )
    _add_source_args(converge)
    converge.add_argument(
        "--allow-invalid-artifacts",
        action="store_true",
        help="Do not fail the convergence gate on malformed Claude-flow JSON artifacts",
    )
    converge.add_argument(
        "--allow-stale-indexes",
        action="store_true",
        help="Do not fail the convergence gate when AgentDB index health is stale",
    )
    converge.add_argument(
        "--skip-rebuild-indexes",
        action="store_true",
        help="Skip FTS/HNSW rebuild during convergence",
    )
    converge.add_argument(
        "--json",
        action="store_true",
        help="Print the full convergence report as JSON",
    )

    stats_cmd = sub.add_parser("stats", help="Show AgentDB statistics")  # noqa: F841

    args = parser.parse_args()

    if args.command == "migrate":
        sys.exit(_run_migrate(args))

    elif args.command == "converge":
        sys.exit(_run_converge(args))

    elif args.command == "stats":
        _, db = _build_db()
        print(json.dumps(db.stats(), indent=2, default=str))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
