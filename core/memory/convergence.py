"""
Operational convergence helpers for the unified memory system.

This module closes the gap between "migration exists" and
"live memory is trustworthy". It provides:
- source inspection for Claude-flow inputs
- one-shot convergence execution
- policy-bound pass/fail reporting
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .agent_db import AgentDB
from .config import MemoryConfig
from .orchestrator import MigrationOrchestrator


@dataclass(frozen=True)
class ConvergencePolicy:
    """Operational policy for live memory convergence."""

    strict_json: bool = True
    require_artifact_clean: bool = True
    require_healthy_indexes: bool = True
    rebuild_indexes: bool = True


def inspect_claude_flow_sources(db_path: Path, artifact_dir: Path) -> dict[str, Any]:
    """Inspect live Claude-flow sources without mutating them."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "db": _inspect_sqlite_source(db_path),
        "artifacts": _inspect_artifact_dir(artifact_dir),
    }


def run_convergence(
    *,
    config: Optional[MemoryConfig] = None,
    v1_path: Optional[Path] = None,
    claude_flow_db: Optional[Path] = None,
    claude_flow_dir: Optional[Path] = None,
    dry_run: bool = False,
    policy: Optional[ConvergencePolicy] = None,
    on_progress=None,
    report_path: Optional[Path] = None,
) -> tuple[int, dict[str, Any]]:
    """Run the full operational convergence sequence and return (exit_code, report)."""
    cfg = config or MemoryConfig()
    gate = policy or ConvergencePolicy()
    db_path = claude_flow_db or (Path(".swarm") / "memory.db")
    artifact_dir = claude_flow_dir or (Path(".claude-flow") / "memory")

    agent_db = AgentDB(cfg)
    agent_db.initialize()

    source_inspection = inspect_claude_flow_sources(db_path, artifact_dir)
    stats_before = agent_db.stats()

    orch = MigrationOrchestrator(agent_db, on_progress=on_progress)
    if v1_path:
        orch.set_v1_database(v1_path)
    orch.set_claude_flow_db(db_path)
    orch.set_claude_flow_artifact_dir(artifact_dir)
    orch.set_strict_json(gate.strict_json)

    migration = orch.run(dry_run=dry_run)
    rebuild = None
    if not dry_run and gate.rebuild_indexes:
        rebuild = agent_db.rebuild_indexes()
    stats_after = agent_db.stats()

    artifact_issues = source_inspection["artifacts"]["issues"]
    gate_result = _evaluate_gate(
        migration_errors=migration.total_errors,
        artifact_issue_count=len(artifact_issues),
        index_health=stats_after["index_health"]["status"],
        policy=gate,
    )

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "policy": {
            "strict_json": gate.strict_json,
            "require_artifact_clean": gate.require_artifact_clean,
            "require_healthy_indexes": gate.require_healthy_indexes,
            "rebuild_indexes": gate.rebuild_indexes,
        },
        "sources": source_inspection,
        "stats_before": stats_before,
        "migration": migration.to_dict(),
        "rebuild": rebuild,
        "stats_after": stats_after,
        "gate": gate_result,
    }

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )

    return (0 if gate_result["passed"] else 1), report


def format_convergence_summary(report: dict[str, Any]) -> str:
    """Human-readable convergence summary for CLI use."""
    db = report["sources"]["db"]
    artifacts = report["sources"]["artifacts"]
    migration = report["migration"]
    stats_after = report["stats_after"]
    gate = report["gate"]

    lines = [
        "Memory convergence report:",
        (
            f"  source_db: exists={db['exists']}, "
            f"memory_entries={db['table_counts'].get('memory_entries', 0)}, "
            f"patterns={db['table_counts'].get('patterns', 0)}, "
            f"sessions={db['table_counts'].get('sessions', 0)}"
        ),
        (
            f"  artifacts: exists={artifacts['exists']}, "
            f"json_files={artifacts['json_file_count']}, "
            f"issues={len(artifacts['issues'])}"
        ),
        (
            f"  migration: imported={migration['total_imported']}, "
            f"errors={migration['total_errors']}"
        ),
        (
            f"  indexes: status={stats_after['index_health']['status']}, "
            f"fts_rows={stats_after['fts_row_count']}, "
            f"vectors={stats_after['indexed_vectors']}"
        ),
        f"  gate: {'PASSED' if gate['passed'] else 'BLOCKED'}",
    ]

    for reason in gate["reasons"]:
        lines.append(f"    - {reason}")

    return "\n".join(lines)


def _inspect_sqlite_source(db_path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {
        "path": str(db_path),
        "exists": db_path.exists(),
        "table_counts": {},
        "vector_indexes": [],
    }
    if not db_path.exists():
        return data

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        table_names = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table in (
            "memory_entries",
            "patterns",
            "sessions",
            "records",
            "records_fts",
        ):
            if table in table_names:
                data["table_counts"][table] = int(
                    conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )

        if "vector_indexes" in table_names:
            rows = conn.execute(
                "SELECT id, dimensions, total_vectors FROM vector_indexes ORDER BY id"
            ).fetchall()
            data["vector_indexes"] = [dict(row) for row in rows]
    finally:
        conn.close()

    return data


def _inspect_artifact_dir(artifact_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = {
        "path": str(artifact_dir),
        "exists": artifact_dir.exists(),
        "json_file_count": 0,
        "issues": [],
    }
    if not artifact_dir.exists():
        return data

    json_files = sorted(artifact_dir.glob("*.json"))
    data["json_file_count"] = len(json_files)
    for path in json_files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            data["issues"].append(
                {
                    "path": str(path),
                    "code": "invalid_json",
                    "message": str(exc),
                }
            )
    return data


def _evaluate_gate(
    *,
    migration_errors: int,
    artifact_issue_count: int,
    index_health: str,
    policy: ConvergencePolicy,
) -> dict[str, Any]:
    reasons: list[str] = []
    if migration_errors > 0:
        reasons.append(f"migration_errors={migration_errors}")
    if policy.require_artifact_clean and artifact_issue_count > 0:
        reasons.append(f"artifact_issues={artifact_issue_count}")
    if policy.require_healthy_indexes and index_health != "healthy":
        reasons.append(f"index_health={index_health}")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "artifact_clean": artifact_issue_count == 0,
        "index_health": index_health,
        "migration_errors": migration_errors,
    }
