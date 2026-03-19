"""Tests for operational memory convergence helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from core.memory.config import HNSWConfig, MemoryConfig
from core.memory.convergence import (
    ConvergencePolicy,
    format_convergence_summary,
    inspect_claude_flow_sources,
    run_convergence,
)


def _create_source_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE memory_entries (
            id TEXT PRIMARY KEY,
            key TEXT,
            namespace TEXT,
            type TEXT,
            content TEXT,
            tags TEXT,
            metadata TEXT,
            owner_id TEXT,
            created_at INTEGER,
            updated_at INTEGER,
            last_accessed_at INTEGER,
            access_count INTEGER,
            status TEXT
        );
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            project_path TEXT
        );
        """)
    conn.execute(
        """
        INSERT INTO memory_entries (
            id, key, namespace, type, content, tags, metadata, owner_id,
            created_at, updated_at, last_accessed_at, access_count, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "mem-1",
            "alpha",
            "bizra",
            "semantic",
            "Alpha bridge memory",
            "ops",
            "{}",
            "owner",
            1_773_210_000_000,
            1_773_210_100_000,
            1_773_210_200_000,
            1,
            "active",
        ),
    )
    conn.execute(
        "INSERT INTO sessions (id, project_path) VALUES (?, ?)",
        ("session-1", "/tmp/project"),
    )
    conn.commit()
    conn.close()


def test_inspect_claude_flow_sources_reports_counts_and_issues(tmp_path: Path) -> None:
    db_path = tmp_path / ".swarm" / "memory.db"
    db_path.parent.mkdir(parents=True)
    _create_source_db(db_path)

    artifact_dir = tmp_path / ".claude-flow" / "memory"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "project-patterns.json").write_text(
        '{"patterns":{"alpha":{"value":1}}}\n{"extra":true}',
        encoding="utf-8",
    )

    inspection = inspect_claude_flow_sources(db_path, artifact_dir)

    assert inspection["db"]["table_counts"]["memory_entries"] == 1
    assert inspection["db"]["table_counts"]["sessions"] == 1
    assert inspection["artifacts"]["json_file_count"] == 1
    assert len(inspection["artifacts"]["issues"]) == 1


def test_run_convergence_blocks_on_invalid_artifacts(tmp_path: Path) -> None:
    config = MemoryConfig(
        data_dir=tmp_path / "agent_db",
        hnsw=HNSWConfig(dimensions=8, max_elements=128),
    )
    config.auto_embed = False

    db_path = tmp_path / ".swarm" / "memory.db"
    db_path.parent.mkdir(parents=True)
    _create_source_db(db_path)

    artifact_dir = tmp_path / ".claude-flow" / "memory"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "project-patterns.json").write_text(
        '{"patterns":{"alpha":{"value":1}}}\n{"extra":true}',
        encoding="utf-8",
    )

    exit_code, report = run_convergence(
        config=config,
        claude_flow_db=db_path,
        claude_flow_dir=artifact_dir,
        dry_run=True,
    )

    assert exit_code == 1
    assert report["gate"]["passed"] is False
    assert "artifact_issues=1" in report["gate"]["reasons"]
    assert "migration_errors=1" in report["gate"]["reasons"]


def test_run_convergence_imports_and_reports_healthy(tmp_path: Path) -> None:
    config = MemoryConfig(
        data_dir=tmp_path / "agent_db",
        hnsw=HNSWConfig(dimensions=8, max_elements=128),
    )
    config.auto_embed = False

    db_path = tmp_path / ".swarm" / "memory.db"
    db_path.parent.mkdir(parents=True)
    _create_source_db(db_path)

    artifact_dir = tmp_path / ".claude-flow" / "memory"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "session-index.json").write_text(
        '{"sessions":[{"session_id":"s1","timestamp":"2026-03-11T07:00:00Z","summary":"Session summary"}]}',
        encoding="utf-8",
    )
    report_path = tmp_path / "reports" / "convergence.json"

    exit_code, report = run_convergence(
        config=config,
        claude_flow_db=db_path,
        claude_flow_dir=artifact_dir,
        dry_run=False,
        policy=ConvergencePolicy(),
        report_path=report_path,
    )

    assert exit_code == 0
    assert report["gate"]["passed"] is True
    assert report["migration"]["total_imported"] == 2
    assert report["stats_after"]["active_records"] == 2
    assert report["stats_after"]["index_health"]["status"] == "healthy"
    assert report_path.exists()
    assert "gate: PASSED" in format_convergence_summary(report)
