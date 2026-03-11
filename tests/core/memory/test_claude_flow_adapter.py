from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from core.memory.adapters.claude_flow import ClaudeFlowAdapter


def _create_claude_flow_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
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
        CREATE TABLE patterns (
            id TEXT PRIMARY KEY,
            name TEXT,
            pattern_type TEXT,
            description TEXT,
            tags TEXT,
            created_at INTEGER,
            updated_at INTEGER,
            confidence REAL
        );
        """
    )
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
            "Alpha memory from claude-flow",
            json.dumps(["ops", "memory"]),
            json.dumps({"related_ids": ["ctx-1"]}),
            "owner",
            1_773_210_000_000,
            1_773_210_100_000,
            1_773_210_200_000,
            3,
            "active",
        ),
    )
    conn.execute(
        """
        INSERT INTO patterns (
            id, name, pattern_type, description, tags, created_at, updated_at, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "pat-1",
            "Observe/Throttle",
            "governance",
            "Quarantine-not-evict pattern",
            json.dumps(["sape"]),
            1_773_210_000_000,
            1_773_210_050_000,
            0.92,
        ),
    )
    conn.commit()
    conn.close()


def test_export_db_preserves_provenance(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    _create_claude_flow_db(db_path)

    batch = ClaudeFlowAdapter(db_path=db_path).export_db()

    assert batch.issues == []
    assert len(batch.records) == 2
    assert {record.source for record in batch.records} == {
        "claude_flow_db",
        "claude_flow_patterns",
    }
    db_record = next(record for record in batch.records if record.source == "claude_flow_db")
    assert db_record.metadata["table_name"] == "memory_entries"
    assert db_record.metadata["source_path"] == str(db_path)
    assert "bizra" in db_record.tags


def test_export_artifacts_reports_malformed_json_without_crashing(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "memory"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "session-index.json").write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "session_id": "s1",
                        "timestamp": "2026-03-11T07:00:00Z",
                        "summary": "Session summary",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "project-patterns.json").write_text(
        '{"patterns": {"alpha": {"value": 1}}}\n{"extra": true}',
        encoding="utf-8",
    )

    batch = ClaudeFlowAdapter(artifact_dir=artifact_dir).export_artifacts()

    assert len(batch.records) == 1
    assert batch.records[0].source == "claude_flow_session_index"
    assert len(batch.issues) == 1
    assert batch.issues[0].code == "invalid_json"
