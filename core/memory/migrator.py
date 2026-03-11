"""
Memory Migrator — SQLite v1 -> v2 + HNSW index build.

Migrates data from the Phase 18 LivingMemory SQLite v1 store
(core/living_memory/persistence.py) into the AgentDB v2 store,
preserving all data. The old database is never modified — a .bak
copy is made before migration begins.

Usage:
    from core.memory.migrator import MemoryMigrator
    migrator = MemoryMigrator(agent_db, source_db_path)
    result = migrator.migrate()
    print(result)  # {"migrated": 1234, "skipped": 0, "errors": 0}
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .agent_db import AgentDB
from .config import MemoryConfig
from .types import MemoryKind, MemoryRecord, RecordState

logger = logging.getLogger(__name__)

# Map v1 memory_type to v2 MemoryKind
_V1_TYPE_MAP = {
    "episodic": MemoryKind.EPISODIC,
    "semantic": MemoryKind.SEMANTIC,
    "procedural": MemoryKind.PROCEDURAL,
    "working": MemoryKind.WORKING,
    "prospective": MemoryKind.PROSPECTIVE,
}

_V1_STATE_MAP = {
    "active": RecordState.ACTIVE,
    "archived": RecordState.ARCHIVED,
    "deleted": RecordState.DELETED,
    "consolidating": RecordState.ACTIVE,
    "decaying": RecordState.ACTIVE,
    "corrupted": RecordState.ACTIVE,
}


class MigrationResult:
    """Results from a migration run."""

    def __init__(self) -> None:
        self.migrated: int = 0
        self.skipped: int = 0
        self.errors: int = 0
        self.source_count: int = 0

    def __repr__(self) -> str:
        return (
            f"MigrationResult(migrated={self.migrated}, "
            f"skipped={self.skipped}, errors={self.errors})"
        )

    def to_dict(self) -> dict:
        return {
            "migrated": self.migrated,
            "skipped": self.skipped,
            "errors": self.errors,
            "source_count": self.source_count,
        }


class MemoryMigrator:
    """Migrates SQLite v1 (LivingMemory) data into AgentDB v2."""

    def __init__(
        self,
        agent_db: AgentDB,
        source_path: Optional[Path] = None,
        config: Optional[MemoryConfig] = None,
    ) -> None:
        self._db = agent_db
        self._config = config or MemoryConfig()
        self._source_path = source_path or self._config.living_memory_db

    def migrate(self, backup: bool = True) -> MigrationResult:
        """Run the migration.

        Args:
            backup: If True, copy source DB to .bak before reading.

        Returns:
            MigrationResult with counts.
        """
        result = MigrationResult()

        if self._source_path is None or not self._source_path.exists():
            logger.info("No source database found — skipping migration")
            return result

        # Create backup (copy, never move)
        if backup:
            bak_path = self._source_path.with_suffix(".db.bak")
            if not bak_path.exists():
                shutil.copy2(str(self._source_path), str(bak_path))
                logger.info(f"Source DB backed up to {bak_path}")

        # Open source DB read-only
        try:
            source_conn = sqlite3.connect(f"file:{self._source_path}?mode=ro", uri=True)
            source_conn.row_factory = sqlite3.Row
        except (OSError, ValueError) as e:  # SEC-003 — file_io boundary
            logger.error(f"Failed to open source DB: {e}")
            result.errors += 1
            return result

        try:
            cursor = source_conn.execute(
                "SELECT COUNT(*) FROM memories WHERE state != 'deleted'"
            )
            result.source_count = cursor.fetchone()[0]
            logger.info(f"Migrating {result.source_count} records from v1 to v2...")

            # Read in batches
            batch_size = 500
            offset = 0

            while True:
                cursor = source_conn.execute(
                    "SELECT * FROM memories WHERE state != 'deleted' LIMIT ? OFFSET ?",
                    (batch_size, offset),
                )
                rows = cursor.fetchall()
                if not rows:
                    break

                batch = []
                for row in rows:
                    try:
                        record = self._v1_row_to_record(row)
                        batch.append(record)
                    except Exception as e:  # noqa: BLE001 — boundary boundary
                        logger.warning(f"Failed to convert row {row['id']}: {e}")
                        result.errors += 1

                # Batch upsert into AgentDB
                if batch:
                    count = self._db.backend.upsert_batch(batch)
                    # Index embeddings in HNSW
                    for record in batch:
                        if record.embedding is not None:
                            self._db.hnsw.add(record.id, record.embedding)
                    result.migrated += count

                offset += batch_size

            # Save HNSW index
            self._db.save()

            logger.info(
                f"Migration complete: {result.migrated} migrated, "
                f"{result.errors} errors, {result.skipped} skipped"
            )

        except Exception as e:  # noqa: BLE001 — boundary boundary
            logger.error(f"Migration failed: {e}")
            result.errors += 1
        finally:
            source_conn.close()

        return result

    def _v1_row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        """Convert a v1 SQLite row to a v2 MemoryRecord."""
        kind = _V1_TYPE_MAP.get(row["memory_type"], MemoryKind.SEMANTIC)
        state = _V1_STATE_MAP.get(row["state"], RecordState.ACTIVE)

        embedding = None
        if row["embedding"] is not None:
            embedding = list(np.frombuffer(row["embedding"], dtype=np.float32))

        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            kind=kind,
            state=state,
            embedding=embedding,
            ihsan_score=row["ihsan_score"],
            snr_score=row["snr_score"],
            importance=row["importance"],
            source=row["source"],
            source_id=row["id"],
            related_ids=json.loads(row["related_ids"] or "[]"),
            tags=[],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["last_accessed"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            access_count=row["access_count"],
            metadata={
                "emotional_weight": row["emotional_weight"],
                "confidence": row["confidence"],
                "parent_id": row["parent_id"],
                "origin": "living_memory_v1",
            },
        )
