"""
State Checkpointer — Fault-Tolerant State Persistence
=====================================================
Provides checkpoint/restore capabilities for sovereign runtime
enabling recovery from failures and session continuity.

Supports two backends:
- File-based (JSON): Simple, human-readable, default
- SQLite: ACID-compliant, better for high-frequency checkpoints

Standing on Giants: Event Sourcing + Snapshot Pattern + Write-Ahead Logging
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Storage backend for checkpoints."""
    FILE = auto()    # JSON files (default, human-readable)
    SQLITE = auto()  # SQLite database (ACID, faster queries)


@dataclass
class Checkpoint:
    """A point-in-time state snapshot."""
    id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    state: Dict[str, Any] = field(default_factory=dict)
    version: int = 0
    checksum: str = ""
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_checksum(self) -> str:
        """Compute SHA-256 checksum of state."""
        state_str = json.dumps(self.state, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]


class SQLiteCheckpointStore:
    """
    SQLite-based checkpoint storage.

    Provides ACID-compliant storage with:
    - WAL mode for concurrent reads
    - Indexed queries for fast lookups
    - Automatic schema migration
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database and create tables."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrency
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        # Create tables
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                state TEXT NOT NULL,
                version INTEGER NOT NULL,
                checksum TEXT NOT NULL,
                source TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_checkpoints_version
                ON checkpoints(version DESC);
            CREATE INDEX IF NOT EXISTS idx_checkpoints_timestamp
                ON checkpoints(timestamp DESC);

            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)

        # Check/update schema version
        cursor = self._conn.execute(
            "SELECT value FROM schema_info WHERE key = 'version'"
        )
        row = cursor.fetchone()
        if not row:
            self._conn.execute(
                "INSERT INTO schema_info (key, value) VALUES ('version', ?)",
                (str(self.SCHEMA_VERSION),)
            )

        self._conn.commit()

    def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint to SQLite."""
        self._conn.execute("""
            INSERT OR REPLACE INTO checkpoints
            (id, timestamp, state, version, checksum, source, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            checkpoint.id,
            checkpoint.timestamp.isoformat(),
            json.dumps(checkpoint.state, default=str),
            checkpoint.version,
            checkpoint.checksum,
            checkpoint.source,
            json.dumps(checkpoint.metadata, default=str),
        ))
        self._conn.commit()

    def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a specific checkpoint."""
        cursor = self._conn.execute(
            "SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,)
        )
        row = cursor.fetchone()
        return self._row_to_checkpoint(row) if row else None

    def load_latest(self) -> Optional[Checkpoint]:
        """Load the most recent checkpoint."""
        cursor = self._conn.execute(
            "SELECT * FROM checkpoints ORDER BY version DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return self._row_to_checkpoint(row) if row else None

    def list_checkpoints(self, limit: int = 100) -> List[Checkpoint]:
        """List recent checkpoints."""
        cursor = self._conn.execute(
            "SELECT * FROM checkpoints ORDER BY version DESC LIMIT ?",
            (limit,)
        )
        return [self._row_to_checkpoint(row) for row in cursor.fetchall()]

    def count(self) -> int:
        """Count total checkpoints."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM checkpoints")
        return cursor.fetchone()[0]

    def delete_old(self, keep_count: int) -> int:
        """Delete old checkpoints, keeping the most recent N."""
        cursor = self._conn.execute("""
            DELETE FROM checkpoints WHERE id NOT IN (
                SELECT id FROM checkpoints ORDER BY version DESC LIMIT ?
            )
        """, (keep_count,))
        deleted = cursor.rowcount
        self._conn.commit()
        return deleted

    def _row_to_checkpoint(self, row: sqlite3.Row) -> Checkpoint:
        """Convert SQLite row to Checkpoint."""
        return Checkpoint(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            state=json.loads(row["state"]),
            version=row["version"],
            checksum=row["checksum"],
            source=row["source"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def vacuum(self) -> None:
        """Compact the database."""
        self._conn.execute("VACUUM")


class StateCheckpointer:
    """
    Manages state checkpoints for fault tolerance.

    Features:
    - Periodic automatic checkpoints
    - Manual checkpoint triggers
    - State restoration
    - Checkpoint rotation (keep N most recent)
    - Dual backend support: FILE (JSON) or SQLITE

    Usage:
        # File-based (default)
        checkpointer = StateCheckpointer()

        # SQLite-based (recommended for production)
        checkpointer = StateCheckpointer(backend=StorageBackend.SQLITE)
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        max_checkpoints: int = 10,
        auto_interval_seconds: float = 300.0,
        backend: StorageBackend = StorageBackend.FILE,
    ):
        self.checkpoint_dir = checkpoint_dir or Path("sovereign_state/checkpoints")
        self.max_checkpoints = max_checkpoints
        self.auto_interval = auto_interval_seconds
        self.backend = backend

        self._current_version = 0
        self._latest_checkpoint: Optional[Checkpoint] = None
        self._running = False

        # Ensure directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite store if using that backend
        self._sqlite_store: Optional[SQLiteCheckpointStore] = None
        if backend == StorageBackend.SQLITE:
            db_path = self.checkpoint_dir / "checkpoints.db"
            self._sqlite_store = SQLiteCheckpointStore(db_path)
            # Load current version from DB
            latest = self._sqlite_store.load_latest()
            if latest:
                self._current_version = latest.version
                self._latest_checkpoint = latest

    async def checkpoint(
        self,
        state: Dict[str, Any],
        source: str = "manual",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        """Create a new checkpoint."""
        self._current_version += 1

        cp = Checkpoint(
            id=f"cp-{self._current_version:06d}",
            state=state,
            version=self._current_version,
            source=source,
            metadata=metadata or {},
        )
        cp.checksum = cp.compute_checksum()

        # Persist using appropriate backend
        await self._save_checkpoint(cp)
        self._latest_checkpoint = cp

        # Rotate old checkpoints
        await self._rotate_checkpoints()

        logger.info(f"Checkpoint created: {cp.id} (checksum: {cp.checksum}, backend: {self.backend.name})")
        return cp

    async def _save_checkpoint(self, cp: Checkpoint) -> None:
        """Save checkpoint using configured backend."""
        if self.backend == StorageBackend.SQLITE:
            # Use SQLite backend
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._sqlite_store.save, cp)
        else:
            # Use file backend
            await self._save_checkpoint_file(cp)

    async def _save_checkpoint_file(self, cp: Checkpoint) -> None:
        """Save checkpoint to JSON file."""
        filename = self.checkpoint_dir / f"{cp.id}.json"

        data = {
            "id": cp.id,
            "timestamp": cp.timestamp.isoformat(),
            "state": cp.state,
            "version": cp.version,
            "checksum": cp.checksum,
            "source": cp.source,
            "metadata": cp.metadata,
        }

        # Write atomically (write to temp, then rename)
        loop = asyncio.get_event_loop()

        def write_file():
            temp_file = filename.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
            temp_file.rename(filename)

        await loop.run_in_executor(None, write_file)

    async def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        if self.backend == StorageBackend.SQLITE:
            # SQLite handles rotation
            loop = asyncio.get_event_loop()
            deleted = await loop.run_in_executor(
                None, self._sqlite_store.delete_old, self.max_checkpoints
            )
            if deleted > 0:
                logger.debug(f"Rotated {deleted} old checkpoints (SQLite)")
        else:
            # File-based rotation
            checkpoints = sorted(self.checkpoint_dir.glob("cp-*.json"))

            if len(checkpoints) > self.max_checkpoints:
                to_remove = checkpoints[:-self.max_checkpoints]
                for cp_file in to_remove:
                    cp_file.unlink()
                    logger.debug(f"Rotated checkpoint: {cp_file.name}")

    async def restore(self, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]:
        """Restore from a checkpoint. If no ID given, restore latest."""
        if self.backend == StorageBackend.SQLITE:
            return await self._restore_sqlite(checkpoint_id)
        else:
            return await self._restore_file(checkpoint_id)

    async def _restore_sqlite(self, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]:
        """Restore from SQLite backend."""
        loop = asyncio.get_event_loop()

        if checkpoint_id:
            cp = await loop.run_in_executor(
                None, self._sqlite_store.load, checkpoint_id
            )
        else:
            cp = await loop.run_in_executor(
                None, self._sqlite_store.load_latest
            )

        if not cp:
            logger.warning("No checkpoints found in SQLite")
            return None

        # Verify checksum
        computed = cp.compute_checksum()
        if computed != cp.checksum:
            logger.error(f"Checkpoint checksum mismatch: {cp.id}")
            return None

        self._latest_checkpoint = cp
        self._current_version = cp.version
        logger.info(f"Restored checkpoint: {cp.id} (SQLite)")
        return cp

    async def _restore_file(self, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]:
        """Restore from file backend."""
        if checkpoint_id:
            filename = self.checkpoint_dir / f"{checkpoint_id}.json"
        else:
            # Find latest
            checkpoints = sorted(self.checkpoint_dir.glob("cp-*.json"))
            if not checkpoints:
                logger.warning("No checkpoints found")
                return None
            filename = checkpoints[-1]

        if not filename.exists():
            logger.error(f"Checkpoint not found: {filename}")
            return None

        loop = asyncio.get_event_loop()

        def read_file():
            with open(filename) as f:
                return json.load(f)

        data = await loop.run_in_executor(None, read_file)

        cp = Checkpoint(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            state=data["state"],
            version=data["version"],
            checksum=data["checksum"],
            source=data["source"],
            metadata=data.get("metadata", {}),
        )

        # Verify checksum
        computed = cp.compute_checksum()
        if computed != cp.checksum:
            logger.error(f"Checkpoint checksum mismatch: {cp.id}")
            return None

        self._latest_checkpoint = cp
        self._current_version = cp.version
        logger.info(f"Restored checkpoint: {cp.id}")
        return cp

    async def list_checkpoints(self, limit: int = 100) -> List[Checkpoint]:
        """List recent checkpoints."""
        if self.backend == StorageBackend.SQLITE:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._sqlite_store.list_checkpoints, limit
            )
        else:
            # File-based listing
            checkpoints = sorted(self.checkpoint_dir.glob("cp-*.json"), reverse=True)
            result = []
            for cp_file in checkpoints[:limit]:
                cp = await self._restore_file(cp_file.stem)
                if cp:
                    result.append(cp)
            return result

    async def auto_checkpoint_loop(
        self,
        state_provider: callable,
    ) -> None:
        """Background loop for automatic checkpoints."""
        self._running = True
        logger.info(f"Auto-checkpoint started (interval: {self.auto_interval}s)")

        while self._running:
            await asyncio.sleep(self.auto_interval)
            try:
                state = state_provider()
                await self.checkpoint(state, source="auto")
            except Exception as e:
                logger.error(f"Auto-checkpoint failed: {e}")

    def stop(self) -> None:
        """Stop auto-checkpoint loop."""
        self._running = False

    @property
    def latest(self) -> Optional[Checkpoint]:
        """Get the latest checkpoint."""
        return self._latest_checkpoint

    def stats(self) -> Dict[str, Any]:
        """Get checkpointer statistics."""
        if self.backend == StorageBackend.SQLITE:
            checkpoint_count = self._sqlite_store.count()
        else:
            checkpoint_count = len(list(self.checkpoint_dir.glob("cp-*.json")))

        return {
            "checkpoint_count": checkpoint_count,
            "current_version": self._current_version,
            "latest_id": self._latest_checkpoint.id if self._latest_checkpoint else None,
            "auto_interval": self.auto_interval,
            "running": self._running,
            "backend": self.backend.name,
            "max_checkpoints": self.max_checkpoints,
        }

    def close(self) -> None:
        """Close resources (SQLite connection if used)."""
        self._running = False
        if self._sqlite_store:
            self._sqlite_store.close()
            self._sqlite_store = None

    async def vacuum(self) -> None:
        """Compact SQLite database (no-op for file backend)."""
        if self.backend == StorageBackend.SQLITE:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._sqlite_store.vacuum)
            logger.info("SQLite database vacuumed")


__all__ = [
    "Checkpoint",
    "SQLiteCheckpointStore",
    "StateCheckpointer",
    "StorageBackend",
]
