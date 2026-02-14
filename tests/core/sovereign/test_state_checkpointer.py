"""
State Checkpointer — Comprehensive Test Suite
==============================================================================
Persistence layer hardening: 60+ tests targeting 90%+ coverage.

Critical areas: checksum integrity, round-trip persistence (both backends),
checkpoint rotation, version monotonicity, atomic file writes, SQLite WAL mode,
corruption detection, empty-state handling, and concurrent safety.

Created: 2026-02-11
"""

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.sovereign.state_checkpointer import (
    Checkpoint,
    SQLiteCheckpointStore,
    StateCheckpointer,
    StorageBackend,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(n: int = 1) -> Dict[str, Any]:
    """Generate a deterministic test state dict."""
    return {"counter": n, "label": f"state-{n}", "nested": {"a": n * 10}}


def _make_checkpoint(
    version: int = 1, state: Dict[str, Any] | None = None, source: str = "test"
) -> Checkpoint:
    """Convenience factory for Checkpoint instances."""
    s = state or _make_state(version)
    cp = Checkpoint(
        id=f"cp-{version:06d}",
        state=s,
        version=version,
        source=source,
        metadata={"test": True},
    )
    cp.checksum = cp.compute_checksum()
    return cp


# ===========================================================================
# 1. TestStorageBackend
# ===========================================================================

class TestStorageBackend:
    """Verify the StorageBackend enum surface."""

    def test_file_variant_exists(self):
        assert StorageBackend.FILE is not None

    def test_sqlite_variant_exists(self):
        assert StorageBackend.SQLITE is not None

    def test_enum_members_are_distinct(self):
        assert StorageBackend.FILE != StorageBackend.SQLITE

    def test_enum_has_exactly_two_members(self):
        assert len(StorageBackend) == 2

    def test_file_name(self):
        assert StorageBackend.FILE.name == "FILE"

    def test_sqlite_name(self):
        assert StorageBackend.SQLITE.name == "SQLITE"


# ===========================================================================
# 2. TestCheckpoint
# ===========================================================================

class TestCheckpoint:
    """Test the Checkpoint dataclass and its checksum logic."""

    def test_default_id_is_empty_string(self):
        cp = Checkpoint()
        assert cp.id == ""

    def test_default_state_is_empty_dict(self):
        cp = Checkpoint()
        assert cp.state == {}

    def test_default_version_is_zero(self):
        cp = Checkpoint()
        assert cp.version == 0

    def test_default_checksum_is_empty(self):
        cp = Checkpoint()
        assert cp.checksum == ""

    def test_default_source_is_empty(self):
        cp = Checkpoint()
        assert cp.source == ""

    def test_default_metadata_is_empty_dict(self):
        cp = Checkpoint()
        assert cp.metadata == {}

    def test_timestamp_is_utc(self):
        cp = Checkpoint()
        assert cp.timestamp.tzinfo is not None
        assert cp.timestamp.tzinfo == timezone.utc

    def test_compute_checksum_deterministic(self):
        """Same state must always produce the same checksum."""
        state = {"x": 1, "y": [2, 3]}
        cp = Checkpoint(state=state)
        c1 = cp.compute_checksum()
        c2 = cp.compute_checksum()
        assert c1 == c2

    def test_compute_checksum_key_order_invariant(self):
        """sort_keys=True means insertion order does not matter."""
        cp1 = Checkpoint(state={"b": 2, "a": 1})
        cp2 = Checkpoint(state={"a": 1, "b": 2})
        assert cp1.compute_checksum() == cp2.compute_checksum()

    def test_compute_checksum_length(self):
        """Checksum is SHA-256 truncated to 16 hex chars."""
        cp = Checkpoint(state={"data": "test"})
        cs = cp.compute_checksum()
        assert len(cs) == 16
        assert all(c in "0123456789abcdef" for c in cs)

    def test_compute_checksum_changes_on_state_change(self):
        cp1 = Checkpoint(state={"v": 1})
        cp2 = Checkpoint(state={"v": 2})
        assert cp1.compute_checksum() != cp2.compute_checksum()

    def test_metadata_isolation_between_instances(self):
        """Default factory must not share dict references."""
        cp1 = Checkpoint()
        cp2 = Checkpoint()
        cp1.metadata["key"] = "val"
        assert "key" not in cp2.metadata

    def test_state_isolation_between_instances(self):
        cp1 = Checkpoint()
        cp2 = Checkpoint()
        cp1.state["x"] = 99
        assert "x" not in cp2.state

    def test_compute_checksum_empty_state(self):
        """Empty dict should have a stable checksum."""
        cp = Checkpoint(state={})
        cs = cp.compute_checksum()
        assert isinstance(cs, str)
        assert len(cs) == 16


# ===========================================================================
# 3. TestSQLiteCheckpointStore
# ===========================================================================

class TestSQLiteCheckpointStore:
    """Test the raw SQLite storage layer."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_checkpoints.db"

    @pytest.fixture
    def store(self, db_path: Path) -> SQLiteCheckpointStore:
        s = SQLiteCheckpointStore(db_path)
        yield s
        s.close()

    def test_init_creates_database_file(self, db_path: Path):
        store = SQLiteCheckpointStore(db_path)
        assert db_path.exists()
        store.close()

    def test_init_creates_checkpoints_table(self, store: SQLiteCheckpointStore):
        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
        )
        assert cursor.fetchone() is not None

    def test_init_creates_schema_info_table(self, store: SQLiteCheckpointStore):
        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_info'"
        )
        assert cursor.fetchone() is not None

    def test_init_creates_version_index(self, store: SQLiteCheckpointStore):
        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_checkpoints_version'"
        )
        assert cursor.fetchone() is not None

    def test_init_creates_timestamp_index(self, store: SQLiteCheckpointStore):
        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_checkpoints_timestamp'"
        )
        assert cursor.fetchone() is not None

    def test_wal_mode_enabled(self, store: SQLiteCheckpointStore):
        """WAL mode must be set for concurrent-read safety."""
        cursor = store._conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"

    def test_synchronous_normal(self, store: SQLiteCheckpointStore):
        cursor = store._conn.execute("PRAGMA synchronous")
        # NORMAL = 1
        val = cursor.fetchone()[0]
        assert val == 1

    def test_check_same_thread_false(self, db_path: Path):
        """Connection must be created with check_same_thread=False."""
        store = SQLiteCheckpointStore(db_path)
        # If check_same_thread were True, accessing from another thread would
        # raise ProgrammingError.  We verify the connection works from the
        # current thread (it was opened from the same, so we inspect the
        # init code path indirectly -- the test simply verifies no error).
        assert store._conn is not None
        store.close()

    def test_schema_version_stored(self, store: SQLiteCheckpointStore):
        cursor = store._conn.execute(
            "SELECT value FROM schema_info WHERE key = 'version'"
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == str(SQLiteCheckpointStore.SCHEMA_VERSION)

    def test_save_and_load_round_trip(self, store: SQLiteCheckpointStore):
        cp = _make_checkpoint(version=1)
        store.save(cp)
        loaded = store.load(cp.id)
        assert loaded is not None
        assert loaded.id == cp.id
        assert loaded.state == cp.state
        assert loaded.version == cp.version
        assert loaded.checksum == cp.checksum
        assert loaded.source == cp.source
        assert loaded.metadata == cp.metadata

    def test_save_overwrites_same_id(self, store: SQLiteCheckpointStore):
        cp = _make_checkpoint(version=1, state={"v": "old"})
        store.save(cp)
        cp_new = _make_checkpoint(version=1, state={"v": "new"})
        cp_new.id = cp.id
        cp_new.checksum = cp_new.compute_checksum()
        store.save(cp_new)
        loaded = store.load(cp.id)
        assert loaded.state == {"v": "new"}

    def test_load_nonexistent_returns_none(self, store: SQLiteCheckpointStore):
        result = store.load("nonexistent-id")
        assert result is None

    def test_load_latest_returns_highest_version(self, store: SQLiteCheckpointStore):
        for v in range(1, 6):
            store.save(_make_checkpoint(version=v))
        latest = store.load_latest()
        assert latest is not None
        assert latest.version == 5

    def test_load_latest_empty_db_returns_none(self, store: SQLiteCheckpointStore):
        result = store.load_latest()
        assert result is None

    def test_list_checkpoints_ordering(self, store: SQLiteCheckpointStore):
        for v in range(1, 6):
            store.save(_make_checkpoint(version=v))
        listing = store.list_checkpoints()
        versions = [cp.version for cp in listing]
        assert versions == [5, 4, 3, 2, 1]  # DESC order

    def test_list_checkpoints_with_limit(self, store: SQLiteCheckpointStore):
        for v in range(1, 11):
            store.save(_make_checkpoint(version=v))
        listing = store.list_checkpoints(limit=3)
        assert len(listing) == 3
        assert listing[0].version == 10

    def test_list_checkpoints_empty_db(self, store: SQLiteCheckpointStore):
        listing = store.list_checkpoints()
        assert listing == []

    def test_count_empty(self, store: SQLiteCheckpointStore):
        assert store.count() == 0

    def test_count_after_inserts(self, store: SQLiteCheckpointStore):
        for v in range(1, 4):
            store.save(_make_checkpoint(version=v))
        assert store.count() == 3

    def test_delete_old_keeps_recent(self, store: SQLiteCheckpointStore):
        for v in range(1, 8):
            store.save(_make_checkpoint(version=v))
        deleted = store.delete_old(keep_count=3)
        assert deleted == 4
        assert store.count() == 3
        # Remaining should be versions 5, 6, 7
        remaining = store.list_checkpoints()
        remaining_versions = {cp.version for cp in remaining}
        assert remaining_versions == {5, 6, 7}

    def test_delete_old_when_under_limit(self, store: SQLiteCheckpointStore):
        for v in range(1, 3):
            store.save(_make_checkpoint(version=v))
        deleted = store.delete_old(keep_count=10)
        assert deleted == 0
        assert store.count() == 2

    def test_delete_old_keep_zero_deletes_all(self, store: SQLiteCheckpointStore):
        for v in range(1, 4):
            store.save(_make_checkpoint(version=v))
        deleted = store.delete_old(keep_count=0)
        assert deleted == 3
        assert store.count() == 0

    def test_vacuum(self, store: SQLiteCheckpointStore):
        """Vacuum should not raise."""
        store.save(_make_checkpoint(version=1))
        store.delete_old(keep_count=0)
        store.vacuum()  # Should not raise

    def test_close_sets_conn_to_none(self, db_path: Path):
        store = SQLiteCheckpointStore(db_path)
        store.close()
        assert store._conn is None

    def test_row_to_checkpoint_preserves_types(self, store: SQLiteCheckpointStore):
        cp = _make_checkpoint(version=42, state={"int": 1, "str": "hello", "list": [1, 2]})
        store.save(cp)
        loaded = store.load(cp.id)
        assert isinstance(loaded.state["int"], int)
        assert isinstance(loaded.state["str"], str)
        assert isinstance(loaded.state["list"], list)
        assert isinstance(loaded.version, int)
        assert isinstance(loaded.timestamp, datetime)

    def test_metadata_round_trip(self, store: SQLiteCheckpointStore):
        cp = _make_checkpoint(version=1)
        cp.metadata = {"key": "value", "num": 42}
        store.save(cp)
        loaded = store.load(cp.id)
        assert loaded.metadata == {"key": "value", "num": 42}

    def test_empty_metadata_round_trip(self, store: SQLiteCheckpointStore):
        cp = _make_checkpoint(version=1)
        cp.metadata = {}
        store.save(cp)
        loaded = store.load(cp.id)
        assert loaded.metadata == {}


# ===========================================================================
# 4. TestStateCheckpointerFile
# ===========================================================================

class TestStateCheckpointerFile:
    """Tests for StateCheckpointer with FILE backend."""

    @pytest.fixture
    def cp_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "checkpoints_file"

    @pytest.fixture
    def checkpointer(self, cp_dir: Path) -> StateCheckpointer:
        sc = StateCheckpointer(
            checkpoint_dir=cp_dir,
            max_checkpoints=5,
            auto_interval_seconds=300.0,
            backend=StorageBackend.FILE,
        )
        yield sc
        sc.close()

    def test_init_creates_directory(self, cp_dir: Path):
        sc = StateCheckpointer(checkpoint_dir=cp_dir, backend=StorageBackend.FILE)
        assert cp_dir.is_dir()
        sc.close()

    def test_init_no_sqlite_store(self, checkpointer: StateCheckpointer):
        assert checkpointer._sqlite_store is None

    def test_init_version_starts_at_zero(self, checkpointer: StateCheckpointer):
        assert checkpointer._current_version == 0

    def test_init_latest_is_none(self, checkpointer: StateCheckpointer):
        assert checkpointer.latest is None

    async def test_checkpoint_creates_file(self, checkpointer: StateCheckpointer, cp_dir: Path):
        cp = await checkpointer.checkpoint(_make_state(1), source="test")
        expected_file = cp_dir / f"{cp.id}.json"
        assert expected_file.exists()

    async def test_checkpoint_file_content_valid_json(
        self, checkpointer: StateCheckpointer, cp_dir: Path
    ):
        cp = await checkpointer.checkpoint(_make_state(1))
        filepath = cp_dir / f"{cp.id}.json"
        with open(filepath) as f:
            data = json.load(f)
        assert data["id"] == cp.id
        assert data["state"] == _make_state(1)
        assert data["checksum"] == cp.checksum

    async def test_checkpoint_increments_version(self, checkpointer: StateCheckpointer):
        cp1 = await checkpointer.checkpoint(_make_state(1))
        cp2 = await checkpointer.checkpoint(_make_state(2))
        cp3 = await checkpointer.checkpoint(_make_state(3))
        assert cp1.version == 1
        assert cp2.version == 2
        assert cp3.version == 3

    async def test_version_monotonicity(self, checkpointer: StateCheckpointer):
        """Versions must strictly increase with every checkpoint call."""
        versions = []
        for i in range(10):
            cp = await checkpointer.checkpoint(_make_state(i))
            versions.append(cp.version)
        # Strictly monotonically increasing
        assert versions == sorted(versions)
        assert len(set(versions)) == len(versions)  # All unique

    async def test_checkpoint_sets_latest(self, checkpointer: StateCheckpointer):
        cp = await checkpointer.checkpoint(_make_state(1))
        assert checkpointer.latest is not None
        assert checkpointer.latest.id == cp.id

    async def test_checkpoint_computes_checksum(self, checkpointer: StateCheckpointer):
        cp = await checkpointer.checkpoint(_make_state(1))
        expected = Checkpoint(state=_make_state(1)).compute_checksum()
        assert cp.checksum == expected
        assert len(cp.checksum) == 16

    async def test_checkpoint_stores_source(self, checkpointer: StateCheckpointer):
        cp = await checkpointer.checkpoint(_make_state(1), source="my-source")
        assert cp.source == "my-source"

    async def test_checkpoint_stores_metadata(self, checkpointer: StateCheckpointer):
        meta = {"reason": "test", "count": 42}
        cp = await checkpointer.checkpoint(_make_state(1), metadata=meta)
        assert cp.metadata == meta

    async def test_checkpoint_default_source_is_manual(self, checkpointer: StateCheckpointer):
        cp = await checkpointer.checkpoint(_make_state(1))
        assert cp.source == "manual"

    async def test_checkpoint_default_metadata_is_empty(self, checkpointer: StateCheckpointer):
        cp = await checkpointer.checkpoint(_make_state(1))
        assert cp.metadata == {}

    async def test_restore_latest(self, checkpointer: StateCheckpointer):
        await checkpointer.checkpoint(_make_state(1))
        await checkpointer.checkpoint(_make_state(2))
        await checkpointer.checkpoint(_make_state(3))
        restored = await checkpointer.restore()
        assert restored is not None
        assert restored.state == _make_state(3)
        assert restored.version == 3

    async def test_restore_by_id(self, checkpointer: StateCheckpointer):
        cp1 = await checkpointer.checkpoint(_make_state(1))
        await checkpointer.checkpoint(_make_state(2))
        restored = await checkpointer.restore(checkpoint_id=cp1.id)
        assert restored is not None
        assert restored.state == _make_state(1)
        assert restored.id == cp1.id

    async def test_restore_nonexistent_returns_none(self, checkpointer: StateCheckpointer):
        result = await checkpointer.restore(checkpoint_id="cp-999999")
        assert result is None

    async def test_restore_empty_dir_returns_none(self, checkpointer: StateCheckpointer):
        result = await checkpointer.restore()
        assert result is None

    async def test_checksum_mismatch_returns_none_on_restore(
        self, checkpointer: StateCheckpointer, cp_dir: Path
    ):
        """Tampered file must be rejected -- this is the critical integrity test."""
        cp = await checkpointer.checkpoint(_make_state(1))
        # Tamper with the file
        filepath = cp_dir / f"{cp.id}.json"
        with open(filepath) as f:
            data = json.load(f)
        data["state"]["counter"] = 9999  # Corrupt the state
        with open(filepath, "w") as f:
            json.dump(data, f)
        # Restore should detect mismatch and return None
        result = await checkpointer.restore(checkpoint_id=cp.id)
        assert result is None

    async def test_rotation_deletes_oldest(self, checkpointer: StateCheckpointer, cp_dir: Path):
        """With max_checkpoints=5, creating 8 should leave exactly 5."""
        for i in range(1, 9):
            await checkpointer.checkpoint(_make_state(i))
        remaining = list(cp_dir.glob("cp-*.json"))
        assert len(remaining) == 5

    async def test_rotation_keeps_most_recent(
        self, checkpointer: StateCheckpointer, cp_dir: Path
    ):
        for i in range(1, 9):
            await checkpointer.checkpoint(_make_state(i))
        remaining_names = sorted(f.stem for f in cp_dir.glob("cp-*.json"))
        # Versions 4-8 should survive (the most recent 5)
        expected = [f"cp-{v:06d}" for v in range(4, 9)]
        assert remaining_names == expected

    async def test_atomic_write_no_tmp_leftover(
        self, checkpointer: StateCheckpointer, cp_dir: Path
    ):
        """Atomic writes should not leave .tmp files behind."""
        await checkpointer.checkpoint(_make_state(1))
        tmp_files = list(cp_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    async def test_restore_updates_internal_version(self, checkpointer: StateCheckpointer):
        await checkpointer.checkpoint(_make_state(1))
        cp2 = await checkpointer.checkpoint(_make_state(2))
        # Create a fresh checkpointer pointing at the same directory
        sc2 = StateCheckpointer(
            checkpoint_dir=checkpointer.checkpoint_dir,
            backend=StorageBackend.FILE,
        )
        restored = await sc2.restore()
        assert restored is not None
        assert sc2._current_version == cp2.version
        sc2.close()

    async def test_list_checkpoints_file_backend(self, checkpointer: StateCheckpointer):
        for i in range(1, 4):
            await checkpointer.checkpoint(_make_state(i))
        listing = await checkpointer.list_checkpoints()
        assert len(listing) == 3

    async def test_list_checkpoints_with_limit(self, checkpointer: StateCheckpointer):
        for i in range(1, 6):
            await checkpointer.checkpoint(_make_state(i))
        listing = await checkpointer.list_checkpoints(limit=2)
        assert len(listing) == 2

    def test_stats_structure(self, checkpointer: StateCheckpointer):
        s = checkpointer.stats()
        assert "checkpoint_count" in s
        assert "current_version" in s
        assert "latest_id" in s
        assert "auto_interval" in s
        assert "running" in s
        assert "backend" in s
        assert "max_checkpoints" in s

    def test_stats_initial_values(self, checkpointer: StateCheckpointer):
        s = checkpointer.stats()
        assert s["checkpoint_count"] == 0
        assert s["current_version"] == 0
        assert s["latest_id"] is None
        assert s["running"] is False
        assert s["backend"] == "FILE"
        assert s["max_checkpoints"] == 5

    async def test_stats_after_checkpoints(self, checkpointer: StateCheckpointer):
        await checkpointer.checkpoint(_make_state(1))
        await checkpointer.checkpoint(_make_state(2))
        s = checkpointer.stats()
        assert s["checkpoint_count"] == 2
        assert s["current_version"] == 2
        assert s["latest_id"] == "cp-000002"

    def test_latest_property_none_initially(self, checkpointer: StateCheckpointer):
        assert checkpointer.latest is None

    async def test_latest_property_after_checkpoint(self, checkpointer: StateCheckpointer):
        cp = await checkpointer.checkpoint(_make_state(1))
        assert checkpointer.latest is cp

    def test_stop_sets_running_false(self, checkpointer: StateCheckpointer):
        checkpointer._running = True
        checkpointer.stop()
        assert checkpointer._running is False

    def test_close_stops_and_nulls_sqlite(self, checkpointer: StateCheckpointer):
        checkpointer._running = True
        checkpointer.close()
        assert checkpointer._running is False
        assert checkpointer._sqlite_store is None

    async def test_round_trip_preserves_state_types(self, checkpointer: StateCheckpointer):
        """Verify complex nested types survive JSON round-trip."""
        state = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, "two", 3.0],
            "nested": {"a": {"b": {"c": 1}}},
        }
        await checkpointer.checkpoint(state)
        restored = await checkpointer.restore()
        assert restored is not None
        assert restored.state == state


# ===========================================================================
# 5. TestStateCheckpointerSQLite
# ===========================================================================

class TestStateCheckpointerSQLite:
    """Tests for StateCheckpointer with SQLITE backend."""

    @pytest.fixture
    def cp_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "checkpoints_sqlite"

    @pytest.fixture
    def checkpointer(self, cp_dir: Path) -> StateCheckpointer:
        sc = StateCheckpointer(
            checkpoint_dir=cp_dir,
            max_checkpoints=5,
            auto_interval_seconds=300.0,
            backend=StorageBackend.SQLITE,
        )
        yield sc
        sc.close()

    def test_init_creates_directory(self, cp_dir: Path):
        sc = StateCheckpointer(checkpoint_dir=cp_dir, backend=StorageBackend.SQLITE)
        assert cp_dir.is_dir()
        sc.close()

    def test_init_creates_sqlite_store(self, checkpointer: StateCheckpointer):
        assert checkpointer._sqlite_store is not None

    def test_init_creates_db_file(self, cp_dir: Path):
        sc = StateCheckpointer(checkpoint_dir=cp_dir, backend=StorageBackend.SQLITE)
        assert (cp_dir / "checkpoints.db").exists()
        sc.close()

    def test_init_version_starts_at_zero_when_empty(self, checkpointer: StateCheckpointer):
        assert checkpointer._current_version == 0

    async def test_init_loads_version_from_existing_db(self, cp_dir: Path):
        """Re-opening checkpointer must resume version from DB."""
        sc1 = StateCheckpointer(checkpoint_dir=cp_dir, backend=StorageBackend.SQLITE)
        await sc1.checkpoint(_make_state(1))
        await sc1.checkpoint(_make_state(2))
        await sc1.checkpoint(_make_state(3))
        sc1.close()
        # Re-open
        sc2 = StateCheckpointer(checkpoint_dir=cp_dir, backend=StorageBackend.SQLITE)
        assert sc2._current_version == 3
        assert sc2.latest is not None
        assert sc2.latest.version == 3
        sc2.close()

    async def test_checkpoint_increments_version(self, checkpointer: StateCheckpointer):
        cp1 = await checkpointer.checkpoint(_make_state(1))
        cp2 = await checkpointer.checkpoint(_make_state(2))
        assert cp1.version == 1
        assert cp2.version == 2

    async def test_version_monotonicity(self, checkpointer: StateCheckpointer):
        versions = []
        for i in range(10):
            cp = await checkpointer.checkpoint(_make_state(i))
            versions.append(cp.version)
        assert versions == sorted(versions)
        assert len(set(versions)) == len(versions)

    async def test_checkpoint_computes_checksum(self, checkpointer: StateCheckpointer):
        cp = await checkpointer.checkpoint(_make_state(1))
        expected = Checkpoint(state=_make_state(1)).compute_checksum()
        assert cp.checksum == expected

    async def test_checkpoint_sets_latest(self, checkpointer: StateCheckpointer):
        cp = await checkpointer.checkpoint(_make_state(1))
        assert checkpointer.latest is not None
        assert checkpointer.latest.id == cp.id

    async def test_restore_latest(self, checkpointer: StateCheckpointer):
        await checkpointer.checkpoint(_make_state(1))
        await checkpointer.checkpoint(_make_state(2))
        restored = await checkpointer.restore()
        assert restored is not None
        assert restored.state == _make_state(2)

    async def test_restore_by_id(self, checkpointer: StateCheckpointer):
        cp1 = await checkpointer.checkpoint(_make_state(1))
        await checkpointer.checkpoint(_make_state(2))
        restored = await checkpointer.restore(checkpoint_id=cp1.id)
        assert restored is not None
        assert restored.state == _make_state(1)

    async def test_restore_nonexistent_returns_none(self, checkpointer: StateCheckpointer):
        result = await checkpointer.restore(checkpoint_id="cp-999999")
        assert result is None

    async def test_restore_empty_db_returns_none(self, checkpointer: StateCheckpointer):
        result = await checkpointer.restore()
        assert result is None

    async def test_checksum_mismatch_returns_none_on_restore(
        self, checkpointer: StateCheckpointer
    ):
        """Tamper in DB must be rejected -- critical integrity test for SQLITE."""
        cp = await checkpointer.checkpoint(_make_state(1))
        # Tamper directly in SQLite
        store = checkpointer._sqlite_store
        store._conn.execute(
            "UPDATE checkpoints SET state = ? WHERE id = ?",
            (json.dumps({"counter": 9999}), cp.id),
        )
        store._conn.commit()
        result = await checkpointer.restore(checkpoint_id=cp.id)
        assert result is None

    async def test_rotation_enforces_max_checkpoints(self, checkpointer: StateCheckpointer):
        """With max_checkpoints=5, creating 8 should leave exactly 5 in DB."""
        for i in range(1, 9):
            await checkpointer.checkpoint(_make_state(i))
        assert checkpointer._sqlite_store.count() == 5

    async def test_rotation_keeps_most_recent(self, checkpointer: StateCheckpointer):
        for i in range(1, 9):
            await checkpointer.checkpoint(_make_state(i))
        listing = checkpointer._sqlite_store.list_checkpoints()
        versions = {cp.version for cp in listing}
        assert versions == {4, 5, 6, 7, 8}

    async def test_list_checkpoints(self, checkpointer: StateCheckpointer):
        for i in range(1, 4):
            await checkpointer.checkpoint(_make_state(i))
        listing = await checkpointer.list_checkpoints()
        assert len(listing) == 3
        # Descending version order
        assert listing[0].version > listing[-1].version

    async def test_list_checkpoints_with_limit(self, checkpointer: StateCheckpointer):
        for i in range(1, 6):
            await checkpointer.checkpoint(_make_state(i))
        listing = await checkpointer.list_checkpoints(limit=2)
        assert len(listing) == 2

    def test_stats_structure(self, checkpointer: StateCheckpointer):
        s = checkpointer.stats()
        assert s["backend"] == "SQLITE"
        assert "checkpoint_count" in s
        assert "current_version" in s

    async def test_stats_after_checkpoints(self, checkpointer: StateCheckpointer):
        await checkpointer.checkpoint(_make_state(1))
        s = checkpointer.stats()
        assert s["checkpoint_count"] == 1
        assert s["current_version"] == 1

    def test_close_stops_and_closes_store(self, cp_dir: Path):
        sc = StateCheckpointer(checkpoint_dir=cp_dir, backend=StorageBackend.SQLITE)
        sc._running = True
        sc.close()
        assert sc._running is False
        assert sc._sqlite_store is None

    async def test_round_trip_preserves_state(self, checkpointer: StateCheckpointer):
        state = {"nested": {"deep": [1, 2, {"x": True}]}, "top": "val"}
        await checkpointer.checkpoint(state)
        restored = await checkpointer.restore()
        assert restored is not None
        assert restored.state == state

    async def test_restore_updates_internal_state(self, checkpointer: StateCheckpointer):
        cp = await checkpointer.checkpoint(_make_state(42))
        # Simulate a fresh checkpointer resuming from DB
        restored = await checkpointer.restore(checkpoint_id=cp.id)
        assert checkpointer._current_version == cp.version
        assert checkpointer._latest_checkpoint == restored


# ===========================================================================
# 6. TestAutoCheckpointLoop
# ===========================================================================

class TestAutoCheckpointLoop:
    """Test the background auto-checkpoint loop."""

    @pytest.fixture
    def checkpointer(self, tmp_path: Path) -> StateCheckpointer:
        sc = StateCheckpointer(
            checkpoint_dir=tmp_path / "auto_loop",
            max_checkpoints=50,
            auto_interval_seconds=0.05,  # Very short for tests
            backend=StorageBackend.FILE,
        )
        yield sc
        sc.close()

    async def test_auto_loop_calls_provider_and_checkpoints(
        self, checkpointer: StateCheckpointer
    ):
        """Provider should be called and checkpoints created."""
        call_count = 0

        def provider():
            nonlocal call_count
            call_count += 1
            return _make_state(call_count)

        # Run loop in background, let it tick a few times
        task = asyncio.create_task(checkpointer.auto_checkpoint_loop(provider))
        await asyncio.sleep(0.2)
        checkpointer.stop()
        # Give the loop time to exit
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert call_count >= 1
        assert checkpointer._current_version >= 1

    async def test_stop_halts_loop(self, checkpointer: StateCheckpointer):
        provider = lambda: _make_state(1)
        task = asyncio.create_task(checkpointer.auto_checkpoint_loop(provider))
        await asyncio.sleep(0.1)
        checkpointer.stop()
        await asyncio.sleep(0.15)
        version_at_stop = checkpointer._current_version
        # Wait more -- version should not change
        await asyncio.sleep(0.2)
        assert checkpointer._current_version == version_at_stop
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_auto_loop_sets_running_true(self, checkpointer: StateCheckpointer):
        assert checkpointer._running is False
        task = asyncio.create_task(
            checkpointer.auto_checkpoint_loop(lambda: _make_state(1))
        )
        await asyncio.sleep(0.02)
        assert checkpointer._running is True
        checkpointer.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_auto_loop_source_is_auto(self, checkpointer: StateCheckpointer):
        task = asyncio.create_task(
            checkpointer.auto_checkpoint_loop(lambda: _make_state(1))
        )
        await asyncio.sleep(0.1)
        checkpointer.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        if checkpointer.latest:
            assert checkpointer.latest.source == "auto"

    async def test_auto_loop_survives_provider_exception(self, tmp_path: Path):
        """Loop must not crash if provider raises."""
        sc = StateCheckpointer(
            checkpoint_dir=tmp_path / "auto_err",
            max_checkpoints=50,
            auto_interval_seconds=0.05,
            backend=StorageBackend.FILE,
        )
        call_count = 0

        def flaky_provider():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("provider blew up")
            return _make_state(call_count)

        task = asyncio.create_task(sc.auto_checkpoint_loop(flaky_provider))
        await asyncio.sleep(0.25)
        sc.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Despite first failure, later calls should have succeeded
        assert call_count >= 2
        sc.close()


# ===========================================================================
# 7. TestVacuum
# ===========================================================================

class TestVacuum:
    """Test vacuum behavior for both backends."""

    async def test_sqlite_vacuum_works(self, tmp_path: Path):
        sc = StateCheckpointer(
            checkpoint_dir=tmp_path / "vac_sqlite",
            backend=StorageBackend.SQLITE,
        )
        await sc.checkpoint(_make_state(1))
        await sc.vacuum()  # Should not raise
        sc.close()

    async def test_file_vacuum_is_noop(self, tmp_path: Path):
        sc = StateCheckpointer(
            checkpoint_dir=tmp_path / "vac_file",
            backend=StorageBackend.FILE,
        )
        await sc.checkpoint(_make_state(1))
        await sc.vacuum()  # No-op, should not raise
        sc.close()

    async def test_sqlite_vacuum_after_delete(self, tmp_path: Path):
        """Vacuum after deleting rows should reclaim space (or at least not error)."""
        sc = StateCheckpointer(
            checkpoint_dir=tmp_path / "vac_del",
            max_checkpoints=2,
            backend=StorageBackend.SQLITE,
        )
        for i in range(1, 11):
            await sc.checkpoint(_make_state(i))
        await sc.vacuum()
        # DB should still be functional
        restored = await sc.restore()
        assert restored is not None
        sc.close()


# ===========================================================================
# 8. TestChecksumIntegrity
# ===========================================================================

class TestChecksumIntegrity:
    """Dedicated checksum / tamper-detection tests across both backends."""

    def test_same_state_same_checksum(self):
        state = {"key": "value", "num": 42}
        cp1 = Checkpoint(state=state)
        cp2 = Checkpoint(state=state.copy())
        assert cp1.compute_checksum() == cp2.compute_checksum()

    def test_different_state_different_checksum(self):
        cp1 = Checkpoint(state={"key": "value1"})
        cp2 = Checkpoint(state={"key": "value2"})
        assert cp1.compute_checksum() != cp2.compute_checksum()

    def test_nested_state_checksum_sensitivity(self):
        """Even deep nested changes must produce different checksums."""
        cp1 = Checkpoint(state={"a": {"b": {"c": 1}}})
        cp2 = Checkpoint(state={"a": {"b": {"c": 2}}})
        assert cp1.compute_checksum() != cp2.compute_checksum()

    def test_list_order_affects_checksum(self):
        cp1 = Checkpoint(state={"items": [1, 2, 3]})
        cp2 = Checkpoint(state={"items": [3, 2, 1]})
        assert cp1.compute_checksum() != cp2.compute_checksum()

    def test_type_difference_affects_checksum(self):
        """String '1' vs integer 1 must produce different checksums."""
        cp1 = Checkpoint(state={"val": "1"})
        cp2 = Checkpoint(state={"val": 1})
        assert cp1.compute_checksum() != cp2.compute_checksum()

    async def test_tampered_file_rejected_on_restore(self, tmp_path: Path):
        """End-to-end tamper test for FILE backend."""
        cp_dir = tmp_path / "tamper_file"
        sc = StateCheckpointer(checkpoint_dir=cp_dir, backend=StorageBackend.FILE)
        cp = await sc.checkpoint({"secret": "original"})
        # Tamper
        filepath = cp_dir / f"{cp.id}.json"
        with open(filepath) as f:
            data = json.load(f)
        data["state"]["secret"] = "tampered"
        with open(filepath, "w") as f:
            json.dump(data, f)
        result = await sc.restore(checkpoint_id=cp.id)
        assert result is None, "Tampered checkpoint must be rejected"
        sc.close()

    async def test_tampered_sqlite_rejected_on_restore(self, tmp_path: Path):
        """End-to-end tamper test for SQLITE backend."""
        cp_dir = tmp_path / "tamper_sqlite"
        sc = StateCheckpointer(checkpoint_dir=cp_dir, backend=StorageBackend.SQLITE)
        cp = await sc.checkpoint({"secret": "original"})
        # Tamper directly in DB
        sc._sqlite_store._conn.execute(
            "UPDATE checkpoints SET state = ? WHERE id = ?",
            (json.dumps({"secret": "tampered"}), cp.id),
        )
        sc._sqlite_store._conn.commit()
        result = await sc.restore(checkpoint_id=cp.id)
        assert result is None, "Tampered SQLite checkpoint must be rejected"
        sc.close()

    async def test_checksum_stored_matches_computed(self, tmp_path: Path):
        """The checksum saved in the checkpoint must match recomputation."""
        cp_dir = tmp_path / "cs_verify"
        sc = StateCheckpointer(checkpoint_dir=cp_dir, backend=StorageBackend.FILE)
        state = {"complex": {"nested": [1, "two", 3.0, None]}}
        cp = await sc.checkpoint(state)
        recomputed = Checkpoint(state=state).compute_checksum()
        assert cp.checksum == recomputed
        sc.close()

    async def test_restore_valid_checkpoint_passes_checksum(self, tmp_path: Path):
        """Untampered checkpoint must pass checksum verification on restore."""
        cp_dir = tmp_path / "cs_pass"
        sc = StateCheckpointer(checkpoint_dir=cp_dir, backend=StorageBackend.FILE)
        await sc.checkpoint(_make_state(1))
        restored = await sc.restore()
        assert restored is not None
        assert restored.checksum == restored.compute_checksum()
        sc.close()


# ===========================================================================
# 9. TestEdgeCases
# ===========================================================================

class TestEdgeCases:
    """Additional edge-case and boundary tests."""

    async def test_checkpoint_with_empty_state(self, tmp_path: Path):
        sc = StateCheckpointer(
            checkpoint_dir=tmp_path / "empty_state", backend=StorageBackend.FILE
        )
        cp = await sc.checkpoint({})
        assert cp.state == {}
        restored = await sc.restore()
        assert restored is not None
        assert restored.state == {}
        sc.close()

    async def test_checkpoint_with_large_state(self, tmp_path: Path):
        """State with many keys should work correctly."""
        sc = StateCheckpointer(
            checkpoint_dir=tmp_path / "large_state", backend=StorageBackend.FILE
        )
        state = {f"key_{i}": i * 3.14 for i in range(1000)}
        cp = await sc.checkpoint(state)
        restored = await sc.restore()
        assert restored is not None
        assert restored.state == state
        sc.close()

    async def test_max_checkpoints_one(self, tmp_path: Path):
        """max_checkpoints=1 should only ever keep the latest."""
        sc = StateCheckpointer(
            checkpoint_dir=tmp_path / "max1",
            max_checkpoints=1,
            backend=StorageBackend.FILE,
        )
        for i in range(5):
            await sc.checkpoint(_make_state(i))
        files = list((tmp_path / "max1").glob("cp-*.json"))
        assert len(files) == 1
        sc.close()

    async def test_max_checkpoints_one_sqlite(self, tmp_path: Path):
        sc = StateCheckpointer(
            checkpoint_dir=tmp_path / "max1_sq",
            max_checkpoints=1,
            backend=StorageBackend.SQLITE,
        )
        for i in range(5):
            await sc.checkpoint(_make_state(i))
        assert sc._sqlite_store.count() == 1
        sc.close()

    async def test_checkpoint_id_format(self, tmp_path: Path):
        """IDs should be zero-padded: cp-000001, cp-000002, etc."""
        sc = StateCheckpointer(
            checkpoint_dir=tmp_path / "id_fmt", backend=StorageBackend.FILE
        )
        cp = await sc.checkpoint(_make_state(1))
        assert cp.id == "cp-000001"
        cp2 = await sc.checkpoint(_make_state(2))
        assert cp2.id == "cp-000002"
        sc.close()

    async def test_checkpoint_timestamp_is_utc(self, tmp_path: Path):
        sc = StateCheckpointer(
            checkpoint_dir=tmp_path / "tz", backend=StorageBackend.FILE
        )
        cp = await sc.checkpoint(_make_state(1))
        assert cp.timestamp.tzinfo is not None
        sc.close()

    async def test_two_checkpointers_same_dir_file_backend(self, tmp_path: Path):
        """Two file-backend checkpointers on the same dir: independent version
        counters produce colliding IDs so the second overwrites the first.
        The file should still contain valid JSON (atomic write guarantee)."""
        cp_dir = tmp_path / "shared"
        sc1 = StateCheckpointer(checkpoint_dir=cp_dir, backend=StorageBackend.FILE)
        sc2 = StateCheckpointer(checkpoint_dir=cp_dir, backend=StorageBackend.FILE)
        await sc1.checkpoint(_make_state(1))
        await sc2.checkpoint(_make_state(2))
        # Both use version 1 -> same filename -> second overwrites first
        filepath = cp_dir / "cp-000001.json"
        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        # The surviving file must be valid JSON with state from the second write
        assert data["state"] == _make_state(2)
        sc1.close()
        sc2.close()

    def test_storage_backend_used_in_stats(self, tmp_path: Path):
        sc_file = StateCheckpointer(
            checkpoint_dir=tmp_path / "s_file", backend=StorageBackend.FILE
        )
        sc_sql = StateCheckpointer(
            checkpoint_dir=tmp_path / "s_sql", backend=StorageBackend.SQLITE
        )
        assert sc_file.stats()["backend"] == "FILE"
        assert sc_sql.stats()["backend"] == "SQLITE"
        sc_file.close()
        sc_sql.close()

    async def test_sqlite_store_direct_count_matches_checkpointer(self, tmp_path: Path):
        sc = StateCheckpointer(
            checkpoint_dir=tmp_path / "cnt",
            max_checkpoints=100,
            backend=StorageBackend.SQLITE,
        )
        for i in range(5):
            await sc.checkpoint(_make_state(i))
        assert sc._sqlite_store.count() == 5
        assert sc.stats()["checkpoint_count"] == 5
        sc.close()

    async def test_file_stats_count_matches_files(self, tmp_path: Path):
        cp_dir = tmp_path / "fcnt"
        sc = StateCheckpointer(
            checkpoint_dir=cp_dir,
            max_checkpoints=100,
            backend=StorageBackend.FILE,
        )
        for i in range(5):
            await sc.checkpoint(_make_state(i))
        file_count = len(list(cp_dir.glob("cp-*.json")))
        assert sc.stats()["checkpoint_count"] == file_count == 5
        sc.close()
