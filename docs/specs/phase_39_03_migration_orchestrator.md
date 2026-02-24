# Phase 39 — Pseudocode Module 03: Migration Orchestrator

**FR-03** | Priority: 3 | Risk: Low | New files: 1

---

## Overview

One-command orchestration of all legacy memory → AgentDB migration.
Coordinates existing adapters + migrator in sequence, with progress
callbacks and dry-run mode.

---

## Flow Diagram

```
MigrationOrchestrator.run()
  │
  ├── Phase 1: LivingMemoryAdapter.export_all()
  │     └── AgentDB.store_record() for each
  │
  ├── Phase 2: ExperienceLedgerAdapter.export_all()
  │     └── AgentDB.store_record() for each
  │
  ├── Phase 3: PatternMemoryAdapter.export_all()
  │     └── AgentDB.store_record() for each
  │
  ├── Phase 4: MemoryMigrator.migrate()  (SQLite v1 → v2)
  │     └── Batch upsert + HNSW index
  │
  └── AgentDB.save()  (persist HNSW)
```

---

## Pseudocode: `core/memory/orchestrator.py`

```
MODULE orchestrator

IMPORT logging
FROM dataclasses IMPORT dataclass, field
FROM datetime IMPORT datetime, timezone
FROM pathlib IMPORT Path
FROM typing IMPORT Callable, Dict, List, Optional

FROM .agent_db IMPORT AgentDB
FROM .adapters.living_memory IMPORT LivingMemoryAdapter
FROM .adapters.experience_ledger IMPORT ExperienceLedgerAdapter
FROM .adapters.pattern_memory IMPORT PatternMemoryAdapter
FROM .migrator IMPORT MemoryMigrator, MigrationResult
FROM .types IMPORT MemoryRecord

LOG = logging.getLogger(__name__)


@dataclass
CLASS MigrationPhaseResult:
    source: str
    records_found: int = 0
    records_imported: int = 0
    records_skipped: int = 0
    errors: int = 0
    duration_ms: float = 0.0


@dataclass
CLASS OrchestratorResult:
    phases: List[MigrationPhaseResult] = field(default_factory=list)
    total_imported: int = 0
    total_errors: int = 0
    duration_ms: float = 0.0
    dry_run: bool = False

    METHOD summary() -> str:
        lines = [f"Migration {'(DRY RUN) ' IF self.dry_run ELSE ''}complete:"]
        FOR phase IN self.phases:
            lines.append(
                f"  {phase.source}: {phase.records_imported}/{phase.records_found} "
                f"imported ({phase.errors} errors, {phase.duration_ms:.0f}ms)"
            )
        lines.append(f"  Total: {self.total_imported} imported, {self.total_errors} errors")
        RETURN "\n".join(lines)


# Progress callback type
ProgressFn = Callable[[str, int, int], None]  # (source, migrated_so_far, total)


CLASS MigrationOrchestrator:
    """Coordinates all-source migration into AgentDB.

    Usage:
        db = AgentDB(config)
        db.initialize()

        orch = MigrationOrchestrator(db)
        orch.set_living_memory(lm_core)
        orch.set_experience_ledger(sel)
        result = orch.run()
        print(result.summary())
    """

    CONSTRUCTOR(
        agent_db: AgentDB,
        on_progress: Optional[ProgressFn] = None,
    ):
        self._db = agent_db
        self._on_progress = on_progress

        # Source systems (optional — set before run())
        self._living_memory = None
        self._experience_ledger = None
        self._pattern_memory = None
        self._v1_db_path: Optional[Path] = None

    METHOD set_living_memory(lm) -> "MigrationOrchestrator":
        self._living_memory = lm
        RETURN self

    METHOD set_experience_ledger(sel) -> "MigrationOrchestrator":
        self._experience_ledger = sel
        RETURN self

    METHOD set_pattern_memory(pm) -> "MigrationOrchestrator":
        self._pattern_memory = pm
        RETURN self

    METHOD set_v1_database(path: Path) -> "MigrationOrchestrator":
        self._v1_db_path = path
        RETURN self

    METHOD run(dry_run: bool = False) -> OrchestratorResult:
        """Execute all migration phases in sequence.

        Args:
            dry_run: If True, count records without writing to AgentDB.

        Returns:
            OrchestratorResult with per-phase breakdown.
        """
        result = OrchestratorResult(dry_run=dry_run)
        start = _now_ms()

        # Phase 1: LivingMemory
        IF self._living_memory IS NOT None:
            phase = self._migrate_adapter(
                source="living_memory",
                adapter=LivingMemoryAdapter(self._living_memory),
                dry_run=dry_run,
            )
            result.phases.append(phase)

        # Phase 2: Experience Ledger
        IF self._experience_ledger IS NOT None:
            phase = self._migrate_adapter(
                source="experience_ledger",
                adapter=ExperienceLedgerAdapter(self._experience_ledger),
                dry_run=dry_run,
            )
            result.phases.append(phase)

        # Phase 3: Pattern Memory
        IF self._pattern_memory IS NOT None:
            adapter = PatternMemoryAdapter(self._pattern_memory)
            IF adapter.available:
                phase = self._migrate_adapter(
                    source="pattern_memory",
                    adapter=adapter,
                    dry_run=dry_run,
                )
                result.phases.append(phase)

        # Phase 4: SQLite v1 → v2 (uses existing MemoryMigrator)
        IF self._v1_db_path IS NOT None AND self._v1_db_path.exists():
            phase = self._migrate_v1(dry_run=dry_run)
            result.phases.append(phase)

        # Flush HNSW to disk
        IF NOT dry_run:
            self._db.save()

        # Aggregate totals
        result.total_imported = sum(p.records_imported FOR p IN result.phases)
        result.total_errors = sum(p.errors FOR p IN result.phases)
        result.duration_ms = _now_ms() - start

        LOG.info(result.summary())
        RETURN result

    METHOD _migrate_adapter(
        self,
        source: str,
        adapter,
        dry_run: bool,
    ) -> MigrationPhaseResult:
        """Generic migration from any adapter that has export_all()."""
        phase = MigrationPhaseResult(source=source)
        start = _now_ms()

        TRY:
            records: List[MemoryRecord] = adapter.export_all()
            phase.records_found = len(records)

            IF dry_run:
                phase.records_imported = 0  # Dry run — don't write
                phase.duration_ms = _now_ms() - start
                RETURN phase

            FOR i, record IN enumerate(records):
                TRY:
                    self._db.store_record(record)
                    phase.records_imported += 1
                EXCEPT Exception as e:
                    LOG.warning(f"Failed to import {source} record {record.id}: {e}")
                    phase.errors += 1

                # Progress callback
                IF self._on_progress AND (i + 1) % 100 == 0:
                    self._on_progress(source, i + 1, len(records))

        EXCEPT Exception as e:
            LOG.error(f"Adapter {source} failed: {e}")
            phase.errors += 1

        phase.duration_ms = _now_ms() - start
        RETURN phase

    METHOD _migrate_v1(dry_run: bool) -> MigrationPhaseResult:
        """Migrate SQLite v1 using existing MemoryMigrator."""
        phase = MigrationPhaseResult(source="sqlite_v1")
        start = _now_ms()

        IF dry_run:
            # Count records in source DB without migrating
            TRY:
                IMPORT sqlite3
                conn = sqlite3.connect(f"file:{self._v1_db_path}?mode=ro", uri=True)
                count = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE state != 'deleted'"
                ).fetchone()[0]
                conn.close()
                phase.records_found = count
            EXCEPT Exception as e:
                LOG.warning(f"v1 DB count failed: {e}")
                phase.errors += 1
            phase.duration_ms = _now_ms() - start
            RETURN phase

        TRY:
            migrator = MemoryMigrator(self._db, source_path=self._v1_db_path)
            result: MigrationResult = migrator.migrate()
            phase.records_found = result.source_count
            phase.records_imported = result.migrated
            phase.records_skipped = result.skipped
            phase.errors = result.errors
        EXCEPT Exception as e:
            LOG.error(f"SQLite v1 migration failed: {e}")
            phase.errors += 1

        phase.duration_ms = _now_ms() - start
        RETURN phase


FUNCTION _now_ms() -> float:
    RETURN datetime.now(timezone.utc).timestamp() * 1000
```

---

## CLI Entry Point: `core/memory/__main__.py`

```
MODULE __main__

"""CLI: python -m core.memory migrate [--dry-run] [--v1-path PATH]"""

IMPORT argparse, sys
FROM pathlib IMPORT Path

FROM .agent_db IMPORT AgentDB
FROM .config IMPORT MemoryConfig
FROM .orchestrator IMPORT MigrationOrchestrator

FUNCTION main():
    parser = argparse.ArgumentParser(prog="python -m core.memory")
    sub = parser.add_subparsers(dest="command")

    mig = sub.add_parser("migrate", help="Migrate all legacy memory into AgentDB")
    mig.add_argument("--dry-run", action="store_true", help="Count without writing")
    mig.add_argument("--v1-path", type=Path, help="Path to SQLite v1 database")

    args = parser.parse_args()

    IF args.command == "migrate":
        config = MemoryConfig()
        db = AgentDB(config)
        db.initialize()

        orch = MigrationOrchestrator(
            db,
            on_progress=LAMBDA src, done, total: print(f"  {src}: {done}/{total}")
        )

        IF args.v1_path:
            orch.set_v1_database(args.v1_path)
        ELIF config.living_memory_db:
            orch.set_v1_database(config.living_memory_db)

        result = orch.run(dry_run=args.dry_run)
        print(result.summary())
        sys.exit(1 IF result.total_errors > 0 ELSE 0)

    ELSE:
        parser.print_help()

IF __name__ == "__main__":
    main()
```

---

## TDD Anchors

```
TEST test_orchestrator_all_sources:
    # Setup: create mock LivingMemory, SEL, PatternMemory with sample data
    db = AgentDB(config)
    db.initialize()

    orch = MigrationOrchestrator(db)
    orch.set_living_memory(mock_lm)
    orch.set_experience_ledger(mock_sel)
    result = orch.run()

    ASSERT result.total_imported > 0
    ASSERT result.total_errors == 0
    ASSERT len(result.phases) == 2

TEST test_dry_run_no_writes:
    db = AgentDB(config)
    db.initialize()
    initial_count = db.count

    orch = MigrationOrchestrator(db)
    orch.set_living_memory(mock_lm)
    result = orch.run(dry_run=True)

    ASSERT result.dry_run == True
    ASSERT result.phases[0].records_found > 0
    ASSERT result.phases[0].records_imported == 0
    ASSERT db.count == initial_count  # No records written

TEST test_idempotent_double_run:
    db = AgentDB(config)
    db.initialize()

    orch = MigrationOrchestrator(db)
    orch.set_living_memory(mock_lm)

    result1 = orch.run()
    count_after_first = db.count

    result2 = orch.run()
    count_after_second = db.count

    ASSERT count_after_first == count_after_second  # Upsert = no duplication

TEST test_progress_callback_fires:
    progress_calls = []
    orch = MigrationOrchestrator(
        db,
        on_progress=LAMBDA src, done, total: progress_calls.append((src, done, total))
    )
    orch.set_living_memory(mock_lm_with_200_records)
    orch.run()

    ASSERT len(progress_calls) >= 1  # At least one callback for 200 records

TEST test_missing_source_skipped:
    orch = MigrationOrchestrator(db)
    # No sources set
    result = orch.run()
    ASSERT len(result.phases) == 0
    ASSERT result.total_imported == 0
```
