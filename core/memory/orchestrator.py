"""
Migration Orchestrator — One-command migration from all legacy memory systems.

Coordinates existing adapters (LivingMemory, SEL, PatternMemory) plus
the SQLite v1→v2 MemoryMigrator into a unified migration pipeline.

Usage:
    from core.memory.orchestrator import MigrationOrchestrator
    orch = MigrationOrchestrator(agent_db)
    orch.set_living_memory(lm_core)
    result = orch.run()
    print(result.summary())

CLI:
    python -m core.memory migrate [--dry-run] [--v1-path PATH]

Standing on Giants: ADR-006 (Unified Memory Service)
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from .adapters.claude_flow import ClaudeFlowAdapter
from .adapters.experience_ledger import ExperienceLedgerAdapter
from .adapters.living_memory import LivingMemoryAdapter
from .adapters.pattern_memory import PatternMemoryAdapter
from .adapters.reasoning_bank import ReasoningBankAdapter
from .agent_db import AgentDB
from .migrator import MemoryMigrator
from .types import MemoryRecord

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]


@dataclass
class MigrationPhaseResult:
    """Results from a single migration phase."""

    source: str
    records_found: int = 0
    records_imported: int = 0
    records_skipped: int = 0
    errors: int = 0
    duration_ms: float = 0.0
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "records_found": self.records_found,
            "records_imported": self.records_imported,
            "records_skipped": self.records_skipped,
            "errors": self.errors,
            "duration_ms": round(self.duration_ms, 3),
            "issues": self.issues,
        }


@dataclass
class OrchestratorResult:
    """Aggregated results from all migration phases."""

    phases: List[MigrationPhaseResult] = field(default_factory=list)
    total_imported: int = 0
    total_errors: int = 0
    duration_ms: float = 0.0
    dry_run: bool = False

    def summary(self) -> str:
        prefix = "(DRY RUN) " if self.dry_run else ""
        lines = [f"Migration {prefix}complete:"]
        for p in self.phases:
            issue_suffix = f", {len(p.issues)} issues" if p.issues else ""
            lines.append(
                f"  {p.source}: {p.records_imported}/{p.records_found} "
                f"imported ({p.errors} errors{issue_suffix}, {p.duration_ms:.0f}ms)"
            )
        lines.append(
            f"  Total: {self.total_imported} imported, {self.total_errors} errors"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "dry_run": self.dry_run,
            "total_imported": self.total_imported,
            "total_errors": self.total_errors,
            "duration_ms": round(self.duration_ms, 3),
            "phases": [phase.to_dict() for phase in self.phases],
        }


class MigrationOrchestrator:
    """Coordinates all-source migration into AgentDB.

    Each source runs independently; failure in one does not block others.
    Content-addressable IDs make the entire pipeline idempotent.
    """

    def __init__(
        self,
        agent_db: AgentDB,
        on_progress: Optional[ProgressFn] = None,
    ) -> None:
        self._db = agent_db
        self._on_progress = on_progress
        self._living_memory = None
        self._experience_ledger = None
        self._pattern_memory = None
        self._v1_db_path: Optional[Path] = None
        self._claude_flow_db_path: Optional[Path] = None
        self._claude_flow_artifact_dir: Optional[Path] = None
        self._reasoning_bank_dir: Optional[Path] = None
        self._strict_json = False

    def set_living_memory(self, lm) -> MigrationOrchestrator:
        self._living_memory = lm
        return self

    def set_experience_ledger(self, sel) -> MigrationOrchestrator:
        self._experience_ledger = sel
        return self

    def set_pattern_memory(self, pm) -> MigrationOrchestrator:
        self._pattern_memory = pm
        return self

    def set_v1_database(self, path: Path) -> MigrationOrchestrator:
        self._v1_db_path = path
        return self

    def set_claude_flow_db(self, path: Path) -> MigrationOrchestrator:
        self._claude_flow_db_path = path
        return self

    def set_claude_flow_artifact_dir(self, path: Path) -> MigrationOrchestrator:
        self._claude_flow_artifact_dir = path
        return self

    def set_reasoning_bank_dir(self, path: Path) -> MigrationOrchestrator:
        """Set the ReasoningBank state directory (.claude/state/reasoning_bank/)."""
        self._reasoning_bank_dir = path
        return self

    def set_strict_json(self, strict_json: bool) -> MigrationOrchestrator:
        self._strict_json = strict_json
        return self

    def run(self, dry_run: bool = False) -> OrchestratorResult:
        """Execute all migration phases in sequence."""
        result = OrchestratorResult(dry_run=dry_run)
        start = _now_ms()

        if self._living_memory is not None:
            phase = self._migrate_adapter(
                "living_memory",
                LivingMemoryAdapter(self._living_memory),
                dry_run,
            )
            result.phases.append(phase)

        if self._experience_ledger is not None:
            phase = self._migrate_adapter(
                "experience_ledger",
                ExperienceLedgerAdapter(self._experience_ledger),
                dry_run,
            )
            result.phases.append(phase)

        if self._pattern_memory is not None:
            adapter = PatternMemoryAdapter(self._pattern_memory)
            if adapter.available:
                phase = self._migrate_adapter("pattern_memory", adapter, dry_run)
                result.phases.append(phase)

        if self._v1_db_path is not None and self._v1_db_path.exists():
            phase = self._migrate_v1(dry_run)
            result.phases.append(phase)

        if self._claude_flow_db_path is not None and self._claude_flow_db_path.exists():
            phase = self._migrate_claude_flow_db(dry_run)
            result.phases.append(phase)

        if (
            self._claude_flow_artifact_dir is not None
            and self._claude_flow_artifact_dir.exists()
        ):
            phase = self._migrate_claude_flow_artifacts(dry_run)
            result.phases.append(phase)

        if self._reasoning_bank_dir is not None:
            adapter = ReasoningBankAdapter(state_dir=self._reasoning_bank_dir)
            if adapter.available:
                phase = self._migrate_adapter("reasoning_bank", adapter, dry_run)
                result.phases.append(phase)

        if not dry_run:
            self._db.save()

        result.total_imported = sum(p.records_imported for p in result.phases)
        result.total_errors = sum(p.errors for p in result.phases)
        result.duration_ms = _now_ms() - start

        logger.info(result.summary())
        return result

    def _migrate_adapter(
        self,
        source: str,
        adapter,
        dry_run: bool,
    ) -> MigrationPhaseResult:
        phase = MigrationPhaseResult(source=source)
        start = _now_ms()

        try:
            records: List[MemoryRecord] = adapter.export_all()
            phase.records_found = len(records)

            if dry_run:
                phase.duration_ms = _now_ms() - start
                return phase

            for i, record in enumerate(records):
                try:
                    self._db.store_record(record)
                    phase.records_imported += 1
                except Exception as e:  # noqa: BLE001 — boundary boundary
                    logger.warning(f"Failed to import {source} record {record.id}: {e}")
                    phase.errors += 1

                if self._on_progress and (i + 1) % 100 == 0:
                    self._on_progress(source, i + 1, len(records))

        except Exception as e:  # noqa: BLE001 — boundary boundary
            logger.error(f"Adapter {source} failed: {e}")
            phase.errors += 1

        phase.duration_ms = _now_ms() - start
        return phase

    def _migrate_claude_flow_db(self, dry_run: bool) -> MigrationPhaseResult:
        adapter = ClaudeFlowAdapter(
            db_path=self._claude_flow_db_path,
            artifact_dir=self._claude_flow_artifact_dir,
            strict_json=self._strict_json,
        )
        batch = adapter.export_db()
        phase = MigrationPhaseResult(
            source="claude_flow_db",
            records_found=len(batch.records),
            issues=[issue.to_message() for issue in batch.issues],
        )
        start = _now_ms()
        if dry_run:
            phase.duration_ms = _now_ms() - start
            return phase

        for record in batch.records:
            try:
                self._db.store_record(record)
                phase.records_imported += 1
            except Exception as exc:  # noqa: BLE001 — boundary boundary
                logger.warning(
                    "Failed to import claude_flow_db record %s: %s", record.id, exc
                )
                phase.errors += 1

        phase.duration_ms = _now_ms() - start
        return phase

    def _migrate_claude_flow_artifacts(self, dry_run: bool) -> MigrationPhaseResult:
        adapter = ClaudeFlowAdapter(
            db_path=self._claude_flow_db_path,
            artifact_dir=self._claude_flow_artifact_dir,
            strict_json=self._strict_json,
        )
        batch = adapter.export_artifacts()
        phase = MigrationPhaseResult(
            source="claude_flow_artifacts",
            records_found=len(batch.records),
            records_skipped=len(batch.issues),
            issues=[issue.to_message() for issue in batch.issues],
        )
        if self._strict_json:
            phase.errors = len(batch.issues)

        start = _now_ms()
        if dry_run:
            phase.duration_ms = _now_ms() - start
            return phase

        for record in batch.records:
            try:
                self._db.store_record(record)
                phase.records_imported += 1
            except Exception as exc:  # noqa: BLE001 — boundary boundary
                logger.warning(
                    "Failed to import claude_flow artifact record %s: %s",
                    record.id,
                    exc,
                )
                phase.errors += 1

        phase.duration_ms = _now_ms() - start
        return phase

    def _migrate_v1(self, dry_run: bool) -> MigrationPhaseResult:
        phase = MigrationPhaseResult(source="sqlite_v1")
        start = _now_ms()

        if dry_run:
            try:
                conn = sqlite3.connect(f"file:{self._v1_db_path}?mode=ro", uri=True)
                count = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE state != 'deleted'"
                ).fetchone()[0]
                conn.close()
                phase.records_found = count
            except (OSError, ConnectionError) as e:  # SEC-003 — connection boundary
                logger.warning(f"v1 DB count failed: {e}")
                phase.errors += 1
            phase.duration_ms = _now_ms() - start
            return phase

        try:
            migrator = MemoryMigrator(self._db, source_path=self._v1_db_path)
            mr = migrator.migrate()
            phase.records_found = mr.source_count
            phase.records_imported = mr.migrated
            phase.records_skipped = mr.skipped
            phase.errors = mr.errors
        except Exception as e:  # noqa: BLE001 — boundary boundary
            logger.error(f"SQLite v1 migration failed: {e}")
            phase.errors += 1

        phase.duration_ms = _now_ms() - start
        return phase


def _now_ms() -> float:
    return datetime.now(timezone.utc).timestamp() * 1000
