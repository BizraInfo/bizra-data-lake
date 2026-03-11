"""
MissionScheduler — Cron-like daemon for proactive mission execution
===================================================================
Reads proactive command definitions from slash_commands.yaml and
schedules them through ProactiveScheduler with PEK confidence gates.

Wire: slash_commands.yaml -> MissionScheduler -> ProactiveScheduler ->
      Node0ProactiveKernel._execute_mission() -> ActionReceipt

Standing on Giants:
- Boyd (OODA: observe-orient-decide-act loop)
- Shannon (SNR: signal quality gates)
- Lamport (hash-chained receipts)

Created: 2026-02-22 | BIZRA MissionScheduler v1.0
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD
from core.sovereign.proactive_scheduler import (
    JobPriority,
    ProactiveScheduler,
    ScheduleType,
)

logger = logging.getLogger("bizra.mission_scheduler")

# Default paths
DEFAULT_COMMANDS_YAML = Path("bizra-omega/bizra-cli/config/slash_commands.yaml")
DEFAULT_MISSION_DB = Path("sovereign_state/missions.db")


@dataclass
class MissionDefinition:
    """A proactive mission loaded from slash_commands.yaml."""

    name: str
    description: str
    schedule: str  # "08:00", "09:00", "Friday 17:00", "*/15 * * * *"
    agents: list[str] = field(default_factory=list)
    includes: list[str] = field(default_factory=list)
    alerts_on: list[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class MissionRecord:
    """Persisted mission execution record."""

    id: int = 0
    mission_name: str = ""
    cron: str = ""
    last_run: Optional[float] = None
    next_run: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed, skipped
    result_summary: str = ""
    ihsan_score: float = 0.0
    snr_score: float = 0.0
    total_tokens: int = 0
    agents_used: str = ""
    created_at: float = field(default_factory=time.time)


class MissionPersistence:
    """SQLite persistence for mission queue (Task 2.2).

    Missions survive restart. Table: proactive_missions.
    """

    def __init__(self, db_path: Path = DEFAULT_MISSION_DB):
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS proactive_missions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mission_name TEXT NOT NULL,
                cron TEXT NOT NULL DEFAULT '',
                last_run REAL,
                next_run REAL,
                status TEXT NOT NULL DEFAULT 'pending',
                result_summary TEXT DEFAULT '',
                ihsan_score REAL DEFAULT 0.0,
                snr_score REAL DEFAULT 0.0,
                total_tokens INTEGER DEFAULT 0,
                agents_used TEXT DEFAULT '',
                created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
            )
        """)
        self._conn.commit()

    def upsert(self, record: MissionRecord) -> int:
        """Insert or update a mission record."""
        if record.id:
            self._conn.execute(
                """UPDATE proactive_missions
                   SET last_run = ?, next_run = ?, status = ?,
                       result_summary = ?, ihsan_score = ?, snr_score = ?,
                       total_tokens = ?, agents_used = ?
                   WHERE id = ?""",
                (
                    record.last_run,
                    record.next_run,
                    record.status,
                    record.result_summary,
                    record.ihsan_score,
                    record.snr_score,
                    record.total_tokens,
                    record.agents_used,
                    record.id,
                ),
            )
            self._conn.commit()
            return record.id
        else:
            cursor = self._conn.execute(
                """INSERT INTO proactive_missions
                   (mission_name, cron, last_run, next_run, status,
                    result_summary, ihsan_score, snr_score, total_tokens,
                    agents_used, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.mission_name,
                    record.cron,
                    record.last_run,
                    record.next_run,
                    record.status,
                    record.result_summary,
                    record.ihsan_score,
                    record.snr_score,
                    record.total_tokens,
                    record.agents_used,
                    record.created_at,
                ),
            )
            self._conn.commit()
            return cursor.lastrowid or 0

    def get_by_name(self, name: str) -> Optional[MissionRecord]:
        """Get most recent mission record by name."""
        row = self._conn.execute(
            """SELECT * FROM proactive_missions
               WHERE mission_name = ?
               ORDER BY created_at DESC LIMIT 1""",
            (name,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def get_pending(self) -> list[MissionRecord]:
        """Get all pending missions ordered by next_run."""
        rows = self._conn.execute("""SELECT * FROM proactive_missions
               WHERE status = 'pending'
               ORDER BY next_run ASC""").fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_history(self, limit: int = 20) -> list[MissionRecord]:
        """Get recent mission history."""
        rows = self._conn.execute(
            """SELECT * FROM proactive_missions
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row: sqlite3.Row) -> MissionRecord:
        return MissionRecord(
            id=row["id"],
            mission_name=row["mission_name"],
            cron=row["cron"],
            last_run=row["last_run"],
            next_run=row["next_run"],
            status=row["status"],
            result_summary=row["result_summary"],
            ihsan_score=row["ihsan_score"],
            snr_score=row["snr_score"],
            total_tokens=row["total_tokens"],
            agents_used=row["agents_used"],
            created_at=row["created_at"],
        )

    def close(self) -> None:
        self._conn.close()


def _parse_schedule(
    schedule: str,
) -> tuple[ScheduleType, Optional[float], Optional[datetime]]:
    """Parse a schedule string into ProactiveScheduler parameters.

    Supports:
    - "HH:MM" -> daily at that time
    - "DayName HH:MM" -> weekly on that day
    - "*/N * * * *" -> every N minutes (cron-like)
    """
    import re

    # Every N minutes: "*/15 * * * *"
    cron_match = re.match(r"\*/(\d+)", schedule)
    if cron_match:
        minutes = int(cron_match.group(1))
        return ScheduleType.RECURRING, float(minutes * 60), None

    # Weekly: "Friday 17:00"
    day_match = re.match(
        r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+(\d{1,2}):(\d{2})",
        schedule,
        re.IGNORECASE,
    )
    if day_match:
        day_name = day_match.group(1)
        hour = int(day_match.group(2))
        minute = int(day_match.group(3))
        # Calculate next occurrence
        next_run = _next_weekday_time(day_name, hour, minute)
        # Weekly interval
        return ScheduleType.RECURRING, 7 * 24 * 3600.0, next_run

    # Daily: "08:00"
    time_match = re.match(r"(\d{1,2}):(\d{2})", schedule)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        next_run = _next_daily_time(hour, minute)
        return ScheduleType.RECURRING, 24 * 3600.0, next_run

    # Fallback: one-time immediate
    logger.warning("Unparseable schedule '%s', defaulting to one-time", schedule)
    return ScheduleType.ONE_TIME, None, None


def _next_daily_time(hour: int, minute: int) -> datetime:
    """Next occurrence of HH:MM (UTC)."""
    now = datetime.now(timezone.utc)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def _next_weekday_time(day_name: str, hour: int, minute: int) -> datetime:
    """Next occurrence of 'DayName HH:MM' (UTC)."""
    days = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    target_day = days.get(day_name.lower(), 0)
    now = datetime.now(timezone.utc)
    days_ahead = (target_day - now.weekday()) % 7
    if days_ahead == 0:
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            days_ahead = 7
        else:
            return target
    return (now + timedelta(days=days_ahead)).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )


class MissionScheduler:
    """Cron-like daemon reading proactive commands from slash_commands.yaml.

    Lifecycle:
    1. load_missions() — parse YAML, create MissionDefinitions
    2. start() — register jobs with ProactiveScheduler, begin loop
    3. stop() — graceful shutdown

    Each mission fires through the PEK confidence gate:
    - confidence >= auto_execute_tau (0.75) -> auto-execute
    - confidence >= min_confidence (0.58) -> propose to user
    - confidence < min_confidence -> skip

    Standing on Giants:
    - Boyd (OODA: scheduled missions are the "orient" phase)
    - Shannon (SNR: missions must pass quality gates)
    """

    def __init__(
        self,
        commands_yaml: Path = DEFAULT_COMMANDS_YAML,
        mission_db: Path = DEFAULT_MISSION_DB,
        execute_fn: Optional[Callable] = None,
        auto_execute_tau: float = 0.75,
        min_confidence: float = 0.58,
    ):
        self._commands_yaml = commands_yaml
        self._persistence = MissionPersistence(mission_db)
        self._scheduler = ProactiveScheduler(max_concurrent=3, check_interval=5.0)
        self._missions: Dict[str, MissionDefinition] = {}
        self._job_ids: Dict[str, str] = {}  # mission_name -> job_id
        self._execute_fn = execute_fn  # Node0ProactiveKernel._execute_mission
        self._auto_execute_tau = auto_execute_tau
        self._min_confidence = min_confidence
        self._approval_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._running = False

    def load_missions(self) -> int:
        """Load proactive mission definitions from slash_commands.yaml.

        Returns the number of missions loaded.
        """
        try:
            import yaml

            with open(self._commands_yaml) as f:
                data = yaml.safe_load(f) or {}
        except (FileNotFoundError, ImportError) as exc:
            logger.warning("Cannot load slash_commands.yaml: %s", exc)
            # Fallback: hardcoded default missions
            return self._load_default_missions()

        proactive = data.get("proactive", {})
        count = 0
        for name, config in proactive.items():
            mission = MissionDefinition(
                name=name,
                description=config.get("description", name),
                schedule=config.get("schedule", "08:00"),
                agents=config.get("agents", []),
                includes=config.get("includes", []),
                alerts_on=config.get("alerts_on", []),
            )
            self._missions[name] = mission
            count += 1
            logger.info("Loaded mission: %s @ %s", name, mission.schedule)

        return count

    def _load_default_missions(self) -> int:
        """Hardcoded fallback missions when YAML is unavailable."""
        defaults = [
            MissionDefinition(
                name="morning-brief",
                description="Daily morning briefing",
                schedule="08:00",
                agents=["strategist", "analyst", "guardian"],
                includes=["overnight_alerts", "priority_tasks", "calendar_summary"],
            ),
            MissionDefinition(
                name="standup",
                description="Daily standup summary",
                schedule="09:00",
                agents=["developer", "reviewer"],
                includes=["yesterday_completed", "today_planned", "blockers"],
            ),
            MissionDefinition(
                name="health-check",
                description="System health check",
                schedule="*/15 * * * *",
                agents=["guardian"],
                alerts_on=["service_down", "high_latency"],
            ),
        ]
        for m in defaults:
            self._missions[m.name] = m
        return len(defaults)

    async def start(self) -> None:
        """Register all missions with ProactiveScheduler and start the loop."""
        if not self._missions:
            self.load_missions()

        for name, mission in self._missions.items():
            if not mission.enabled:
                continue

            schedule_type, interval, run_at = _parse_schedule(mission.schedule)

            job_id = self._scheduler.schedule(
                name=f"mission:{name}",
                handler=self._make_handler(name, mission),
                schedule_type=schedule_type,
                priority=JobPriority.NORMAL,
                run_at=run_at,
                interval=interval,
                metadata={
                    "mission_name": name,
                    "agents": mission.agents,
                    "includes": mission.includes,
                },
            )
            self._job_ids[name] = job_id

            # Persist the scheduled mission
            record = MissionRecord(
                mission_name=name,
                cron=mission.schedule,
                next_run=run_at.timestamp() if run_at else time.time(),
                status="pending",
            )
            self._persistence.upsert(record)

        self._running = True
        logger.info("MissionScheduler started with %d missions", len(self._missions))

        # Start scheduler in background
        asyncio.create_task(self._scheduler.start())

    def _make_handler(self, name: str, mission: MissionDefinition) -> Callable:
        """Create an async handler for a mission that respects PEK gates.

        PEK confidence gate (from runtime_types.py):
        - confidence >= auto_execute_tau (0.75) -> auto-execute
        - confidence >= min_confidence (0.58) -> propose to user via approval queue
        - confidence < min_confidence -> skip (below attention threshold)

        Standing on Giants: Boyd (OODA decide phase)
        """

        async def _handler() -> dict[str, Any]:
            logger.info("Mission firing: %s", name)

            # PEK confidence estimation based on mission readiness signals
            confidence = self._estimate_confidence(name, mission)
            logger.info(
                "Mission %s confidence=%.2f (auto_tau=%.2f, min=%.2f)",
                name,
                confidence,
                self._auto_execute_tau,
                self._min_confidence,
            )

            # Gate 1: Below minimum confidence -> skip
            if confidence < self._min_confidence:
                logger.info(
                    "Mission %s skipped: confidence %.2f < min %.2f",
                    name,
                    confidence,
                    self._min_confidence,
                )
                record = MissionRecord(
                    mission_name=name,
                    cron=mission.schedule,
                    last_run=time.time(),
                    status="skipped",
                    result_summary=f"confidence={confidence:.2f} < min={self._min_confidence:.2f}",
                )
                self._persistence.upsert(record)
                return {"status": "skipped", "confidence": confidence}

            # Gate 2: Between min and auto_execute -> propose to user
            if confidence < self._auto_execute_tau:
                logger.info(
                    "Mission %s proposed for approval: confidence %.2f",
                    name,
                    confidence,
                )
                approval_item = {
                    "name": name,  # React reads mission.name
                    "mission_name": name,  # Backward compat
                    "description": mission.description,
                    "agents": mission.agents,
                    "includes": mission.includes,
                    "schedule": mission.schedule,
                    "confidence": confidence,
                    "timestamp": time.time(),
                }
                await self._approval_queue.put(approval_item)
                record = MissionRecord(
                    mission_name=name,
                    cron=mission.schedule,
                    last_run=time.time(),
                    status="pending_approval",
                    result_summary=f"confidence={confidence:.2f}, awaiting user approval",
                )
                self._persistence.upsert(record)
                return {"status": "pending_approval", "confidence": confidence}

            # Gate 3: Above auto_execute_tau -> execute automatically
            logger.info(
                "Mission %s auto-executing: confidence %.2f >= %.2f",
                name,
                confidence,
                self._auto_execute_tau,
            )

            # Build mission dict for execute_fn
            mission_dict = {
                "id": f"mission-{name}-{int(time.time())}",
                "description": mission.description,
                "priority": "normal",
                "status": "running",
                "includes": mission.includes,
                "confidence": confidence,
            }

            record = MissionRecord(
                mission_name=name,
                cron=mission.schedule,
                last_run=time.time(),
                status="running",
                agents_used=",".join(mission.agents),
            )
            record_id = self._persistence.upsert(record)
            record.id = record_id

            # Execute via the wired execution function
            if self._execute_fn:
                try:
                    result = await self._execute_fn(mission_dict, mission.agents)
                    ihsan = result.get("ihsan_score", 0.0)
                    snr = result.get("snr_score", 0.0)

                    record.status = "completed"
                    record.ihsan_score = ihsan
                    record.snr_score = snr
                    record.total_tokens = result.get("total_tokens", 0)
                    record.result_summary = (
                        f"agents={result.get('success_count', 0)}/"
                        f"{result.get('total_count', 0)} "
                        f"ihsan={ihsan:.2f} snr={snr:.2f} "
                        f"confidence={confidence:.2f}"
                    )
                    self._persistence.upsert(record)

                    logger.info(
                        "Mission %s completed: ihsan=%.2f snr=%.2f tokens=%d",
                        name,
                        ihsan,
                        snr,
                        record.total_tokens,
                    )
                    return result

                except Exception as exc:  # noqa: BLE001 — boundary boundary
                    record.status = "failed"
                    record.result_summary = str(exc)[:256]
                    self._persistence.upsert(record)
                    logger.error("Mission %s failed: %s", name, exc)
                    return {"error": str(exc)}
            else:
                # No execute_fn wired — log-only mode
                record.status = "completed"
                record.result_summary = (
                    f"dry-run (no execute_fn) confidence={confidence:.2f}"
                )
                self._persistence.upsert(record)
                return {"status": "dry-run", "mission": name, "confidence": confidence}

        return _handler

    def _estimate_confidence(self, name: str, mission: MissionDefinition) -> float:
        """Estimate mission execution confidence based on readiness signals.

        Factors:
        - Previous execution success rate (historical)
        - Time-of-day appropriateness (morning missions at 08:00 are higher)
        - Agent availability (all agents present = higher)
        - Recent error rate (failures reduce confidence)

        Returns a float between 0.0 and 1.0.
        """
        confidence = 0.70  # Base confidence for scheduled missions

        # Historical success boost
        recent = self._persistence.get_by_name(name)
        if recent and recent.status == "completed":
            confidence += 0.15  # Previously succeeded
            if recent.ihsan_score >= UNIFIED_IHSAN_THRESHOLD:
                confidence += 0.05  # High quality previous run
        elif recent and recent.status == "failed":
            confidence -= 0.20  # Recent failure reduces confidence

        # Health-check missions are always high confidence (automated)
        if name == "health-check":
            confidence = max(confidence, 0.85)

        # Morning-brief at scheduled time gets a boost
        if name == "morning-brief":
            now_hour = datetime.now(timezone.utc).hour
            if 6 <= now_hour <= 10:
                confidence += 0.10  # Within morning window

        # Cap at 1.0
        return min(1.0, max(0.0, confidence))

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        self._scheduler.stop()
        self._persistence.close()
        logger.info("MissionScheduler stopped")

    def get_approval_queue(self) -> asyncio.Queue[dict[str, Any]]:
        """Queue for missions requiring human approval (confidence < auto_execute_tau)."""
        return self._approval_queue

    def get_mission(self, name: str) -> Optional[MissionDefinition]:
        """Get a mission definition by name."""
        return self._missions.get(name)

    def list_missions(self) -> list[dict[str, Any]]:
        """List all registered missions with their status."""
        result = []
        for name, mission in self._missions.items():
            record = self._persistence.get_by_name(name)
            result.append(
                {
                    "name": name,
                    "description": mission.description,
                    "schedule": mission.schedule,
                    "agents": mission.agents,
                    "enabled": mission.enabled,
                    "last_run": record.last_run if record else None,
                    "status": record.status if record else "never_run",
                    "ihsan_score": record.ihsan_score if record else 0.0,
                }
            )
        return result

    def get_history(self, limit: int = 20) -> list[MissionRecord]:
        """Get recent mission execution history."""
        return self._persistence.get_history(limit)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def mission_count(self) -> int:
        return len(self._missions)
