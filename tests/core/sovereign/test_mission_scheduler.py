"""
Tests for the MissionScheduler (Task 2.1 + 2.2).

Covers:
- Mission loading from YAML and fallback defaults
- Schedule parsing (daily, weekly, cron-like)
- Mission persistence (SQLite survive restart)
- Scheduler lifecycle (start, stop)
- Mission execution with mock execute_fn
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.sovereign.mission_scheduler import (
    MissionDefinition,
    MissionPersistence,
    MissionRecord,
    MissionScheduler,
    _next_daily_time,
    _next_weekday_time,
    _parse_schedule,
)
from core.sovereign.proactive_scheduler import ScheduleType

# ---------------------------------------------------------------------------
# Schedule parsing
# ---------------------------------------------------------------------------


class TestScheduleParsing:
    def test_daily_time(self) -> None:
        stype, interval, run_at = _parse_schedule("08:00")
        assert stype == ScheduleType.RECURRING
        assert interval == 24 * 3600.0
        assert run_at is not None
        assert run_at.hour == 8
        assert run_at.minute == 0

    def test_weekly_time(self) -> None:
        stype, interval, run_at = _parse_schedule("Friday 17:00")
        assert stype == ScheduleType.RECURRING
        assert interval == 7 * 24 * 3600.0
        assert run_at is not None
        assert run_at.hour == 17
        assert run_at.minute == 0
        # Should be a Friday
        assert run_at.strftime("%A") == "Friday"

    def test_cron_interval(self) -> None:
        stype, interval, run_at = _parse_schedule("*/15 * * * *")
        assert stype == ScheduleType.RECURRING
        assert interval == 15 * 60.0
        assert run_at is None

    def test_invalid_schedule_fallback(self) -> None:
        stype, interval, run_at = _parse_schedule("invalid")
        assert stype == ScheduleType.ONE_TIME

    def test_next_daily_time_is_future(self) -> None:
        target = _next_daily_time(8, 0)
        assert target > datetime.now(timezone.utc)

    def test_next_weekday_time_is_future(self) -> None:
        target = _next_weekday_time("Monday", 9, 0)
        assert target > datetime.now(timezone.utc)
        assert target.strftime("%A") == "Monday"


# ---------------------------------------------------------------------------
# Mission persistence (Task 2.2)
# ---------------------------------------------------------------------------


class TestMissionPersistence:
    @pytest.fixture
    def db(self, tmp_path: Path) -> MissionPersistence:
        p = MissionPersistence(tmp_path / "test_missions.db")
        yield p
        p.close()

    def test_create_and_retrieve(self, db: MissionPersistence) -> None:
        record = MissionRecord(
            mission_name="morning-brief",
            cron="08:00",
            status="pending",
        )
        record_id = db.upsert(record)
        assert record_id > 0

        retrieved = db.get_by_name("morning-brief")
        assert retrieved is not None
        assert retrieved.mission_name == "morning-brief"
        assert retrieved.cron == "08:00"

    def test_update_existing(self, db: MissionPersistence) -> None:
        record = MissionRecord(
            mission_name="health-check",
            cron="*/15 * * * *",
            status="pending",
        )
        record_id = db.upsert(record)
        record.id = record_id
        record.status = "completed"
        record.ihsan_score = 0.97
        db.upsert(record)

        retrieved = db.get_by_name("health-check")
        assert retrieved is not None
        assert retrieved.status == "completed"
        assert retrieved.ihsan_score == 0.97

    def test_get_pending(self, db: MissionPersistence) -> None:
        for name in ["a", "b", "c"]:
            db.upsert(MissionRecord(mission_name=name, status="pending"))
        db.upsert(MissionRecord(mission_name="done", status="completed"))

        pending = db.get_pending()
        assert len(pending) == 3
        assert all(r.status == "pending" for r in pending)

    def test_get_history(self, db: MissionPersistence) -> None:
        for i in range(5):
            db.upsert(
                MissionRecord(
                    mission_name=f"mission-{i}",
                    status="completed",
                    created_at=time.time() + i,
                )
            )
        history = db.get_history(limit=3)
        assert len(history) == 3

    def test_survive_restart(self, tmp_path: Path) -> None:
        """Missions survive restart (close and reopen DB)."""
        db_path = tmp_path / "restart_test.db"
        db1 = MissionPersistence(db_path)
        db1.upsert(MissionRecord(mission_name="persist-test", status="pending"))
        db1.close()

        db2 = MissionPersistence(db_path)
        retrieved = db2.get_by_name("persist-test")
        db2.close()

        assert retrieved is not None
        assert retrieved.mission_name == "persist-test"


# ---------------------------------------------------------------------------
# MissionScheduler
# ---------------------------------------------------------------------------


class TestMissionScheduler:
    @pytest.fixture
    def scheduler(self, tmp_path: Path) -> MissionScheduler:
        return MissionScheduler(
            commands_yaml=Path("nonexistent.yaml"),  # Will use defaults
            mission_db=tmp_path / "scheduler_test.db",
        )

    def test_load_default_missions(self, scheduler: MissionScheduler) -> None:
        count = scheduler.load_missions()
        assert count >= 3  # morning-brief, standup, health-check

    def test_list_missions(self, scheduler: MissionScheduler) -> None:
        scheduler.load_missions()
        missions = scheduler.list_missions()
        assert len(missions) >= 3
        names = [m["name"] for m in missions]
        assert "morning-brief" in names
        assert "health-check" in names

    def test_get_mission(self, scheduler: MissionScheduler) -> None:
        scheduler.load_missions()
        morning = scheduler.get_mission("morning-brief")
        assert morning is not None
        assert morning.schedule == "08:00"
        assert "strategist" in morning.agents

    @pytest.mark.asyncio
    async def test_start_and_stop(self, scheduler: MissionScheduler) -> None:
        scheduler.load_missions()
        # Start in background task
        task = asyncio.create_task(scheduler.start())
        await asyncio.sleep(0.1)
        assert scheduler.is_running
        await scheduler.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_mission_with_execute_fn(self, tmp_path: Path) -> None:
        """Mission fires through execute_fn and records result."""
        mock_execute = AsyncMock(
            return_value={
                "ihsan_score": 0.96,
                "snr_score": 0.92,
                "total_tokens": 150,
                "success_count": 3,
                "total_count": 3,
            }
        )

        scheduler = MissionScheduler(
            commands_yaml=Path("nonexistent.yaml"),
            mission_db=tmp_path / "exec_test.db",
            execute_fn=mock_execute,
            auto_execute_tau=0.60,  # Lower threshold so base confidence (0.70) passes
        )
        scheduler.load_missions()

        # Manually invoke the morning-brief handler
        mission = scheduler.get_mission("morning-brief")
        assert mission is not None
        handler = scheduler._make_handler("morning-brief", mission)
        result = await handler()

        assert result["ihsan_score"] == 0.96
        assert result["total_tokens"] == 150
        mock_execute.assert_called_once()

        # Check persistence
        history = scheduler.get_history(limit=5)
        completed = [h for h in history if h.status == "completed"]
        assert len(completed) >= 1
        assert completed[0].ihsan_score == 0.96

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_pek_gate_skips_low_confidence(self, tmp_path: Path) -> None:
        """Missions with low confidence are skipped."""
        mock_execute = AsyncMock()

        scheduler = MissionScheduler(
            commands_yaml=Path("nonexistent.yaml"),
            mission_db=tmp_path / "skip_test.db",
            execute_fn=mock_execute,
            min_confidence=0.99,  # Impossibly high threshold
        )
        scheduler.load_missions()

        mission = scheduler.get_mission("standup")
        assert mission is not None
        handler = scheduler._make_handler("standup", mission)
        result = await handler()

        assert result["status"] == "skipped"
        mock_execute.assert_not_called()
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_pek_gate_proposes_medium_confidence(self, tmp_path: Path) -> None:
        """Missions between min and auto_execute go to approval queue."""
        mock_execute = AsyncMock()

        scheduler = MissionScheduler(
            commands_yaml=Path("nonexistent.yaml"),
            mission_db=tmp_path / "approve_test.db",
            execute_fn=mock_execute,
            min_confidence=0.50,
            auto_execute_tau=0.90,  # Base confidence 0.70 lands in "propose" zone
        )
        scheduler.load_missions()

        mission = scheduler.get_mission("standup")
        assert mission is not None
        handler = scheduler._make_handler("standup", mission)
        result = await handler()

        assert result["status"] == "pending_approval"
        mock_execute.assert_not_called()

        # Check approval queue has an item
        item = await scheduler.get_approval_queue().get()
        assert item["name"] == "standup"  # React reads mission.name
        assert item["mission_name"] == "standup"  # Backward compat
        assert "confidence" in item
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_health_check_always_high_confidence(self, tmp_path: Path) -> None:
        """Health-check missions always have high confidence."""
        mock_execute = AsyncMock(
            return_value={
                "ihsan_score": 0.98,
                "snr_score": 0.95,
                "total_tokens": 10,
                "success_count": 1,
                "total_count": 1,
            }
        )

        scheduler = MissionScheduler(
            commands_yaml=Path("nonexistent.yaml"),
            mission_db=tmp_path / "health_test.db",
            execute_fn=mock_execute,
        )
        scheduler.load_missions()

        mission = scheduler.get_mission("health-check")
        assert mission is not None
        handler = scheduler._make_handler("health-check", mission)
        result = await handler()

        # health-check confidence is >= 0.85 (above auto_execute_tau of 0.75)
        assert result.get("ihsan_score") == 0.98
        mock_execute.assert_called_once()
        await scheduler.stop()

    def test_mission_count(self, scheduler: MissionScheduler) -> None:
        scheduler.load_missions()
        assert scheduler.mission_count >= 3
