"""Scheduler Integration — Bridge ProactiveScheduler to SovereignRuntime."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .proactive_scheduler import JobPriority, ProactiveScheduler, ScheduleType
from .runtime_engines.sovereign_runtime import (
    RuntimeDecision,
    RuntimeInput,
    SovereignRuntime,
    get_sovereign_runtime,
)

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """A scheduled RuntimeInput task."""

    task_id: str
    cron_pattern: Optional[str]
    runtime_input: RuntimeInput
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    last_result: Optional[RuntimeDecision] = None
    execution_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def parse_cron_to_interval(pattern: str) -> float:
    """Parse cron to interval. Supports */N minute or 0 */N hour patterns."""
    parts = pattern.strip().split()
    if len(parts) == 5:
        minute, hour = parts[0], parts[1]
        if minute.startswith("*/") and hour == "*":
            return int(minute[2:]) * 60.0
        if minute == "0" and hour.startswith("*/"):
            return int(hour[2:]) * 3600.0
    return 3600.0  # Default hourly


class ProactiveSchedulerBridge:
    """Bridge connecting ProactiveScheduler to SovereignRuntime."""

    def __init__(
        self,
        scheduler: Optional[ProactiveScheduler] = None,
        runtime: Optional[SovereignRuntime] = None,
    ):
        self._scheduler = scheduler or ProactiveScheduler()
        self._runtime = runtime or get_sovereign_runtime()
        self._tasks: Dict[str, ScheduledTask] = {}
        self._history: List[Dict[str, Any]] = []

    def schedule_task(
        self,
        task_id: str,
        query: str,
        cron_pattern: Optional[str] = None,
        run_at: Optional[datetime] = None,
        priority: JobPriority = JobPriority.NORMAL,
        context: Optional[Dict[str, Any]] = None,
    ) -> ScheduledTask:
        """Schedule a RuntimeInput query for execution."""
        runtime_input = RuntimeInput(
            query=query,
            context=context or {},
            source="scheduler",
            priority=priority.value / 5.0,
        )
        schedule_type = (
            ScheduleType.RECURRING if cron_pattern else ScheduleType.ONE_TIME
        )
        interval = parse_cron_to_interval(cron_pattern) if cron_pattern else None

        task = ScheduledTask(
            task_id=task_id,
            cron_pattern=cron_pattern,
            runtime_input=runtime_input,
            next_run=run_at or datetime.now(timezone.utc),
        )
        self._tasks[task_id] = task

        async def handler() -> RuntimeDecision:
            return await self._execute_task(task_id)

        self._scheduler.schedule(
            name=f"sovereign:{task_id}",
            handler=handler,
            schedule_type=schedule_type,
            priority=priority,
            run_at=run_at,
            interval=interval,
            metadata={"task_id": task_id},
        )
        logger.info(f"Scheduled task: {task_id}, pattern={cron_pattern}")
        return task

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        if task_id not in self._tasks:
            return False
        for job_id, job in self._scheduler._jobs.items():
            if job.metadata.get("task_id") == task_id:
                self._scheduler.cancel(job_id)
                break
        del self._tasks[task_id]
        return True

    def list_scheduled(self) -> List[ScheduledTask]:
        """List all scheduled tasks."""
        return list(self._tasks.values())

    async def run_due_tasks(self) -> List[RuntimeDecision]:
        """Run all tasks that are due for execution."""
        now = datetime.now(timezone.utc)
        results = []
        for task_id, task in self._tasks.items():
            if task.next_run and task.next_run <= now:
                results.append(await self._execute_task(task_id))
        return results

    async def _execute_task(self, task_id: str) -> RuntimeDecision:
        """Execute a task through the SovereignRuntime."""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        start = datetime.now(timezone.utc)
        result = await self._runtime.process(task.runtime_input)

        task.last_run = start
        task.last_result = result
        task.execution_count += 1
        if task.cron_pattern:
            task.next_run = start + timedelta(
                seconds=parse_cron_to_interval(task.cron_pattern)
            )

        self._history.append(
            {
                "task_id": task_id,
                "executed_at": start.isoformat(),
                "decision_id": result.id,
                "action": result.action,
                "ihsan_score": result.ihsan_score,
                "execution_allowed": result.execution_allowed,
            }
        )
        return result

    def get_history(self, task_id: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """Get execution history, optionally filtered by task_id."""
        hist = [h for h in self._history if not task_id or h["task_id"] == task_id]
        return hist[-limit:]

    def stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "total_tasks": len(self._tasks),
            "total_executions": sum(t.execution_count for t in self._tasks.values()),
            "scheduler_stats": self._scheduler.stats(),
        }


__all__ = ["ScheduledTask", "parse_cron_to_interval", "ProactiveSchedulerBridge"]
