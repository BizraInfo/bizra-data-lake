"""
Proactive Scheduler — Anticipatory Job Scheduling
=================================================
Schedules and executes proactive tasks based on predicted needs,
patterns, and opportunities rather than just reactive commands.

Standing on Giants: Cron + Predictive Scheduling + Reactive Systems
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional
import uuid
import heapq

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of scheduled jobs."""
    ONE_TIME = auto()     # Execute once at specified time
    RECURRING = auto()    # Execute on interval
    TRIGGERED = auto()    # Execute when condition met
    PROACTIVE = auto()    # Execute when opportunity detected


class JobPriority(Enum):
    """Job execution priority."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class ScheduledJob:
    """A job in the proactive schedule."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    handler: Optional[Callable[[], Coroutine[Any, Any, Any]]] = None
    schedule_type: ScheduleType = ScheduleType.ONE_TIME
    priority: JobPriority = JobPriority.NORMAL

    # Timing
    next_run: Optional[datetime] = None
    interval_seconds: Optional[float] = None
    last_run: Optional[datetime] = None

    # Trigger condition (for TRIGGERED type)
    trigger_condition: Optional[Callable[[], bool]] = None

    # Execution
    enabled: bool = True
    run_count: int = 0
    error_count: int = 0
    max_retries: int = 3

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "ScheduledJob") -> bool:
        """For heap ordering by next_run time."""
        if self.next_run is None:
            return False
        if other.next_run is None:
            return True
        return (self.next_run, self.priority.value) < (other.next_run, other.priority.value)


@dataclass
class JobResult:
    """Result of a job execution."""
    job_id: str = ""
    success: bool = True
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ProactiveScheduler:
    """
    Scheduler for proactive and anticipatory task execution.

    Features:
    - Time-based scheduling (one-time, recurring)
    - Condition-based triggers
    - Proactive opportunity-based execution
    - Priority-aware execution queue
    - Graceful degradation under load
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        check_interval: float = 1.0,
    ):
        self.max_concurrent = max_concurrent
        self.check_interval = check_interval

        self._jobs: Dict[str, ScheduledJob] = {}
        self._job_heap: List[ScheduledJob] = []  # Min-heap by next_run
        self._results: Dict[str, List[JobResult]] = {}
        self._running = False
        self._active_count = 0

    def schedule(
        self,
        name: str,
        handler: Callable[[], Coroutine[Any, Any, Any]],
        schedule_type: ScheduleType = ScheduleType.ONE_TIME,
        priority: JobPriority = JobPriority.NORMAL,
        run_at: Optional[datetime] = None,
        interval: Optional[float] = None,
        trigger: Optional[Callable[[], bool]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Schedule a new job."""
        job = ScheduledJob(
            name=name,
            handler=handler,
            schedule_type=schedule_type,
            priority=priority,
            interval_seconds=interval,
            trigger_condition=trigger,
            metadata=metadata or {},
        )

        # Set next run time
        if schedule_type == ScheduleType.ONE_TIME:
            job.next_run = run_at or datetime.now(timezone.utc)
        elif schedule_type == ScheduleType.RECURRING:
            job.next_run = run_at or datetime.now(timezone.utc)
            if not interval:
                raise ValueError("RECURRING jobs require interval")
        elif schedule_type == ScheduleType.TRIGGERED:
            if not trigger:
                raise ValueError("TRIGGERED jobs require trigger condition")
            job.next_run = datetime.now(timezone.utc)  # Check immediately
        elif schedule_type == ScheduleType.PROACTIVE:
            job.next_run = datetime.now(timezone.utc)  # Opportunity-driven

        self._jobs[job.id] = job
        heapq.heappush(self._job_heap, job)
        self._results[job.id] = []

        logger.info(f"Scheduled job: {name} ({schedule_type.name})")
        return job.id

    def cancel(self, job_id: str) -> bool:
        """Cancel a scheduled job."""
        if job_id in self._jobs:
            self._jobs[job_id].enabled = False
            del self._jobs[job_id]
            logger.info(f"Cancelled job: {job_id}")
            return True
        return False

    def pause(self, job_id: str) -> bool:
        """Pause a job (won't execute until resumed)."""
        if job_id in self._jobs:
            self._jobs[job_id].enabled = False
            return True
        return False

    def resume(self, job_id: str) -> bool:
        """Resume a paused job."""
        if job_id in self._jobs:
            self._jobs[job_id].enabled = True
            return True
        return False

    async def _execute_job(self, job: ScheduledJob) -> JobResult:
        """Execute a single job."""
        result = JobResult(job_id=job.id)
        start_time = datetime.now(timezone.utc)

        try:
            self._active_count += 1
            if job.handler:
                result.result = await job.handler()
            result.success = True
            job.run_count += 1
            job.last_run = datetime.now(timezone.utc)

        except Exception as e:
            result.success = False
            result.error = str(e)
            job.error_count += 1
            logger.error(f"Job {job.name} failed: {e}")

        finally:
            self._active_count -= 1
            result.duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        self._results[job.id].append(result)

        # Keep only recent results
        if len(self._results[job.id]) > 100:
            self._results[job.id] = self._results[job.id][-50:]

        return result

    def _update_next_run(self, job: ScheduledJob) -> None:
        """Update job's next run time after execution."""
        now = datetime.now(timezone.utc)

        if job.schedule_type == ScheduleType.ONE_TIME:
            job.next_run = None  # No more runs
            job.enabled = False

        elif job.schedule_type == ScheduleType.RECURRING:
            if job.interval_seconds:
                job.next_run = now + timedelta(seconds=job.interval_seconds)
                heapq.heappush(self._job_heap, job)

        elif job.schedule_type == ScheduleType.TRIGGERED:
            # Check again after short delay
            job.next_run = now + timedelta(seconds=self.check_interval * 5)
            heapq.heappush(self._job_heap, job)

        elif job.schedule_type == ScheduleType.PROACTIVE:
            # Re-queue for next opportunity check
            job.next_run = now + timedelta(seconds=self.check_interval * 10)
            heapq.heappush(self._job_heap, job)

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            now = datetime.now(timezone.utc)

            # Process ready jobs
            while self._job_heap and self._active_count < self.max_concurrent:
                # Peek at next job
                if not self._job_heap:
                    break

                job = self._job_heap[0]

                # Check if job should run
                if job.next_run and job.next_run > now:
                    break  # No jobs ready yet

                # Pop the job
                heapq.heappop(self._job_heap)

                # Skip disabled jobs
                if not job.enabled or job.id not in self._jobs:
                    continue

                # For triggered jobs, check condition
                if job.schedule_type == ScheduleType.TRIGGERED:
                    if job.trigger_condition and not job.trigger_condition():
                        self._update_next_run(job)
                        continue

                # Execute asynchronously
                asyncio.create_task(self._run_job(job))

            await asyncio.sleep(self.check_interval)

    async def _run_job(self, job: ScheduledJob) -> None:
        """Run a job and update its schedule."""
        await self._execute_job(job)
        self._update_next_run(job)

    async def start(self) -> None:
        """Start the scheduler."""
        self._running = True
        logger.info("Proactive scheduler started")
        await self._scheduler_loop()

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        logger.info("Proactive scheduler stopped")

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def get_results(self, job_id: str, limit: int = 10) -> List[JobResult]:
        """Get recent results for a job."""
        return self._results.get(job_id, [])[-limit:]

    def stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        total_runs = sum(j.run_count for j in self._jobs.values())
        total_errors = sum(j.error_count for j in self._jobs.values())

        return {
            "total_jobs": len(self._jobs),
            "enabled_jobs": sum(1 for j in self._jobs.values() if j.enabled),
            "pending_jobs": len(self._job_heap),
            "active_jobs": self._active_count,
            "total_runs": total_runs,
            "total_errors": total_errors,
            "success_rate": (total_runs - total_errors) / max(total_runs, 1),
            "running": self._running,
        }


__all__ = [
    "JobPriority",
    "JobResult",
    "ProactiveScheduler",
    "ScheduledJob",
    "ScheduleType",
]
