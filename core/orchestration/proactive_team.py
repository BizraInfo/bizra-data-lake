"""
Proactive Team — Loop Integration Layer
=======================================
Integrates proactive components (scheduler, monitor, collective intelligence)
with the autonomous OODA loop for coordinated proactive operation.

Standing on Giants: OODA Loop + Proactive Computing + Team Coordination
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .event_bus import EventBus, Event, EventPriority, get_event_bus
from .team_planner import TeamPlanner, Goal, TeamTask
from core.bridges.dual_agentic_bridge import DualAgenticBridge, ConsensusResult
from core.reasoning.collective_intelligence import CollectiveIntelligence, AgentContribution
from .proactive_scheduler import ProactiveScheduler, ScheduleType, JobPriority
from .predictive_monitor import PredictiveMonitor, TrendDirection

logger = logging.getLogger(__name__)


@dataclass
class ProactiveCycleResult:
    """Result of one proactive team cycle."""
    cycle_number: int = 0
    opportunities_detected: int = 0
    tasks_created: int = 0
    tasks_executed: int = 0
    consensus_approved: int = 0
    consensus_vetoed: int = 0
    collective_decisions: int = 0
    alerts_generated: int = 0
    synergy_score: float = 0.0
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ProactiveTeam:
    """
    Coordinates proactive team operation.

    Integrates:
    - TeamPlanner for goal decomposition
    - DualAgenticBridge for PAT/SAT coordination
    - CollectiveIntelligence for team synthesis
    - ProactiveScheduler for anticipatory execution
    - PredictiveMonitor for trend detection
    - EventBus for component communication
    """

    def __init__(
        self,
        ihsan_threshold: float = 0.95,
        event_bus: Optional[EventBus] = None,
    ):
        self.ihsan_threshold = ihsan_threshold
        self.event_bus = event_bus or get_event_bus()

        # Initialize components
        self.planner = TeamPlanner(ihsan_threshold=ihsan_threshold)
        self.bridge = DualAgenticBridge(ihsan_threshold=ihsan_threshold)
        self.collective = CollectiveIntelligence()
        self.scheduler = ProactiveScheduler()
        self.monitor = PredictiveMonitor()

        # State
        self._running = False
        self._cycle_count = 0
        self._cycle_results: List[ProactiveCycleResult] = []

        # Subscribe to events
        self._setup_event_handlers()

    def _setup_event_handlers(self) -> None:
        """Set up event bus subscriptions."""

        async def handle_opportunity(event: Event):
            """Handle detected opportunity."""
            await self._process_opportunity(event.payload)

        async def handle_alert(event: Event):
            """Handle predictive alert."""
            await self._process_alert(event.payload)

        self.event_bus.subscribe("proactive.opportunity", handle_opportunity)
        self.event_bus.subscribe("proactive.alert", handle_alert)
        self.event_bus.subscribe("monitor.*", handle_alert)

    async def _process_opportunity(self, opportunity: Dict[str, Any]) -> None:
        """Process a detected opportunity."""
        logger.info(f"Processing opportunity: {opportunity.get('description', 'unknown')}")

        # Create goal from opportunity
        goal = Goal(
            description=opportunity.get("description", ""),
            success_criteria=opportunity.get("criteria", []),
            priority=opportunity.get("priority", 0.5),
        )

        # Decompose into tasks
        tasks = await self.planner.decompose_goal(goal)

        # Allocate and validate each task
        for task in tasks:
            allocations = self.planner.allocate_task(task)

            # Validate through bridge
            outcome = await self.bridge.propose_and_validate(
                task=task,
                action_type="execute_proactive_task",
                parameters={"task": task.name, "allocations": [a.role.value for a in allocations]},
                ihsan_estimate=0.95,
            )

            if outcome.result == ConsensusResult.APPROVED:
                # Schedule for execution
                self.scheduler.schedule(
                    name=task.name,
                    handler=self._create_task_handler(task),
                    schedule_type=ScheduleType.ONE_TIME,
                    priority=JobPriority.NORMAL,
                )

    async def _process_alert(self, alert: Dict[str, Any]) -> None:
        """Process a predictive alert."""
        logger.info(f"Processing alert: {alert.get('message', 'unknown')}")

        # Emit event for handlers
        await self.event_bus.emit(
            topic="proactive.alert.processed",
            payload=alert,
            priority=EventPriority.HIGH,
        )

    def _create_task_handler(self, task: TeamTask) -> Callable:
        """Create an async handler for a task."""

        async def handler():
            logger.info(f"Executing proactive task: {task.name}")
            # Task execution logic would go here
            self.planner.complete_task(task.id)
            return {"task_id": task.id, "status": "completed"}

        return handler

    async def run_cycle(self) -> ProactiveCycleResult:
        """Execute one proactive team cycle."""
        self._cycle_count += 1
        start_time = datetime.now(timezone.utc)

        result = ProactiveCycleResult(cycle_number=self._cycle_count)

        # 1. Monitor Analysis
        analyses = self.monitor.analyze_all()
        alerts = self.monitor.check_alerts()
        result.alerts_generated = len(alerts)

        # Emit alerts as events
        for alert in alerts:
            await self.event_bus.emit(
                topic="proactive.alert",
                payload={
                    "id": alert.id,
                    "metric": alert.metric_name,
                    "severity": alert.severity.name,
                    "message": alert.message,
                },
                priority=EventPriority.HIGH,
            )

        # 2. Opportunity Detection
        opportunities = self._detect_opportunities(analyses)
        result.opportunities_detected = len(opportunities)

        # 3. Process opportunities
        for opp in opportunities:
            await self._process_opportunity(opp)

        # 4. Get scheduler stats
        scheduler_stats = self.scheduler.stats()
        result.tasks_created = scheduler_stats["total_jobs"]
        result.tasks_executed = scheduler_stats["total_runs"]

        # 5. Get bridge stats
        bridge_stats = self.bridge.stats()
        result.consensus_approved = bridge_stats["approved"]
        result.consensus_vetoed = bridge_stats["vetoed"]

        # 6. Get collective intelligence stats
        ci_stats = self.collective.stats()
        result.collective_decisions = ci_stats["total_decisions"]
        result.synergy_score = ci_stats["average_synergy"]

        # Calculate duration
        result.duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        self._cycle_results.append(result)
        return result

    def _detect_opportunities(
        self,
        analyses: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Detect opportunities from metric analyses."""
        opportunities = []

        for name, analysis in analyses.items():
            # Optimization opportunities
            if analysis.direction == TrendDirection.RISING:
                if name in ("memory_usage", "cpu_usage", "latency_ms"):
                    opportunities.append({
                        "type": "optimization",
                        "description": f"Optimize {name} - trending up",
                        "metric": name,
                        "priority": 0.7,
                        "criteria": [f"Reduce {name} trend"],
                    })

            # Improvement opportunities
            if analysis.direction == TrendDirection.STABLE:
                if name in ("snr_score", "ihsan_score") and analysis.forecast_1h:
                    if analysis.forecast_1h < 0.98:
                        opportunities.append({
                            "type": "improvement",
                            "description": f"Improve {name} from stable baseline",
                            "metric": name,
                            "priority": 0.5,
                            "criteria": [f"Increase {name} above 0.98"],
                        })

        return opportunities

    async def start(self) -> None:
        """Start the proactive team."""
        self._running = True
        logger.info("Proactive team started")

        # Start event bus
        event_bus_task = asyncio.create_task(self.event_bus.start())

        # Start scheduler
        scheduler_task = asyncio.create_task(self.scheduler.start())

        # Main proactive loop
        while self._running:
            try:
                result = await self.run_cycle()
                logger.debug(
                    f"Cycle {result.cycle_number}: "
                    f"opps={result.opportunities_detected}, "
                    f"tasks={result.tasks_executed}, "
                    f"synergy={result.synergy_score:.2f}"
                )
            except Exception as e:
                logger.error(f"Proactive cycle error: {e}")

            await asyncio.sleep(5.0)  # Cycle interval

    def stop(self) -> None:
        """Stop the proactive team."""
        self._running = False
        self.event_bus.stop()
        self.scheduler.stop()
        logger.info("Proactive team stopped")

    def stats(self) -> Dict[str, Any]:
        """Get team statistics."""
        return {
            "cycles": self._cycle_count,
            "running": self._running,
            "planner": self.planner.stats(),
            "bridge": self.bridge.stats(),
            "collective": self.collective.stats(),
            "scheduler": self.scheduler.stats(),
            "monitor": self.monitor.stats(),
        }


__all__ = [
    "ProactiveCycleResult",
    "ProactiveTeam",
]
