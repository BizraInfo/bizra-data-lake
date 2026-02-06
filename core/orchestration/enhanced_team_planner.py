"""
Enhanced Team Planner — Proactive Goal Detection and Execution
==============================================================
Extends the base TeamPlanner with proactive goal detection,
autonomy-aware execution, and integration with the Muraqabah engine.

Standing on Giants: HTN Planning + Proactive Computing + GOAP
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional
import uuid

from .team_planner import TeamPlanner, TeamTask, Goal, TaskComplexity, AgentRole, TaskAllocation
from core.governance.autonomy_matrix import AutonomyMatrix, AutonomyLevel, ActionContext, AutonomyDecision
from .muraqabah_engine import MuraqabahEngine, Opportunity, MonitorDomain
from core.bridges.dual_agentic_bridge import DualAgenticBridge, ConsensusResult
from .event_bus import EventBus, EventPriority, get_event_bus

logger = logging.getLogger(__name__)


@dataclass
class ProactiveGoal:
    """A proactively detected goal from opportunity analysis."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    opportunity_id: str = ""
    domain: MonitorDomain = MonitorDomain.ENVIRONMENTAL
    description: str = ""
    urgency: float = 0.5           # 0-1, higher = more urgent
    estimated_value: float = 0.5   # 0-1, relative value
    deadline: Optional[datetime] = None
    autonomy_level: AutonomyLevel = AutonomyLevel.SUGGESTER
    constitutional_score: float = 0.95
    required_agents: List[AgentRole] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """A plan for executing a proactive goal."""
    goal_id: str = ""
    tasks: List[TeamTask] = field(default_factory=list)
    allocations: Dict[str, List[TaskAllocation]] = field(default_factory=dict)
    autonomy_decision: Optional[AutonomyDecision] = None
    consensus_required: bool = True
    estimated_duration_seconds: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExecutionResult:
    """Result of executing a proactive goal."""
    goal_id: str = ""
    success: bool = False
    tasks_completed: int = 0
    tasks_failed: int = 0
    consensus_approved: bool = False
    autonomy_used: AutonomyLevel = AutonomyLevel.OBSERVER
    value_delivered: float = 0.0
    error: Optional[str] = None
    duration_ms: float = 0.0
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EnhancedTeamPlanner(TeamPlanner):
    """
    Enhanced planner with proactive goal detection and autonomous execution.

    Extends TeamPlanner with:
    - Opportunity-to-goal conversion
    - Autonomy-aware execution
    - Constitutional validation via DualAgenticBridge
    - Proactive scheduling
    """

    def __init__(
        self,
        ihsan_threshold: float = 0.95,
        autonomy_matrix: Optional[AutonomyMatrix] = None,
        bridge: Optional[DualAgenticBridge] = None,
        muraqabah: Optional[MuraqabahEngine] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(ihsan_threshold=ihsan_threshold)

        self.autonomy = autonomy_matrix or AutonomyMatrix(default_level=AutonomyLevel.SUGGESTER)
        self.bridge = bridge or DualAgenticBridge(ihsan_threshold=ihsan_threshold)
        self.muraqabah = muraqabah
        self.event_bus = event_bus or get_event_bus()

        # Proactive state
        self._proactive_goals: Dict[str, ProactiveGoal] = {}
        self._execution_plans: Dict[str, ExecutionPlan] = {}
        self._execution_results: List[ExecutionResult] = []
        self._goal_queue: List[ProactiveGoal] = []

        # Handlers
        self._approval_handlers: List[Callable[[ProactiveGoal], bool]] = []
        self._execution_handlers: Dict[str, Callable[[TeamTask], bool]] = {}

    async def handle_opportunity(self, opportunity: Opportunity) -> Optional[ProactiveGoal]:
        """Convert an opportunity into a proactive goal."""
        # Assess autonomy level based on opportunity characteristics
        context = ActionContext(
            action_type=f"proactive_{opportunity.domain.value}",
            description=opportunity.description,
            cost_percent=opportunity.estimated_value * 5,  # Rough cost estimate
            risk_score=1.0 - opportunity.confidence,
            ihsan_score=opportunity.confidence,
            is_reversible=True,
            is_emergency=opportunity.urgency > 0.9,
            domain=opportunity.domain.value,
        )

        autonomy_decision = self.autonomy.determine_autonomy(context)

        # Create proactive goal
        goal = ProactiveGoal(
            opportunity_id=opportunity.id,
            domain=opportunity.domain,
            description=opportunity.description,
            urgency=opportunity.urgency,
            estimated_value=opportunity.estimated_value,
            autonomy_level=autonomy_decision.determined_level,
            constitutional_score=opportunity.confidence,
            success_criteria=[f"Complete: {opportunity.action_required}"],
        )

        # Set deadline based on urgency
        if opportunity.urgency > 0.8:
            goal.deadline = datetime.now(timezone.utc) + timedelta(hours=1)
        elif opportunity.urgency > 0.5:
            goal.deadline = datetime.now(timezone.utc) + timedelta(hours=24)

        self._proactive_goals[goal.id] = goal
        logger.info(f"Created proactive goal: {goal.id} ({goal.domain.value})")

        return goal

    async def plan_goal(self, goal: ProactiveGoal) -> ExecutionPlan:
        """Create an execution plan for a proactive goal."""
        plan = ExecutionPlan(goal_id=goal.id)

        # Convert to base Goal for decomposition
        base_goal = Goal(
            id=goal.id,
            description=goal.description,
            success_criteria=goal.success_criteria,
            priority=goal.urgency,
            deadline=goal.deadline,
        )

        # Decompose into tasks
        tasks = await self.decompose_goal(base_goal)
        plan.tasks = tasks

        # Allocate each task
        for task in tasks:
            allocations = self.allocate_task(task)
            plan.allocations[task.id] = allocations

        # Determine autonomy for execution
        context = ActionContext(
            action_type=f"execute_{goal.domain.value}_goal",
            description=goal.description,
            cost_percent=goal.estimated_value * 10,
            risk_score=1.0 - goal.constitutional_score,
            ihsan_score=goal.constitutional_score,
            is_reversible=True,
            domain=goal.domain.value,
        )
        plan.autonomy_decision = self.autonomy.determine_autonomy(context)

        # Consensus required unless AUTOLOW or below
        plan.consensus_required = (
            plan.autonomy_decision.determined_level > AutonomyLevel.AUTOLOW
        )

        # Estimate duration
        plan.estimated_duration_seconds = len(tasks) * 30  # Rough estimate

        self._execution_plans[goal.id] = plan
        return plan

    async def execute_autonomously(self, goal: ProactiveGoal) -> ExecutionResult:
        """Execute a goal autonomously based on autonomy level."""
        result = ExecutionResult(goal_id=goal.id)
        start_time = datetime.now(timezone.utc)

        try:
            # Get or create plan
            plan = self._execution_plans.get(goal.id)
            if not plan:
                plan = await self.plan_goal(goal)

            result.autonomy_used = plan.autonomy_decision.determined_level

            # Check if we can execute automatically
            if not plan.autonomy_decision.can_execute:
                # Need approval
                approved = await self._request_approval(goal, plan)
                if not approved:
                    result.error = "Approval denied"
                    return result

            # If consensus required, validate through bridge
            if plan.consensus_required:
                for task in plan.tasks:
                    outcome = await self.bridge.propose_and_validate(
                        task=task,
                        action_type="proactive_task_execution",
                        parameters={
                            "goal_id": goal.id,
                            "task_name": task.name,
                            "autonomy_level": goal.autonomy_level.name,
                        },
                        ihsan_estimate=goal.constitutional_score,
                        risk_estimate=1.0 - goal.constitutional_score,
                    )

                    if outcome.result != ConsensusResult.APPROVED:
                        result.error = f"Consensus denied for task {task.id}"
                        result.consensus_approved = False
                        return result

                result.consensus_approved = True

            # Execute tasks
            for task in plan.tasks:
                try:
                    # Notify before if required
                    if plan.autonomy_decision.notify_before:
                        await self.event_bus.emit(
                            topic="proactive.task.starting",
                            payload={
                                "goal_id": goal.id,
                                "task_id": task.id,
                                "task_name": task.name,
                            },
                            priority=EventPriority.NORMAL,
                        )

                    # Execute task
                    success = await self._execute_task(task)

                    if success:
                        result.tasks_completed += 1
                        self.complete_task(task.id)
                    else:
                        result.tasks_failed += 1

                    # Notify after if required
                    if plan.autonomy_decision.notify_after:
                        await self.event_bus.emit(
                            topic="proactive.task.completed",
                            payload={
                                "goal_id": goal.id,
                                "task_id": task.id,
                                "success": success,
                            },
                            priority=EventPriority.NORMAL,
                        )

                except Exception as e:
                    logger.error(f"Task execution error: {e}")
                    result.tasks_failed += 1

            # Calculate success
            result.success = result.tasks_completed > 0 and result.tasks_failed == 0
            result.value_delivered = (
                goal.estimated_value * result.tasks_completed / max(len(plan.tasks), 1)
            )

        except Exception as e:
            result.error = str(e)
            logger.error(f"Goal execution error: {e}")

        finally:
            result.duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            result.completed_at = datetime.now(timezone.utc)
            self._execution_results.append(result)

        return result

    async def _request_approval(
        self,
        goal: ProactiveGoal,
        plan: ExecutionPlan,
    ) -> bool:
        """Request approval for goal execution."""
        # Emit approval request event
        await self.event_bus.emit(
            topic="proactive.approval.requested",
            payload={
                "goal_id": goal.id,
                "description": goal.description,
                "urgency": goal.urgency,
                "autonomy_level": goal.autonomy_level.name,
                "tasks_count": len(plan.tasks),
            },
            priority=EventPriority.HIGH,
        )

        # Call registered approval handlers
        for handler in self._approval_handlers:
            try:
                if handler(goal):
                    return True
            except Exception as e:
                logger.error(f"Approval handler error: {e}")

        # Default: not approved without explicit handler
        return False

    async def _execute_task(self, task: TeamTask) -> bool:
        """Execute a single task."""
        # Check for registered handler
        handler_key = task.name.split(":")[0] if ":" in task.name else "default"
        handler = self._execution_handlers.get(handler_key, self._default_task_handler)

        try:
            return handler(task)
        except Exception as e:
            logger.error(f"Task handler error: {e}")
            return False

    def _default_task_handler(self, task: TeamTask) -> bool:
        """Default task handler - logs and returns success."""
        logger.info(f"Executing task: {task.name}")
        return True

    def register_approval_handler(self, handler: Callable[[ProactiveGoal], bool]) -> None:
        """Register an approval handler."""
        self._approval_handlers.append(handler)

    def register_execution_handler(
        self,
        task_type: str,
        handler: Callable[[TeamTask], bool],
    ) -> None:
        """Register a task execution handler."""
        self._execution_handlers[task_type] = handler

    def queue_goal(self, goal: ProactiveGoal) -> None:
        """Add goal to execution queue."""
        self._goal_queue.append(goal)
        self._goal_queue.sort(key=lambda g: (-g.urgency, g.created_at))

    def get_next_goal(self) -> Optional[ProactiveGoal]:
        """Get next goal from queue."""
        if self._goal_queue:
            return self._goal_queue.pop(0)
        return None

    def stats(self) -> Dict[str, Any]:
        """Get enhanced planner statistics."""
        base_stats = super().stats()

        total_results = len(self._execution_results)
        successful = sum(1 for r in self._execution_results if r.success)

        return {
            **base_stats,
            "proactive_goals": len(self._proactive_goals),
            "queued_goals": len(self._goal_queue),
            "execution_plans": len(self._execution_plans),
            "total_executions": total_results,
            "successful_executions": successful,
            "success_rate": successful / max(total_results, 1),
            "total_value_delivered": sum(r.value_delivered for r in self._execution_results),
            "autonomy_stats": self.autonomy.stats(),
            "bridge_stats": self.bridge.stats(),
        }


__all__ = [
    "EnhancedTeamPlanner",
    "ExecutionPlan",
    "ExecutionResult",
    "ProactiveGoal",
]
