"""
Team Planner — Goal Decomposition and Task Allocation
=====================================================
Decomposes high-level goals into agent-specific tasks and
coordinates allocation across the dual-agentic team (PAT + SAT).

Standing on Giants: HTN Planning + Game Theory + Load Balancing
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Roles in the dual-agentic team."""
    # PAT - Primary Action Team (7 agents)
    MASTER_REASONER = "master_reasoner"
    DATA_ANALYZER = "data_analyzer"
    EXECUTION_PLANNER = "execution_planner"
    ETHICS_GUARDIAN = "ethics_guardian"
    COMMUNICATOR = "communicator"
    MEMORY_ARCHITECT = "memory_architect"
    FUSION = "fusion"

    # SAT - Secondary Action Team (5 validators)
    SECURITY_GUARDIAN = "security_guardian"
    ETHICS_VALIDATOR = "ethics_validator"
    PERFORMANCE_MONITOR = "performance_monitor"
    CONSISTENCY_CHECKER = "consistency_checker"
    RESOURCE_OPTIMIZER = "resource_optimizer"


class TaskComplexity(Enum):
    """Task complexity levels."""
    TRIVIAL = auto()    # Single agent, immediate
    SIMPLE = auto()     # Single agent, some work
    MODERATE = auto()   # 2-3 agents, coordination
    COMPLEX = auto()    # Full team, orchestration
    CRITICAL = auto()   # All hands, veto-enabled


@dataclass
class TeamTask:
    """A task allocated to team agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    complexity: TaskComplexity = TaskComplexity.SIMPLE
    assigned_roles: Set[AgentRole] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)  # Task IDs
    deadline: Optional[datetime] = None
    priority: float = 0.5  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Goal:
    """A high-level goal to be decomposed."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    success_criteria: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    deadline: Optional[datetime] = None


@dataclass
class TaskAllocation:
    """Result of task allocation to agents."""
    task_id: str = ""
    role: AgentRole = AgentRole.MASTER_REASONER
    estimated_effort: float = 1.0  # Relative effort units
    confidence: float = 0.9


class TeamPlanner:
    """
    Coordinates goal decomposition and task allocation.

    Capabilities:
    - Break goals into achievable tasks
    - Allocate tasks to appropriate agents
    - Handle dependencies and ordering
    - Balance load across team
    """

    # Role capabilities for task matching
    ROLE_CAPABILITIES = {
        AgentRole.MASTER_REASONER: {"reasoning", "planning", "synthesis", "strategy"},
        AgentRole.DATA_ANALYZER: {"data", "analysis", "patterns", "statistics"},
        AgentRole.EXECUTION_PLANNER: {"workflow", "execution", "scheduling", "resources"},
        AgentRole.ETHICS_GUARDIAN: {"ethics", "safety", "compliance", "validation"},
        AgentRole.COMMUNICATOR: {"output", "formatting", "explanation", "translation"},
        AgentRole.MEMORY_ARCHITECT: {"memory", "knowledge", "retrieval", "indexing"},
        AgentRole.FUSION: {"integration", "synthesis", "merging", "consensus"},
        AgentRole.SECURITY_GUARDIAN: {"security", "access", "authentication", "encryption"},
        AgentRole.ETHICS_VALIDATOR: {"ethics", "fairness", "bias", "harm"},
        AgentRole.PERFORMANCE_MONITOR: {"performance", "latency", "throughput", "metrics"},
        AgentRole.CONSISTENCY_CHECKER: {"consistency", "validation", "verification", "integrity"},
        AgentRole.RESOURCE_OPTIMIZER: {"optimization", "efficiency", "allocation", "scaling"},
    }

    def __init__(self, ihsan_threshold: float = 0.95):
        self.ihsan_threshold = ihsan_threshold
        self._pending_goals: List[Goal] = []
        self._active_tasks: Dict[str, TeamTask] = {}
        self._completed_tasks: Set[str] = set()
        self._allocations: Dict[str, List[TaskAllocation]] = {}
        self._agent_loads: Dict[AgentRole, float] = {role: 0.0 for role in AgentRole}

    async def decompose_goal(self, goal: Goal) -> List[TeamTask]:
        """
        Decompose a goal into executable tasks.

        Uses hierarchical task decomposition based on goal complexity.
        """
        tasks = []

        # Determine complexity from goal description and constraints
        complexity = self._assess_complexity(goal)

        if complexity == TaskComplexity.TRIVIAL:
            # Single task
            tasks.append(TeamTask(
                name=f"Execute: {goal.description[:50]}",
                description=goal.description,
                complexity=complexity,
                priority=goal.priority,
                deadline=goal.deadline,
            ))

        elif complexity == TaskComplexity.SIMPLE:
            # Analysis + Execution
            tasks.extend([
                TeamTask(
                    name=f"Analyze: {goal.description[:30]}",
                    description=f"Analyze requirements for: {goal.description}",
                    complexity=TaskComplexity.TRIVIAL,
                    priority=goal.priority,
                ),
                TeamTask(
                    name=f"Execute: {goal.description[:30]}",
                    description=f"Execute plan for: {goal.description}",
                    complexity=TaskComplexity.SIMPLE,
                    priority=goal.priority,
                    deadline=goal.deadline,
                ),
            ])
            # Set dependency
            tasks[1].dependencies.add(tasks[0].id)

        elif complexity in (TaskComplexity.MODERATE, TaskComplexity.COMPLEX):
            # Full decomposition: Plan + Validate + Execute + Review
            plan_task = TeamTask(
                name=f"Plan: {goal.description[:25]}",
                description=f"Create execution plan for: {goal.description}",
                complexity=TaskComplexity.SIMPLE,
                priority=goal.priority,
            )
            tasks.append(plan_task)

            validate_task = TeamTask(
                name=f"Validate plan",
                description=f"Validate plan against constraints and ethics",
                complexity=TaskComplexity.SIMPLE,
                priority=goal.priority,
                dependencies={plan_task.id},
            )
            tasks.append(validate_task)

            execute_task = TeamTask(
                name=f"Execute: {goal.description[:25]}",
                description=f"Execute validated plan for: {goal.description}",
                complexity=complexity,
                priority=goal.priority,
                deadline=goal.deadline,
                dependencies={validate_task.id},
            )
            tasks.append(execute_task)

            review_task = TeamTask(
                name=f"Review execution",
                description=f"Review results and validate success criteria",
                complexity=TaskComplexity.SIMPLE,
                priority=goal.priority,
                dependencies={execute_task.id},
            )
            tasks.append(review_task)

        else:  # CRITICAL
            # All of the above plus security and consensus
            # Add security pre-check
            security_task = TeamTask(
                name=f"Security assessment",
                description=f"Assess security implications for: {goal.description}",
                complexity=TaskComplexity.MODERATE,
                priority=goal.priority * 1.2,  # Higher priority
            )
            tasks.append(security_task)

            # Standard decomposition with security dependency
            for task in await self.decompose_goal(Goal(
                description=goal.description,
                priority=goal.priority,
                deadline=goal.deadline,
                constraints={**goal.constraints, "_complexity_override": TaskComplexity.COMPLEX},
            )):
                task.dependencies.add(security_task.id)
                tasks.append(task)

        logger.info(f"Decomposed goal '{goal.id}' into {len(tasks)} tasks")
        return tasks

    def _assess_complexity(self, goal: Goal) -> TaskComplexity:
        """Assess goal complexity based on description and constraints."""
        if "_complexity_override" in goal.constraints:
            return goal.constraints["_complexity_override"]

        desc_lower = goal.description.lower()

        # Keyword-based complexity assessment
        critical_keywords = {"critical", "urgent", "emergency", "security", "financial"}
        complex_keywords = {"multi", "coordinate", "integrate", "analyze"}
        moderate_keywords = {"update", "modify", "create", "generate"}

        if any(kw in desc_lower for kw in critical_keywords):
            return TaskComplexity.CRITICAL
        elif any(kw in desc_lower for kw in complex_keywords):
            return TaskComplexity.COMPLEX
        elif any(kw in desc_lower for kw in moderate_keywords):
            return TaskComplexity.MODERATE
        elif len(goal.description) < 50:
            return TaskComplexity.TRIVIAL
        else:
            return TaskComplexity.SIMPLE

    def allocate_task(self, task: TeamTask) -> List[TaskAllocation]:
        """Allocate a task to appropriate agent roles."""
        allocations = []

        # Find matching roles based on task description
        task_keywords = set(task.description.lower().split())
        matched_roles = []

        for role, capabilities in self.ROLE_CAPABILITIES.items():
            overlap = task_keywords & capabilities
            if overlap:
                score = len(overlap) / len(capabilities)
                matched_roles.append((role, score))

        # Sort by match score
        matched_roles.sort(key=lambda x: x[1], reverse=True)

        # Determine how many roles based on complexity
        role_count = {
            TaskComplexity.TRIVIAL: 1,
            TaskComplexity.SIMPLE: 1,
            TaskComplexity.MODERATE: 2,
            TaskComplexity.COMPLEX: 3,
            TaskComplexity.CRITICAL: 5,
        }.get(task.complexity, 1)

        # Apply load balancing
        selected_roles = []
        for role, score in matched_roles[:role_count * 2]:  # Consider more candidates
            if len(selected_roles) >= role_count:
                break
            # Prefer less loaded agents
            load_factor = 1.0 + self._agent_loads.get(role, 0)
            adjusted_score = score / load_factor
            selected_roles.append((role, adjusted_score))

        selected_roles.sort(key=lambda x: x[1], reverse=True)

        # Create allocations
        for role, score in selected_roles[:role_count]:
            allocation = TaskAllocation(
                task_id=task.id,
                role=role,
                estimated_effort=task.complexity.value * 0.5,
                confidence=min(0.99, score + 0.5),
            )
            allocations.append(allocation)
            self._agent_loads[role] += allocation.estimated_effort

        # Default to master reasoner if no match
        if not allocations:
            allocations.append(TaskAllocation(
                task_id=task.id,
                role=AgentRole.MASTER_REASONER,
                confidence=0.7,
            ))

        task.assigned_roles = {a.role for a in allocations}
        self._active_tasks[task.id] = task
        self._allocations[task.id] = allocations

        logger.debug(f"Allocated task {task.id} to {[a.role.value for a in allocations]}")
        return allocations

    def complete_task(self, task_id: str) -> None:
        """Mark a task as completed and update loads."""
        if task_id in self._allocations:
            for allocation in self._allocations[task_id]:
                self._agent_loads[allocation.role] -= allocation.estimated_effort
                self._agent_loads[allocation.role] = max(0, self._agent_loads[allocation.role])

        self._completed_tasks.add(task_id)
        if task_id in self._active_tasks:
            del self._active_tasks[task_id]

    def get_ready_tasks(self) -> List[TeamTask]:
        """Get tasks whose dependencies are satisfied."""
        ready = []
        for task in self._active_tasks.values():
            if task.dependencies.issubset(self._completed_tasks):
                ready.append(task)
        return sorted(ready, key=lambda t: -t.priority)

    def stats(self) -> Dict[str, Any]:
        """Get planner statistics."""
        return {
            "pending_goals": len(self._pending_goals),
            "active_tasks": len(self._active_tasks),
            "completed_tasks": len(self._completed_tasks),
            "agent_loads": {r.value: round(l, 2) for r, l in self._agent_loads.items()},
        }


__all__ = [
    "AgentRole",
    "Goal",
    "TaskAllocation",
    "TaskComplexity",
    "TeamPlanner",
    "TeamTask",
]
