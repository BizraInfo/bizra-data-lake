"""
Autonomous Agent — Self-Directing Task Executor

Core agent abstraction for BIZRA agentic system:
- Goal-directed behavior
- Self-monitoring and correction
- Constitutional constraints (Ihsān)
- Tool usage capabilities

Standing on Giants: BDI Architecture + Constitutional AI + ReAct Pattern
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """State of an autonomous agent."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING = "waiting"
    REFLECTING = "reflecting"
    ERROR = "error"
    HALTED = "halted"


class TaskPriority(str, Enum):
    """Priority levels for agent tasks."""
    CRITICAL = "critical"    # Immediate attention
    HIGH = "high"            # Soon
    NORMAL = "normal"        # Regular queue
    LOW = "low"              # When available
    BACKGROUND = "background"  # Only when idle


class TaskStatus(str, Enum):
    """Status of an agent task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentTask:
    """A task for an autonomous agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING

    # Constraints
    deadline: Optional[datetime] = None
    max_retries: int = 3
    timeout_seconds: float = 300.0

    # Execution
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0

    # Results
    result: Optional[Any] = None
    error: Optional[str] = None

    # Dependencies
    depends_on: Set[str] = field(default_factory=set)
    blocks: Set[str] = field(default_factory=set)

    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return self.depends_on.issubset(completed_tasks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retries": self.retries,
            "error": self.error,
        }


@dataclass
class AgentThought:
    """A thought in the agent's reasoning chain."""
    content: str
    thought_type: str  # observation, reasoning, decision, reflection
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentAction:
    """An action taken by the agent."""
    tool: str
    input: Dict[str, Any]
    output: Optional[Any] = None
    success: bool = True
    error: Optional[str] = None
    duration_ms: float = 0.0


class AutonomousAgent(ABC):
    """
    Base class for autonomous agents in BIZRA.

    Implements the OODA loop (Observe-Orient-Decide-Act) with
    Constitutional AI constraints.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "Agent",
        llm_fn: Optional[Callable[[str], str]] = None,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
    ):
        self.id = agent_id or str(uuid.uuid4())[:8]
        self.name = name
        self.llm_fn = llm_fn
        self.ihsan_threshold = ihsan_threshold

        # State
        self.state = AgentState.IDLE
        self._current_task: Optional[AgentTask] = None

        # Task management
        self._task_queue: List[AgentTask] = []
        self._completed_tasks: Set[str] = set()

        # Reasoning trace
        self._thoughts: List[AgentThought] = []
        self._actions: List[AgentAction] = []

        # Tools
        self._tools: Dict[str, Callable] = {}

        # Metrics
        self._total_tasks: int = 0
        self._successful_tasks: int = 0
        self._total_actions: int = 0
        self._ihsan_violations: int = 0

    def register_tool(
        self,
        name: str,
        fn: Callable,
        description: str = "",
    ) -> None:
        """Register a tool for agent use."""
        self._tools[name] = fn
        logger.debug(f"Agent {self.id} registered tool: {name}")

    def add_task(self, task: AgentTask) -> None:
        """Add a task to the queue."""
        self._task_queue.append(task)
        self._task_queue.sort(key=lambda t: (
            -list(TaskPriority).index(t.priority),
            t.created_at,
        ))
        logger.debug(f"Agent {self.id} queued task: {task.name}")

    def _add_thought(
        self,
        content: str,
        thought_type: str,
        confidence: float = 1.0,
    ) -> None:
        """Record a thought in the reasoning chain."""
        self._thoughts.append(AgentThought(
            content=content,
            thought_type=thought_type,
            confidence=confidence,
        ))
        # Keep only recent thoughts
        if len(self._thoughts) > 100:
            self._thoughts.pop(0)

    async def _validate_ihsan(self, action: str) -> bool:
        """
        Validate action against Ihsān constraints.

        Constitutional check before any action.
        """
        # Basic safety checks
        dangerous_patterns = [
            "delete all",
            "drop table",
            "rm -rf",
            "format",
            "shutdown",
            "reboot",
        ]

        action_lower = action.lower()
        for pattern in dangerous_patterns:
            if pattern in action_lower:
                self._ihsan_violations += 1
                self._add_thought(
                    f"BLOCKED: Action contains dangerous pattern '{pattern}'",
                    "ihsan_check",
                    confidence=1.0,
                )
                return False

        # If LLM available, do constitutional check
        if self.llm_fn:
            prompt = f"""Evaluate this action for safety and benefit:
Action: {action}

Is this action:
1. Safe (no harm to user or system)?
2. Beneficial (helps user's goals)?
3. Ethical (respects privacy, fairness)?

Respond with YES if all conditions are met, NO otherwise."""

            try:
                response = self.llm_fn(prompt)
                if "NO" in response.upper():
                    self._ihsan_violations += 1
                    self._add_thought(
                        f"BLOCKED by constitutional check: {response[:100]}",
                        "ihsan_check",
                        confidence=0.8,
                    )
                    return False
            except Exception as e:
                logger.warning(f"Constitutional check failed: {e}")
                # Fail-closed: don't allow if can't verify
                return False

        return True

    async def _use_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> Tuple[bool, Any]:
        """Use a registered tool."""
        if tool_name not in self._tools:
            return False, f"Tool not found: {tool_name}"

        # Validate before using
        action_desc = f"Use tool '{tool_name}' with input: {tool_input}"
        if not await self._validate_ihsan(action_desc):
            return False, "Action blocked by constitutional check"

        import time
        start = time.time()

        try:
            tool_fn = self._tools[tool_name]
            if asyncio.iscoroutinefunction(tool_fn):
                result = await tool_fn(**tool_input)
            else:
                result = tool_fn(**tool_input)

            duration = (time.time() - start) * 1000

            self._actions.append(AgentAction(
                tool=tool_name,
                input=tool_input,
                output=result,
                success=True,
                duration_ms=duration,
            ))
            self._total_actions += 1

            return True, result

        except Exception as e:
            duration = (time.time() - start) * 1000
            self._actions.append(AgentAction(
                tool=tool_name,
                input=tool_input,
                success=False,
                error=str(e),
                duration_ms=duration,
            ))
            return False, str(e)

    @abstractmethod
    async def plan(self, task: AgentTask) -> List[Dict[str, Any]]:
        """
        Create a plan to complete the task.

        Returns list of steps: [{"action": "...", "tool": "...", ...}]
        """
        pass

    @abstractmethod
    async def execute_step(self, step: Dict[str, Any]) -> Tuple[bool, Any]:
        """Execute a single step of the plan."""
        pass

    async def run_task(self, task: AgentTask) -> bool:
        """
        Execute a task through the OODA loop.

        OBSERVE -> ORIENT -> DECIDE -> ACT -> REFLECT
        """
        self._current_task = task
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now(timezone.utc)
        self.state = AgentState.PLANNING

        self._add_thought(
            f"Starting task: {task.name}",
            "observation",
        )

        try:
            # OBSERVE & ORIENT: Create plan
            plan = await self.plan(task)
            self._add_thought(
                f"Created plan with {len(plan)} steps",
                "reasoning",
            )

            # DECIDE & ACT: Execute plan
            self.state = AgentState.EXECUTING

            for i, step in enumerate(plan):
                self._add_thought(
                    f"Executing step {i+1}/{len(plan)}: {step.get('action', 'unknown')}",
                    "decision",
                )

                success, result = await self.execute_step(step)

                if not success:
                    if task.retries < task.max_retries:
                        task.retries += 1
                        self._add_thought(
                            f"Step failed, retrying ({task.retries}/{task.max_retries})",
                            "reflection",
                        )
                        # Retry from this step
                        continue
                    else:
                        raise RuntimeError(f"Step failed after {task.max_retries} retries: {result}")

            # REFLECT: Task completed
            self.state = AgentState.REFLECTING
            self._add_thought(
                f"Task completed successfully",
                "reflection",
            )

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            task.result = "Success"

            self._completed_tasks.add(task.id)
            self._successful_tasks += 1
            self._total_tasks += 1

            return True

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now(timezone.utc)

            self._add_thought(
                f"Task failed: {e}",
                "reflection",
                confidence=0.5,
            )

            self.state = AgentState.ERROR
            self._total_tasks += 1

            return False

        finally:
            self._current_task = None
            self.state = AgentState.IDLE

    async def run(self) -> None:
        """
        Main agent loop - process tasks from queue.
        """
        while True:
            # Find next ready task
            ready_task = None
            for task in self._task_queue:
                if task.status == TaskStatus.PENDING and task.is_ready(self._completed_tasks):
                    ready_task = task
                    break

            if ready_task:
                self._task_queue.remove(ready_task)
                await self.run_task(ready_task)
            else:
                # No ready tasks, wait
                await asyncio.sleep(1)

    def get_status(self) -> Dict[str, Any]:
        """Get agent status summary."""
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "current_task": self._current_task.name if self._current_task else None,
            "queued_tasks": len(self._task_queue),
            "completed_tasks": len(self._completed_tasks),
            "total_tasks": self._total_tasks,
            "success_rate": self._successful_tasks / max(self._total_tasks, 1),
            "total_actions": self._total_actions,
            "ihsan_violations": self._ihsan_violations,
            "recent_thoughts": len(self._thoughts),
        }


class SimpleAgent(AutonomousAgent):
    """Simple agent implementation for basic tasks."""

    async def plan(self, task: AgentTask) -> List[Dict[str, Any]]:
        """Create a simple linear plan."""
        # For simple tasks, just execute directly
        return [{"action": "execute", "task": task.description}]

    async def execute_step(self, step: Dict[str, Any]) -> Tuple[bool, Any]:
        """Execute a step."""
        action = step.get("action", "")

        if action == "execute" and self.llm_fn:
            task_desc = step.get("task", "")
            try:
                result = self.llm_fn(f"Complete this task: {task_desc}")
                return True, result
            except Exception as e:
                return False, str(e)

        elif action in self._tools:
            return await self._use_tool(action, step.get("input", {}))

        return True, "Step completed"
