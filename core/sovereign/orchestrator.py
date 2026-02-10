"""
Sovereign Orchestrator — Task Decomposition & Agent Routing

Standing on the Shoulders of:
- Claude-Flow V3 (ruv.io, 2026) — Multi-agent swarm coordination
- DSPy (Stanford NLP, 2024) — Self-optimizing prompt pipelines
- LangGraph (LangChain, 2024) — Stateful agent workflows
- Crew AI (2024) — Role-based agent collaboration
- AutoGen (Microsoft, 2024) — Multi-agent conversation patterns

The Orchestrator is the conductor of the sovereign symphony.
It decomposes complex tasks, routes to specialized agents,
and coordinates the flow of information with SNR maximization.
"""

from __future__ import annotations

import asyncio
import heapq
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class TaskComplexity(Enum):
    """Complexity levels for task classification."""

    TRIVIAL = auto()  # Single-step, no decomposition needed
    SIMPLE = auto()  # 2-3 steps, linear execution
    MODERATE = auto()  # 4-7 steps, some parallelization
    COMPLEX = auto()  # 8-15 steps, significant parallelization
    RESEARCH = auto()  # Open-ended, iterative refinement
    SOVEREIGN = auto()  # Multi-domain, requires full council


class TaskStatus(Enum):
    """Status of a task in the orchestration pipeline."""

    PENDING = auto()
    QUEUED = auto()
    ASSIGNED = auto()
    IN_PROGRESS = auto()
    WAITING_DEPENDENCY = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class AgentType(Enum):
    """Types of specialized agents in the swarm."""

    RESEARCHER = auto()  # Information gathering, RAG queries
    ANALYST = auto()  # Data analysis, pattern recognition
    SYNTHESIZER = auto()  # Combining information, summarization
    CREATOR = auto()  # Content generation, code writing
    VALIDATOR = auto()  # Quality assurance, fact-checking
    PLANNER = auto()  # Strategic planning, goal decomposition
    EXECUTOR = auto()  # Action execution, tool usage
    COMMUNICATOR = auto()  # User interaction, clarification
    SPECIALIST = auto()  # Domain-specific expertise


class RoutingStrategy(Enum):
    """Strategies for routing tasks to agents."""

    ROUND_ROBIN = auto()  # Equal distribution
    LOAD_BALANCED = auto()  # Based on current workload
    EXPERTISE_MATCHED = auto()  # Based on task requirements
    PRIORITY_QUEUE = auto()  # Based on task urgency
    ADAPTIVE = auto()  # ML-based dynamic routing


@dataclass
class TaskNode:
    """A single task in the decomposition graph."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    complexity: TaskComplexity = TaskComplexity.SIMPLE
    status: TaskStatus = TaskStatus.PENDING
    priority: float = 0.5  # [0, 1] — higher is more urgent
    assigned_agent: Optional[AgentType] = None
    dependencies: list[str] = field(default_factory=list)  # Task IDs
    outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    snr_score: float = 0.0

    def __lt__(self, other: TaskNode) -> bool:
        """For priority queue ordering."""
        return self.priority > other.priority  # Higher priority first


@dataclass
class AgentCapability:
    """Describes an agent's capabilities and current state."""

    agent_type: AgentType
    name: str
    expertise_domains: list[str]
    max_concurrent_tasks: int = 3
    current_tasks: int = 0
    success_rate: float = 0.95
    average_latency_ms: float = 1000.0
    specialization_score: dict[str, float] = field(default_factory=dict)

    @property
    def available_capacity(self) -> int:
        return self.max_concurrent_tasks - self.current_tasks

    @property
    def load_factor(self) -> float:
        return self.current_tasks / self.max_concurrent_tasks


@dataclass
class DecompositionResult:
    """Result of task decomposition."""

    root_task: TaskNode
    subtasks: list[TaskNode]
    dependency_graph: dict[str, list[str]]  # task_id -> [dependent_task_ids]
    estimated_parallelism: float
    critical_path: list[str]
    total_estimated_steps: int


@dataclass
class RoutingDecision:
    """Decision about how to route a task."""

    task_id: str
    selected_agent: AgentType
    reasoning: str
    confidence: float
    alternative_agents: list[AgentType]
    estimated_latency_ms: float


# Agent role prompts for specialized execution
_AGENT_ROLE_PROMPTS: dict[AgentType, str] = {
    AgentType.RESEARCHER: (
        "Your role: Gather, verify, and synthesize information from available knowledge. "
        "Focus on accuracy and source attribution. Cite evidence."
    ),
    AgentType.ANALYST: (
        "Your role: Analyze data, identify patterns, and extract insights. "
        "Use quantitative reasoning. Present findings with confidence levels."
    ),
    AgentType.SYNTHESIZER: (
        "Your role: Combine information from multiple sources into a coherent narrative. "
        "Resolve contradictions. Identify the signal in the noise."
    ),
    AgentType.CREATOR: (
        "Your role: Generate high-quality content, code, or designs. "
        "Be original yet grounded. Follow best practices."
    ),
    AgentType.VALIDATOR: (
        "Your role: Verify correctness, check for errors, and ensure quality. "
        "Apply rigorous standards. Flag issues with specific evidence."
    ),
    AgentType.PLANNER: (
        "Your role: Decompose goals into actionable steps. "
        "Consider dependencies, risks, and resource constraints."
    ),
    AgentType.EXECUTOR: (
        "Your role: Execute the task directly and efficiently. "
        "Focus on completion. Report results clearly."
    ),
    AgentType.COMMUNICATOR: (
        "Your role: Translate technical content for the intended audience. "
        "Be clear, concise, and empathetic."
    ),
    AgentType.SPECIALIST: (
        "Your role: Apply deep domain expertise to the problem. "
        "Provide expert-level analysis unavailable from generalists."
    ),
}


class TaskDecomposer:
    """
    Decomposes complex tasks into executable subtasks.

    Uses hierarchical decomposition with dependency tracking
    to enable maximum parallelization while respecting constraints.
    """

    # Decomposition templates for different complexity levels
    DECOMPOSITION_PATTERNS = {
        TaskComplexity.TRIVIAL: 1,
        TaskComplexity.SIMPLE: 3,
        TaskComplexity.MODERATE: 6,
        TaskComplexity.COMPLEX: 12,
        TaskComplexity.RESEARCH: 20,
        TaskComplexity.SOVEREIGN: 30,
    }

    def __init__(self):
        self.decomposition_history: list[DecompositionResult] = []

    async def decompose(
        self,
        task: TaskNode,
        max_depth: int = 3,
        min_subtask_size: float = 0.1,
    ) -> DecompositionResult:
        """
        Decompose a task into subtasks with dependency tracking.

        Args:
            task: The root task to decompose
            max_depth: Maximum levels of decomposition
            min_subtask_size: Minimum relative size of a subtask

        Returns:
            DecompositionResult with subtasks and dependency graph
        """
        subtasks: List[TaskNode] = []
        dependency_graph: Dict[str, List[str]] = {}

        # Determine target subtask count based on complexity
        target_count = self.DECOMPOSITION_PATTERNS.get(task.complexity, 5)

        # Generate subtasks (placeholder — real implementation uses LLM)
        subtask_templates = self._get_subtask_templates(task.complexity)

        for i, template in enumerate(subtask_templates[:target_count]):
            subtask = TaskNode(
                title=f"{task.title} — Step {i+1}: {template['name']}",
                description=template["description"],
                complexity=template["complexity"],
                priority=task.priority * template.get("priority_factor", 1.0),
                assigned_agent=template.get("suggested_agent"),
                metadata={
                    "parent_task": task.id,
                    "step_index": i,
                    "template": template["name"],
                },
            )
            subtasks.append(subtask)
            dependency_graph[subtask.id] = []

        # Establish dependencies (linear by default, override with graph analysis)
        for i in range(1, len(subtasks)):
            subtasks[i].dependencies.append(subtasks[i - 1].id)
            dependency_graph[subtasks[i - 1].id].append(subtasks[i].id)

        # Identify parallelizable tasks
        parallelism = self._calculate_parallelism(subtasks, dependency_graph)

        # Find critical path
        critical_path = self._find_critical_path(subtasks, dependency_graph)

        result = DecompositionResult(
            root_task=task,
            subtasks=subtasks,
            dependency_graph=dependency_graph,
            estimated_parallelism=parallelism,
            critical_path=critical_path,
            total_estimated_steps=len(subtasks),
        )

        self.decomposition_history.append(result)
        return result

    def _get_subtask_templates(
        self, complexity: TaskComplexity
    ) -> list[dict[str, Any]]:
        """Get subtask templates based on complexity."""
        base_templates = [
            {
                "name": "Understand",
                "description": "Analyze and understand the requirements",
                "complexity": TaskComplexity.SIMPLE,
                "suggested_agent": AgentType.ANALYST,
            },
            {
                "name": "Research",
                "description": "Gather relevant information and context",
                "complexity": TaskComplexity.SIMPLE,
                "suggested_agent": AgentType.RESEARCHER,
            },
            {
                "name": "Plan",
                "description": "Create execution strategy",
                "complexity": TaskComplexity.SIMPLE,
                "suggested_agent": AgentType.PLANNER,
            },
            {
                "name": "Execute",
                "description": "Perform the core task",
                "complexity": TaskComplexity.MODERATE,
                "suggested_agent": AgentType.EXECUTOR,
            },
            {
                "name": "Validate",
                "description": "Verify outputs meet requirements",
                "complexity": TaskComplexity.SIMPLE,
                "suggested_agent": AgentType.VALIDATOR,
            },
            {
                "name": "Synthesize",
                "description": "Combine results into coherent output",
                "complexity": TaskComplexity.SIMPLE,
                "suggested_agent": AgentType.SYNTHESIZER,
            },
        ]

        if complexity in [
            TaskComplexity.COMPLEX,
            TaskComplexity.RESEARCH,
            TaskComplexity.SOVEREIGN,
        ]:
            base_templates.extend(
                [
                    {
                        "name": "Deep Research",
                        "description": "Extended information gathering",
                        "complexity": TaskComplexity.MODERATE,
                        "suggested_agent": AgentType.RESEARCHER,
                    },
                    {
                        "name": "Cross-Reference",
                        "description": "Validate across multiple sources",
                        "complexity": TaskComplexity.MODERATE,
                        "suggested_agent": AgentType.VALIDATOR,
                    },
                    {
                        "name": "Iterate",
                        "description": "Refine based on intermediate results",
                        "complexity": TaskComplexity.MODERATE,
                        "suggested_agent": AgentType.CREATOR,
                    },
                    {
                        "name": "Expert Review",
                        "description": "Domain specialist evaluation",
                        "complexity": TaskComplexity.MODERATE,
                        "suggested_agent": AgentType.SPECIALIST,
                    },
                ]
            )

        if complexity == TaskComplexity.SOVEREIGN:
            base_templates.extend(
                [
                    {
                        "name": "Council Review",
                        "description": "Guardian Council deliberation",
                        "complexity": TaskComplexity.COMPLEX,
                        "suggested_agent": AgentType.VALIDATOR,
                        "priority_factor": 1.2,
                    },
                    {
                        "name": "Ethical Audit",
                        "description": "Ihsān compliance verification",
                        "complexity": TaskComplexity.MODERATE,
                        "suggested_agent": AgentType.VALIDATOR,
                        "priority_factor": 1.3,
                    },
                ]
            )

        return base_templates

    def _calculate_parallelism(
        self, subtasks: list[TaskNode], dependency_graph: dict[str, list[str]]
    ) -> float:
        """Calculate the degree of parallelism possible."""
        if not subtasks:
            return 1.0

        # Count tasks at each level
        levels = self._topological_levels(subtasks, dependency_graph)
        if not levels:
            return 1.0

        max_level_size = max(len(level) for level in levels)
        return max_level_size / len(subtasks) if subtasks else 1.0

    def _topological_levels(
        self, subtasks: list[TaskNode], dependency_graph: dict[str, list[str]]
    ) -> list[list[str]]:
        """Group tasks into parallel execution levels."""
        task_ids = {t.id for t in subtasks}
        in_degree = {tid: 0 for tid in task_ids}

        # Calculate in-degrees
        for task in subtasks:
            for dep in task.dependencies:
                if dep in task_ids:
                    in_degree[task.id] = in_degree.get(task.id, 0) + 1

        levels = []
        remaining = set(task_ids)

        while remaining:
            # Find tasks with no remaining dependencies
            level = [tid for tid in remaining if in_degree.get(tid, 0) == 0]
            if not level:
                break  # Cycle detected

            levels.append(level)

            # Remove completed tasks and update in-degrees
            for tid in level:
                remaining.discard(tid)
                for dependent in dependency_graph.get(tid, []):
                    in_degree[dependent] = in_degree.get(dependent, 1) - 1

        return levels

    def _find_critical_path(
        self, subtasks: list[TaskNode], dependency_graph: dict[str, list[str]]
    ) -> list[str]:
        """Find the longest path through the dependency graph."""
        if not subtasks:
            return []

        {t.id: t for t in subtasks}

        # Dynamic programming for longest path
        def dfs(task_id: str, memo: dict) -> tuple[int, list[str]]:
            if task_id in memo:
                return memo[task_id]

            dependents = dependency_graph.get(task_id, [])
            if not dependents:
                memo[task_id] = (1, [task_id])
                return memo[task_id]

            best_length = 0
            best_path = []

            for dep_id in dependents:
                length, path = dfs(dep_id, memo)
                if length > best_length:
                    best_length = length
                    best_path = path

            result = (best_length + 1, [task_id] + best_path)
            memo[task_id] = result
            return result

        # Find starting tasks (no dependencies)
        starts = [t.id for t in subtasks if not t.dependencies]

        memo: dict = {}
        longest_path: list[str] = []
        max_length = 0

        for start in starts:
            length, path = dfs(start, memo)
            if length > max_length:
                max_length = length
                longest_path = path

        return longest_path


class AgentRouter:
    """
    Routes tasks to appropriate agents based on capabilities and load.

    Implements multiple routing strategies with adaptive optimization.
    """

    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.agents: dict[AgentType, list[AgentCapability]] = {}
        self.routing_history: list[RoutingDecision] = []

    def register_agent(self, capability: AgentCapability):
        """Register an agent with its capabilities."""
        if capability.agent_type not in self.agents:
            self.agents[capability.agent_type] = []
        self.agents[capability.agent_type].append(capability)

    def _match_expertise(self, task: TaskNode, agent: AgentCapability) -> float:
        """Calculate expertise match score between task and agent."""
        task_domains = task.metadata.get("domains", [])
        if not task_domains:
            return 0.5  # Default match

        scores = []
        for domain in task_domains:
            if domain in agent.specialization_score:
                scores.append(agent.specialization_score[domain])
            elif domain in agent.expertise_domains:
                scores.append(0.8)
            else:
                scores.append(0.3)

        return sum(scores) / len(scores) if scores else 0.5

    async def route(self, task: TaskNode) -> RoutingDecision:
        """Route a task to the most appropriate agent."""
        if task.assigned_agent and task.assigned_agent in self.agents:
            # Pre-assigned agent — verify availability
            candidates = self.agents[task.assigned_agent]
            available = [a for a in candidates if a.available_capacity > 0]
            if available:
                selected = min(available, key=lambda a: a.load_factor)
                return RoutingDecision(
                    task_id=task.id,
                    selected_agent=task.assigned_agent,
                    reasoning="Pre-assigned agent with available capacity",
                    confidence=0.95,
                    alternative_agents=[],
                    estimated_latency_ms=selected.average_latency_ms,
                )

        # Route based on strategy
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._route_round_robin(task)
        elif self.strategy == RoutingStrategy.LOAD_BALANCED:
            return self._route_load_balanced(task)
        elif self.strategy == RoutingStrategy.EXPERTISE_MATCHED:
            return self._route_expertise_matched(task)
        elif self.strategy == RoutingStrategy.PRIORITY_QUEUE:
            return self._route_priority(task)
        else:  # ADAPTIVE
            return self._route_adaptive(task)

    def _route_round_robin(self, task: TaskNode) -> RoutingDecision:
        """Simple round-robin routing."""
        for agent_type, agents in self.agents.items():
            available = [a for a in agents if a.available_capacity > 0]
            if available:
                selected = available[0]
                return RoutingDecision(
                    task_id=task.id,
                    selected_agent=agent_type,
                    reasoning="Round-robin selection",
                    confidence=0.7,
                    alternative_agents=list(self.agents.keys())[:3],
                    estimated_latency_ms=selected.average_latency_ms,
                )

        # Fallback to first agent type
        return RoutingDecision(
            task_id=task.id,
            selected_agent=AgentType.EXECUTOR,
            reasoning="Fallback — no available agents",
            confidence=0.3,
            alternative_agents=[],
            estimated_latency_ms=5000.0,
        )

    def _route_load_balanced(self, task: TaskNode) -> RoutingDecision:
        """Route to agent with lowest load."""
        all_agents = []
        for agent_type, agents in self.agents.items():
            for agent in agents:
                all_agents.append((agent_type, agent))

        if not all_agents:
            return self._route_round_robin(task)

        # Sort by load factor
        all_agents.sort(key=lambda x: x[1].load_factor)
        selected_type, selected = all_agents[0]

        return RoutingDecision(
            task_id=task.id,
            selected_agent=selected_type,
            reasoning=f"Lowest load: {selected.load_factor:.2f}",
            confidence=0.8,
            alternative_agents=[t for t, _ in all_agents[1:4]],
            estimated_latency_ms=selected.average_latency_ms,
        )

    def _route_expertise_matched(self, task: TaskNode) -> RoutingDecision:
        """Route based on expertise match."""
        best_score = 0.0
        best_type = None
        best_agent = None

        for agent_type, agents in self.agents.items():
            for agent in agents:
                if agent.available_capacity > 0:
                    score = self._match_expertise(task, agent)
                    if score > best_score:
                        best_score = score
                        best_type = agent_type
                        best_agent = agent

        if best_type and best_agent:
            return RoutingDecision(
                task_id=task.id,
                selected_agent=best_type,
                reasoning=f"Best expertise match: {best_score:.2f}",
                confidence=best_score,
                alternative_agents=[],
                estimated_latency_ms=best_agent.average_latency_ms,
            )

        return self._route_round_robin(task)

    def _route_priority(self, task: TaskNode) -> RoutingDecision:
        """Route high-priority tasks to fastest agents."""
        if task.priority > 0.8:
            # High priority — find fastest available agent
            fastest = None
            fastest_type = None
            fastest_latency = float("inf")

            for agent_type, agents in self.agents.items():
                for agent in agents:
                    if (
                        agent.available_capacity > 0
                        and agent.average_latency_ms < fastest_latency
                    ):
                        fastest = agent
                        fastest_type = agent_type
                        fastest_latency = agent.average_latency_ms

            if fastest and fastest_type:
                return RoutingDecision(
                    task_id=task.id,
                    selected_agent=fastest_type,
                    reasoning=f"Priority routing: fastest agent ({fastest_latency:.0f}ms)",
                    confidence=0.9,
                    alternative_agents=[],
                    estimated_latency_ms=fastest_latency,
                )

        return self._route_load_balanced(task)

    def _route_adaptive(self, task: TaskNode) -> RoutingDecision:
        """
        Adaptive routing combining multiple factors.

        Score = w1*expertise + w2*(1-load) + w3*success_rate + w4*(1/latency)
        """
        candidates = []

        for agent_type, agents in self.agents.items():
            for agent in agents:
                if agent.available_capacity <= 0:
                    continue

                expertise_score = self._match_expertise(task, agent)
                load_score = 1.0 - agent.load_factor
                success_score = agent.success_rate
                latency_score = 1000.0 / (agent.average_latency_ms + 100)  # Normalize

                # Weighted combination
                w1, w2, w3, w4 = 0.35, 0.25, 0.25, 0.15
                total_score = (
                    w1 * expertise_score
                    + w2 * load_score
                    + w3 * success_score
                    + w4 * latency_score
                )

                candidates.append((total_score, agent_type, agent))

        if not candidates:
            return self._route_round_robin(task)

        candidates.sort(reverse=True, key=lambda x: x[0])
        best_score, best_type, best_agent = candidates[0]

        return RoutingDecision(
            task_id=task.id,
            selected_agent=best_type,
            reasoning=f"Adaptive score: {best_score:.3f}",
            confidence=min(1.0, best_score),
            alternative_agents=[t for _, t, _ in candidates[1:4]],
            estimated_latency_ms=best_agent.average_latency_ms,
        )


class SovereignOrchestrator:
    """
    The Master Orchestrator — Coordinates all sovereign operations.

    Integrates task decomposition, agent routing, and execution
    with SNR maximization and Ihsān compliance.
    """

    def __init__(
        self,
        routing_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
        max_concurrent_tasks: int = 10,
        snr_threshold: float = 0.95,
    ):
        self.decomposer = TaskDecomposer()
        self.router = AgentRouter(strategy=routing_strategy)
        self.max_concurrent = max_concurrent_tasks
        self.snr_threshold = snr_threshold

        self.task_queue: list[TaskNode] = []
        self.active_tasks: dict[str, TaskNode] = {}
        self.completed_tasks: dict[str, TaskNode] = {}
        self.task_results: dict[str, Any] = {}

        self._running = False
        self._execution_loop: Optional[asyncio.Task] = None

        # External integrations (injected via set_gateway / set_memory)
        self._gateway: Any = None  # InferenceGateway
        self._memory: Any = None  # LivingMemoryCore

    def register_default_agents(self):
        """Register a default set of agents."""
        default_agents = [
            AgentCapability(
                AgentType.RESEARCHER, "Researcher-1", ["information", "rag", "web"], 5
            ),
            AgentCapability(
                AgentType.ANALYST, "Analyst-1", ["data", "patterns", "metrics"], 3
            ),
            AgentCapability(
                AgentType.SYNTHESIZER, "Synthesizer-1", ["summary", "integration"], 3
            ),
            AgentCapability(
                AgentType.CREATOR, "Creator-1", ["code", "content", "design"], 3
            ),
            AgentCapability(
                AgentType.VALIDATOR, "Validator-1", ["quality", "testing", "review"], 4
            ),
            AgentCapability(
                AgentType.PLANNER, "Planner-1", ["strategy", "goals", "roadmap"], 2
            ),
            AgentCapability(
                AgentType.EXECUTOR, "Executor-1", ["tools", "actions", "automation"], 5
            ),
            AgentCapability(
                AgentType.SPECIALIST, "Specialist-1", ["domain", "expert"], 2
            ),
        ]

        for agent in default_agents:
            self.router.register_agent(agent)

    async def submit(self, task: TaskNode) -> str:
        """Submit a task for orchestration."""
        # Decompose if complex
        if task.complexity.value >= TaskComplexity.MODERATE.value:
            result = await self.decomposer.decompose(task)
            for subtask in result.subtasks:
                heapq.heappush(self.task_queue, subtask)
        else:
            heapq.heappush(self.task_queue, task)

        return task.id

    async def execute_task(self, task: TaskNode) -> dict[str, Any]:
        """
        Execute a single task through the inference pipeline.

        Routes task to specialized agent, builds a focused prompt,
        and calls the inference gateway for real LLM completion.
        Falls back to heuristic execution when no gateway available.
        """
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        self.active_tasks[task.id] = task

        # Route to appropriate agent
        routing = await self.router.route(task)

        try:
            # Build agent-specific prompt
            prompt = self._build_agent_prompt(task, routing)

            # Try real inference via gateway
            content = await self._execute_via_gateway(prompt)

            if content is None:
                # Fallback: heuristic execution (no LLM available)
                content = self._heuristic_execute(task, routing)

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.snr_score = self._score_output(content)
            task.outputs["content"] = content
            task.outputs["agent"] = routing.selected_agent.name

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.snr_score = 0.0
            content = f"[ERROR] {e}"

        result = {
            "task_id": task.id,
            "agent": routing.selected_agent.name,
            "status": task.status.name.lower(),
            "content": content,
            "snr_score": task.snr_score,
            "latency_ms": (
                (task.completed_at - task.started_at).total_seconds() * 1000
                if task.completed_at and task.started_at
                else 0
            ),
        }

        self.task_results[task.id] = result
        if task.id in self.active_tasks:
            del self.active_tasks[task.id]
        self.completed_tasks[task.id] = task

        return result

    # ── Inference integration ───────────────────────────────────────────

    def set_gateway(self, gateway: Any) -> None:
        """Inject an InferenceGateway for real LLM execution."""
        self._gateway = gateway

    def set_memory(self, memory: Any) -> None:
        """Inject a LivingMemoryCore for context retrieval."""
        self._memory = memory

    def _build_agent_prompt(self, task: TaskNode, routing: RoutingDecision) -> str:
        """Build a focused prompt for the assigned agent type."""
        agent_name = routing.selected_agent.name
        role_instructions = _AGENT_ROLE_PROMPTS.get(
            routing.selected_agent, _AGENT_ROLE_PROMPTS[AgentType.EXECUTOR]
        )

        # Gather dependency outputs for context
        dep_context = ""
        for dep_id in task.dependencies:
            dep_result = self.task_results.get(dep_id, {})
            dep_content = dep_result.get("content", "")
            if dep_content:
                dep_agent = dep_result.get("agent", "unknown")
                dep_context += f"\n[{dep_agent}]: {dep_content[:500]}"

        prompt = (
            f"You are {agent_name}, a specialized agent in the BIZRA sovereign system.\n"
            f"{role_instructions}\n\n"
            f"TASK: {task.title}\n"
            f"DESCRIPTION: {task.description}\n"
            f"PRIORITY: {task.priority:.2f}\n"
        )

        if dep_context:
            prompt += f"\nPREVIOUS AGENT OUTPUTS:{dep_context}\n"

        prompt += "\nProvide a concise, high-SNR response. Be specific and actionable."

        return prompt

    async def _execute_via_gateway(self, prompt: str) -> Optional[str]:
        """Execute prompt through InferenceGateway if available."""
        gateway = getattr(self, "_gateway", None)
        if gateway is None:
            return None

        try:
            infer_method = getattr(gateway, "infer", None)
            if infer_method is None:
                return None

            result = await infer_method(prompt, max_tokens=512)
            content = getattr(result, "content", None) or str(result)
            return content if content else None
        except Exception:
            return None

    def _heuristic_execute(self, task: TaskNode, routing: RoutingDecision) -> str:
        """Fallback execution using heuristics when no LLM available."""
        agent = routing.selected_agent

        # Gather any dependency outputs
        dep_summaries = []
        for dep_id in task.dependencies:
            dep_result = self.task_results.get(dep_id, {})
            dep_content = dep_result.get("content", "")
            if dep_content:
                dep_summaries.append(dep_content[:200])

        context_str = "; ".join(dep_summaries) if dep_summaries else "no prior context"

        return (
            f"[{agent.name}] Processed: {task.title}. "
            f"Context: {context_str}. "
            f"Description: {task.description}"
        )

    def _score_output(self, content: str) -> float:
        """Quick SNR score for task output."""
        if not content:
            return 0.0
        words = content.split()
        if len(words) < 3:
            return 0.3
        unique_ratio = len(set(words)) / len(words)
        length_score = min(len(words) / 50, 1.0)
        return min(1.0, 0.5 * unique_ratio + 0.5 * length_score)

    async def run(self):
        """Start the orchestration loop."""
        self._running = True

        while self._running:
            # Check for available capacity
            while self.task_queue and len(self.active_tasks) < self.max_concurrent:
                task = heapq.heappop(self.task_queue)

                # Check dependencies
                deps_satisfied = all(
                    dep_id in self.completed_tasks for dep_id in task.dependencies
                )

                if deps_satisfied:
                    asyncio.create_task(self.execute_task(task))
                else:
                    task.status = TaskStatus.WAITING_DEPENDENCY
                    heapq.heappush(self.task_queue, task)

            await asyncio.sleep(0.01)  # Yield control

    def stop(self):
        """Stop the orchestration loop."""
        self._running = False

    def get_status(self) -> dict[str, Any]:
        """Get current orchestration status."""
        return {
            "queued": len(self.task_queue),
            "active": len(self.active_tasks),
            "completed": len(self.completed_tasks),
            "running": self._running,
        }


# Factory function
def create_orchestrator(
    routing_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
    register_defaults: bool = True,
) -> SovereignOrchestrator:
    """Create and configure a SovereignOrchestrator."""
    orchestrator = SovereignOrchestrator(routing_strategy=routing_strategy)
    if register_defaults:
        orchestrator.register_default_agents()
    return orchestrator
