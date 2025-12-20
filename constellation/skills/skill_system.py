# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Constellation - Sub-Agents & Skills System v1.0
# ═══════════════════════════════════════════════════════════════════════════════
"""
Modular skill system and sub-agent orchestration:
- Skill definitions and registry
- Sub-agent spawning and management
- Skill composition and chaining
- Capability delegation
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any, Callable, Awaitable
from enum import Enum
from abc import ABC, abstractmethod
import uuid


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SKILL TYPES
# ─────────────────────────────────────────────────────────────────────────────

class SkillCategory(str, Enum):
    """Categories of skills."""
    REASONING = "reasoning"      # Logical analysis, deduction
    RESEARCH = "research"        # Information gathering
    COMPUTATION = "computation"  # Mathematical operations
    WRITING = "writing"          # Text generation
    ANALYSIS = "analysis"        # Data/text analysis
    SYNTHESIS = "synthesis"      # Combining information
    VERIFICATION = "verification"  # Fact checking
    TRANSLATION = "translation"  # Language/format conversion
    MEMORY = "memory"           # Knowledge management
    COMMUNICATION = "communication"  # Inter-agent messaging


class SkillLevel(str, Enum):
    """Proficiency levels for skills."""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"


# ─────────────────────────────────────────────────────────────────────────────
# SKILL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SkillInput:
    """Definition of skill input."""
    name: str
    type_hint: str  # "str", "int", "list[str]", etc.
    required: bool = True
    default: Optional[Any] = None
    description: str = ""


@dataclass
class SkillOutput:
    """Definition of skill output."""
    name: str
    type_hint: str
    description: str = ""


SkillHandler = Callable[..., Awaitable[dict[str, Any]]]


@dataclass
class Skill:
    """A modular skill that can be used by agents."""
    id: str
    name: str
    category: SkillCategory
    handler: SkillHandler
    inputs: list[SkillInput] = field(default_factory=list)
    outputs: list[SkillOutput] = field(default_factory=list)
    description: str = ""
    
    # Requirements
    min_level: SkillLevel = SkillLevel.NOVICE
    dependencies: list[str] = field(default_factory=list)  # Other skill IDs
    
    # Metadata
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    
    # Stats
    invocation_count: int = 0
    total_time_ms: float = 0.0
    
    async def execute(self, **kwargs) -> dict[str, Any]:
        """Execute the skill."""
        import time
        
        # Validate required inputs
        for inp in self.inputs:
            if inp.required and inp.name not in kwargs:
                raise ValueError(f"Missing required input: {inp.name}")
                
        # Apply defaults
        for inp in self.inputs:
            if inp.name not in kwargs and inp.default is not None:
                kwargs[inp.name] = inp.default
                
        start = time.perf_counter()
        
        try:
            result = await self.handler(**kwargs)
            
            # Update stats
            self.invocation_count += 1
            self.total_time_ms += (time.perf_counter() - start) * 1000
            
            return result
            
        except Exception as e:
            logger.error(f"Skill {self.name} failed: {e}", exc_info=True)
            raise
            
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "inputs": [{"name": i.name, "type": i.type_hint, "required": i.required} for i in self.inputs],
            "outputs": [{"name": o.name, "type": o.type_hint} for o in self.outputs],
            "min_level": self.min_level.value,
            "dependencies": self.dependencies,
            "version": self.version,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SKILL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class SkillRegistry:
    """Registry for all available skills."""
    
    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._by_category: dict[SkillCategory, list[str]] = {c: [] for c in SkillCategory}
        
    def register(self, skill: Skill) -> None:
        """Register a skill."""
        self._skills[skill.id] = skill
        self._by_category[skill.category].append(skill.id)
        logger.debug(f"Registered skill: {skill.name} ({skill.id})")
        
    def get(self, skill_id: str) -> Optional[Skill]:
        """Get skill by ID."""
        return self._skills.get(skill_id)
        
    def get_by_category(self, category: SkillCategory) -> list[Skill]:
        """Get all skills in a category."""
        return [self._skills[sid] for sid in self._by_category[category]]
        
    def search(
        self,
        query: str,
        category: Optional[SkillCategory] = None,
    ) -> list[Skill]:
        """Search skills by name/description."""
        results = []
        query_lower = query.lower()
        
        for skill in self._skills.values():
            if category and skill.category != category:
                continue
            if query_lower in skill.name.lower() or query_lower in skill.description.lower():
                results.append(skill)
                
        return results
        
    def list_all(self) -> list[Skill]:
        """List all registered skills."""
        return list(self._skills.values())


# ─────────────────────────────────────────────────────────────────────────────
# SUB-AGENT
# ─────────────────────────────────────────────────────────────────────────────

class SubAgentState(str, Enum):
    """State of a sub-agent."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubAgentTask:
    """A task assigned to a sub-agent."""
    id: str
    description: str
    skill_id: str
    inputs: dict[str, Any]
    priority: int = 0
    timeout_seconds: Optional[int] = None
    
    # Results
    state: SubAgentState = SubAgentState.IDLE
    output: Optional[dict] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class SubAgent:
    """
    A sub-agent spawned by a parent agent for specialized tasks.
    
    Sub-agents:
    - Have focused capabilities (subset of skills)
    - Report back to parent agent
    - Can be cancelled or prioritized
    """
    id: str
    name: str
    parent_agent: str
    skills: list[str]  # Skill IDs
    
    # State
    state: SubAgentState = SubAgentState.IDLE
    current_task: Optional[SubAgentTask] = None
    completed_tasks: list[SubAgentTask] = field(default_factory=list)
    
    # Configuration
    max_concurrent_tasks: int = 1
    auto_terminate: bool = True  # Terminate when all tasks complete
    
    # Lifecycle
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    terminated_at: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "parent_agent": self.parent_agent,
            "skills": self.skills,
            "state": self.state.value,
            "current_task": self.current_task.id if self.current_task else None,
            "completed_tasks": len(self.completed_tasks),
            "created_at": self.created_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SUB-AGENT MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class SubAgentManager:
    """
    Manages sub-agent lifecycle and task execution.
    
    Provides:
    - Sub-agent spawning
    - Task assignment and monitoring
    - Result collection
    - Resource management
    """
    
    def __init__(self, skill_registry: SkillRegistry):
        self.skill_registry = skill_registry
        self._sub_agents: dict[str, SubAgent] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
    def spawn(
        self,
        name: str,
        parent_agent: str,
        skills: list[str],
        auto_terminate: bool = True,
    ) -> SubAgent:
        """Spawn a new sub-agent."""
        sub_agent = SubAgent(
            id=f"sub_{uuid.uuid4().hex[:8]}",
            name=name,
            parent_agent=parent_agent,
            skills=skills,
            auto_terminate=auto_terminate,
        )
        
        self._sub_agents[sub_agent.id] = sub_agent
        logger.info(f"Spawned sub-agent: {name} ({sub_agent.id}) for {parent_agent}")
        
        return sub_agent
        
    def assign_task(
        self,
        sub_agent_id: str,
        description: str,
        skill_id: str,
        inputs: dict[str, Any],
        priority: int = 0,
        timeout_seconds: Optional[int] = None,
    ) -> SubAgentTask:
        """Assign a task to a sub-agent."""
        sub_agent = self._sub_agents.get(sub_agent_id)
        if not sub_agent:
            raise ValueError(f"Sub-agent not found: {sub_agent_id}")
            
        if skill_id not in sub_agent.skills:
            raise ValueError(f"Sub-agent {sub_agent_id} doesn't have skill: {skill_id}")
            
        task = SubAgentTask(
            id=f"task_{uuid.uuid4().hex[:8]}",
            description=description,
            skill_id=skill_id,
            inputs=inputs,
            priority=priority,
            timeout_seconds=timeout_seconds,
        )
        
        sub_agent.current_task = task
        return task
        
    async def execute_task(
        self,
        sub_agent_id: str,
        task: SubAgentTask,
    ) -> dict[str, Any]:
        """Execute a task on a sub-agent."""
        sub_agent = self._sub_agents.get(sub_agent_id)
        if not sub_agent:
            raise ValueError(f"Sub-agent not found: {sub_agent_id}")
            
        skill = self.skill_registry.get(task.skill_id)
        if not skill:
            raise ValueError(f"Skill not found: {task.skill_id}")
            
        # Update state
        task.state = SubAgentState.RUNNING
        task.started_at = datetime.now(timezone.utc).isoformat()
        sub_agent.state = SubAgentState.RUNNING
        
        try:
            # Execute with optional timeout
            if task.timeout_seconds:
                result = await asyncio.wait_for(
                    skill.execute(**task.inputs),
                    timeout=task.timeout_seconds,
                )
            else:
                result = await skill.execute(**task.inputs)
                
            # Success
            task.state = SubAgentState.COMPLETED
            task.output = result
            task.completed_at = datetime.now(timezone.utc).isoformat()
            
            # Move to completed
            sub_agent.completed_tasks.append(task)
            sub_agent.current_task = None
            sub_agent.state = SubAgentState.IDLE
            
            # Auto-terminate check
            if sub_agent.auto_terminate and not sub_agent.current_task:
                await self.terminate(sub_agent_id)
                
            return result
            
        except asyncio.TimeoutError:
            task.state = SubAgentState.FAILED
            task.error = "Task timed out"
            task.completed_at = datetime.now(timezone.utc).isoformat()
            sub_agent.state = SubAgentState.FAILED
            raise
            
        except Exception as e:
            task.state = SubAgentState.FAILED
            task.error = str(e)
            task.completed_at = datetime.now(timezone.utc).isoformat()
            sub_agent.state = SubAgentState.FAILED
            raise
            
    async def terminate(self, sub_agent_id: str) -> bool:
        """Terminate a sub-agent."""
        sub_agent = self._sub_agents.get(sub_agent_id)
        if not sub_agent:
            return False
            
        # Cancel current task if running
        if sub_agent.current_task and sub_agent.state == SubAgentState.RUNNING:
            sub_agent.current_task.state = SubAgentState.CANCELLED
            
        sub_agent.state = SubAgentState.COMPLETED
        sub_agent.terminated_at = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Terminated sub-agent: {sub_agent.name} ({sub_agent_id})")
        return True
        
    def get_sub_agent(self, sub_agent_id: str) -> Optional[SubAgent]:
        """Get sub-agent by ID."""
        return self._sub_agents.get(sub_agent_id)
        
    def get_by_parent(self, parent_agent: str) -> list[SubAgent]:
        """Get all sub-agents for a parent agent."""
        return [
            sa for sa in self._sub_agents.values()
            if sa.parent_agent == parent_agent
        ]
        
    def get_active(self) -> list[SubAgent]:
        """Get all active sub-agents."""
        return [
            sa for sa in self._sub_agents.values()
            if sa.state in [SubAgentState.IDLE, SubAgentState.RUNNING]
        ]


# ─────────────────────────────────────────────────────────────────────────────
# SKILL CHAIN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SkillChainStep:
    """A step in a skill chain."""
    skill_id: str
    input_mapping: dict[str, str] = field(default_factory=dict)  # step_input -> chain_input or prev_output
    condition: Optional[Callable[[dict], bool]] = None  # Optional condition to execute


class SkillChain:
    """
    Chain multiple skills together for complex operations.
    
    Provides:
    - Sequential execution
    - Input/output mapping between steps
    - Conditional execution
    - Error handling
    """
    
    def __init__(
        self,
        name: str,
        skill_registry: SkillRegistry,
    ):
        self.name = name
        self.skill_registry = skill_registry
        self.steps: list[SkillChainStep] = []
        
    def add_step(
        self,
        skill_id: str,
        input_mapping: Optional[dict[str, str]] = None,
        condition: Optional[Callable[[dict], bool]] = None,
    ) -> "SkillChain":
        """Add a step to the chain."""
        step = SkillChainStep(
            skill_id=skill_id,
            input_mapping=input_mapping or {},
            condition=condition,
        )
        self.steps.append(step)
        return self
        
    async def execute(self, **initial_inputs) -> dict[str, Any]:
        """Execute the skill chain."""
        context = {"inputs": initial_inputs, "outputs": {}}
        
        for i, step in enumerate(self.steps):
            # Check condition
            if step.condition and not step.condition(context):
                logger.debug(f"Skipping step {i} ({step.skill_id}): condition not met")
                continue
                
            # Get skill
            skill = self.skill_registry.get(step.skill_id)
            if not skill:
                raise ValueError(f"Skill not found: {step.skill_id}")
                
            # Map inputs
            step_inputs = {}
            for step_key, source in step.input_mapping.items():
                if source.startswith("input."):
                    step_inputs[step_key] = initial_inputs.get(source[6:])
                elif source.startswith("output."):
                    parts = source[7:].split(".", 1)
                    step_idx = int(parts[0])
                    output_key = parts[1] if len(parts) > 1 else None
                    prev_output = context["outputs"].get(step_idx, {})
                    step_inputs[step_key] = prev_output.get(output_key) if output_key else prev_output
                else:
                    step_inputs[step_key] = source
                    
            # Execute
            try:
                result = await skill.execute(**step_inputs)
                context["outputs"][i] = result
                
            except Exception as e:
                logger.error(f"Chain step {i} ({step.skill_id}) failed: {e}")
                raise
                
        # Return final output
        if self.steps:
            return context["outputs"].get(len(self.steps) - 1, {})
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL INSTANCES
# ─────────────────────────────────────────────────────────────────────────────

_skill_registry: Optional[SkillRegistry] = None
_sub_agent_manager: Optional[SubAgentManager] = None


def get_skill_registry() -> SkillRegistry:
    """Get the global skill registry."""
    global _skill_registry
    if _skill_registry is None:
        _skill_registry = SkillRegistry()
        _register_builtin_skills()
    return _skill_registry


def get_sub_agent_manager() -> SubAgentManager:
    """Get the global sub-agent manager."""
    global _sub_agent_manager
    if _sub_agent_manager is None:
        _sub_agent_manager = SubAgentManager(get_skill_registry())
    return _sub_agent_manager


# ─────────────────────────────────────────────────────────────────────────────
# BUILTIN SKILLS
# ─────────────────────────────────────────────────────────────────────────────

def _register_builtin_skills() -> None:
    """Register built-in skills."""
    registry = get_skill_registry()
    
    # Summarize skill
    async def summarize_handler(text: str, max_length: int = 200) -> dict:
        """Summarize text to specified length."""
        # Placeholder - would use LLM
        if len(text) <= max_length:
            return {"summary": text}
        return {"summary": text[:max_length] + "..."}
        
    registry.register(Skill(
        id="skill_summarize",
        name="Summarize",
        category=SkillCategory.WRITING,
        handler=summarize_handler,
        inputs=[
            SkillInput("text", "str", description="Text to summarize"),
            SkillInput("max_length", "int", required=False, default=200),
        ],
        outputs=[SkillOutput("summary", "str")],
        description="Summarize text to a specified length",
    ))
    
    # Extract entities skill
    async def extract_entities_handler(text: str) -> dict:
        """Extract named entities from text."""
        # Placeholder
        return {"entities": [], "count": 0}
        
    registry.register(Skill(
        id="skill_extract_entities",
        name="Extract Entities",
        category=SkillCategory.ANALYSIS,
        handler=extract_entities_handler,
        inputs=[SkillInput("text", "str", description="Text to analyze")],
        outputs=[
            SkillOutput("entities", "list[dict]"),
            SkillOutput("count", "int"),
        ],
        description="Extract named entities from text",
    ))
    
    # Verify claim skill
    async def verify_claim_handler(claim: str, context: str = "") -> dict:
        """Verify a claim against context."""
        # Placeholder
        return {"verified": None, "confidence": 0.0, "evidence": []}
        
    registry.register(Skill(
        id="skill_verify_claim",
        name="Verify Claim",
        category=SkillCategory.VERIFICATION,
        handler=verify_claim_handler,
        inputs=[
            SkillInput("claim", "str", description="Claim to verify"),
            SkillInput("context", "str", required=False, default=""),
        ],
        outputs=[
            SkillOutput("verified", "bool"),
            SkillOutput("confidence", "float"),
            SkillOutput("evidence", "list[str]"),
        ],
        description="Verify a claim against knowledge and context",
    ))
    
    # Calculate skill
    async def calculate_handler(expression: str) -> dict:
        """Evaluate mathematical expression."""
        try:
            # Safe eval for simple math
            result = eval(expression, {"__builtins__": {}}, {})
            return {"result": result, "expression": expression}
        except Exception as e:
            return {"error": str(e), "expression": expression}
            
    registry.register(Skill(
        id="skill_calculate",
        name="Calculate",
        category=SkillCategory.COMPUTATION,
        handler=calculate_handler,
        inputs=[SkillInput("expression", "str", description="Math expression")],
        outputs=[SkillOutput("result", "float")],
        description="Evaluate mathematical expressions",
    ))
