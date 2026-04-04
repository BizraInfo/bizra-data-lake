"""
OpenProse Workflow Engine - Extracted from BIZRA-copilot

Standing on the Shoulders of Giants Protocol:
This module synthesizes multi-agent workflow patterns from:
https://github.com/BizraInfo/BIZRA-copilot.git

Key Patterns Implemented:
1. Pipeline Composition (|, filter, map, reduce)
2. Model Tiering (opus/sonnet/haiku routing)
3. Persistent Agents (execution-scoped, user-scoped, global)
4. Workflow Blocks (parameterized reusable workflows)
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openprose.workflow")


# ============================================================================
# MODEL TIERING (Standing on Giants Protocol)
# ============================================================================


class ModelTier(Enum):
    """
    Model capability tiers - extracted from BIZRA-copilot OpenProse patterns.

    Usage guidance:
    - OPUS: Hard analytical work, complex reasoning, coordination
    - SONNET: General-purpose, good balance of quality/speed
    - HAIKU: Simple transformations (use sparingly to avoid overuse)
    """

    OPUS = "opus"  # Top-tier: complex reasoning, coordination
    SONNET = "sonnet"  # Mid-tier: general purpose, balanced
    HAIKU = "haiku"  # Fast-tier: simple transformations only


class PersistenceScope(Enum):
    """
    Agent persistence scopes - extracted from OpenProse patterns.
    """

    EPHEMERAL = "ephemeral"  # Dies after single use
    EXECUTION = "execution"  # Dies with workflow run
    USER = "user"  # Survives across projects (user-scoped)
    GLOBAL = "global"  # Survives across users (system-scoped)


@dataclass
class ModelConfig:
    """Model configuration for agent routing."""

    tier: ModelTier
    temperature: float = 0.7
    max_tokens: int = 4096

    # Model mapping (from model-family-genesis-v1-SEALED.yaml)
    MODEL_MAP: Dict[ModelTier, str] = field(
        default_factory=lambda: {
            ModelTier.OPUS: "anthropic/claude-opus-4-5",
            ModelTier.SONNET: "anthropic/claude-sonnet-4-5",
            ModelTier.HAIKU: "anthropic/claude-haiku",
        }
    )

    @property
    def model_id(self) -> str:
        return self.MODEL_MAP.get(self.tier, self.MODEL_MAP[ModelTier.SONNET])


# ============================================================================
# AGENT FRAMEWORK
# ============================================================================


@dataclass
class AgentDefinition:
    """
    Agent definition - extracted from OpenProse 'agent' blocks.

    Example in OpenProse:
        agent captain:
          model: opus
          persist: true
          prompt: "You coordinate the team"
    """

    name: str
    model: ModelTier = ModelTier.SONNET
    persist: PersistenceScope = PersistenceScope.EXECUTION
    prompt: str = ""
    context: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.id = (
            f"agent_{self.name}_{hashlib.sha256(self.prompt.encode()).hexdigest()[:8]}"
        )


T = TypeVar("T")


class Agent:
    """
    Agent instance - executes prompts with model tiering.
    """

    def __init__(self, definition: AgentDefinition):
        self.definition = definition
        self.history: List[Dict[str, str]] = []
        self.created_at = datetime.now(timezone.utc)

    async def session(self, prompt: str, context: Optional[List[str]] = None) -> str:
        """
        Execute a session (single inference call).

        Maps to OpenProse: session "prompt" context: [...]
        """
        # Build context
        full_context = self.definition.context.copy()
        if context:
            full_context.extend(context)

        # Construct message
        system_prompt = self.definition.prompt
        user_prompt = prompt

        if full_context:
            user_prompt = f"Context:\n{chr(10).join(full_context)}\n\nTask: {prompt}"

        # Log execution (actual LLM call would go here)
        logger.info(
            f"🤖 Agent [{self.definition.name}] executing session "
            f"(model={self.definition.model.value}, persist={self.definition.persist.value})"
        )

        # Record in history
        self.history.append(
            {
                "role": "user",
                "content": user_prompt,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Placeholder for actual LLM response
        # In production, this would call the appropriate model via BIZRA's model routing
        response = f"[{self.definition.model.value}] Response to: {prompt[:50]}..."

        self.history.append(
            {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return response


# ============================================================================
# PIPELINE OPERATIONS
# ============================================================================


class PipelineOp(ABC, Generic[T]):
    """Base class for pipeline operations."""

    @abstractmethod
    async def execute(self, input_data: T, agent: Agent) -> Any:
        pass


@dataclass
class FilterOp(PipelineOp[List[T]]):
    """
    Filter operation - extracted from OpenProse '| filter:' pattern.

    Example:
        candidates | filter: session "Is this feasible?"
    """

    prompt: str

    async def execute(self, items: List[T], agent: Agent) -> List[T]:
        result = []
        for item in items:
            response = await agent.session(self.prompt, context=[str(item)])
            # Simple heuristic: keep if response contains positive signal
            if any(
                word in response.lower()
                for word in ["yes", "true", "feasible", "valid"]
            ):
                result.append(item)
        return result


@dataclass
class MapOp(PipelineOp[List[T]]):
    """
    Map operation - extracted from OpenProse '| map:' pattern.

    Example:
        candidates | map: session "Expand into one-page pitch"
    """

    prompt: str

    async def execute(self, items: List[T], agent: Agent) -> List[str]:
        results = []
        for item in items:
            response = await agent.session(self.prompt, context=[str(item)])
            results.append(response)
        return results


@dataclass
class ReduceOp(PipelineOp[List[T]]):
    """
    Reduce operation - extracted from OpenProse '| reduce:' pattern.

    Example:
        candidates | reduce: session "Compare and return stronger"
    """

    prompt: str

    async def execute(self, items: List[T], agent: Agent) -> Optional[T]:
        if not items:
            return None
        if len(items) == 1:
            return items[0]

        # Pairwise reduction
        current = items[0]
        for item in items[1:]:
            response = await agent.session(
                self.prompt, context=[str(current), str(item)]
            )
            # Use response as the "winner" of comparison
            current = response
        return current


# ============================================================================
# PIPELINE BUILDER
# ============================================================================


class Pipeline(Generic[T]):
    """
    Pipeline builder - implements OpenProse's pipe composition.

    Example:
        let result = candidates
          | filter: session "Is this feasible?"
          | map: session "Expand into pitch"
          | reduce: session "Compare and return stronger"
    """

    def __init__(self, data: T, agent: Agent):
        self.data = data
        self.agent = agent
        self.operations: List[PipelineOp] = []

    def filter(self, prompt: str) -> "Pipeline[List[Any]]":
        """Add filter operation."""
        self.operations.append(FilterOp(prompt))
        return self

    def map(self, prompt: str) -> "Pipeline[List[str]]":
        """Add map operation."""
        self.operations.append(MapOp(prompt))
        return self

    def reduce(self, prompt: str) -> "Pipeline[Optional[Any]]":
        """Add reduce operation."""
        self.operations.append(ReduceOp(prompt))
        return self

    async def execute(self) -> Any:
        """Execute the pipeline."""
        result = self.data
        for op in self.operations:
            result = await op.execute(result, self.agent)
            logger.info(
                f"Pipeline step {type(op).__name__}: {len(result) if isinstance(result, list) else 1} items"
            )
        return result


# ============================================================================
# WORKFLOW BLOCKS (Reusable Parameterized Workflows)
# ============================================================================


@dataclass
class WorkflowBlock:
    """
    Workflow block - extracted from OpenProse 'block' pattern.

    Example:
        block research-report(topic, depth):
          let research = session "Research {topic}"
          let analysis = session "Analyze findings"
          let report = session "Write report"
    """

    name: str
    parameters: List[str]
    steps: List[Callable[..., str]]

    def __post_init__(self):
        self.id = f"block_{self.name}"


class WorkflowEngine:
    """
    Workflow engine - orchestrates multi-agent workflows.

    Implements the OpenProse VM pattern for complex agentic pipelines.
    """

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.blocks: Dict[str, WorkflowBlock] = {}
        self.execution_id = hashlib.sha256(
            datetime.now(timezone.utc).isoformat().encode()
        ).hexdigest()[:12]

    def define_agent(self, definition: AgentDefinition) -> Agent:
        """
        Define a new agent in the workflow.

        Example:
            engine.define_agent(AgentDefinition(
                name="captain",
                model=ModelTier.OPUS,
                persist=PersistenceScope.EXECUTION,
                prompt="You coordinate the team"
            ))
        """
        agent = Agent(definition)
        self.agents[definition.name] = agent
        logger.info(
            f"🎯 Defined agent: {definition.name} (model={definition.model.value})"
        )
        return agent

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get agent by name."""
        return self.agents.get(name)

    def register_block(self, block: WorkflowBlock) -> None:
        """Register a reusable workflow block."""
        self.blocks[block.name] = block
        logger.info(f"📦 Registered block: {block.name}({', '.join(block.parameters)})")

    def pipeline(self, data: Any, agent_name: str) -> Pipeline:
        """
        Create a pipeline with the specified agent.

        Example:
            result = await engine.pipeline(
                candidates,
                "researcher"
            ).filter("Is this feasible?").map("Expand").execute()
        """
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(
                f"Agent '{agent_name}' not found. Define it first with define_agent()"
            )
        return Pipeline(data, agent)

    async def run_session(
        self, agent_name: str, prompt: str, context: Optional[List[str]] = None
    ) -> str:
        """
        Run a single session with an agent.

        Example:
            result = await engine.run_session(
                "captain",
                "Analyze the market trends",
                context=["Previous research..."]
            )
        """
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")
        return await agent.session(prompt, context)

    def cleanup_ephemeral(self) -> int:
        """Clean up ephemeral agents."""
        ephemeral = [
            name
            for name, agent in self.agents.items()
            if agent.definition.persist == PersistenceScope.EPHEMERAL
        ]
        for name in ephemeral:
            del self.agents[name]
        logger.info(f"🧹 Cleaned up {len(ephemeral)} ephemeral agents")
        return len(ephemeral)


# ============================================================================
# SNR-OPTIMIZED WORKFLOW PATTERNS
# ============================================================================


class SnrOptimizedWorkflow:
    """
    SNR-optimized workflow patterns - combines SAPE scoring with OpenProse.

    Implements the "standing on giants" pattern: leverage existing patterns
    and elevate them with BIZRA's SNR/Ihsān scoring system.
    """

    SNR_TARGET = 7.8  # Phase 0 target
    SNR_FLOOR = 7.0  # Safe mode trigger
    IHSAN_THRESHOLD = 0.95  # Constitutional requirement

    def __init__(self, engine: WorkflowEngine):
        self.engine = engine

    async def research_report(
        self,
        topic: str,
        depth: str = "detailed",
        *,
        researcher_tier: ModelTier = ModelTier.OPUS,
        formatter_tier: ModelTier = ModelTier.SONNET,
    ) -> Dict[str, Any]:
        """
        Research report workflow - extracted from OpenProse patterns.

        Pattern:
            block research-report(topic, depth):
              let research = session "Research {topic}"
              let analysis = session "Analyze findings"
              let report = session "Write {depth} report"
        """
        # Define agents
        researcher = self.engine.define_agent(
            AgentDefinition(
                name="researcher",
                model=researcher_tier,
                persist=PersistenceScope.EXECUTION,
                prompt="You perform deep research and analysis with high accuracy.",
            )
        )

        formatter = self.engine.define_agent(
            AgentDefinition(
                name="formatter",
                model=formatter_tier,
                persist=PersistenceScope.EPHEMERAL,
                prompt="You format research into well-structured reports.",
            )
        )

        # Execute workflow
        research = await researcher.session(f"Research {topic} comprehensively")
        analysis = await researcher.session(
            f"Analyze the findings about {topic}", context=[research]
        )
        report = await formatter.session(
            f"Write a {depth}-level report on {topic}", context=[research, analysis]
        )

        # Cleanup
        self.engine.cleanup_ephemeral()

        return {
            "topic": topic,
            "depth": depth,
            "research": research,
            "analysis": analysis,
            "report": report,
            "workflow": "research_report",
            "models_used": {
                "researcher": researcher_tier.value,
                "formatter": formatter_tier.value,
            },
        }

    async def idea_tournament(
        self,
        domain: str,
        count: int = 10,
    ) -> Dict[str, Any]:
        """
        Idea tournament workflow - implements OpenProse pipeline composition.

        Pattern:
            let candidates = session "Generate ideas"
              | filter: session "Is feasible?"
              | map: session "Expand into pitch"
              | reduce: session "Compare and select best"
        """
        # Define generator agent
        generator = self.engine.define_agent(
            AgentDefinition(
                name="generator",
                model=ModelTier.OPUS,
                persist=PersistenceScope.EXECUTION,
                prompt="You generate creative and innovative ideas.",
            )
        )

        # Define evaluator agent
        evaluator = self.engine.define_agent(
            AgentDefinition(
                name="evaluator",
                model=ModelTier.SONNET,
                persist=PersistenceScope.EXECUTION,
                prompt="You evaluate ideas for feasibility and potential.",
            )
        )

        # Generate initial candidates
        candidates_response = await generator.session(
            f"Generate {count} innovative ideas in the domain: {domain}"
        )

        # Parse into list (simplified - real impl would use structured output)
        candidates = [f"Idea from {domain}: {i+1}" for i in range(count)]

        # Execute pipeline
        pipeline = self.engine.pipeline(candidates, "evaluator")
        result = await (
            pipeline.filter(
                "Is this idea technically feasible and valuable? Answer yes/no."
            )
            .map("Expand this idea into a detailed one-page pitch.")
            .reduce(
                "Compare these two pitches. Which is stronger? Return only the stronger one."
            )
            .execute()
        )

        return {
            "domain": domain,
            "initial_count": count,
            "winner": result,
            "workflow": "idea_tournament",
        }


# ============================================================================
# CONVENIENCE FACTORY
# ============================================================================


def create_workflow_engine() -> WorkflowEngine:
    """Factory function to create a configured workflow engine."""
    engine = WorkflowEngine()
    logger.info(f"🚀 WorkflowEngine initialized (execution_id={engine.execution_id})")
    return engine


def create_snr_workflow() -> SnrOptimizedWorkflow:
    """Factory function to create SNR-optimized workflow."""
    engine = create_workflow_engine()
    return SnrOptimizedWorkflow(engine)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================


async def demo_workflow():
    """Demonstrate the OpenProse workflow patterns."""

    # Create SNR-optimized workflow
    workflow = create_snr_workflow()

    # Run research report
    report = await workflow.research_report(
        topic="AI Agent Architectures",
        depth="comprehensive",
        researcher_tier=ModelTier.OPUS,
    )

    print(f"📊 Research Report: {report['topic']}")
    print(f"   Depth: {report['depth']}")
    print(f"   Models: {report['models_used']}")

    # Run idea tournament
    tournament = await workflow.idea_tournament(
        domain="AI-powered developer tools",
        count=5,
    )

    print(f"\n🏆 Idea Tournament: {tournament['domain']}")
    print(f"   Winner: {tournament['winner']}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(demo_workflow())
