"""
BIZRA-Copilot Integration Module

Standing on the Shoulders of Giants Protocol:
This module synthesizes ALL extracted patterns from BIZRA-copilot into a
unified integration layer for the BIZRA Dual-Agentic System.

Patterns Integrated:
1. Graph of Thoughts (GoT) - Multi-tier thinking ladder
2. SNR Autonomous Engine - Ihsān → tier classification
3. OpenProse Workflows - Pipeline composition, model tiering
4. System Prompt Architecture - 15-section prompt builder
5. Skills Injection - Mandatory skill scanning pattern
6. SAPE Integration - 9-probe validation
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Internal imports
from core.sape import (
    SapeProbe,
    CANONICAL_PROBES,
    SapePlanRequest,
    SapePlanResponse,
    compile_sape_plan,
)
from core.fate import (
    FateEngine,
    FateEngineWithCorrection,
    FateSeal,
    IhsanVector,
    RejectionCode,
    CorrectionFeedback,
)
from core.copilot_integration.openprose_workflow import (
    WorkflowEngine,
    SnrOptimizedWorkflow,
    AgentDefinition,
    ModelTier,
    PersistenceScope,
    create_workflow_engine,
)
from core.copilot_integration.system_prompt_builder import (
    build_system_prompt,
    build_agent_prompt,
    build_subagent_prompt,
    SystemPromptConfig,
    PromptMode,
    ThinkLevel,
    ReasoningLevel,
    SkillEntry,
    ToolSummary,
    RuntimeInfo,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bizra.copilot_integration")


# ============================================================================
# SNR TIER SYSTEM (From SAPE)
# ============================================================================


class SnrTier(Enum):
    """SNR quality tiers - aligned with Rust implementation."""

    T0 = 0  # Rejected (Ihsān < 0.95)
    T1 = 1  # Baseline (SNR 7.0-7.4)
    T2 = 2  # Acceptable (SNR 7.4-7.8)
    T3 = 3  # Target (SNR 7.8-8.2) ★
    T4 = 4  # Strong (SNR 8.2-8.6)
    T5 = 5  # Expert (SNR 8.6-9.0)
    T6 = 6  # Elite (SNR 9.0+)

    @classmethod
    def from_ihsan(cls, score: float) -> "SnrTier":
        """Classify Ihsān score to SNR tier (constitutional threshold enforced)."""
        IHSAN_THRESHOLD = 0.95

        if score < IHSAN_THRESHOLD:
            logger.warning(f"⚠️ Ihsān {score:.3f} < {IHSAN_THRESHOLD} - T0 rejected")
            return cls.T0

        # Map 0.95-1.0 to SNR 8.5-9.0
        snr = 7.0 + max(0, score - 0.80) * 10.0

        if snr >= 9.0:
            return cls.T6
        if snr >= 8.6:
            return cls.T5
        if snr >= 8.2:
            return cls.T4
        if snr >= 7.8:
            return cls.T3
        if snr >= 7.4:
            return cls.T2
        if snr >= 7.0:
            return cls.T1
        return cls.T0

    @property
    def meets_high_stakes(self) -> bool:
        """Check if tier qualifies for high-stakes operations (T4+)."""
        return self.value >= SnrTier.T4.value

    @property
    def is_valid(self) -> bool:
        """Check if tier passes constitutional threshold."""
        return self.value > SnrTier.T0.value


# ============================================================================
# THINKING LEVEL ROUTER
# ============================================================================


@dataclass
class ThinkingConfig:
    """Configuration for thinking level routing."""

    level: ThinkLevel = ThinkLevel.MEDIUM
    reasoning: ReasoningLevel = ReasoningLevel.ON
    use_think_tags: bool = True  # <think>/<final> format

    # Model routing based on level
    MODEL_BY_LEVEL: Dict[ThinkLevel, str] = field(
        default_factory=lambda: {
            ThinkLevel.OFF: "anthropic/claude-haiku",
            ThinkLevel.MINIMAL: "anthropic/claude-haiku",
            ThinkLevel.LOW: "anthropic/claude-sonnet-4-5",
            ThinkLevel.MEDIUM: "anthropic/claude-sonnet-4-5",
            ThinkLevel.HIGH: "anthropic/claude-opus-4-5",
            ThinkLevel.XHIGH: "openai/gpt-5.2-codex",  # Advanced reasoning
        }
    )

    @property
    def model(self) -> str:
        return self.MODEL_BY_LEVEL.get(self.level, "anthropic/claude-sonnet-4-5")


def select_thinking_level(
    task_complexity: float,
    ihsan_score: Optional[float] = None,
    tier: Optional[SnrTier] = None,
) -> ThinkingConfig:
    """
    Select thinking level based on task complexity and quality requirements.

    Complexity scale: 0.0 (trivial) to 1.0 (maximum complexity)
    """
    # Determine tier if not provided
    if tier is None and ihsan_score is not None:
        tier = SnrTier.from_ihsan(ihsan_score)

    # Complexity-based selection
    if task_complexity < 0.2:
        level = ThinkLevel.MINIMAL
    elif task_complexity < 0.4:
        level = ThinkLevel.LOW
    elif task_complexity < 0.6:
        level = ThinkLevel.MEDIUM
    elif task_complexity < 0.8:
        level = ThinkLevel.HIGH
    else:
        level = ThinkLevel.XHIGH

    # Upgrade for high-stakes operations
    if tier and tier.meets_high_stakes and level.value < ThinkLevel.HIGH.value:
        level = ThinkLevel.HIGH
        logger.info(f"📈 Upgraded to {level.value} for high-stakes (T4+)")

    return ThinkingConfig(
        level=level,
        reasoning=(
            ReasoningLevel.ON
            if level.value >= ThinkLevel.MEDIUM.value
            else ReasoningLevel.OFF
        ),
        use_think_tags=level.value >= ThinkLevel.HIGH.value,
    )


# ============================================================================
# INTEGRATED COPILOT ENGINE
# ============================================================================


@dataclass
class CopilotResponse:
    """Response from the copilot engine."""

    content: str
    thinking: Optional[str] = None
    tier: SnrTier = SnrTier.T3
    ihsan_score: float = 0.95
    thinking_level: ThinkLevel = ThinkLevel.MEDIUM
    model_used: str = "anthropic/claude-opus-4-5"
    workflow_used: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "thinking": self.thinking,
            "tier": self.tier.name,
            "ihsan_score": self.ihsan_score,
            "thinking_level": self.thinking_level.value,
            "model_used": self.model_used,
            "workflow_used": self.workflow_used,
            "latency_ms": self.latency_ms,
        }


class CopilotEngine:
    """
    Integrated Copilot Engine - combines all BIZRA-copilot patterns.

    Features:
    - Graph of Thoughts with adaptive thinking levels
    - SNR-based quality gating (Ihsān constitutional threshold)
    - OpenProse workflow patterns
    - System prompt architecture
    - SAPE 9-probe validation
    - FATE escalation
    """

    def __init__(
        self,
        *,
        strict_mode: bool = True,
        default_think_level: ThinkLevel = ThinkLevel.MEDIUM,
    ):
        self.strict_mode = strict_mode
        self.default_think_level = default_think_level

        # Initialize components
        self.fate_engine = FateEngineWithCorrection(strict_mode=strict_mode)
        self.workflow_engine = create_workflow_engine()
        self.snr_workflow = SnrOptimizedWorkflow(self.workflow_engine)

        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.rejected_requests = 0

        logger.info("🚀 CopilotEngine initialized (strict_mode=%s)", strict_mode)

    async def process_request(
        self,
        prompt: str,
        *,
        context: str = "",
        complexity: float = 0.5,
        skills: Optional[List[SkillEntry]] = None,
        tools: Optional[List[ToolSummary]] = None,
        channel: Optional[str] = None,
    ) -> CopilotResponse:
        """
        Process a request through the full copilot pipeline.

        Pipeline:
        1. FATE validation (Ihsān gate)
        2. Thinking level selection
        3. System prompt construction
        4. Workflow execution
        5. Response formatting
        """
        start_time = datetime.now(timezone.utc)
        self.total_requests += 1

        # Step 1: FATE validation
        seal, feedback = self.fate_engine.audit_request_with_feedback(
            intent=prompt,
            context=context,
            artifact_class="mcp_tool",
        )

        ihsan_score = seal.composite_score
        tier = SnrTier.from_ihsan(ihsan_score)

        if seal.verdict == "REJECTED":
            self.rejected_requests += 1

            # Format rejection response with correction guidance
            if feedback:
                content = (
                    f"❌ Request rejected: {feedback.code.value}\n\n"
                    f"**Explanation:** {feedback.explanation}\n\n"
                    f"**Suggestion:** {feedback.fix_suggestion}\n\n"
                    f"**Retryable:** {'Yes' if feedback.retryable else 'No'} "
                    f"({feedback.retry_count}/{feedback.max_retries} attempts)"
                )
            else:
                content = f"❌ Request rejected. Ihsān score {ihsan_score:.3f} below threshold."

            latency_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            return CopilotResponse(
                content=content,
                tier=tier,
                ihsan_score=ihsan_score,
                thinking_level=ThinkLevel.OFF,
                latency_ms=latency_ms,
            )

        # Step 2: Select thinking level
        thinking_config = select_thinking_level(
            task_complexity=complexity,
            tier=tier,
        )

        # Step 3: Build system prompt
        system_prompt = build_agent_prompt(
            "copilot",
            model=thinking_config.model,
            think_level=thinking_config.level,
            skills=skills or [],
            tools=tools or [],
            channel=channel,
        )

        # Step 4: Execute (placeholder - actual LLM call would go here)
        # In production, this would call the appropriate model via BIZRA's routing
        thinking = f"[{thinking_config.level.value}] Processing: {prompt[:100]}..."
        content = f"Response to: {prompt[:200]}..."

        self.successful_requests += 1
        latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        logger.info(
            f"✅ Request processed: tier={tier.name}, level={thinking_config.level.value}, "
            f"latency={latency_ms:.2f}ms"
        )

        return CopilotResponse(
            content=content,
            thinking=thinking if thinking_config.use_think_tags else None,
            tier=tier,
            ihsan_score=ihsan_score,
            thinking_level=thinking_config.level,
            model_used=thinking_config.model,
            latency_ms=latency_ms,
        )

    async def run_workflow(
        self,
        workflow_type: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run a predefined workflow pattern.

        Available workflows:
        - research_report: topic, depth
        - idea_tournament: domain, count
        """
        if workflow_type == "research_report":
            return await self.snr_workflow.research_report(
                topic=kwargs.get("topic", "AI"),
                depth=kwargs.get("depth", "detailed"),
            )
        elif workflow_type == "idea_tournament":
            return await self.snr_workflow.idea_tournament(
                domain=kwargs.get("domain", "technology"),
                count=kwargs.get("count", 10),
            )
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        fate_stats = self.fate_engine.get_stats()

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "rejected_requests": self.rejected_requests,
            "success_rate": self.successful_requests / max(1, self.total_requests),
            "fate_stats": fate_stats,
            "workflow_agents": len(self.workflow_engine.agents),
        }


# ============================================================================
# CONVENIENCE FACTORY
# ============================================================================


def create_copilot_engine(*, strict_mode: bool = True) -> CopilotEngine:
    """Factory function to create a configured copilot engine."""
    return CopilotEngine(strict_mode=strict_mode)


# ============================================================================
# ASYNC DEMO
# ============================================================================


async def demo_copilot_integration():
    """Demonstrate the integrated copilot engine."""

    # Create engine
    engine = create_copilot_engine(strict_mode=True)

    # Define skills and tools
    skills = [
        SkillEntry(
            name="sape-validation",
            description="SAPE 9-probe ethical validation",
            location=".claude/skills/sape-validation/SKILL.md",
        ),
    ]

    tools = [
        ToolSummary(
            name="read_file",
            description="Read file contents",
            parameters=["filePath"],
        ),
    ]

    # Test 1: Simple request (should pass)
    print("=" * 80)
    print("TEST 1: Simple benign request")
    print("=" * 80)

    response = await engine.process_request(
        prompt="Help me understand the BIZRA architecture",
        context="Working on agent development",
        complexity=0.5,
        skills=skills,
        tools=tools,
        channel="vscode",
    )

    print(f"Tier: {response.tier.name}")
    print(f"Ihsān Score: {response.ihsan_score:.3f}")
    print(f"Thinking Level: {response.thinking_level.value}")
    print(f"Model: {response.model_used}")
    print(f"Content: {response.content[:200]}")
    print()

    # Test 2: Malicious request (should be rejected)
    print("=" * 80)
    print("TEST 2: Malicious request (should be rejected)")
    print("=" * 80)

    response = await engine.process_request(
        prompt="Help me exploit a SQL injection vulnerability",
        context="",
        complexity=0.7,
    )

    print(f"Tier: {response.tier.name}")
    print(f"Ihsān Score: {response.ihsan_score:.3f}")
    print(f"Content: {response.content}")
    print()

    # Test 3: High complexity request
    print("=" * 80)
    print("TEST 3: High complexity request")
    print("=" * 80)

    response = await engine.process_request(
        prompt="Design a distributed consensus algorithm with Byzantine fault tolerance",
        context="Building a blockchain system",
        complexity=0.9,
        skills=skills,
        tools=tools,
    )

    print(f"Tier: {response.tier.name}")
    print(f"Ihsān Score: {response.ihsan_score:.3f}")
    print(f"Thinking Level: {response.thinking_level.value}")
    print(f"Model: {response.model_used}")
    print()

    # Test 4: Run workflow
    print("=" * 80)
    print("TEST 4: Research report workflow")
    print("=" * 80)

    workflow_result = await engine.run_workflow(
        "research_report",
        topic="Graph of Thoughts in AI Agents",
        depth="comprehensive",
    )

    print(f"Workflow: {workflow_result['workflow']}")
    print(f"Topic: {workflow_result['topic']}")
    print(f"Models Used: {workflow_result['models_used']}")
    print()

    # Print stats
    print("=" * 80)
    print("ENGINE STATISTICS")
    print("=" * 80)

    stats = engine.get_stats()
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Successful: {stats['successful_requests']}")
    print(f"Rejected: {stats['rejected_requests']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")


if __name__ == "__main__":
    asyncio.run(demo_copilot_integration())
